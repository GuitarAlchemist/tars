namespace Tars.Llm

open System
open System.Security.Cryptography
open System.Text
open System.Text.Json

/// TypeSafe AI "System One" (Jev): small, repeated judgements answered as closed types
/// rather than prose. A request carries a state and independent questions, and the model
/// may only answer with values we defined.
///
/// The rule this module exists to keep: a model estimates, deterministic code validates,
/// gates and acts. Every answer is checked against the question that asked for it, so an
/// option we never offered, a probability distribution that does not sum to one, or a
/// choice that is not the model's own argmax is an error rather than a decision.
module SystemOne =

    /// The pinned model. `jev-latest` and `jev-preview` move, which would silently
    /// re-evaluate every threshold tuned against a fixed version.
    [<Literal>]
    let PinnedModel = "jev-1.13.0"

    /// A question the caller asks about the state. Ids are the caller's own.
    type Question =
        /// Pick exactly one option id. Each option carries the criterion for choosing it.
        | Choice of instructions: string * options: (string * string) list
        /// Place the state on an ordered scale, one label per level.
        | Score of instructions: string * levels: string list
        /// How true is this of the state, from 0 to 1.
        | Noul of instructions: string

    /// A validated answer. Confidence is the model's, never an authority.
    type Answer =
        | Chose of choice: string * confidence: float * probabilities: Map<string, float>
        | Scored of score: float * confidence: float * probabilities: Map<string, float>
        | Nouled of noul: float

    type Usage = { InputTokens: int; OutputTokens: int }

    type Reply =
        { Model: string
          Answers: Map<string, Answer>
          Usage: Usage }

        /// The answer to `questionId`, if the reply carried one.
        member this.TryAnswer(questionId: string) = this.Answers.TryFind questionId

    // ---------------------------------------------------------------- request

    /// A state field. Deliberately narrow: what the model reads should be data we
    /// assembled, not a serialised object graph we stopped tracking.
    type StateValue =
        | Text of string
        | Items of string list

    let private writeState (writer: Utf8JsonWriter) (state: (string * StateValue) list) =
        writer.WriteStartObject()

        for key, value in state |> List.sortBy fst do
            match value with
            | Text text -> writer.WriteString(key, text)
            | Items items ->
                writer.WriteStartArray(key)
                for item in items do
                    writer.WriteStringValue(item)
                writer.WriteEndArray()

        writer.WriteEndObject()

    let private writeQuestion (writer: Utf8JsonWriter) (id: string) (question: Question) =
        writer.WriteStartObject(id)

        match question with
        | Choice(instructions, options) ->
            writer.WriteString("type", "choice")
            writer.WriteString("instructions", instructions)
            writer.WriteStartObject("criteria")

            for optionId, criterion in options do
                writer.WriteString(optionId, criterion)

            writer.WriteEndObject()
        | Score(instructions, levels) ->
            writer.WriteString("type", "score")
            writer.WriteString("instructions", instructions)
            writer.WriteStartArray("criteria")

            for level in levels do
                writer.WriteStringValue(level)

            writer.WriteEndArray()
        | Noul instructions ->
            writer.WriteString("type", "noul")
            writer.WriteString("instructions", instructions)

        writer.WriteEndObject()

    /// The request body, with keys in a fixed order so its digest is stable across runs.
    let payload (state: (string * StateValue) list) (questions: (string * Question) list) : string =
        use stream = new IO.MemoryStream()
        use writer = new Utf8JsonWriter(stream)
        writer.WriteStartObject()
        writer.WriteString("model", PinnedModel)
        writer.WritePropertyName("questions")
        writer.WriteStartObject()

        for id, question in questions |> List.sortBy fst do
            writeQuestion writer id question

        writer.WriteEndObject()
        writer.WritePropertyName("state")
        writeState writer state
        writer.WriteEndObject()
        writer.Flush()
        Encoding.UTF8.GetString(stream.ToArray())

    /// SHA-256 of a request body, for journalling which question was asked.
    let digest (payloadJson: string) : string =
        use sha = SHA256.Create()

        sha.ComputeHash(Encoding.UTF8.GetBytes payloadJson)
        |> Array.map (fun b -> b.ToString("x2"))
        |> String.concat ""

    /// UTF-8 size of a request body, the only input cost we can bound before sending.
    let payloadBytes (payloadJson: string) : int = Encoding.UTF8.GetByteCount payloadJson

    // ------------------------------------------------------------- validation

    /// Floating-point slack only. Rounding slack is derived per reply, below.
    let private tolerance = 1e-9

    /// How precisely a number was published. Jev rounds what it sends, so a rule that
    /// ignores the rounding rejects replies that are perfectly consistent: the first
    /// live probe answered score 0.05 over a 0.96/0.04/0.00 distribution, whose exact
    /// expectation is 0.04 — one hundredth apart, and both true to two decimals.
    let private decimalsOf (element: JsonElement) =
        let raw = element.GetRawText()

        if raw.Contains "e" || raw.Contains "E" then
            15 // exponent form is already more precision than any rounding grid
        else
            match raw.IndexOf '.' with
            | -1 -> 0
            | dot -> raw.Length - dot - 1

    /// Half of the last published digit: the most a rounded number can be off by.
    let private halfUlp (decimals: int) = 0.5 * Math.Pow(10.0, float -decimals)

    let private finite (value: float) =
        not (Double.IsNaN value) && not (Double.IsInfinity value)

    let private tryProp (name: string) (element: JsonElement) =
        match element.TryGetProperty name with
        | true, value -> Some value
        | _ -> None

    let private number (context: string) (element: JsonElement) =
        if element.ValueKind = JsonValueKind.Number then
            let value = element.GetDouble()
            if finite value then Ok value else Error $"{context} must be a finite number"
        else
            Error $"{context} must be a number"

    let private unitInterval context element =
        number context element
        |> Result.bind (fun value ->
            if value >= 0.0 && value <= 1.0 then
                Ok value
            else
                Error $"{context} must be between 0 and 1")

    /// Probabilities must cover exactly the options offered, each in [0, 1], summing to 1
    /// once the grid they were rounded onto is allowed for. Returns the distribution and
    /// that grid's half-digit, which the score check needs too.
    let private probabilities (context: string) (expected: Set<string>) (answer: JsonElement) =
        match tryProp "probabilities" answer with
        | None -> Error $"{context}: probabilities are missing"
        | Some element when element.ValueKind <> JsonValueKind.Object ->
            Error $"{context}: probabilities must be an object"
        | Some element ->
            let entries = element.EnumerateObject() |> Seq.toList

            let keys = entries |> List.map (fun p -> p.Name) |> Set.ofList

            // The finest precision published is the grid; coarser-looking values are
            // just trailing zeros dropped, not a coarser grid.
            let slack =
                match entries with
                | [] -> tolerance
                | _ -> entries |> List.map (fun p -> decimalsOf p.Value) |> List.max |> halfUlp

            if keys <> expected then
                let offered = expected |> Set.toList |> String.concat ", "
                Error $"{context}: probability keys must be exactly the options asked for ({offered})"
            else
                let folded =
                    entries
                    |> List.fold
                        (fun acc property ->
                            match acc with
                            | Error _ -> acc
                            | Ok map ->
                                unitInterval $"{context}: probability '{property.Name}'" property.Value
                                |> Result.map (fun value -> map |> Map.add property.Name value))
                        (Ok Map.empty)

                folded
                |> Result.bind (fun map ->
                    let total = map |> Map.fold (fun sum _ value -> sum + value) 0.0

                    // Every rounded term can be off by half a digit, so the sum can be
                    // off by that much per option — and no more.
                    if abs (total - 1.0) <= slack * float entries.Length + tolerance then
                        Ok(map, slack)
                    else
                        Error $"{context}: probabilities must sum to 1, got %.6f{total}")

    let private confidenceOf context answer =
        match tryProp "confidence" answer with
        | None -> Error $"{context}: confidence is missing"
        | Some element -> unitInterval $"{context}: confidence" element

    let private expectType (context: string) (expected: string) (answer: JsonElement) =
        match tryProp "type" answer with
        | Some element when element.ValueKind = JsonValueKind.String && element.GetString() = expected -> Ok()
        | _ -> Error $"{context}: expected a {expected} answer"

    let private parseChoice context (options: (string * string) list) (answer: JsonElement) =
        let ids = options |> List.map fst |> Set.ofList

        expectType context "choice" answer
        |> Result.bind (fun () ->
            match tryProp "choice" answer with
            | Some element when element.ValueKind = JsonValueKind.String -> Ok(element.GetString())
            | _ -> Error $"{context}: choice must be a string")
        |> Result.bind (fun chosen ->
            if ids.Contains chosen then
                Ok chosen
            else
                // The whole point of a closed answer: an invented option is not a decision.
                Error $"{context}: '{chosen}' is not one of the options asked for")
        |> Result.bind (fun chosen ->
            probabilities context ids answer
            |> Result.bind (fun (distribution, slack) ->
                let best = distribution |> Map.fold (fun best _ value -> max best value) 0.0

                // Two options genuinely tied can land a digit apart once rounded; a
                // real gap is wider than the grid.
                if distribution.[chosen] < best - (2.0 * slack + tolerance) then
                    Error $"{context}: chose '{chosen}' while naming another option as more likely"
                else
                    confidenceOf context answer
                    |> Result.map (fun confidence -> Chose(chosen, confidence, distribution))))

    let private parseScore context (levels: string list) (answer: JsonElement) =
        let top = float (List.length levels - 1)
        let levelKeys = levels |> List.mapi (fun index _ -> string index) |> Set.ofList

        expectType context "score" answer
        |> Result.bind (fun () ->
            match tryProp "score" answer with
            | None -> Error $"{context}: score is missing"
            | Some element -> number $"{context}: score" element)
        |> Result.bind (fun score ->
            if score < 0.0 || score > top then
                Error $"{context}: score must be between 0 and %.0f{top}"
            else
                Ok score)
        |> Result.bind (fun score ->
            match tryProp "legend" answer with
            | Some element when element.ValueKind = JsonValueKind.Object ->
                let returned =
                    element.EnumerateObject()
                    |> Seq.map (fun p -> p.Name, (if p.Value.ValueKind = JsonValueKind.String then p.Value.GetString() else ""))
                    |> Map.ofSeq

                let expected = levels |> List.mapi (fun index label -> string index, label) |> Map.ofList

                if returned = expected then
                    Ok score
                else
                    Error $"{context}: legend must repeat the levels asked for"
            | _ -> Error $"{context}: legend is missing")
        |> Result.bind (fun score ->
            let scoreSlack =
                tryProp "score" answer
                |> Option.map (decimalsOf >> halfUlp)
                |> Option.defaultValue tolerance

            probabilities context levelKeys answer
            |> Result.bind (fun (distribution, slack) ->
                let weighted =
                    distribution |> Map.fold (fun sum level value -> sum + float (int level) * value) 0.0

                // Each level's rounding is multiplied by that level's weight, and the
                // score carries its own rounding on top.
                let weightedSlack =
                    distribution
                    |> Map.fold (fun sum level _ -> sum + float (int level) * slack) 0.0

                if abs (score - weighted) > weightedSlack + scoreSlack + tolerance then
                    Error $"{context}: score %.6f{score} does not match its own distribution (%.6f{weighted})"
                else
                    confidenceOf context answer
                    |> Result.map (fun confidence -> Scored(score, confidence, distribution))))

    let private parseNoul context (answer: JsonElement) =
        expectType context "noul" answer
        |> Result.bind (fun () ->
            match tryProp "noul" answer with
            | None -> Error $"{context}: noul is missing"
            | Some element -> unitInterval $"{context}: noul" element |> Result.map Nouled)

    let private parseUsage (root: JsonElement) =
        match tryProp "usage" root with
        | Some element when element.ValueKind = JsonValueKind.Object ->
            let names = element.EnumerateObject() |> Seq.map (fun p -> p.Name) |> Set.ofSeq

            if names <> set [ "input_tokens"; "output_tokens" ] then
                Error "usage must carry exactly input_tokens and output_tokens"
            else
                let count name =
                    let value = element.GetProperty(name: string)

                    match value.TryGetInt32() with
                    | true, tokens when tokens >= 0 -> Ok tokens
                    | _ -> Error $"usage.{name} must be a non-negative whole number"

                match count "input_tokens", count "output_tokens" with
                | Ok input, Ok output ->
                    Ok
                        { InputTokens = input
                          OutputTokens = output }
                | Error e, _
                | _, Error e -> Error e
        | _ -> Error "usage is missing"

    /// Parse a reply and check it against the questions that were asked. Every failure
    /// names what was wrong, because a rejected reply is a fact worth logging.
    let parseReply (questions: (string * Question) list) (json: string) : Result<Reply, string> =
        try
            use document = JsonDocument.Parse(json: string)
            let root = document.RootElement

            let model =
                match tryProp "model" root with
                | Some element when element.ValueKind = JsonValueKind.String && element.GetString() <> "" ->
                    Ok(element.GetString())
                | _ -> Error "model must be a non-empty string"

            let answers =
                match tryProp "answers" root with
                | Some element when element.ValueKind = JsonValueKind.Object ->
                    let returned = element.EnumerateObject() |> Seq.map (fun p -> p.Name) |> Set.ofSeq
                    let asked = questions |> List.map fst |> Set.ofList

                    if returned <> asked then
                        Error "answers must cover exactly the question ids asked"
                    else
                        questions
                        |> List.fold
                            (fun acc (id, question) ->
                                match acc with
                                | Error _ -> acc
                                | Ok map ->
                                    let answer = element.GetProperty(id: string)

                                    let parsed =
                                        match question with
                                        | Choice(_, options) -> parseChoice id options answer
                                        | Score(_, levels) -> parseScore id levels answer
                                        | Noul _ -> parseNoul id answer

                                    parsed |> Result.map (fun a -> map |> Map.add id a))
                            (Ok Map.empty)
                | _ -> Error "answers must be an object"

            match model, answers, parseUsage root with
            | Ok model, Ok answers, Ok usage ->
                Ok
                    { Model = model
                      Answers = answers
                      Usage = usage }
            | Error e, _, _
            | _, Error e, _
            | _, _, Error e -> Error e
        with :? JsonException as ex ->
            Error $"reply is not JSON: {ex.Message}"

    /// As `parseReply`, and the reply must come from the pinned model. Use this for
    /// anything whose thresholds were tuned against a fixed version.
    let parsePinnedReply questions json =
        parseReply questions json
        |> Result.bind (fun reply ->
            if reply.Model = PinnedModel then
                Ok reply
            else
                Error $"reply came from '{reply.Model}', not the pinned {PinnedModel}")

    // ------------------------------------------------------------------- port

    /// One question set evaluated against one state. Implementations own transport,
    /// the pinned model, timeouts and redaction; callers own thresholds and effects.
    type ISystemOne =
        abstract Evaluate: state: (string * StateValue) list * questions: (string * Question) list -> Async<Result<Reply, string>>

    /// Answers from a saved reply: the default everywhere, including CI, so a decision
    /// seam can be exercised without a key, a network or a bill.
    type ReplaySystemOne(replyJson: string) =

        /// Read the reply from a file written by an earlier live probe.
        static member FromFile(path: string) = ReplaySystemOne(IO.File.ReadAllText path)

        interface ISystemOne with
            member _.Evaluate(_, questions) =
                async { return parseReply questions replyJson }

    // -------------------------------------------------------------- transport

    /// Published input price. The lab's proxy counts one token per byte, which
    /// overestimates on purpose: a bound is only useful when it cannot be undershot.
    [<Literal>]
    let InputPricePerMillionUsd = 0.042

    /// Worst-case input cost of a payload, in US dollars.
    let inputCostProxyUsd (bytes: int) =
        float bytes * InputPricePerMillionUsd / 1_000_000.0

    /// The bounds a live call is held to, so a decision seam can never quietly become
    /// an unbounded spend or a slow path in the middle of a plan.
    type JevLimits =
        { Endpoint: string
          MaxPayloadBytes: int
          Timeout: TimeSpan }

        static member Default =
            { Endpoint = "https://api.typesafe.ai/v1/systemone"
              MaxPayloadBytes = 2500
              Timeout = TimeSpan.FromSeconds 20.0 }

    /// The live model. The request is refused before it leaves when it is over the byte
    /// cap, redirects are not followed so the bearer stays with the origin it was issued
    /// for, and the key appears in no log, message or error.
    type JevClient(apiKey: string, limits: JevLimits) =
        let handler = new Net.Http.HttpClientHandler(AllowAutoRedirect = false)
        let http = new Net.Http.HttpClient(handler, Timeout = limits.Timeout)

        new(apiKey) = new JevClient(apiKey, JevLimits.Default)

        interface IDisposable with
            member _.Dispose() = http.Dispose()

        /// The reply exactly as it arrived, before it is read as answers. A probe
        /// saves this as the fixture the offline tests replay.
        member _.EvaluateRaw(state, questions) : Async<Result<string, string>> =
            async {
                let body = payload state questions
                let size = payloadBytes body

                if size > limits.MaxPayloadBytes then
                    return Error $"request is {size} bytes, over the {limits.MaxPayloadBytes}-byte cap"
                else
                    try
                        use request =
                            new Net.Http.HttpRequestMessage(Net.Http.HttpMethod.Post, limits.Endpoint)

                        request.Headers.Authorization <-
                            Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiKey)

                        request.Content <- new Net.Http.StringContent(body, Encoding.UTF8, "application/json")

                        let! response = http.SendAsync request |> Async.AwaitTask
                        let! text = response.Content.ReadAsStringAsync() |> Async.AwaitTask

                        if response.IsSuccessStatusCode then
                            return Ok text
                        else
                            // The error body can quote the request back; the status is
                            // what a caller acts on anyway.
                            return Error $"Jev answered {int response.StatusCode}"
                    with ex ->
                        return Error $"Jev call failed: {ex.GetType().Name}"
            }

        interface ISystemOne with
            member this.Evaluate(state, questions) =
                async {
                    let! raw = this.EvaluateRaw(state, questions)
                    return raw |> Result.bind (parsePinnedReply questions)
                }

    /// The decider this process is configured for: a live client only when `TARS_JEV`
    /// is on *and* `TYPESAFE_API_KEY` is set. Every other case is `None`, which leaves
    /// callers on the path they already had. The key is read here and goes nowhere else.
    let deciderFromEnvironment () : ISystemOne option =
        let isOn (value: string) =
            [ "1"; "true"; "yes"; "on" ] |> List.contains (value.Trim().ToLowerInvariant())

        match Environment.GetEnvironmentVariable "TARS_JEV" with
        | null -> None
        | flag when not (isOn flag) -> None
        | _ ->
            match Environment.GetEnvironmentVariable "TYPESAFE_API_KEY" with
            | null -> None
            | key when String.IsNullOrWhiteSpace key -> None
            | key -> Some(new JevClient(key.Trim()) :> ISystemOne)
