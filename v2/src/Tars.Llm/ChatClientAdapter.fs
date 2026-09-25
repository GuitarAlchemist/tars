namespace Tars.Llm

// Bidirectional adapters between TARS ILlmService and Microsoft.Extensions.AI IChatClient.
// - LlmServiceChatClient: wraps ILlmService as an IChatClient (for new MAF code)
// - ChatClientLlmService: wraps IChatClient as an ILlmService (for existing TARS code)

open System
open System.Collections.Generic
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.AI
open Tars.Llm
open Tars.Llm.Routing

// ─────────────────────────────────────────────────────────────────────
// Mapping helpers
// ─────────────────────────────────────────────────────────────────────
module internal ChatClientMapping =

    /// Map TARS Role to M.E.AI ChatRole
    let toAIRole (role: Role) : ChatRole =
        match role with
        | Role.System -> ChatRole.System
        | Role.User -> ChatRole.User
        | Role.Assistant -> ChatRole.Assistant
        | Role.Tool _ -> ChatRole.Tool
        | Role.AssistantCalling _ -> ChatRole.Assistant

    /// Map M.E.AI ChatRole to TARS Role. A tool result keeps the id of the call it
    /// answers; without it the next request cannot say what was answered.
    let fromAIRole (role: ChatRole) (callId: string option) : Role =
        if role = ChatRole.System then Role.System
        elif role = ChatRole.Assistant then Role.Assistant
        elif role = ChatRole.Tool then Role.Tool(callId |> Option.defaultValue "")
        else Role.User

    /// Convert a TARS LlmMessage to an M.E.AI ChatMessage
    let toAIChatMessage (msg: LlmMessage) : ChatMessage =
        let message = ChatMessage(toAIRole msg.Role, msg.Content)

        match msg.Role with
        | Role.AssistantCalling calls ->
            for call in calls do
                let arguments =
                    try
                        JsonSerializer.Deserialize<Dictionary<string, obj>>(call.ArgumentsJson)
                        :> IDictionary<string, obj>
                    with _ ->
                        dict []

                message.Contents.Add(FunctionCallContent(call.Id, call.Name, arguments))
        | _ -> ()

        message

    /// What a function result carries: the id it answers, and its result as text.
    let private functionResult (msg: ChatMessage) =
        msg.Contents
        |> Seq.tryPick (fun content ->
            match content with
            | :? FunctionResultContent as result ->
                let text =
                    match result.Result with
                    | null -> ""
                    | :? string as s -> s
                    | other -> JsonSerializer.Serialize other

                Some(result.CallId, text)
            | _ -> None)

    /// The calls an assistant turn asked for, kept so the exchange stays answerable.
    let private functionCalls (msg: ChatMessage) =
        msg.Contents
        |> Seq.choose (fun content ->
            match content with
            | :? FunctionCallContent as call ->
                let arguments =
                    match call.Arguments with
                    | null -> "{}"
                    | args -> JsonSerializer.Serialize args

                Some
                    { Id = call.CallId
                      Name = call.Name
                      ArgumentsJson = arguments }
            | _ -> None)
        |> List.ofSeq

    /// Convert an M.E.AI ChatMessage to a TARS LlmMessage
    let fromAIChatMessage (msg: ChatMessage) : LlmMessage =
        match functionResult msg with
        | Some(callId, text) ->
            // A tool result's payload lives in its content part, not in `Text`.
            { Role = Role.Tool callId
              Content = text }
        | None ->
            let text = msg.Text |> Option.ofObj |> Option.defaultValue ""

            match functionCalls msg with
            | [] ->
                { Role = fromAIRole msg.Role None
                  Content = text }
            | calls ->
                // The assistant turn that asked for the tools. Mapping it on `Text`
                // alone — normally empty — left the next request with an empty turn
                // followed by a result answering nothing, which OpenAI-shaped
                // endpoints reject outright.
                { Role = Role.AssistantCalling calls
                  Content = text }

    /// The tools of a request, in the shape every OpenAI-descended wire expects
    /// (Ollama included). `AIFunction` already carries the JSON schema the provider
    /// wants; anything else in `ChatOptions.Tools` is not a function we can offer.
    let toolsOf (options: ChatOptions) : obj list =
        if isNull options || isNull options.Tools then
            []
        else
            options.Tools
            |> Seq.choose (fun tool ->
                match tool with
                | :? AIFunction as f ->
                    let description: obj =
                        dict
                            [ "name", box f.Name
                              "description", box (f.Description |> Option.ofObj |> Option.defaultValue "")
                              "parameters", box f.JsonSchema ]

                    Some(box (dict [ "type", box "function"; "function", description ]))
                | _ -> None)
            |> Seq.toList

    /// A tool call as it comes back over the wire.
    type WireToolCall =
        { CallId: string
          Name: string
          Arguments: IDictionary<string, obj> }

    let private argumentsOf (element: JsonElement) =
        // Ollama sends an object; OpenAI sends that object as a JSON string.
        let asObject =
            match element.ValueKind with
            | JsonValueKind.String ->
                try
                    Some(JsonDocument.Parse(element.GetString()).RootElement)
                with _ ->
                    None
            | JsonValueKind.Object -> Some element
            | _ -> None

        match asObject with
        | Some o when o.ValueKind = JsonValueKind.Object ->
            o.EnumerateObject()
            |> Seq.map (fun property ->
                let value: obj =
                    match property.Value.ValueKind with
                    | JsonValueKind.String -> box (property.Value.GetString())
                    | JsonValueKind.Number -> box (property.Value.GetDouble())
                    | JsonValueKind.True -> box true
                    | JsonValueKind.False -> box false
                    | JsonValueKind.Null -> null
                    | _ -> box (property.Value.GetRawText())

                property.Name, value)
            |> dict
        | _ -> dict []

    /// Read tool calls out of a provider's raw reply.
    ///
    /// `LlmResponse` has no typed channel for them (#305 keeps that open), but the raw
    /// body does, and dropping the calls here is what made the whole tool pipeline
    /// inert. Both shapes in use are read: Ollama's `message.tool_calls` and the
    /// OpenAI family's `choices[].message.tool_calls`.
    let toolCallsOf (raw: string option) : WireToolCall list =
        match raw with
        | None -> []
        | Some body ->
            try
                let root = JsonDocument.Parse(body).RootElement

                let messages =
                    seq {
                        match root.TryGetProperty "message" with
                        | true, message -> yield message
                        | _ -> ()

                        match root.TryGetProperty "choices" with
                        | true, choices when choices.ValueKind = JsonValueKind.Array ->
                            for choice in choices.EnumerateArray() do
                                match choice.TryGetProperty "message" with
                                | true, message -> yield message
                                | _ -> ()
                        | _ -> ()
                    }

                [ for message in messages do
                      match message.TryGetProperty "tool_calls" with
                      | true, calls when calls.ValueKind = JsonValueKind.Array ->
                          for index, call in calls.EnumerateArray() |> Seq.indexed do
                              match call.TryGetProperty "function" with
                              | true, fn ->
                                  let name =
                                      match fn.TryGetProperty "name" with
                                      | true, n when n.ValueKind = JsonValueKind.String -> n.GetString()
                                      | _ -> ""

                                  let arguments =
                                      match fn.TryGetProperty "arguments" with
                                      | true, args -> argumentsOf args
                                      | _ -> dict []

                                  // Ollama sends no id; the position is what pairs the
                                  // result back, and M.E.AI needs *some* id to match on.
                                  let callId =
                                      match call.TryGetProperty "id" with
                                      | true, id when id.ValueKind = JsonValueKind.String -> id.GetString()
                                      | _ -> $"call_{index}"

                                  if not (String.IsNullOrWhiteSpace name) then
                                      yield
                                          { CallId = callId
                                            Name = name
                                            Arguments = arguments }
                              | _ -> ()
                      | _ -> () ]
            with _ ->
                // A body we cannot read is a body with no calls in it, not a failure:
                // the text answer still stands.
                []

    /// Wire names for the constraints M.E.AI has no first-class channel for.
    /// Private to this adapter pair: no provider reads them, and they are NOT the
    /// shape `OpenAiCompatibleClient` puts on the wire (that is a nested
    /// `structured_outputs: { grammar = ... }` object). They exist only so the
    /// write and read halves here cannot drift apart.
    [<Literal>]
    let GrammarKey = "structured_outputs_grammar"

    [<Literal>]
    let RegexKey = "structured_outputs_regex"

    /// A JSON schema must be a JSON *object*. `"null"`, `"42"` and `"[1,2]"` all
    /// parse happily and would sail through as a schema, producing
    /// `json_schema: { schema: null }` and a 400 at the provider rather than the
    /// intended degrade to plain JSON mode.
    let private tryParseSchema (schema: string) =
        try
            let root = System.Text.Json.JsonDocument.Parse(schema).RootElement

            if root.ValueKind = System.Text.Json.JsonValueKind.Object then
                Some root
            else
                None
        with _ ->
            None

    /// Build M.E.AI ChatOptions from an LlmRequest
    let toChatOptions (req: LlmRequest) : ChatOptions =
        let opts = ChatOptions()
        req.Temperature |> Option.iter (fun t -> opts.Temperature <- Nullable(float32 t))
        req.MaxTokens |> Option.iter (fun m -> opts.MaxOutputTokens <- Nullable m)
        req.Model |> Option.iter (fun m -> opts.ModelId <- m)
        req.Seed |> Option.iter (fun s -> opts.Seed <- Nullable(int64 s))

        if not req.Stop.IsEmpty then
            opts.StopSequences <- req.Stop |> ResizeArray

        let carry key (value: string) =
            let dict = Dictionary<string, obj>()
            dict.[key] <- box value
            opts.AdditionalProperties <- AdditionalPropertiesDictionary(dict)

        match req.ResponseFormat with
        | Some ResponseFormat.Json -> opts.ResponseFormat <- ChatResponseFormat.Json
        // Text must be stated, not left to the catch-all: falling through would let
        // `req.JsonMode` overwrite it, turning an explicit request for prose into
        // one demanding JSON.
        | Some ResponseFormat.Text -> opts.ResponseFormat <- ChatResponseFormat.Text
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema schema)) ->
            // `ForJsonSchema` is the channel providers actually enforce;
            // AdditionalProperties is not, so a schema left there is never applied.
            // A schema we cannot use degrades to plain JSON mode rather than
            // throwing out of a format mapping.
            match tryParseSchema schema with
            | Some element ->
                opts.ResponseFormat <-
                    ChatResponseFormat.ForJsonSchema(element, "tars_structured_output", "TARS constrained response schema")
            | None -> opts.ResponseFormat <- ChatResponseFormat.Json
        | Some (ResponseFormat.Constrained (Grammar.Ebnf grammar)) ->
            // Backend selection is the server's job, so no backend key here.
            carry GrammarKey grammar
        | Some (ResponseFormat.Constrained (Grammar.Regex pattern)) -> carry RegexKey pattern
        | None ->
            if req.JsonMode then
                opts.ResponseFormat <- ChatResponseFormat.Json

        opts

    /// Recover a TARS ResponseFormat from M.E.AI ChatOptions — the inverse of
    /// `toChatOptions`.
    ///
    /// The trap this exists to avoid: `ChatResponseFormat.ForJsonSchema` returns a
    /// NEW `ChatResponseFormatJson`, not the `ChatResponseFormat.Json` singleton, so
    /// a reference comparison against that singleton is false in precisely the case
    /// that carries a schema. Distinguish on `.Schema`, never on identity.
    ///
    /// Precedence, for options that carry both channels — `toChatOptions` never
    /// emits both, so this only arises from an external producer:
    ///   1. a schema-bearing ResponseFormat — the most specific constraint, and one
    ///      the caller set through the typed API on purpose
    ///   2. the grammar/regex side channel — used only because M.E.AI has no slot
    ///      for them, so it should not beat an explicit schema
    ///   3. a bare Json/Text ResponseFormat — weaker than a grammar, so it loses
    let fromChatOptions (options: ChatOptions) : ResponseFormat option =
        let carried key =
            match options.AdditionalProperties with
            | null -> None
            | props ->
                match props.TryGetValue key with
                | true, (:? string as s) when not (String.IsNullOrWhiteSpace s) -> Some s
                | _ -> None

        // `Undefined` is reachable: M.E.AI accepts `ForJsonSchema(default)` without
        // validation, and GetRawText() throws on it. A schema we cannot read is no
        // schema, not an exception escaping the adapter.
        let schemaOf (json: ChatResponseFormatJson) =
            Option.ofNullable json.Schema
            |> Option.filter (fun s -> s.ValueKind <> System.Text.Json.JsonValueKind.Undefined)

        let typed =
            match box options.ResponseFormat with
            | null -> None
            | :? ChatResponseFormatJson as json ->
                match schemaOf json with
                | Some schema -> Some(ResponseFormat.Constrained(Grammar.JsonSchema(schema.GetRawText())))
                | None -> Some ResponseFormat.Json
            | _ -> Some ResponseFormat.Text

        match typed with
        | Some (ResponseFormat.Constrained (Grammar.JsonSchema _)) -> typed
        | _ ->
            carried GrammarKey
            |> Option.map (Grammar.Ebnf >> ResponseFormat.Constrained)
            |> Option.orElseWith (fun () ->
                carried RegexKey |> Option.map (Grammar.Regex >> ResponseFormat.Constrained))
            |> Option.orElse typed

    /// M.E.AI seeds are `int64`; `LlmRequest.Seed` is `int`. A plain `int` cast
    /// truncates silently — `int 4294967297L` is `1` — so a caller asking for a
    /// 64-bit seed would get a DIFFERENT seed than they requested, defeating the
    /// only thing a seed is for. Out-of-range seeds are dropped instead: no seed is
    /// honest, a wrong seed is not.
    let seedOf (options: ChatOptions) (fallback: int option) =
        options.Seed
        |> Option.ofNullable
        |> Option.filter (fun s -> s >= int64 Int32.MinValue && s <= int64 Int32.MaxValue)
        |> Option.map int
        |> Option.orElse fallback

    /// Stop sequences, evaluated eagerly into an immutable list.
    let stopOf (options: ChatOptions) (fallback: string list) =
        if isNull options.StopSequences then fallback
        else options.StopSequences |> Seq.toList

    /// Apply a recovered format to a request, setting `ResponseFormat` and the
    /// legacy `JsonMode` flag together.
    ///
    /// Both fields live on LlmRequest and every backend matches `ResponseFormat`
    /// first, consulting `JsonMode` only in the `None` branch — so `JsonMode` is
    /// dead whenever `ResponseFormat` is set. It is maintained here anyway so the
    /// two never state different things to a future reader or backend.
    let applyFormat (options: ChatOptions) (req: LlmRequest) =
        match fromChatOptions options with
        | None -> req
        | Some format ->
            { req with
                ResponseFormat = Some format
                JsonMode =
                    match format with
                    | ResponseFormat.Json
                    | ResponseFormat.Constrained (Grammar.JsonSchema _) -> true
                    | _ -> false }

    /// Convert an M.E.AI ChatResponse to a TARS LlmResponse
    let toLlmResponse (resp: ChatResponse) : LlmResponse =
        let text = resp.Text |> Option.ofObj |> Option.defaultValue ""

        let usage =
            resp.Usage
            |> Option.ofObj
            |> Option.map (fun u ->
                { PromptTokens = u.InputTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0
                  CompletionTokens = u.OutputTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0
                  TotalTokens = u.TotalTokenCount |> Option.ofNullable |> Option.map int |> Option.defaultValue 0 })

        let finishReason =
            resp.FinishReason
            |> Option.ofNullable
            |> Option.map (fun fr ->
                if fr = ChatFinishReason.Stop then "stop"
                elif fr = ChatFinishReason.Length then "length"
                elif fr = ChatFinishReason.ContentFilter then "content_filter"
                elif fr = ChatFinishReason.ToolCalls then "tool_calls"
                else "unknown")

        { Text = text
          FinishReason = finishReason
          Usage = usage
          Raw = None }

// ─────────────────────────────────────────────────────────────────────
// Adapter 1: ILlmService -> IChatClient
// Allows existing ILlmService implementations to be consumed as IChatClient.
// ─────────────────────────────────────────────────────────────────────
type LlmServiceChatClient(inner: ILlmService) =

    interface IChatClient with

        member this.GetService(serviceType: Type, serviceKey: obj) : obj =
            if serviceType = typeof<IChatClient> && isNull serviceKey then
                box this
            else
                null

        member _.GetResponseAsync(messages: IEnumerable<ChatMessage>, options: ChatOptions, cancellationToken: CancellationToken) : Task<ChatResponse> =
            task {
                let tarsMessages =
                    messages
                    |> Seq.map ChatClientMapping.fromAIChatMessage
                    |> Seq.toList

                let mutable req = Prompt.ofMessages tarsMessages

                if not (isNull options) then
                    req <-
                        { req with
                            Temperature =
                                options.Temperature
                                |> Option.ofNullable
                                |> Option.map float
                                |> Option.orElse req.Temperature
                            MaxTokens =
                                options.MaxOutputTokens
                                |> Option.ofNullable
                                |> Option.orElse req.MaxTokens
                            Model =
                                options.ModelId
                                |> Option.ofObj
                                |> Option.orElse req.Model
                            Seed = ChatClientMapping.seedOf options req.Seed
                            Stop = ChatClientMapping.stopOf options req.Stop
                            Tools = ChatClientMapping.toolsOf options }
                        |> ChatClientMapping.applyFormat options

                let! llmResp = inner.CompleteAsync(req)

                let responseMsg = ChatMessage(ChatRole.Assistant, llmResp.Text)

                // Without these the invoking client sees prose where the model asked
                // for a tool, and the loop never starts.
                let toolCalls = ChatClientMapping.toolCallsOf llmResp.Raw

                for call in toolCalls do
                    responseMsg.Contents.Add(FunctionCallContent(call.CallId, call.Name, call.Arguments))

                let chatResp = ChatResponse(responseMsg)
                chatResp.ModelId <- req.Model |> Option.defaultValue null

                if not toolCalls.IsEmpty then
                    // Providers disagree about saying so; the calls themselves are the
                    // fact, and the invoking client keys off this.
                    chatResp.FinishReason <- Nullable ChatFinishReason.ToolCalls
                else
                    llmResp.FinishReason
                    |> Option.iter (fun fr ->
                        chatResp.FinishReason <-
                            Nullable(
                                match fr with
                                | "stop" -> ChatFinishReason.Stop
                                | "length" -> ChatFinishReason.Length
                                | "content_filter" -> ChatFinishReason.ContentFilter
                                | "tool_calls" -> ChatFinishReason.ToolCalls
                                | _ -> ChatFinishReason.Stop))

                llmResp.Usage
                |> Option.iter (fun u ->
                    let usage = UsageDetails()
                    usage.InputTokenCount <- Nullable(int64 u.PromptTokens)
                    usage.OutputTokenCount <- Nullable(int64 u.CompletionTokens)
                    usage.TotalTokenCount <- Nullable(int64 u.TotalTokens)
                    chatResp.Usage <- usage)

                return chatResp
            }

        member _.GetStreamingResponseAsync(messages: IEnumerable<ChatMessage>, options: ChatOptions, cancellationToken: CancellationToken) : IAsyncEnumerable<ChatResponseUpdate> =
            let inner = inner

            { new IAsyncEnumerable<ChatResponseUpdate> with
                member _.GetAsyncEnumerator(ct) =
                    let tarsMessages =
                        messages
                        |> Seq.map ChatClientMapping.fromAIChatMessage
                        |> Seq.toList

                    let mutable req = Prompt.ofMessages tarsMessages |> Prompt.withStream true

                    if not (isNull options) then
                        req <-
                            { req with
                                Temperature =
                                    options.Temperature
                                    |> Option.ofNullable
                                    |> Option.map float
                                    |> Option.orElse req.Temperature
                                MaxTokens =
                                    options.MaxOutputTokens
                                    |> Option.ofNullable
                                    |> Option.orElse req.MaxTokens
                                Model = options.ModelId |> Option.ofObj |> Option.orElse req.Model
                                // Streaming carries the same sampling controls as the
                                // non-streaming path. Dropping them here meant an
                                // identical ChatOptions produced two different requests
                                // depending only on whether the caller streamed.
                                Stop = ChatClientMapping.stopOf options req.Stop
                                Seed = ChatClientMapping.seedOf options req.Seed
                                Tools = ChatClientMapping.toolsOf options }
                            |> ChatClientMapping.applyFormat options

                    // Tool calls do not survive being flattened into a token stream:
                    // providers put them in the final message, and our streaming path
                    // forwards content only. Offering tools therefore takes the
                    // buffered call and yields it as one update, which keeps the loop
                    // working for a streaming caller at the cost of the tokens
                    // arriving together. Reassembling calls from deltas is #317.
                    let offersTools = not req.Tools.IsEmpty

                    let pending = System.Collections.Concurrent.ConcurrentQueue<ChatResponseUpdate>()
                    let mutable current = ChatResponseUpdate()
                    let mutable completion: Task<LlmResponse> = null
                    let mutable closed = false

                    let textUpdate (text: string) =
                        let update = ChatResponseUpdate()
                        update.Role <- Nullable ChatRole.Assistant
                        update.Contents.Add(TextContent(text))
                        update

                    /// The update that closes the stream: whatever the streamed tokens
                    /// could not carry — the tool calls, the finish reason, the usage.
                    let closingUpdate (response: LlmResponse) =
                        let update = ChatResponseUpdate()
                        update.Role <- Nullable ChatRole.Assistant

                        if offersTools && not (String.IsNullOrEmpty response.Text) then
                            update.Contents.Add(TextContent(response.Text))

                        let calls = ChatClientMapping.toolCallsOf response.Raw

                        for call in calls do
                            update.Contents.Add(FunctionCallContent(call.CallId, call.Name, call.Arguments))

                        if not calls.IsEmpty then
                            update.FinishReason <- Nullable ChatFinishReason.ToolCalls
                        else
                            response.FinishReason
                            |> Option.iter (fun reason ->
                                update.FinishReason <-
                                    Nullable(
                                        match reason with
                                        | "length" -> ChatFinishReason.Length
                                        | "content_filter" -> ChatFinishReason.ContentFilter
                                        | "tool_calls" -> ChatFinishReason.ToolCalls
                                        | _ -> ChatFinishReason.Stop))

                        update

                    { new IAsyncEnumerator<ChatResponseUpdate> with
                        // Handing out the same update twice is what a consumer expects;
                        // dequeuing here meant reading `Current` twice ate a token.
                        member _.Current = current

                        member _.MoveNextAsync() =
                            if isNull completion then
                                completion <-
                                    if offersTools then
                                        inner.CompleteAsync(req)
                                    else
                                        inner.CompleteStreamAsync(req, (fun token -> pending.Enqueue(textUpdate token)))

                            let rec advance () =
                                task {
                                    match pending.TryDequeue() with
                                    | true, update ->
                                        current <- update
                                        return true
                                    | _ ->
                                        if closed then
                                            return false
                                        elif completion.IsCompleted then
                                            closed <- true
                                            // Awaited, not inspected: a provider that
                                            // failed used to end the stream as though
                                            // it had simply finished.
                                            let! response = completion
                                            pending.Enqueue(closingUpdate response)
                                            return! advance ()
                                        else
                                            do! Task.Delay(10, ct)
                                            return! advance ()
                                }

                            ValueTask<bool>(advance ())

                        member _.DisposeAsync() = ValueTask() }
            }

        member _.Dispose() = ()

// ─────────────────────────────────────────────────────────────────────
// Adapter 2: IChatClient -> ILlmService
// Allows any M.E.AI provider to be used through the TARS ILlmService interface.
// ─────────────────────────────────────────────────────────────────────
type ChatClientLlmService(chatClient: IChatClient) =

    interface ILlmService with

        member _.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
            task {
                let messages = ResizeArray<ChatMessage>()

                req.SystemPrompt
                |> Option.iter (fun sp ->
                    messages.Add(ChatMessage(ChatRole.System, sp)))

                for msg in req.Messages do
                    messages.Add(ChatClientMapping.toAIChatMessage msg)

                let options = ChatClientMapping.toChatOptions req

                let! resp = chatClient.GetResponseAsync(messages, options, CancellationToken.None)

                return ChatClientMapping.toLlmResponse resp
            }

        member _.CompleteStreamAsync(req: LlmRequest, onToken: string -> unit) : Task<LlmResponse> =
            task {
                let messages = ResizeArray<ChatMessage>()

                req.SystemPrompt
                |> Option.iter (fun sp ->
                    messages.Add(ChatMessage(ChatRole.System, sp)))

                for msg in req.Messages do
                    messages.Add(ChatClientMapping.toAIChatMessage msg)

                let options = ChatClientMapping.toChatOptions req

                let mutable fullText = ""

                let updates = chatClient.GetStreamingResponseAsync(messages, options, CancellationToken.None)
                let enumerator = updates.GetAsyncEnumerator(CancellationToken.None)

                try
                    let mutable hasMore = true
                    while hasMore do
                        let! next = enumerator.MoveNextAsync()
                        hasMore <- next
                        if hasMore then
                            let update = enumerator.Current
                            let token = update.Text |> Option.ofObj |> Option.defaultValue ""
                            if token <> "" then
                                fullText <- fullText + token
                                onToken token
                finally
                    enumerator.DisposeAsync().AsTask().Wait()

                return
                    { Text = fullText
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }
            }

        member _.EmbedAsync(_text: string) : Task<float32[]> =
            raise (NotSupportedException("IChatClient does not support embeddings. Use IEmbeddingGenerator instead."))

        member _.RouteAsync(_req: LlmRequest) : Task<RoutedBackend> =
            task {
                return
                    { Backend = Ollama "unknown"
                      Endpoint = Uri("http://localhost:11434")
                      ApiKey = None }
            }
