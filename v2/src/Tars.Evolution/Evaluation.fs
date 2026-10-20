namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm

type IEvaluationStrategy =
    abstract member Evaluate: taskDef: TaskDefinition * result: TaskResult -> Task<EvaluationResult>

type SemanticEvaluation(llm: ILlmService, ?minConfidence: float, ?logger: string -> unit) =
    let minConfidence = defaultArg minConfidence 0.6
    let log = defaultArg logger (fun _ -> ())

    let tryGetProperty (name: string) (elem: JsonElement) =
        let mutable prop = Unchecked.defaultof<JsonElement>
        if elem.TryGetProperty(name, &prop) then
            Some prop
        elif elem.ValueKind = JsonValueKind.Object then
            elem.EnumerateObject()
            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun p -> p.Value)
        else
            None

    let getBool name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.True -> Some true
            | JsonValueKind.False -> Some false
            | JsonValueKind.String ->
                match Boolean.TryParse(prop.GetString()) with
                | true, value -> Some value
                | _ -> None
            | _ -> None
        | None -> None

    let getString name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop when prop.ValueKind = JsonValueKind.String -> prop.GetString()
        | _ -> ""

    let getDouble name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Number -> prop.GetDouble()
            | JsonValueKind.String ->
                match Double.TryParse(prop.GetString()) with
                | true, value -> value
                | _ -> 0.0
            | _ -> 0.0
        | None -> 0.0

    let getStringList name (elem: JsonElement) =
        match tryGetProperty name elem with
        | Some prop ->
            match prop.ValueKind with
            | JsonValueKind.Array ->
                prop.EnumerateArray()
                |> Seq.choose (fun item ->
                    if item.ValueKind = JsonValueKind.String then
                        Some(item.GetString())
                    else
                        None)
                |> Seq.toList
            | JsonValueKind.String -> [ prop.GetString() ]
            | _ -> []
        | None -> []

    interface IEvaluationStrategy with
        member _.Evaluate(taskDef, result) =
            task {
                let constraints =
                    if taskDef.Constraints.IsEmpty then
                        "None"
                    else
                        String.concat "; " taskDef.Constraints

                let jsonTemplate =
                    "{\"passed\":true,\"confidence\":0.0,\"summary\":\"...\",\"issues\":[\"...\"],\"suggested_fixes\":[\"...\"]}"

                // With --run-code the examples in the validation criteria may have run with the code and
                // passed (result.Evaluation). They check its values, not what else the goal asks.
                let examplesPassed =
                    match result.Evaluation with
                    | Some examples when examples.Passed ->
                        [ ""
                          $"The code in the output was run with the examples in the validation criteria: {examples.Summary}"
                          "Do not judge its results again. Judge only what the examples do not check, such as how the goal or the constraints say the code must be written. If they ask nothing more, it passes." ]
                    | _ -> []

                let prompt =
                    String.concat
                        "\n"
                        [ "You are a strict evaluator of task outputs."
                          $"Goal: {taskDef.Goal}"
                          $"Constraints: {constraints}"
                          $"Validation Criteria: {taskDef.ValidationCriteria}"
                          yield! examplesPassed
                          "Judge the code, not how the answer presents it: the agent starts its answers with an `ACT:` line and may write text around its code. Ignore both, and any constraint on that presentation (the `ACT:` line, text around the code, the number of code blocks). Constraints on what the code returns or prints still apply."
                          ""
                          "Output:"
                          "```"
                          result.Output
                          "```"
                          ""
                          "Return ONLY this JSON object:"
                          jsonTemplate ]

                let req: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = Some "Evaluate task output for semantic correctness."
                      MaxTokens = Some 400
                      Temperature = Some 0.2
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema EvolutionSchemas.evaluationSchema))
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                try
                    let! response = llm.CompleteAsync(req)

                    match JsonParsing.tryParseElement response.Text with
                    | Ok elem ->
                        let rawPassed = getBool "passed" elem |> Option.defaultValue false
                        let confidence = getDouble "confidence" elem |> max 0.0 |> min 1.0
                        let summary = getString "summary" elem |> fun s -> if String.IsNullOrWhiteSpace s then "No summary provided." else s
                        let issues = getStringList "issues" elem
                        let fixes = getStringList "suggested_fixes" elem

                        let passed =
                            if rawPassed && confidence >= minConfidence then
                                true
                            elif rawPassed then
                                false
                            else
                                false

                        let summary =
                            if rawPassed && confidence < minConfidence then
                                summary + $" (Confidence {confidence:F2} below threshold {minConfidence:F2}.)"
                            else
                                summary

                        return
                            { Passed = passed
                              Confidence = confidence
                              Summary = summary
                              Issues = issues
                              SuggestedFixes = fixes
                              EvaluatedAt = DateTime.UtcNow }
                    | Error err ->
                        log $"[Evaluation] Failed to parse JSON: {err}"

                        return
                            { Passed = false
                              Confidence = 0.0
                              Summary = $"Failed to parse evaluation JSON: {err}"
                              Issues = [ "llm_output_parse_error" ]
                              SuggestedFixes = []
                              EvaluatedAt = DateTime.UtcNow }
                with ex ->
                    log $"[Evaluation] Error during evaluation: {ex.Message}"

                    return
                        { Passed = false
                          Confidence = 0.0
                          Summary = $"Evaluation failed: {ex.Message}"
                          Issues = [ "evaluation_error" ]
                          SuggestedFixes = []
                          EvaluatedAt = DateTime.UtcNow }
            }

/// Runs an answer's code and checks its examples, `call = expected`: the code prints the calls'
/// values, and they are compared with expected values computed in processes without it (#394).
/// Evolve (Engine) and the benchmark (BenchmarkRunner) share it.
module ExampleCheck =

    open System.Threading

    /// Runs an F# script with dotnet fsi and `options` (such as "--typecheck-only "), in a temporary
    /// directory, with no input, for at most `timeout`. Ok with what it printed when it exits with 0;
    /// otherwise Error with what went wrong.
    /// System is opened first: the model writes Char or String.IsNullOrEmpty without opening it.
    /// A leading namespace or top-level module cannot follow that open, but fsi rejects both
    /// anyway, and scriptOfAnswer has already dropped the first and turned the second into
    /// `module X =`, which can. The #line directive keeps error positions on the lines of the
    /// script itself.
    let private runFsiWith (options: string) (timeout: TimeSpan) (script: string) : Task<Result<string, string>> =
        task {
            let dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "tars-evolve-run", Guid.NewGuid().ToString("N"))
            System.IO.Directory.CreateDirectory dir |> ignore
            let path = System.IO.Path.Combine(dir, "answer.fsx")

            try
                System.IO.File.WriteAllText(path, "open System\n#line 1 \"answer.fsx\"\n" + script)
                let psi = System.Diagnostics.ProcessStartInfo("dotnet", $"fsi {options}\"{path}\"")
                psi.WorkingDirectory <- dir
                psi.RedirectStandardInput <- true
                psi.RedirectStandardOutput <- true
                psi.RedirectStandardError <- true
                psi.UseShellExecute <- false
                psi.CreateNoWindow <- true

                use proc = System.Diagnostics.Process.Start psi
                proc.StandardInput.Close()
                // Both streams are read while the script runs, so a full pipe cannot block it.
                let stdout = proc.StandardOutput.ReadToEndAsync()
                let stderr = proc.StandardError.ReadToEndAsync()
                use cts = new CancellationTokenSource(timeout)

                try
                    do! proc.WaitForExitAsync(cts.Token)
                    let! output = stdout
                    let! errors = stderr

                    if proc.ExitCode = 0 then
                        return Result.Ok output
                    else
                        let errors = errors.Replace(dir + string System.IO.Path.DirectorySeparatorChar, "").Trim()

                        return
                            Result.Error(
                                if errors.Length > 2000 then
                                    errors.Substring(0, 2000) + "\n..."
                                else
                                    errors
                            )
                with :? OperationCanceledException ->
                    try
                        proc.Kill true
                    with _ ->
                        ()

                    return Result.Error $"It did not finish within {timeout.TotalSeconds:F0} s."
            finally
                try
                    System.IO.Directory.Delete(dir, true)
                with _ ->
                    ()
        }

    let private runFsi timeout script = runFsiWith "" timeout script

    /// Runs an F# script as runFsi does: Ok when it exits with 0; otherwise Error with what went wrong.
    let runScript (timeout: TimeSpan) (script: string) : Task<Result<unit, string>> =
        task {
            let! run = runFsi timeout script
            return run |> Result.map ignore
        }

    /// Indices of the characters of `text` outside brackets, strings and chars.
    let private topLevel (text: string) : int list =
        let found = ResizeArray<int>()
        let mutable depth = 0
        let mutable i = 0

        while i < text.Length do
            match text.[i] with
            | '"' ->
                i <- i + 1

                while i < text.Length && text.[i] <> '"' do
                    if text.[i] = '\\' then
                        i <- i + 1

                    i <- i + 1
            | '\'' when i + 2 < text.Length && text.[i + 2] = '\'' -> i <- i + 2
            | '(' | '[' | '{' -> depth <- depth + 1
            | ')' | ']' | '}' -> depth <- max 0 (depth - 1)
            | _ ->
                if depth = 0 then
                    found.Add i

            i <- i + 1

        List.ofSeq found

    /// The examples in a task's validation criteria: the parts, split on a `;` or a new line
    /// outside brackets, strings and chars, that call a function the goal names: with its
    /// signature (`isEven : int -> bool`), in the code a refactor gives (`let sign x = ...`), or
    /// quoted after "named" (a function named `countVowels`, a tool named 'StringAnalyzer'). The
    /// curriculum writes them as `isEven 4 = true; isEven 5 = false`.
    let examplesOf (goal: string) (criteria: string) : string list =
        let names =
            System.Text.RegularExpressions.Regex.Matches(
                goal,
                @"`\s*([A-Za-z_]\w*)\s*`?\s*:|`let\s+(?:rec\s+)?([A-Za-z_]\w*)|\bnamed\s+['""`]([A-Za-z_]\w*)['""`]"
            )
            |> Seq.map (fun m ->
                [ 1; 2; 3 ]
                |> List.map (fun i -> m.Groups.[i])
                |> List.find (fun g -> g.Success)
                |> fun g -> g.Value)
            |> Seq.distinct
            |> List.ofSeq

        if names.IsEmpty then
            []
        else
            let cuts =
                topLevel criteria
                |> List.filter (fun i -> criteria.[i] = ';' || criteria.[i] = '\n')

            (-1 :: cuts) @ [ criteria.Length ]
            |> List.pairwise
            |> List.map (fun (a, b) -> criteria.Substring(a + 1, b - a - 1).Trim().Trim('`').Trim())
            |> List.filter (fun part ->
                names
                |> List.exists (fun name ->
                    System.Text.RegularExpressions.Regex.IsMatch(
                        part,
                        "^" + System.Text.RegularExpressions.Regex.Escape name + @"(\s|\(|$)"
                    )))

    /// The call and the expected value of an example `call = expected`, split at its top-level `=`.
    /// None for an example written as a condition.
    let private callAndExpected (example: string) =
        topLevel example
        |> List.tryFind (fun i ->
            example.[i] = '='
            && (i = 0 || not ("<>=!:".Contains example.[i - 1]))
            && (i + 1 >= example.Length || not ("=>".Contains example.[i + 1])))
        |> Option.map (fun at -> example.Substring(0, at).Trim(), example.Substring(at + 1).Trim())

    /// The F# line that checks one example. For `call = expected` it reports the value it got.
    /// It names F#'s own `<>` and `not`, which the answer before it may have redefined.
    let private exampleCheck (example: string) =
        let quoted =
            "\"" + example.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

        match callAndExpected example with
        | Some(call, expected) ->
            // A match, not a let: a let of `reverse []` would not compile (value restriction).
            $"match ({call}) with actual when global.Microsoft.FSharp.Core.Operators.(<>) actual ({expected}) -> failwithf \"Example failed: %%s, got %%A\" {quoted} actual | _ -> ()"
        | None -> $"if global.Microsoft.FSharp.Core.Operators.not ({example}) then failwithf \"Example failed: %%s\" {quoted}"

    /// What running an answer's code followed by checks of its examples showed.
    type ExampleRun =
        /// Every example gave the expected value.
        | ExamplesPassed
        /// The code ran and an example gave another value: "Example failed: <example>, got <value>".
        | ExampleFailed of failure: string
        /// The code did not compile, threw, did not finish or ended the script: what went wrong.
        | CodeFailed of errors: string
        /// The examples do not compile against the code (a name or a signature it does not have),
        /// so nothing ran: they say nothing about it.
        | ExamplesUnusable
        /// The examples passed, but in the code's process: the process without the code did not
        /// compute their expected values (they use the code's types, or it failed), or computed
        /// values that differ from one process to another, or the type check did not give the type
        /// each example compares at, so this process could not compare them.
        /// `reason` is the first line of that process's errors.
        | ExamplesNotCompared of reason: string

    /// The first line of `errors` that is not blank.
    let private firstLine (errors: string) =
        errors.Split('\n')
        |> Array.map (fun l -> l.Trim())
        |> Array.tryFind (fun l -> l <> "")
        |> Option.defaultValue errors

    let private endedEarly =
        "The script ended inside the code, so the examples after it never ran (does the code call exit?)."

    /// F# that writes a value in full, for valueLine, so that two values get the same text when `=`
    /// holds them equal, and only then. It writes every item of a list, array, map or set (%A stops
    /// after 100), with the bounds of an array of several dimensions, the fields of tuples, records
    /// and unions, and strings escaped. Numbers are written by value alone: the type check in
    /// runWithExamples holds both sides to types `=` accepts, and `=` widens an int to a float. A
    /// float zero has one text, as -0.0 = 0.0, and a decimal no trailing zeros. A date is written by
    /// its ticks, and a date with an offset by its instant. A value `=` compares by reference, such
    /// as a ResizeArray or a seq, or one it cannot write exactly, such as NaN, which equals nothing,
    /// stops the script, so its examples are not compared. Where a value's type holds obj or an
    /// abstract type (an interface, System.Array), `=` compares runtime types: a boxed 2 is not a
    /// boxed 2L, though two arrays with equal items are equal. So `typed` also writes the runtime
    /// type of each value but an array, and tarsValueLine prints both texts, and whether the
    /// static type holds obj or an abstract type.
    let private valueText =
        """let rec tarsValueText (typed: bool) (value: obj) : string =
    if typed && not (isNull value) && not (value :? System.Array) then value.GetType().FullName + " " + tarsValueOf typed value
    else tarsValueOf typed value
and tarsValueOf (typed: bool) (value: obj) : string =
    let items (xs: obj) =
        System.String.Join("; ", Seq.cast<obj> (xs :?> System.Collections.IEnumerable) |> Seq.map (tarsValueText typed))
    let invariant = System.Globalization.CultureInfo.InvariantCulture
    let t = if isNull value then null else value.GetType()
    let definition = if isNull t || not t.IsGenericType then null else t.GetGenericTypeDefinition()
    match value with
    | null -> "null"
    | :? string as s -> System.String.Concat("\"", s.Replace("\\", "\\\\").Replace("\"", "\\\""), "\"")
    | :? float as f when System.Double.IsNaN f -> failwith "NaN equals nothing, not even itself"
    | :? float32 as f when System.Single.IsNaN f -> failwith "NaN equals nothing, not even itself"
    | :? float as f -> if f = 0.0 then "0" else f.ToString("R", invariant)
    | :? float32 as f -> if f = 0.0f then "0" else f.ToString("R", invariant)
    | :? decimal as d -> (d / 1.0000000000000000000000000000M).ToString(invariant)
    | :? System.Numerics.BigInteger as b -> b.ToString(invariant)
    | :? System.DateTime as d -> "DateTime " + string d.Ticks
    | :? System.DateTimeOffset as d -> "DateTimeOffset " + string d.UtcTicks
    | :? System.TimeSpan as d -> "TimeSpan " + string d.Ticks
    | :? bool | :? char | :? System.Guid -> sprintf "%A" value
    | _ when t.IsEnum -> sprintf "%A" value
    | _ when t.IsPrimitive -> System.Convert.ToString(value, invariant)
    | :? System.Array as a ->
        let shape =
            if a.Rank = 1 && a.GetLowerBound 0 = 0 then ""
            else System.String.Join(", ", [ for d in 0 .. a.Rank - 1 -> string (a.GetLowerBound d) + ".." + string (a.GetUpperBound d) ]) + ": "
        "[|" + shape + items value + "|]"
    | _ when definition = typedefof<list<_>> -> "[" + items value + "]"
    | _ when definition = typedefof<Map<_, _>> -> "map [" + items value + "]"
    | _ when definition = typedefof<Set<_>> -> "set [" + items value + "]"
    | _ when definition = typedefof<System.Collections.Generic.KeyValuePair<_, _>> ->
        "(" + tarsValueText typed (t.GetProperty("Key").GetValue value) + ", " + tarsValueText typed (t.GetProperty("Value").GetValue value) + ")"
    | _ when Microsoft.FSharp.Reflection.FSharpType.IsTuple t ->
        "(" + System.String.Join(", ", Microsoft.FSharp.Reflection.FSharpValue.GetTupleFields value |> Array.map (tarsValueText typed)) + ")"
    | _ when Microsoft.FSharp.Reflection.FSharpType.IsRecord t ->
        let names = Microsoft.FSharp.Reflection.FSharpType.GetRecordFields t |> Array.map (fun p -> p.Name)
        let values = Microsoft.FSharp.Reflection.FSharpValue.GetRecordFields value |> Array.map (tarsValueText typed)
        "{ " + System.String.Join("; ", Array.map2 (fun n v -> n + " = " + v) names values) + " }"
    | _ when Microsoft.FSharp.Reflection.FSharpType.IsUnion t ->
        let case, fields = Microsoft.FSharp.Reflection.FSharpValue.GetUnionFields(value, t)
        if fields.Length = 0 then case.Name else case.Name + " (" + System.String.Join(", ", fields |> Array.map (tarsValueText typed)) + ")"
    | _ -> failwithf "No exact text for a value of type %s" t.FullName
let tarsOpen (t: System.Type) =
    let seen = System.Collections.Generic.HashSet<System.Type>()
    let rec isOpen (t: System.Type) =
        if not (seen.Add t) then false
        elif t = typeof<obj> || (t.IsAbstract && not (Microsoft.FSharp.Reflection.FSharpType.IsUnion t)) then true
        elif t.IsArray then isOpen (t.GetElementType())
        elif Microsoft.FSharp.Reflection.FSharpType.IsRecord t then
            Microsoft.FSharp.Reflection.FSharpType.GetRecordFields t |> Array.exists (fun p -> isOpen p.PropertyType)
        elif Microsoft.FSharp.Reflection.FSharpType.IsUnion t then
            Microsoft.FSharp.Reflection.FSharpType.GetUnionCases t
            |> Array.exists (fun c -> c.GetFields() |> Array.exists (fun p -> isOpen p.PropertyType))
        else t.IsGenericType && Array.exists isOpen (t.GetGenericArguments())
    isOpen t
type TarsProbe = | TarsProbe
type TarsSlot<'T> = TarsSlot of 'T
let tarsTyped (value: 'T) (_: 'T -> bool) = value
let tarsValueLine (mark: string) (index: int) (value: 'T) =
    let base64 (text: string) = System.Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes text)
    printfn "\n%s %d %s %s %b" mark index (base64 (tarsValueText false (box value))) (base64 (tarsValueText true (box value))) (tarsOpen typeof<'T>)"""

    /// The F# line that prints the value of `expr` with tarsValueLine, on a line of its own: `mark`,
    /// the example's index, the value's text and typed text, in base64 so they stay on that line,
    /// and whether its static type holds obj or an abstract type.
    let private valueLine (mark: string) (index: int) (expr: string) =
        $"V{mark}.tarsValueLine \"{mark}\" {index} ({expr})"

    /// The call of an example whose expected value is `expected`, written next to `expected` in a
    /// function that is never called, so that it gets the type the example gives it: `make () = 0`
    /// makes `make ()` an int, which alone would hold obj. The expected value is not computed.
    /// Only the type check writes it: a script the code runs from would show the code `expected`.
    let private typedCall (mark: string) (call: string) (expected: string) =
        $"V{mark}.tarsTyped ({call}) (fun tarsCalled{mark} -> global.Microsoft.FSharp.Core.Operators.(=) tarsCalled{mark} ({expected}))"

    /// F# after the examples that makes the type check report the type `=` compares each example
    /// at, so that the expected values can be computed at it: alone, `Unchecked.defaultof<_>` holds
    /// obj where `f () = Unchecked.defaultof<_>` has an int. Each call, typed as typedCall types
    /// it, goes in a tuple of TarsSlot, which is then given a type it cannot have. fsi stops at that
    /// error, which names the tuple's type.
    let private typeProbe (mark: string) (pairs: (string * string) list) =
        let slots =
            pairs
            |> List.map (fun (call, expected) -> $"V{mark}.TarsSlot ({typedCall mark call expected})")
            |> String.concat ", "

        $"let tarsTypes{mark} () = ({slots})\nlet _ : V{mark}.TarsProbe = tarsTypes{mark} ()"

    /// The types typeProbe's error gives its slots, in order. They are read after the TarsProbe the
    /// error names first, because the words between depend on the compiler: for a tuple, the
    /// .NET 10.0.402 SDK says "but is a tuple of type" where the .NET 11 preview says "but here
    /// has type". None when it does not give one for each of `count` examples, or gives one with a
    /// backtick or a quote, which F# prints only in a name the answer chose.
    let private probedTypes (mark: string) (count: int) (errors: string) : string list option =
        let probe = $"V{mark}.TarsProbe"
        let at = errors.IndexOf(probe, StringComparison.Ordinal)
        let slot = $"V{mark}.TarsSlot<"

        // The slot's type ends at the > that closes its <.
        let rec close (text: string) i depth =
            if i >= text.Length then None
            elif text.[i] = '<' then close text (i + 1) (depth + 1)
            elif text.[i] = '>' then (if depth = 0 then Some i else close text (i + 1) (depth - 1))
            else close text (i + 1) depth

        let rec slots (text: string) from found =
            match text.IndexOf(slot, from, StringComparison.Ordinal) with
            | -1 -> List.rev found
            | start ->
                let first = start + slot.Length

                match close text first 0 with
                | Some last -> slots text (last + 1) (text.Substring(first, last - first) :: found)
                | None -> List.rev found

        if at < 0 then
            None
        else
            let types = slots (errors.Substring(at + probe.Length)) 0 []

            if types.Length = count && types |> List.forall (fun t -> not (t.Contains '`' || t.Contains '"')) then
                Some types
            else
                None

    /// valueText in a module named after `mark`, which valueLine names, so a name the answer
    /// defines cannot hide it.
    let private valuePrinter (mark: string) =
        $"module V{mark} =\n" + (valueText.Split('\n') |> Array.map (fun line -> "    " + line) |> String.concat "\n")

    /// A value a run printed with valueLine.
    type private PrintedValue =
        { Text: string
          Typed: string
          /// Its static type holds obj or an abstract type. On the code's side, the type of the call
          /// fixes the type `=` compares at, so there `=` compares runtime types.
          Open: bool }

    /// The values a run printed with valueLine, decoded, by example index. Other lines, and lines
    /// that do not decode, are left out.
    let private printedValues (mark: string) (output: string) : Map<int, PrintedValue list> =
        let decode (value: string) =
            try
                Some(System.Text.Encoding.UTF8.GetString(Convert.FromBase64String value))
            with :? FormatException ->
                None

        output.Split('\n')
        |> Array.choose (fun line ->
            match line.Trim().Split(' ') with
            | [| m; index; text; typed; isOpen |] when m = mark ->
                match Int32.TryParse index, decode text, decode typed with
                | (true, i), Some text, Some typed -> Some(i, { Text = text; Typed = typed; Open = (isOpen = "true") })
                | _ -> None
            | _ -> None)
        |> Array.toList
        |> List.groupBy fst
        |> List.map (fun (i, values) -> i, List.map snd values)
        |> Map.ofList

    /// What the values the code printed for its examples show against the expected values, both
    /// by example index.
    let private compareValues
        (examples: string list)
        (expected: Map<int, PrintedValue list>)
        (actual: Map<int, PrintedValue list>)
        =
        let printed = examples |> List.mapi (fun i example -> i, example, Map.tryFind i actual)

        if printed |> List.exists (fun (_, _, values) -> values.IsNone) then
            CodeFailed endedEarly
        else
            printed
            |> List.tryPick (fun (i, example, values) ->
                match values, Map.tryFind i expected with
                | Some [ value ], Some [ wanted ] when (if value.Open then value.Typed = wanted.Typed else value.Text = wanted.Text) ->
                    None
                | Some [ value ], _ ->
                    let got = if value.Open then value.Typed else value.Text
                    Some $"Example failed: {example}, got {got}"
                | _, _ -> Some $"Example failed: {example}, the code printed more than one value for it")
            |> Option.map ExampleFailed
            |> Option.defaultValue ExamplesPassed

    /// Runs `script` followed by its examples, and says what they show (#394). Three dotnet fsi
    /// processes share `timeout`, the caller's deadline, and the code runs in the last only:
    /// - the first type-checks the code followed by the examples' checks, and runs nothing: `=` did
    ///   not compile across types, and the values compared below as text do not show them. It also
    ///   reports the type `=` compares each example at (typeProbe);
    /// - the second prints the examples' expected values at those types, without the code; an
    ///   example written as a condition is expected to be true;
    /// - the code's process prints only the values of the examples' calls, and this process
    ///   compares them. Each call is given the type the type check reported, `((call) : int)`, so
    ///   that process's script holds no expected value, and the code passes only by printing the
    ///   right values. The examples are in evolve's executor prompt, though, so there a pass does
    ///   not show that the code computed them rather than printed values it was shown.
    /// When the expected values do not compute without the code, the code's process runs the
    /// checks itself, which end by printing a mark, since fsi also exits with 0 when the code
    /// calls `exit 0`. The code can read that mark from its script, so there a failure counts and
    /// a pass does not.
    let runWithExamples (timeout: TimeSpan) (script: string) (examples: string list) : Task<ExampleRun> =
        task {
            let mark = Guid.NewGuid().ToString("N")

            let pairs =
                examples
                |> List.map (fun example -> callAndExpected example |> Option.defaultValue (example, "true"))

            let valueLines exprs =
                List.mapi (valueLine mark) exprs |> String.concat "\n"

            let checks =
                (examples |> List.map exampleCheck |> String.concat "\n") + $"\nprintfn \"{mark}\""

            let withExamples lines =
                script + "\n#line 1 \"examples.fsx\"\n" + lines

            let started = System.Diagnostics.Stopwatch.StartNew()
            let left () = max TimeSpan.Zero (timeout - started.Elapsed)

            let compileErrorIn (file: string) (errors: string) =
                System.Text.RegularExpressions.Regex.IsMatch(
                    errors,
                    System.Text.RegularExpressions.Regex.Escape file + @"\(\d+,\d+\): error"
                )

            let fromErrors (errors: string) =
                let failed = System.Text.RegularExpressions.Regex.Match(errors, @"Example failed: [^\r\n]*")

                if compileErrorIn "answer.fsx" errors then CodeFailed errors
                elif compileErrorIn "examples.fsx" errors then ExamplesUnusable
                elif failed.Success then ExampleFailed failed.Value
                else CodeFailed errors

            let printer = valuePrinter mark
            let beforeAnswer = "#line 1 \"values.fsx\"\n" + printer + "\n#line 1 \"answer.fsx\"\n"

            // The type check goes first, as the expected values are computed at the types it reports.
            // Its probe always fails, so a passing check stops in types.fsx; when it fails before,
            // nothing else starts.
            let! typed =
                runFsiWith
                    "--typecheck-only --preferreduilang:en-US "
                    (left ())
                    (beforeAnswer + withExamples checks + "\n#line 1 \"types.fsx\"\n" + typeProbe mark pairs)

            match typed with
            | Result.Error errors when not (compileErrorIn "types.fsx" errors) -> return fromErrors errors
            | _ ->
                let types =
                    match typed with
                    | Result.Error errors -> probedTypes mark pairs.Length errors
                    | Result.Ok _ -> None

                // They are computed in two processes at once: values that differ from one process to
                // another, such as a process id or a working directory, cannot be compared with the
                // code's.
                let! expected =
                    match types with
                    | None ->
                        Task.FromResult(Result.Error "The type check did not give the type each example compares at.")
                    | Some types ->
                        task {
                            let expectedValues =
                                printer
                                + "\n"
                                + valueLines (List.map2 (fun (_, expected) t -> $"({expected}) : {t}") pairs types)

                            let computing = runFsi (left ()) expectedValues
                            let computingAgain = runFsi (left ()) expectedValues
                            let! first = computing
                            let! second = computingAgain

                            return
                                match first, second with
                                | Result.Ok one, Result.Ok other when printedValues mark one <> printedValues mark other ->
                                    Result.Error "The expected values differ from one process to another."
                                | Result.Ok _, Result.Error errors -> Result.Error errors
                                | _ -> first
                        }

                let! run =
                    match expected, types with
                    | Result.Ok _, Some types ->
                        // The printer goes before the code, so the names it uses are F#'s, not ones the
                        // code redefined.
                        let values =
                            withExamples (valueLines (List.map2 (fun (call, _) t -> $"({call}) : {t}") pairs types))
                        runFsi (left ()) (beforeAnswer + values)
                    | _ -> runFsi (left ()) (withExamples checks)

                match run, expected with
                | Result.Ok output, Result.Ok expectedOutput ->
                    return compareValues examples (printedValues mark expectedOutput) (printedValues mark output)
                | Result.Ok output, Result.Error errors when output.Contains mark ->
                    return ExamplesNotCompared(firstLine errors)
                | Result.Ok _, Result.Error _ -> return CodeFailed endedEarly
                | Result.Error errors, _ -> return fromErrors errors
        }

    /// The evaluation an examples run gives: None when the examples say nothing about the code.
    let verdictOfExamples (count: int) (run: ExampleRun) : EvaluationResult option =
        let verdict passed (summary: string) =
            Some
                { Passed = passed
                  Confidence = 1.0
                  Summary = summary
                  Issues = (if passed then [] else [ summary ])
                  SuggestedFixes = []
                  EvaluatedAt = DateTime.UtcNow }

        match run with
        | ExamplesPassed -> verdict true $"The code ran and gave the expected value for all {count} examples."
        | ExampleFailed failure -> verdict false failure
        | CodeFailed errors -> verdict false $"The code did not run: {firstLine errors}"
        | ExamplesUnusable
        | ExamplesNotCompared _ -> None

    /// Runs `script` and then checks `examples`. Some verdict when they ran: passed when each gave
    /// the expected value; failed when one did not, or the code did not compile or run. None when
    /// there are no examples, or they do not compile against the code.
    let checkExamples (timeout: TimeSpan) (script: string) (examples: string list) : Task<EvaluationResult option> =
        task {
            if examples.IsEmpty then
                return None
            else
                let! run = runWithExamples timeout script examples
                return verdictOfExamples examples.Length run
        }
