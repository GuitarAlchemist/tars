namespace Tars.Evolution

open System
open System.Threading
open System.Threading.Tasks
open Tars.Core
open Tars.Core.Knowledge
open Tars.Graph
open Tars.Llm
open System.Text.Json
open Tars.Kernel
open Tars.Cortex
open Tars.Connectors.EpisodeIngestion
open Tars.Knowledge
open Tars.Core.WorkflowOfThought

module Engine =

    /// Item to be buffered for memory storage
    type MemoryItem =
        | Belief of collection: string * id: string * vector: float32[] * payload: Map<string, string>
        | Legacy of collection: string * id: string * vector: float32[] * payload: Map<string, string>

    type MemoryServices =
        { SemanticMemory: ISemanticMemory option
          KnowledgeBase: KnowledgeBase option
          KnowledgeGraph: TemporalKnowledgeGraph.TemporalGraph option
          MemoryBuffer: BufferAgent<MemoryItem> option
          EpisodeService: IEpisodeIngestionService option
          Ledger: KnowledgeLedger option
          EvidenceStore: IEvidenceStore option }

    type GovernanceServices =
        { Epistemic: IEpistemicGovernor option
          PreLlm: PreLlmPipeline option
          Budget: BudgetGovernor option
          OutputGuard: IOutputGuard option
          Evaluator: IEvaluationStrategy option }

    type RunOptions =
        { RunId: RunId option
          Verbose: bool
          ShowSemanticMessage: Message -> bool -> unit
          Focus: string option
          ToolRegistry: Tars.Tools.ToolRegistry option
          ResearchEnhanced: bool
          SelfImprovement: bool
          /// Run the F# code in the executor's answers with dotnet fsi (`evolve --run-code`), with the
          /// task's examples, and ask for new answers while they fail. It runs with the user's rights,
          /// outside any sandbox, so it is off unless asked for.
          RunCode: bool }

    /// The context for the evolution engine
    type EvolutionContext =
        { Registry: IAgentRegistry
          Llm: ILlmService
          /// The model that writes the tasks (`evolve --teacher`). None: Llm writes them too.
          CurriculumLlm: ILlmService option
          VectorStore: IVectorStore
          Logger: string -> unit
          Memory: MemoryServices
          Governance: GovernanceServices
          Options: RunOptions }

    let private scoreTask
        (ctx: EvolutionContext)
        (taskDef: TaskDefinition)
        (recentVectors: float32[] list)
        : Task<float> =
        task {
            // Base Score: Difficulty gives a small boost
            let baseScore = 1.0 + (0.1 * float taskDef.DifficultyLevel)

            if recentVectors.IsEmpty then
                return baseScore
            else
                try
                    // Calculate embedding for the new task
                    let! currentVector = ctx.Llm.EmbedAsync(taskDef.Goal)

                    // Find maximum similarity to any recent task
                    let maxSimilarity =
                        recentVectors
                        |> List.map (fun v -> MetricSpace.cosineSimilarity currentVector v)
                        |> List.max

                    // Semantic Fan-out Limiting:
                    // If similarity is too high (> 0.85), it means we are repeating ourselves.
                    // Penalty scales with similarity.
                    let penalty =
                        if maxSimilarity > 0.9f then 10.0 // Hard block (score becomes negative)
                        elif maxSimilarity > 0.8f then 3.0 // Strong discouragement
                        elif maxSimilarity > 0.7f then 0.5 // Minimal discouragement
                        else 0.0 // Novel task

                    let finalScore = baseScore - penalty
                    // ctx.Logger($"[Scoring] '{taskDef.Goal}' Sim: {maxSimilarity:F2} -> Score: {finalScore:F2}")
                    return finalScore
                with ex ->
                    ctx.Logger($"[Scoring] Failed for '{taskDef.Goal}': {ex.Message}")
                    return baseScore
        }

    let private tryGetPropertyInsensitive (name: string) (elem: JsonElement) =
        if elem.ValueKind <> JsonValueKind.Object then
            None
        else
            elem.EnumerateObject()
            |> Seq.tryFind (fun p -> p.Name.Equals(name, StringComparison.OrdinalIgnoreCase))
            |> Option.map (fun p -> p.Value)

    let private readBoolFromJson names (elem: JsonElement) =
        names
        |> List.tryPick (fun name ->
            match tryGetPropertyInsensitive name elem with
            | Some prop ->
                match prop.ValueKind with
                | JsonValueKind.True -> Some true
                | JsonValueKind.False -> Some false
                | JsonValueKind.String ->
                    match Boolean.TryParse(prop.GetString()) with
                    | true, value -> Some value
                    | _ -> None
                | _ -> None
            | None -> None)

    let private readStringFromJson names (elem: JsonElement) =
        names
        |> List.tryPick (fun name ->
            match tryGetPropertyInsensitive name elem with
            | Some prop when prop.ValueKind = JsonValueKind.String -> Some(prop.GetString())
            | _ -> None)

    let private looksLikeFollowUpRequest (text: string) =
        if String.IsNullOrWhiteSpace text then
            false
        else
            let trimmed = text.Trim()
            let lowered = trimmed.ToLowerInvariant()

            trimmed.EndsWith("?")
            || lowered.StartsWith("please provide")
            || lowered.StartsWith("could you")
            || lowered.StartsWith("can you")
            || lowered.StartsWith("would you")
            || lowered.StartsWith("i need")
            || lowered.StartsWith("i require")
            || lowered.StartsWith("share the")
            || lowered.StartsWith("send the")

    let private tryExtractJsonElement (text: string) =
        // Remove conversational filler
        let cleanText =
            let lowered = text.ToLowerInvariant()
            let filler = [ "certainly", "here is", "here are", "sure", "ok", "i can help" ]
            let mutable result = text

            if text.StartsWith("ACT:") then
                let idx = text.IndexOf(":", 4)

                if idx > 0 then
                    result <- text.Substring(idx + 1).Trim()

            // If it still looks like it has filler before { or [
            let firstBrace = result.IndexOfAny([| '{'; '[' |])

            if firstBrace > 0 then
                result <- result.Substring(firstBrace).Trim()

            result

        let trimmed = cleanText.Trim()

        let tryParse payload =
            match JsonParsing.tryParseElement payload with
            | Result.Ok elem -> Some elem
            | Result.Error _ -> None

        let tryExtract (openChar: char, closeChar: char) =
            let startIdx = trimmed.IndexOf(openChar)
            let endIdx = trimmed.LastIndexOf(closeChar)

            if startIdx >= 0 && endIdx > startIdx then
                trimmed.Substring(startIdx, endIdx - startIdx + 1) |> Some
            else
                None

        match tryParse trimmed with
        | Some elem -> Result.Ok elem
        | None ->
            match tryExtract ('{', '}') |> Option.bind tryParse with
            | Some elem -> Result.Ok elem
            | None ->
                match tryExtract ('[', ']') |> Option.bind tryParse with
                | Some elem -> Result.Ok elem
                | None -> Result.Error "Response was not valid JSON."

    /// Whether the curriculum agent's answer holds a task list the parser below can
    /// read: a JSON array of tasks, or an object with a "tasks" array, where every
    /// task has a string "goal", an array of strings "constraints" and a string
    /// "validation_criteria". Anything else, whether prose ("Please provide the
    /// current context...") or JSON of another shape, is retried as a direct request
    /// constrained to the task schema rather than replaced by a canned task.
    let curriculumAnswerHasTasks (text: string) =
        let has (t: JsonElement) (name: string) (kind: JsonValueKind) =
            let mutable value = Unchecked.defaultof<JsonElement>
            t.TryGetProperty(name, &value) && value.ValueKind = kind

        let isTask (t: JsonElement) =
            t.ValueKind = JsonValueKind.Object
            && has t "goal" JsonValueKind.String
            && has t "validation_criteria" JsonValueKind.String
            && has t "constraints" JsonValueKind.Array
            && (t.GetProperty("constraints").EnumerateArray()
                |> Seq.forall (fun c -> c.ValueKind = JsonValueKind.String))

        let taskList (root: JsonElement) =
            if root.ValueKind = JsonValueKind.Array then
                Some root
            elif root.ValueKind = JsonValueKind.Object && has root "tasks" JsonValueKind.Array then
                Some(root.GetProperty("tasks"))
            else
                None

        not (String.IsNullOrWhiteSpace text)
        && (match tryExtractJsonElement text with
            | Result.Ok root ->
                match taskList root with
                | Some tasks -> tasks.GetArrayLength() > 0 && (tasks.EnumerateArray() |> Seq.forall isTask)
                | None -> false
            | Result.Error _ -> false)

    /// Whether an executor answer shows code: a fenced block other than a tool call. The
    /// semantic evaluation reads only the answer, so a solution that is described, planned
    /// or asked about, rather than shown, fails it.
    let answerHasCode (answer: string) =
        System.Text.RegularExpressions.Regex.Matches(answer, @"```([^\n`]*)\n[\s\S]*?```")
        |> Seq.exists (fun m -> m.Groups.[1].Value.Trim() <> "tool")

    /// Whether a task asks for code: a verb that makes something (write, implement, create,
    /// refactor...) followed by a code artifact or "F#". A task that scans, analyzes or
    /// summarizes ("Read Domain.fs and write a summary...") wants prose, so an answer without
    /// code is not sent back for it.
    let taskAsksForCode (goal: string) =
        System.Text.RegularExpressions.Regex.IsMatch(
            goal,
            @"\b(write|implement|create|build|refactor|fix|add|modify|extend)\b.*\b(functions?|methods?|modules?|types?|class(es)?|tools?|scripts?|code|programs?|parsers?|tests?|F#)",
            System.Text.RegularExpressions.RegexOptions.IgnoreCase
        )

    /// Whether a task can be done from its own text. In live evolve runs, every task that asked
    /// to refactor or fix code it did not include ("Refactor the existing code...", "Refactor the
    /// provided F# code...") or to create a tool it did not name ("Create a tool for analyzing F#
    /// code complexity") failed: the executor asked for the code or the specification, or refused.
    let taskIsSpecified (taskDef: TaskDefinition) =
        let isMatch (input: string) (pattern: string) =
            System.Text.RegularExpressions.Regex.IsMatch(
                input,
                pattern,
                System.Text.RegularExpressions.RegexOptions.IgnoreCase
            )

        let text = String.concat "\n" (taskDef.Goal :: taskDef.Constraints)

        let needsCode =
            isMatch taskDef.Goal @"\b(refactor|rewrite|optimi[sz]e|improve|fix)"
            || isMatch
                taskDef.Goal
                @"\b(existing|provided|given|current|previous|failing)\s+(\w+\s+){0,2}(code|codebase|function|module|implementation|tool)\b"

        let hasCode = isMatch text @"```|`[^`\n]*\b(let|fun|match|type)\b[^`\n]*`"
        let isTool = isMatch taskDef.Goal @"\btools?\b"
        let namesTool = isMatch taskDef.Goal @"['""`][a-z_][a-z0-9_]*['""`]"
        (not needsCode || hasCode) && (not isTool || namesTool)

    /// Concrete exercises, as (goal, validation criteria), for when the curriculum gives no task
    /// that is new and specified. Each names its function and gives examples to check it by.
    let concreteTasks =
        [ ("Write `isPalindrome : string -> bool` in F#. It ignores case and every character that is not a letter or a digit.",
           "isPalindrome \"A man, a plan, a canal: Panama\" = true; isPalindrome \"race a car\" = false; isPalindrome \"\" = true")
          ("Write `fizzBuzz : int -> string` in F#: \"Fizz\" for multiples of 3, \"Buzz\" for multiples of 5, \"FizzBuzz\" for both, otherwise the number.",
           "fizzBuzz 9 = \"Fizz\"; fizzBuzz 10 = \"Buzz\"; fizzBuzz 15 = \"FizzBuzz\"; fizzBuzz 7 = \"7\"")
          ("Write `gcd : int -> int -> int` in F#, using Euclid's algorithm.",
           "gcd 48 18 = 6; gcd 17 5 = 1; gcd 0 9 = 9")
          ("Write `isPrime : int -> bool` in F#.",
           "isPrime 2 = true; isPrime 15 = false; isPrime 97 = true; isPrime 1 = false")
          ("Write `binarySearch : int[] -> int -> int option` in F#. The array is sorted; the result is the index of the value.",
           "binarySearch [| 1; 3; 5; 7 |] 5 = Some 2; binarySearch [| 1; 3; 5; 7 |] 4 = None; binarySearch [||] 1 = None")
          ("Write `wordCount : string -> Map<string, int>` in F#. Words are separated by spaces and compared in lower case.",
           "wordCount \"a B a\" = Map [ \"a\", 2; \"b\", 1 ]; wordCount \"\" = Map.empty")
          ("Write `runLength : string -> (char * int) list` in F#, the run-length encoding of a string.",
           "runLength \"aaabcc\" = [ ('a', 3); ('b', 1); ('c', 2) ]; runLength \"\" = []")
          ("Write `romanToInt : string -> int` in F#, for Roman numerals up to 3999.",
           "romanToInt \"III\" = 3; romanToInt \"XIV\" = 14; romanToInt \"MCMXC\" = 1990")
          ("Write `balanced : string -> bool` in F#: whether the brackets (), [] and {} in a string are balanced.",
           "balanced \"([]{})\" = true; balanced \"([)]\" = false; balanced \"((\" = false")
          ("Write `flatten : int list list -> int list` in F#, without List.concat or List.collect.",
           "flatten [ [ 1; 2 ]; []; [ 3 ] ] = [ 1; 2; 3 ]; flatten [] = []")
          ("Refactor this function in F# to use pattern matching instead of if/elif, keeping its name and results: `let sign x = if x > 0 then 1 elif x < 0 then -1 else 0`",
           "sign 5 = 1; sign -3 = -1; sign 0 = 0; the body uses match")
          ("Refactor this function in F# to be tail-recursive, keeping its name and results: `let rec sumTo (n: int64) = if n <= 0L then 0L else n + sumTo (n - 1L)`",
           "sumTo 0L = 0L; sumTo 10L = 55L; sumTo 1000000L = 500000500000L; no stack overflow") ]

    /// The first concrete task whose goal is not among `completedGoals`. When every one is done,
    /// they come round again.
    let nextConcreteTask (completedGoals: string list) =
        let finished =
            completedGoals |> List.map (fun g -> g.Trim().ToLowerInvariant()) |> Set.ofList

        concreteTasks
        |> List.tryFind (fun (goal, _) -> not (finished.Contains(goal.ToLowerInvariant())))
        |> Option.defaultWith (fun () -> concreteTasks.[completedGoals.Length % concreteTasks.Length])

    /// What a line of code is in, where it starts.
    type private Literal =
        | Code
        | TripleQuoted
        | Verbatim
        | Ordinary

    /// For each line, whether it starts inside a string an earlier line opened: triple-quoted,
    /// verbatim or ordinary. What follows `//` is a comment.
    let private continuesString (lines: string array) : bool array =
        let starts = Array.zeroCreate lines.Length
        let mutable literal = Code

        for n in 0 .. lines.Length - 1 do
            let line = lines.[n]
            starts.[n] <- literal <> Code
            let mutable i = 0

            while i < line.Length do
                let rest = line.Substring i
                let at (s: string) = rest.StartsWith(s, StringComparison.Ordinal)

                match literal with
                | TripleQuoted when at "\"\"\"" ->
                    literal <- Code
                    i <- i + 3
                | Verbatim when at "\"\"" -> i <- i + 2
                | Verbatim when at "\"" ->
                    literal <- Code
                    i <- i + 1
                | Ordinary when at "\\" -> i <- i + 2
                | Ordinary when at "\"" ->
                    literal <- Code
                    i <- i + 1
                | Code when at "//" -> i <- line.Length
                | Code when at "\"\"\"" ->
                    literal <- TripleQuoted
                    i <- i + 3
                | Code when at "@\"" ->
                    literal <- Verbatim
                    i <- i + 2
                | Code when at "\"" ->
                    literal <- Ordinary
                    i <- i + 1
                | Code when at "'" && rest.Length > 2 && rest.[2] = '\'' -> i <- i + 3
                | Code when at "'\\" && rest.Length > 3 && rest.[3] = '\'' -> i <- i + 4
                | _ -> i <- i + 1

        starts

    /// One fenced block as dotnet fsi accepts it. A .fs file may start with a namespace line
    /// or a top-level `module X`, and fsi rejects both: the namespace line is dropped, and the
    /// module becomes `module X =` with the rest of the block indented under it, then opened, so
    /// the next blocks and the examples see its functions as the answer wrote them. A line that
    /// continues a multi-line string is not indented: the examples compare its text.
    let private asScript (block: string) =
        let lines =
            block.Replace("\r\n", "\n").TrimEnd().Split('\n')
            |> Array.filter (fun l -> not (System.Text.RegularExpressions.Regex.IsMatch(l, @"^namespace\s")))

        let topModule =
            lines
            |> Array.tryFindIndex (fun l -> System.Text.RegularExpressions.Regex.IsMatch(l, @"^module\s+(rec\s+)?[\w.]+\s*$"))

        match topModule with
        | Some i ->
            let name = lines.[i].Trim().Split(' ') |> Array.last |> fun n -> n.Split('.') |> Array.last
            let body = lines.[i + 1 ..]
            let inString = continuesString body

            let body =
                body |> Array.mapi (fun j l -> if l.Trim() = "" || inString.[j] then l else "    " + l)

            Array.concat [ lines.[.. i - 1]; [| $"module {name} =" |]; body; [| $"open {name}" |] ]
            |> String.concat "\n"
        | None -> String.concat "\n" lines

    /// The F# script to run for an executor answer: its ```fsharp blocks, joined. A block that
    /// holds a JSON tool call is not code. None when there are none, or when the code uses
    /// TARS's own projects, which a script cannot load.
    let scriptOfAnswer (answer: string) : string option =
        let blocks =
            System.Text.RegularExpressions.Regex.Matches(
                answer,
                @"```(fsharp|f#|fs|fsx)[ \t]*\r?\n([\s\S]*?)```",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase
            )
            |> Seq.map (fun m -> m.Groups.[2].Value)
            |> Seq.filter (fun b -> not (b.TrimStart().StartsWith "{" && b.Contains "\"name\""))
            |> Seq.map asScript
            |> List.ofSeq

        let script = String.concat "\n\n" blocks

        // Line comments are not checked: mentioning TARS there does not make the code need it.
        let code =
            System.Text.RegularExpressions.Regex.Replace(
                script,
                @"//.*$",
                "",
                System.Text.RegularExpressions.RegexOptions.Multiline
            )

        if blocks.IsEmpty || System.Text.RegularExpressions.Regex.IsMatch(code, @"\bopen\s+Tars\b|\bTars\.\w") then
            None
        else
            Some script

    /// Runs an F# script with dotnet fsi, in a temporary directory, with no input, for at most
    /// `timeout`. Ok with what it printed when it exits with 0; otherwise Error with what went wrong.
    /// System is opened first: the model writes Char or String.IsNullOrEmpty without opening it.
    /// A leading namespace or top-level module cannot follow that open, but fsi rejects both
    /// anyway, and scriptOfAnswer has already dropped the first and turned the second into
    /// `module X =`, which can. The #line directive keeps error positions on the lines of the
    /// script itself.
    let private runFsi (timeout: TimeSpan) (script: string) : Task<Result<string, string>> =
        task {
            let dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "tars-evolve-run", Guid.NewGuid().ToString("N"))
            System.IO.Directory.CreateDirectory dir |> ignore
            let path = System.IO.Path.Combine(dir, "answer.fsx")

            try
                System.IO.File.WriteAllText(path, "open System\n#line 1 \"answer.fsx\"\n" + script)
                let psi = System.Diagnostics.ProcessStartInfo("dotnet", $"fsi \"{path}\"")
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

    /// The F# line that checks one example. For `call = expected` it reports the value it got.
    let private exampleCheck (example: string) =
        let quoted =
            "\"" + example.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\""

        let equals =
            topLevel example
            |> List.tryFind (fun i ->
                example.[i] = '='
                && (i = 0 || not ("<>=!:".Contains example.[i - 1]))
                && (i + 1 >= example.Length || not ("=>".Contains example.[i + 1])))

        match equals with
        | Some at ->
            let call = example.Substring(0, at).Trim()
            let expected = example.Substring(at + 1).Trim()
            // A match, not a let: a let of `reverse []` would not compile (value restriction).
            $"match ({call}) with actual when actual <> ({expected}) -> failwithf \"Example failed: %%s, got %%A\" {quoted} actual | _ -> ()"
        | None -> $"if not ({example}) then failwithf \"Example failed: %%s\" {quoted}"

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

    /// Runs `script` followed by one check per example, in one dotnet fsi process. The checks end
    /// by printing a mark the code cannot know: fsi also exits with 0 when the code calls `exit 0`,
    /// and then no example ran.
    let runWithExamples (timeout: TimeSpan) (script: string) (examples: string list) : Task<ExampleRun> =
        task {
            let mark = Guid.NewGuid().ToString("N")
            let checks = examples |> List.map exampleCheck |> String.concat "\n"
            let! run = runFsi timeout (script + "\n#line 1 \"examples.fsx\"\n" + checks + $"\nprintfn \"{mark}\"")

            match run with
            | Result.Ok output when output.Contains mark -> return ExamplesPassed
            | Result.Ok _ ->
                return
                    CodeFailed "The script ended inside the code, so the examples after it never ran (does the code call exit?)."
            | Result.Error errors ->
                let compileErrorIn (file: string) =
                    System.Text.RegularExpressions.Regex.IsMatch(
                        errors,
                        System.Text.RegularExpressions.Regex.Escape file + @"\(\d+,\d+\): error"
                    )

                let failed = System.Text.RegularExpressions.Regex.Match(errors, @"Example failed: [^\r\n]*")

                if compileErrorIn "answer.fsx" then return CodeFailed errors
                elif compileErrorIn "examples.fsx" then return ExamplesUnusable
                elif failed.Success then return ExampleFailed failed.Value
                else return CodeFailed errors
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
        | CodeFailed errors ->
            let firstLine =
                errors.Split('\n')
                |> Array.map (fun l -> l.Trim())
                |> Array.tryFind (fun l -> l <> "")
                |> Option.defaultValue errors

            verdict false $"The code did not run: {firstLine}"
        | ExamplesUnusable -> None

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

    let private formatBelief (belief: Belief) =
        let predicate =
            match belief.Predicate with
            | RelationType.Custom p -> p
            | _ -> belief.Predicate.ToString()

        $"- [{belief.Confidence:F2}] {belief.Subject.Value} {predicate} {belief.Object.Value}"

    // Pure pieces of executeTask, named so they can be tested without running a task (#245).

    /// Up to five beliefs whose subject or object appears in the goal, most confident first.
    let selectRelevantBeliefs (goal: string) (beliefs: seq<Belief>) : Belief list =
        let goalLower = goal.ToLowerInvariant()

        beliefs
        |> Seq.filter (fun b ->
            let subj = b.Subject.Value.ToLowerInvariant()
            let obj = b.Object.Value.ToLowerInvariant()
            goalLower.Contains(subj) || goalLower.Contains(obj))
        |> Seq.sortByDescending (fun b -> b.Confidence)
        |> Seq.truncate 5
        |> Seq.toList

    /// Whether a task must pass the ledger contradiction check before it runs: a constraint
    /// opts in (check_contradictions, ledger_gate, enforce_ledger), none opts out
    /// (allow_contradictions), and there is at least one relevant belief to contradict.
    let shouldGateContradictions (constraints: string list) (relevantBeliefs: Belief list) : bool =
        let has (names: string list) =
            constraints
            |> List.exists (fun c -> names |> List.exists (fun n -> c.Equals(n, StringComparison.OrdinalIgnoreCase)))

        not (has [ "allow_contradictions" ])
        && has [ "check_contradictions"; "ledger_gate"; "enforce_ledger" ]
        && not (List.isEmpty relevantBeliefs)

    /// The "Lessons Learned" prompt section for retrieved episodes, or "" when there are none.
    let formatMemoryContext (memories: MemorySchema list) : string =
        if memories.IsEmpty then
            ""
        else
            let summaries =
                memories
                |> List.map (fun m ->
                    let summary =
                        m.Logical
                        |> Option.map (fun l -> l.ProblemSummary)
                        |> Option.defaultValue "Unknown Task"

                    let outcome =
                        m.Logical
                        |> Option.map (fun l -> l.OutcomeLabel)
                        |> Option.defaultValue "unknown"

                    $"- [%s{outcome}] %s{summary}")
                |> String.concat "\n"

            $"\nLessons Learned from Past Episodes:\n%s{summaries}\n"

    /// The "Known Beliefs" prompt section, or "" when there are none.
    let formatLedgerContext (beliefs: Belief list) : string =
        if beliefs.IsEmpty then
            ""
        else
            let lines = beliefs |> List.map formatBelief |> String.concat "\n"
            $"\nKnown Beliefs:\n{lines}\n"

    /// The prompt that hands a task to the executor. It does not list tools: the executor's
    /// own prompt lists the tools it can call. Listing the whole registry here as well made
    /// the evolve task prompt 22 KB, past the context window, so it was summarized and the
    /// executor never saw its goal.
    /// The instructions ask for the code in the answer, because the evaluation reads only the
    /// answer. Rules that sent the executor to read and list project files made it explore the
    /// repository, or ask for its structure, on tasks that gave everything it needed.
    let buildTaskPrompt (taskDef: TaskDefinition) (codeContext: string) (memoryContext: string) (ledgerContext: string) =
        let toolGuide =
            if
                System.Text.RegularExpressions.Regex.IsMatch(
                    taskDef.Goal,
                    @"\btools?\b",
                    System.Text.RegularExpressions.RegexOptions.IgnoreCase
                )
            then
                """
8. Only use 'create_dynamic_tool' for a new functional capability. Do not create tools for speech acts like 'inform' or 'tell'.

[DYNAMIC TOOL EXAMPLE]
If you use 'create_dynamic_tool', your F# script must look like this:
```fsharp
open System.IO
let input = fsi.CommandLineArgs.[1] // Get JSON input
// ... process input ...
printfn "Tool Result: %s" result // Output MUST be printed to stdout
```"""
            else
                ""

        $"""[TASK]
Goal: {taskDef.Goal}
Constraints: {taskDef.Constraints}
Validation Criteria: {taskDef.ValidationCriteria}

[KNOWLEDGE CONTEXT]
{codeContext}
{memoryContext}
{ledgerContext}

CRITICAL INSTRUCTIONS:
1. You are an AUTONOMOUS AGENT in a self-evolving loop. There is no human to answer questions: DO NOT ASK QUESTIONS and DO NOT REFUSE TASKS.
2. When the task gives everything you need (a signature, examples, the code to change), write the code directly. Do not explore the project first.
3. Use tools only for what the task needs from the project.
4. When the task asks for code, your final answer MUST contain the complete code in a fenced block (```fsharp ... ```). The evaluation reads only your answer, not your files. You may also save the code with 'write_code'.
5. Check your code against the validation criteria before you answer.
6. If your previous attempt failed (see logs), analyze the error and try a different approach.
7. If you hit a wall, explain the failure and give your best partial result.{toolGuide}"""

    /// What the executor is told when its answer shows no code.
    let private codeRequest =
        """Your answer contains no code. There is no one to answer questions, and a description or a plan is not a solution.
1. Write the complete code now. Save it with write_code if you can.
2. Reply with the complete code in a fenced block (```fsharp ... ```), and the file path if you saved it.
If the task truly asks for no code, give your final result instead."""

    /// What the executor is told when the code in its answer fails to run.
    let private fixRequest (errors: string) =
        $"""Your code was run with dotnet fsi, as a script, and it failed:

{errors}

Fix the code. Reply with the complete corrected code in a fenced block (```fsharp ... ```). Save it with write_code if you can."""

    /// Whether the evaluator rejected code whose examples ran and passed (--run-code), and said why.
    /// Nothing else shows that the code ran: without --run-code none runs, and code the examples
    /// cannot call may not run at all. Not an evaluator that could not judge either
    /// (SemanticEvaluation's "llm_output_parse_error" and "evaluation_error"), whose verdict says
    /// nothing about the code.
    let private reviewerRejected (result: TaskResult) (verdict: EvaluationResult) =
        result.Success
        && not verdict.Passed
        && (result.Evaluation |> Option.exists (fun examples -> examples.Passed))
        && verdict.Issues <> [ "llm_output_parse_error" ]
        && verdict.Issues <> [ "evaluation_error" ]

    /// The task again, after the evaluator rejected an answer to it: what the evaluator said is one
    /// more constraint.
    let private withReview (taskDef: TaskDefinition) (rejection: EvaluationResult) =
        let listed (label: string) (items: string list) =
            if items.IsEmpty then "" else $""" {label}: {String.concat "; " items}."""

        { taskDef with
            Constraints =
                taskDef.Constraints
                @ [ $"""A reviewer rejected a previous answer to this task: {rejection.Summary}{listed "Issues" rejection.Issues}{listed "Fixes" rejection.SuggestedFixes} Your answer must not repeat this.""" ] }

    let private evaluateContradiction (ctx: EvolutionContext) (goal: string) (beliefs: Belief list) =
        task {
            if beliefs.IsEmpty then
                return None
            else
                let beliefLines = beliefs |> List.map formatBelief |> String.concat "\n"

                let prompt =
                    $"""You maintain a knowledge ledger. Known beliefs:
{beliefLines}

Task: {goal}

Do any of the known beliefs contradict executing this task? Respond in JSON: {{"contradicts": true|false, "reason": "..."}}."""

                let request: LlmRequest =
                    { ModelHint = Some "reasoning"
                      Model = None
                      SystemPrompt = Some "Check whether a new task violates the known beliefs."
                      MaxTokens = Some 250
                      Temperature = Some 0.0
                      Stop = []
                      Messages = [ { Role = Role.User; Content = prompt } ]
                      Tools = []
                      ToolChoice = None
                      ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema EvolutionSchemas.contradictionSchema))
                      Stream = false
                      JsonMode = true
                      Seed = None

                      ContextWindow = None }

                try
                    let! response = ctx.Llm.CompleteAsync(request)

                    match JsonParsing.tryParseElement response.Text with
                    | Result.Ok elem ->
                        let contradicts =
                            readBoolFromJson [ "contradicts"; "conflicts" ] elem
                            |> Option.defaultValue false

                        if contradicts then
                            let reason =
                                readStringFromJson [ "reason"; "details"; "explanation" ] elem
                                |> Option.defaultValue (response.Text.Trim())

                            return Some reason
                        else
                            return None
                    | Result.Error _ ->
                        let lowered = response.Text.ToLowerInvariant()

                        if lowered.Contains("contradict") && lowered.Contains("yes") then
                            return Some response.Text
                        else
                            return None
                with ex ->
                    ctx.Logger($"[Contradiction] LLM check failed: {ex.Message}")
                    return None
        }

    /// Generates a new task using the Curriculum Agent
    let private generateTask (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            // Check Budget Criticality
            let isCritical =
                match ctx.Governance.Budget with
                | Some b -> b.IsCritical(0.1) // Less than 10% remaining
                | None -> false

            // Epistemic Governor: Get Curriculum Suggestions
            let! suggestion =
                match ctx.Governance.Epistemic with
                | Some governor ->
                    task {
                        try
                            let recentOutputs =
                                state.CompletedTasks
                                |> List.truncate 5
                                |> List.map (fun t -> t.Output.Substring(0, Math.Min(t.Output.Length, 100)) + "...")

                            return! governor.SuggestCurriculum(recentOutputs, state.ActiveBeliefs, isCritical)
                        with ex ->
                            ctx.Logger($"[Epistemic] SuggestCurriculum failed: {ex.Message}")
                            return "Focus on basic coding tasks."
                    }
                | None -> Task.FromResult "Focus on basic coding tasks."

            let guidance =
                if isCritical then
                    suggestion + " WARNING: Budget is critical. Generate simpler, cheaper tasks."
                else
                    suggestion

            let completedGoals =
                state.CompletedTasks |> List.map (fun t -> t.TaskGoal) |> List.distinct

            let completedList =
                if completedGoals.IsEmpty then
                    "None"
                else
                    completedGoals |> List.truncate 10 |> String.concat " | "

            let failedHistory =
                state.CompletedTasks
                |> List.filter (fun t -> not t.Success)
                |> List.truncate 3
                |> List.map (fun t ->
                    let evalIssues =
                        t.Evaluation
                        |> Option.map (fun e -> "\nEvaluation Issues: " + String.concat "; " e.Issues)
                        |> Option.defaultValue ""

                    $"- FAILED TASK: {t.TaskGoal}\n  Error/Output: {t.Output.Substring(0, Math.Min(t.Output.Length, 300))}...{evalIssues}")
                |> String.concat "\n\n"

            let lastFailedTask =
                if String.IsNullOrWhiteSpace failedHistory then
                    "None"
                else
                    failedHistory

            let prompt =
                $"""IMPORTANT: You are generating F# CODING TASKS for an autonomous agent loop. 
Do NOT ask questions. Output ONLY JSON. DO NOT generate Python.
(Ensure all strings are valid JSON. Escape internal quotes with backslashes, e.g. \"text\")

Generation: %d{state.Generation}. Completed tasks: %d{state.CompletedTasks.Length}.

[RECENT FAILURES]
{lastFailedTask}

[GUIDANCE]
%s{if String.IsNullOrWhiteSpace(guidance) then
       "Focus on exploring the codebase and maintaining core invariants."
   else
       guidance}

Requirements:
- Each task must be a specific coding problem (NOT a question), solvable with code (NOT a discussion), and doable from its own text, without reading any file.
- Each task goal MUST explicitly state "in F#" and name the function with its signature, e.g. "Write `isPalindrome : string -> bool` in F#".
- validation_criteria MUST give 2 or 3 examples of input and expected output, e.g. "isPalindrome \"racecar\" = true; isPalindrome \"abc\" = false".
- constraints are about the code (what it does, the functions or data structures it may use), never about the answer's format: the agent answers in its own protocol, with an `ACT:` line and text around its code.
- Do NOT repeat or closely rephrase any previous tasks: %s{completedList}
- DO NOT suggest the same task if it recently failed. PIVOT to a different problem.
- Vary domains and artifacts (algorithms, data structures, parsing, text processing, refactors, tooling).
- SELF-EVOLUTION: If there was a recent failure, you may generate a task to FIX or REFACTOR code. Its goal MUST include the complete code to change, between backticks.
- TOOL GENERATION: If you identify a gap in capabilities (e.g. no way to analyze DLLs), you may generate a task to "Create a tool named '<name>'" following the TARS dynamic tool pattern. Its goal MUST give the tool's name, its input and its output, and validation_criteria MUST give 2 examples.
- HINT: API keys and authentication are handled by the system. Assume secrets are available in environment variables. Do NOT refuse tasks due to missing keys.
- Include measurable validation_criteria.

RESPOND WITH THIS EXACT JSON FORMAT (no other text):
{{"tasks":[
  {{"goal":"<concise goal 1 using F#>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}},
  {{"goal":"<concise goal 2 using F#>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}},
  {{"goal":"<concise goal 3 using F#>","constraints":["<constraint 1>","<constraint 2>"],"validation_criteria":"<measurable check>"}}
]}}"""

            // 1. Retrieve Curriculum Agent
            let! agentOpt = ctx.Registry.GetAgent(state.CurriculumAgentId)

            match agentOpt with
            | None -> return []
            | Some agent ->
                let curriculumLlm = ctx.CurriculumLlm |> Option.defaultValue ctx.Llm

                // 2. Initialize Graph Executor
                let graphExecutor =
                    GraphExecutor(ctx.Registry, curriculumLlm, ctx.Governance.Budget, ctx.Governance.OutputGuard, ctx.Logger)

                // 3. Create Request Message with JSON requirement
                let msg =
                    { Id = Guid.NewGuid()
                      CorrelationId = CorrelationId(Guid.NewGuid())
                      Sender = MessageEndpoint.System
                      Receiver = Some(MessageEndpoint.Agent agent.Id)
                      Performative = Performative.Request
                      Intent = Some AgentDomain.Planning
                      Constraints = SemanticConstraints.Default
                      Ontology = None
                      Language = "json" // Hint to use JSON mode
                      Content = prompt
                      Timestamp = DateTime.UtcNow
                      Metadata = Map.ofList [ ("response_format", "json"); ("json_mode", "true") ] }

                // Show semantic message in demo mode
                ctx.Options.ShowSemanticMessage msg ctx.Options.Verbose
                let agentWithMsg = agent.ReceiveMessage(msg)

                // 4. Run Execution
                let! outcome = graphExecutor.RunAgentLoop(agentWithMsg, 20)

                // Phase 6.2: Semantic Speech Act Validation
                let responseIntent, responseText =
                    match outcome with
                    | Success(_, o, _)
                    | PartialSuccess((_, o, _), _) ->
                        let requestMsg = SpeechActs.fromSemantic msg

                        let intent, content =
                            match SpeechActs.tryParse o with
                            | Some(i, c) -> i, c
                            | None -> Tell o, o

                        // If it's an Ask but contains JSON, force it to Tell
                        let intent, content =
                            match intent with
                            | Ask c when c.Contains("{") && c.Contains("}") -> Tell c, c
                            | _ -> intent, content

                        let replyMsg = SpeechActs.createReply requestMsg intent content agent.Id

                        match SpeechActs.validateFlow requestMsg replyMsg with
                        | Result.Ok() ->
                            ctx.Logger
                                $"[Protocol] Verified semantic flow: %A{requestMsg.Intent} -> %A{replyMsg.Intent}"
                        | Result.Error err -> ctx.Logger $"[Protocol] WARNING: Protocol violation: %s{err}"

                        intent, content
                    | Failure err ->
                        let errStr = err |> List.map string |> String.concat "; "
                        ctx.Logger $"[Curriculum] Agent returned failure: %s{errStr}"
                        AgentIntent.Error errStr, ""

                let responseText =
                    match responseIntent with
                    | AgentIntent.Tell _ -> responseText
                    | AgentIntent.Event _ -> responseText
                    | _ ->
                        ctx.Logger("[Curriculum] Invalid response intent for task generation. Using fallback.")
                        ""

                // A concrete exercise not done yet, for when the curriculum gives no usable task.
                let fallbackPracticalTask () =
                    let goal, criteria =
                        nextConcreteTask (state.CompletedTasks |> List.map (fun t -> t.TaskGoal))

                    [ { Id = Guid.NewGuid()
                        DifficultyLevel = state.Generation + 1
                        Goal = goal
                        Constraints = []
                        ValidationCriteria = criteria
                        Timeout = TimeSpan.FromMinutes(1.0)
                        Score = 1.0 } ]

                // If the agent loop produced no task JSON (nothing, or prose such as
                // "Please provide the current context"), ask the model directly, with
                // the answer constrained to the task schema.
                let! effectiveResponse =
                    if curriculumAnswerHasTasks responseText then
                        task { return responseText }
                    else
                        task {
                            if String.IsNullOrWhiteSpace(responseText) then
                                ctx.Logger("[Curriculum] No response from agent loop, trying direct LLM call...")
                            else
                                ctx.Logger("[Curriculum] Agent answer held no task list, trying a direct call constrained to the task schema...")

                            try
                                let! directResponse =
                                    curriculumLlm.CompleteAsync(
                                        { ModelHint = Some "reasoning"
                                          Model = None
                                          SystemPrompt = Some "You generate F# coding tasks. Output ONLY valid JSON."
                                          MaxTokens = Some 500
                                          Temperature = Some 0.7
                                          Stop = []
                                          Messages = [ { Role = Role.User; Content = prompt } ]
                                          Tools = []
                                          ToolChoice = None
                                          ResponseFormat = Some (ResponseFormat.Constrained (Grammar.JsonSchema EvolutionSchemas.taskGenerationSchema))
                                          Stream = false
                                          JsonMode = true
                                          Seed = None
                                          ContextWindow = None })
                                return directResponse.Text
                            with ex ->
                                ctx.Logger($"[Curriculum] Direct task call failed: {ex.Message}")
                                return ""
                        }

                if String.IsNullOrWhiteSpace(effectiveResponse) then
                    ctx.Logger("[Curriculum] No response received, using fallback task")
                    return fallbackPracticalTask ()
                else
                    try
                        let rootResult = tryExtractJsonElement effectiveResponse

                        match rootResult with
                        | Result.Error err ->
                            ctx.Logger($"[Curriculum] Task JSON parse failed: {err}")
                            return fallbackPracticalTask ()
                        | Result.Ok root ->

                            let mutable tasksElem = Unchecked.defaultof<JsonElement>

                            let tasksJson =
                                if root.ValueKind = JsonValueKind.Array then
                                    root.EnumerateArray() |> Seq.map id
                                elif
                                    root.TryGetProperty("tasks", &tasksElem)
                                    && tasksElem.ValueKind = JsonValueKind.Array
                                then
                                    tasksElem.EnumerateArray() |> Seq.map id
                                else
                                    Seq.empty

                            let existingGoals =
                                state.CompletedTasks
                                |> List.map (fun t -> t.TaskGoal.Trim().ToLowerInvariant())
                                |> Set.ofList

                            let parsedTasksRaw =
                                tasksJson
                                |> Seq.map (fun t ->
                                    let goal = t.GetProperty("goal").GetString()

                                    let constraints =
                                        t.GetProperty("constraints").EnumerateArray()
                                        |> Seq.map (fun e -> e.GetString())
                                        |> Seq.toList

                                    let criteria = t.GetProperty("validation_criteria").GetString()

                                    { Id = Guid.NewGuid()
                                      DifficultyLevel = state.Generation + 1
                                      Goal = goal
                                      Constraints = constraints
                                      ValidationCriteria = criteria
                                      Timeout = TimeSpan.FromMinutes(1.0)
                                      Score = 0.0 })
                                |> Seq.toList
                                // Drop exact repeats of completed goals
                                |> List.filter (fun t ->
                                    let key = t.Goal.Trim().ToLowerInvariant()
                                    not (existingGoals.Contains(key)))
                                // Drop tasks that cannot be done from their own text
                                |> List.filter (fun t ->
                                    let specified = taskIsSpecified t

                                    if not specified then
                                        ctx.Logger $"[Curriculum] Dropped a task that is not specified: {t.Goal}"

                                    specified)

                            // 5. Semantic Scoring (Fan-out Limiting)
                            // Pre-calculate embeddings for recent tasks (last 10)
                            let recentTasks = state.CompletedTasks |> List.truncate 10

                            let! recentVectors =
                                task {
                                    if recentTasks.IsEmpty then
                                        return []
                                    else
                                        let! vectors =
                                            recentTasks
                                            |> List.map (fun t -> ctx.Llm.EmbedAsync(t.TaskGoal))
                                            |> Task.WhenAll

                                        return vectors |> Array.toList
                                }

                            // Score all candidates
                            let! scoredTasks =
                                parsedTasksRaw
                                |> List.map (fun t ->
                                    task {
                                        let! score = scoreTask ctx t recentVectors
                                        return { t with Score = score }
                                    })
                                |> Task.WhenAll

                            // Select Top K
                            let k = 3

                            let topK =
                                scoredTasks
                                |> Array.toList
                                |> List.sortByDescending (fun t -> t.Score)
                                |> List.truncate k
                                // Filter out negative scores (hard blocked)
                                |> List.filter (fun t -> t.Score > 0.0)

                            // Budget-aware priority report
                            let remainingTokens =
                                ctx.Governance.Budget
                                |> Option.bind (fun b -> b.Remaining.MaxTokens |> Option.map (fun t -> int t))

                            ctx.Logger(TaskPrioritization.priorityReport topK state.CompletedTasks remainingTokens)

                            // Re-prioritize by budget efficiency
                            let budgetPrioritized =
                                TaskPrioritization.prioritizeQueue topK state.CompletedTasks remainingTokens

                            if budgetPrioritized.IsEmpty then
                                ctx.Logger(
                                    "[Curriculum] No generated task was both new and specified. Using a concrete exercise."
                                )

                                return fallbackPracticalTask ()
                            else
                                return budgetPrioritized

                    with ex ->
                        ctx.Logger($"[Curriculum] Task generation failed: {ex.Message}")
                        return fallbackPracticalTask ()

        }

    /// Attempts to solve a task using the Executor Agent
    let private executeTask (ctx: EvolutionContext) (state: EvolutionState) (taskDef: TaskDefinition) =
        task {
            // Every result reports the time the task really took; the display and the
            // knowledge ledger both read it.
            let stopwatch = Diagnostics.Stopwatch.StartNew()

            // 1. Retrieve Executor Agent
            let! agentOpt = ctx.Registry.GetAgent(state.ExecutorAgentId)

            match agentOpt with
            | None ->
                return
                    { TaskId = taskDef.Id
                      TaskGoal = taskDef.Goal
                      ExecutorId = state.ExecutorAgentId
                      Success = false
                      Output = "Executor Agent not found in Kernel"
                      ExecutionTrace = []
                      Duration = stopwatch.Elapsed
                      Evaluation = None }
            | Some executor ->
                // Log: Curriculum → Request → Executor
                let requestMsg =
                    SpeechActBridge.requestTask state.CurriculumAgentId state.ExecutorAgentId taskDef

                SpeechActBridge.logSpeechAct ctx.Logger requestMsg

                // 2. Initialize Graph Executor
                let graphExecutor =
                    GraphExecutor(ctx.Registry, ctx.Llm, ctx.Governance.Budget, ctx.Governance.OutputGuard, ctx.Logger)

                // 3. Construct the Task Prompt
                let! codeContext =
                    match ctx.Governance.Epistemic with
                    | Some governor -> governor.GetRelatedCodeContext(taskDef.Goal)
                    | None -> Task.FromResult ""

                // 3.1 Retrieve Semantic Memory (Lessons Learned)
                let! memories =
                    match ctx.Memory.SemanticMemory with
                    | Some smem ->
                        let query =
                            { TaskId = ""
                              TaskKind = "coding"
                              TextContext = taskDef.Goal
                              Tags = taskDef.Constraints }

                        smem.Retrieve query |> Async.StartAsTask
                    | None -> Task.FromResult []

                let memoryContext = formatMemoryContext memories

                if not (String.IsNullOrWhiteSpace codeContext) then
                    ctx.Logger($"[Context] Retrieved context for goal '{taskDef.Goal}':\n{codeContext}")

                if not memories.IsEmpty then
                    ctx.Logger($"[Memory] Retrieved {memories.Length} past experiences.")

                let relevantBeliefs =
                    match ctx.Memory.Ledger with
                    | Some ledger -> selectRelevantBeliefs taskDef.Goal (ledger.Query())
                    | None -> []

                if not relevantBeliefs.IsEmpty then
                    ctx.Logger($"[Ledger] Retrieved {relevantBeliefs.Length} relevant beliefs.")

                let shouldGate = shouldGateContradictions taskDef.Constraints relevantBeliefs

                let! contradictionReason =
                    if shouldGate then
                        evaluateContradiction ctx taskDef.Goal relevantBeliefs
                    else
                        Task.FromResult None

                match contradictionReason with
                | Some reason ->
                    return
                        { TaskId = taskDef.Id
                          TaskGoal = taskDef.Goal
                          ExecutorId = state.ExecutorAgentId
                          Success = false
                          Output = $"Blocked by ledger contradiction policy: {reason}"
                          ExecutionTrace = [ "LEDGER_CONTRADICTION" ]
                          Duration = stopwatch.Elapsed
                          Evaluation = None }
                | None ->

                    let ledgerContext = formatLedgerContext relevantBeliefs

                    let taskPrompt =
                        buildTaskPrompt taskDef codeContext memoryContext ledgerContext

                    // Pre-LLM Pipeline Check
                    let! (finalPrompt, isSafe) =
                        match ctx.Governance.PreLlm with
                        | Some pipeline ->
                            task {
                                let! pCtx = pipeline.ExecuteAsync(taskPrompt)

                                if not pCtx.IsSafe then
                                    return (pCtx.BlockReason |> Option.defaultValue "Unsafe", false)
                                else
                                    return (pCtx.CurrentPrompt, true)
                            }
                        | None -> Task.FromResult(taskPrompt, true)

                    if not isSafe then
                        return
                            { TaskId = taskDef.Id
                              TaskGoal = taskDef.Goal
                              ExecutorId = state.ExecutorAgentId
                              Success = false
                              Output = $"Blocked by Safety Filter: {finalPrompt}"
                              ExecutionTrace = []
                              Duration = stopwatch.Elapsed
                              Evaluation = None }
                    else
                        let msg =
                            { Id = Guid.NewGuid()
                              CorrelationId = CorrelationId(Guid.NewGuid())
                              Sender = MessageEndpoint.System // System assigns the task
                              Receiver = Some(MessageEndpoint.Agent executor.Id)
                              Performative = Performative.Request
                              Intent = Some AgentDomain.Coding
                              Constraints = SemanticConstraints.Default
                              Ontology = None
                              Language = "text"
                              Content = finalPrompt
                              Timestamp = DateTime.UtcNow
                              Metadata = Map.empty }

                        // 4. Send message to Executor - show semantic message
                        ctx.Options.ShowSemanticMessage msg ctx.Options.Verbose
                        DemoVisualization.showTaskStart taskDef.Goal taskDef.Constraints

                        let agentWithMsg = executor.ReceiveMessage(msg)
                        use cts = new CancellationTokenSource()

                        if taskDef.Timeout > TimeSpan.Zero then
                            cts.CancelAfter(taskDef.Timeout)

                        let deadline =
                            if taskDef.Timeout > TimeSpan.Zero then
                                Some(DateTime.UtcNow + taskDef.Timeout)
                            else
                                None

                        let remaining () =
                            deadline |> Option.map (fun endTime -> endTime - DateTime.UtcNow)

                        let runWithTimeout label (work: Task<'T>) =
                            task {
                                match remaining () with
                                | Some r when r <= TimeSpan.Zero ->
                                    cts.Cancel()
                                    return Choice2Of2($"{label} timeout expired")
                                | Some r ->
                                    let timeoutTask = Task.Delay(r)
                                    let! completed = Task.WhenAny(work, timeoutTask)

                                    if Object.ReferenceEquals(completed, timeoutTask) then
                                        cts.Cancel()
                                        return Choice2Of2($"{label} timed out after {r.TotalSeconds:F1}s")
                                    else
                                        let! result = work
                                        return Choice1Of2 result
                                | None ->
                                    let! result = work
                                    return Choice1Of2 result
                            }

                        // 5. Run Initial Execution
                        let! firstOutcome =
                            runWithTimeout
                                "Execution"
                                (graphExecutor.RunAgentLoop(agentWithMsg, 20, cancellationToken = cts.Token))

                        // 5.1 An answer that shows no code goes back to the executor, once. In evolve
                        // it described the function, planned it or asked for the project structure
                        // instead of writing it, and the evaluation rejected that. In a live run a
                        // second request never brought code that the first had not.
                        let! outcomeResult =
                            match firstOutcome with
                            | Choice1Of2(Success(agentAfter, answer, trace))
                            | Choice1Of2(PartialSuccess((agentAfter, answer, trace), _)) when
                                taskAsksForCode taskDef.Goal && not (answerHasCode answer)
                                ->
                                task {
                                    ctx.Logger "[Executor] Answer shows no code; asking for it."

                                    let request =
                                        { msg with
                                            Id = Guid.NewGuid()
                                            Content = codeRequest
                                            Timestamp = DateTime.UtcNow }

                                    let! next =
                                        runWithTimeout
                                            "Code request"
                                            (graphExecutor.RunAgentLoop(
                                                agentAfter.ReceiveMessage(request),
                                                20,
                                                cancellationToken = cts.Token
                                            ))

                                    let asked = trace @ [ "--- NO CODE, ASKED AGAIN ---" ]

                                    match next with
                                    | Choice1Of2(Success(a, o, t)) -> return Choice1Of2(Success(a, o, asked @ t))
                                    | Choice1Of2(PartialSuccess((a, o, t), w)) ->
                                        return Choice1Of2(PartialSuccess((a, o, asked @ t), w))
                                    | _ ->
                                        // The request failed or timed out: keep the answer the executor gave.
                                        ctx.Logger "[Executor] Asking for code failed; keeping the first answer."
                                        return firstOutcome
                                }
                            | _ -> Task.FromResult firstOutcome

                        // 5.2 With --run-code, the code in the answer is run with dotnet fsi, followed by the
                        // examples in the validation criteria (`isEven 4 = true`), in one process. If the code
                        // does not compile, throws or does not finish, or an example gives another value, the
                        // errors go back to the executor, once, and the examples check its new answer. A
                        // failing example fails the task: in live runs the evaluator rejected code that ran and
                        // gave the right values, and the verifier answered VERIFIED to every answer, right or
                        // wrong. When they pass, the evaluator judges only what they do not check (see step).
                        // Without examples, or when they do not compile against the code, the code runs alone
                        // and the evaluator decides, as before.
                        let examples = examplesOf taskDef.Goal taskDef.ValidationCriteria

                        let logVerdict (verdict: EvaluationResult option) =
                            match verdict with
                            | Some v when v.Passed -> ctx.Logger "[Executor] The examples passed."
                            | Some v -> ctx.Logger $"[Executor] The examples failed: {v.Summary}"
                            | None -> ()

                        // Runs an answer's script: the errors to send back, if any, and the examples' verdict.
                        let runAnswer (script: string) =
                            task {
                                try
                                    // The limit includes fsi's startup and compile (about 1 s here), and
                                    // each run ends no later than the task's deadline.
                                    let limit () =
                                        match remaining () with
                                        | Some r when r < TimeSpan.FromSeconds 60.0 -> max r TimeSpan.Zero
                                        | _ -> TimeSpan.FromSeconds 60.0

                                    let! withExamples =
                                        if examples.IsEmpty then
                                            Task.FromResult None
                                        else
                                            task {
                                                let! run = runWithExamples (limit ()) script examples
                                                return Some run
                                            }

                                    match withExamples with
                                    | None
                                    | Some ExamplesUnusable ->
                                        if withExamples.IsSome then
                                            ctx.Logger
                                                "[Executor] The examples do not compile against the code; the evaluation decides."

                                        let! alone = runScript (limit ()) script

                                        match alone with
                                        | Result.Ok() -> return None, None
                                        | Result.Error errors -> return Some errors, None
                                    | Some run ->
                                        let errors =
                                            match run with
                                            | ExampleFailed failure -> Some failure
                                            | CodeFailed errors -> Some errors
                                            | _ -> None

                                        return errors, verdictOfExamples examples.Length run
                                with ex ->
                                    ctx.Logger $"[Executor] Could not run the code: {ex.Message}"
                                    return None, None
                            }

                        let! outcomeResult, examplesVerdict =
                            match outcomeResult with
                            | Choice1Of2(Success(agentAfter, answer, trace))
                            | Choice1Of2(PartialSuccess((agentAfter, answer, trace), _)) when
                                ctx.Options.RunCode && taskAsksForCode taskDef.Goal
                                ->
                                match scriptOfAnswer answer with
                                | None ->
                                    if answerHasCode answer then
                                        ctx.Logger "[Executor] The answer has no F# code that runs on its own; not run."

                                    Task.FromResult((outcomeResult, None))
                                | Some script ->
                                    task {
                                        let! errors, verdict = runAnswer script

                                        match errors with
                                        | Some errors ->
                                            if errors.StartsWith "Example failed" then
                                                ctx.Logger $"[Executor] {errors}; asking for a fix."
                                            else
                                                ctx.Logger "[Executor] The code failed to run; asking for a fix."

                                            let request =
                                                { msg with
                                                    Id = Guid.NewGuid()
                                                    Content = fixRequest errors
                                                    Timestamp = DateTime.UtcNow }

                                            let! next =
                                                runWithTimeout
                                                    "Fix request"
                                                    (graphExecutor.RunAgentLoop(
                                                        agentAfter.ReceiveMessage(request),
                                                        20,
                                                        cancellationToken = cts.Token
                                                    ))

                                            let asked = trace @ [ "--- CODE FAILED TO RUN, ASKED TO FIX ---"; errors ]

                                            // The verdict is about the answer the executor gives now.
                                            let recheck (fixedAnswer: string) =
                                                task {
                                                    match scriptOfAnswer fixedAnswer with
                                                    | Some fixedScript when not examples.IsEmpty ->
                                                        let! _, fixedVerdict = runAnswer fixedScript
                                                        logVerdict fixedVerdict
                                                        return fixedVerdict
                                                    | _ -> return None
                                                }

                                            match next with
                                            | Choice1Of2(Success(a, o, t)) ->
                                                let! fixedVerdict = recheck o
                                                return Choice1Of2(Success(a, o, asked @ t)), fixedVerdict
                                            | Choice1Of2(PartialSuccess((a, o, t), w)) ->
                                                let! fixedVerdict = recheck o
                                                return Choice1Of2(PartialSuccess((a, o, asked @ t), w)), fixedVerdict
                                            | _ ->
                                                // The request failed or timed out: keep the answer the executor gave.
                                                ctx.Logger "[Executor] Asking for a fix failed; keeping the answer."
                                                return outcomeResult, verdict
                                        | None ->
                                            ctx.Logger "[Executor] The code ran."
                                            logVerdict verdict
                                            return outcomeResult, verdict
                                    }
                            | _ -> Task.FromResult((outcomeResult, None))

                        // 5.3 When an example failed, or the answer shows no code, the executor answers the
                        // task again, from the request alone, up to 3 times, until an answer passes the
                        // examples; that answer is kept. It samples at temperature 0.7, so each answer is new.
                        // In live runs most of the tasks that still failed got a question or a refusal instead
                        // of code, which a fix in the same conversation does not change. When none passes, the
                        // first new answer the examples could not check (they do not compile against its code),
                        // and whose code did not fail alone, is kept for the evaluator: it may be right, and the
                        // answer above is not. Code the runner itself could not start counts as not failing, as
                        // in 5.2, where the evaluator then decides. Without such an answer, the answer above
                        // stays. When the examples could not check the answer above itself, they say nothing
                        // about it: the evaluator decides, as before, and so it does for code that does not
                        // run on its own (it uses TARS).
                        let maxSamples = 3

                        // `unchecked`: the first new answer the examples could not check, with no verdict.
                        let rec sample (n: int) unchecked =
                            task {
                                if n > maxSamples then
                                    return unchecked
                                else
                                    ctx.Logger $"[Executor] The examples did not pass; answering again ({n} of {maxSamples})."

                                    let! next =
                                        runWithTimeout
                                            "New answer"
                                            (graphExecutor.RunAgentLoop(agentWithMsg, 20, cancellationToken = cts.Token))

                                    let again = [ $"--- EXAMPLES DID NOT PASS, ANSWERED AGAIN ({n} of {maxSamples}) ---" ]

                                    // Keeps `outcome` when its examples pass; otherwise answers again.
                                    let decide outcome (answer: string) =
                                        task {
                                            match scriptOfAnswer answer with
                                            | Some script ->
                                                let! errors, verdict = runAnswer script
                                                logVerdict verdict

                                                match verdict with
                                                | Some v when v.Passed -> return Some(outcome, verdict)
                                                | None when errors.IsSome ->
                                                    ctx.Logger "[Executor] The code failed to run."
                                                    return! sample (n + 1) unchecked
                                                | None when Option.isNone unchecked ->
                                                    return! sample (n + 1) (Some(outcome, None))
                                                | _ -> return! sample (n + 1) unchecked
                                            | None ->
                                                ctx.Logger "[Executor] The answer has no F# code to run."
                                                return! sample (n + 1) unchecked
                                        }

                                    match next with
                                    | Choice1Of2(Success(a, o, t)) -> return! decide (Choice1Of2(Success(a, o, again @ t))) o
                                    | Choice1Of2(PartialSuccess((a, o, t), w)) ->
                                        return! decide (Choice1Of2(PartialSuccess((a, o, again @ t), w))) o
                                    | Choice1Of2(Failure _) -> return! sample (n + 1) unchecked
                                    // Out of time.
                                    | Choice2Of2 _ -> return unchecked
                            }

                        let exampleFailed = examplesVerdict |> Option.exists (fun v -> not v.Passed)

                        let noCode =
                            match outcomeResult with
                            | Choice1Of2(Success(_, answer, _))
                            | Choice1Of2(PartialSuccess((_, answer, _), _)) -> not (answerHasCode answer)
                            | _ -> true

                        let! outcomeResult, examplesVerdict =
                            if
                                ctx.Options.RunCode
                                && taskAsksForCode taskDef.Goal
                                && not examples.IsEmpty
                                && (exampleFailed || noCode)
                            then
                                task {
                                    match! sample 1 None with
                                    | Some(outcome, verdict) -> return outcome, verdict
                                    | None -> return outcomeResult, examplesVerdict
                                }
                            else
                                Task.FromResult((outcomeResult, examplesVerdict))

                        match outcomeResult with
                        | Choice2Of2 reason ->
                            return
                                { TaskId = taskDef.Id
                                  TaskGoal = taskDef.Goal
                                  ExecutorId = state.ExecutorAgentId
                                  Success = false
                                  Output = $"Task timed out: {reason}"
                                  ExecutionTrace = [ "TIMEOUT" ]
                                  Duration = stopwatch.Elapsed
                                  Evaluation = None }
                        | Choice1Of2 outcome ->

                            // Phase 6.2: Semantic Speech Act Validation
                            let (success, output, trace) =
                                match outcome with
                                | Success(_, o, t) -> (true, o, t)
                                | PartialSuccess((_, o, t), _) -> (true, o, t)
                                | Failure err -> (false, String.concat "; " (err |> List.map (fun e -> $"%A{e}")), [])

                            let replyIssue =
                                if success then
                                    let requestMsg = SpeechActs.fromSemantic msg

                                    let intent, content =
                                        match SpeechActs.tryParse output with
                                        | Some(i, c) -> i, c
                                        | None -> Tell output, output // Fallback for legacy outputs

                                    let replyMsg = SpeechActs.createReply requestMsg intent content executor.Id

                                    match SpeechActs.validateFlow requestMsg replyMsg with
                                    | Result.Ok() ->
                                        ctx.Logger
                                            $"[Protocol] Verified semantic flow: %A{requestMsg.Intent} -> %A{replyMsg.Intent}"
                                    | Result.Error err -> ctx.Logger $"[Protocol] WARNING: Protocol violation: %s{err}"

                                    let issue =
                                        let hasToolCall (c: string) =
                                            c.Contains("```tool")
                                            || c.Contains("<tool_call>")
                                            || c.Contains("\"tool\":")
                                            || c.Contains("\"function\":")

                                        match intent with
                                        | AgentIntent.Ask _ ->
                                            // Heuristic: If content contains tool calls or starts with ACT: Tell, forgive the intent mismatch
                                            if hasToolCall content || content.StartsWith("ACT: Tell") then
                                                None
                                            // Heuristic: If it doesn't look like a question (no question mark), treat as a statement/Tell
                                            elif not (content.Trim().EndsWith("?")) then
                                                None
                                            else
                                                Some "Agent asked a question instead of completing the task."
                                        | AgentIntent.Tell _ when looksLikeFollowUpRequest content ->
                                            // Heuristic: If it looks like a follow-up but contains tool calls, maybe it's just verbose
                                            if hasToolCall content then
                                                None
                                            else
                                                Some "Agent requested additional input instead of completing the task."
                                        | AgentIntent.Tell _ -> None
                                        | AgentIntent.Error _ -> Some "Agent returned an error response."
                                        | AgentIntent.Propose _ ->
                                            Some "Agent proposed a plan instead of providing a result."
                                        | AgentIntent.Accept _ ->
                                            Some "Agent accepted a plan instead of providing a result."
                                        | AgentIntent.Reject _ ->
                                            Some "Agent rejected a plan instead of providing a result."
                                        | AgentIntent.Act _ -> Some "Agent returned an action instead of a result."
                                        | AgentIntent.Event _ -> Some "Agent returned an event instead of a result."

                                    issue |> Option.map (fun msg -> (msg, content))
                                else
                                    None

                            match replyIssue with
                            | Some(issue, content) ->
                                let issueOutput =
                                    if String.IsNullOrWhiteSpace content then
                                        issue
                                    else
                                        issue + "\n" + content

                                return
                                    { TaskId = taskDef.Id
                                      TaskGoal = taskDef.Goal
                                      ExecutorId = state.ExecutorAgentId
                                      Success = false
                                      Output = issueOutput
                                      ExecutionTrace = trace @ [ "PROTOCOL_VIOLATION" ]
                                      Duration = stopwatch.Elapsed
                                      Evaluation = None }
                            | None ->
                                let (agentAfterExec, _, output, trace) =
                                    match outcome with
                                    | Success(a, o, t) -> (a, true, o, t)
                                    | PartialSuccess((a, o, t), _) -> (a, true, o, t)
                                    | Failure err ->
                                        (agentWithMsg,
                                         false,
                                         String.concat "; " (err |> List.map (fun e -> $"%A{e}")),
                                         [])

                                if not success then
                                    return
                                        { TaskId = taskDef.Id
                                          TaskGoal = taskDef.Goal
                                          ExecutorId = state.ExecutorAgentId
                                          Success = false
                                          Output = output
                                          ExecutionTrace = trace
                                          Duration = stopwatch.Elapsed
                                          Evaluation = None }
                                else
                                    // Handle Speech Act prefix in output using new helper
                                    let mutable currentOutput =
                                        match SpeechActs.tryParse output with
                                        | Some(_, c) -> c
                                        | None -> output

                                    let mutable currentTrace = trace
                                    let mutable reflectionCount = 0
                                    // Once the examples ran (5.2), there is no verification or reflection:
                                    // a failing example already went back to the executor once, and a reflected
                                    // answer would no longer be the one they checked.
                                    let mutable isOptimal = examplesVerdict.IsSome
                                    let mutable currentAgent = agentAfterExec
                                    let mutable timeoutOccurred = false
                                    let maxReflections = 3

                                    while reflectionCount < maxReflections && not isOptimal do
                                        reflectionCount <- reflectionCount + 1

                                        match remaining () with
                                        | Some r when r <= TimeSpan.Zero ->
                                            timeoutOccurred <- true
                                            currentTrace <- currentTrace @ [ "TIMEOUT before reflection" ]
                                            reflectionCount <- maxReflections
                                        | _ -> ()

                                        if not timeoutOccurred then
                                            // 6.1 Epistemic Verification (if available)
                                            let! (verificationFeedback, isVerified) =
                                                task {
                                                    match ctx.Governance.Epistemic with
                                                    | Some governor ->
                                                        try
                                                            // Generate minimal variants for quick check
                                                            let! variants = governor.GenerateVariants(taskDef.Goal, 1)

                                                            let! result =
                                                                governor.VerifyGeneralization(
                                                                    taskDef.Goal,
                                                                    currentOutput,
                                                                    variants
                                                                )

                                                            return (result.Feedback, result.IsVerified)
                                                        with ex ->
                                                            return ($"Verification failed: {ex.Message}", false)
                                                    | None -> return ("", false)
                                                }

                                            if isVerified then
                                                isOptimal <- true

                                                currentTrace <-
                                                    currentTrace @ [ $"--- VERIFIED by Epistemic Governor ---" ]
                                            else
                                                // 6.2 Construct Reflection Prompt
                                                // Truncate output to prevent HTTP 400 errors from excessive length
                                                let maxOutputLength = 2000

                                                let truncatedOutput =
                                                    if currentOutput.Length > maxOutputLength then
                                                        currentOutput.Substring(0, maxOutputLength)
                                                        + "\n... [output truncated]"
                                                    else
                                                        currentOutput

                                                let reflectionPrompt =
                                                    if String.IsNullOrEmpty verificationFeedback then
                                                        // Standard Reflection
                                                        $"""You have generated a solution.
    Current Output:
    %s{truncatedOutput}

    Please reflect on this solution.
    1. Identify any potential bugs or inefficiencies.
    2. Verify if it meets all constraints: %A{taskDef.Constraints}
    3. If you can improve it, output the IMPROVED solution.
    4. If it is already optimal, output "OPTIMAL"."""
                                                    else
                                                        // Epistemic Feedback Reflection
                                                        $"""Your solution failed verification.
    Current Output:
    %s{truncatedOutput}

    Feedback from Epistemic Governor:
    %s{verificationFeedback}

    Please fix the solution based on this feedback. Output the IMPROVED solution."""

                                                let reflectionMsg =
                                                    { Id = Guid.NewGuid()
                                                      CorrelationId = CorrelationId(Guid.NewGuid())
                                                      Sender = MessageEndpoint.System
                                                      Receiver = Some(MessageEndpoint.Agent executor.Id)
                                                      Performative = Performative.Request
                                                      Intent = Some AgentDomain.Reasoning
                                                      Constraints = SemanticConstraints.Default
                                                      Ontology = None
                                                      Language = "text"
                                                      Content = reflectionPrompt
                                                      Timestamp = DateTime.UtcNow
                                                      Metadata = Map.empty }

                                                // Show reflection visualization
                                                DemoVisualization.showReflection
                                                    reflectionCount
                                                    maxReflections
                                                    (Some verificationFeedback)

                                                // Phase 6.8: Epistemic Reflection Recording
                                                match ctx.Memory.EpisodeService with
                                                | Some svc ->
                                                    let episode =
                                                        Tars.Core.Episode.Reflection(
                                                            state.ExecutorAgentId.ToString(),
                                                            reflectionPrompt,
                                                            DateTime.UtcNow
                                                        )

                                                    svc.Queue(episode)
                                                | None -> ()

                                                let agentWithReflection = currentAgent.ReceiveMessage(reflectionMsg)

                                                let! reflectOutcomeResult =
                                                    runWithTimeout
                                                        "Reflection"
                                                        (graphExecutor.RunAgentLoop(
                                                            agentWithReflection,
                                                            20,
                                                            cancellationToken = cts.Token
                                                        ))

                                                match reflectOutcomeResult with
                                                | Choice2Of2 reason ->
                                                    timeoutOccurred <- true

                                                    currentTrace <-
                                                        currentTrace @ [ $"TIMEOUT during reflection: {reason}" ]

                                                    reflectionCount <- maxReflections
                                                    currentOutput <- currentOutput + "\n[TIMEOUT during reflection]"
                                                | Choice1Of2 reflectOutcome ->
                                                    match reflectOutcome with
                                                    | Success(nextAgent, reflectOutput, reflectTrace) ->
                                                        currentAgent <- nextAgent

                                                        currentTrace <-
                                                            currentTrace
                                                            @ [ $"--- REFLECTION {reflectionCount} ---" ]
                                                            @ reflectTrace

                                                        if
                                                            reflectOutput.Contains("OPTIMAL")
                                                            && String.IsNullOrEmpty verificationFeedback
                                                        then
                                                            isOptimal <- true
                                                        else
                                                            currentOutput <- reflectOutput
                                                    | PartialSuccess((nextAgent, reflectOutput, reflectTrace), _) ->
                                                        currentAgent <- nextAgent

                                                        currentTrace <-
                                                            currentTrace
                                                            @ [ $"--- REFLECTION {reflectionCount} (Partial) ---" ]
                                                            @ reflectTrace

                                                        currentOutput <- reflectOutput
                                                    | Failure err ->
                                                        // If reflection fails, stop and keep previous result
                                                        currentTrace <-
                                                            currentTrace
                                                            @ [ $"--- REFLECTION {reflectionCount} FAILED ---" ]
                                                        // Don't update output, just stop
                                                        reflectionCount <- maxReflections

                                    return
                                        { TaskId = taskDef.Id
                                          TaskGoal = taskDef.Goal
                                          ExecutorId = state.ExecutorAgentId
                                          Success = not timeoutOccurred
                                          Output =
                                            if timeoutOccurred then
                                                currentOutput + "\n[TIMEOUT]"
                                            else
                                                currentOutput
                                          ExecutionTrace = currentTrace
                                          Duration = stopwatch.Elapsed
                                          Evaluation = examplesVerdict }
        }

    /// Runs a Darwin-lite mutation loop on a failed workflow (Phase 15.4)
    let private runDarwinLoop
        (ctx: EvolutionContext)
        (state: EvolutionState)
        (taskDef: TaskDefinition)
        (failedResult: TaskResult)
        =
        task {
            // 1. Log failure to Symbolic Memory
            let metadata =
                Map.ofList
                    [ "goal", taskDef.Goal
                      "executor", string failedResult.ExecutorId
                      "reason", "task_failure" ]

            do!
                SymbolicMemory.logFailure
                    (ctx.Options.RunId
                     |> Option.map (fun (RunId.RunId r) -> r)
                     |> Option.defaultValue (Guid.NewGuid()))
                    None
                    failedResult.Output
                    metadata
                |> Async.StartAsTask

            // 2. Identify target file for mutation
            // We look at ctx.Options.Focus or try to find a .trsx file mentioned in the trace
            let targetFile =
                match ctx.Options.Focus with
                | Some f when f.EndsWith(".trsx") -> Some f
                | _ ->
                    // Fallback: look for any .trsx in the current working directory if it's a small project
                    None

            match targetFile with
            | Some path ->
                ctx.Logger($"[Darwin] Workflow failure detected in {path}. Attempting self-improvement...")

                let! proposalResult =
                    SelfImprovement.analyzeAndPropose
                        ctx.Llm
                        path
                        failedResult.Output
                        (String.concat "\n" failedResult.ExecutionTrace)
                        (ctx.Options.RunId
                         |> Option.map (fun (RunId.RunId r) -> r)
                         |> Option.defaultValue (Guid.NewGuid()))

                match proposalResult with
                | SelfImprovement.Success(p, variantPath) ->
                    ctx.Logger($"[Darwin] Mutation proposed: {p.Rationale}")
                    ctx.Logger($"[Darwin] Applied to variant: {variantPath}")

                    do! SelfImprovement.logImprovement p variantPath true |> Async.StartAsTask
                    return Some p
                | SelfImprovement.Failure err ->
                    ctx.Logger($"[Darwin] Self-improvement failed: {err}")
                    return None
            | None -> return None
        }

    /// The main tick of the evolutionary loop
    let rec step (ctx: EvolutionContext) (state: EvolutionState) =
        task {
            match state.CurrentTask with
            | Some taskDef ->
                // The verdict on an answer: a failing example's, or else the evaluator's.
                let judge (result: TaskResult) =
                    match result.Evaluation, ctx.Governance.Evaluator with
                    // The examples ran with the code (--run-code) and one failed: the task failed.
                    | Some examples, _ when not examples.Passed -> Task.FromResult(Some examples)
                    // They passed, or did not run: the evaluator judges. After passing examples it sees
                    // them in result.Evaluation and judges only what they do not check, such as a
                    // `flatten` the goal asks for without List.concat.
                    | _, Some evaluator ->
                        task {
                            try
                                let! evaluated = evaluator.Evaluate(taskDef, result)
                                return Some evaluated
                            with ex ->
                                ctx.Logger($"[Evaluation] Failed: {ex.Message}")
                                return None
                        }
                    | examples, None -> Task.FromResult examples

                // 2. Execution Phase: Attempt to solve
                let watch = Diagnostics.Stopwatch.StartNew()
                let! result = executeTask ctx state taskDef
                let! evaluation = judge result

                // Both answers share the task's deadline. A Timeout of zero or less means no deadline.
                let left = taskDef.Timeout - watch.Elapsed
                let hasTime = taskDef.Timeout <= TimeSpan.Zero || left > TimeSpan.Zero

                // The evaluator rejected code whose examples passed, and said why: the executor answers
                // once more, told what it said, within the time the task has left, and the new answer is
                // judged against the task as given. In live runs the judge rejected answers that broke a
                // constraint, and the executor never knew.
                let! (result, evaluation) =
                    match evaluation with
                    | Some rejection when reviewerRejected result rejection && hasTime ->
                        task {
                            ctx.Logger($"[Evaluation] Rejected, the executor answers again: {rejection.Summary}")

                            let revisedTask =
                                { withReview taskDef rejection with
                                    Timeout = if taskDef.Timeout > TimeSpan.Zero then left else taskDef.Timeout }

                            let! revised = executeTask ctx state revisedTask
                            let! verdict = judge revised
                            return { revised with Duration = result.Duration + revised.Duration }, verdict
                        }
                    | _ -> Task.FromResult((result, evaluation))

                let resultWithEvaluation = { result with Evaluation = evaluation }

                let evaluationPassed =
                    evaluation
                    |> Option.map (fun e -> e.Passed)
                    |> Option.defaultValue result.Success

                // Update success flag based on BOTH execution and semantic evaluation
                // A task is only truly successful if it executed AND passed semantic validation
                let finalResult =
                    { resultWithEvaluation with
                        Success = result.Success && evaluationPassed }

                match ctx.Memory.Ledger with
                | Some ledger -> do! LedgerIngestion.recordTaskResult ledger ctx.Memory.EvidenceStore ctx.Options.RunId taskDef finalResult ctx.Logger
                | None -> ()

                let resultForDisplay =
                    match evaluation with
                    | Some e when not e.Passed ->
                        { finalResult with
                            Output = finalResult.Output + "\n[SEMANTIC EVAL FAILED] " + e.Summary }
                    | _ -> finalResult

                // 3. Evaluation Phase (semantic)
                if result.Success && evaluationPassed then
                    let mutable newBeliefs = state.ActiveBeliefs

                    // Epistemic Governor: Extract Principle
                    match ctx.Governance.Epistemic with
                    | Some governor ->
                        try
                            let! belief = governor.ExtractPrinciple(taskDef.Goal, finalResult.Output)

                            // Store belief in VectorStore (Buffered)
                            let! embedding = ctx.Llm.EmbedAsync belief.Statement

                            let payload =
                                Map
                                    [ "type", "belief"
                                      "statement", belief.Statement
                                      "context", belief.Context
                                      "confidence", string belief.Confidence
                                      "derived_from", string taskDef.Id ]

                            match ctx.Memory.MemoryBuffer with
                            | Some buffer ->
                                buffer.Accumulate(Belief("tars-beliefs", string belief.Id, embedding, payload))
                            | None ->
                                do! ctx.VectorStore.SaveAsync("tars-beliefs", string belief.Id, embedding, payload)

                            // Phase 6.8: Record Belief Update to Knowledge Graph
                            match ctx.Memory.EpisodeService with
                            | Some svc ->
                                let episode =
                                    Tars.Core.Episode.BeliefUpdate(
                                        "EpistemicGovernor",
                                        belief.Statement,
                                        belief.Confidence,
                                        DateTime.UtcNow
                                    )

                                svc.Queue(episode)
                            | None -> ()

                            // Update Active Beliefs (keep last 10)
                            newBeliefs <- (belief.Statement :: newBeliefs) |> List.truncate 10

                            match ctx.Memory.Ledger with
                            | Some ledger ->
                                do!
                                    LedgerIngestion.recordEpistemicBelief
                                        ledger
                                        ctx.Options.RunId
                                        belief
                                        (Some taskDef.Id)
                                        ctx.Logger
                            | None -> ()
                        with ex ->
                            // Log error but continue
                            printfn $"Epistemic extraction failed: %s{ex.Message}"
                    | None -> ()

                    // Construct a trace object (simplified for now)
                    let trace: MemoryTrace =
                        { TaskId = string taskDef.Id
                          Variables =
                            Map
                                [ "output", box resultWithEvaluation.Output
                                  "trace", box resultWithEvaluation.ExecutionTrace ]
                          StepOutputs = Map.empty }

                    // Save to Semantic Memory (Grow)
                    match ctx.Memory.SemanticMemory with
                    | Some smem ->
                        try
                            let! schemaId = smem.Grow(trace, obj ())
                            ctx.Logger($"[Memory] Grew new memory schema: {schemaId}")
                        with ex ->
                            ctx.Logger($"[Memory] Failed to grow memory: {ex.Message}")
                    | None -> ()

                    // Save to Knowledge            // Ingest trace into KG
                    // Save to Knowledge            // Ingest trace into KG
                    match ctx.Memory.KnowledgeGraph with
                    | Some kg ->
                        try
                            let taskEntity =
                                TarsEntity.ConceptE
                                    { Name = $"Task: {taskDef.Goal}"
                                      Description = taskDef.Goal
                                      RelatedConcepts = [] }

                            let _ = kg.AddNode(taskEntity)

                            let resultEntity =
                                if resultWithEvaluation.Success then
                                    TarsEntity.ConceptE
                                        { Name = "Success"
                                          Description = "Task Success"
                                          RelatedConcepts = [] }
                                else
                                    TarsEntity.ConceptE
                                        { Name = "Failure"
                                          Description = "Task Failure"
                                          RelatedConcepts = [] }

                            let _ = kg.AddFact(TarsFact.DerivedFrom(resultEntity, taskEntity))

                            // Map code structure if available
                            match trace.Variables |> Map.tryFind "code_structure" with
                            | Some(:? CodeStructure as cs) ->
                                for m in cs.Modules do
                                    let modEntity =
                                        TarsEntity.CodeModuleE
                                            { Path = m
                                              Namespace = ""
                                              Dependencies = []
                                              Complexity = 0.0
                                              LineCount = 0 }

                                    let _ = kg.AddFact(TarsFact.BelongsTo(taskEntity, m))
                                    ()

                                for t in cs.Types do
                                    let typeEntity =
                                        TarsEntity.ConceptE
                                            { Name = t
                                              Description = "Type"
                                              RelatedConcepts = [] }

                                    let _ = kg.AddFact(TarsFact.DerivedFrom(typeEntity, taskEntity))
                                    ()

                                for f in cs.Functions do
                                    let funcEntity =
                                        TarsEntity.ConceptE
                                            { Name = f
                                              Description = "Function"
                                              RelatedConcepts = [] }

                                    let _ = kg.AddFact(TarsFact.DerivedFrom(funcEntity, taskEntity))
                                    ()
                            | _ -> ()

                            ctx.Logger($"[KnowledgeGraph] Ingested episode for task: {taskDef.Id}")
                        with ex ->
                            ctx.Logger($"[KnowledgeGraph] Failed to ingest episode: {ex.Message}")
                    | None -> ()

                    // Save to Legacy Memory (Backup)
                    try
                        let! embedding = ctx.Llm.EmbedAsync taskDef.Goal

                        let payload =
                            Map
                                [ "goal", taskDef.Goal
                                  "output", resultWithEvaluation.Output
                                  "generation", string state.Generation ]

                        match ctx.Memory.MemoryBuffer with
                        | Some buffer ->
                            buffer.Accumulate(Legacy("tars-evolution-memory", string taskDef.Id, embedding, payload))
                        | None ->
                            do!
                                ctx.VectorStore.SaveAsync(
                                    "tars-evolution-memory",
                                    string taskDef.Id,
                                    embedding,
                                    payload
                                )
                    with ex ->
                        printfn $"Failed to save to memory: %s{ex.Message}"

                    // Log: Executor → Inform/Failure → Curriculum
                    let responseMsg =
                        SpeechActBridge.informResult
                            state.ExecutorAgentId
                            state.CurriculumAgentId
                            taskDef.Id
                            resultWithEvaluation

                    SpeechActBridge.logSpeechAct ctx.Logger responseMsg

                    // Feature C: Epistemic Verification Checkpoint
                    let! isVerified =
                        match ctx.Governance.Epistemic, resultWithEvaluation.Success with
                        | Some governor, true ->
                            task {
                                try
                                    let statement =
                                        sprintf
                                            "Task '%s' was completed with output: %s"
                                            (taskDef.Goal.Substring(0, min 50 taskDef.Goal.Length))
                                            (resultWithEvaluation.Output.Substring(
                                                0,
                                                min 100 resultWithEvaluation.Output.Length
                                            ))

                                    let! verified = governor.Verify(statement)

                                    if not verified then
                                        ctx.Logger("[Epistemic] ⚠️ Output verification FAILED - possible quality issue")
                                    else
                                        ctx.Logger("[Epistemic] ✓ Output verified")

                                    return verified
                                with ex ->
                                    ctx.Logger($"[Epistemic] Verification skipped: {ex.Message}")
                                    return true // Skip on error, don't block
                            }
                        | _ -> Task.FromResult(true)

                    // Adjust result based on verification (add metadata)
                    let verifiedResult =
                        let baseResult = resultForDisplay

                        if isVerified then
                            baseResult
                        else
                            { baseResult with
                                Output = baseResult.Output + "\n[UNVERIFIED - Review Recommended]" }

                    // Display task completion with generated solution
                    DemoVisualization.showTaskComplete
                        verifiedResult.Success
                        verifiedResult.Output
                        verifiedResult.Duration

                    // Capture episode to Graphiti knowledge graph
                    match ctx.Memory.EpisodeService with
                    | Some svc ->
                        let episode =
                            Tars.Core.Episode.AgentInteraction(
                                "Evolution",
                                taskDef.Goal,
                                (if finalResult.Success then "SUCCESS: " else "FAILED: ")
                                + resultForDisplay.Output,
                                DateTime.UtcNow
                            )

                        svc.Queue(episode)
                        let! _ = svc.FlushAsync()
                        ()
                    | None -> ()

                    return
                        { state with
                            Generation = state.Generation + 1
                            CompletedTasks = resultForDisplay :: state.CompletedTasks
                            CurrentTask = None
                            ActiveBeliefs = newBeliefs }
                else
                    // Retail Darwin logic (Phase 15)
                    let! mutation =
                        if ctx.Options.SelfImprovement then
                            runDarwinLoop ctx state taskDef resultForDisplay
                        else
                            Task.FromResult None

                    // Retry or fail? For now, just log and clear
                    DemoVisualization.showTaskComplete
                        resultForDisplay.Success
                        resultForDisplay.Output
                        resultForDisplay.Duration

                    return
                        { state with
                            CompletedTasks = resultForDisplay :: state.CompletedTasks
                            CurrentTask = None }

            | None ->
                // Check Queue first
                match state.TaskQueue with
                | nextTask :: remainingQueue ->
                    // Execute task immediately instead of just setting it
                    ctx.Logger
                        $"[Evolution] Picking task from queue: {nextTask.Goal.Substring(0, Math.Min(nextTask.Goal.Length, 50))}..."

                    let stateWithTask =
                        { state with
                            CurrentTask = Some nextTask
                            TaskQueue = remainingQueue }
                    // Recursively call step to execute the task
                    return! step ctx stateWithTask
                | [] ->
                    // 1. Curriculum Phase: Generate new tasks
                    let! newTasks = generateTask ctx state

                    match newTasks with
                    | first :: rest ->
                        ctx.Logger
                            $"[Evolution] Generated {newTasks.Length} tasks, executing first: {first.Goal.Substring(0, Math.Min(first.Goal.Length, 50))}..."

                        let stateWithTask =
                            { state with
                                CurrentTask = Some first
                                TaskQueue = rest }
                        // Recursively call step to execute the task
                        return! step ctx stateWithTask
                    | [] ->
                        ctx.Logger "[Evolution] No tasks generated"
                        return state
        }
