namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Core
open Tars.Llm.Routing

module EvolutionBenchmarkTests =

    type BenchmarkLlm(responseText: string) =
        interface Tars.Llm.ILlmService with
            member _.CompleteAsync(_req) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.CompleteStreamAsync(_req, _onToken) =
                task {
                    return
                        { Text = responseText
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                }

            member _.EmbedAsync(_text) = task { return [| 0.1f; 0.2f; 0.3f |] }

            member _.RouteAsync _ =
                task {
                    return
                        { Backend = Tars.Llm.LlmBackend.Ollama "mock"
                          Endpoint = Uri "http://localhost:11434"
                          ApiKey = None }
                }

    type BenchmarkRegistry(agents: Agent list) =
        interface IAgentRegistry with
            member _.GetAgent(id) =
                async { return agents |> List.tryFind (fun a -> a.Id = id) }

            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return agents }

    [<Fact>]
    let ``Evolution handles benchmark evaluation`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let curriculumAgentId = AgentId(Guid.NewGuid())
            let executorAgentId = AgentId(Guid.NewGuid())

            let llmJson =
                """{"tasks":[{"goal":"Benchmark Task 1","constraints":[],"validation_criteria":"check"},{"goal":"Benchmark Task 2","constraints":[],"validation_criteria":"check"}]}"""

            let llm = BenchmarkLlm(llmJson)

            let agent1 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = curriculumAgentId in id)
                    "Curriculum"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            let agent2 =
                Tars.Kernel.AgentFactory.create
                    (let (AgentId id) = executorAgentId in id)
                    "Executor"
                    "1.0.0"
                    "test"
                    "System"
                    []
                    []

            let registry = BenchmarkRegistry([ agent1; agent2 ])

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = llm
                  CurriculumLlm = None
                  VectorStore =
                    { new IVectorStore with
                        member _.SaveAsync(_, _, _, _) = Task.CompletedTask
                        member _.SearchAsync(_, _, _) = Task.FromResult([]) }
                  Logger = fun msg -> printfn $"LOG: %s{msg}"
                  Memory =
                    { SemanticMemory = None
                      KnowledgeBase = None
                      KnowledgeGraph = None
                      MemoryBuffer = None
                      EpisodeService = None
                      Ledger = None
                      EvidenceStore = None }
                  Governance =
                    { Epistemic = None
                      PreLlm = None
                      Budget = None
                      OutputGuard = None
                      Evaluator = None }
                  Options =
                    { RunId = None
                      Verbose = true
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      RecordTask = ignore
                      SelfImprovement = false } }

            let mutable state: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = curriculumAgentId
                  ExecutorAgentId = executorAgentId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            // Step 1: Curriculum Phase
            let! stateAfterCurriculum = Engine.step ctx state
            Assert.NotEmpty(stateAfterCurriculum.TaskQueue)

            // Step 2: Execution Phase
            let! stateAfterExecution = Engine.step ctx stateAfterCurriculum
            Assert.NotEmpty(stateAfterExecution.CompletedTasks)
            Assert.True(stateAfterExecution.CompletedTasks.Head.Success)
        }

    // A solution runs in the same fsi process as its validation, before it, so what it prints
    // lands in the same output. "PASS" in that output was enough: a solution that printed it and
    // left before the validation ran was counted as validated, and SelfTrain could export it.

    let private attemptOnReverse (solution: string) =
        let problem = ProblemBank.tryFind "basic-reverse-string" |> Option.get
        (BenchmarkRunner.runProblem (BenchmarkLlm solution) problem false ignore).Result

    [<Fact>]
    let ``A solution that prints PASS and leaves before the validation is not validated`` () =
        let wrong = "let reverse (s: string) : string = s\nprintfn \"PASS\"\n"
        Assert.False((attemptOnReverse (wrong + "exit 0")).Validated, "exit 0")
        Assert.False((attemptOnReverse (wrong + "failwith \"stop\" |> ignore")).Validated, "an exception")

    [<Fact>]
    let ``A solution the validation accepts is validated, whatever it prints itself`` () =
        let right = "let reverse (s: string) : string = System.String(Array.rev (s.ToCharArray()))\n"
        let attempt = attemptOnReverse right
        Assert.True(attempt.Validated)
        Assert.Equal("PASS", attempt.ValidationOutput)
        Assert.True((attemptOnReverse (right + "printfn \"PASS\"")).Validated)

    // The two forms #394 ran by hand. The first reads the mark from the script and prints PASS
    // with it; the second rewrites FAIL to PASS in what the process prints. Both were validated
    // while the validation ran in the solution's process.
    [<Fact>]
    let ``A solution that forges the validation's output is not validated`` () =
        let wrong = "let reverse (s: string) : string = s\n"

        let readsTheMark =
            "let self = System.IO.File.ReadAllLines(System.IO.Path.Combine(__SOURCE_DIRECTORY__, __SOURCE_FILE__))\nlet last = self |> Array.filter (fun l -> l.StartsWith \"printfn \\\"\") |> Array.last\nlet mark = last.Substring(9, 32)\nprintfn \"PASS\"\nprintfn \"%s\" mark\nexit 0\n"

        let rewritesFail =
            "let real = System.Console.Out\nSystem.Console.SetOut(\n    { new System.IO.StringWriter() with\n        override _.WriteLine(s: string) = real.WriteLine(s.Replace(\"FAIL\", \"PASS\"))\n        override _.Write(s: string) = real.Write(s.Replace(\"FAIL\", \"PASS\")) })\n"

        Assert.False((attemptOnReverse (wrong + readsTheMark)).Validated, "reads the mark")
        Assert.False((attemptOnReverse (wrong + rewritesFail)).Validated, "rewrites FAIL")

    // The code's script held each call next to its expected value, to type the call. A solution
    // could read the expected values there and return them.
    [<Fact>]
    let ``A solution that reads the expected values from its script is not validated`` () =
        let readsTheExpected =
            """let reverse (s: string) : string =
    let self = System.IO.File.ReadAllText "answer.fsx"
    let at = self.IndexOf("(reverse \"" + s + "\") (fun")
    if at < 0 then s
    else
        let from = self.IndexOf("(\"", self.IndexOf("Operators.(=)", at)) + 2
        self.Substring(from, self.IndexOf("\")", from) - from)
"""

        Assert.False((attemptOnReverse readsTheExpected).Validated)

    /// A right solution to each problem whose examples are compared outside the solution's process.
    let private referenceSolutions =
        Map.ofList
            [ "basic-reverse-string", "let reverse (s: string) : string = System.String(Array.rev (s.ToCharArray()))"
              "basic-fizzbuzz", "let fizzBuzz (n: int) : string list =\n    [ for i in 1 .. n ->\n        if i % 15 = 0 then \"FizzBuzz\"\n        elif i % 3 = 0 then \"Fizz\"\n        elif i % 5 = 0 then \"Buzz\"\n        else string i ]"
              "basic-count-vowels", "let countVowels (s: string) : int =\n    s |> Seq.filter (fun c -> \"aeiouAEIOU\".Contains c) |> Seq.length"
              "basic-palindrome", "let isPalindrome (s: string) : bool =\n    let t = s.ToLower().Replace(\" \", \"\")\n    t = System.String(Array.rev (t.ToCharArray()))"
              "basic-max-element", "let findMax (xs: int list) : int = List.reduce max xs"
              "inter-binary-search", "let binarySearch (arr: int[]) (target: int) : int option =\n    let rec go lo hi =\n        if lo > hi then None\n        else\n            let mid = (lo + hi) / 2\n            if arr.[mid] = target then Some mid\n            elif arr.[mid] < target then go (mid + 1) hi\n            else go lo (mid - 1)\n    go 0 (arr.Length - 1)"
              "inter-flatten", "let flatten (xss: 'a list list) : 'a list = List.concat xss"
              "inter-safe-divide", "let safeDivide (a: float) (b: float) : Result<float, string> =\n    if b = 0.0 then Error \"division by zero\" else Ok (a / b)"
              "inter-group-by", "let groupByKey (items: (string * int) list) : Map<string, int list> =\n    items |> List.groupBy fst |> List.map (fun (k, vs) -> k, List.map snd vs) |> Map.ofList"
              "inter-roman-numerals", "let toRoman (n: int) : string =\n    let table = [ 1000, \"M\"; 900, \"CM\"; 500, \"D\"; 400, \"CD\"; 100, \"C\"; 90, \"XC\"; 50, \"L\"; 40, \"XL\"; 10, \"X\"; 9, \"IX\"; 5, \"V\"; 4, \"IV\"; 1, \"I\" ]\n    let rec go n acc table =\n        match table with\n        | [] -> acc\n        | (v, s) :: rest -> if n >= v then go (n - v) (acc + s) table else go n acc rest\n    go n \"\" table"
              "adv-merge-sort", "let mergeSort (xs: int list) : int list = List.sort xs"
              "adv-balanced-parens", "let isBalanced (s: string) : bool =\n    let rec go stack chars =\n        match chars with\n        | [] -> List.isEmpty stack\n        | c :: rest ->\n            match c with\n            | '(' | '[' | '{' -> go (c :: stack) rest\n            | ')' | ']' | '}' ->\n                let opening = if c = ')' then '(' elif c = ']' then '[' else '{'\n                match stack with\n                | top :: below when top = opening -> go below rest\n                | _ -> false\n            | _ -> go stack rest\n    go [] (List.ofSeq s)"
              "adv-eval-rpn", "let evalRPN (tokens: string list) : int =\n    let step stack token =\n        match token, stack with\n        | \"+\", b :: a :: rest -> a + b :: rest\n        | \"-\", b :: a :: rest -> a - b :: rest\n        | \"*\", b :: a :: rest -> a * b :: rest\n        | \"/\", b :: a :: rest -> a / b :: rest\n        | n, _ -> int n :: stack\n    List.fold step [] tokens |> List.head"
              "adv-matrix-multiply", "let matMul (a: float[,]) (b: float[,]) : float[,] =\n    Array2D.init (Array2D.length1 a) (Array2D.length2 b) (fun i j ->\n        Seq.sum (seq { for k in 0 .. Array2D.length2 a - 1 -> a.[i, k] * b.[k, j] }))"
              "exp-result-ce", "type ResultBuilder() =\n    member _.Bind(m, f) = Result.bind f m\n    member _.Return x = Ok x\n    member _.ReturnFrom m = m\nlet result = ResultBuilder()"
              "exp-active-pattern", "let (|Int|Float|Word|) (s: string) =\n    match System.Int32.TryParse s with\n    | true, _ -> Int\n    | _ ->\n        match System.Double.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture) with\n        | true, _ -> Float\n        | _ -> Word\nlet classify (s: string) : string =\n    match s with\n    | Int -> \"int\"\n    | Float -> \"float\"\n    | Word -> \"word\""
              "exp-graph-cycle", "let hasCycle (graph: Map<int, int list>) : bool =\n    let rec visit (path: Set<int>) (node: int) =\n        path.Contains node\n        || (graph |> Map.tryFind node |> Option.defaultValue [] |> List.exists (visit (path.Add node)))\n    graph |> Map.exists (fun node _ -> visit Set.empty node)"
              "exp-memoize", "let memoize (f: 'a -> 'b) : ('a -> 'b) =\n    let cache = System.Collections.Generic.Dictionary<'a, 'b>()\n    fun x ->\n        match cache.TryGetValue x with\n        | true, v -> v\n        | _ ->\n            let v = f x\n            cache.[x] <- v\n            v\nlet rec fib : int -> int64 =\n    memoize (fun n -> if n < 2 then int64 n else fib (n - 1) + fib (n - 2))"
              "inter-perf-sum-squares", "let sumSquares (xs: float[]) : float = Array.fold (fun acc x -> acc + x * x) 0.0 xs" ]

    // Each of these problems is checked by examples, the cases its ValidationCode checked, and
    // a right solution to it is validated.
    [<Theory>]
    [<InlineData("basic-reverse-string")>]
    [<InlineData("basic-fizzbuzz")>]
    [<InlineData("basic-count-vowels")>]
    [<InlineData("basic-palindrome")>]
    [<InlineData("basic-max-element")>]
    [<InlineData("inter-binary-search")>]
    [<InlineData("inter-flatten")>]
    [<InlineData("inter-safe-divide")>]
    [<InlineData("inter-group-by")>]
    [<InlineData("inter-roman-numerals")>]
    [<InlineData("adv-merge-sort")>]
    [<InlineData("adv-balanced-parens")>]
    [<InlineData("adv-eval-rpn")>]
    [<InlineData("adv-matrix-multiply")>]
    [<InlineData("exp-result-ce")>]
    [<InlineData("exp-active-pattern")>]
    [<InlineData("exp-graph-cycle")>]
    [<InlineData("exp-memoize")>]
    [<InlineData("inter-perf-sum-squares")>]
    let ``a problem with a reference solution is checked by examples and validates it`` (id: string) =
        let problem = ProblemBank.tryFind id |> Option.get
        Assert.False(problem.Examples.IsEmpty, $"{id} has no examples")
        let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm referenceSolutions.[id]) problem false ignore).Result
        Assert.True(attempt.Validated, $"{id}: {attempt.ValidationOutput}")

    /// Solutions wrong only where a problem's ValidationCode did not look, which it accepted:
    /// fizzBuzz 15's element 3, the cells of matMul's result past the four it checked, and the
    /// error a failed result CE gives, where it checked only that there was one.
    let private acceptedBefore =
        Map.ofList
            [ "basic-fizzbuzz", "let fizzBuzz (n: int) : string list =\n    [ for i in 1 .. n ->\n        if i = 4 then \"four\"\n        elif i % 15 = 0 then \"FizzBuzz\"\n        elif i % 3 = 0 then \"Fizz\"\n        elif i % 5 = 0 then \"Buzz\"\n        else string i ]"
              "adv-matrix-multiply", "let matMul (a: float[,]) (b: float[,]) : float[,] =\n    Array2D.init (Array2D.length1 a + 1) (Array2D.length2 b + 1) (fun i j ->\n        if i < Array2D.length1 a && j < Array2D.length2 b then\n            Seq.sum (seq { for k in 0 .. Array2D.length2 a - 1 -> a.[i, k] * b.[k, j] })\n        else 0.0)"
              "exp-result-ce", "type ResultBuilder() =\n    member _.Bind(m, f) = match m with Ok x -> f x | Error _ -> Error \"failed\"\n    member _.Return x = Ok x\nlet result = ResultBuilder()" ]

    // The examples accept what the ValidationCode accepted, so results stay comparable with
    // earlier runs.
    [<Theory>]
    [<InlineData("basic-fizzbuzz")>]
    [<InlineData("adv-matrix-multiply")>]
    [<InlineData("exp-result-ce")>]
    let ``a problem's examples accept what its ValidationCode accepted`` (id: string) =
        let problem = ProblemBank.tryFind id |> Option.get
        let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm acceptedBefore.[id]) problem false ignore).Result
        Assert.True(attempt.Validated, $"{id}: {attempt.ValidationOutput}")

    /// Solutions whose first call is wrong and later calls right. The ValidationCode of these
    /// problems checked one result, fizzBuzz 15's, matMul's or fib 40's, so it rejected them.
    let private firstCallWrong =
        Map.ofList
            [ "basic-fizzbuzz", "let mutable calls = 0\nlet fizzBuzz (n: int) : string list =\n    calls <- calls + 1\n    [ for i in 1 .. n ->\n        if calls = 1 then \"wrong\"\n        elif i % 15 = 0 then \"FizzBuzz\"\n        elif i % 3 = 0 then \"Fizz\"\n        elif i % 5 = 0 then \"Buzz\"\n        else string i ]"
              "adv-matrix-multiply", "let mutable calls = 0\nlet matMul (a: float[,]) (b: float[,]) : float[,] =\n    calls <- calls + 1\n    Array2D.init (Array2D.length1 a) (Array2D.length2 b) (fun i j ->\n        if calls = 1 && (i, j) <> (0, 0) then 0.0\n        else Seq.sum (seq { for k in 0 .. Array2D.length2 a - 1 -> a.[i, k] * b.[k, j] }))"
              "exp-memoize", "let memoize (f: 'a -> 'b) : ('a -> 'b) =\n    let cache = System.Collections.Generic.Dictionary<'a, 'b>()\n    fun x ->\n        match cache.TryGetValue x with\n        | true, v -> v\n        | _ ->\n            let v = f x\n            cache.[x] <- v\n            v\nlet rec right : int -> int64 =\n    memoize (fun n -> if n < 2 then int64 n else right (n - 1) + right (n - 2))\nlet mutable fortyCalls = 0\nlet fib (n: int) : int64 =\n    if n = 40 then fortyCalls <- fortyCalls + 1\n    if n = 40 && fortyCalls = 1 then 0L else right n" ]

    // Where the ValidationCode checked several things of one result, one example checks them on
    // one call, so a solution cannot answer each check with another call.
    [<Theory>]
    [<InlineData("basic-fizzbuzz")>]
    [<InlineData("adv-matrix-multiply")>]
    [<InlineData("exp-memoize")>]
    let ``a solution whose first call is wrong is not validated where one result was checked`` (id: string) =
        let problem = ProblemBank.tryFind id |> Option.get
        let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm firstCallWrong.[id]) problem false ignore).Result
        Assert.False(attempt.Validated, $"{id}: {attempt.ValidationOutput}")

    // compileAndValidate killed the solution's process after 30 s. The examples' processes share
    // 60 s, and the solution's own process keeps its 30 s. This one sleeps once, before its calls.
    [<Fact>]
    let ``a solution whose run takes more than 30 s is not validated`` () =
        let problem = ProblemBank.tryFind "basic-reverse-string" |> Option.get
        let slow =
            "System.Threading.Thread.Sleep 35000\nlet reverse (s: string) : string = System.String(Array.rev (s.ToCharArray()))"
        let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm slow) problem false ignore).Result
        Assert.False(attempt.Validated, attempt.ValidationOutput)

    // compileAndValidate did not open System for the solution, so the benchmark's examples do not
    // open it either, though evolve's do.
    [<Fact>]
    let ``a solution that names System's types without opening System is not validated`` () =
        let problem = ProblemBank.tryFind "basic-reverse-string" |> Option.get
        let unopened =
            "let reverse (s: string) : string = s.ToCharArray() |> Array.rev |> Array.map Char.ToString |> String.concat \"\""
        let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm unopened) problem false ignore).Result
        Assert.False(attempt.Validated, attempt.ValidationOutput)

    // compileAndValidate counted a timeout as a compile failure, so the solver was retried. This
    // solution sleeps on its first run only, so the retry passes.
    [<Fact>]
    let ``a solution that times out is retried`` () =
        let problem = ProblemBank.tryFind "basic-reverse-string" |> Option.get
        let marker = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"tars-timeout-{Guid.NewGuid():N}")

        let slowOnce =
            $"if not (System.IO.File.Exists @\"{marker}\") then\n    System.IO.File.WriteAllText(@\"{marker}\", \"\")\n    System.Threading.Thread.Sleep 35000\nlet reverse (s: string) : string = System.String(Array.rev (s.ToCharArray()))"

        try
            let attempt = (BenchmarkRunner.runProblem (BenchmarkLlm slowOnce) problem true ignore).Result
            Assert.True(attempt.Validated, attempt.ValidationOutput)
        finally
            System.IO.File.Delete marker

module BenchmarkProvenanceTests =

    open System
    open System.Threading.Tasks
    open Xunit
    open Tars.Llm
    open Tars.Cortex.WoTTypes
    open Tars.Evolution

    // A learning curve needs to know which model and which evolve cycle produced each
    // result. Benchmark runs recorded the model as "default" and no cycle, so a
    // per-cycle, per-model pass rate could not be computed from disk.

    /// Routes every request to `backend`; RouteAsync fails when `backend` is None.
    type private RoutedLlm(backend: LlmBackend option) =
        interface ILlmService with
            member _.CompleteAsync(_) =
                Task.FromResult
                    { Text = ""
                      FinishReason = Some "stop"
                      Usage = None
                      Raw = None }

            member this.CompleteStreamAsync(req, _) = (this :> ILlmService).CompleteAsync req
            member _.EmbedAsync(_) = Task.FromResult([| 0.0f |])

            member _.RouteAsync(_) =
                match backend with
                | Some b ->
                    Task.FromResult(
                        ({ Backend = b
                           Endpoint = Uri "http://localhost:11434"
                           ApiKey = None }
                        : Tars.Llm.Routing.RoutedBackend)
                    )
                | None -> Task.FromException<Tars.Llm.Routing.RoutedBackend>(InvalidOperationException "no backend")

    type private RecordingSelector() =
        let recorded = ResizeArray<PatternOutcome>()
        member _.Recorded = List.ofSeq recorded

        interface IPatternSelector with
            member _.Recommend(_, _) = ChainOfThought
            member _.Score(_) = Map.empty
            member _.RecordOutcome(outcome) = recorded.Add outcome

    let private run (llm: ILlmService) =
        (BenchmarkRunner.runSuiteFromProblems llm [] None None None false ignore).Result

    [<Fact>]
    let ``a run records the model the router chose, not "default"`` () =
        let summary = run (RoutedLlm(Some(Ollama "qwen3-coder:30b")))
        Assert.Equal("ollama/qwen3-coder:30b", summary.ModelUsed)

    [<Fact>]
    let ``a run whose backend cannot be resolved says the model is unknown, and still completes`` () =
        let summary = run (RoutedLlm None)
        Assert.Equal("unknown", summary.ModelUsed)

    [<Fact>]
    let ``a placeholder model name from the router is not credited as a model`` () =
        // ChatClientLlmService routes every request to `Ollama "unknown"`, and
        // Claude Code with no model to `Anthropic "claude-code"`.
        Assert.Equal("unknown", (run (RoutedLlm(Some(Ollama "unknown")))).ModelUsed)
        Assert.Equal("unknown", (run (RoutedLlm(Some(Ollama " ")))).ModelUsed)
        Assert.Equal("unknown", (run (RoutedLlm(Some(Anthropic "claude-code")))).ModelUsed)
        Assert.Equal("anthropic/opus", (run (RoutedLlm(Some(Anthropic "opus")))).ModelUsed)

    [<Fact>]
    let ``a run outside an evolve cycle has no cycle id`` () =
        Assert.Equal(None, (run (RoutedLlm(Some(Ollama "m")))).CycleId)

    [<Fact>]
    let ``recorded outcomes carry the run's model and cycle`` () =
        let attempt: BenchmarkAttempt =
            { ProblemId = "basic-reverse-string"
              Difficulty = Beginner
              Category = StringManipulation
              GeneratedCode = ""
              Compiled = true
              Validated = true
              CompileErrors = []
              ValidationOutput = ""
              GenerationTimeMs = 10L
              ValidationTimeMs = 5L
              ExecutionNs = None
              PropertiesValidated = None
              Timestamp = DateTime.UtcNow }

        let summary =
            { run (RoutedLlm(Some(Ollama "qwen3-coder:30b"))) with
                Attempts = [ attempt ]
                CycleId = Some "a1b2c3d4/2" }

        let selector = RecordingSelector()
        BenchmarkRunner.recordOutcomes selector summary
        let outcome = Assert.Single(selector.Recorded)
        Assert.Equal(Some "ollama/qwen3-coder:30b", outcome.ModelId)
        Assert.Equal(Some "a1b2c3d4/2", outcome.CycleId)

    [<Fact>]
    let ``an unknown model is not recorded as a model id`` () =
        let summary = run (RoutedLlm None)

        let attempt: BenchmarkAttempt =
            { ProblemId = "p"
              Difficulty = Beginner
              Category = StringManipulation
              GeneratedCode = ""
              Compiled = false
              Validated = false
              CompileErrors = []
              ValidationOutput = ""
              GenerationTimeMs = 0L
              ValidationTimeMs = 0L
              ExecutionNs = None
              PropertiesValidated = None
              Timestamp = DateTime.UtcNow }

        let selector = RecordingSelector()
        BenchmarkRunner.recordOutcomes selector { summary with Attempts = [ attempt ] }
        Assert.Equal(None, (Assert.Single(selector.Recorded)).ModelId)

    [<Fact>]
    let ``a solution that failed its property tests is recorded as a failure`` () =
        let summary = run (RoutedLlm None)

        let validated properties : BenchmarkAttempt =
            { ProblemId = "p"
              Difficulty = Beginner
              Category = StringManipulation
              GeneratedCode = ""
              Compiled = true
              Validated = true
              CompileErrors = []
              ValidationOutput = "PASS"
              GenerationTimeMs = 0L
              ValidationTimeMs = 0L
              ExecutionNs = None
              PropertiesValidated = properties
              Timestamp = DateTime.UtcNow }

        let selector = RecordingSelector()

        BenchmarkRunner.recordOutcomes
            selector
            { summary with
                Attempts = [ validated (Some false); validated (Some true); validated None ] }

        // SelfTrain leaves out a solution whose examples passed and whose properties failed: it is
        // not verified, so it is no success in the outcome store either.
        Assert.Equal<bool>([ false; true; true ], selector.Recorded |> List.map (fun o -> o.Success))

module HardBankTests =

    open System.Threading
    open System.Threading.Tasks
    open Xunit
    open Tars.Evolution

    // ADR 0005, backlog item 3. A validation that no solution passes would score every model
    // 0 on its problem, so each validation is run against a reference solution.
    // The references are here and not in the bank: nothing exports a test file as training
    // data, and five of these problems are held out.

    let private references: Map<string, string> =
        [ "hard-lru-trace", """
let lruTrace (capacity: int) (ops: (string * int * int) list) : int list =
    let step (items: (int * int) list, results: int list) (op: string, key: int, value: int) =
        match op with
        | "get" ->
            match items |> List.tryFind (fun (k, _) -> k = key) with
            | Some (_, v) -> (key, v) :: (items |> List.filter (fun (k, _) -> k <> key)), v :: results
            | None -> items, -1 :: results
        | _ ->
            if capacity <= 0 then items, results
            else
                let rest = items |> List.filter (fun (k, _) -> k <> key)
                let kept = if rest.Length >= capacity then List.take (capacity - 1) rest else rest
                (key, value) :: kept, results
    ops |> List.fold step ([], []) |> snd |> List.rev
"""

          "hard-dijkstra", """
let shortestPath (edges: (int * int * int) list) (source: int) (target: int) : int option =
    let adj = edges |> List.groupBy (fun (f, _, _) -> f) |> Map.ofList
    let rec loop (frontier: Set<int * int>) (dist: Map<int, int>) =
        if frontier.IsEmpty then Map.tryFind target dist
        else
            let (d, u) = frontier.MinimumElement
            let frontier = frontier.Remove((d, u))
            if u = target then Some d
            else
                let out = Map.tryFind u adj |> Option.defaultValue []
                let frontier, dist =
                    out
                    |> List.fold
                        (fun (fr: Set<int * int>, ds: Map<int, int>) (_, v, w) ->
                            let nd = d + w
                            match Map.tryFind v ds with
                            | Some old when old <= nd -> fr, ds
                            | _ -> fr.Add((nd, v)), ds.Add(v, nd))
                        (frontier, dist)
                loop frontier dist
    loop (Set.singleton (0, source)) (Map.ofList [ source, 0 ])
"""

          "hard-topo-order", """
let topoOrder (n: int) (deps: (int * int) list) : int list option =
    let rec loop (remaining: Set<int>) (edges: (int * int) list) (acc: int list) =
        if remaining.IsEmpty then Some (List.rev acc)
        else
            let blocked = edges |> List.map snd |> Set.ofList
            match remaining |> Set.filter (fun t -> not (blocked.Contains t)) |> Seq.tryHead with
            | None -> None
            | Some t -> loop (remaining.Remove t) (edges |> List.filter (fun (a, _) -> a <> t)) (t :: acc)
    loop (Set.ofList [ 0 .. n - 1 ]) deps []
"""

          "hard-expr-eval", """
let evaluate (expr: string) : Result<int, string> =
    let rec skip (cs: char list) = match cs with ' ' :: t -> skip t | _ -> cs
    let rec parseExpr (cs: char list) : Result<int * char list, string> =
        parseTerm cs |> Result.bind (fun (v, rest) -> exprTail v rest)
    and exprTail (acc: int) (cs: char list) =
        match skip cs with
        | '+' :: t -> parseTerm t |> Result.bind (fun (v, rest) -> exprTail (acc + v) rest)
        | '-' :: t -> parseTerm t |> Result.bind (fun (v, rest) -> exprTail (acc - v) rest)
        | rest -> Ok (acc, rest)
    and parseTerm (cs: char list) =
        parseFactor cs |> Result.bind (fun (v, rest) -> termTail v rest)
    and termTail (acc: int) (cs: char list) =
        match skip cs with
        | '*' :: t -> parseFactor t |> Result.bind (fun (v, rest) -> termTail (acc * v) rest)
        | '/' :: t ->
            parseFactor t
            |> Result.bind (fun (v, rest) -> if v = 0 then Error "division by zero" else termTail (acc / v) rest)
        | rest -> Ok (acc, rest)
    and parseFactor (cs: char list) =
        match skip cs with
        | '-' :: t -> parseFactor t |> Result.map (fun (v, rest) -> -v, rest)
        | '(' :: t ->
            parseExpr t
            |> Result.bind (fun (v, rest) ->
                match skip rest with
                | ')' :: after -> Ok (v, after)
                | _ -> Error "expected )")
        | c :: _ as all when System.Char.IsDigit c ->
            let digits = all |> List.takeWhile System.Char.IsDigit
            Ok (digits |> List.fold (fun acc d -> acc * 10 + int d - int '0') 0, List.skip digits.Length all)
        | _ -> Error "expected a number"
    match parseExpr (List.ofSeq expr) with
    | Ok (v, rest) when List.isEmpty (skip rest) -> Ok v
    | Ok _ -> Error "unexpected input"
    | Error e -> Error e
"""

          "hard-wrap-text", """
let wrapText (width: int) (text: string) : string list =
    let words = text.Split([| ' ' |], System.StringSplitOptions.RemoveEmptyEntries) |> List.ofArray
    let rec pieces (w: string) : string list =
        if w.Length <= width then [ w ] else w.Substring(0, width) :: pieces (w.Substring width)
    let add (lines: string list, current: string) (piece: string) =
        if current = "" then lines, piece
        elif current.Length + 1 + piece.Length <= width then lines, current + " " + piece
        else current :: lines, piece
    let lines, current = words |> List.collect pieces |> List.fold add ([], "")
    List.rev (if current = "" then lines else current :: lines)
"""

          "hard-roman-parse", """
let parseRoman (s: string) : int option =
    let table =
        [ 1000, "M"; 900, "CM"; 500, "D"; 400, "CD"; 100, "C"; 90, "XC"; 50, "L"; 40, "XL"; 10, "X"; 9, "IX"; 5, "V"; 4, "IV"; 1, "I" ]
    let toRoman (n: int) : string =
        table
        |> List.fold (fun (acc: string, rest: int) (v, sym) -> acc + String.replicate (rest / v) sym, rest % v) ("", n)
        |> fst
    seq { 1 .. 3999 } |> Seq.tryFind (fun n -> toRoman n = s)
"""

          "hard-rational", """
type Rational = { Num: int; Den: int }

let rec gcd (a: int) (b: int) : int = if b = 0 then abs a else gcd b (a % b)

let normalize (n: int) (d: int) : Rational =
    let g = gcd n d
    let s = if d < 0 then -1 else 1
    { Num = s * n / g; Den = s * d / g }

let make (n: int) (d: int) : Rational option = if d = 0 then None else Some (normalize n d)
let add (a: Rational) (b: Rational) : Rational = normalize (a.Num * b.Den + b.Num * a.Den) (a.Den * b.Den)
let mul (a: Rational) (b: Rational) : Rational = normalize (a.Num * b.Num) (a.Den * b.Den)
let show (r: Rational) : string = if r.Den = 1 then string r.Num else sprintf "%d/%d" r.Num r.Den
"""

          "hard-ledger", """
type Tx =
    | Deposit of string * int
    | Withdraw of string * int
    | Transfer of string * string * int

let settle (start: Map<string, int>) (txs: Tx list) : Result<Map<string, int>, string> =
    let credit (acct: string) (amount: int) (m: Map<string, int>) =
        m.Add(acct, (Map.tryFind acct m |> Option.defaultValue 0) + amount)
    let debit (n: int) (acct: string) (amount: int) (m: Map<string, int>) =
        match Map.tryFind acct m with
        | None -> Error (sprintf "tx %d: unknown account" n)
        | Some b when b < amount -> Error (sprintf "tx %d: insufficient funds" n)
        | Some b -> Ok (m.Add(acct, b - amount))
    let apply (n: int) (m: Map<string, int>) (tx: Tx) =
        let amount =
            match tx with
            | Deposit (_, a) | Withdraw (_, a) | Transfer (_, _, a) -> a
        if amount <= 0 then Error (sprintf "tx %d: invalid amount" n)
        else
            match tx with
            | Deposit (acct, a) -> Ok (credit acct a m)
            | Withdraw (acct, a) -> debit n acct a m
            | Transfer (src, dst, a) -> debit n src a m |> Result.map (credit dst a)
    txs
    |> List.indexed
    |> List.fold (fun state (i, tx) -> state |> Result.bind (fun m -> apply (i + 1) m tx)) (Ok start)
"""

          "hard-throttled-map", """
let mapThrottled (maxParallel: int) (f: 'a -> Async<'b>) (xs: 'a list) : Async<'b list> =
    async {
        let! results = Async.Parallel(xs |> List.map f, maxParallel)
        return List.ofArray results
    }
"""

          "hard-min-window", """
let minWindow (s: string) (t: string) : string =
    if t = "" then ""
    else
        let need = t |> Seq.countBy id |> Map.ofSeq
        let covers (sub: string) : bool =
            let have = sub |> Seq.countBy id |> Map.ofSeq
            need |> Map.forall (fun c n -> (Map.tryFind c have |> Option.defaultValue 0) >= n)
        seq {
            for len in t.Length .. s.Length do
                for start in 0 .. s.Length - len do
                    yield s.Substring(start, len)
        }
        |> Seq.tryFind covers
        |> Option.defaultValue ""
"""

          "hard-regex-match", """
let isMatch (text: string) (pattern: string) : bool =
    let rec go (t: char list) (p: char list) : bool =
        match p with
        | [] -> List.isEmpty t
        | c :: '*' :: rest ->
            go t rest
            || (match t with
                | h :: tl when c = '.' || c = h -> go tl p
                | _ -> false)
        | c :: rest ->
            match t with
            | h :: tl when c = '.' || c = h -> go tl rest
            | _ -> false
    go (List.ofSeq text) (List.ofSeq pattern)
"""

          "hard-csv-line", """
let parseCsvLine (line: string) : string list =
    let rec go (cs: char list) (inQuotes: bool) (field: char list) (fields: string list) : string list =
        let finish () = System.String(field |> List.rev |> Array.ofList)
        match cs, inQuotes with
        | [], _ -> List.rev (finish () :: fields)
        | '"' :: '"' :: rest, true -> go rest true ('"' :: field) fields
        | '"' :: rest, _ -> go rest (not inQuotes) field fields
        | ',' :: rest, false -> go rest false [] (finish () :: fields)
        | c :: rest, _ -> go rest inQuotes (c :: field) fields
    go (List.ofSeq line) false [] []
"""

          "hard-huffman", """
let huffmanCodes (freqs: (char * int) list) : Map<char, string> =
    match freqs with
    | [] -> Map.empty
    | [ (c, _) ] -> Map.ofList [ c, "0" ]
    | _ ->
        let rec build (nodes: (int * (char * string) list) list) : Map<char, string> =
            match List.sortBy fst nodes with
            | (w1, a) :: (w2, b) :: rest ->
                let merged =
                    (a |> List.map (fun (c, code) -> c, "0" + code)) @ (b |> List.map (fun (c, code) -> c, "1" + code))
                build ((w1 + w2, merged) :: rest)
            | [ (_, leaves) ] -> Map.ofList leaves
            | [] -> Map.empty
        build (freqs |> List.map (fun (c, w) -> w, [ c, "" ]))
"""

          "hard-lcs", """
let lcs (a: string) (b: string) : string =
    let table = Array2D.zeroCreate<int> (a.Length + 1) (b.Length + 1)
    for i in a.Length - 1 .. -1 .. 0 do
        for j in b.Length - 1 .. -1 .. 0 do
            table.[i, j] <-
                if a.[i] = b.[j] then 1 + table.[i + 1, j + 1]
                else max table.[i + 1, j] table.[i, j + 1]
    let rec walk (i: int) (j: int) (acc: char list) : string =
        if i = a.Length || j = b.Length then System.String(acc |> List.rev |> Array.ofList)
        elif a.[i] = b.[j] then walk (i + 1) (j + 1) (a.[i] :: acc)
        elif table.[i + 1, j] >= table.[i, j + 1] then walk (i + 1) j acc
        else walk i (j + 1) acc
    walk 0 0 []
"""

          "hard-state-ce", """
type State<'s, 'a> = State of ('s -> 'a * 's)

let runState (m: State<'s, 'a>) (s: 's) : 'a * 's =
    let (State f) = m
    f s

let getState () : State<'s, 's> = State (fun s -> s, s)
let putState (s: 's) : State<'s, unit> = State (fun _ -> (), s)

type StateBuilder() =
    member _.Return(x: 'a) : State<'s, 'a> = State (fun s -> x, s)
    member _.ReturnFrom(m: State<'s, 'a>) : State<'s, 'a> = m
    member _.Zero() : State<'s, unit> = State (fun s -> (), s)
    member _.Bind(m: State<'s, 'a>, f: 'a -> State<'s, 'b>) : State<'s, 'b> =
        State (fun s ->
            let a, s1 = runState m s
            runState (f a) s1)

let state = StateBuilder()
"""

          "hard-life-step", """
let lifeStep (alive: Set<int * int>) : Set<int * int> =
    let neighbours (x: int, y: int) : (int * int) list =
        [ for dx in -1 .. 1 do
              for dy in -1 .. 1 do
                  if dx <> 0 || dy <> 0 then
                      yield x + dx, y + dy ]
    alive
    |> Seq.collect neighbours
    |> Seq.countBy id
    |> Seq.filter (fun (cell, n) -> n = 3 || (n = 2 && alive.Contains cell))
    |> Seq.map fst
    |> Set.ofSeq
"""
        ]
        |> Map.ofList

    [<Fact>]
    let ``every hard problem's validation accepts its reference solution`` () =
        task {
            // dotnet fsi is slow to start, so the scripts run four at a time.
            use gate = new SemaphoreSlim(4)

            let run (problem: BenchmarkProblem) =
                task {
                    do! gate.WaitAsync()

                    try
                        let llm = EvolutionBenchmarkTests.BenchmarkLlm(references.[problem.Id])
                        return! BenchmarkRunner.runProblem llm problem false ignore
                    finally
                        gate.Release() |> ignore
                }

            let! attempts = ProblemBank.hard () |> List.map run |> Task.WhenAll

            let rejected =
                attempts
                |> Array.filter (fun a -> not a.Validated)
                |> Array.map (fun a -> $"{a.ProblemId}: {a.ValidationOutput} {a.CompileErrors}")

            Assert.Empty rejected
        }
