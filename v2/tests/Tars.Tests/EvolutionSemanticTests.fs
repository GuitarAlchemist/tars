namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Evolution
open Tars.Core

module EvolutionSemanticTests =

    type SuccessLlm(responseText: string) =
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

    type MockRegistry(agents: Agent list) =
        interface IAgentRegistry with
            member _.GetAgent(id) =
                async { return agents |> List.tryFind (fun a -> a.Id = id) }

            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return agents }

    /// Answers like SuccessLlm and counts the requests it answers.
    type CountingLlm(responseText: string) =
        let answers = SuccessLlm(responseText) :> Tars.Llm.ILlmService
        let mutable requests = 0
        member _.Requests = requests

        interface Tars.Llm.ILlmService with
            member _.CompleteAsync(req) =
                requests <- requests + 1
                answers.CompleteAsync(req)

            member _.CompleteStreamAsync(req, onToken) =
                requests <- requests + 1
                answers.CompleteStreamAsync(req, onToken)

            member _.EmbedAsync(text) = answers.EmbedAsync(text)
            member _.RouteAsync(req) = answers.RouteAsync(req)

    [<Fact>]
    let ``The teacher's model writes the tasks and the executor's model answers them`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let curriculumAgentId = AgentId(Guid.NewGuid())
            let executorAgentId = AgentId(Guid.NewGuid())

            let agent (AgentId id) name =
                Tars.Kernel.AgentFactory.create id name "1.0.0" "test" "System" [] []

            let teacher =
                CountingLlm(
                    "{\"tasks\":[{\"goal\":\"Write `isEven : int -> bool` in F#\",\"constraints\":[],\"validation_criteria\":\"isEven 4 = true; isEven 5 = false\"}]}"
                )

            let executor =
                CountingLlm("ACT: INFORM: Done.\n```fsharp\nlet isEven (n: int) = n % 2 = 0\n```")

            let ctx: Engine.EvolutionContext =
                { Registry = MockRegistry([ agent curriculumAgentId "Curriculum"; agent executorAgentId "Executor" ])
                  Llm = executor
                  CurriculumLlm = Some(teacher :> Tars.Llm.ILlmService)
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
                      Verbose = false
                      ShowSemanticMessage = fun _ _ -> ()
                      Focus = None
                      ToolRegistry = None
                      ResearchEnhanced = false
                      RunCode = false
                      RecordTask = ignore
                      SelfImprovement = false } }

            let state: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = curriculumAgentId
                  ExecutorAgentId = executorAgentId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            let! nextState = Engine.step ctx state

            Assert.Equal("Write `isEven : int -> bool` in F#", nextState.CompletedTasks.Head.TaskGoal)
            Assert.True(executor.Requests > 0, "the executor's model answered nothing")
        }

    [<Fact>]
    let ``A local teacher answers the reasoning requests of the curriculum and the judge`` () =
        // Every other route configured: a reasoning model, a GGUF model (which takes every local
        // route first), Docker Model Runner and llama.cpp (which take their own hints).
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                ReasoningModel = Some "deepseek-r1:8b"
                LlamaSharpModelPath = Some "model.gguf"
                DockerModelRunnerBaseUri = Some(Uri "http://localhost:12434")
                DefaultDockerModelRunnerModel = Some "ai/smollm2"
                LlamaCppBaseUri = Some(Uri "http://localhost:8080")
                DefaultLlamaCppModel = Some "llama.gguf"
                PreferredProvider = "Ollama" }

        let pinned = global.Tars.Interface.Cli.LlmFactory.pinnedTo "qwen3:14b" cfg

        for hint in [ "reasoning"; "coding"; "fast"; "cheap"; "docker"; "llamacpp"; "" ] do
            let routed =
                Tars.Llm.Routing.chooseBackend pinned { Tars.Llm.LlmRequest.Default with ModelHint = Some hint }

            Assert.True((routed.Backend = Tars.Llm.LlmBackend.Ollama "qwen3:14b"), $"hint '{hint}' went to {routed.Backend}")

    [<Fact>]
    let ``An API teacher is named with its provider`` () =
        let route =
            global.Tars.Interface.Cli.LlmFactory.apiRoute (fun _ -> None) Tars.Llm.Routing.RoutingConfig.Default
        let backendOf name = route name |> Option.map (fun r -> r.Backend)

        Assert.True((backendOf "openai:o3" = Some(Tars.Llm.LlmBackend.OpenAI "o3")))
        Assert.True((backendOf "gemini:gemini-2.5-pro" = Some(Tars.Llm.LlmBackend.GoogleGemini "gemini-2.5-pro")))
        Assert.True((backendOf "anthropic:claude-sonnet-5-5" = Some(Tars.Llm.LlmBackend.Anthropic "claude-sonnet-5-5")))

        // Other names stay local, even an Ollama model named like an API one; `claude:` is Claude Code.
        for name in [ "gpt-oss:20b"; "qwen3:14b"; "claude:sonnet"; "openai:" ] do
            Assert.True((route name).IsNone, $"{name} went to an API")

    [<Fact>]
    let ``An API teacher only gets its own provider's key`` () =
        // RoutingConfig.fromTarsConfig copies Llm:ApiKey, which is OPENAI_API_KEY, into every slot.
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                OpenAIKey = Some "openai-key"
                GoogleGeminiKey = Some "openai-key"
                AnthropicKey = Some "openai-key" }

        let secret =
            function
            | "OPENAI_API_KEY" -> Some "openai-key"
            | _ -> None

        let keyOf name =
            (global.Tars.Interface.Cli.LlmFactory.apiRoute secret cfg name).Value.ApiKey

        Assert.True((keyOf "openai:gpt-4o" = Some "openai-key"))
        Assert.True((keyOf "anthropic:claude-sonnet-5-5" = None), "the OpenAI key went to Anthropic")
        Assert.True((keyOf "gemini:gemini-2.5-pro" = None), "the OpenAI key went to Gemini")

    /// A local HTTP server standing in for a provider's API. It answers `response` and records
    /// each request as (path, Authorization header, x-goog-api-key header, body).
    let fakeProvider (response: string) =
        let port =
            let probe = new Net.Sockets.TcpListener(Net.IPAddress.Loopback, 0)
            probe.Start()
            let port = (probe.LocalEndpoint :?> Net.IPEndPoint).Port
            probe.Stop()
            port

        let baseUri = Uri($"http://localhost:{port}/")
        let listener = new Net.HttpListener()
        listener.Prefixes.Add(string baseUri)
        listener.Start()
        let received = Collections.Concurrent.ConcurrentQueue<string * string * string * string>()

        task {
            while listener.IsListening do
                try
                    let! context = listener.GetContextAsync()
                    use reader = new IO.StreamReader(context.Request.InputStream)
                    let! body = reader.ReadToEndAsync()
                    let header (name: string) = context.Request.Headers.[name]
                    received.Enqueue((context.Request.Url.AbsolutePath, header "Authorization", header "x-goog-api-key", body))
                    let bytes = Text.Encoding.UTF8.GetBytes response
                    context.Response.ContentType <- "application/json"
                    context.Response.OutputStream.Write(bytes, 0, bytes.Length)
                    context.Response.Close()
                with _ ->
                    ()
        }
        |> ignore

        baseUri, received, listener

    let private userSays (text: string) : Tars.Llm.LlmMessage list =
        [ { Role = Tars.Llm.Role.User; Content = text } ]

    [<Fact>]
    let ``An API teacher answers every request on its provider, with its key`` () =
        let baseUri, received, listener =
            fakeProvider """{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"""

        use listener = listener

        // A reasoning model is configured too: the teacher's hints must not reach it.
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                OpenAIBaseUri = baseUri
                ReasoningModel = Some "deepseek-r1:8b" }

        let route =
            (global.Tars.Interface.Cli.LlmFactory.apiRoute (fun _ -> Some "test-key") cfg "openai:gpt-4.1").Value
        let teacher = global.Tars.Interface.Cli.LlmFactory.onRoute cfg route

        for hint in [ "reasoning"; "coding"; "" ] do
            let request =
                { Tars.Llm.LlmRequest.Default with
                    ModelHint = Some hint
                    Messages = userSays "hi" }

            Assert.Equal("ok", teacher.CompleteAsync(request).Result.Text)

        Assert.Equal(3, received.Count)

        for path, authorization, _, body in received do
            Assert.Equal("/v1/chat/completions", path)
            Assert.Equal("Bearer test-key", authorization)
            Assert.Contains("\"gpt-4.1\"", body)

    [<Fact>]
    let ``A Gemini teacher gets the judge's instructions, and JSON mode for its schema, with a warning`` () =
        let baseUri, received, listener =
            fakeProvider
                """{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP","index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"thoughtsTokenCount":200,"totalTokenCount":215}}"""

        use listener = listener

        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                GoogleGeminiBaseUri = baseUri }

        let route =
            (global.Tars.Interface.Cli.LlmFactory.apiRoute (fun _ -> Some "test-key") cfg "gemini:gemini-2.5-pro").Value
        let teacher = global.Tars.Interface.Cli.LlmFactory.onRoute cfg route

        // The judge's request (Evaluation.fs): a system prompt and a strict JSON schema.
        let request =
            { Tars.Llm.LlmRequest.Default with
                ModelHint = Some "reasoning"
                SystemPrompt = Some "Evaluate task output for semantic correctness."
                Messages = userSays "hi"
                Temperature = Some 0.0
                ResponseFormat =
                    Some(Tars.Llm.ResponseFormat.Constrained(Tars.Llm.Grammar.JsonSchema EvolutionSchemas.evaluationSchema))
                JsonMode = true }

        let warnings = Collections.Generic.List<string>()
        Tars.Llm.Routing.ConstraintDowngradeLog.setSink warnings.Add

        let response =
            try
                teacher.CompleteAsync(request).Result
            finally
                Tars.Llm.Routing.ConstraintDowngradeLog.resetSink ()

        Assert.Equal("ok", response.Text)

        // Its thinking tokens are billed as output, as its answer's are.
        let usage = response.Usage.Value
        Assert.Equal(10, usage.PromptTokens)
        Assert.Equal(205, usage.CompletionTokens)

        // The schema is given up, so it is reported, as on every other route.
        Assert.Contains(warnings, fun w -> w.Contains "json_schema grammar discarded — backend GoogleGemini")

        let path, _, key, body = Seq.exactlyOne received
        Assert.Equal("/v1beta/models/gemini-2.5-pro:generateContent", path)
        Assert.Equal("test-key", key)

        // The body as Gemini reads it: F# options must be plain values, not {"value": ...}.
        use json = Text.Json.JsonDocument.Parse body
        let root = json.RootElement

        Assert.Equal(
            "Evaluate task output for semantic correctness.",
            root.GetProperty("systemInstruction").GetProperty("parts").[0].GetProperty("text").GetString()
        )

        let generationConfig = root.GetProperty "generationConfig"
        Assert.Equal("application/json", generationConfig.GetProperty("responseMimeType").GetString())
        Assert.Equal(0.0, generationConfig.GetProperty("temperature").GetDouble())
        // Gemini's response schema has no additionalProperties, which every TARS schema carries.
        Assert.False(fst (generationConfig.TryGetProperty "responseSchema"))
        Assert.DoesNotContain("additionalProperties", body)

    [<Fact>]
    let ``OpenAI's reasoning models are refused, its chat models are not`` () =
        let unsupported = global.Tars.Interface.Cli.LlmFactory.unsupportedApiModel

        for name in [ "openai:o3"; "openai:o4-mini"; "openai:o1"; "openai:gpt-5"; "openai:gpt-5-mini" ] do
            Assert.True((unsupported name).IsSome, name)

        for name in
            [ "openai:gpt-4.1"
              "openai:gpt-4o"
              "anthropic:claude-sonnet-5-5"
              "gemini:gemini-2.5-pro"
              "gpt-oss:20b"
              "o3" ] do
            Assert.True((unsupported name).IsNone, name)

    [<Fact>]
    let ``A paid call reserves the schema and the tools it sends too`` () =
        let sent = ref 0

        let llm =
            { new Tars.Llm.ILlmService with
                member _.CompleteAsync _ =
                    sent.Value <- sent.Value + 1

                    let usage: Tars.Llm.TokenUsage =
                        { PromptTokens = 500
                          CompletionTokens = 0
                          TotalTokens = 500 }

                    let response: Tars.Llm.LlmResponse =
                        { Text = "{}"
                          FinishReason = None
                          Usage = Some usage
                          Raw = None }

                    Task.FromResult response

                member _.CompleteStreamAsync(_, _) = failwith "not used"
                member _.EmbedAsync _ = Task.FromResult [||]
                member _.RouteAsync _ = failwith "not used" }

        // 1 USD per input token, output free: 200 USD pays for 200 input tokens. The messages are
        // 2 bytes, but the provider also bills the schema (about 400 bytes) or the tool (over 300).
        let budget = BudgetGovernor({ Budget.Infinite with MaxMoney = Some 200m<usd> })
        let paid = global.Tars.Interface.Cli.LlmFactory.charged budget (1_000_000m, 0m) llm

        let withSchema =
            { Tars.Llm.LlmRequest.Default with
                MaxTokens = Some 100
                Messages = userSays "hi"
                ResponseFormat =
                    Some(Tars.Llm.ResponseFormat.Constrained(Tars.Llm.Grammar.JsonSchema EvolutionSchemas.evaluationSchema)) }

        let withTool =
            { Tars.Llm.LlmRequest.Default with
                MaxTokens = Some 100
                Messages = userSays "hi"
                Tools = [ box {| name = "read_code"; description = String.replicate 300 "x" |} ] }

        for request in [ withSchema; withTool ] do
            Assert.ThrowsAny<exn>(Action(fun () -> paid.CompleteAsync(request).Result |> ignore))
            |> ignore

        Assert.Equal(0, sent.Value)
        Assert.True((budget.Consumed.Money = 0m<usd>), $"charged {budget.Consumed.Money}")

    [<Fact>]
    let ``A price is USD per million input and output tokens`` () =
        let parse = global.Tars.Interface.Cli.LlmFactory.parsePrice

        Assert.True((parse "2.5/10" = Some(2.5m, 10m)))

        for text in [ "2,5/10"; "2.5"; "1/2/3"; "-1/2"; "a/b" ] do
            Assert.True((parse text).IsNone, text)

    [<Fact>]
    let ``A paid model's calls are charged to the budget, which they never exceed`` () =
        let sent = Collections.Concurrent.ConcurrentQueue<Tars.Llm.LlmRequest>()

        let llm =
            { new Tars.Llm.ILlmService with
                member _.CompleteAsync request =
                    sent.Enqueue request

                    let usage: Tars.Llm.TokenUsage =
                        { PromptTokens = 10
                          CompletionTokens = 1_000_000
                          TotalTokens = 1_000_010 }

                    let response: Tars.Llm.LlmResponse =
                        { Text = "ok"
                          FinishReason = None
                          Usage = Some usage
                          Raw = None }

                    Task.FromResult response

                member _.CompleteStreamAsync(_, _) = failwith "not used"
                member _.EmbedAsync _ = Task.FromResult [||]
                member _.RouteAsync _ = failwith "not used" }

        // At 2 USD per million input tokens and 8 per million output tokens, a call that may write
        // 1M tokens may cost about 8 USD, and this one does (10 input tokens, 1M output tokens).
        let budget = BudgetGovernor({ Budget.Infinite with MaxMoney = Some 20m<usd> })
        let paid = global.Tars.Interface.Cli.LlmFactory.charged budget (2m, 8m) llm

        let request =
            { Tars.Llm.LlmRequest.Default with
                MaxTokens = Some 1_000_000
                Messages = userSays "hi" }

        paid.CompleteAsync(request).Result |> ignore
        paid.CompleteAsync(request).Result |> ignore

        Assert.True((budget.Consumed.Money = 16.00004m<usd>), $"charged {budget.Consumed.Money}")

        // The 4 USD left cannot pay for a call that may cost 8: it is refused before it is sent.
        let refused =
            Assert.ThrowsAny<exn>(Action(fun () -> paid.CompleteAsync(request).Result |> ignore))

        Assert.Contains("budget is spent", refused.ToString())
        Assert.Equal(2, sent.Count)
        Assert.True((budget.Consumed.Money = 16.00004m<usd>), $"charged {budget.Consumed.Money}")

        // A request without an output limit gets one, so its cost has a worst case.
        let roomy = BudgetGovernor({ Budget.Infinite with MaxMoney = Some 100m<usd> })

        (global.Tars.Interface.Cli.LlmFactory.charged roomy (2m, 8m) llm)
            .CompleteAsync(Tars.Llm.LlmRequest.Default)
            .Result
        |> ignore

        Assert.Equal(Some 4096, (Seq.last sent).MaxTokens)

        // A call the provider refused (4xx) gives its reservation back. One that failed on the
        // way back (a timeout, a dropped connection) keeps it: the provider may have billed it.
        let chargedAfter (error: exn) =
            let failing =
                { new Tars.Llm.ILlmService with
                    member _.CompleteAsync _ =
                        Task.FromException<Tars.Llm.LlmResponse> error

                    member _.CompleteStreamAsync(_, _) = failwith "not used"
                    member _.EmbedAsync _ = Task.FromResult [||]
                    member _.RouteAsync _ = failwith "not used" }

            let failed = BudgetGovernor({ Budget.Infinite with MaxMoney = Some 100m<usd> })
            let failingPaid = global.Tars.Interface.Cli.LlmFactory.charged failed (2m, 8m) failing

            Assert.ThrowsAny<exn>(Action(fun () -> failingPaid.CompleteAsync(request).Result |> ignore))
            |> ignore

            failed.Consumed.Money

        let refused =
            chargedAfter (Net.Http.HttpRequestException("Bad Request", null, Nullable Net.HttpStatusCode.BadRequest))

        Assert.True((refused = 0m<usd>), $"kept {refused}")

        // The reservation: 64 + 2 + 8 input tokens at 2 USD per million, 1M output tokens at 8.
        let timedOut = chargedAfter (TaskCanceledException "timed out")
        Assert.True((timedOut = 8.000148m<usd>), $"kept {timedOut}")

        // A response without usage keeps its whole reservation: nothing else bounds what it cost.
        let withoutUsage =
            { new Tars.Llm.ILlmService with
                member _.CompleteAsync _ =
                    let response: Tars.Llm.LlmResponse =
                        { Text = "ok"
                          FinishReason = None
                          Usage = None
                          Raw = None }

                    Task.FromResult response

                member _.CompleteStreamAsync(_, _) = failwith "not used"
                member _.EmbedAsync _ = Task.FromResult [||]
                member _.RouteAsync _ = failwith "not used" }

        let kept = BudgetGovernor({ Budget.Infinite with MaxMoney = Some 2000m<usd> })

        (global.Tars.Interface.Cli.LlmFactory.charged kept (1_000_000m, 1_000_000m) withoutUsage)
            .CompleteAsync(
                { Tars.Llm.LlmRequest.Default with
                    MaxTokens = Some 1000
                    Messages = userSays "hi" }
            )
            .Result
        |> ignore

        // At 1 USD a token: 64 + 2 + 8 input tokens and 1000 output tokens.
        Assert.True((kept.Consumed.Money = 1074m<usd>), $"kept {kept.Consumed.Money}")

    [<Fact>]
    let ``The llama.cpp server gets every request for its configured model, and none for another .gguf`` () =
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                LlamaCppBaseUri = Some(Uri "http://localhost:8080")
                DefaultLlamaCppModel = Some "served-model"
                PreferredProvider = "Ollama" }

        // The server answers with the model it loaded, whatever a request names. Identifiers may be
        // paths, so `Served-Model` is another model.
        for model, onServer in [ "served-model", true; "other.gguf", false; "Served-Model", false ] do
            let pinned = global.Tars.Interface.Cli.LlmFactory.pinnedTo model cfg
            let pin = global.Tars.Interface.Cli.LlmFactory.pinnedRequest model cfg

            // The executor's requests carry its model's name as their hint; the curriculum's and
            // the judge's, "reasoning".
            for hint in [ model; "reasoning"; "fast"; "" ] do
                let routed =
                    Tars.Llm.Routing.chooseBackend pinned (pin { Tars.Llm.LlmRequest.Default with ModelHint = Some hint })

                let expected =
                    match routed.Backend with
                    | Tars.Llm.LlmBackend.LlamaCpp(m, _) ->
                        onServer && m = model && routed.Endpoint = Uri "http://localhost:8080"
                    | backend -> not onServer && backend = Tars.Llm.LlmBackend.Ollama model

                Assert.True(expected, $"'{model}' with hint '{hint}' went to {routed.Backend}")

    [<Fact>]
    let ``The configured LlamaSharp model gets every request when it is the pinned model`` () =
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                LlamaSharpModelPath = Some "/models/A.gguf"
                PreferredProvider = "Ollama" }

        // On a case-sensitive file system, `/models/a.gguf` is another file.
        for model, onLlamaSharp in [ "/models/A.gguf", true; "/models/a.gguf", false ] do
            let pinned = global.Tars.Interface.Cli.LlmFactory.pinnedTo model cfg
            let pin = global.Tars.Interface.Cli.LlmFactory.pinnedRequest model cfg

            let expected =
                if onLlamaSharp then
                    Tars.Llm.LlmBackend.LlamaSharp model
                else
                    Tars.Llm.LlmBackend.Ollama model

            for hint in [ model; "reasoning"; "fast"; "" ] do
                let routed =
                    Tars.Llm.Routing.chooseBackend pinned (pin { Tars.Llm.LlmRequest.Default with ModelHint = Some hint })

                Assert.True((routed.Backend = expected), $"'{model}' with hint '{hint}' went to {routed.Backend}")

    [<Fact>]
    let ``A local --model answers the executor's requests, not the configured CodingModel`` () =
        // The executor's requests carry its model's name as their hint; the curriculum's and the
        // judge's, "reasoning".
        let llm = global.Tars.Interface.Cli.Commands.Evolve.llmFor Serilog.Log.Logger "qwen2.5-coder:14b"

        for hint in [ "qwen2.5-coder:14b"; "reasoning" ] do
            let routed = llm.RouteAsync({ Tars.Llm.LlmRequest.Default with ModelHint = Some hint }).Result

            // Whichever local backend the configuration has: llama.cpp or LlamaSharp too, when it
            // serves this model.
            let model =
                match routed.Backend with
                | Tars.Llm.LlmBackend.Ollama m
                | Tars.Llm.LlmBackend.Vllm m
                | Tars.Llm.LlmBackend.LlamaCpp(m, _)
                | Tars.Llm.LlmBackend.LlamaSharp m -> m
                | other -> string other

            Assert.True((model = "qwen2.5-coder:14b"), $"hint '{hint}' went to {routed.Backend}")

    [<Fact>]
    let ``Evolve keeps one JSON line per task, with its examples' verdict apart from the evaluator's`` () =
        let dir = IO.Path.Combine(IO.Path.GetTempPath(), Guid.NewGuid().ToString "N")
        let path = IO.Path.Combine(dir, "results.jsonl")

        let verdict passed summary : EvaluationResult =
            { Passed = passed
              Confidence = 1.0
              Summary = summary
              Issues = []
              SuggestedFixes = []
              EvaluatedAt = DateTime.UtcNow }

        let taskDef: TaskDefinition =
            { Id = Guid.NewGuid()
              DifficultyLevel = 1
              Goal = "Write `fact : int -> int` in F#"
              Constraints = [ "Use recursion"; "No mutable"; "No List.rev"; "No loops" ]
              ValidationCriteria = "fact 5 = 120"
              Timeout = TimeSpan.FromMinutes 2.0
              Score = 1.0 }

        let rejected: TaskResult =
            { TaskId = taskDef.Id
              TaskGoal = taskDef.Goal
              ExecutorId = AgentId(Guid.NewGuid())
              Success = false
              Output = "```fsharp\nlet rec fact n = if n <= 1 then 1 else n * fact (n - 1)\n```"
              ExecutionTrace = []
              Duration = TimeSpan.FromSeconds 3.0
              Evaluation = Some(verdict false "Uses a loop.") }

        let entry result examples : Engine.TaskRecord =
            { Definition = taskDef
              Result = result
              Examples = examples }

        try
            let record =
                global.Tars.Interface.Cli.Commands.Evolve.recordTo Serilog.Log.Logger path "run1" "qwen2.5-coder:14b"

            // Rejected although its examples passed; then accepted with no examples run.
            record (entry rejected (Some(verdict true "The examples passed.")))
            record (entry { rejected with Success = true; Evaluation = None } None)

            let lines =
                IO.File.ReadAllLines path
                |> Array.map (fun line -> Text.Json.JsonDocument.Parse(line).RootElement)

            Assert.Equal(2, lines.Length)

            for line in lines do
                let text (name: string) = line.GetProperty(name).GetString()
                Assert.Equal("run1", text "run")
                Assert.Equal("qwen2.5-coder:14b", text "model")
                Assert.Equal(string taskDef.Id, text "taskId")
                Assert.Equal(taskDef.Goal, text "goal")

                Assert.Equal<string>(
                    taskDef.Constraints,
                    [ for c in line.GetProperty("constraints").EnumerateArray() -> c.GetString() ]
                )

                Assert.Equal("fact 5 = 120", text "criteria")
                Assert.Equal(rejected.Output, text "output")

            Assert.True(lines.[0].GetProperty("examplesPassed").GetBoolean())
            Assert.True(lines.[0].GetProperty("examplesComparedOutside").GetBoolean())
            Assert.False(lines.[0].GetProperty("success").GetBoolean())
            Assert.Equal("Uses a loop.", lines.[0].GetProperty("verdict").GetString())
            Assert.Equal(Text.Json.JsonValueKind.Null, lines.[1].GetProperty("examplesPassed").ValueKind)
            Assert.Equal(Text.Json.JsonValueKind.Null, lines.[1].GetProperty("examplesComparedOutside").ValueKind)
            Assert.True(lines.[1].GetProperty("success").GetBoolean())
            Assert.Equal(Text.Json.JsonValueKind.Null, lines.[1].GetProperty("verdict").ValueKind)
        finally
            if IO.Directory.Exists dir then
                IO.Directory.Delete(dir, true)

    /// A file under v2/, found from the test directory.
    let private v2File (parts: string list) =
        let rec v2 (dir: IO.DirectoryInfo) =
            if isNull dir then failwith "Could not locate v2/ (no Tars.sln above the test directory)"
            elif dir.GetFiles("Tars.sln").Length > 0 then dir
            else v2 dir.Parent

        IO.Path.Combine(Array.ofList ((v2 (IO.DirectoryInfo AppContext.BaseDirectory)).FullName :: parts))

    let private cliSettings () =
        Text.Json.JsonDocument.Parse(IO.File.ReadAllText(v2File [ "src"; "Tars.Interface.Cli"; "appsettings.json" ]))

    [<Fact>]
    let ``The CLI's default model answers its own executor's requests`` () =
        // The executor's requests carry its model's name as their hint, and a coder model's name
        // routes to the CodingModel: appsettings.json must not leave that on another model.
        use settings = cliSettings ()
        let llm = settings.RootElement.GetProperty "Llm"

        let setting (name: string) =
            match llm.TryGetProperty name with
            | true, value -> Some(value.GetString())
            | _ -> None

        let defaults = Tars.Core.ConfigurationDefaults.createDefault ()
        let model = (setting "Model").Value

        let config =
            { defaults with
                Llm =
                    { defaults.Llm with
                        Model = model
                        CodingModel = setting "CodingModel" |> Option.orElse defaults.Llm.CodingModel } }

        let routed =
            Tars.Llm.Routing.chooseBackend
                (Tars.Llm.Routing.RoutingConfig.fromTarsConfig config)
                { Tars.Llm.LlmRequest.Default with ModelHint = Some model }

        Assert.True((routed.Backend = Tars.Llm.LlmBackend.Ollama model), $"'{model}' went to {routed.Backend}")

    [<Fact>]
    let ``The setup script pulls the CLI's default model`` () =
        use settings = cliSettings ()
        let model = settings.RootElement.GetProperty("Llm").GetProperty("Model").GetString()
        let setup = IO.File.ReadAllText(v2File [ "scripts"; "setup-tars.ps1" ])

        Assert.Contains($"\"{model}\"", setup)

    [<Fact>]
    let ``With --trace, the teacher's calls are traced like the executor's`` () =
        let llm = SuccessLlm("ok") :> Tars.Llm.ILlmService
        let recorder = TraceRecorder()

        let executor, teacher =
            global.Tars.Interface.Cli.Commands.Evolve.tracedServices true recorder llm (Some llm)

        Assert.IsType<Tars.Llm.TracingLlmService>(executor) |> ignore
        Assert.IsType<Tars.Llm.TracingLlmService>(teacher.Value) |> ignore

        // Each call says which of the two answered it.
        executor.CompleteAsync(Tars.Llm.LlmRequest.Default).Result |> ignore
        teacher.Value.CompleteAsync(Tars.Llm.LlmRequest.Default).Result |> ignore

        let roles () =
            ((recorder :> ITraceRecorder).GetTraceAsync() |> Async.RunSynchronously).Value.Events
            |> List.choose (fun e -> e.Metadata.TryFind "role")
            |> List.sort

        // The recording is fire-and-forget.
        let watch = Diagnostics.Stopwatch.StartNew()

        while roles().Length < 2 && watch.Elapsed < TimeSpan.FromSeconds 5.0 do
            Threading.Thread.Sleep 20

        Assert.Equal<string>([ "executor"; "teacher" ], roles ())

    [<Fact>]
    let ``An evolve run counts its Claude calls by role, and not a local model's`` () =
        let calls = global.Tars.Interface.Cli.Commands.Evolve.ClaudeCalls()
        let llm = SuccessLlm("ok") :> Tars.Llm.ILlmService
        let curriculum = calls.Count "curriculum" (Some "claude:sonnet") llm
        let judge = calls.Count "judge" (Some "claude:sonnet") llm
        let governor = calls.Count "governor" (Some "claude:sonnet") llm
        let executor = calls.Count "executor" (Some "qwen2.5-coder:14b") llm
        let request = Tars.Llm.LlmRequest.Default

        curriculum.CompleteAsync(request).Result |> ignore
        curriculum.CompleteStreamAsync(request, ignore).Result |> ignore

        (judge :?> Tars.Llm.ICancellableLlmService).CompleteAsync(request, Threading.CancellationToken.None).Result
        |> ignore

        executor.CompleteAsync(request).Result |> ignore
        // An embedding and a route are not calls to Claude.
        curriculum.EmbedAsync("text").Result |> ignore
        curriculum.RouteAsync(request).Result |> ignore

        // A Claude role that made no call is still listed, with 0.
        Assert.Equal<string * int>([ "curriculum", 2; "governor", 0; "judge", 1 ], calls.ByRole)
        Assert.Equal(3, calls.Total)

    [<Fact>]
    let ``An evolve run's Claude calls are reported against ADR 0005's budget`` () =
        let calls = global.Tars.Interface.Cli.Commands.Evolve.ClaudeCalls()
        let llm = SuccessLlm("ok") :> Tars.Llm.ILlmService
        let curriculum = calls.Count "curriculum" (Some "claude:sonnet") llm
        let judge = calls.Count "judge" (Some "claude:sonnet") llm

        for _ in 1..2 do
            curriculum.CompleteAsync(Tars.Llm.LlmRequest.Default).Result |> ignore

        judge.CompleteAsync(Tars.Llm.LlmRequest.Default).Result |> ignore

        let report = global.Tars.Interface.Cli.Commands.Evolve.claudeCallsReport calls

        // D3: about 30 calls per 10-task run, so 3 per task the run finished. A task that
        // failed is finished too: evolve records it.
        Assert.Equal(Some(false, "Claude calls: 3 (curriculum 2, judge 1), within ADR 0005's budget of 3 for 1 task finished"), report 1)
        Assert.Equal(Some(false, "Claude calls: 3 (curriculum 2, judge 1), within ADR 0005's budget of 30 for 10 tasks finished"), report 10)
        // A run whose curriculum gave no task spent its calls on nothing: no budget for them.
        Assert.Equal(Some(true, "Claude calls: 3 (curriculum 2, judge 1), over ADR 0005's budget of 0 for 0 tasks finished"), report 0)

        // A run with no Claude role has nothing to report.
        Assert.Equal(None, global.Tars.Interface.Cli.Commands.Evolve.claudeCallsReport (global.Tars.Interface.Cli.Commands.Evolve.ClaudeCalls()) 10)

    [<Fact>]
    let ``With --trace, each call's request is recorded`` () =
        let recorder = TraceRecorder()

        let llm =
            Tars.Llm.TracingLlmService(SuccessLlm("ok"), recorder, "teacher") :> Tars.Llm.ILlmService

        let request =
            { Tars.Llm.LlmRequest.Default with
                SystemPrompt = Some "Judge the code."
                Messages = [ { Role = Tars.Llm.Role.User; Content = "let x = 1" } ]
                ResponseFormat = Some Tars.Llm.ResponseFormat.Json }

        llm.CompleteAsync(request).Result |> ignore

        (llm :?> Tars.Llm.ICancellableLlmService).CompleteAsync(request, Threading.CancellationToken.None).Result
        |> ignore

        let inputs () =
            ((recorder :> ITraceRecorder).GetTraceAsync() |> Async.RunSynchronously).Value.Events
            |> List.map (fun e -> e.Input)

        // The recording is fire-and-forget.
        let watch = Diagnostics.Stopwatch.StartNew()

        while inputs().Length < 2 && watch.Elapsed < TimeSpan.FromSeconds 5.0 do
            Threading.Thread.Sleep 20

        Assert.Equal(2, inputs().Length)

        for input in inputs () do
            // The roles and the response format are F# unions, which System.Text.Json cannot write.
            Assert.DoesNotContain("Serialization failed", input)
            use json = Text.Json.JsonDocument.Parse input
            Assert.Equal("Judge the code.", json.RootElement.GetProperty("SystemPrompt").GetString())
            let message = json.RootElement.GetProperty("Messages").[0]
            Assert.Equal("User", message.GetProperty("Role").GetString())
            Assert.Equal("let x = 1", message.GetProperty("Content").GetString())
            Assert.Equal("Json", json.RootElement.GetProperty("ResponseFormat").GetString())

    [<Fact>]
    let ``Evolution loop validates speech acts in response`` () =
        task {
            if not (TestHelpers.requireTools()) then () else
            let curriculumAgentId = AgentId(Guid.NewGuid())
            let executorAgentId = AgentId(Guid.NewGuid())

            // Curriculum Agent returns tasks in JSON, but prefixed with ACT: INFORM:
            let curriculumLlm =
                SuccessLlm(
                    "ACT: INFORM: {\"tasks\":[{\"goal\":\"Task 1\",\"constraints\":[],\"validation_criteria\":\"ok\"}, {\"goal\":\"Task 2\",\"constraints\":[],\"validation_criteria\":\"ok\"}]}"
                )

            // Custom Registry to return LLM based on agent ID if needed,
            // but for this simple test we'll just use the context's LLM which will be SuccessLlm.

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

            let registry = MockRegistry([ agent1; agent2 ])

            let ctx: Engine.EvolutionContext =
                { Registry = registry
                  Llm = curriculumLlm
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

            let state: EvolutionState =
                { Generation = 0
                  CurriculumAgentId = curriculumAgentId
                  ExecutorAgentId = executorAgentId
                  CompletedTasks = []
                  CurrentTask = None
                  TaskQueue = []
                  ActiveBeliefs = [] }

            // Run one step (Curriculum Phase)
            let! nextState = Engine.step ctx state

            // Verify tasks were generated despite the ACT: prefix
            Assert.NotEmpty(nextState.TaskQueue)
            Assert.Equal("Task 2", nextState.TaskQueue.Head.Goal)
            Assert.NotEmpty(nextState.CompletedTasks)
            Assert.Equal("Task 1", nextState.CompletedTasks.Head.TaskGoal)
        }
