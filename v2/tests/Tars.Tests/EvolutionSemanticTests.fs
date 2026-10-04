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
        let route = global.Tars.Interface.Cli.LlmFactory.apiRoute Tars.Llm.Routing.RoutingConfig.Default
        let backendOf name = route name |> Option.map (fun r -> r.Backend)

        Assert.True((backendOf "openai:o3" = Some(Tars.Llm.LlmBackend.OpenAI "o3")))
        Assert.True((backendOf "gemini:gemini-2.5-pro" = Some(Tars.Llm.LlmBackend.GoogleGemini "gemini-2.5-pro")))
        Assert.True((backendOf "anthropic:claude-sonnet-5-5" = Some(Tars.Llm.LlmBackend.Anthropic "claude-sonnet-5-5")))

        // Other names stay local, even an Ollama model named like an API one; `claude:` is Claude Code.
        for name in [ "gpt-oss:20b"; "qwen3:14b"; "claude:sonnet"; "openai:" ] do
            Assert.True((route name).IsNone, $"{name} went to an API")

    [<Fact>]
    let ``An API teacher answers every request on its provider, with its key`` () =
        let port =
            let probe = new Net.Sockets.TcpListener(Net.IPAddress.Loopback, 0)
            probe.Start()
            let port = (probe.LocalEndpoint :?> Net.IPEndPoint).Port
            probe.Stop()
            port

        let baseUri = Uri($"http://localhost:{port}/")
        use listener = new Net.HttpListener()
        listener.Prefixes.Add(string baseUri)
        listener.Start()
        let received = Collections.Concurrent.ConcurrentQueue<string>()

        let _server =
            task {
                while listener.IsListening do
                    try
                        let! context = listener.GetContextAsync()
                        use reader = new IO.StreamReader(context.Request.InputStream)
                        let! body = reader.ReadToEndAsync()
                        let authorization = context.Request.Headers.["Authorization"]
                        received.Enqueue($"{context.Request.Url.AbsolutePath} {authorization} {body}")

                        let bytes =
                            Text.Encoding.UTF8.GetBytes
                                """{"id":"1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}"""

                        context.Response.ContentType <- "application/json"
                        context.Response.OutputStream.Write(bytes, 0, bytes.Length)
                        context.Response.Close()
                    with _ ->
                        ()
            }

        try
            // A reasoning model is configured too: the teacher's hints must not reach it.
            let cfg =
                { Tars.Llm.Routing.RoutingConfig.Default with
                    OpenAIBaseUri = baseUri
                    OpenAIKey = Some "test-key"
                    ReasoningModel = Some "deepseek-r1:8b" }

            let route = (global.Tars.Interface.Cli.LlmFactory.apiRoute cfg "openai:o3").Value
            let teacher = global.Tars.Interface.Cli.LlmFactory.onRoute cfg route

            for hint in [ "reasoning"; "coding"; "" ] do
                let request =
                    { Tars.Llm.LlmRequest.Default with
                        ModelHint = Some hint
                        Messages = [ { Tars.Llm.LlmMessage.Role = Tars.Llm.Role.User; Content = "hi" } ] }

                Assert.Equal("ok", teacher.CompleteAsync(request).Result.Text)

            Assert.Equal(3, received.Count)

            for request in received do
                Assert.StartsWith("/v1/chat/completions Bearer test-key ", request)
                Assert.Contains("\"o3\"", request)
        finally
            listener.Stop()

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
