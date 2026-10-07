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
    let ``The llama.cpp server gets every request for its configured model, and none for another .gguf`` () =
        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                LlamaCppBaseUri = Some(Uri "http://localhost:8080")
                DefaultLlamaCppModel = Some "served-model"
                PreferredProvider = "Ollama" }

        // The server answers with the model it loaded, whatever a request names.
        for model, onServer in [ "served-model", true; "other.gguf", false ] do
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
        let model = "C:/models/tars.gguf"

        let cfg =
            { Tars.Llm.Routing.RoutingConfig.Default with
                LlamaSharpModelPath = Some model
                PreferredProvider = "Ollama" }

        let pinned = global.Tars.Interface.Cli.LlmFactory.pinnedTo model cfg
        let pin = global.Tars.Interface.Cli.LlmFactory.pinnedRequest model cfg

        for hint in [ model; "reasoning"; "fast"; "" ] do
            let routed =
                Tars.Llm.Routing.chooseBackend pinned (pin { Tars.Llm.LlmRequest.Default with ModelHint = Some hint })

            Assert.True((routed.Backend = Tars.Llm.LlmBackend.LlamaSharp model), $"hint '{hint}' went to {routed.Backend}")

    [<Fact>]
    let ``A local --model answers the executor's requests, not the configured CodingModel`` () =
        // The executor's requests carry its model's name as their hint; the curriculum's and the
        // judge's, "reasoning".
        let llm = global.Tars.Interface.Cli.Commands.Evolve.llmFor Serilog.Log.Logger "qwen2.5-coder:14b"

        for hint in [ "qwen2.5-coder:14b"; "reasoning" ] do
            let routed = llm.RouteAsync({ Tars.Llm.LlmRequest.Default with ModelHint = Some hint }).Result

            // Whichever local backend the configuration has: llama.cpp too, when it serves this model.
            let model =
                match routed.Backend with
                | Tars.Llm.LlmBackend.Ollama m
                | Tars.Llm.LlmBackend.Vllm m
                | Tars.Llm.LlmBackend.LlamaCpp(m, _) -> m
                | other -> string other

            Assert.True((model = "qwen2.5-coder:14b"), $"hint '{hint}' went to {routed.Backend}")

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
