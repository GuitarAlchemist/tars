module Tars.Tests.PatternTests

open System
open System.Threading
open Xunit
open Xunit.Abstractions
open Tars.Core
open Tars.Cortex
open Tars.Cortex.Patterns
open Tars.Llm
open Tars.Llm.LlmService
open System.Threading
open System.Threading.Tasks
open Tars.Core.Knowledge

type QueueLlm(responses: string list, recorder: System.Collections.Generic.List<string>) =
    let mutable queue = responses

    interface ILlmService with
        member _.CompleteAsync(req: LlmRequest) =
            task {
                let lastContent =
                    req.Messages
                    |> List.tryLast
                    |> Option.map (fun m -> m.Content)
                    |> Option.defaultValue ""

                recorder.Add(lastContent)

                match queue with
                | next :: rest ->
                    queue <- rest
                    return
                        { Text = next
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
                | [] ->
                    return
                        { Text = "Thought: done\nAction: Finish\nAction Input: finished"
                          FinishReason = Some "stop"
                          Usage = None
                          Raw = None }
            }

        member _.CompleteStreamAsync(_req, _onToken) = raise (NotImplementedException())
        member _.EmbedAsync(_text) = Task.FromResult(Array.empty<float32>)

type StubSemanticMemory(memories: MemorySchema list) =
    interface ISemanticMemory with
        member _.Retrieve(_query) = async { return memories }
        member _.Grow(_trace, _report) = async { return "id" }
        member _.Refine() = async { return () }

type StubGraphService(facts: TarsFact list) =
    interface IGraphService with
        member _.AddNodeAsync(_e) = Task.FromResult("")
        member _.AddFactAsync(_f) = Task.FromResult(Guid.NewGuid())
        member _.AddEpisodeAsync(_e) = Task.FromResult("")
        member _.QueryAsync(_q) = Task.FromResult(facts)
        member _.PersistAsync() = Task.FromResult(())

type StubToolRegistry() =
    let tools = System.Collections.Generic.Dictionary<string, Tool>()

    member _.Register(tool: Tool) = tools.[tool.Name] <- tool

    interface IToolRegistry with
        member _.Register(tool) = tools.[tool.Name] <- tool
        member _.Get(name) =
            match tools.TryGetValue name with
            | true, t -> Some t
            | _ -> None
        member _.GetAll() = tools.Values |> Seq.toList

/// Unit tests for the agentic patterns (ReAct, Chain of Thought, Plan & Execute)
type PatternTests(output: ITestOutputHelper) =

    // Create a mock AgentContext for testing
    let createMockContext () =
        let agent: Agent =
            { Id = AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "test-model"
              SystemPrompt = "You are a test agent."
              Tools = []
              Capabilities = []
              State = AgentState.Idle
              Memory = [] }

        let mockRegistry =
            { new IAgentRegistry with
                member _.GetAgent(_) = async { return None }
                member _.FindAgents(_) = async { return [] }
                member _.GetAllAgents() = async { return [] } }

        let mockExecutor =
            { new IAgentExecutor with
                member _.Execute(_, _) = async { return Success "mock result" } }

        { Self = agent
          Registry = mockRegistry
          Executor = mockExecutor
          Logger = fun msg -> output.WriteLine(msg)
          Budget = None
          Epistemic = None
          SemanticMemory = None
          KnowledgeGraph = None
          CapabilityStore = None
          CancellationToken = CancellationToken.None }

    // =========================================================================
    // Chain of Thought Tests
    // =========================================================================

    [<Fact>]
    member _.``ChainOfThought: Executes steps in sequence``() =
        async {
            let ctx = createMockContext ()

            // Create simple steps that append to the input
            let step1 input =
                fun _ -> async { return Success(input + " -> step1") }

            let step2 input =
                fun _ -> async { return Success(input + " -> step2") }

            let step3 input =
                fun _ -> async { return Success(input + " -> step3") }

            let workflow = chainOfThought [ step1; step2; step3 ] "start"
            let! result = workflow ctx

            match result with
            | Success value ->
                output.WriteLine(sprintf "Output: %s" value)
                Assert.Equal("start -> step1 -> step2 -> step3", value)
            | PartialSuccess(value, _) -> Assert.Equal("start -> step1 -> step2 -> step3", value)
            | Failure errors -> Assert.Fail(sprintf "Unexpected failure: %A" errors)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``ChainOfThought: Accumulates warnings``() =
        async {
            let ctx = createMockContext ()

            let step1 input =
                fun _ -> async { return Success(input + " -> step1") }

            let step2 input =
                fun _ ->
                    async {
                        return PartialSuccess(input + " -> step2", [ PartialFailure.Warning "Warning from step2" ])
                    }

            let step3 input =
                fun _ -> async { return Success(input + " -> step3") }

            let workflow = chainOfThought [ step1; step2; step3 ] "start"
            let! result = workflow ctx

            match result with
            | PartialSuccess(output, warnings) ->
                Assert.Equal("start -> step1 -> step2 -> step3", output)
                Assert.Single(warnings) |> ignore
            | Success _ -> Assert.Fail("Expected PartialSuccess with warnings")
            | Failure _ -> Assert.Fail("Unexpected failure")
        }
        |> Async.RunSynchronously

    // =========================================================================
    // ReAct Safety + Context Tests
    // =========================================================================

    [<Fact>]
    member _.``ReAct prepends semantic context and emits safety warning for risky tool``() =
        async {
            let calls = System.Collections.Generic.List<string>()

            let llm =
                QueueLlm(
                    [ "Thought: try\nAction: write_code\nAction Input: dangerous change"
                      "Thought: done\nAction: Finish\nAction Input: final answer" ],
                    calls
                )
                :> ILlmService

            let mem =
                { Id = "m1"
                  Logical =
                    Some
                        { ProblemSummary = "Fixed bug"
                          StrategySummary = "Used tests"
                          ErrorKinds = []
                          ErrorTags = []
                          OutcomeLabel = "success"
                          Score = None
                          CostTokens = None
                          Embedding = Array.empty
                          Tags = [] }
                  Perceptual = None
                  CreatedAt = DateTime.UtcNow
                  LastUsedAt = None
                  UsageCount = 0 }

            let semMem = StubSemanticMemory([ mem ]) :> ISemanticMemory

            let fact =
                TarsFact.DerivedFrom(TarsEntity.ConceptE { Name = "A"; Description = ""; RelatedConcepts = [] },
                                     TarsEntity.ConceptE { Name = "B"; Description = ""; RelatedConcepts = [] })

            let kg = StubGraphService([ fact ]) :> IGraphService

            let toolRegistry = StubToolRegistry()

            let writeTool: Tool =
                { Name = "write_code"
                  Description = "writes code"
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun _ -> async { return Result.Ok "ok" } }

            toolRegistry.Register(writeTool)

            let agent: Agent =
                { Id = AgentId(Guid.NewGuid())
                  Name = "TestAgent"
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Model = "test"
                  SystemPrompt = "test"
                  Tools = []
                  Capabilities = []
                  State = AgentState.Idle
                  Memory = [] }

            let ctx: AgentContext =
                { Self = agent
                  Registry =
                    { new IAgentRegistry with
                        member _.GetAgent _ = async { return Some agent }
                        member _.FindAgents _ = async { return [ agent ] }
                        member _.GetAllAgents() = async { return [ agent ] } }
                  Executor =
                    { new IAgentExecutor with
                        member _.Execute(_, _) = async { return Success "noop" } }
                  Logger = fun _ -> ()
                  Budget = None
                  Epistemic = None
                  SemanticMemory = Some semMem
                  KnowledgeGraph = Some kg
                  CapabilityStore = None
                  CancellationToken = CancellationToken.None }

            let workflow = reAct llm (toolRegistry :> IToolRegistry) 5 "Do the thing"
            let! result = workflow ctx

            // Assert semantic prelude was included in first LLM call
            Assert.True(calls.Count >= 1)
            let firstPrompt = calls |> Seq.head
            Assert.Contains("Lessons:", firstPrompt)
            Assert.Contains("Facts:", firstPrompt)

            match result with
            | PartialSuccess(answer, warnings) ->
                Assert.Equal("final answer", answer)
                Assert.True(warnings |> List.exists (fun w -> match w with | PartialFailure.Warning msg -> msg.Contains("SafetyGate") | _ -> false))
            | _ -> Assert.Fail("Expected PartialSuccess with safety warning")
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``ChainOfThought: Short-circuits on failure``() =
        async {
            let ctx = createMockContext ()

            let step1 input =
                fun _ -> async { return Success(input + " -> step1") }

            let step2 _ =
                fun _ -> async { return Failure [ PartialFailure.Error "Step 2 failed" ] }

            let step3 input =
                fun _ -> async { return Success(input + " -> step3") }

            let workflow = chainOfThought [ step1; step2; step3 ] "start"
            let! result = workflow ctx

            match result with
            | Failure errors ->
                Assert.Single(errors) |> ignore

                match errors.Head with
                | PartialFailure.Error msg -> Assert.Equal("Step 2 failed", msg)
                | _ -> Assert.Fail("Wrong error type")
            | _ -> Assert.Fail("Expected failure")
        }
        |> Async.RunSynchronously

    // =========================================================================
    // Plan & Execute Tests
    // =========================================================================

    [<Fact>]
    member _.``PlanAndExecute: Executes all plan steps``() =
        async {
            let ctx = createMockContext ()

            // Planner returns a fixed list of steps
            let planner: AgentWorkflow<string list> =
                fun _ -> async { return Success [ "Step 1"; "Step 2"; "Step 3" ] }

            // Executor echoes the step
            let executor step : AgentWorkflow<string> =
                fun _ -> async { return Success(sprintf "Executed: %s" step) }

            let workflow = planAndExecute planner executor
            let! result = workflow ctx

            match result with
            | Success results ->
                Assert.Equal(3, results.Length)
                Assert.Equal("Executed: Step 1", results.[0])
                Assert.Equal("Executed: Step 2", results.[1])
                Assert.Equal("Executed: Step 3", results.[2])
            | PartialSuccess(results, _) -> Assert.Equal(3, results.Length)
            | Failure errors -> Assert.Fail(sprintf "Unexpected failure: %A" errors)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``PlanAndExecute: Stops on executor failure``() =
        async {
            let ctx = createMockContext ()

            let planner: AgentWorkflow<string list> =
                fun _ -> async { return Success [ "Step 1"; "Step 2"; "Step 3" ] }

            let executor step : AgentWorkflow<string> =
                fun _ ->
                    async {
                        if step = "Step 2" then
                            return Failure [ PartialFailure.Error "Step 2 failed" ]
                        else
                            return Success(sprintf "Executed: %s" step)
                    }

            let workflow = planAndExecute planner executor
            let! result = workflow ctx

            match result with
            | Failure errors -> Assert.True(errors.Length >= 1)
            | _ -> Assert.Fail("Expected failure")
        }
        |> Async.RunSynchronously

    // =========================================================================
    // ReAct Tests (require mock LLM and tools)
    // =========================================================================

    [<Fact>]
    member _.``ReAct: Stops after max steps with no answer``() =
        async {
            let ctx = createMockContext ()

            // Mock LLM that always returns a malformed response
            let mockLlm =
                { new ILlmService with
                    member _.CompleteAsync(_) =
                        task {
                            return
                                { Text = "Just some random text"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(_, _) =
                        task {
                            return
                                { Text = "streaming"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.EmbedAsync(_) = task { return [| 0.1f |] } }

            // Empty tool registry
            let mockTools =
                { new IToolRegistry with
                    member _.Register(_) = ()
                    member _.Get(_) = None
                    member _.GetAll() = [] }

            let workflow = reAct mockLlm mockTools 3 "What is 2+2?"
            let! result = workflow ctx

            match result with
            | Failure errors ->
                output.WriteLine(sprintf "Errors: %A" errors)
                Assert.True(errors.Length >= 1, "Should have at least one error about max steps")
            | _ -> Assert.Fail("Expected failure after max steps")
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``ReAct: Finishes when LLM returns Finish action``() =
        async {
            let ctx = createMockContext ()

            // Mock LLM that returns a Finish action
            let mockLlm =
                { new ILlmService with
                    member _.CompleteAsync(_) =
                        task {
                            return
                                { Text = "Thought: I know the answer.\nAction: Finish\nAction Input: The answer is 4"
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.CompleteStreamAsync(_, _) =
                        task {
                            return
                                { Text = ""
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.EmbedAsync(_) = task { return [| 0.1f |] } }

            let mockTools =
                { new IToolRegistry with
                    member _.Register(_) = ()
                    member _.Get(_) = None
                    member _.GetAll() = [] }

            let workflow = reAct mockLlm mockTools 5 "What is 2+2?"
            let! result = workflow ctx

            match result with
            | Success answer ->
                output.WriteLine(sprintf "Answer: %s" answer)
                Assert.Equal("The answer is 4", answer)
            | PartialSuccess(answer, _) -> Assert.Equal("The answer is 4", answer)
            | Failure errors -> Assert.Fail(sprintf "Unexpected failure: %A" errors)
        }
        |> Async.RunSynchronously

    [<Fact>]
    member _.``ReAct: Can execute tools and observe results``() =
        async {
            let ctx = createMockContext ()
            let mutable callCount = 0

            // Mock LLM that first calls a tool, then finishes
            let mockLlm =
                { new ILlmService with
                    member _.CompleteAsync(_) =
                        task {
                            callCount <- callCount + 1

                            if callCount = 1 then
                                return
                                    { Text = "Thought: I need to calculate.\nAction: calculator\nAction Input: 2+2"
                                      FinishReason = Some "stop"
                                      Usage = None
                                      Raw = None }
                            else
                                return
                                    { Text = "Thought: The calculator says 4.\nAction: Finish\nAction Input: 4"
                                      FinishReason = Some "stop"
                                      Usage = None
                                      Raw = None }
                        }

                    member _.CompleteStreamAsync(_, _) =
                        task {
                            return
                                { Text = ""
                                  FinishReason = Some "stop"
                                  Usage = None
                                  Raw = None }
                        }

                    member _.EmbedAsync(_) = task { return [| 0.1f |] } }

            // Mock tool registry with a calculator tool
            let calculatorTool: Tool =
                { Name = "calculator"
                  Description = "Calculates math expressions"
                  Version = "1.0.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun _ -> async { return Result.Ok "4" } }

            let mockTools =
                { new IToolRegistry with
                    member _.Register(_) = ()

                    member _.Get(name) =
                        if name = "calculator" then Some calculatorTool else None

                    member _.GetAll() = [ calculatorTool ] }

            let workflow = reAct mockLlm mockTools 5 "What is 2+2?"
            let! result = workflow ctx

            match result with
            | Success answer ->
                output.WriteLine(sprintf "Answer: %s" answer)
                Assert.Equal("4", answer)
            | PartialSuccess(answer, _) -> Assert.Equal("4", answer)
            | Failure errors -> Assert.Fail(sprintf "Unexpected failure: %A" errors)
        }
        |> Async.RunSynchronously
