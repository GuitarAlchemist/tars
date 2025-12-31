namespace Tars.Tests

open System
open System.Net.Http
open System.Text.Json
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Llm.Routing
open Tars.Cortex
open Tars.Cortex.Patterns
open Tars.Tests.TestHelpers

type WotIntegrationTests(output: Xunit.Abstractions.ITestOutputHelper) =

    // Local resolver removed in favor of TestHelpers.resolveTestModel

    let createTestContext (log: string -> unit) (llm: ILlmService) =
        let logger msg = log msg

        let agent =
            { Id = AgentId(Guid.NewGuid())
              Name = "TestAgent"
              Version = "1.0"
              ParentVersion = None
              CreatedAt = DateTime.UtcNow
              Model = "mock"
              SystemPrompt = "System"
              Tools = []
              Capabilities = []
              State = AgentState.Idle
              Memory = []
              Fitness = 1.0
              Drives =
                { Accuracy = 1.0
                  Speed = 1.0
                  Creativity = 1.0
                  Safety = 1.0 }
              Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) }

        { Self = agent
          Registry = Unchecked.defaultof<_>
          Executor = Unchecked.defaultof<_>
          Logger = logger
          Budget = None
          Epistemic = None
          SemanticMemory = None
          KnowledgeGraph = None
          CapabilityStore = None
          Audit = None
          Critique = None
          Cycle = None
          CancellationToken = System.Threading.CancellationToken.None }

    [<Fact>]
    member this.``WorkflowOfThought solves 2+2 with real LLM``() =
        task {
            // Use real configuration from default values which points to local Ollama
            // but override the model to one that is known to exist in the environment
            let! model = TestHelpers.resolveTestModel ()

            let routingConfig =
                { RoutingConfig.Default with
                    DefaultOllamaModel = model
                    DefaultVllmModel = model // Just in case
                    ReasoningModel = Some model }

            let config = { Routing = routingConfig }

            // Create the REAL service
            let httpClient = new System.Net.Http.HttpClient()
            let llm = DefaultLlmService(httpClient, config) :> ILlmService

            // Configure WoT
            let wotConfig =
                { BaseConfig =
                    { BranchingFactor = 3
                      MaxDepth = 3 // Increase depth to allow for reasoning emergence
                      TopK = 2
                      ScoreThreshold = 0.3 // Lower threshold to accept LLM outputs more easily
                      MinConfidence = 0.3
                      DiversityThreshold = 0.9
                      DiversityPenalty = 0.1
                      Constraints = []
                      EnableCritique = false // Simplify for basic test
                      EnablePolicyChecks = false
                      EnableMemoryRecall = false
                      TrackEdges = true }
                  RequiredPolicies = []
                  AvailableTools = []
                  RoleAssignments = Map.empty
                  MemoryNamespace = None
                  MaxEscalations = 0
                  TimeoutMs = None }

            let goal = "Calculate 2+2"
            let workflow = Patterns.workflowOfThought llm wotConfig goal

            let ctx = createTestContext (fun msg -> output.WriteLine(msg)) llm

            // Execute
            let! result = workflow ctx

            match result with
            | Success answer ->
                output.WriteLine($"Result: {answer}")
                // Flexible assertion as LLM output varies
                Assert.True(answer.Contains("4"), $"Expected result to contain '4', but got: {answer}")
            | Failure errors ->
                output.WriteLine("Failed:")

                for e in errors do
                    output.WriteLine($"  {e}")

                Assert.Fail("Workflow returned failure")
            | PartialSuccess(ans, _) ->
                output.WriteLine($"Partial: {ans}")
                Assert.True(ans.Contains("4"), $"Expected partial result to contain '4', but got: {ans}")
        }

    [<Fact>]
    member this.``WorkflowOfThought answers capital of France with real LLM``() =
        task {
            let! model = TestHelpers.resolveTestModel ()

            let routingConfig =
                { RoutingConfig.Default with
                    DefaultOllamaModel = model
                    DefaultVllmModel = model
                    ReasoningModel = Some model }

            let config = { Routing = routingConfig }

            let httpClient = new System.Net.Http.HttpClient()
            let llm = DefaultLlmService(httpClient, config) :> ILlmService

            let wotConfig =
                { BaseConfig =
                    { BranchingFactor = 2
                      MaxDepth = 2
                      TopK = 1
                      ScoreThreshold = 0.3
                      MinConfidence = 0.3
                      DiversityThreshold = 0.9
                      DiversityPenalty = 0.1
                      Constraints = []
                      EnableCritique = false
                      EnablePolicyChecks = false
                      EnableMemoryRecall = false
                      TrackEdges = true }
                  RequiredPolicies = []
                  AvailableTools = []
                  RoleAssignments = Map.empty
                  MemoryNamespace = None
                  MaxEscalations = 0
                  TimeoutMs = None }

            let goal = "What is the capital of France?"
            let workflow = Patterns.workflowOfThought llm wotConfig goal

            let ctx = createTestContext (fun msg -> output.WriteLine(msg)) llm

            let! result = workflow ctx

            match result with
            | Success answer ->
                output.WriteLine($"Result: {answer}")

                Assert.True(
                    answer.Contains("Paris", StringComparison.OrdinalIgnoreCase),
                    $"Expected 'Paris' in output, got: {answer}"
                )
            | Failure errors ->
                output.WriteLine("Failed:")

                for e in errors do
                    output.WriteLine($"  {e}")

                Assert.Fail("Workflow returned failure")
            | PartialSuccess(ans, _) ->
                output.WriteLine($"Partial: {ans}")

                Assert.True(
                    ans.Contains("Paris", StringComparison.OrdinalIgnoreCase),
                    $"Expected 'Paris' in partial output, got: {ans}"
                )
        }
