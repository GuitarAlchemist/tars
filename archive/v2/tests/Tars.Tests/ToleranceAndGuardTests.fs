namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Kernel
open Tars.Core.ToleranceEngineering

module ToleranceAndGuardTests =

    let createMockLlm (response: string) =
        { new ILlmService with
            member _.CompleteAsync req =
                task {
                    return
                        { Text = response
                          Usage =
                            Some
                                { PromptTokens = 10
                                  CompletionTokens = 10
                                  TotalTokens = 20 }
                          FinishReason = Some "stop"
                          Raw = None }
                }

            member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
            member _.EmbedAsync text = Task.FromResult [| 0.1f |]

            member _.RouteAsync _ =
                Task.FromResult
                    { Backend = Ollama "mock"
                      Endpoint = Uri "http://localhost:11434"
                      ApiKey = None } }

    let createTestAgent () =
        { Id = AgentId( Guid.NewGuid())
          Name = "TestAgent"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "test-model"
          SystemPrompt = "You are a test agent."
          Tools = []
          Capabilities = []
          State = Idle
          Memory = []
          Fitness = 1.0
          Drives =
            { Accuracy = 0.5
              Speed = 0.5
              Creativity = 0.5
              Safety = 0.5 }
          Constitution = AgentConstitution.Create(AgentId(Guid.NewGuid()), NeuralRole.GeneralReasoning) }

    [<Fact>]
    let ``Protocol Guard intercepts questions and retries with hint`` () =
        task {
            // This response contains a question ("Ask" speech act)
            let questionResponse = "ACT: QUERY: What time is it?"
            let agent = createTestAgent ()
            let history = []
            
            let mutable logMessages = []
            let logger msg = logMessages <- logMessages @ [msg]

            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<_>
                  Llm = createMockLlm questionResponse
                  MaxSteps = 10
                  BudgetGovernor = None
                  OutputGuard = None
                  CancellationToken = System.Threading.CancellationToken.None
                  Logger = logger
                  ToleranceMetrics = None
                  ToleranceProfile = None }

            let thinkingAgent = { agent with State = Thinking history }

            // Running step should trigger the loop. 
            // Since the mock LLM ALWAYS returns a question, it should hit max attempts and fail.
            // But we can check the logs for the protocol violation message.
            let! outcome = GraphRuntime.step thinkingAgent ctx

            // Verify protocol violation was logged
            Assert.Contains(logMessages, fun (m: string) -> m.Contains("[Protocol] \u26a0\ufe0f Violation detected"))
            
            // Verify it attempted to retry (max attempts = 3)
            // The logs should show multiple "Agent ... is thinking" entries or protocol violations
            let violationCount = logMessages |> List.filter (fun m -> m.Contains("[Protocol]")) |> List.length
            Assert.Equal(3, violationCount)
        }

    [<Fact>]
    let ``Entropy penalty triggers tolerance retry for repetitive output`` () =
        task {
            // Highly repetitive output should trigger entropy penalty
            let repetitiveResponse = "word word word word word word word word word word"
            let agent = createTestAgent ()
            let history = []
            
            let mutable logMessages = []
            let logger msg = logMessages <- logMessages @ [msg]

            let metrics = MetricsAggregator()
            // Profile that requires 0.80 confidence, Base is 0.70, Entropy penalty -0.30 -> Net 0.40
            // This should trigger BelowTolerance(0.40, Retry 3)
            let profile = Custom { 
                ConfidenceThreshold = 0.80
                MaxVariance = 0.20
                MaxRetries = 3
                HumanReviewThreshold = 0.30
                SafetyFactor = 1.0
                AllowDegraded = false
            }

            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<_>
                  Llm = createMockLlm repetitiveResponse
                  MaxSteps = 10
                  BudgetGovernor = None
                  OutputGuard = None
                  CancellationToken = System.Threading.CancellationToken.None
                  Logger = logger
                  ToleranceMetrics = Some metrics
                  ToleranceProfile = Some profile }

            let thinkingAgent = { agent with State = Thinking history }

            let! _ = GraphRuntime.step thinkingAgent ctx

            // Verify entropy-induced tolerance retry was logged
            // Check for "Below tolerance: conf=0.30"
            Assert.Contains(logMessages, fun (m: string) -> m.Contains("[Tolerance] \ud83d\udd34 Below tolerance: conf=0.30"))
            Assert.Contains(logMessages, fun (m: string) -> m.Contains("[Tolerance] \ud83d\udd04 Retrying due to low confidence"))
        }
