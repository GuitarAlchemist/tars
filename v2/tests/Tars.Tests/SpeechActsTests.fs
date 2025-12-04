namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open Serilog

module SpeechActsTests =

    let createNullLogger () =
        LoggerConfiguration().CreateLogger() :> ILogger

    let createMockLlm (response: string) =
        { new ILlmService with
            member _.CompleteAsync req =
                task {
                    return
                        { Text = response
                          Usage = None
                          FinishReason = Some "stop"
                          Raw = None }
                }

            member _.CompleteStreamAsync(req, handler) = raise (NotImplementedException())
            member _.EmbedAsync text = raise (NotImplementedException()) }

    let createTestAgent () =
        { Id = AgentId(Guid.NewGuid())
          Name = "TestAgent"
          Version = "1.0.0"
          ParentVersion = None
          CreatedAt = DateTime.UtcNow
          Model = "test-model"
          SystemPrompt = "You are a test agent."
          Tools = []
          Capabilities = []
          State = Idle
          Memory = [] }

    [<Fact>]
    let ``Agent processes Request performative by entering Thinking state`` () =
        task {
            let agent = createTestAgent ()

            let msg =
                { Id = Guid.NewGuid()
                  CorrelationId = CorrelationId(Guid.NewGuid())
                  Sender = MessageEndpoint.User
                  Receiver = Some(MessageEndpoint.Agent agent.Id)
                  Performative = Performative.Request
                  Intent = None
                  Constraints = SemanticConstraints.Default
                  Ontology = None
                  Language = "text"
                  Content = "Do something"
                  Timestamp = DateTime.UtcNow
                  Metadata = Map.empty }

            let agentWithMsg = agent.ReceiveMessage(msg)

            // Act: Step the graph (Idle -> ?)
            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<IAgentRegistry>
                  Llm = createMockLlm "OK"
                  MaxSteps = 10
                  BudgetGovernor = None
                  OutputGuard = None
                  Logger = fun _ -> () }

            let! outcome = GraphRuntime.step agentWithMsg ctx

            match outcome with
            | Success nextAgent ->
                match nextAgent.State with
                | Thinking _ -> Assert.True(true, "Agent should be Thinking")
                | _ -> Assert.Fail($"Expected Thinking state, got {nextAgent.State}")
            | _ -> Assert.Fail("Step failed")
        }

    [<Fact>]
    let ``Agent can emit a Propose speech act`` () =
        task {
            // Setup an agent in Thinking state
            let agent = createTestAgent ()
            let thinkingState = Thinking []
            let agentThinking = { agent with State = thinkingState }

            // Mock LLM to return a PROPOSE act
            let llmResponse = "ACT: PROPOSE: I can do this for $50"

            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<IAgentRegistry>
                  Llm = createMockLlm llmResponse
                  MaxSteps = 10
                  BudgetGovernor = None
                  OutputGuard = None
                  Logger = fun _ -> () }

            let! outcome = GraphRuntime.step agentThinking ctx

            match outcome with
            | Success nextAgent ->
                // We expect the agent to have transitioned to WaitingForUser (or a new state)
                // AND the output should reflect the speech act.
                // Currently GraphRuntime puts text response into WaitingForUser.
                // We want to verify that the system *recognized* it as a speech act if we implement that logic.

                // For this test to pass AFTER implementation, we might expect a specific state or memory update.
                // For now, let's assume we map it to WaitingForUser but with metadata?
                // Or maybe we introduce a new state 'Negotiating'?

                // Let's just check if it handled the text for now, we will refine this test as we implement.
                match nextAgent.State with
                | WaitingForUser content ->
                    Assert.Contains("Propose", content, StringComparison.OrdinalIgnoreCase)

                    // Verify memory
                    let lastMsg = nextAgent.Memory |> List.last
                    Assert.Equal(Performative.Propose, lastMsg.Performative)
                    Assert.Equal("I can do this for $50", lastMsg.Content)
                | _ -> Assert.Fail($"Expected WaitingForUser, got {nextAgent.State}")
            | _ -> Assert.Fail("Step failed")
        }
