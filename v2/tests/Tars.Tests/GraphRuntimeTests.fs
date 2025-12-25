namespace Tars.Tests

open System
open System.Threading.Tasks
open Xunit
open Tars.Core
open Tars.Graph
open Tars.Llm
open Tars.Llm.LlmService
open Tars.Kernel

module GraphRuntimeTests =

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
            member _.RouteAsync _ = Task.FromResult { Backend = Ollama "mock"; Endpoint = Uri "http://localhost:11434"; ApiKey = None } }

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
    let ``handleThinking returns Failure when budget is exhausted`` () =
        task {
            let agent = createTestAgent ()
            let history = []

            let exhaustedBudget =
                let b =
                    BudgetGovernor(
                        { Budget.Infinite with
                            MaxTokens = Some 0<token> }
                    )
                // Consume all
                b.Consume { Cost.Zero with Tokens = 100<token> } |> ignore
                b

            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<_>
                  Llm = createMockLlm "Response"
                  MaxSteps = 10
                  BudgetGovernor = Some exhaustedBudget
                  OutputGuard = None
                  CancellationToken = System.Threading.CancellationToken.None
                  Logger = fun _ -> () }

            let thinkingAgent = { agent with State = Thinking history }

            let! outcome = GraphRuntime.step thinkingAgent ctx

            match outcome with
            | Failure errs ->
                Assert.Contains(
                    errs,
                    fun e ->
                        match e with
                        | PartialFailure.Error msg -> msg.Contains("Budget")
                        | _ -> false
                )
            | _ -> Assert.Fail("Expected Failure due to budget exhaustion")
        }

    [<Fact>]
    let ``handleActing returns PartialSuccess when tool execution fails`` () =
        task {
            let tool =
                { Name = "FailingTool"
                  Description = "Always fails"
                  Version = "1.0"
                  ParentVersion = None
                  CreatedAt = DateTime.UtcNow
                  Execute = fun _ -> async { return Result.Error "Tool crashed" }
                  ThingDescription = None }

            let agent =
                { createTestAgent () with
                    Tools = [ tool ]
                    State = Acting(tool, "input") }

            let ctx: GraphRuntime.GraphContext =
                { Registry = Unchecked.defaultof<_>
                  Llm = createMockLlm "Response"
                  MaxSteps = 10
                  BudgetGovernor = None
                  OutputGuard = None
                  CancellationToken = System.Threading.CancellationToken.None
                  Logger = fun _ -> () }

            let! outcome = GraphRuntime.step agent ctx

            match outcome with
            | PartialSuccess(nextAgent, warnings) ->
                // Check state transition
                match nextAgent.State with
                | Observing(t, output) ->
                    Assert.Equal("FailingTool", t.Name)
                    Assert.Contains("Error: Tool crashed", output)
                | _ -> Assert.Fail($"Expected Observing state, got {nextAgent.State}")

                // Check warnings
                Assert.Contains(
                    warnings,
                    fun w ->
                        match w with
                        | PartialFailure.ToolError(name, err) -> name = "FailingTool" && err = "Tool crashed"
                        | _ -> false
                )
            | _ -> Assert.Fail("Expected PartialSuccess due to tool failure")
        }
