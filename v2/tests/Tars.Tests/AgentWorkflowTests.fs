namespace Tars.Tests

open System
open System.Threading
open Xunit
open Tars.Core

module AgentWorkflowTests =

    // Stub implementations for testing
    type StubRegistry() =
        interface IAgentRegistry with
            member _.GetAgent(_) = async { return None }
            member _.FindAgents(_) = async { return [] }
            member _.GetAllAgents() = async { return [] }

    type StubExecutor() =
        interface IAgentExecutor with
            member _.Execute(_, _) = async { return Success "stub result" }

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
    let ``Agent workflow returns success`` () =
        let workflow = agent { return 42 }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(42, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Agent workflow binds correctly`` () =
        let workflow =
            agent {
                let! a = agent { return 10 }
                let! b = agent { return 20 }
                return a + b
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(30, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Agent workflow checks budget`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        // Consume 90
        governor.Consume({ Cost.Zero with Tokens = 90<token> }) |> ignore

        let workflow =
            agent {
                // Try to spend 20 (should fail)
                do! AgentWorkflow.checkBudget { Cost.Zero with Tokens = 20<token> }
                return "Success"
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = Some governor
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Failure errs ->
            Assert.Contains(
                errs,
                fun e ->
                    match e with
                    | PartialFailure.Warning msg -> msg.Contains("Budget exceeded")
                    | _ -> false
            )
        | _ -> Assert.Fail($"Should be Failure, but was {result}")

    [<Fact>]
    let ``Agent workflow accumulates warnings`` () =
        let workflow =
            agent {
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "Watch out")
                return 42
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal(42, v)
            Assert.Contains(PartialFailure.Warning "Watch out", warnings)
        | _ -> Assert.Fail("Should be PartialSuccess")

    // === Edge Case Tests ===

    [<Fact>]
    let ``Agent workflow respects cancellation token`` () =
        let cts = new CancellationTokenSource()
        cts.Cancel() // Pre-cancel

        // Use a bind (let!) to trigger cancellation check
        let workflow =
            agent {
                let! a = agent { return 10 }
                return a + 32
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = cts.Token }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Failure errs ->
            Assert.Contains(
                errs,
                fun e ->
                    match e with
                    | PartialFailure.Warning msg -> msg.Contains("cancelled")
                    | _ -> false
            )
        | _ -> Assert.Fail("Should be Failure due to cancellation")

    [<Fact>]
    let ``Agent workflow chains multiple warnings`` () =
        let workflow =
            agent {
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "Warning 1")
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "Warning 2")
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "Warning 3")
                return "done"
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal("done", v)
            Assert.Equal(3, warnings.Length)
        | _ -> Assert.Fail("Should be PartialSuccess with 3 warnings")

    [<Fact>]
    let ``Agent workflow fail stops execution`` () =
        let mutable reached = false

        let workflow =
            agent {
                do! AgentWorkflow.fail (PartialFailure.Error "Early exit")
                reached <- true
                return "should not reach"
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        Assert.False(reached, "Code after fail should not execute")

        match result with
        | Failure _ -> ()
        | _ -> Assert.Fail("Should be Failure")

    [<Fact>]
    let ``Agent workflow succeed helper works`` () =
        let workflow = AgentWorkflow.succeed 99

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(99, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Agent workflow with no budget skips budget check`` () =
        let workflow =
            agent {
                do!
                    AgentWorkflow.checkBudget
                        { Cost.Zero with
                            Tokens = 1000000<token> }

                return "passed"
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None // No budget = unlimited
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal("passed", v)
        | _ -> Assert.Fail("Should be Success when no budget is set")

    // === Circuit Combinator Tests ===

    [<Fact>]
    let ``Transform: Maps success value`` () =
        let workflow = AgentWorkflow.succeed 10
        let transformed = workflow |> AgentWorkflow.transform (fun x -> x * 2)

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = transformed ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(20, v)
        | _ -> Assert.Fail("Should be Success with transformed value")

    [<Fact>]
    let ``Transform: Maps partial success preserving warnings`` () =
        let workflow =
            agent {
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "watch out")
                return 5
            }

        let transformed = workflow |> AgentWorkflow.transform (fun x -> x.ToString())

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = transformed ctx |> Async.RunSynchronously

        match result with
        | PartialSuccess(v, warnings) ->
            Assert.Equal("5", v)
            Assert.Single(warnings) |> ignore
        | _ -> Assert.Fail("Should be PartialSuccess with warnings preserved")

    [<Fact>]
    let ``Transform: Passes through failure`` () =
        let workflow: AgentWorkflow<int> =
            fun _ -> async { return Failure [ PartialFailure.Error "failed" ] }

        let transformed = workflow |> AgentWorkflow.transform (fun x -> x * 2)

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = transformed ctx |> Async.RunSynchronously

        match result with
        | Failure _ -> ()
        | _ -> Assert.Fail("Should pass through Failure")

    [<Fact>]
    let ``Stabilize: Logs with high inertia`` () =
        let mutable logged = false
        let workflow = AgentWorkflow.succeed "stable"
        let stabilized = workflow |> AgentWorkflow.stabilize 0.7

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger =
                fun msg ->
                    if msg.Contains("Stabilizing") then
                        logged <- true
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = stabilized ctx |> Async.RunSynchronously

        Assert.True(logged, "Should log stabilization message for high inertia")

        match result with
        | Success v -> Assert.Equal("stable", v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Stabilize: No log with low inertia`` () =
        let mutable logged = false
        let workflow = AgentWorkflow.succeed "fast"
        let stabilized = workflow |> AgentWorkflow.stabilize 0.3

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger =
                fun msg ->
                    if msg.Contains("Stabilizing") then
                        logged <- true
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = stabilized ctx |> Async.RunSynchronously

        Assert.False(logged, "Should not log for low inertia")

        match result with
        | Success v -> Assert.Equal("fast", v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``ForwardOnly: Logs forward-only enforcement`` () =
        let mutable logged = false
        let workflow = AgentWorkflow.succeed "forward"
        let protected' = workflow |> AgentWorkflow.forwardOnly

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger =
                fun msg ->
                    if msg.Contains("forward-only") then
                        logged <- true
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = protected' ctx |> Async.RunSynchronously

        Assert.True(logged, "Should log forward-only message")

        match result with
        | Success v -> Assert.Equal("forward", v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``ForwardOnly: Preserves failure`` () =
        let workflow: AgentWorkflow<int> =
            fun _ -> async { return Failure [ PartialFailure.Error "blocked" ] }

        let protected' = workflow |> AgentWorkflow.forwardOnly

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = protected' ctx |> Async.RunSynchronously

        match result with
        | Failure errs ->
            Assert.Contains(
                errs,
                fun e ->
                    match e with
                    | PartialFailure.Error "blocked" -> true
                    | _ -> false
            )
        | _ -> Assert.Fail("Should preserve Failure")

    [<Fact>]
    let ``Grounded: Logs verification for success`` () =
        let mutable logged = false
        let workflow = AgentWorkflow.succeed "verified"
        let grounded = workflow |> AgentWorkflow.grounded

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger =
                fun msg ->
                    if msg.Contains("Grounding") then
                        logged <- true
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = grounded ctx |> Async.RunSynchronously

        Assert.True(logged, "Should log grounding verification")

        match result with
        | Success v -> Assert.Equal("verified", v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Grounded: Logs verification for partial success`` () =
        let mutable logged = false

        let workflow =
            agent {
                do! AgentWorkflow.warnWith () (PartialFailure.Warning "uncertain")
                return "maybe"
            }

        let grounded = workflow |> AgentWorkflow.grounded

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger =
                fun msg ->
                    if msg.Contains("Grounding") then
                        logged <- true
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = grounded ctx |> Async.RunSynchronously

        match result with
        | PartialSuccess(v, _) -> Assert.Equal("maybe", v)
        | _ -> Assert.Fail("Should be PartialSuccess")

    [<Fact>]
    let ``Circuit combinators can be composed`` () =
        let workflow = AgentWorkflow.succeed 5

        let composed =
            workflow
            |> AgentWorkflow.transform (fun x -> x * 2) // 10
            |> AgentWorkflow.stabilize 0.3 // pass through
            |> AgentWorkflow.forwardOnly // pass through
            |> AgentWorkflow.transform (fun x -> x + 1) // 11
            |> AgentWorkflow.grounded // verify

        let mutable logs = []

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun msg -> logs <- msg :: logs
              Budget = None
              Epistemic = None
              CancellationToken = CancellationToken.None }

        let result = composed ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(11, v)
        | _ -> Assert.Fail("Should be Success")

        Assert.True(logs |> List.exists (fun s -> s.Contains("forward-only")), "Should have forward-only log")
        Assert.True(logs |> List.exists (fun s -> s.Contains("Grounding")), "Should have grounding log")
