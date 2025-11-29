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

        let workflow = agent { return 42 }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None
              CancellationToken = cts.Token }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Failure errs ->
            Assert.Contains(errs, fun e ->
                match e with
                | PartialFailure.Warning msg -> msg.Contains("cancelled")
                | _ -> false)
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
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal(99, v)
        | _ -> Assert.Fail("Should be Success")

    [<Fact>]
    let ``Agent workflow with no budget skips budget check`` () =
        let workflow =
            agent {
                do! AgentWorkflow.checkBudget { Cost.Zero with Tokens = 1000000<token> }
                return "passed"
            }

        let ctx =
            { Self = createTestAgent ()
              Registry = StubRegistry()
              Executor = StubExecutor()
              Logger = fun _ -> ()
              Budget = None // No budget = unlimited
              CancellationToken = CancellationToken.None }

        let result = workflow ctx |> Async.RunSynchronously

        match result with
        | Success v -> Assert.Equal("passed", v)
        | _ -> Assert.Fail("Should be Success when no budget is set")
