namespace Tars.Tests

open System
open Xunit
open Tars.Core

module GovernanceTests =

    [<Fact>]
    let ``BudgetGovernor tracks token usage`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        let cost = { Cost.Zero with Tokens = 50<token> }
        let result = governor.Consume(cost)

        Assert.True(
            match result with
            | Result.Ok _ -> true
            | _ -> false
        )

        let remaining = governor.Remaining
        Assert.Equal(Some 50<token>, remaining.MaxTokens)

    [<Fact>]
    let ``BudgetGovernor prevents spending over limit`` () =
        let budget =
            { Budget.Infinite with
                MaxTokens = Some 100<token> }

        let governor = BudgetGovernor(budget)

        // Spend 80
        let _ = governor.Consume({ Cost.Zero with Tokens = 80<token> })

        // Try to spend 30 (should fail)
        let resultFail = governor.TryConsume({ Cost.Zero with Tokens = 30<token> })

        Assert.True(
            match resultFail with
            | Result.Error _ -> true
            | _ -> false
        )

        // Try to spend 10 (should succeed)
        let resultSuccess = governor.TryConsume({ Cost.Zero with Tokens = 10<token> })

        Assert.True(
            match resultSuccess with
            | Result.Ok _ -> true
            | _ -> false
        )

    [<Fact>]
    let ``BudgetGovernor tracks calls`` () =
        let budget =
            { Budget.Infinite with
                MaxCalls = Some 100<requests> }

        let governor = BudgetGovernor(budget)

        // Consume 99 calls
        let cost =
            { Cost.Zero with
                CallCount = 99<requests> }

        let _ = governor.Consume(cost)

        // 100th call should succeed
        let result100 =
            governor.TryConsume(
                { Cost.Zero with
                    CallCount = 1<requests> }
            )

        Assert.True(
            match result100 with
            | Result.Ok _ -> true
            | _ -> false
        )

        // Actually consume it
        let _ =
            governor.Consume(
                { Cost.Zero with
                    CallCount = 1<requests> }
            )

        // 101st call should fail
        let result101 =
            governor.TryConsume(
                { Cost.Zero with
                    CallCount = 1<requests> }
            )

        Assert.True(
            match result101 with
            | Result.Error _ -> true
            | _ -> false
        )
