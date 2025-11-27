namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Kernel

type GovernanceTests() =

    [<Fact>]
    member _.``BudgetGovernor tracks tokens correctly with units``() =
        // Arrange
        let maxTokens = 100
        let governor = BudgetGovernor(maxTokens)
        let correlationId = Guid.NewGuid()

        // Act
        // Spend 50 tokens
        let canSpend1 = governor.CanSpend(correlationId, 50<token>)
        governor.RecordUsage(correlationId, 50<token>)

        // Check remaining
        let (remTokens, _, _) = governor.GetRemainingBudget(correlationId)

        // Spend 60 tokens (should fail check if we were strict, but CanSpend just checks if current + est <= max)
        // 50 used + 60 est = 110 > 100. Should return false.
        let canSpend2 = governor.CanSpend(correlationId, 60<token>)

        // Assert
        Assert.True(canSpend1)
        Assert.Equal(50<token>, remTokens)
        Assert.False(canSpend2)
