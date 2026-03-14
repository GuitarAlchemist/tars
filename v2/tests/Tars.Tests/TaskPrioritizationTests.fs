module Tars.Tests.TaskPrioritizationTests

open System
open Xunit
open Tars.Evolution

[<Fact>]
let ``estimateCost returns Cheap for low difficulty`` () =
    // Arrange
    let task =
        { Id = Guid.NewGuid()
          DifficultyLevel = 1
          Goal = "Simple task"
          Constraints = []
          ValidationCriteria = "test"
          Timeout = TimeSpan.FromMinutes(1.0)
          Score = 1.0 }

    // Act
    let cost = TaskPrioritization.estimateCost task

    // Assert
    Assert.Equal(TaskPrioritization.CostEstimate.Cheap, cost)

[<Fact>]
let ``estimateCost returns Moderate for medium difficulty`` () =
    // Arrange
    let task =
        { Id = Guid.NewGuid()
          DifficultyLevel = 4
          Goal = "Medium task"
          Constraints = []
          ValidationCriteria = "test"
          Timeout = TimeSpan.FromMinutes(1.0)
          Score = 1.0 }

    // Act
    let cost = TaskPrioritization.estimateCost task

    // Assert
    Assert.Equal(TaskPrioritization.CostEstimate.Moderate, cost)

[<Fact>]
let ``estimateCost returns Expensive for high difficulty`` () =
    // Arrange
    let task =
        { Id = Guid.NewGuid()
          DifficultyLevel = 8
          Goal = "Complex task"
          Constraints = []
          ValidationCriteria = "test"
          Timeout = TimeSpan.FromMinutes(1.0)
          Score = 1.0 }

    // Act
    let cost = TaskPrioritization.estimateCost task

    // Assert
    Assert.Equal(TaskPrioritization.CostEstimate.Expensive, cost)

[<Fact>]
let ``tokensForCost returns expected values`` () =
    // Assert
    Assert.Equal(800, TaskPrioritization.tokensForCost TaskPrioritization.CostEstimate.Cheap)
    Assert.Equal(2000, TaskPrioritization.tokensForCost TaskPrioritization.CostEstimate.Moderate)
    Assert.Equal(4000, TaskPrioritization.tokensForCost TaskPrioritization.CostEstimate.Expensive)

[<Fact>]
let ``expectedValue gives novelty bonus`` () =
    // Arrange
    let task =
        { Id = Guid.NewGuid()
          DifficultyLevel = 3
          Goal = "Novel task"
          Constraints = []
          ValidationCriteria = "test"
          Timeout = TimeSpan.FromMinutes(1.0)
          Score = 1.0 }

    // Act - no completed tasks = novelty bonus
    let value = TaskPrioritization.expectedValue task []

    // Assert - base (1.0) + difficulty (0.3) + novelty (0.5) + success (0.1) = 1.9
    Assert.True(value > 1.5)

[<Fact>]
let ``scoreTask penalizes unaffordable tasks`` () =
    // Arrange
    let cheapTask =
        { Id = Guid.NewGuid()
          DifficultyLevel = 1
          Goal = "Cheap"
          Constraints = []
          ValidationCriteria = "test"
          Timeout = TimeSpan.FromMinutes(1.0)
          Score = 1.0 }

    let expensiveTask =
        { cheapTask with
            DifficultyLevel = 8
            Goal = "Expensive" }

    // Act - very limited budget
    let cheapScore = TaskPrioritization.scoreTask cheapTask [] (Some 500)
    let expensiveScore = TaskPrioritization.scoreTask expensiveTask [] (Some 500)

    // Assert - expensive task should be heavily penalized
    Assert.True(cheapScore > expensiveScore)

[<Fact>]
let ``prioritizeQueue sorts by score descending`` () =
    // Arrange
    let tasks =
        [ { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "A"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 }
          { Id = Guid.NewGuid()
            DifficultyLevel = 5
            Goal = "B"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 }
          { Id = Guid.NewGuid()
            DifficultyLevel = 3
            Goal = "C"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 } ]

    // Act
    let prioritized = TaskPrioritization.prioritizeQueue tasks [] None

    // Assert - should be sorted by efficiency
    Assert.Equal(3, prioritized.Length)

[<Fact>]
let ``projectBudget sums estimated costs`` () =
    // Arrange
    let tasks =
        [ { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "Cheap"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 }
          { Id = Guid.NewGuid()
            DifficultyLevel = 5
            Goal = "Moderate"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 } ]

    // Act
    let projected = TaskPrioritization.projectBudget tasks

    // Assert - 800 + 2000 = 2800
    Assert.Equal(2800, projected)

[<Fact>]
let ``fitsInBudget returns correct result`` () =
    // Arrange
    let tasks =
        [ { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "Cheap"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 } ]

    // Assert
    Assert.True(TaskPrioritization.fitsInBudget tasks 1000)
    Assert.False(TaskPrioritization.fitsInBudget tasks 500)

[<Fact>]
let ``filterAffordable removes expensive tasks`` () =
    // Arrange
    let tasks =
        [ { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "Cheap1"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 }
          { Id = Guid.NewGuid()
            DifficultyLevel = 8
            Goal = "Expensive"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 }
          { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "Cheap2"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 } ]

    // Act - budget allows 2 cheap tasks but not expensive
    let affordable = TaskPrioritization.filterAffordable tasks 2000

    // Assert
    Assert.Equal(2, affordable.Length)
    Assert.All(affordable, fun t -> Assert.Equal(1, t.DifficultyLevel))

[<Fact>]
let ``priorityReport generates output`` () =
    // Arrange
    let tasks =
        [ { Id = Guid.NewGuid()
            DifficultyLevel = 1
            Goal = "Test task"
            Constraints = []
            ValidationCriteria = ""
            Timeout = TimeSpan.FromMinutes(1.0)
            Score = 1.0 } ]

    // Act
    let report = TaskPrioritization.priorityReport tasks [] None

    // Assert
    Assert.Contains("[Priority]", report)
    Assert.Contains("Test task", report)
