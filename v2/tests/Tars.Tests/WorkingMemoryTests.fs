module Tars.Tests.WorkingMemoryTests

open System
open Xunit
open Tars.Core

[<Fact>]
let ``WorkingMemory respects capacity`` () =
    // Arrange
    let memory = WorkingMemory<string>(3)

    // Act
    memory.AddSimple("item1", 1.0)
    memory.AddSimple("item2", 1.0)
    memory.AddSimple("item3", 1.0)
    memory.AddSimple("item4", 1.0) // Should trigger pruning

    // Assert
    Assert.True(memory.Count <= 3)

[<Fact>]
let ``WorkingMemory keeps highest importance items`` () =
    // Arrange
    let memory = WorkingMemory<string>(2)

    // Act
    memory.AddSimple("low", 0.1)
    memory.AddSimple("high", 0.9)
    memory.AddSimple("medium", 0.5)

    // Assert
    let items = memory.GetAll()
    Assert.Equal(2, items.Length)
    // High importance item should be first
    Assert.Equal("high", items.[0].Content)

[<Fact>]
let ``WorkingMemory GetTop returns correct count`` () =
    // Arrange
    let memory = WorkingMemory<string>(10)

    for i in 1..5 do
        memory.AddSimple($"item{i}", float i * 0.1)

    // Act
    let top2 = memory.GetTop(2)

    // Assert
    Assert.Equal(2, top2.Length)

[<Fact>]
let ``WorkingMemory FindByTag filters correctly`` () =
    // Arrange
    let memory = WorkingMemory<string>(10)

    let importance =
        { BaseImportance = 1.0
          Recency = 0.0
          Relevance = 0.0
          SuccessWeight = 0.0 }

    memory.Add("tagged", importance, [ "important" ])
    memory.Add("untagged", importance)

    // Act
    let tagged = memory.FindByTag("important")

    // Assert
    Assert.Single(tagged) |> ignore
    Assert.Equal("tagged", tagged.[0].Content)

[<Fact>]
let ``WorkingMemory Touch updates access time`` () =
    // Arrange
    let memory = WorkingMemory<string>(10)
    memory.AddSimple("test", 1.0)
    let before = memory.GetAll().[0].LastAccessedAt

    // Act
    System.Threading.Thread.Sleep(10) // Small delay
    memory.Touch(fun s -> s = "test")
    let after = memory.GetAll().[0].LastAccessedAt

    // Assert
    Assert.True(after > before)

[<Fact>]
let ``WorkingMemory Clear removes all items`` () =
    // Arrange
    let memory = WorkingMemory<string>(10)
    memory.AddSimple("item1", 1.0)
    memory.AddSimple("item2", 1.0)

    // Act
    memory.Clear()

    // Assert
    Assert.Equal(0, memory.Count)

[<Fact>]
let ``WorkingMemory Statistics returns info`` () =
    // Arrange
    let memory = WorkingMemory<string>(10)
    memory.AddSimple("test", 1.0)

    // Act
    let stats = memory.Statistics()

    // Assert
    Assert.Contains("Count:", stats)

[<Fact>]
let ``ImportanceScore Total calculates correctly`` () =
    // Arrange
    let score =
        { BaseImportance = 1.0
          Recency = 0.5
          Relevance = 0.3
          SuccessWeight = 0.2 }

    // Act & Assert
    Assert.Equal(2.0, score.Total)
