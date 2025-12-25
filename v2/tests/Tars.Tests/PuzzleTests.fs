namespace Tars.Tests

open System
open Xunit
open Tars.Interface.Cli.Commands.PuzzleDemo

module PuzzleTests =

    [<Fact>]
    let ``All puzzles have valid definitions`` () =
        Assert.NotEmpty(allPuzzles)
        for puzzle in allPuzzles do
            Assert.False(String.IsNullOrWhiteSpace(puzzle.Name))
            Assert.False(String.IsNullOrWhiteSpace(puzzle.Description))
            Assert.False(String.IsNullOrWhiteSpace(puzzle.Prompt))
            Assert.True(puzzle.Difficulty >= 1 && puzzle.Difficulty <= 5)

    [<Fact>]
    let ``River Crossing validator works correctly`` () =
        let puzzle = allPuzzles |> List.find (fun p -> p.Name = "River Crossing")
        Assert.True(puzzle.Validator "Step 1: Goat. Step 2: Wolf. Step 3: Cabbage. Step 7: Finally everyone is across.")
        Assert.False(puzzle.Validator "Take goat across.") // Missing wolf, cabbage, step 7

    [<Fact>]
    let ``Monty Hall validator works correctly`` () =
        let puzzle = allPuzzles |> List.find (fun p -> p.Name = "Monty Hall Problem")
        Assert.True(puzzle.Validator "I should switch because the probability of winning increases to 2/3.")
        Assert.True(puzzle.Validator "Switching gives a 66% chance.")
        Assert.False(puzzle.Validator "I should stay.")

    [<Fact>]
    let ``Cheryl's Birthday validator works correctly`` () =
        let puzzle = allPuzzles |> List.find (fun p -> p.Name = "Cheryl's Birthday")
        Assert.True(puzzle.Validator "The date is July 16")
        Assert.True(puzzle.Validator "Cheryl's birthday is on July 16th.")
        Assert.False(puzzle.Validator "August 17")

    [<Fact>]
    let ``Space Station Maintenance validator works correctly`` () =
        let puzzle = allPuzzles |> List.find (fun p -> p.Name = "Space Station Maintenance")
        Assert.True(puzzle.Validator "Conclusion: Everything will be finished by 16:00.")
        Assert.True(puzzle.Validator "Conclusion: The earliest finish time is 4:00 PM.")
        Assert.False(puzzle.Validator "Everything will be finished by 16:00.") // Missing 'conclusion'
        Assert.False(puzzle.Validator "Conclusion: 17:00")
