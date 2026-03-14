namespace Tars.Tests

open Xunit
open Xunit.Abstractions
open Tars.Cortex.Grammar

type GrammarTests (output: ITestOutputHelper) =

    [<Fact>]
    member _.``Can parse simple goal with one task`` () =
        output.WriteLine("Starting test: Can parse simple goal with one task")
        let input = """
goal "MyGoal" {
    task "Task1"
}
"""
        output.WriteLine("Input grammar:")
        output.WriteLine(input)

        let goals = Parser.parse input
        output.WriteLine($"Parsed {goals.Length} goals.")
        
        Assert.Equal(1, goals.Length)
        let goal = goals[0]
        Assert.Equal("MyGoal", goal.Name)
        Assert.Equal(1, goal.Tasks.Length)
        Assert.Equal("Task1", goal.Tasks[0].Name)
        output.WriteLine("Verified goal name and task.")

    [<Fact>]
    member _.``Can parse multiple goals`` () =
        output.WriteLine("Starting test: Can parse multiple goals")
        let input = """
goal "G1" {
    task "T1"
}
goal "G2" {
    task "T2"
    task "T3"
}
"""
        output.WriteLine("Input grammar:")
        output.WriteLine(input)

        let goals = Parser.parse input
        output.WriteLine($"Parsed {goals.Length} goals.")
        Assert.Equal(2, goals.Length)
        
        // Note: The current implementation uses recursion accumulating to a list, check order.
        // "parseGoals ({ ... } :: acc) remaining" -> effectively reverse order if not reversed at end.
        // The code does "List.rev acc" in parseGoals, so order should be preserved.
        
        Assert.Equal("G1", goals[0].Name)
        Assert.Equal(1, goals[0].Tasks.Length)
        Assert.Equal("G2", goals[1].Name)
        Assert.Equal(2, goals[1].Tasks.Length)
        output.WriteLine("Verified multiple goals and their tasks.")

    [<Fact>]
    member _.``Can parse tasks with spaces`` () =
        output.WriteLine("Starting test: Can parse tasks with spaces")
        let input = """
goal "Complex Goal" {
    task "Task One"
    task "Task Two"
}
"""
        output.WriteLine("Input grammar:")
        output.WriteLine(input)
        
        let goals = Parser.parse input
        output.WriteLine($"Parsed {goals.Length} goals.")
        Assert.Equal("Complex Goal", goals[0].Name)
        Assert.Equal("Task One", goals[0].Tasks[0].Name)
        output.WriteLine("Verified names with spaces.")

    [<Fact>]
    member _.``Ignores empty lines and spaces`` () =
        output.WriteLine("Starting test: Ignores empty lines and spaces")
        let input = """
goal "Spaced" {

    task "T1"

}
"""
        output.WriteLine("Input grammar:")
        output.WriteLine(input)

        let goals = Parser.parse input
        output.WriteLine($"Parsed {goals.Length} goals.")
        Assert.Equal(1, goals[0].Tasks.Length)
        output.WriteLine("Verified parsing with extra whitespace.")
