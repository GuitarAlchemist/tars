module TarsEngine.DSL.Tests.AdvancedFeaturesIntegrationTests

open System
open System.Collections.Generic
open Xunit
open TarsEngine.DSL.SimpleDsl

[<Fact>]
let ``Test While Loop`` () =
    // Create a program with a while loop
    let program = {
        Blocks = [
            {
                Type = BlockType.Variable
                Name = Some "counter"
                Content = ""
                Properties = Map.ofList [
                    "value", NumberValue(0.0)
                ]
                NestedBlocks = []
            }
            {
                Type = BlockType.While
                Name = None
                Content = ""
                Properties = Map.ofList [
                    "condition", StringValue("${counter < 3}")
                ]
                NestedBlocks = [
                    {
                        Type = BlockType.Action
                        Name = None
                        Content = ""
                        Properties = Map.ofList [
                            "type", StringValue("log")
                            "message", StringValue("Counter: ${counter}")
                        ]
                        NestedBlocks = []
                    }
                    {
                        Type = BlockType.Variable
                        Name = Some "counter"
                        Content = ""
                        Properties = Map.ofList [
                            "value", StringValue("${counter + 1}")
                        ]
                        NestedBlocks = []
                    }
                ]
            }
        ]
    }

    // Execute the program
    let result = executeProgram program

    // Check the result
    match result with
    | Success _ ->
        // Test passes if no error occurred
        Assert.True(true)
    | Error msg ->
        Assert.True(false, $"Program execution failed: {msg}")

[<Fact>]
let ``Test Function Definition and Call`` () =
    // Create a program with a function definition and call
    let program = {
        Blocks = [
            {
                Type = BlockType.Function
                Name = Some "add"
                Content = ""
                Properties = Map.ofList [
                    "parameters", StringValue("a, b")
                ]
                NestedBlocks = [
                    {
                        Type = BlockType.Return
                        Name = None
                        Content = ""
                        Properties = Map.ofList [
                            "value", StringValue("${a + b}")
                        ]
                        NestedBlocks = []
                    }
                ]
            }
            {
                Type = BlockType.Variable
                Name = Some "result"
                Content = ""
                Properties = Map.ofList [
                    "value", NumberValue(0.0)
                ]
                NestedBlocks = []
            }
            {
                Type = BlockType.Call
                Name = None
                Content = ""
                Properties = Map.ofList [
                    "function", StringValue("add")
                    "arguments", ObjectValue(Map.ofList [
                        "a", NumberValue(2.0)
                        "b", NumberValue(3.0)
                    ])
                    "result_variable", StringValue("result")
                ]
                NestedBlocks = []
            }
            {
                Type = BlockType.Action
                Name = None
                Content = ""
                Properties = Map.ofList [
                    "type", StringValue("log")
                    "message", StringValue("Result: ${result}")
                ]
                NestedBlocks = []
            }
        ]
    }

    // Execute the program
    let result = executeProgram program

    // Check the result
    match result with
    | Success _ ->
        // Test passes if no error occurred
        Assert.True(true)
    | Error msg ->
        Assert.True(false, $"Program execution failed: {msg}")

[<Fact>]
let ``Test Try Catch`` () =
    // Create a program with a try/catch block
    let program = {
        Blocks = [
            {
                Type = BlockType.Try
                Name = None
                Content = ""
                Properties = Map.empty
                NestedBlocks = [
                    {
                        Type = BlockType.Action
                        Name = None
                        Content = ""
                        Properties = Map.ofList [
                            "type", StringValue("unknown_action")
                            "message", StringValue("This will fail")
                        ]
                        NestedBlocks = []
                    }
                    {
                        Type = BlockType.Catch
                        Name = None
                        Content = ""
                        Properties = Map.empty
                        NestedBlocks = [
                            {
                                Type = BlockType.Variable
                                Name = Some "caught"
                                Content = ""
                                Properties = Map.ofList [
                                    "value", StringValue("true")
                                ]
                                NestedBlocks = []
                            }
                            {
                                Type = BlockType.Action
                                Name = None
                                Content = ""
                                Properties = Map.ofList [
                                    "type", StringValue("log")
                                    "message", StringValue("Caught error: ${error}")
                                ]
                                NestedBlocks = []
                            }
                        ]
                    }
                ]
            }
        ]
    }

    // Execute the program
    let result = executeProgram program

    // Check the result
    match result with
    | Success _ ->
        // Test passes if no error occurred
        Assert.True(true)
    | Error msg ->
        Assert.True(false, $"Program execution failed: {msg}")

[<Fact>]
let ``Test For Loop`` () =
    // Create a program with a for loop
    let program = {
        Blocks = [
            {
                Type = BlockType.Variable
                Name = Some "sum"
                Content = ""
                Properties = Map.ofList [
                    "value", NumberValue(0.0)
                ]
                NestedBlocks = []
            }
            {
                Type = BlockType.For
                Name = None
                Content = ""
                Properties = Map.ofList [
                    "variable", StringValue("i")
                    "from", NumberValue(1.0)
                    "to", NumberValue(5.0)
                    "step", NumberValue(1.0)
                ]
                NestedBlocks = [
                    {
                        Type = BlockType.Variable
                        Name = Some "sum"
                        Content = ""
                        Properties = Map.ofList [
                            "value", StringValue("${sum + i}")
                        ]
                        NestedBlocks = []
                    }
                    {
                        Type = BlockType.Action
                        Name = None
                        Content = ""
                        Properties = Map.ofList [
                            "type", StringValue("log")
                            "message", StringValue("Sum: ${sum}, i: ${i}")
                        ]
                        NestedBlocks = []
                    }
                ]
            }
        ]
    }

    // Execute the program
    let result = executeProgram program

    // Check the result
    match result with
    | Success _ ->
        // Test passes if no error occurred
        Assert.True(true)
    | Error msg ->
        Assert.True(false, $"Program execution failed: {msg}")
