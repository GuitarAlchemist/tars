module TarsEngine.DSL.Tests.AdvancedDslUnitTests

open System
open System.Collections.Generic
open Xunit
open TarsEngine.DSL.SimpleDsl
open TarsEngine.DSL.AdvancedDsl

[<Fact>]
let ``Test While Loop`` () =
    // Create a test environment
    let environment = Dictionary<string, PropertyValue>()
    environment.["counter"] <- NumberValue(0.0)
    
    // Create a while loop body
    let body = [
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
    
    // Execute the while loop
    let result = executeWhileLoop "${counter < 3}" body environment
    
    // Check the result
    match result with
    | Success _ -> 
        // Check that the counter was incremented to 3
        Assert.Equal(NumberValue(3.0), environment.["counter"])
    | Error msg -> 
        Assert.True(false, $"While loop failed: {msg}")

[<Fact>]
let ``Test For Loop`` () =
    // Create a test environment
    let environment = Dictionary<string, PropertyValue>()
    environment.["sum"] <- NumberValue(0.0)
    
    // Create a for loop body
    let body = [
        {
            Type = BlockType.Variable
            Name = Some "sum"
            Content = ""
            Properties = Map.ofList [
                "value", StringValue("${sum + i}")
            ]
            NestedBlocks = []
        }
    ]
    
    // Execute the for loop
    let result = executeForLoop "i" (NumberValue(1.0)) (NumberValue(5.0)) (Some (NumberValue(1.0))) body environment
    
    // Check the result
    match result with
    | Success _ -> 
        // Check that the sum is 1 + 2 + 3 + 4 + 5 = 15
        Assert.Equal(NumberValue(15.0), environment.["sum"])
    | Error msg -> 
        Assert.True(false, $"For loop failed: {msg}")

[<Fact>]
let ``Test Function Call`` () =
    // Create a test environment
    let environment = Dictionary<string, PropertyValue>()
    
    // Register a test function
    let functionBody = [
        {
            Type = BlockType.Variable
            Name = Some "result"
            Content = ""
            Properties = Map.ofList [
                "value", StringValue("${a + b}")
            ]
            NestedBlocks = []
        }
        {
            Type = BlockType.Return
            Name = None
            Content = ""
            Properties = Map.ofList [
                "value", StringValue("${result}")
            ]
            NestedBlocks = []
        }
    ]
    
    registerFunction "add" ["a"; "b"] functionBody
    
    // Execute the function call
    let args = Map.ofList [
        "a", NumberValue(2.0)
        "b", NumberValue(3.0)
    ]
    
    let result = executeFunction "add" args environment
    
    // Check the result
    match result with
    | Success (NumberValue n) -> 
        Assert.Equal(5.0, n)
    | Success _ -> 
        Assert.True(false, "Function call returned wrong type")
    | Error msg -> 
        Assert.True(false, $"Function call failed: {msg}")
    
    // Clean up
    clearFunctionRegistry()

[<Fact>]
let ``Test Try Catch`` () =
    // Create a test environment
    let environment = Dictionary<string, PropertyValue>()
    
    // Create a try block that will fail
    let tryBody = [
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
    ]
    
    // Create a catch block
    let catchBody = [
        {
            Type = BlockType.Variable
            Name = Some "caught"
            Content = ""
            Properties = Map.ofList [
                "value", StringValue("true")
            ]
            NestedBlocks = []
        }
    ]
    
    // Execute the try/catch
    let result = executeTryCatch tryBody catchBody environment
    
    // Check the result
    match result with
    | Success _ -> 
        // Check that the error was caught
        Assert.Equal(StringValue("true"), environment.["caught"])
    | Error msg -> 
        Assert.True(false, $"Try/catch failed: {msg}")
