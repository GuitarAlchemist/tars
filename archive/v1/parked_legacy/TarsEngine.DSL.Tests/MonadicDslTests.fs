module TarsEngine.DSL.Tests.MonadicDslTests

open System
open System.Collections.Generic
open Xunit
open TarsEngine.DSL.SimpleDsl
open TarsEngine.DSL.Monads
open TarsEngine.DSL.MonadicDsl

[<Fact>]
let ``Test basic monadic DSL operations`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()

    // Define a simple computation
    let computation =
        dsl {
            // Set a variable
            do! Dsl.setVariable "x" (NumberValue 42.0)

            // Get the variable
            let! x = Dsl.getVariable "x"

            // Return the variable
            return x
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Ok (NumberValue n) ->
        Assert.Equal(42.0, n)
    | _ ->
        Assert.True(false, "Expected NumberValue 42.0")

    // Check the environment
    Assert.True(finalState.Environment.ContainsKey("x"))
    Assert.Equal(NumberValue 42.0, finalState.Environment.["x"])

[<Fact>]
let ``Test error handling in monadic DSL`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()

    // Define a computation that will fail
    let computation =
        dsl {
            // Set a variable
            do! Dsl.setVariable "x" (NumberValue 42.0)

            // Try to get a non-existent variable
            let! y = Dsl.getVariable "y"

            // This should not be executed
            return y
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Error msg ->
        Assert.Contains("not found", msg)
    | _ ->
        Assert.True(false, "Expected an error")

    // Check the environment (x should still be set)
    Assert.True(finalState.Environment.ContainsKey("x"))
    Assert.Equal(NumberValue 42.0, finalState.Environment.["x"])

[<Fact>]
let ``Test executing a block in monadic DSL`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()

    // Create a block
    let block = {
        Type = BlockType.Variable
        Name = Option.Some "x"
        Content = ""
        Properties = Map.ofList [
            "value", NumberValue 42.0
        ]
        NestedBlocks = []
    }

    // Define a computation that executes the block
    let computation =
        dsl {
            // Execute the block
            let! result = Dsl.executeBlock block

            // Get the variable
            let! x = Dsl.getVariable "x"

            // Return the variable
            return x
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Ok (NumberValue n) ->
        Assert.Equal(42.0, n)
    | _ ->
        Assert.True(false, "Expected NumberValue 42.0")

    // Check the environment
    Assert.True(finalState.Environment.ContainsKey("x"))
    Assert.Equal(NumberValue 42.0, finalState.Environment.["x"])

[<Fact>]
let ``Test for loop in monadic DSL`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()

    // Define a computation with a for loop
    let computation =
        dsl {
            // Initialize sum
            do! Dsl.setVariable "sum" (NumberValue 0.0)

            // Loop from 1 to 5
            for i in 1..5 do
                // Get current sum
                let! sum = Dsl.getVariable "sum"

                // Update sum
                match sum with
                | NumberValue n ->
                    do! Dsl.setVariable "sum" (NumberValue (n + float i))
                | _ ->
                    return! Dsl.error "Expected NumberValue"

            // Return the sum
            let! sum = Dsl.getVariable "sum"
            return sum
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Ok (NumberValue n) ->
        Assert.Equal(15.0, n) // 1 + 2 + 3 + 4 + 5 = 15
    | _ ->
        Assert.True(false, "Expected NumberValue 15.0")

    // Check the environment
    Assert.True(finalState.Environment.ContainsKey("sum"))
    Assert.Equal(NumberValue 15.0, finalState.Environment.["sum"])

[<Fact>]
let ``Test while loop in monadic DSL`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()
    env.["counter"] <- NumberValue 0.0

    // Define a computation with a while loop
    let computation =
        dsl {
            // Get initial counter value
            let! initialCounter = Dsl.getVariable "counter"
            let mutable continueLoop =
                match initialCounter with
                | NumberValue n -> n < 5.0
                | _ -> false

            // Loop while counter < 5
            while continueLoop do
                // Get current counter
                let! counter = Dsl.getVariable "counter"

                // Increment counter
                match counter with
                | NumberValue n ->
                    do! Dsl.setVariable "counter" (NumberValue (n + 1.0))
                    // Update continueLoop
                    continueLoop <- n + 1.0 < 5.0
                | _ ->
                    return! Dsl.error "Expected NumberValue"

            // Return the counter
            let! counter = Dsl.getVariable "counter"
            return counter
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Ok (NumberValue n) ->
        Assert.Equal(5.0, n)
    | _ ->
        Assert.True(false, "Expected NumberValue 5.0")

    // Check the environment
    Assert.True(finalState.Environment.ContainsKey("counter"))
    Assert.Equal(NumberValue 5.0, finalState.Environment.["counter"])

[<Fact>]
let ``Test try-with in monadic DSL`` () =
    // Create an environment
    let env = Dictionary<string, PropertyValue>()

    // Define a computation with try-with
    let computation =
        dsl {
            try
                // This will fail
                let! y = Dsl.getVariable "y"
                return y
            with ex ->
                // Set an error flag
                do! Dsl.setVariable "error_occurred" (BoolValue true)
                // Return a default value
                return NumberValue 0.0
        }

    // Run the computation
    let (result, finalState) = Dsl.run computation env

    // Check the result
    match result with
    | Ok (NumberValue n) ->
        Assert.Equal(0.0, n)
    | _ ->
        Assert.True(false, "Expected NumberValue 0.0")

    // Check the environment
    Assert.True(finalState.Environment.ContainsKey("error_occurred"))
    Assert.Equal(BoolValue true, finalState.Environment.["error_occurred"])
