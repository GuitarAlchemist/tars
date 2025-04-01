namespace TarsEngine.DSL

/// Module containing advanced features for the TARS DSL
module AdvancedDsl =
    open System
    open System.Collections.Generic
    open System.Text.RegularExpressions
    open TarsEngine.DSL.SimpleDsl
    
    /// Block types for advanced features
    type AdvancedBlockType =
        | While
        | For
        | Function
        | Call
        | Try
        | Catch
        | Return
    
    /// Convert an advanced block type to a string
    let advancedBlockTypeToString (blockType: AdvancedBlockType) =
        match blockType with
        | While -> "WHILE"
        | For -> "FOR"
        | Function -> "FUNCTION"
        | Call -> "CALL"
        | Try -> "TRY"
        | Catch -> "CATCH"
        | Return -> "RETURN"
    
    /// A function definition
    type FunctionDef = {
        Name: string
        Parameters: string list
        Body: Block list
    }
    
    /// Global function registry
    let mutable functionRegistry = Map.empty<string, FunctionDef>
    
    /// Execute a while loop
    let executeWhileLoop (condition: string) (body: Block list) (environment: Dictionary<string, PropertyValue>) =
        // Execute the loop
        let mutable result = Success(StringValue(""))
        let mutable continueLoop = true
        let mutable loopCount = 0
        let maxLoops = 1000 // Safety limit to prevent infinite loops
        
        while continueLoop && loopCount < maxLoops do
            // Substitute variables in the condition
            let substitutedCondition = substituteVariables condition environment
            
            // Evaluate the condition
            let conditionResult = evaluateCondition substitutedCondition
            
            if conditionResult then
                // Execute the nested blocks
                let mutable blockResult = Success(StringValue(""))
                let mutable continueExecution = true
                
                for nestedBlock in body do
                    if continueExecution then
                        match executeBlock nestedBlock environment with
                        | Success value -> blockResult <- Success value
                        | Error msg ->
                            blockResult <- Error msg
                            // Stop execution on error
                            continueExecution <- false
                            continueLoop <- false
                
                result <- blockResult
                loopCount <- loopCount + 1
            else
                // Condition is false, exit the loop
                continueLoop <- false
        
        // Check if we hit the max loop count
        if loopCount >= maxLoops then
            Error($"While loop exceeded maximum iteration count ({maxLoops})")
        else
            result
    
    /// Execute a for loop
    let executeForLoop (variable: string) (fromValue: PropertyValue) (toValue: PropertyValue) (stepValue: PropertyValue option) (body: Block list) (environment: Dictionary<string, PropertyValue>) =
        // Parse from and to values
        let fromNum = 
            match fromValue with
            | NumberValue n -> n
            | StringValue s -> 
                match Double.TryParse(substituteVariables s environment) with
                | true, n -> n
                | _ -> 0.0
            | _ -> 0.0
            
        let toNum = 
            match toValue with
            | NumberValue n -> n
            | StringValue s -> 
                match Double.TryParse(substituteVariables s environment) with
                | true, n -> n
                | _ -> 0.0
            | _ -> 0.0
            
        // Parse step value (default to 1.0)
        let stepNum = 
            match stepValue with
            | Some (NumberValue n) -> n
            | Some (StringValue s) -> 
                match Double.TryParse(substituteVariables s environment) with
                | true, n -> n
                | _ -> 1.0
            | _ -> 1.0
            
        // Execute the loop
        let mutable result = Success(StringValue(""))
        let mutable loopCount = 0
        let maxLoops = 1000 // Safety limit to prevent infinite loops
        
        let mutable i = fromNum
        while (stepNum > 0.0 && i <= toNum) || (stepNum < 0.0 && i >= toNum) do
            if loopCount >= maxLoops then
                result <- Error($"For loop exceeded maximum iteration count ({maxLoops})")
                i <- if stepNum > 0.0 then toNum + 1.0 else toNum - 1.0 // Exit the loop
            else
                // Set the loop variable
                environment.[variable] <- NumberValue(i)
                
                // Execute the nested blocks
                let mutable blockResult = Success(StringValue(""))
                let mutable continueExecution = true
                
                for nestedBlock in body do
                    if continueExecution then
                        match executeBlock nestedBlock environment with
                        | Success value -> blockResult <- Success value
                        | Error msg ->
                            blockResult <- Error msg
                            // Stop execution on error
                            continueExecution <- false
                            i <- if stepNum > 0.0 then toNum + 1.0 else toNum - 1.0 // Exit the loop
                
                result <- blockResult
                loopCount <- loopCount + 1
                i <- i + stepNum
        
        result
    
    /// Execute a function call
    let executeFunction (functionName: string) (args: Map<string, PropertyValue>) (environment: Dictionary<string, PropertyValue>) =
        // Look up the function in the registry
        match functionRegistry.TryFind(functionName) with
        | Some functionDef ->
            // Create a new environment for the function call
            let functionEnv = Dictionary<string, PropertyValue>()
            
            // Copy global variables to function environment
            for KeyValue(key, value) in environment do
                functionEnv.[key] <- value
            
            // Bind arguments to parameters
            for paramName in functionDef.Parameters do
                match args.TryFind(paramName) with
                | Some argValue -> 
                    // Substitute variables in string arguments
                    match argValue with
                    | StringValue s -> functionEnv.[paramName] <- StringValue(substituteVariables s environment)
                    | _ -> functionEnv.[paramName] <- argValue
                | None -> 
                    // Parameter not provided
                    functionEnv.[paramName] <- StringValue("")
            
            // Execute the function body
            let mutable result = Success(StringValue(""))
            let mutable continueExecution = true
            
            for nestedBlock in functionDef.Body do
                if continueExecution then
                    match executeBlock nestedBlock functionEnv with
                    | Success value -> result <- Success value
                    | Error msg ->
                        result <- Error msg
                        // Stop execution on error
                        continueExecution <- false
            
            result
        | None ->
            Error($"Function '{functionName}' not found")
    
    /// Execute a try/catch block
    let executeTryCatch (tryBody: Block list) (catchBody: Block list) (environment: Dictionary<string, PropertyValue>) =
        // Execute the try block
        let mutable result = Success(StringValue(""))
        let mutable errorOccurred = false
        let mutable errorMessage = ""
        
        // Execute the try block
        for nestedBlock in tryBody do
            if not errorOccurred then
                match executeBlock nestedBlock environment with
                | Success value -> result <- Success value
                | Error msg ->
                    errorOccurred <- true
                    errorMessage <- msg
        
        if errorOccurred then
            // Store the error message in the environment
            environment.["error"] <- StringValue(errorMessage)
            
            // Execute the catch block
            let mutable catchResult = Success(StringValue(""))
            let mutable continueExecution = true
            
            for nestedBlock in catchBody do
                if continueExecution then
                    match executeBlock nestedBlock environment with
                    | Success value -> catchResult <- Success value
                    | Error msg ->
                        catchResult <- Error msg
                        // Stop execution on error
                        continueExecution <- false
            
            catchResult
        else
            // No error occurred, return the result
            result
    
    /// Register a function
    let registerFunction (name: string) (parameters: string list) (body: Block list) =
        let functionDef = {
            Name = name
            Parameters = parameters
            Body = body
        }
        functionRegistry <- functionRegistry.Add(name, functionDef)
    
    /// Clear the function registry
    let clearFunctionRegistry () =
        functionRegistry <- Map.empty
