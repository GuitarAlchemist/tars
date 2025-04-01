namespace TarsEngine.DSL

open Ast
open AgentInterfaces
open System
open System.Collections.Generic

/// Module containing the interpreter for the TARS DSL
module Interpreter =
    /// Environment for storing variables during execution
    type Environment = Map<string, PropertyValue>

    /// Result of executing a TARS program
    type ExecutionResult =
        | Success of PropertyValue
        | Error of string

    /// Function call stack frame
    type StackFrame = {
        FunctionName: string
        LocalVariables: Environment
        ReturnValue: PropertyValue option
    }

    /// Execution context
    type ExecutionContext = {
        Environment: Environment
        CallStack: StackFrame list
        BreakPoints: Set<string * int>  // (file, line) pairs
        StepMode: bool
        CurrentFile: string
        CurrentLine: int
        Paused: bool
    }

    /// Create a new execution context
    let createContext() = {
        Environment = Map.empty
        CallStack = []
        BreakPoints = Set.empty
        StepMode = false
        CurrentFile = ""
        CurrentLine = 0
        Paused = false
    }

    /// Global execution context
    let mutable globalContext = createContext()

    /// Global agent registry
    let mutable agentRegistry: IAgentRegistry option = None

    /// Set the agent registry
    let setAgentRegistry registry =
        agentRegistry <- Some registry

    /// Execute a block with the given environment
    let rec executeBlock (block: TarsBlock) (env: Environment) =
        // Update global context
        globalContext <- { globalContext with Environment = env }

        // Check if we should pause for debugging
        if globalContext.StepMode || globalContext.BreakPoints.Contains(globalContext.CurrentFile, globalContext.CurrentLine) then
            globalContext <- { globalContext with Paused = true }
            // In a real implementation, we would wait for user input here
            // For now, we'll just continue
            globalContext <- { globalContext with Paused = false }

        // Execute the current block based on its type
        match block.Type with
        | BlockType.Config ->
            // Add all properties to the environment
            let mutable newEnv = env
            for KeyValue(key, value) in block.Properties do
                newEnv <- newEnv.Add(key, value)

            // Update global context
            globalContext <- { globalContext with Environment = newEnv }

            // Execute nested blocks
            let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b newEnv)
            let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

            if nestedErrors.Length > 0 then
                Error(String.Join("; ", nestedErrors))
            else
                Success(StringValue("Config block executed"))

        | BlockType.Prompt ->
            // Get the prompt text from the properties
            match block.Properties.TryFind("text") with
            | Some(StringValue(text)) ->
                // Evaluate the text with variable interpolation
                let evaluatedText = Evaluator.evaluateString text env

                // In a real implementation, this would send the prompt to an AI model
                Success(StringValue($"Prompt executed: {evaluatedText}"))
            | _ ->
                Error("Prompt block must have a 'text' property")

        | BlockType.Variable ->
            // Get the variable name and value
            match block.Name, block.Properties.TryFind("value") with
            | Some(name), Some(value) ->
                // Evaluate the value with variable interpolation
                let evaluatedValue = Evaluator.evaluatePropertyValue value env
                // Store the variable in the environment
                let newEnv = env.Add(name, evaluatedValue)
                // Update global context
                globalContext <- { globalContext with Environment = newEnv }
                Success(StringValue($"Variable defined: {name}"))
            | None, _ ->
                Error("Variable block must have a name")
            | _, None ->
                Error("Variable block must have a 'value' property")

        | BlockType.If ->
            // Get the condition from the properties
            match block.Properties.TryFind("condition") with
            | Some(value) ->
                // Evaluate the condition
                let conditionStr =
                    match value with
                    | StringValue s -> s
                    | _ -> "false"

                let condition = Evaluator.evaluateBooleanExpression conditionStr env

                if condition then
                    // Execute the nested blocks if the condition is true
                    let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b env)
                    let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                    if nestedErrors.Length > 0 then
                        Error(String.Join("; ", nestedErrors))
                    else
                        Success(StringValue("If block executed"))
                else
                    // Skip the nested blocks if the condition is false
                    Success(StringValue("If block skipped"))
            | None ->
                Error("If block must have a 'condition' property")

        | BlockType.Else ->
            // Execute the nested blocks
            let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b env)
            let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

            if nestedErrors.Length > 0 then
                Error(String.Join("; ", nestedErrors))
            else
                Success(StringValue("Else block executed"))

        | BlockType.For ->
            // Get the loop variable and range
            match block.Properties.TryFind("variable"), block.Properties.TryFind("range") with
            | Some(StringValue(variable)), Some(value) ->
                // Evaluate the range
                let evaluatedRange = Evaluator.evaluatePropertyValue value env

                match evaluatedRange with
                | ListValue(range) ->
                    // Execute the nested blocks for each value in the range
                    let mutable forErrors = []

                    for value in range do
                        // Store the loop variable in the environment
                        let loopEnv = env.Add(variable, value)

                        // Execute the nested blocks
                        let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b loopEnv)
                        let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                        if nestedErrors.Length > 0 then
                            forErrors <- forErrors @ nestedErrors

                    if forErrors.Length > 0 then
                        Error(String.Join("; ", forErrors))
                    else
                        Success(StringValue("For block executed"))
                | _ ->
                    Error("For block 'range' property must evaluate to a list")
            | _ ->
                Error("For block must have 'variable' and 'range' properties")

        | BlockType.While ->
            // Get the condition from the properties
            match block.Properties.TryFind("condition") with
            | Some(value) ->
                // Evaluate the condition
                let conditionStr =
                    match value with
                    | StringValue s -> s
                    | _ -> "false"

                let mutable currentCondition = Evaluator.evaluateBooleanExpression conditionStr env
                let mutable whileErrors = []
                let mutable currentEnv = env

                while currentCondition && whileErrors.Length = 0 do
                    // Execute the nested blocks
                    let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b currentEnv)
                    let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                    if nestedErrors.Length > 0 then
                        whileErrors <- whileErrors @ nestedErrors
                    else
                        // Re-evaluate the condition
                        currentCondition <- Evaluator.evaluateBooleanExpression conditionStr currentEnv

                if whileErrors.Length > 0 then
                    Error(String.Join("; ", whileErrors))
                else
                    Success(StringValue("While block executed"))
            | None ->
                Error("While block must have a 'condition' property")

        | BlockType.Function ->
            // Get the function name
            match block.Name with
            | Some(name) ->
                // Store the function in the environment
                let functionValue = ObjectValue(Map.empty
                    .Add("type", StringValue("function"))
                    .Add("name", StringValue(name))
                    .Add("block", ObjectValue(Map.empty
                        .Add("type", StringValue(block.Type.ToString()))
                        .Add("name", match block.Name with Some n -> StringValue(n) | None -> StringValue(""))
                        .Add("content", StringValue(block.Content))
                        .Add("properties", ObjectValue(block.Properties))
                        .Add("nestedBlocks", ListValue([])))))

                let newEnv = env.Add(name, functionValue)
                // Update global context
                globalContext <- { globalContext with Environment = newEnv }
                Success(StringValue($"Function defined: {name}"))
            | None ->
                Error("Function block must have a name")

        | BlockType.Return ->
            // Get the return value from the properties
            match block.Properties.TryFind("value") with
            | Some(value) ->
                // Evaluate the value with variable interpolation
                let evaluatedValue = Evaluator.evaluatePropertyValue value env

                // If we're in a function, update the stack frame
                match globalContext.CallStack with
                | frame :: rest ->
                    let newFrame = { frame with ReturnValue = Some evaluatedValue }
                    globalContext <- { globalContext with CallStack = newFrame :: rest }
                | [] -> ()

                Success(evaluatedValue)
            | None ->
                Error("Return block must have a 'value' property")

        | BlockType.Import ->
            // Get the module name from the properties
            match block.Properties.TryFind("module") with
            | Some(StringValue(moduleName)) ->
                // In a real implementation, this would import a module
                Success(StringValue($"Module imported: {moduleName}"))
            | _ ->
                Error("Import block must have a 'module' property")

        | BlockType.Export ->
            // Get the export name from the properties
            match block.Properties.TryFind("name") with
            | Some(StringValue(exportName)) ->
                // In a real implementation, this would export a value
                Success(StringValue($"Value exported: {exportName}"))
            | _ ->
                Error("Export block must have a 'name' property")

        | BlockType.Agent ->
            // Check if agent registry is available
            match agentRegistry with
            | Some registry ->
                // In a real implementation, this would create an agent from the block
                // For now, we'll just return a success result
                registry.RegisterAgent(block)
                Success(StringValue($"Agent registered"))
            | None ->
                Error("Agent registry not available")

        | BlockType.Task ->
            // Tasks are handled as part of agent creation
            Success(StringValue("Task defined"))

        | BlockType.Action ->
            // Get the action type from the properties
            match block.Properties.TryFind("type") with
            | Some(StringValue("execute")) ->
                // Execute an agent task
                match block.Properties.TryFind("agent"), block.Properties.TryFind("task") with
                | Some(StringValue(agentName)), Some(StringValue(taskName)) ->
                    let functionName =
                        match block.Properties.TryFind("function") with
                        | Some(StringValue(name)) -> Some name
                        | _ -> None

                    // Get parameters
                    let parameters =
                        match block.Properties.TryFind("parameters") with
                        | Some(ObjectValue(paramMap)) -> paramMap
                        | _ -> Map.empty

                    // Check if agent registry is available
                    match agentRegistry with
                    | Some registry ->
                        // Execute the task
                        match registry.ExecuteTask(agentName, taskName, functionName, parameters, env) with
                        | AgentResult.Success result ->
                            // Store the result in the output variable if specified
                            match block.Properties.TryFind("output_variable") with
                            | Some(StringValue(varName)) ->
                                let newEnv = env.Add(varName, result)
                                // Update global context
                                globalContext <- { globalContext with Environment = newEnv }
                            | _ -> ()

                            Success(result)
                        | AgentResult.Error msg -> Error(msg)
                    | None -> Error("Agent registry not available")
                | _ ->
                    Error("Action block with type 'execute' must have 'agent' and 'task' properties")
            | Some(StringValue(actionType)) ->
                // In a real implementation, this would execute the action
                Success(StringValue($"Action executed: {actionType}"))
            | _ ->
                Error("Action block must have a 'type' property")

        | BlockType.Describe ->
            // Process description block
            let name = block.Properties.TryFind("name") |> Option.map (function StringValue s -> s | _ -> "")
            let version = block.Properties.TryFind("version") |> Option.map (function StringValue s -> s | _ -> "")
            let description = block.Properties.TryFind("description") |> Option.map (function StringValue s -> s | _ -> "")

            let nameStr = name |> Option.defaultValue ""
            let versionStr = version |> Option.defaultValue ""
            Success(StringValue($"Description: {nameStr} v{versionStr}"))

        | BlockType.SpawnAgent ->
            // Get agent ID and type
            match block.Properties.TryFind("id"), block.Properties.TryFind("type") with
            | Some(StringValue(id)), Some(StringValue(agentType)) ->
                // In a real implementation, this would spawn a new agent
                Success(StringValue($"Agent spawned: {id} of type {agentType}"))
            | _ ->
                Error("SpawnAgent block must have 'id' and 'type' properties")

        | BlockType.Message ->
            // Get agent and message
            match block.Properties.TryFind("agent"), block.Properties.TryFind("text") with
            | Some(StringValue(agent)), Some(StringValue(text)) ->
                // In a real implementation, this would send a message to the agent
                Success(StringValue($"Message sent to {agent}: {text}"))
            | _ ->
                Error("Message block must have 'agent' and 'text' properties")

        | BlockType.SelfImprove ->
            // Get agent and instructions
            match block.Properties.TryFind("agent"), block.Properties.TryFind("instructions") with
            | Some(StringValue(agent)), Some(StringValue(instructions)) ->
                // In a real implementation, this would trigger self-improvement
                Success(StringValue($"Self-improvement triggered for {agent}: {instructions}"))
            | _ ->
                Error("SelfImprove block must have 'agent' and 'instructions' properties")

        | BlockType.Tars ->
            // Execute TARS block (container for other blocks)
            let nestedResults = block.NestedBlocks |> List.map (fun b -> executeBlock b env)
            let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

            if nestedErrors.Length > 0 then
                Error(String.Join("; ", nestedErrors))
            else
                Success(StringValue("TARS block executed"))

        | BlockType.Communication ->
            // Process communication configuration
            match block.Properties.TryFind("protocol"), block.Properties.TryFind("endpoint") with
            | Some(StringValue(protocol)), Some(StringValue(endpoint)) ->
                Success(StringValue($"Communication configured: {protocol} at {endpoint}"))
            | _ ->
                Error("Communication block must have 'protocol' and 'endpoint' properties")

        | BlockType.AutoImprove ->
            // In a real implementation, this would trigger auto-improvement
            Success(StringValue("Auto-improvement triggered"))

        | BlockType.Unknown(blockType) ->
            Error($"Unknown block type: {blockType}")

        | _ ->
            // These block types are handled elsewhere or not yet implemented
            Success(StringValue($"Block type {block.Type} executed"))

    /// Execute a TARS program
    let execute (program: TarsProgram) =
        // Initialize the environment
        let env = Map.empty<string, PropertyValue>

        // Reset the global context
        globalContext <- createContext()

        // Execute each block and collect the results
        let results =
            program.Blocks
            |> List.map (fun block -> executeBlock block env)

        // Check if any block execution resulted in an error
        let errors =
            results
            |> List.choose (function
                | Error(msg) -> Some(msg)
                | _ -> None)

        if errors.Length > 0 then
            Error(String.Join("; ", errors))
        else
            // Return the result of the last block
            match List.tryLast results with
            | Some(result) -> result
            | None -> Success(StringValue("Program executed successfully"))

    /// Execute a TARS program with debugging
    let executeWithDebugging (program: TarsProgram) (breakPoints: Set<string * int>) (stepMode: bool) =
        // Initialize the environment
        let env = Map.empty<string, PropertyValue>

        // Reset the global context
        globalContext <- createContext()

        // Set debugging options
        globalContext <- { globalContext with BreakPoints = breakPoints; StepMode = stepMode }

        // Execute the program
        execute program
