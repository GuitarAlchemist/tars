namespace TarsEngine.DSL

open Ast
open System

/// Module containing the interpreter for the TARS DSL
module Interpreter =
    /// Environment for storing variables during execution
    type Environment = Map<string, PropertyValue>

    /// Result of executing a TARS program
    type ExecutionResult =
        | Success of PropertyValue
        | Error of string

    /// Execute a TARS program
    let execute (program: TarsProgram) =
        // Initialize the environment
        let mutable env = Map.empty<string, PropertyValue>
        let mutable agents = Map.empty<string, TarsBlock>

        // Execute a block and its nested blocks
        let rec executeBlock (block: TarsBlock) =
            // Execute nested blocks first if needed
            let nestedResults =
                block.NestedBlocks
                |> List.map executeBlock

            // Check if any nested block execution resulted in an error
            let nestedErrors =
                nestedResults
                |> List.choose (function
                    | Error(msg) -> Some(msg)
                    | _ -> None)

            if nestedErrors.Length > 0 then
                Error(String.Join("; ", nestedErrors))
            else
                // Execute the current block
                match block.Type with
                | BlockType.Config ->
                    // Add all properties to the environment
                    for KeyValue(key, value) in block.Properties do
                        env <- env.Add(key, value)
                    Success(StringValue("Config block executed"))

                | BlockType.Prompt ->
                    // Get the prompt text from the properties
                    match block.Properties.TryFind("text") with
                    | Some(StringValue(text)) ->
                        // In a real implementation, this would send the prompt to an AI model
                        Success(StringValue($"Prompt executed: {text}"))
                    | _ ->
                        Error("Prompt block must have a 'text' property")

                | BlockType.Action ->
                    // Get the action type from the properties
                    match block.Properties.TryFind("type") with
                    | Some(StringValue(actionType)) ->
                        // In a real implementation, this would execute the action
                        Success(StringValue($"Action executed: {actionType}"))
                    | _ ->
                        Error("Action block must have a 'type' property")

                | BlockType.Task ->
                    // Get the task description from the properties
                    match block.Properties.TryFind("description") with
                    | Some(StringValue(description)) ->
                        // In a real implementation, this would execute the task
                        Success(StringValue($"Task executed: {description}"))
                    | _ ->
                        Error("Task block must have a 'description' property")

                | BlockType.Agent ->
                    // Get the agent name
                    let agentName =
                        match block.Name, block.Properties.TryFind("name") with
                        | Some(name), _ -> name
                        | _, Some(StringValue(name)) -> name
                        | _ -> "unnamed_agent_" + Guid.NewGuid().ToString("N")

                    // Store the agent for later use
                    agents <- agents.Add(agentName, block)

                    // In a real implementation, this would create an agent
                    Success(StringValue($"Agent created: {agentName}"))

                | BlockType.AutoImprove ->
                    // Get the target from the properties
                    match block.Properties.TryFind("target") with
                    | Some(StringValue(target)) ->
                        // In a real implementation, this would trigger auto-improvement
                        Success(StringValue($"Auto-improvement triggered for: {target}"))
                    | _ ->
                        Error("AutoImprove block must have a 'target' property")

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
                    Success(StringValue("TARS block executed"))

                | BlockType.Communication ->
                    // Process communication configuration
                    match block.Properties.TryFind("protocol"), block.Properties.TryFind("endpoint") with
                    | Some(StringValue(protocol)), Some(StringValue(endpoint)) ->
                        Success(StringValue($"Communication configured: {protocol} at {endpoint}"))
                    | _ ->
                        Error("Communication block must have 'protocol' and 'endpoint' properties")

                | BlockType.Variable ->
                    // Get the variable name and value
                    match block.Name, block.Properties.TryFind("value") with
                    | Some(name), Some(value) ->
                        // Store the variable in the environment
                        env <- env.Add(name, value)
                        Success(StringValue($"Variable defined: {name}"))
                    | None, _ ->
                        Error("Variable block must have a name")
                    | _, None ->
                        Error("Variable block must have a 'value' property")

                | BlockType.If ->
                    // Get the condition from the properties
                    match block.Properties.TryFind("condition") with
                    | Some(BoolValue(true)) ->
                        // Execute the nested blocks if the condition is true
                        let nestedResults = block.NestedBlocks |> List.map executeBlock
                        let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                        if nestedErrors.Length > 0 then
                            Error(String.Join("; ", nestedErrors))
                        else
                            Success(StringValue("If block executed"))
                    | Some(BoolValue(false)) ->
                        // Skip the nested blocks if the condition is false
                        Success(StringValue("If block skipped"))
                    | _ ->
                        Error("If block must have a 'condition' property with a boolean value")

                | BlockType.Else ->
                    // Execute the nested blocks
                    let nestedResults = block.NestedBlocks |> List.map executeBlock
                    let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                    if nestedErrors.Length > 0 then
                        Error(String.Join("; ", nestedErrors))
                    else
                        Success(StringValue("Else block executed"))

                | BlockType.For ->
                    // Get the loop variable and range
                    match block.Properties.TryFind("variable"), block.Properties.TryFind("range") with
                    | Some(StringValue(variable)), Some(ListValue(range)) ->
                        // Execute the nested blocks for each value in the range
                        let mutable forErrors = []

                        for value in range do
                            // Store the loop variable in the environment
                            env <- env.Add(variable, value)

                            // Execute the nested blocks
                            let nestedResults = block.NestedBlocks |> List.map executeBlock
                            let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                            if nestedErrors.Length > 0 then
                                forErrors <- forErrors @ nestedErrors

                        if forErrors.Length > 0 then
                            Error(String.Join("; ", forErrors))
                        else
                            Success(StringValue("For block executed"))
                    | _ ->
                        Error("For block must have 'variable' and 'range' properties")

                | BlockType.While ->
                    // Get the condition from the properties
                    match block.Properties.TryFind("condition") with
                    | Some(BoolValue(condition)) ->
                        // Execute the nested blocks while the condition is true
                        let mutable whileErrors = []
                        let mutable currentCondition = condition

                        while currentCondition do
                            // Execute the nested blocks
                            let nestedResults = block.NestedBlocks |> List.map executeBlock
                            let nestedErrors = nestedResults |> List.choose (function Error(msg) -> Some(msg) | _ -> None)

                            if nestedErrors.Length > 0 then
                                whileErrors <- whileErrors @ nestedErrors
                                currentCondition <- false
                            else
                                // Re-evaluate the condition
                                match block.Properties.TryFind("condition") with
                                | Some(BoolValue(newCondition)) -> currentCondition <- newCondition
                                | _ -> currentCondition <- false

                        if whileErrors.Length > 0 then
                            Error(String.Join("; ", whileErrors))
                        else
                            Success(StringValue("While block executed"))
                    | _ ->
                        Error("While block must have a 'condition' property with a boolean value")

                | BlockType.Function ->
                    // Get the function name
                    match block.Name with
                    | Some(name) ->
                        // Store the function in the environment
                        env <- env.Add(name, ObjectValue(Map.empty.Add("type", StringValue("function")).Add("block", ObjectValue(Map.empty.Add("type", StringValue(block.Type.ToString())).Add("name", StringValue(name)).Add("content", StringValue(block.Content)).Add("properties", ObjectValue(block.Properties)).Add("nestedBlocks", ListValue([])))))) // Simplified for now
                        Success(StringValue($"Function defined: {name}"))
                    | None ->
                        Error("Function block must have a name")

                | BlockType.Return ->
                    // Get the return value from the properties
                    match block.Properties.TryFind("value") with
                    | Some(value) ->
                        Success(value)
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

                | BlockType.Unknown(blockType) ->
                    Error($"Unknown block type: {blockType}")

        // Execute each block and collect the results
        let results =
            program.Blocks
            |> List.map executeBlock

        // Check if any block execution resulted in an error
        let errors =
            results
            |> List.choose (function
                | Error(msg) -> Some(msg)
                | _ -> None)

        if errors.Length > 0 then
            Error(System.String.Join("; ", errors))
        else
            Success(StringValue("Program executed successfully"))

    /// Execute a TARS program from a file
    let executeFile (filePath: string) =
        let program = Parser.parseFile filePath
        execute program
