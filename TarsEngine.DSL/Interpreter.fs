namespace TarsEngine.DSL

open Ast

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
        
        // Execute each block in the program
        let executeBlock (block: TarsBlock) =
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
                // Get the agent name from the properties
                match block.Properties.TryFind("name") with
                | Some(StringValue(name)) ->
                    // In a real implementation, this would create an agent
                    Success(StringValue($"Agent created: {name}"))
                | _ ->
                    Error("Agent block must have a 'name' property")
                    
            | BlockType.AutoImprove ->
                // Get the target from the properties
                match block.Properties.TryFind("target") with
                | Some(StringValue(target)) ->
                    // In a real implementation, this would trigger auto-improvement
                    Success(StringValue($"Auto-improvement triggered for: {target}"))
                | _ ->
                    Error("AutoImprove block must have a 'target' property")
                    
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
