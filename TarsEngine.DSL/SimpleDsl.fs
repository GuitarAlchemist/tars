namespace TarsEngine.DSL

open System
open System.Text.RegularExpressions
open System.Collections.Generic

/// Module containing a simplified implementation of the TARS DSL
module SimpleDsl =
    /// Block types in the DSL
    type BlockType =
        | Describe
        | Config
        | Prompt
        | Action
        | Variable
        | If
        | Else
        | Unknown of string

    /// Property value types in the DSL
    type PropertyValue =
        | StringValue of string
        | NumberValue of float
        | BoolValue of bool
        | ListValue of PropertyValue list
        | ObjectValue of Map<string, PropertyValue>

    /// A block in the DSL
    type Block = {
        Type: BlockType
        Name: string option
        Content: string
        Properties: Map<string, PropertyValue>
        NestedBlocks: Block list
    }

    /// A TARS program consisting of blocks
    type Program = {
        Blocks: Block list
    }

    /// Result of executing a block
    type ExecutionResult =
        | Success of PropertyValue
        | Error of string

    /// Parse a string into a block type
    let parseBlockType (blockType: string) =
        match blockType.ToUpper() with
        | "DESCRIBE" -> BlockType.Describe
        | "CONFIG" -> BlockType.Config
        | "PROMPT" -> BlockType.Prompt
        | "ACTION" -> BlockType.Action
        | "VARIABLE" -> BlockType.Variable
        | "IF" -> BlockType.If
        | "ELSE" -> BlockType.Else
        | _ -> BlockType.Unknown blockType

    /// Parse a property value
    let rec parsePropertyValue (value: string) =
        // Try to parse as number
        match Double.TryParse(value) with
        | true, num -> NumberValue(num)
        | _ ->
            // Try to parse as boolean
            match Boolean.TryParse(value) with
            | true, b -> BoolValue(b)
            | _ ->
                // Parse as string - remove quotes if present
                if value.StartsWith('"') && value.EndsWith('"') && value.Length >= 2 then
                    StringValue(value.Substring(1, value.Length - 2))
                else
                    StringValue(value)

    /// Parse properties from a string
    let parseProperties (propertiesText: string) =
        let properties = Map.empty<string, PropertyValue>

        // Match property lines like "name: value"
        let propertyRegex = Regex(@"^\s*(\w+)\s*:\s*(.+)$", RegexOptions.Multiline)
        let matches = propertyRegex.Matches(propertiesText)

        // Add each property to the map
        matches
        |> Seq.cast<Match>
        |> Seq.fold (fun props m ->
            let name = m.Groups.[1].Value
            let value = m.Groups.[2].Value.Trim()
            Map.add name (parsePropertyValue value) props
        ) properties

    /// Parse block content into properties and nested blocks
    let rec parseBlockContent (content: string) =
        // Find all nested blocks
        let blockRegex = Regex(@"^\s*(\w+)(?:\s+(\w+))?\s*\{", RegexOptions.Multiline)
        let blockMatches = blockRegex.Matches(content)

        let mutable properties = Map.empty<string, PropertyValue>
        let mutable nestedBlocks = []

        if blockMatches.Count > 0 then
            // There are nested blocks
            let mutable lastIndex = 0

            for m in blockMatches do
                // Parse properties before this block
                let propertiesText = content.Substring(lastIndex, m.Index - lastIndex)
                // Merge the properties
                for KeyValue(key, value) in parseProperties propertiesText do
                    properties <- Map.add key value properties

                // Parse the nested block
                let blockText = content.Substring(m.Index)
                let block = parseBlock blockText
                nestedBlocks <- nestedBlocks @ [block]

                // Update the last index
                lastIndex <- m.Index + block.Content.Length + 2 // +2 for the braces
        else
            // No nested blocks, just properties
            properties <- parseProperties content

        (properties, nestedBlocks)

    /// Parse a block from a string
    and parseBlock (blockText: string) =
        // Match block header like "BLOCKTYPE name {"
        let headerRegex = Regex(@"^\s*(\w+)(?:\s+(\w+))?\s*\{", RegexOptions.Multiline)
        let headerMatch = headerRegex.Match(blockText)

        if headerMatch.Success then
            let blockType = parseBlockType headerMatch.Groups.[1].Value
            let name =
                if headerMatch.Groups.[2].Success then
                    Some headerMatch.Groups.[2].Value
                else
                    None

            // Find the end of the block
            let startIndex = headerMatch.Index + headerMatch.Length
            let endIndex = findMatchingBrace blockText startIndex

            if endIndex > startIndex then
                let blockContent = blockText.Substring(startIndex, endIndex - startIndex)

                // Parse properties and nested blocks
                let (properties, nestedBlocks) = parseBlockContent blockContent

                // Create the block
                {
                    Type = blockType
                    Name = name
                    Content = blockContent
                    Properties = properties
                    NestedBlocks = nestedBlocks
                }
            else
                // Invalid block
                {
                    Type = BlockType.Unknown "Invalid"
                    Name = None
                    Content = ""
                    Properties = Map.empty
                    NestedBlocks = []
                }
        else
            // Invalid block
            {
                Type = BlockType.Unknown "Invalid"
                Name = None
                Content = ""
                Properties = Map.empty
                NestedBlocks = []
            }

    /// Find the matching closing brace
    and findMatchingBrace (text: string) (startIndex: int) =
        let mutable braceCount = 1
        let mutable index = startIndex

        while braceCount > 0 && index < text.Length do
            match text.[index] with
            | '{' -> braceCount <- braceCount + 1
            | '}' -> braceCount <- braceCount - 1
            | _ -> ()

            index <- index + 1

        index - 1

    // Parse block content is defined above

    /// Parse a program from a string
    let parseProgram (programText: string) =
        // Find all top-level blocks
        let blockRegex = Regex(@"^\s*(\w+)(?:\s+(\w+))?\s*\{", RegexOptions.Multiline)
        let blockMatches = blockRegex.Matches(programText)

        let mutable blocks = []

        for m in blockMatches do
            // Parse the block
            let blockText = programText.Substring(m.Index)
            let block = parseBlock blockText
            blocks <- blocks @ [block]

        { Blocks = blocks }

    /// Substitute variables in a string
    let rec substituteVariables (text: string) (environment: Dictionary<string, PropertyValue>) =
        let variableRegex = Regex(@"\$\{([^}]+)\}")

        variableRegex.Replace(text, fun m ->
            let variableName = m.Groups.[1].Value

            if environment.ContainsKey(variableName) then
                match environment.[variableName] with
                | StringValue s -> s
                | NumberValue n -> n.ToString()
                | BoolValue b -> b.ToString()
                | _ -> m.Value // Keep the original for complex types
            else
                m.Value // Keep the original if variable not found
        )

    /// Evaluate a condition
    and evaluateCondition (condition: string) =
        // Simple condition evaluation
        // In a real implementation, this would be more sophisticated

        // Check for equality
        if condition.Contains("==") then
            let parts = condition.Split([|"=="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = parts.[0].Trim()
                let right = parts.[1].Trim()
                left = right
            else
                false
        // Check for inequality
        elif condition.Contains("!=") then
            let parts = condition.Split([|"!="|], StringSplitOptions.None)
            if parts.Length = 2 then
                let left = parts.[0].Trim()
                let right = parts.[1].Trim()
                left <> right
            else
                false
        // Check for boolean value
        elif condition.ToLower() = "true" then
            true
        elif condition.ToLower() = "false" then
            false
        else
            // Default to false for unknown conditions
            false

    /// Execute a block
    and executeBlock (block: Block) (environment: Dictionary<string, PropertyValue>) =
        match block.Type with
        | BlockType.Describe ->
            // Just return success
            Success(StringValue("Description processed"))

        | BlockType.Config ->
            // Store configuration in environment
            for KeyValue(key, value) in block.Properties do
                environment.[key] <- value

            Success(StringValue("Configuration processed"))

        | BlockType.Variable ->
            // Get the variable name
            match block.Name with
            | Some name ->
                // Get the value from properties
                match block.Properties.TryFind("value") with
                | Some value ->
                    // Store in environment
                    environment.[name] <- value
                    Success(value)
                | None ->
                    Error($"Variable '{name}' has no value")
            | None ->
                Error("Variable block has no name")

        | BlockType.Prompt ->
            // Get the prompt text
            match block.Properties.TryFind("text") with
            | Some (StringValue text) ->
                // Substitute variables
                let substitutedText = substituteVariables text environment

                // In a real implementation, this would send the prompt to an LLM
                // For now, just return the prompt text
                Success(StringValue(substitutedText))
            | _ ->
                Error("Prompt block has no text property")

        | BlockType.Action ->
            // Get the action type
            match block.Properties.TryFind("type") with
            | Some (StringValue actionType) ->
                // Debug output
                printfn "Action type: '%s'" actionType

                // Remove quotes if present
                let cleanActionType =
                    if actionType.StartsWith('"') && actionType.EndsWith('"') && actionType.Length >= 2 then
                        actionType.Substring(1, actionType.Length - 2)
                    else
                        actionType

                printfn "Clean action type: '%s'" cleanActionType

                match cleanActionType.ToLower().Trim() with
                | "log" ->
                    // Get the message
                    match block.Properties.TryFind("message") with
                    | Some (StringValue message) ->
                        // Substitute variables
                        let substitutedMessage = substituteVariables message environment

                        // Log the message
                        printfn "%s" substitutedMessage

                        Success(StringValue(substitutedMessage))
                    | _ ->
                        Error("Log action has no message property")

                | "mcp_send" ->
                    // Get the required parameters
                    let targetOpt = block.Properties.TryFind("target")
                    let actionOpt = block.Properties.TryFind("action")
                    let parametersOpt = block.Properties.TryFind("parameters")
                    let resultVarOpt = block.Properties.TryFind("result_variable")

                    match targetOpt, actionOpt with
                    | Some (StringValue target), Some (StringValue action) ->
                        // Substitute variables in target and action
                        let substitutedTarget = substituteVariables target environment
                        let substitutedAction = substituteVariables action environment

                        // Get parameters as object if available
                        let parameters =
                            match parametersOpt with
                            | Some (ObjectValue paramMap) -> paramMap
                            | _ -> Map.empty

                        // Substitute variables in parameters
                        let substitutedParams =
                            parameters |> Map.map (fun _ value ->
                                match value with
                                | StringValue s -> StringValue(substituteVariables s environment)
                                | _ -> value
                            )

                        // In a real implementation, this would send an MCP request
                        // For now, just return a mock response
                        let mockResponse = $"MCP request to {substitutedTarget}, action: {substitutedAction}, parameters: {substitutedParams.Count} parameters"
                        printfn "[MCP] %s" mockResponse

                        // Store the result in the specified variable if provided
                        match resultVarOpt with
                        | Some (StringValue resultVar) ->
                            environment.[resultVar] <- StringValue(mockResponse)
                        | _ -> ()

                        Success(StringValue(mockResponse))
                    | _ ->
                        Error("MCP action requires 'target' and 'action' properties")

                | "mcp_receive" ->
                    // Get the timeout parameter
                    let timeoutOpt = block.Properties.TryFind("timeout")
                    let resultVarOpt = block.Properties.TryFind("result_variable")

                    // Get timeout value or use default
                    let timeout =
                        match timeoutOpt with
                        | Some (NumberValue t) -> t
                        | _ -> 30.0 // Default timeout in seconds

                    // In a real implementation, this would wait for an MCP request
                    // For now, just return a mock response
                    let mockResponse = $"Received MCP request (timeout: {timeout}s)"
                    printfn "[MCP] %s" mockResponse

                    // Store the result in the specified variable if provided
                    match resultVarOpt with
                    | Some (StringValue resultVar) ->
                        environment.[resultVar] <- StringValue(mockResponse)
                    | _ -> ()

                    Success(StringValue(mockResponse))

                | _ ->
                    Error($"Unknown action type: '{actionType}'")


            | Some other ->
                // Debug output for non-string values
                printfn "Action type is not a string: %A" other
                Error("Action type must be a string")

            | _ ->
                Error("Action block has no type property")

        | BlockType.If ->
            // Get the condition
            match block.Properties.TryFind("condition") with
            | Some (StringValue condition) ->
                // Substitute variables
                let substitutedCondition = substituteVariables condition environment

                // Evaluate the condition
                let conditionResult = evaluateCondition substitutedCondition

                if conditionResult then
                    // Execute the nested blocks
                    let mutable result = Success(StringValue(""))
                    let mutable continueExecution = true

                    for nestedBlock in block.NestedBlocks do
                        if continueExecution then
                            match executeBlock nestedBlock environment with
                            | Success value -> result <- Success value
                            | Error msg ->
                                result <- Error msg
                                // Stop execution on error
                                continueExecution <- false

                    result
                else
                    // Find the ELSE block
                    let elseBlock =
                        block.NestedBlocks
                        |> List.tryFind (fun b -> b.Type = BlockType.Else)

                    match elseBlock with
                    | Some eb ->
                        // Execute the ELSE block
                        executeBlock eb environment
                    | None ->
                        // No ELSE block, just return success
                        Success(StringValue(""))
            | _ ->
                Error("If block has no condition property")

        | BlockType.Else ->
            // Execute the nested blocks
            let mutable result = Success(StringValue(""))
            let mutable continueExecution = true

            for nestedBlock in block.NestedBlocks do
                if continueExecution then
                    match executeBlock nestedBlock environment with
                    | Success value -> result <- Success value
                    | Error msg ->
                        result <- Error msg
                        // Stop execution on error
                        continueExecution <- false

            result

        | BlockType.Unknown blockType ->
            Error($"Unknown block type: {blockType}")

    /// Execute a program
    and executeProgram (program: Program) =
        // Create an environment for variable storage
        let environment = Dictionary<string, PropertyValue>()

        // Execute each block
        let mutable result = Success(StringValue(""))
        let mutable continueExecution = true

        for block in program.Blocks do
            if continueExecution then
                match executeBlock block environment with
                | Success value -> result <- Success value
                | Error msg ->
                    result <- Error msg
                    // Stop execution on error
                    continueExecution <- false

        result
