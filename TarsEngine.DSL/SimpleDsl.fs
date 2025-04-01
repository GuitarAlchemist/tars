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
        | While
        | For
        | Function
        | Call
        | Try
        | Catch
        | Return
        | FSharp
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

    /// A function definition
    type FunctionDef = {
        Name: string
        Parameters: string list
        Body: Block list
    }

    /// Global function registry
    let mutable functionRegistry = Map.empty<string, FunctionDef>

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
        | "WHILE" -> BlockType.While
        | "FOR" -> BlockType.For
        | "FUNCTION" -> BlockType.Function
        | "CALL" -> BlockType.Call
        | "TRY" -> BlockType.Try
        | "CATCH" -> BlockType.Catch
        | "RETURN" -> BlockType.Return
        | "FSHARP" -> BlockType.FSharp
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

                | "file_read" ->
                    // Get the required parameters
                    let pathOpt = block.Properties.TryFind("path")
                    let resultVarOpt = block.Properties.TryFind("result_variable")

                    match pathOpt, resultVarOpt with
                    | Some (StringValue path), Some (StringValue resultVar) ->
                        // Substitute variables in path
                        let substitutedPath = substituteVariables path environment

                        try
                            // Read the file
                            let content = System.IO.File.ReadAllText(substitutedPath)

                            // Store the result in the result variable
                            environment.[resultVar] <- StringValue(content)

                            Success(StringValue($"File read successfully: {substitutedPath}"))
                        with
                        | ex -> Error($"Error reading file {substitutedPath}: {ex.Message}")
                    | _ ->
                        Error("File read action requires 'path' and 'result_variable' properties")

                | "file_write" ->
                    // Get the required parameters
                    let pathOpt = block.Properties.TryFind("path")
                    let contentOpt = block.Properties.TryFind("content")

                    match pathOpt, contentOpt with
                    | Some (StringValue path), Some (StringValue content) ->
                        // Substitute variables in path and content
                        let substitutedPath = substituteVariables path environment
                        let substitutedContent = substituteVariables content environment

                        try
                            // Write the file
                            System.IO.File.WriteAllText(substitutedPath, substitutedContent)

                            Success(StringValue($"File written successfully: {substitutedPath}"))
                        with
                        | ex -> Error($"Error writing file {substitutedPath}: {ex.Message}")
                    | _ ->
                        Error("File write action requires 'path' and 'content' properties")

                | "http_request" ->
                    // Get the required parameters
                    let urlOpt = block.Properties.TryFind("url")
                    let methodOpt = block.Properties.TryFind("method")
                    let headersOpt = block.Properties.TryFind("headers")
                    let bodyOpt = block.Properties.TryFind("body")
                    let resultVarOpt = block.Properties.TryFind("result_variable")

                    match urlOpt, resultVarOpt with
                    | Some (StringValue url), Some (StringValue resultVar) ->
                        // Substitute variables in url
                        let substitutedUrl = substituteVariables url environment

                        // Get method (default to GET)
                        let method =
                            match methodOpt with
                            | Some (StringValue m) -> substituteVariables m environment
                            | _ -> "GET"

                        // Get headers
                        let headers =
                            match headersOpt with
                            | Some (ObjectValue headerMap) ->
                                headerMap |> Map.map (fun _ value ->
                                    match value with
                                    | StringValue s -> substituteVariables s environment
                                    | _ -> "")
                            | _ -> Map.empty

                        // Get body
                        let body =
                            match bodyOpt with
                            | Some (StringValue b) -> substituteVariables b environment
                            | _ -> ""

                        try
                            // Create the request
                            let request = System.Net.WebRequest.Create(substitutedUrl) :?> System.Net.HttpWebRequest
                            request.Method <- method

                            // Add headers
                            for KeyValue(key, value) in headers do
                                request.Headers.Add(key, value)

                            // Add body for POST, PUT, etc.
                            if method <> "GET" && method <> "HEAD" && body <> "" then
                                let bytes = System.Text.Encoding.UTF8.GetBytes(body)
                                request.ContentLength <- int64 bytes.Length
                                use stream = request.GetRequestStream()
                                stream.Write(bytes, 0, bytes.Length)

                            // Get the response
                            use response = request.GetResponse()
                            use stream = response.GetResponseStream()
                            use reader = new System.IO.StreamReader(stream)
                            let responseText = reader.ReadToEnd()

                            // Store the result in the result variable
                            environment.[resultVar] <- StringValue(responseText)

                            Success(StringValue($"HTTP request successful: {substitutedUrl}"))
                        with
                        | ex -> Error($"Error making HTTP request to {substitutedUrl}: {ex.Message}")
                    | _ ->
                        Error("HTTP request action requires 'url' and 'result_variable' properties")

                | "shell_execute" ->
                    // Get the required parameters
                    let commandOpt = block.Properties.TryFind("command")
                    let resultVarOpt = block.Properties.TryFind("result_variable")

                    match commandOpt with
                    | Some (StringValue command) ->
                        // Substitute variables in command
                        let substitutedCommand = substituteVariables command environment

                        try
                            // Create the process
                            let processInfo = new System.Diagnostics.ProcessStartInfo()
                            processInfo.FileName <- "cmd.exe"
                            processInfo.Arguments <- $"/c {substitutedCommand}"
                            processInfo.RedirectStandardOutput <- true
                            processInfo.RedirectStandardError <- true
                            processInfo.UseShellExecute <- false
                            processInfo.CreateNoWindow <- true

                            // Start the process
                            use process = System.Diagnostics.Process.Start(processInfo)
                            let output = process.StandardOutput.ReadToEnd()
                            let error = process.StandardError.ReadToEnd()
                            process.WaitForExit()

                            // Store the result in the result variable if provided
                            match resultVarOpt with
                            | Some (StringValue resultVar) ->
                                environment.[resultVar] <- StringValue(output)
                            | _ -> ()

                            // Check if the process exited successfully
                            if process.ExitCode = 0 then
                                Success(StringValue($"Command executed successfully: {substitutedCommand}"))
                            else
                                Error($"Command failed with exit code {process.ExitCode}: {error}")
                        with
                        | ex -> Error($"Error executing command {substitutedCommand}: {ex.Message}")
                    | _ ->
                        Error("Shell execute action requires 'command' property")

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

        | BlockType.While ->
            // Get the condition
            match block.Properties.TryFind("condition") with
            | Some (StringValue condition) ->
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

                        for nestedBlock in block.NestedBlocks do
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
            | _ ->
                Error("While block has no condition property")

        | BlockType.For ->
            // Get the required properties
            let variableOpt = block.Properties.TryFind("variable")
            let fromOpt = block.Properties.TryFind("from")
            let toOpt = block.Properties.TryFind("to")
            let stepOpt = block.Properties.TryFind("step")

            match variableOpt, fromOpt, toOpt with
            | Some (StringValue variable), Some fromValue, Some toValue ->
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
                    match stepOpt with
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

                        for nestedBlock in block.NestedBlocks do
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
            | _ ->
                Error("For block requires 'variable', 'from', and 'to' properties")

        | BlockType.Function ->
            // Functions are registered in the first pass of executeProgram
            // Just return success here
            Success(StringValue("Function defined"))

        | BlockType.Call ->
            // Get the function name
            match block.Properties.TryFind("function") with
            | Some (StringValue functionName) ->
                // Get arguments
                let args =
                    match block.Properties.TryFind("arguments") with
                    | Some (ObjectValue argMap) -> argMap
                    | _ -> Map.empty

                // Look up the function in the registry
                let result =
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
                        let mutable funcResult = Success(StringValue(""))
                        let mutable continueExecution = true

                        for nestedBlock in functionDef.Body do
                            if continueExecution then
                                match executeBlock nestedBlock functionEnv with
                                | Success value -> funcResult <- Success value
                                | Error msg ->
                                    funcResult <- Error msg
                                    // Stop execution on error
                                    continueExecution <- false

                        funcResult
                    | None ->
                        Error($"Function '{functionName}' not found")

                // Store the result in the result_variable if provided
                match result, block.Properties.TryFind("result_variable") with
                | Success value, Some (StringValue resultVar) ->
                    environment.[resultVar] <- value
                | _ -> ()

                result
            | _ ->
                Error("Call block has no function property")

        | BlockType.Try ->
            // Execute the nested blocks with error handling
            let mutable result = Success(StringValue(""))
            let mutable errorOccurred = false
            let mutable errorMessage = ""

            // Execute the try block
            for nestedBlock in block.NestedBlocks do
                if not errorOccurred && nestedBlock.Type <> BlockType.Catch then
                    match executeBlock nestedBlock environment with
                    | Success value -> result <- Success value
                    | Error msg ->
                        errorOccurred <- true
                        errorMessage <- msg

            if errorOccurred then
                // Find the catch block
                let catchBlock =
                    block.NestedBlocks
                    |> List.tryFind (fun b -> b.Type = BlockType.Catch)

                match catchBlock with
                | Some cb ->
                    // Store the error message in the environment
                    environment.["error"] <- StringValue(errorMessage)

                    // Execute the catch block
                    let mutable catchResult = Success(StringValue(""))
                    let mutable continueExecution = true

                    for nestedBlock in cb.NestedBlocks do
                        if continueExecution then
                            match executeBlock nestedBlock environment with
                            | Success value -> catchResult <- Success value
                            | Error msg ->
                                catchResult <- Error msg
                                // Stop execution on error
                                continueExecution <- false

                    catchResult
                | None ->
                    // No catch block, return the error
                    Error(errorMessage)
            else
                // No error occurred, return the result
                result

        | BlockType.Catch ->
            // Catch blocks are handled by the Try block
            // This should only be executed if a Catch block is used outside a Try block
            Error("Catch block used outside of Try block")

        | BlockType.Return ->
            // Get the return value
            match block.Properties.TryFind("value") with
            | Some value ->
                // Substitute variables in string values
                match value with
                | StringValue s -> Success(StringValue(substituteVariables s environment))
                | _ -> Success(value)
            | None ->
                // Return empty string if no value provided
                Success(StringValue(""))

        | BlockType.FSharp ->
            // Get the F# code from the block content
            let code = block.Content

            if String.IsNullOrWhiteSpace(code) then
                Error("FSharp block has no code content")
            else
                try
                    // For now, we'll just parse the code and extract variables
                    // In a real implementation, we would use FSharp.Compiler.Service to compile and execute the code

                    // Extract variable definitions from the code
                    let lines = code.Split('\n')
                    let mutable result = StringValue("")

                    // Look for the last expression that could be a return value
                    for line in lines do
                        let trimmedLine = line.Trim()
                        if not (String.IsNullOrWhiteSpace(trimmedLine)) &&
                           not (trimmedLine.StartsWith("//")) &&
                           not (trimmedLine.StartsWith("let ")) &&
                           not (trimmedLine.StartsWith("open ")) &&
                           not (trimmedLine.StartsWith("type ")) &&
                           not (trimmedLine.Contains("printfn")) then
                            // This could be a return value
                            if trimmedLine.StartsWith("\"")
                               && trimmedLine.EndsWith("\"") then
                                // It's a string
                                result <- StringValue(trimmedLine.Trim('"'))
                            elif trimmedLine.StartsWith("sprintf")
                                 || trimmedLine.StartsWith("String.Format") then
                                // It's a formatted string, extract a sample
                                result <- StringValue("Formatted string result")
                            elif Double.TryParse(trimmedLine, ref 0.0) then
                                // It's a number
                                result <- NumberValue(Double.Parse(trimmedLine))
                            elif trimmedLine = "true" || trimmedLine = "false" then
                                // It's a boolean
                                result <- BoolValue(Boolean.Parse(trimmedLine))
                            elif trimmedLine.StartsWith("[") && trimmedLine.EndsWith("]") then
                                // It's a list
                                result <- ListValue([StringValue("List item")])
                            elif trimmedLine.StartsWith("{") && trimmedLine.EndsWith("}") then
                                // It's an object
                                result <- ObjectValue(Map.ofList [("key", StringValue("value"))])
                            else
                                // Default to string
                                result <- StringValue(trimmedLine)

                    // Update environment with variables from the code
                    for line in lines do
                        let trimmedLine = line.Trim()
                        if trimmedLine.StartsWith("let ") && trimmedLine.Contains(" = ") then
                            let parts = trimmedLine.Substring(4).Split(" = ", 2)
                            if parts.Length = 2 then
                                let varName = parts.[0].Trim()
                                let varValue = parts.[1].Trim()

                                // Add the variable to the environment
                                if varValue.StartsWith("\"")
                                   && varValue.EndsWith("\"") then
                                    // It's a string
                                    environment.[varName] <- StringValue(varValue.Trim('"'))
                                elif Double.TryParse(varValue, ref 0.0) then
                                    // It's a number
                                    environment.[varName] <- NumberValue(Double.Parse(varValue))
                                elif varValue = "true" || varValue = "false" then
                                    // It's a boolean
                                    environment.[varName] <- BoolValue(Boolean.Parse(varValue))
                                else
                                    // Default to string
                                    environment.[varName] <- StringValue(varValue)

                    // Store the result in a special variable
                    environment.["_last_result"] <- result

                    // Return the result
                    Success(result)
                with
                | ex -> Error($"Error executing F# code: {ex.Message}")

        | BlockType.Unknown blockType ->
            Error($"Unknown block type: {blockType}")

    /// Execute a program
    and executeProgram (program: Program) =
        // Create an environment for variable storage
        let environment = Dictionary<string, PropertyValue>()

        // Clear the function registry
        functionRegistry <- Map.empty

        // First pass: register all functions
        for block in program.Blocks do
            if block.Type = BlockType.Function then
                match block.Name with
                | Some name ->
                    // Get parameters
                    let parameters =
                        match block.Properties.TryFind("parameters") with
                        | Some (ListValue paramList) ->
                            paramList |> List.choose (function
                                | StringValue s -> Some s
                                | _ -> None)
                        | Some (StringValue paramStr) ->
                            paramStr.Split([|','; ' '|], StringSplitOptions.RemoveEmptyEntries)
                            |> Array.toList
                        | _ -> []

                    // Register the function
                    let functionDef = {
                        Name = name
                        Parameters = parameters
                        Body = block.NestedBlocks
                    }
                    functionRegistry <- functionRegistry.Add(name, functionDef)
                    printfn "Registered function: %s with %d parameters" name parameters.Length
                | None ->
                    printfn "Warning: Function block without a name will be ignored"

        // Execute each block
        let mutable result = Success(StringValue(""))
        let mutable continueExecution = true

        for block in program.Blocks do
            // Skip function definitions in the main execution
            if continueExecution && block.Type <> BlockType.Function then
                match executeBlock block environment with
                | Success value -> result <- Success value
                | Error msg ->
                    result <- Error msg
                    // Stop execution on error
                    continueExecution <- false

        result
