namespace TarsEngine.FSharp.Core.Metascript.Services

open System
open System.Text.RegularExpressions
open TarsEngine.FSharp.Core.Metascript.Types

/// <summary>
/// Enhanced metascript parser for TARS autonomous coding system
/// Supports F#, TARS, YAML, and other block types
/// </summary>
module TarsMetascriptParser =
    
    /// <summary>
    /// Parses a metascript file into blocks
    /// </summary>
    /// <param name="content">The metascript file content</param>
    /// <returns>List of parsed metascript blocks</returns>
    let parseMetascript (content: string) : MetascriptBlock list =
        let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
        let mutable blocks = []
        let mutable currentBlock = None
        let mutable currentContent = []
        let mutable lineNumber = 0
        
        for line in lines do
            lineNumber <- lineNumber + 1
            let trimmedLine = line.Trim()
            
            // Check for block start
            if isBlockStart trimmedLine then
                // Save previous block if exists
                match currentBlock with
                | Some (blockType, startLine, parameters) ->
                    let blockContent = String.Join("\n", List.rev currentContent)
                    let block = {
                        BlockType = blockType
                        Content = blockContent
                        LineNumber = startLine
                        Parameters = parameters
                    }
                    blocks <- block :: blocks
                | None -> ()
                
                // Start new block
                let (blockType, parameters) = parseBlockHeader trimmedLine
                currentBlock <- Some (blockType, lineNumber, parameters)
                currentContent <- []
            
            // Check for block end
            elif isBlockEnd trimmedLine then
                match currentBlock with
                | Some (blockType, startLine, parameters) ->
                    let blockContent = String.Join("\n", List.rev currentContent)
                    let block = {
                        BlockType = blockType
                        Content = blockContent
                        LineNumber = startLine
                        Parameters = parameters
                    }
                    blocks <- block :: blocks
                    currentBlock <- None
                    currentContent <- []
                | None -> ()
            
            // Add content to current block
            else
                match currentBlock with
                | Some _ -> currentContent <- line :: currentContent
                | None -> () // Ignore content outside blocks
        
        // Handle final block if file doesn't end with closing brace
        match currentBlock with
        | Some (blockType, startLine, parameters) ->
            let blockContent = String.Join("\n", List.rev currentContent)
            let block = {
                BlockType = blockType
                Content = blockContent
                LineNumber = startLine
                Parameters = parameters
            }
            blocks <- block :: blocks
        | None -> ()
        
        List.rev blocks
    
    /// <summary>
    /// Checks if a line starts a new block
    /// </summary>
    and isBlockStart (line: string) : bool =
        let blockPatterns = [
            @"^(FSHARP|TARS|YAML|ACTION|VARIABLE|FUNCTION|DESCRIBE|CONFIG|LLM)\s*\{"
            @"^(FSHARP|TARS|YAML|ACTION|VARIABLE|FUNCTION|DESCRIBE|CONFIG|LLM)\s*$"
        ]
        
        blockPatterns |> List.exists (fun pattern -> Regex.IsMatch(line, pattern, RegexOptions.IgnoreCase))
    
    /// <summary>
    /// Checks if a line ends a block
    /// </summary>
    and isBlockEnd (line: string) : bool =
        line = "}"
    
    /// <summary>
    /// Parses a block header to extract type and parameters
    /// </summary>
    and parseBlockHeader (line: string) : BlockType * Map<string, string> =
        // Extract block type
        let blockTypeMatch = Regex.Match(line, @"^(\w+)", RegexOptions.IgnoreCase)
        let blockTypeStr = if blockTypeMatch.Success then blockTypeMatch.Groups.[1].Value else "UNKNOWN"
        let blockType = Helpers.parseBlockType blockTypeStr
        
        // Extract parameters (if any)
        let parametersMatch = Regex.Match(line, @"\{([^}]*)\}")
        let parameters = 
            if parametersMatch.Success then
                parseParameters parametersMatch.Groups.[1].Value
            else
                Map.empty
        
        (blockType, parameters)
    
    /// <summary>
    /// Parses parameter string into a map
    /// </summary>
    and parseParameters (paramStr: string) : Map<string, string> =
        let mutable parameters = Map.empty
        
        // Simple parameter parsing (key: value pairs)
        let paramPairs = paramStr.Split([|';'; ','|], StringSplitOptions.RemoveEmptyEntries)
        
        for pair in paramPairs do
            let colonIndex = pair.IndexOf(':')
            if colonIndex > 0 then
                let key = pair.Substring(0, colonIndex).Trim()
                let value = pair.Substring(colonIndex + 1).Trim().Trim('"')
                parameters <- parameters.Add(key, value)
        
        parameters
    
    /// <summary>
    /// Validates a metascript structure
    /// </summary>
    let validateMetascript (blocks: MetascriptBlock list) : Result<unit, string list> =
        let mutable errors = []
        
        // Check for required blocks
        let hasDescribe = blocks |> List.exists (fun b -> b.BlockType = Describe)
        if not hasDescribe then
            errors <- "Missing DESCRIBE block" :: errors
        
        // Check for valid block sequences
        let blockTypes = blocks |> List.map (fun b -> b.BlockType)
        
        // Validate F# blocks have proper syntax
        let fsharpBlocks = blocks |> List.filter (fun b -> b.BlockType = FSharp)
        for block in fsharpBlocks do
            match validateFSharpBlock block.Content with
            | Error err -> errors <- sprintf "F# block error at line %d: %s" block.LineNumber err :: errors
            | Ok _ -> ()
        
        // Validate TARS blocks
        let tarsBlocks = blocks |> List.filter (fun b -> b.BlockType = Tars)
        for block in tarsBlocks do
            match validateTarsBlock block.Content with
            | Error err -> errors <- sprintf "TARS block error at line %d: %s" block.LineNumber err :: errors
            | Ok _ -> ()
        
        if List.isEmpty errors then Ok () else Error (List.rev errors)
    
    /// <summary>
    /// Validates F# block syntax
    /// </summary>
    and validateFSharpBlock (content: string) : Result<unit, string> =
        try
            // Basic F# syntax validation
            let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            
            for line in lines do
                let trimmedLine = line.Trim()
                if not (String.IsNullOrEmpty(trimmedLine)) && not (trimmedLine.StartsWith("//")) then
                    // Check for basic F# syntax issues
                    if trimmedLine.Contains(";;") then
                        return Error "F# interactive syntax (;;) not allowed in metascript blocks"
            
            Ok ()
        with
        | ex -> Error ex.Message
    
    /// <summary>
    /// Validates TARS block syntax
    /// </summary>
    and validateTarsBlock (content: string) : Result<unit, string> =
        try
            let lines = content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            
            for line in lines do
                let trimmedLine = line.Trim()
                if not (String.IsNullOrEmpty(trimmedLine)) && not (trimmedLine.StartsWith("//")) then
                    // Check for valid TARS commands
                    let validCommands = [
                        "generate_project:"
                        "analyze_code:"
                        "improve_code:"
                        "autonomous_coding:"
                    ]
                    
                    let isValidCommand = validCommands |> List.exists (fun cmd -> trimmedLine.StartsWith(cmd))
                    if not isValidCommand then
                        return Error (sprintf "Unknown TARS command: %s" trimmedLine)
            
            Ok ()
        with
        | ex -> Error ex.Message
    
    /// <summary>
    /// Extracts variables from metascript blocks
    /// </summary>
    let extractVariables (blocks: MetascriptBlock list) : Map<string, string> =
        let mutable variables = Map.empty
        
        let variableBlocks = blocks |> List.filter (fun b -> b.BlockType = Variable)
        
        for block in variableBlocks do
            // Extract variable name from parameters or content
            let variableName = 
                match block.Parameters.TryFind("name") with
                | Some name -> name
                | None -> 
                    // Try to extract from content
                    let nameMatch = Regex.Match(block.Content, @"name:\s*""?([^""]+)""?")
                    if nameMatch.Success then nameMatch.Groups.[1].Value else "unnamed"
            
            let variableValue = 
                match block.Parameters.TryFind("value") with
                | Some value -> value
                | None ->
                    // Try to extract from content
                    let valueMatch = Regex.Match(block.Content, @"value:\s*""?([^""]+)""?")
                    if valueMatch.Success then valueMatch.Groups.[1].Value else ""
            
            variables <- variables.Add(variableName, variableValue)
        
        variables
    
    /// <summary>
    /// Extracts configuration from metascript blocks
    /// </summary>
    let extractConfiguration (blocks: MetascriptBlock list) : Map<string, string> =
        let mutable config = Map.empty
        
        let configBlocks = blocks |> List.filter (fun b -> b.BlockType = Config)
        
        for block in configBlocks do
            let lines = block.Content.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            
            for line in lines do
                let trimmedLine = line.Trim()
                let colonIndex = trimmedLine.IndexOf(':')
                if colonIndex > 0 then
                    let key = trimmedLine.Substring(0, colonIndex).Trim()
                    let value = trimmedLine.Substring(colonIndex + 1).Trim().Trim('"')
                    config <- config.Add(key, value)
        
        config
