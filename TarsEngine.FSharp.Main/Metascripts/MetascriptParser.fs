namespace TarsEngine.FSharp.Main.Metascripts

open System
open System.Text.RegularExpressions
open TarsEngine.FSharp.Main.Services

/// <summary>
/// Parser for metascripts
/// </summary>
module MetascriptParser =
    /// <summary>
    /// Block type in a metascript
    /// </summary>
    type BlockType =
        | Describe
        | Config
        | Variable
        | Action
        | FSharp
        | CSharp
        | JavaScript
        | Python
        | If
        | Else
        | Loop
        | Function
        | Unknown
    
    /// <summary>
    /// Block in a metascript
    /// </summary>
    type Block = {
        /// <summary>
        /// The type of the block
        /// </summary>
        Type: BlockType
        
        /// <summary>
        /// The name of the block (if any)
        /// </summary>
        Name: string option
        
        /// <summary>
        /// The content of the block
        /// </summary>
        Content: string
        
        /// <summary>
        /// The properties of the block
        /// </summary>
        Properties: Map<string, string>
        
        /// <summary>
        /// The line number where the block starts
        /// </summary>
        StartLine: int
        
        /// <summary>
        /// The line number where the block ends
        /// </summary>
        EndLine: int
    }
    
    /// <summary>
    /// Parses a metascript into blocks
    /// </summary>
    let parseBlocks (metascript: string) : Block list =
        // Split the metascript into lines
        let lines = metascript.Split([|'\r'; '\n'|], StringSplitOptions.None)
        
        // Define regex patterns for different block types
        let blockStartPattern = @"^(\w+)\s*(?:(\w+))?\s*\{$"
        let blockEndPattern = @"^\}$"
        let propertyPattern = @"^\s*(\w+):\s*(?:""([^""]*)""|'([^']*)'|([^,}]*))(?:,|$)"
        
        // Parse the metascript
        let rec parseLines lineIndex blocks currentBlock properties =
            if lineIndex >= lines.Length then
                // End of metascript
                match currentBlock with
                | Some block ->
                    // Add the last block
                    let finalBlock = { block with Properties = properties; EndLine = lineIndex }
                    finalBlock :: blocks
                | None -> blocks
            else
                let line = lines.[lineIndex].Trim()
                
                match currentBlock, line with
                | None, "" ->
                    // Empty line outside a block
                    parseLines (lineIndex + 1) blocks None Map.empty
                
                | None, _ ->
                    // Check if this is the start of a block
                    let blockStartMatch = Regex.Match(line, blockStartPattern)
                    
                    if blockStartMatch.Success then
                        // Start of a block
                        let blockType = blockStartMatch.Groups.[1].Value.ToUpperInvariant()
                        let blockName = 
                            if blockStartMatch.Groups.[2].Success then
                                Some blockStartMatch.Groups.[2].Value
                            else
                                None
                        
                        let blockTypeEnum =
                            match blockType with
                            | "DESCRIBE" -> BlockType.Describe
                            | "CONFIG" -> BlockType.Config
                            | "VARIABLE" -> BlockType.Variable
                            | "ACTION" -> BlockType.Action
                            | "FSHARP" -> BlockType.FSharp
                            | "CSHARP" -> BlockType.CSharp
                            | "JAVASCRIPT" -> BlockType.JavaScript
                            | "PYTHON" -> BlockType.Python
                            | "IF" -> BlockType.If
                            | "ELSE" -> BlockType.Else
                            | "LOOP" -> BlockType.Loop
                            | "FUNCTION" -> BlockType.Function
                            | _ -> BlockType.Unknown
                        
                        let newBlock = {
                            Type = blockTypeEnum
                            Name = blockName
                            Content = ""
                            Properties = Map.empty
                            StartLine = lineIndex
                            EndLine = -1
                        }
                        
                        parseLines (lineIndex + 1) blocks (Some newBlock) Map.empty
                    else
                        // Not a block start, skip this line
                        parseLines (lineIndex + 1) blocks None Map.empty
                
                | Some block, "" ->
                    // Empty line inside a block
                    parseLines (lineIndex + 1) blocks currentBlock properties
                
                | Some block, _ ->
                    // Check if this is the end of a block
                    if Regex.IsMatch(line, blockEndPattern) then
                        // End of a block
                        let finalBlock = { block with Properties = properties; EndLine = lineIndex }
                        
                        // For code blocks, extract the content
                        let finalBlockWithContent =
                            match block.Type with
                            | BlockType.FSharp | BlockType.CSharp | BlockType.JavaScript | BlockType.Python ->
                                // Extract the content between the start and end lines
                                let content = 
                                    lines
                                    |> Array.skip (block.StartLine + 1)
                                    |> Array.take (lineIndex - block.StartLine - 1)
                                    |> String.concat Environment.NewLine
                                
                                { finalBlock with Content = content }
                            | _ -> finalBlock
                        
                        parseLines (lineIndex + 1) (finalBlockWithContent :: blocks) None Map.empty
                    else
                        // Check if this is a property
                        let propertyMatch = Regex.Match(line, propertyPattern)
                        
                        if propertyMatch.Success then
                            // Property
                            let propertyName = propertyMatch.Groups.[1].Value
                            let propertyValue = 
                                if propertyMatch.Groups.[2].Success then
                                    propertyMatch.Groups.[2].Value
                                elif propertyMatch.Groups.[3].Success then
                                    propertyMatch.Groups.[3].Value
                                elif propertyMatch.Groups.[4].Success then
                                    propertyMatch.Groups.[4].Value.Trim()
                                else
                                    ""
                            
                            let updatedProperties = properties.Add(propertyName, propertyValue)
                            
                            parseLines (lineIndex + 1) blocks currentBlock updatedProperties
                        else
                            // Content line for a code block
                            match block.Type with
                            | BlockType.FSharp | BlockType.CSharp | BlockType.JavaScript | BlockType.Python ->
                                // Add this line to the content
                                let updatedBlock = { block with Content = block.Content + line + Environment.NewLine }
                                parseLines (lineIndex + 1) blocks (Some updatedBlock) properties
                            | _ ->
                                // Ignore content for non-code blocks
                                parseLines (lineIndex + 1) blocks currentBlock properties
        
        // Parse the metascript
        parseLines 0 [] None Map.empty |> List.rev
    
    /// <summary>
    /// Converts a block to a DSL element
    /// </summary>
    let blockToDslElement (block: Block) : DslElement =
        match block.Type with
        | BlockType.FSharp ->
            // F# code block
            DslElement.FSharpCode block.Content
        
        | BlockType.CSharp ->
            // C# code block
            DslElement.CSharpCode block.Content
        
        | BlockType.JavaScript ->
            // JavaScript code block
            DslElement.JavaScriptCode block.Content
        
        | BlockType.Python ->
            // Python code block
            DslElement.PythonCode block.Content
        
        | BlockType.Variable ->
            // Variable block
            let name = 
                match block.Name with
                | Some n -> n
                | None -> ""
            
            let value =
                match block.Properties.TryFind "value" with
                | Some v -> 
                    // Try to parse the value
                    match Int32.TryParse(v) with
                    | true, i -> i :> obj
                    | _ ->
                        match Double.TryParse(v) with
                        | true, d -> d :> obj
                        | _ ->
                            match Boolean.TryParse(v) with
                            | true, b -> b :> obj
                            | _ -> v :> obj
                | None -> "" :> obj
            
            DslElement.BinaryOperation(
                DslElement.Variable name,
                "=",
                DslElement.Literal value
            )
        
        | BlockType.Action ->
            // Action block
            let actionType =
                match block.Properties.TryFind "type" with
                | Some t -> t
                | None -> ""
            
            let message =
                match block.Properties.TryFind "message" with
                | Some m -> m
                | None -> ""
            
            DslElement.FunctionCall(actionType, [message :> obj])
        
        | BlockType.If ->
            // If block
            let condition =
                match block.Properties.TryFind "condition" with
                | Some c -> DslElement.Literal (c :> obj)
                | None -> DslElement.Literal (true :> obj)
            
            // TODO: Parse nested blocks
            
            DslElement.Conditional(
                condition,
                DslElement.Block [],
                DslElement.Block []
            )
        
        | _ ->
            // Other block types
            DslElement.Block []
    
    /// <summary>
    /// Parses a metascript into DSL elements
    /// </summary>
    let parseMetascript (metascript: string) : Result<DslElement list, string list> =
        try
            // Parse the metascript into blocks
            let blocks = parseBlocks metascript
            
            // Convert blocks to DSL elements
            let elements = blocks |> List.map blockToDslElement
            
            Ok elements
        with
        | ex -> Error [ex.Message]
