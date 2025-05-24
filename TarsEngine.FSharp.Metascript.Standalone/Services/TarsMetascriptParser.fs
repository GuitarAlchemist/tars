namespace TarsEngine.FSharp.Metascript.Services

open System
open System.IO
open System.Text
open System.Text.RegularExpressions
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Parser for TARS metascript files.
/// </summary>
type TarsMetascriptParser(logger: ILogger<TarsMetascriptParser>) =
    
    /// <summary>
    /// Parses a TARS metascript from text.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="name">The name of the metascript.</param>
    /// <param name="filePath">The file path of the metascript.</param>
    /// <returns>The parsed metascript.</returns>
    member this.ParseTarsMetascriptAsync(text: string, ?name: string, ?filePath: string) =
        task {
            try
                logger.LogInformation("Parsing TARS metascript: {Name}", defaultArg name "unnamed")
                
                // Parse the blocks
                let blocks = this.ParseTarsBlocks(text)
                
                // Extract metadata
                let metadata = this.ExtractTarsMetadata(blocks)
                
                // Create the metascript
                let metascriptName = 
                    match name with
                    | Some n -> n
                    | None -> 
                        match filePath with
                        | Some path -> Path.GetFileNameWithoutExtension(path)
                        | None -> "unnamed"
                
                let metascript = {
                    Name = metascriptName
                    Blocks = blocks
                    FilePath = filePath
                    CreationTime = DateTime.UtcNow
                    LastModificationTime = None
                    Description = Map.tryFind "description" metadata
                    Author = Map.tryFind "author" metadata
                    Version = Map.tryFind "version" metadata
                    Dependencies = 
                        match Map.tryFind "dependencies" metadata with
                        | Some deps -> deps.Split([|','; ';'|], StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun s -> s.Trim()) |> Array.toList
                        | None -> []
                    Imports = 
                        match Map.tryFind "imports" metadata with
                        | Some imports -> imports.Split([|','; ';'|], StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun s -> s.Trim()) |> Array.toList
                        | None -> []
                    Metadata = metadata
                }
                
                return metascript
            with
            | ex ->
                logger.LogError(ex, "Error parsing TARS metascript: {Name}", defaultArg name "unnamed")
                return {
                    Name = defaultArg name "unnamed"
                    Blocks = []
                    FilePath = filePath
                    CreationTime = DateTime.UtcNow
                    LastModificationTime = None
                    Description = None
                    Author = None
                    Version = None
                    Dependencies = []
                    Imports = []
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Parses a TARS metascript from a file.
    /// </summary>
    /// <param name="filePath">The file path to parse.</param>
    /// <returns>The parsed metascript.</returns>
    member this.ParseTarsMetascriptFileAsync(filePath: string) =
        task {
            try
                logger.LogInformation("Parsing TARS metascript file: {FilePath}", filePath)
                
                // Check if the file exists
                if not (File.Exists(filePath)) then
                    logger.LogError("TARS metascript file not found: {FilePath}", filePath)
                    return {
                        Name = Path.GetFileNameWithoutExtension(filePath)
                        Blocks = []
                        FilePath = Some filePath
                        CreationTime = DateTime.UtcNow
                        LastModificationTime = None
                        Description = None
                        Author = None
                        Version = None
                        Dependencies = []
                        Imports = []
                        Metadata = Map.empty
                    }
                else
                    // Read the file
                    let! text = File.ReadAllTextAsync(filePath)
                    
                    // Get the file info
                    let fileInfo = FileInfo(filePath)
                    
                    // Parse the metascript
                    let! metascript = this.ParseTarsMetascriptAsync(text, Path.GetFileNameWithoutExtension(filePath), filePath)
                    
                    // Update the metascript with file info
                    let result = { 
                        metascript with
                            CreationTime = fileInfo.CreationTimeUtc
                            LastModificationTime = Some fileInfo.LastWriteTimeUtc 
                    }
                    
                    return result
            with
            | ex ->
                logger.LogError(ex, "Error parsing TARS metascript file: {FilePath}", filePath)
                return {
                    Name = Path.GetFileNameWithoutExtension(filePath)
                    Blocks = []
                    FilePath = Some filePath
                    CreationTime = DateTime.UtcNow
                    LastModificationTime = None
                    Description = None
                    Author = None
                    Version = None
                    Dependencies = []
                    Imports = []
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Parses blocks from TARS metascript text.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <returns>The parsed blocks.</returns>
    member private this.ParseTarsBlocks(text: string) =
        try
            let blocks = List<MetascriptBlock>()
            
            // Define regex patterns for TARS blocks
            let blockStartPattern = @"(DESCRIBE|CONFIG|VARIABLE|AGENT|TASK|FUNCTION|ACTION|RETURN)\s+(\w+)?\s*\{"
            let blockStartRegex = Regex(blockStartPattern, RegexOptions.IgnoreCase)
            
            // Split the text into lines
            let lines = text.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
            
            // Parse the blocks
            let mutable i = 0
            while i < lines.Length do
                let line = lines.[i]
                
                // Check if this line is a block start
                let startMatch = blockStartRegex.Match(line)
                if startMatch.Success then
                    // Get the block type
                    let blockTypeStr = startMatch.Groups.[1].Value.ToUpperInvariant()
                    let blockType = 
                        match blockTypeStr with
                        | "DESCRIBE" -> MetascriptBlockType.Describe
                        | "CONFIG" -> MetascriptBlockType.Config
                        | "VARIABLE" -> MetascriptBlockType.Variable
                        | "AGENT" -> MetascriptBlockType.Agent
                        | "TASK" -> MetascriptBlockType.Task
                        | "FUNCTION" -> MetascriptBlockType.Function
                        | "ACTION" -> MetascriptBlockType.Action
                        | "RETURN" -> MetascriptBlockType.Return
                        | _ -> MetascriptBlockType.Unknown
                    
                    // Get the block name
                    let blockName = 
                        if startMatch.Groups.Count > 2 && not (String.IsNullOrWhiteSpace(startMatch.Groups.[2].Value)) then
                            startMatch.Groups.[2].Value
                        else
                            ""
                    
                    // Find the end of the block
                    let mutable j = i + 1
                    let mutable braceCount = 1
                    let mutable blockContent = StringBuilder()
                    
                    while j < lines.Length && braceCount > 0 do
                        let currentLine = lines.[j]
                        
                        // Count braces
                        for c in currentLine do
                            if c = '{' then braceCount <- braceCount + 1
                            elif c = '}' then braceCount <- braceCount - 1
                        
                        // Add the line to the block content if it's not the end brace
                        if not (currentLine.Trim() = "}" && braceCount = 0) then
                            blockContent.AppendLine(currentLine) |> ignore
                        
                        j <- j + 1
                    
                    // Create the block
                    let parameters = 
                        if String.IsNullOrWhiteSpace(blockName) then
                            []
                        else
                            [{ Name = "name"; Value = blockName }]
                    
                    blocks.Add({
                        Type = blockType
                        Content = blockContent.ToString().Trim()
                        LineNumber = i + 1
                        ColumnNumber = line.IndexOf(blockTypeStr)
                        Parameters = parameters
                        Id = Guid.NewGuid().ToString()
                        ParentId = None
                        Metadata = Map.empty
                    })
                    
                    // Move to the end of the block
                    i <- j
                else
                    // Move to the next line
                    i <- i + 1
            
            blocks |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error parsing TARS blocks")
            []
    
    /// <summary>
    /// Extracts metadata from TARS blocks.
    /// </summary>
    /// <param name="blocks">The blocks to extract metadata from.</param>
    /// <returns>The extracted metadata.</returns>
    member private this.ExtractTarsMetadata(blocks: MetascriptBlock list) =
        try
            let metadata = Dictionary<string, string>()
            
            // Look for metadata in DESCRIBE blocks
            let describeBlocks = blocks |> List.filter (fun b -> b.Type = MetascriptBlockType.Describe)
            
            for block in describeBlocks do
                // Parse the content as JSON-like key-value pairs
                let lines = block.Content.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
                
                for line in lines do
                    let trimmedLine = line.Trim()
                    if not (String.IsNullOrWhiteSpace(trimmedLine)) && trimmedLine.Contains(':') then
                        let parts = trimmedLine.Split(':', 2)
                        if parts.Length = 2 then
                            let key = parts.[0].Trim().ToLowerInvariant()
                            let value = parts.[1].Trim().TrimEnd(',')
                            
                            // Remove quotes if present
                            let value = 
                                if value.StartsWith("\"") && value.EndsWith("\"") then
                                    value.Substring(1, value.Length - 2)
                                else
                                    value
                            
                            if not (String.IsNullOrWhiteSpace(key)) then
                                metadata.[key] <- value
            
            // Look for metadata in CONFIG blocks
            let configBlocks = blocks |> List.filter (fun b -> b.Type = MetascriptBlockType.Config)
            
            for block in configBlocks do
                // Parse the content as JSON-like key-value pairs
                let lines = block.Content.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
                
                for line in lines do
                    let trimmedLine = line.Trim()
                    if not (String.IsNullOrWhiteSpace(trimmedLine)) && trimmedLine.Contains(':') then
                        let parts = trimmedLine.Split(':', 2)
                        if parts.Length = 2 then
                            let key = parts.[0].Trim().ToLowerInvariant()
                            let value = parts.[1].Trim().TrimEnd(',')
                            
                            // Remove quotes if present
                            let value = 
                                if value.StartsWith("\"") && value.EndsWith("\"") then
                                    value.Substring(1, value.Length - 2)
                                else
                                    value
                            
                            if not (String.IsNullOrWhiteSpace(key)) then
                                metadata.[$"config.{key}"] <- value
            
            // Look for metadata in block parameters
            for block in blocks do
                for param in block.Parameters do
                    let key = $"block.{block.Id}.{param.Name}".ToLowerInvariant()
                    metadata.[key] <- param.Value
            
            metadata |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq
        with
        | ex ->
            logger.LogError(ex, "Error extracting TARS metadata")
            Map.empty
