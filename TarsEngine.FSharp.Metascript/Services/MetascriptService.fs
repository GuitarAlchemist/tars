namespace TarsEngine.FSharp.Metascript.Services

open System
open System.IO
open System.Text.RegularExpressions
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Implementation of the IMetascriptService interface.
/// </summary>
type MetascriptService(logger: ILogger<MetascriptService>, executor: IMetascriptExecutor) =
    
    /// <summary>
    /// Gets the default parser configuration.
    /// </summary>
    /// <returns>The default parser configuration.</returns>
    member this.GetDefaultParserConfig() =
        let blockStartMarkers = Map.ofList [
            (MetascriptBlockType.Text, "```text")
            (MetascriptBlockType.Code, "```code")
            (MetascriptBlockType.FSharp, "```fsharp")
            (MetascriptBlockType.CSharp, "```csharp")
            (MetascriptBlockType.Python, "```python")
            (MetascriptBlockType.JavaScript, "```javascript")
            (MetascriptBlockType.SQL, "```sql")
            (MetascriptBlockType.Markdown, "```markdown")
            (MetascriptBlockType.HTML, "```html")
            (MetascriptBlockType.CSS, "```css")
            (MetascriptBlockType.JSON, "```json")
            (MetascriptBlockType.XML, "```xml")
            (MetascriptBlockType.YAML, "```yaml")
            (MetascriptBlockType.Command, "```command")
            (MetascriptBlockType.Query, "```query")
            (MetascriptBlockType.Transformation, "```transform")
            (MetascriptBlockType.Analysis, "```analyze")
            (MetascriptBlockType.Reflection, "```reflect")
            (MetascriptBlockType.Execution, "```execute")
            (MetascriptBlockType.Import, "```import")
            (MetascriptBlockType.Export, "```export")
        ]
        
        let blockEndMarkers = Map.ofList [
            (MetascriptBlockType.Text, "```")
            (MetascriptBlockType.Code, "```")
            (MetascriptBlockType.FSharp, "```")
            (MetascriptBlockType.CSharp, "```")
            (MetascriptBlockType.Python, "```")
            (MetascriptBlockType.JavaScript, "```")
            (MetascriptBlockType.SQL, "```")
            (MetascriptBlockType.Markdown, "```")
            (MetascriptBlockType.HTML, "```")
            (MetascriptBlockType.CSS, "```")
            (MetascriptBlockType.JSON, "```")
            (MetascriptBlockType.XML, "```")
            (MetascriptBlockType.YAML, "```")
            (MetascriptBlockType.Command, "```")
            (MetascriptBlockType.Query, "```")
            (MetascriptBlockType.Transformation, "```")
            (MetascriptBlockType.Analysis, "```")
            (MetascriptBlockType.Reflection, "```")
            (MetascriptBlockType.Execution, "```")
            (MetascriptBlockType.Import, "```")
            (MetascriptBlockType.Export, "```")
        ]
        
        {
            BlockStartMarkers = blockStartMarkers
            BlockEndMarkers = blockEndMarkers
            DefaultBlockType = MetascriptBlockType.Text
            AllowNestedBlocks = false
            TrimBlockContent = true
            IncludeEmptyBlocks = false
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Parses a metascript from text.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="name">The name of the metascript.</param>
    /// <param name="filePath">The file path of the metascript.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The parsed metascript.</returns>
    member this.ParseMetascriptAsync(text: string, ?name: string, ?filePath: string, ?config: MetascriptParserConfig) =
        task {
            try
                logger.LogInformation("Parsing metascript: {Name}", defaultArg name "unnamed")
                
                // Get the parser configuration
                let config = defaultArg config (this.GetDefaultParserConfig())
                
                // Parse the metascript
                let blocks = this.ParseBlocks(text, config)
                
                // Extract metadata from the blocks
                let metadata = this.ExtractMetadata(blocks)
                
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
                logger.LogError(ex, "Error parsing metascript: {Name}", defaultArg name "unnamed")
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
    /// Parses a metascript from a file.
    /// </summary>
    /// <param name="filePath">The file path to parse.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The parsed metascript.</returns>
    member this.ParseMetascriptFileAsync(filePath: string, ?config: MetascriptParserConfig) =
        task {
            try
                logger.LogInformation("Parsing metascript file: {FilePath}", filePath)
                
                // Check if the file exists
                if not (File.Exists(filePath)) then
                    logger.LogError("Metascript file not found: {FilePath}", filePath)
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
                    let! metascript = this.ParseMetascriptAsync(text, Path.GetFileNameWithoutExtension(filePath), filePath, ?config = config)
                    
                    // Update the metascript with file info
                    return { metascript with
                        CreationTime = fileInfo.CreationTimeUtc
                        LastModificationTime = Some fileInfo.LastWriteTimeUtc
                    }
            with
            | ex ->
                logger.LogError(ex, "Error parsing metascript file: {FilePath}", filePath)
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
    /// Executes a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The execution result.</returns>
    member this.ExecuteMetascriptAsync(metascript: Metascript, ?context: MetascriptContext) =
        task {
            try
                logger.LogInformation("Executing metascript: {Name}", metascript.Name)
                
                // Create a context if none was provided
                let! context = 
                    match context with
                    | Some ctx -> Task.FromResult(ctx)
                    | None -> 
                        let workingDirectory = 
                            match metascript.FilePath with
                            | Some path -> Path.GetDirectoryName(path)
                            | None -> Directory.GetCurrentDirectory()
                        executor.CreateContextAsync(workingDirectory)
                
                // Execute the metascript
                return! executor.ExecuteAsync(metascript, context)
            with
            | ex ->
                logger.LogError(ex, "Error executing metascript: {Name}", metascript.Name)
                return {
                    Metascript = metascript
                    BlockResults = []
                    Status = MetascriptExecutionStatus.Failure
                    Output = ""
                    Error = Some (ex.ToString())
                    ExecutionTimeMs = 0.0
                    ReturnValue = None
                    Variables = Map.empty
                    Context = None
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Executes a metascript from text.
    /// </summary>
    /// <param name="text">The text to execute.</param>
    /// <param name="name">The name of the metascript.</param>
    /// <param name="filePath">The file path of the metascript.</param>
    /// <param name="context">The execution context.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The execution result.</returns>
    member this.ExecuteMetascriptTextAsync(text: string, ?name: string, ?filePath: string, ?context: MetascriptContext, ?config: MetascriptParserConfig) =
        task {
            try
                logger.LogInformation("Executing metascript text: {Name}", defaultArg name "unnamed")
                
                // Parse the metascript
                let! metascript = this.ParseMetascriptAsync(text, ?name = name, ?filePath = filePath, ?config = config)
                
                // Execute the metascript
                return! this.ExecuteMetascriptAsync(metascript, ?context = context)
            with
            | ex ->
                logger.LogError(ex, "Error executing metascript text: {Name}", defaultArg name "unnamed")
                return {
                    Metascript = {
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
                    BlockResults = []
                    Status = MetascriptExecutionStatus.Failure
                    Output = ""
                    Error = Some (ex.ToString())
                    ExecutionTimeMs = 0.0
                    ReturnValue = None
                    Variables = Map.empty
                    Context = None
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Executes a metascript from a file.
    /// </summary>
    /// <param name="filePath">The file path to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The execution result.</returns>
    member this.ExecuteMetascriptFileAsync(filePath: string, ?context: MetascriptContext, ?config: MetascriptParserConfig) =
        task {
            try
                logger.LogInformation("Executing metascript file: {FilePath}", filePath)
                
                // Parse the metascript
                let! metascript = this.ParseMetascriptFileAsync(filePath, ?config = config)
                
                // Execute the metascript
                return! this.ExecuteMetascriptAsync(metascript, ?context = context)
            with
            | ex ->
                logger.LogError(ex, "Error executing metascript file: {FilePath}", filePath)
                return {
                    Metascript = {
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
                    BlockResults = []
                    Status = MetascriptExecutionStatus.Failure
                    Output = ""
                    Error = Some (ex.ToString())
                    ExecutionTimeMs = 0.0
                    ReturnValue = None
                    Variables = Map.empty
                    Context = None
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Creates a new metascript context.
    /// </summary>
    /// <param name="workingDirectory">The working directory.</param>
    /// <param name="variables">The initial variables.</param>
    /// <param name="parent">The parent context.</param>
    /// <returns>The new context.</returns>
    member this.CreateContextAsync(?workingDirectory: string, ?variables: Map<string, MetascriptVariable>, ?parent: MetascriptContext) =
        executor.CreateContextAsync(?workingDirectory = workingDirectory, ?variables = variables, ?parent = parent)
    
    /// <summary>
    /// Validates a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to validate.</param>
    /// <returns>Whether the metascript is valid and any validation errors.</returns>
    member this.ValidateMetascriptAsync(metascript: Metascript) =
        task {
            try
                logger.LogInformation("Validating metascript: {Name}", metascript.Name)
                
                let errors = List<string>()
                
                // Check if the metascript has blocks
                if metascript.Blocks.IsEmpty then
                    errors.Add("Metascript has no blocks")
                
                // Check if all block types are supported
                for block in metascript.Blocks do
                    if not (executor.IsBlockTypeSupported(block.Type)) then
                        errors.Add($"Block type not supported: {block.Type}")
                
                return (errors.Count = 0, errors |> Seq.toList)
            with
            | ex ->
                logger.LogError(ex, "Error validating metascript: {Name}", metascript.Name)
                return (false, [ex.ToString()])
        }
    
    /// <summary>
    /// Parses blocks from text.
    /// </summary>
    /// <param name="text">The text to parse.</param>
    /// <param name="config">The parser configuration.</param>
    /// <returns>The parsed blocks.</returns>
    member private this.ParseBlocks(text: string, config: MetascriptParserConfig) =
        try
            let blocks = List<MetascriptBlock>()
            let lines = text.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
            
            let mutable currentBlockType = config.DefaultBlockType
            let mutable currentBlockContent = List<string>()
            let mutable currentBlockLineNumber = 0
            let mutable currentBlockColumnNumber = 0
            let mutable currentBlockParameters = List<MetascriptBlockParameter>()
            let mutable inBlock = false
            
            for i = 0 to lines.Length - 1 do
                let line = lines.[i]
                
                if inBlock then
                    // Check if this line is a block end marker
                    let isEndMarker =
                        match config.BlockEndMarkers.TryGetValue(currentBlockType) with
                        | true, marker -> line.Trim() = marker
                        | false, _ -> false
                    
                    if isEndMarker then
                        // End the current block
                        let blockContent = String.Join(Environment.NewLine, currentBlockContent)
                        let trimmedContent = if config.TrimBlockContent then blockContent.Trim() else blockContent
                        
                        if not (String.IsNullOrWhiteSpace(trimmedContent) || (String.IsNullOrEmpty(trimmedContent) && not config.IncludeEmptyBlocks)) then
                            blocks.Add({
                                Type = currentBlockType
                                Content = trimmedContent
                                LineNumber = currentBlockLineNumber
                                ColumnNumber = currentBlockColumnNumber
                                Parameters = currentBlockParameters |> Seq.toList
                                Id = Guid.NewGuid().ToString()
                                ParentId = None
                                Metadata = Map.empty
                            })
                        
                        // Reset for the next block
                        currentBlockType <- config.DefaultBlockType
                        currentBlockContent <- List<string>()
                        currentBlockParameters <- List<MetascriptBlockParameter>()
                        inBlock <- false
                    else
                        // Add the line to the current block
                        currentBlockContent.Add(line)
                else
                    // Check if this line is a block start marker
                    let blockTypeAndMarker =
                        config.BlockStartMarkers
                        |> Map.toSeq
                        |> Seq.tryFind (fun (_, marker) -> line.Trim().StartsWith(marker))
                    
                    match blockTypeAndMarker with
                    | Some (blockType, marker) ->
                        // Start a new block
                        currentBlockType <- blockType
                        currentBlockLineNumber <- i + 1
                        currentBlockColumnNumber <- line.IndexOf(marker)
                        
                        // Extract parameters
                        let parameterText = line.Substring(line.IndexOf(marker) + marker.Length).Trim()
                        if not (String.IsNullOrWhiteSpace(parameterText)) then
                            let paramRegex = Regex(@"(\w+)=(?:""([^""]*)""|'([^']*)'|(\S+))")
                            let matches = paramRegex.Matches(parameterText)
                            
                            for m in matches do
                                let name = m.Groups.[1].Value
                                let value = 
                                    if m.Groups.[2].Success then m.Groups.[2].Value
                                    elif m.Groups.[3].Success then m.Groups.[3].Value
                                    else m.Groups.[4].Value
                                
                                currentBlockParameters.Add({ Name = name; Value = value })
                        
                        inBlock <- true
                    | None ->
                        // This is a text line outside of a block
                        if config.DefaultBlockType <> MetascriptBlockType.Unknown then
                            currentBlockContent.Add(line)
            
            // If we're still in a block at the end, add it
            if inBlock then
                let blockContent = String.Join(Environment.NewLine, currentBlockContent)
                let trimmedContent = if config.TrimBlockContent then blockContent.Trim() else blockContent
                
                if not (String.IsNullOrWhiteSpace(trimmedContent) || (String.IsNullOrEmpty(trimmedContent) && not config.IncludeEmptyBlocks)) then
                    blocks.Add({
                        Type = currentBlockType
                        Content = trimmedContent
                        LineNumber = currentBlockLineNumber
                        ColumnNumber = currentBlockColumnNumber
                        Parameters = currentBlockParameters |> Seq.toList
                        Id = Guid.NewGuid().ToString()
                        ParentId = None
                        Metadata = Map.empty
                    })
            
            // If we have text content outside of a block, add it as a text block
            if not inBlock && currentBlockContent.Count > 0 && config.DefaultBlockType <> MetascriptBlockType.Unknown then
                let blockContent = String.Join(Environment.NewLine, currentBlockContent)
                let trimmedContent = if config.TrimBlockContent then blockContent.Trim() else blockContent
                
                if not (String.IsNullOrWhiteSpace(trimmedContent) || (String.IsNullOrEmpty(trimmedContent) && not config.IncludeEmptyBlocks)) then
                    blocks.Add({
                        Type = config.DefaultBlockType
                        Content = trimmedContent
                        LineNumber = 1
                        ColumnNumber = 0
                        Parameters = []
                        Id = Guid.NewGuid().ToString()
                        ParentId = None
                        Metadata = Map.empty
                    })
            
            blocks |> Seq.toList
        with
        | ex ->
            logger.LogError(ex, "Error parsing blocks")
            []
    
    /// <summary>
    /// Extracts metadata from blocks.
    /// </summary>
    /// <param name="blocks">The blocks to extract metadata from.</param>
    /// <returns>The extracted metadata.</returns>
    member private this.ExtractMetadata(blocks: MetascriptBlock list) =
        try
            let metadata = Dictionary<string, string>()
            
            // Look for metadata in YAML blocks
            let yamlBlocks = blocks |> List.filter (fun b -> b.Type = MetascriptBlockType.YAML)
            
            for block in yamlBlocks do
                let lines = block.Content.Split([|"\r\n"; "\n"|], StringSplitOptions.None)
                
                for line in lines do
                    let parts = line.Split(':', 2)
                    if parts.Length = 2 then
                        let key = parts.[0].Trim().ToLowerInvariant()
                        let value = parts.[1].Trim()
                        
                        if not (String.IsNullOrWhiteSpace(key)) then
                            metadata.[key] <- value
            
            // Look for metadata in block parameters
            for block in blocks do
                for param in block.Parameters do
                    let key = $"block.{block.Id}.{param.Name}".ToLowerInvariant()
                    metadata.[key] <- param.Value
            
            metadata |> Seq.map (fun kvp -> (kvp.Key, kvp.Value)) |> Map.ofSeq
        with
        | ex ->
            logger.LogError(ex, "Error extracting metadata")
            Map.empty
    
    interface IMetascriptService with
        member this.ParseMetascriptAsync(text, ?name, ?filePath, ?config) = 
            this.ParseMetascriptAsync(text, ?name = name, ?filePath = filePath, ?config = config)
        
        member this.ParseMetascriptFileAsync(filePath, ?config) = 
            this.ParseMetascriptFileAsync(filePath, ?config = config)
        
        member this.ExecuteMetascriptAsync(metascript, ?context) = 
            this.ExecuteMetascriptAsync(metascript, ?context = context)
        
        member this.ExecuteMetascriptTextAsync(text, ?name, ?filePath, ?context, ?config) = 
            this.ExecuteMetascriptTextAsync(text, ?name = name, ?filePath = filePath, ?context = context, ?config = config)
        
        member this.ExecuteMetascriptFileAsync(filePath, ?context, ?config) = 
            this.ExecuteMetascriptFileAsync(filePath, ?context = context, ?config = config)
        
        member this.CreateContextAsync(?workingDirectory, ?variables, ?parent) = 
            this.CreateContextAsync(?workingDirectory = workingDirectory, ?variables = variables, ?parent = parent)
        
        member this.GetDefaultParserConfig() = 
            this.GetDefaultParserConfig()
        
        member this.ValidateMetascriptAsync(metascript) = 
            this.ValidateMetascriptAsync(metascript)
