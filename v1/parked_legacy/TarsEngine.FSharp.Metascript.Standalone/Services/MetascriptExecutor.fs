namespace TarsEngine.FSharp.Metascript.Services

open System
open System.IO
open System.Diagnostics
open System.Threading.Tasks
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.BlockHandlers

/// <summary>
/// Implementation of the IMetascriptExecutor interface.
/// </summary>
type MetascriptExecutor(logger: ILogger<MetascriptExecutor>, registry: BlockHandlerRegistry) =
    
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The execution result.</returns>
    member this.ExecuteAsync(metascript: Metascript, ?context: MetascriptContext) =
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
                        this.CreateContextAsync(workingDirectory)
                
                // Update the context with the current metascript
                let context = { context with CurrentMetascript = Some metascript }
                
                // Start timing
                let stopwatch = Stopwatch.StartNew()
                
                // Execute each block
                let blockResults = List<MetascriptBlockExecutionResult>()
                let mutable variables = context.Variables
                let mutable status = MetascriptExecutionStatus.Success
                let mutable error = None
                let mutable returnValue = None
                
                for block in metascript.Blocks do
                    // Update the context with the current block and variables
                    let blockContext = { context with CurrentBlock = Some block; Variables = variables }
                    
                    // Execute the block
                    let! blockResult = this.ExecuteBlockAsync(block, blockContext)
                    
                    // Add the block result
                    blockResults.Add(blockResult)
                    
                    // Update variables
                    for KeyValue(name, variable) in blockResult.Variables do
                        variables <- Map.add name variable variables
                    
                    // Update status and error
                    if blockResult.Status = MetascriptExecutionStatus.Failure then
                        status <- MetascriptExecutionStatus.Partial
                        error <- blockResult.Error
                    
                    // Update return value
                    returnValue <- blockResult.ReturnValue
                
                // Stop timing
                stopwatch.Stop()
                
                // Combine outputs
                let output = 
                    blockResults
                    |> Seq.map (fun r -> r.Output)
                    |> String.concat Environment.NewLine
                
                // Create the final context
                let finalContext = { context with Variables = variables }
                
                // Create the result
                return {
                    Metascript = metascript
                    BlockResults = blockResults |> Seq.toList
                    Status = status
                    Output = output
                    Error = error
                    ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds
                    ReturnValue = returnValue
                    Variables = variables
                    Context = Some finalContext
                    Metadata = Map.empty
                }
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
    /// Executes a metascript block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    member this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                logger.LogInformation("Executing block: {Type}", block.Type)
                
                // Get a handler for the block
                match registry.GetHandler(block) with
                | Some handler ->
                    // Execute the block
                    return! handler.ExecuteAsync(block, context)
                | None ->
                    // No handler found
                    logger.LogWarning("No handler found for block type: {Type}", block.Type)
                    return {
                        Block = block
                        Output = ""
                        Error = Some $"No handler found for block type: {block.Type}"
                        Status = MetascriptExecutionStatus.NotExecuted
                        ExecutionTimeMs = 0.0
                        ReturnValue = None
                        Variables = Map.empty
                        Metadata = Map.empty
                    }
            with
            | ex ->
                logger.LogError(ex, "Error executing block: {Type}", block.Type)
                return {
                    Block = block
                    Output = ""
                    Error = Some (ex.ToString())
                    Status = MetascriptExecutionStatus.Failure
                    ExecutionTimeMs = 0.0
                    ReturnValue = None
                    Variables = Map.empty
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
        task {
            try
                logger.LogInformation("Creating context")
                
                // Get the working directory
                let workingDirectory = 
                    match workingDirectory with
                    | Some dir -> dir
                    | None -> 
                        match parent with
                        | Some p -> p.WorkingDirectory
                        | None -> Directory.GetCurrentDirectory()
                
                // Get the variables
                let variables = 
                    match variables, parent with
                    | Some vars, Some p -> 
                        // Merge variables, with the provided variables taking precedence
                        p.Variables
                        |> Map.toSeq
                        |> Seq.filter (fun (k, _) -> not (Map.containsKey k vars))
                        |> Seq.append (vars |> Map.toSeq)
                        |> Map.ofSeq
                    | Some vars, None -> vars
                    | None, Some p -> p.Variables
                    | None, None -> Map.empty
                
                // Create the context
                return {
                    Variables = variables
                    Parent = parent
                    WorkingDirectory = workingDirectory
                    CurrentMetascript = None
                    CurrentBlock = None
                    Metadata = Map.empty
                }
            with
            | ex ->
                logger.LogError(ex, "Error creating context")
                return {
                    Variables = Map.empty
                    Parent = None
                    WorkingDirectory = Directory.GetCurrentDirectory()
                    CurrentMetascript = None
                    CurrentBlock = None
                    Metadata = Map.empty
                }
        }
    
    /// <summary>
    /// Gets the supported block types.
    /// </summary>
    /// <returns>The supported block types.</returns>
    member this.GetSupportedBlockTypes() =
        registry.GetSupportedBlockTypes()
    
    /// <summary>
    /// Checks if a block type is supported.
    /// </summary>
    /// <param name="blockType">The block type to check.</param>
    /// <returns>Whether the block type is supported.</returns>
    member this.IsBlockTypeSupported(blockType: MetascriptBlockType) =
        registry.GetSupportedBlockTypes() |> List.contains blockType
    
    interface IMetascriptExecutor with
        member this.ExecuteAsync(metascript, ?context) = 
            this.ExecuteAsync(metascript, ?context = context)
        
        member this.ExecuteBlockAsync(block, context) = 
            this.ExecuteBlockAsync(block, context)
        
        member this.CreateContextAsync(?workingDirectory, ?variables, ?parent) = 
            this.CreateContextAsync(?workingDirectory = workingDirectory, ?variables = variables, ?parent = parent)
        
        member this.GetSupportedBlockTypes() = 
            this.GetSupportedBlockTypes()
        
        member this.IsBlockTypeSupported(blockType) = 
            this.IsBlockTypeSupported(blockType)
