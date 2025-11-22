namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System
open System.Diagnostics
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Base class for block handlers.
/// </summary>
[<AbstractClass>]
type BlockHandlerBase(logger: ILogger, blockType: MetascriptBlockType, priority: int) =
    /// <summary>
    /// Gets the block type that this handler can handle.
    /// </summary>
    member this.BlockType = blockType
    
    /// <summary>
    /// Gets the priority of this handler.
    /// </summary>
    member this.Priority = priority
    
    /// <summary>
    /// Checks if this handler can handle the given block.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <returns>Whether this handler can handle the block.</returns>
    member this.CanHandle(block: MetascriptBlock) =
        block.Type = blockType
    
    /// <summary>
    /// Executes a block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    abstract member ExecuteBlockAsync : block: MetascriptBlock * context: MetascriptContext -> Task<MetascriptBlockExecutionResult>
    
    /// <summary>
    /// Creates a successful block execution result.
    /// </summary>
    /// <param name="block">The block that was executed.</param>
    /// <param name="output">The output of the block execution.</param>
    /// <param name="returnValue">The return value, if any.</param>
    /// <param name="variables">The variables created or modified by the block.</param>
    /// <param name="executionTimeMs">The execution time in milliseconds.</param>
    /// <returns>The block execution result.</returns>
    member this.CreateSuccessResult(block: MetascriptBlock, output: string, ?returnValue: obj, ?variables: Map<string, MetascriptVariable>, ?executionTimeMs: float) =
        {
            Block = block
            Output = output
            Error = None
            Status = MetascriptExecutionStatus.Success
            ExecutionTimeMs = defaultArg executionTimeMs 0.0
            ReturnValue = returnValue
            Variables = defaultArg variables Map.empty
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Creates a failure block execution result.
    /// </summary>
    /// <param name="block">The block that was executed.</param>
    /// <param name="error">The error message.</param>
    /// <param name="output">The output of the block execution.</param>
    /// <param name="variables">The variables created or modified by the block.</param>
    /// <param name="executionTimeMs">The execution time in milliseconds.</param>
    /// <returns>The block execution result.</returns>
    member this.CreateFailureResult(block: MetascriptBlock, error: string, ?output: string, ?variables: Map<string, MetascriptVariable>, ?executionTimeMs: float) =
        {
            Block = block
            Output = defaultArg output ""
            Error = Some error
            Status = MetascriptExecutionStatus.Failure
            ExecutionTimeMs = defaultArg executionTimeMs 0.0
            ReturnValue = None
            Variables = defaultArg variables Map.empty
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Creates a not executed block execution result.
    /// </summary>
    /// <param name="block">The block that was not executed.</param>
    /// <param name="reason">The reason why the block was not executed.</param>
    /// <returns>The block execution result.</returns>
    member this.CreateNotExecutedResult(block: MetascriptBlock, reason: string) =
        {
            Block = block
            Output = ""
            Error = Some reason
            Status = MetascriptExecutionStatus.NotExecuted
            ExecutionTimeMs = 0.0
            ReturnValue = None
            Variables = Map.empty
            Metadata = Map.empty
        }
    
    /// <summary>
    /// Executes a block with timing.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    member this.ExecuteAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                logger.LogInformation("Executing block: {Type}", block.Type)
                
                // Start timing
                let stopwatch = Stopwatch.StartNew()
                
                // Execute the block
                let! result = this.ExecuteBlockAsync(block, context)
                
                // Stop timing
                stopwatch.Stop()
                
                // Update the execution time
                return { result with ExecutionTimeMs = stopwatch.Elapsed.TotalMilliseconds }
            with
            | ex ->
                logger.LogError(ex, "Error executing block: {Type}", block.Type)
                return this.CreateFailureResult(block, ex.ToString())
        }
    
    interface IBlockHandler with
        member this.BlockType = this.BlockType
        member this.CanHandle(block) = this.CanHandle(block)
        member this.ExecuteAsync(block, context) = this.ExecuteAsync(block, context)
        member this.Priority = this.Priority
