namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Base class for metascript block handlers.
/// </summary>
[<AbstractClass>]
type BlockHandlerBase(logger: ILogger, blockType: string, priority: int) =
    
    /// <summary>
    /// Gets the logger.
    /// </summary>
    member _.Logger = logger
    
    /// <summary>
    /// Logs the start of block execution.
    /// </summary>
    member _.LogExecutionStart(content: string) =
        logger.LogDebug(sprintf "Starting execution of %s block (length: %d)" blockType content.Length)
    
    /// <summary>
    /// Logs the completion of block execution.
    /// </summary>
    member _.LogExecutionComplete(success: bool, duration: System.TimeSpan) =
        let status = if success then "completed" else "failed"
        logger.LogDebug(sprintf "%s block execution %s in %dms" blockType status (int duration.TotalMilliseconds))
    
    /// <summary>
    /// Creates a validation error.
    /// </summary>
    member _.CreateValidationError(message: string) =
        createError (sprintf "%s block validation failed: %s" blockType message) None
    
    /// <summary>
    /// Creates an execution error.
    /// </summary>
    member _.CreateExecutionError(message: string, details: string option) =
        createError (sprintf "%s block execution failed: %s" blockType message) details
    
    interface IBlockHandler with
        member _.BlockType = blockType
        member _.Priority = priority
        
        member this.HandleAsync(content, context) =
            task {
                let startTime = System.DateTime.UtcNow
                this.LogExecutionStart(content)
                
                try
                    let! result = this.ExecuteBlockAsync(content, context)
                    let duration = System.DateTime.UtcNow - startTime
                    this.LogExecutionComplete(Result.isOk result, duration)
                    return result
                with
                | ex ->
                    let duration = System.DateTime.UtcNow - startTime
                    this.LogExecutionComplete(false, duration)
                    return Error (this.CreateExecutionError(ex.Message, Some ex.StackTrace))
            }
        
        member this.Validate(content) =
            try
                this.ValidateBlock(content)
            with
            | ex ->
                Error (this.CreateValidationError(ex.Message))
    
    /// <summary>
    /// Abstract method for executing the block.
    /// </summary>
    abstract member ExecuteBlockAsync: content: string * context: ExecutionContext -> Task<Result<string, TarsError>>
    
    /// <summary>
    /// Abstract method for validating the block.
    /// </summary>
    abstract member ValidateBlock: content: string -> Result<unit, TarsError>
