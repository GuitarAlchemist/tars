namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Handler for TARS-specific blocks.
/// </summary>
type TarsBlockHandler(logger: ILogger<TarsBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.Unknown, 50)
    
    /// <summary>
    /// Executes a TARS block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            try
                // For now, we just output the block content
                let output = 
                    match block.Type with
                    | MetascriptBlockType.Describe -> $"DESCRIBE block: {block.Content}"
                    | MetascriptBlockType.Config -> $"CONFIG block: {block.Content}"
                    | MetascriptBlockType.Variable -> $"VARIABLE block: {block.Content}"
                    | MetascriptBlockType.Agent -> $"AGENT block: {block.Content}"
                    | MetascriptBlockType.Task -> $"TASK block: {block.Content}"
                    | MetascriptBlockType.Function -> $"FUNCTION block: {block.Content}"
                    | MetascriptBlockType.Action -> $"ACTION block: {block.Content}"
                    | MetascriptBlockType.Return -> $"RETURN block: {block.Content}"
                    | _ -> $"Unknown TARS block type: {block.Type}"
                
                return this.CreateSuccessResult(block, output)
            with
            | ex ->
                logger.LogError(ex, "Error executing TARS block")
                return this.CreateFailureResult(block, ex.ToString())
        }
    
    /// <summary>
    /// Checks if this handler can handle the given block.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <returns>Whether this handler can handle the block.</returns>
    member this.CanHandle(block: MetascriptBlock) =
        match block.Type with
        | MetascriptBlockType.Describe
        | MetascriptBlockType.Config
        | MetascriptBlockType.Variable
        | MetascriptBlockType.Agent
        | MetascriptBlockType.Task
        | MetascriptBlockType.Function
        | MetascriptBlockType.Action
        | MetascriptBlockType.Return -> true
        | _ -> false
