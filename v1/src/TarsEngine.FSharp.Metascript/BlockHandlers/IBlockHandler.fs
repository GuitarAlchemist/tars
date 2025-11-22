namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System.Threading.Tasks
open TarsEngine.FSharp.Metascript

/// <summary>
/// Interface for block handlers.
/// </summary>
type IBlockHandler =
    /// <summary>
    /// Gets the block type that this handler can handle.
    /// </summary>
    abstract member BlockType : MetascriptBlockType
    
    /// <summary>
    /// Checks if this handler can handle the given block.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <returns>Whether this handler can handle the block.</returns>
    abstract member CanHandle : block: MetascriptBlock -> bool
    
    /// <summary>
    /// Executes a block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    abstract member ExecuteAsync : block: MetascriptBlock * context: MetascriptContext -> Task<MetascriptBlockExecutionResult>
    
    /// <summary>
    /// Gets the priority of this handler.
    /// </summary>
    /// <remarks>
    /// Handlers with higher priority are tried first.
    /// </remarks>
    abstract member Priority : int
