namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Threading.Tasks
open TarsEngine.FSharp.Core.Working.Types

/// <summary>
/// Interface for metascript block handlers.
/// </summary>
type IBlockHandler =
    /// <summary>
    /// Gets the block type this handler supports.
    /// </summary>
    abstract member BlockType: string
    
    /// <summary>
    /// Handles the execution of a block.
    /// </summary>
    abstract member HandleAsync: content: string * context: ExecutionContext -> Task<Result<string, TarsError>>
    
    /// <summary>
    /// Validates the block content.
    /// </summary>
    abstract member Validate: content: string -> Result<unit, TarsError>
    
    /// <summary>
    /// Gets the priority of this handler (higher priority handlers are tried first).
    /// </summary>
    abstract member Priority: int
