namespace TarsEngine.FSharp.Metascript.Services

open System.Threading.Tasks
open TarsEngine.FSharp.Metascript

/// <summary>
/// Interface for metascript executors.
/// </summary>
type IMetascriptExecutor =
    /// <summary>
    /// Executes a metascript.
    /// </summary>
    /// <param name="metascript">The metascript to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The execution result.</returns>
    abstract member ExecuteAsync : metascript: Metascript * ?context: MetascriptContext -> Task<MetascriptExecutionResult>
    
    /// <summary>
    /// Executes a metascript block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    abstract member ExecuteBlockAsync : block: MetascriptBlock * context: MetascriptContext -> Task<MetascriptBlockExecutionResult>
    
    /// <summary>
    /// Creates a new metascript context.
    /// </summary>
    /// <param name="workingDirectory">The working directory.</param>
    /// <param name="variables">The initial variables.</param>
    /// <param name="parent">The parent context.</param>
    /// <returns>The new context.</returns>
    abstract member CreateContextAsync : ?workingDirectory: string * ?variables: Map<string, MetascriptVariable> * ?parent: MetascriptContext -> Task<MetascriptContext>
    
    /// <summary>
    /// Gets the supported block types.
    /// </summary>
    /// <returns>The supported block types.</returns>
    abstract member GetSupportedBlockTypes : unit -> MetascriptBlockType list
    
    /// <summary>
    /// Checks if a block type is supported.
    /// </summary>
    /// <param name="blockType">The block type to check.</param>
    /// <returns>Whether the block type is supported.</returns>
    abstract member IsBlockTypeSupported : blockType: MetascriptBlockType -> bool
