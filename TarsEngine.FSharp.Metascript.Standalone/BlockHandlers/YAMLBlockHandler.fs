namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Handler for YAML blocks.
/// </summary>
type YAMLBlockHandler(logger: ILogger<YAMLBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.YAML, 20)
    
    /// <summary>
    /// Executes a YAML block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            // YAML blocks don't execute, they just output their content
            return this.CreateSuccessResult(block, block.Content)
        }
