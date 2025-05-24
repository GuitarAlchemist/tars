namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Handler for text blocks.
/// </summary>
type TextBlockHandler(logger: ILogger<TextBlockHandler>) =
    inherit BlockHandlerBase(logger, MetascriptBlockType.Text, 10)
    
    /// <summary>
    /// Executes a text block.
    /// </summary>
    /// <param name="block">The block to execute.</param>
    /// <param name="context">The execution context.</param>
    /// <returns>The block execution result.</returns>
    override this.ExecuteBlockAsync(block: MetascriptBlock, context: MetascriptContext) =
        task {
            // Text blocks don't execute, they just output their content
            return this.CreateSuccessResult(block, block.Content)
        }
    
    /// <summary>
    /// Checks if this handler can handle the given block.
    /// </summary>
    /// <param name="block">The block to check.</param>
    /// <returns>Whether this handler can handle the block.</returns>
    override this.CanHandle(block: MetascriptBlock) =
        block.Type = MetascriptBlockType.Text ||
        block.Type = MetascriptBlockType.Markdown ||
        block.Type = MetascriptBlockType.HTML ||
        block.Type = MetascriptBlockType.CSS ||
        block.Type = MetascriptBlockType.JSON ||
        block.Type = MetascriptBlockType.XML ||
        block.Type = MetascriptBlockType.YAML
