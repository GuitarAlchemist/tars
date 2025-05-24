namespace TarsEngine.FSharp.Metascript.BlockHandlers

open System
open System.Collections.Generic
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript

/// <summary>
/// Registry for block handlers.
/// </summary>
type BlockHandlerRegistry(logger: ILogger<BlockHandlerRegistry>) =
    let handlers = List<IBlockHandler>()
    
    /// <summary>
    /// Registers a block handler.
    /// </summary>
    /// <param name="handler">The handler to register.</param>
    member this.RegisterHandler(handler: IBlockHandler) =
        logger.LogInformation("Registering handler for block type: {BlockType}", handler.BlockType)
        handlers.Add(handler)
        
        // Sort handlers by priority (descending)
        handlers.Sort(fun a b -> b.Priority.CompareTo(a.Priority))
    
    /// <summary>
    /// Gets a handler for the given block.
    /// </summary>
    /// <param name="block">The block to get a handler for.</param>
    /// <returns>The handler, or None if no handler was found.</returns>
    member this.GetHandler(block: MetascriptBlock) =
        handlers
        |> Seq.tryFind (fun h -> h.CanHandle(block))
    
    /// <summary>
    /// Gets all registered handlers.
    /// </summary>
    /// <returns>The registered handlers.</returns>
    member this.GetAllHandlers() =
        handlers |> Seq.toList
    
    /// <summary>
    /// Gets all supported block types.
    /// </summary>
    /// <returns>The supported block types.</returns>
    member this.GetSupportedBlockTypes() =
        handlers
        |> Seq.map (fun h -> h.BlockType)
        |> Seq.distinct
        |> Seq.toList
