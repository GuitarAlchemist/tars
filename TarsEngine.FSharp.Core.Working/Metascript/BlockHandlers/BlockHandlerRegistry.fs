namespace TarsEngine.FSharp.Core.Working.Metascript

open System.Collections.Generic
open Microsoft.Extensions.Logging

/// <summary>
/// Registry for metascript block handlers.
/// </summary>
type BlockHandlerRegistry(logger: ILogger<BlockHandlerRegistry>) =
    let handlers = Dictionary<string, IBlockHandler>()
    
    /// <summary>
    /// Registers a block handler.
    /// </summary>
    member _.RegisterHandler(handler: IBlockHandler) =
        handlers.[handler.BlockType.ToUpperInvariant()] <- handler
        logger.LogDebug(sprintf "Registered handler for %s blocks (priority: %d)" handler.BlockType handler.Priority)
    
    /// <summary>
    /// Gets a handler for the specified block type.
    /// </summary>
    member _.GetHandler(blockType: string) =
        match handlers.TryGetValue(blockType.ToUpperInvariant()) with
        | true, handler -> Some handler
        | false, _ -> None
    
    /// <summary>
    /// Gets all registered handlers.
    /// </summary>
    member _.GetAllHandlers() =
        handlers.Values |> Seq.toList
    
    /// <summary>
    /// Registers default handlers.
    /// </summary>
    member this.RegisterDefaultHandlers() =
        let configHandler = ConfigBlockHandler(logger :?> ILogger<ConfigBlockHandler>)
        let fsharpHandler = FSharpBlockHandler(logger :?> ILogger<FSharpBlockHandler>)
        let commandHandler = CommandBlockHandler(logger :?> ILogger<CommandBlockHandler>)
        let textHandler = TextBlockHandler(logger :?> ILogger<TextBlockHandler>)
        let mlHandler = MLBlockHandler(logger :?> ILogger<MLBlockHandler>)
        
        this.RegisterHandler(configHandler)
        this.RegisterHandler(fsharpHandler)
        this.RegisterHandler(commandHandler)
        this.RegisterHandler(textHandler)
        this.RegisterHandler(mlHandler)
        
        logger.LogInformation(sprintf "Registered %d default block handlers" handlers.Count)
