namespace TarsEngine.FSharp.TARSX.MetaBlocks

/// Meta Block Processor
/// Processes META blocks for script configuration
module MetaBlockProcessor =
    
    /// Process meta block
    let processMetaBlock (properties: (string * obj) list) : Map<string, obj> =
        properties |> Map.ofList
    
    printfn "📋 Meta Block Processor Loaded"
