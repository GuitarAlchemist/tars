namespace TarsEngine.DSL

/// Module for testing the FParsec-based parser
module TestFParsecParser =
    /// Test the FParsec-based parser with a simple TARS program
    let testParser() =
        let sampleCode = """
CONFIG {
    name: "Sample Config",
    version: "1.0"
}

VARIABLE x {
    value: 42
}

IF {
    condition: "x > 10"
    
    ACTION {
        type: "log",
        message: "x is greater than 10"
    }
}

FUNCTION add {
    parameters: "a, b"
    
    RETURN {
        value: "a + b"
    }
}
"""
        
        // Parse with the original parser
        let originalResult = Parser.parse sampleCode
        
        // Parse with the FParsec-based parser
        let fparsecResult = FParsecParser.parse sampleCode
        
        // Compare the results
        let originalBlocks = originalResult.Blocks
        let fparsecBlocks = fparsecResult.Blocks
        
        printfn "Original parser blocks: %d" originalBlocks.Length
        printfn "FParsec parser blocks: %d" fparsecBlocks.Length
        
        // Compare each block
        for i in 0 .. min (originalBlocks.Length - 1) (fparsecBlocks.Length - 1) do
            let originalBlock = originalBlocks.[i]
            let fparsecBlock = fparsecBlocks.[i]
            
            printfn "Block %d:" i
            printfn "  Original: Type=%A, Name=%A, Properties=%d" originalBlock.Type originalBlock.Name originalBlock.Properties.Count
            printfn "  FParsec:  Type=%A, Name=%A, Properties=%d" fparsecBlock.Type fparsecBlock.Name fparsecBlock.Properties.Count
            
            // Compare properties
            for KeyValue(key, value) in originalBlock.Properties do
                match fparsecBlock.Properties.TryFind key with
                | Some fparsecValue ->
                    if value <> fparsecValue then
                        printfn "    Property '%s': Original=%A, FParsec=%A" key value fparsecValue
                | None ->
                    printfn "    Property '%s' missing in FParsec result" key
            
            // Compare nested blocks
            printfn "  Original nested blocks: %d" originalBlock.NestedBlocks.Length
            printfn "  FParsec nested blocks: %d" fparsecBlock.NestedBlocks.Length
        
        // Return the results
        (originalResult, fparsecResult)
