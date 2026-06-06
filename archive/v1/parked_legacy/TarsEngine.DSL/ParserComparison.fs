namespace TarsEngine.DSL

/// Module for comparing the original parser with the FParsec-based parser
module ParserComparison =
    /// Compare the results of parsing a TARS program with both parsers
    let compareParseResults (code: string) =
        try
            // Parse with the original parser
            let originalResult = Parser.parse code
            
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse code
            
            // Compare the results
            let originalBlocks = originalResult.Blocks
            let fparsecBlocks = fparsecResult.Blocks
            
            if originalBlocks.Length <> fparsecBlocks.Length then
                printfn "Different number of blocks: Original=%d, FParsec=%d" originalBlocks.Length fparsecBlocks.Length
                false
            else
                // Compare each block
                let mutable allEqual = true
                for i in 0 .. originalBlocks.Length - 1 do
                    let originalBlock = originalBlocks.[i]
                    let fparsecBlock = fparsecBlocks.[i]
                    
                    if originalBlock.Type <> fparsecBlock.Type then
                        printfn "Block %d: Different types: Original=%A, FParsec=%A" i originalBlock.Type fparsecBlock.Type
                        allEqual <- false
                    
                    if originalBlock.Name <> fparsecBlock.Name then
                        printfn "Block %d: Different names: Original=%A, FParsec=%A" i originalBlock.Name fparsecBlock.Name
                        allEqual <- false
                    
                    if originalBlock.Properties.Count <> fparsecBlock.Properties.Count then
                        printfn "Block %d: Different number of properties: Original=%d, FParsec=%d" i originalBlock.Properties.Count fparsecBlock.Properties.Count
                        allEqual <- false
                    else
                        for KeyValue(key, value) in originalBlock.Properties do
                            match fparsecBlock.Properties.TryFind key with
                            | Some fparsecValue ->
                                if value <> fparsecValue then
                                    printfn "Block %d: Different property values for '%s': Original=%A, FParsec=%A" i key value fparsecValue
                                    allEqual <- false
                            | None ->
                                printfn "Block %d: Property '%s' missing in FParsec result" i key
                                allEqual <- false
                    
                    if originalBlock.NestedBlocks.Length <> fparsecBlock.NestedBlocks.Length then
                        printfn "Block %d: Different number of nested blocks: Original=%d, FParsec=%d" i originalBlock.NestedBlocks.Length fparsecBlock.NestedBlocks.Length
                        allEqual <- false
                
                allEqual
        with
        | ex ->
            printfn "Exception during comparison: %s" ex.Message
            false
    
    /// Run a simple test with a sample TARS program
    let runSimpleTest() =
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
        
        let result = compareParseResults sampleCode
        
        if result then
            printfn "Test passed: Both parsers produced the same result"
        else
            printfn "Test failed: Parsers produced different results"
        
        result
