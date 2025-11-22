namespace TarsEngine.DSL

open System
open Ast
open ParserConfiguration

/// Module for testing the unified parser
module TestUnifiedParser =
    /// Test the unified parser with a simple TARS program
    let testUnifiedParser() =
        let sampleCode = """
CONFIG {
    name: "Unified Parser Test",
    version: "1.0"
}

VARIABLE x {
    value: 42
}

FUNCTION add {
    parameters: "a, b",
    
    RETURN {
        value: @a + @b
    }
}
"""
        
        try
            printfn "Parsing with original parser..."
            // Set the parser type to Original
            ParserConfiguration.setParserType ParserType.Original
            
            // Parse with the unified parser
            let originalResult = UnifiedParser.parseWithCurrentConfig sampleCode
            
            printfn "Parsing with FParsec-based parser..."
            // Set the parser type to FParsec
            ParserConfiguration.setParserType ParserType.FParsec
            
            // Parse with the unified parser
            let fparsecResult = UnifiedParser.parseWithCurrentConfig sampleCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Compare each block
            for i in 0 .. min (originalResult.Blocks.Length - 1) (fparsecResult.Blocks.Length - 1) do
                let originalBlock = originalResult.Blocks.[i]
                let fparsecBlock = fparsecResult.Blocks.[i]
                
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
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
            
    /// Test the unified parser with a more complex TARS program
    let testComplexUnifiedParser() =
        let complexCode = """
// This is a comment
CONFIG {
    name: "Complex Unified Parser Test",
    version: "2.0",
    description: "A more complex configuration"
}

/* This is a multi-line comment
   that spans multiple lines */
VARIABLE x {
    value: 42,
    description: "The answer to life, the universe, and everything"
}

VARIABLE y {
    value: 10,
    description: "Just a number"
}

VARIABLE z {
    value: x + y,
    description: "A complex expression"
}

FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    VARIABLE result {
        value: @a + @b,
        description: "The result of adding a and b"
    }
    
    RETURN {
        value: @result
    }
}

PROMPT {
```
This is a content block.
It can contain multiple lines of text.
It can also contain special characters like { and }.
```
}
"""
        
        try
            printfn "Parsing complex program with original parser..."
            // Set the parser type to Original
            ParserConfiguration.setParserType ParserType.Original
            
            // Parse with the unified parser
            let originalResult = UnifiedParser.parseWithCurrentConfig complexCode
            
            printfn "Parsing complex program with FParsec-based parser..."
            // Set the parser type to FParsec
            ParserConfiguration.setParserType ParserType.FParsec
            
            // Parse with the unified parser
            let fparsecResult = UnifiedParser.parseWithCurrentConfig complexCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
