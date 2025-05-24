namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the variable reference parser
module TestVariableReferenceParser =
    /// Test the variable reference parser with a simple TARS program
    let testVariableReferenceParser() =
        let sampleCode = """
CONFIG {
    name: "Variable Reference Test",
    version: "1.0"
}

VARIABLE x {
    value: 42
}

VARIABLE y {
    value: @x
}

FUNCTION add {
    parameters: "a, b",
    
    VARIABLE result {
        value: @a
    }
    
    RETURN {
        value: @result
    }
}
"""
        
        try
            printfn "Parsing with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse sampleCode
            
            printfn "Parsing with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse sampleCode
            
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
            
    /// Test the variable reference parser with a more complex TARS program
    let testComplexVariableReferenceParser() =
        let complexCode = """
// This is a comment
CONFIG {
    name: "Complex Variable Reference Test",
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
    value: @x,
    description: "A reference to x"
}

FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    VARIABLE result {
        value: @a,
        description: "A reference to parameter a"
    }
    
    RETURN {
        value: @result
    }
}

VARIABLE z {
    value: [1, 2, 3, @x, @y],
    description: "A list with variable references"
}

VARIABLE obj {
    value: {
        a: 1,
        b: @x,
        c: @y,
        d: {
            e: @z,
            f: @add
        }
    },
    description: "An object with variable references"
}
"""
        
        try
            printfn "Parsing complex program with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse complexCode
            
            printfn "Parsing complex program with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse complexCode
            
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
