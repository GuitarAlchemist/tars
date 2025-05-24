namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the expression parser
module TestExpressionParser =
    /// Test the expression parser with a simple TARS program
    let testExpressionParser() =
        let sampleCode = """
CONFIG {
    name: "Expression Test",
    version: "1.0"
}

VARIABLE x {
    value: 42
}

VARIABLE y {
    value: 10
}

VARIABLE z {
    value: x + y
}

VARIABLE a {
    value: x * y
}

VARIABLE b {
    value: x / y
}

VARIABLE c {
    value: x - y
}

VARIABLE d {
    value: x % y
}

VARIABLE e {
    value: x ^ y
}

VARIABLE f {
    value: -x
}

VARIABLE g {
    value: +y
}

VARIABLE h {
    value: (x + y) * (x - y)
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
            
    /// Test the expression parser with a more complex TARS program
    let testComplexExpressionParser() =
        let complexCode = """
// This is a comment
CONFIG {
    name: "Complex Expression Test",
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
    value: x + y * 2,
    description: "A complex expression"
}

VARIABLE a {
    value: (x + y) * 2,
    description: "Another complex expression"
}

VARIABLE b {
    value: x / (y - 5),
    description: "Division with parentheses"
}

VARIABLE c {
    value: -x + y,
    description: "Unary operator"
}

VARIABLE d {
    value: x % y + x ^ y,
    description: "Modulo and power operators"
}

FUNCTION calculate {
    parameters: "a, b",
    description: "Calculates a complex expression",
    
    VARIABLE result {
        value: a * b + (a - b) / (a + b),
        description: "A complex expression with parameters"
    }
    
    RETURN {
        value: result
    }
}

VARIABLE list {
    value: [1, 2, 3, x + y, x * y, -z],
    description: "A list with expressions"
}

VARIABLE obj {
    value: {
        a: 1,
        b: x + y,
        c: x * y,
        d: {
            e: x / y,
            f: x - y
        }
    },
    description: "An object with expressions"
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
