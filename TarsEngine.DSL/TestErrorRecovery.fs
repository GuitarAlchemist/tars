namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the error recovery
module TestErrorRecovery =
    /// Test the error recovery with a simple TARS program
    let testErrorRecovery() =
        let sampleCode = """
CONFIG {
    name: "Error Recovery Test",
    version: "1.0"
}

// Missing closing brace
VARIABLE x {
    value: 42,
    description: "Variable x"

// Missing closing quote
VARIABLE y {
    value: "Hello, world,
    description: "Variable y"
}

// Valid block
VARIABLE z {
    value: 100,
    description: "Variable z"
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
            
    /// Test the error recovery with a more complex TARS program
    let testComplexErrorRecovery() =
        let complexCode = """
CONFIG {
    name: "Complex Error Recovery Test",
    version: "1.0",
    description: "A more complex configuration"
}

// Missing comma
VARIABLE x {
    value: 42
    description: "Variable x"
}

// Missing closing brace for nested block
VARIABLE y {
    value: 10,
    description: "Variable y",
    
    NESTED_BLOCK {
        property1: "value1"
    
}

// Invalid block type
INVALID_BLOCK {
    property1: "value1",
    property2: "value2"
}

// Missing opening brace
VARIABLE z 
    value: 100,
    description: "Variable z"
}

// Valid block
FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    RETURN {
        value: @a + @b
    }
}

// Missing closing quote in string interpolation
VARIABLE message {
    value: "Hello, ${name!",
    description: "A message"
}

// Valid block
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
