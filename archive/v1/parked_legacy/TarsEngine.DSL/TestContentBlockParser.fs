namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the content block parser
module TestContentBlockParser =
    /// Test the content block parser with a simple TARS program
    let testContentBlockParser() =
        let sampleCode = """
CONFIG {
    name: "Content Block Test",
    version: "1.0"
}

PROMPT {
```
This is a content block.
It can contain multiple lines of text.
It can also contain special characters like { and }.
```
}

FUNCTION add {
    parameters: "a, b",
    
    RETURN {
        value: "a + b"
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
                printfn "  Original: Type=%A, Name=%A, Properties=%d, Content=%s" 
                    originalBlock.Type originalBlock.Name originalBlock.Properties.Count 
                    (if originalBlock.Content.Length > 20 then originalBlock.Content.Substring(0, 20) + "..." else originalBlock.Content)
                printfn "  FParsec:  Type=%A, Name=%A, Properties=%d, Content=%s" 
                    fparsecBlock.Type fparsecBlock.Name fparsecBlock.Properties.Count
                    (if fparsecBlock.Content.Length > 20 then fparsecBlock.Content.Substring(0, 20) + "..." else fparsecBlock.Content)
                
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
            
    /// Test the content block parser with a more complex TARS program
    let testComplexContentBlockParser() =
        let complexCode = """
// This is a comment
CONFIG {
    name: "Complex Content Block Test",
    version: "2.0",
    description: "A more complex configuration"
}

/* This is a multi-line comment
   that spans multiple lines */
PROMPT {
```
This is a complex content block.
It can contain multiple lines of text.
It can also contain special characters like { and }.
It can even contain code:

function hello() {
    console.log("Hello, world!");
}

Or markdown:

# Heading 1
## Heading 2

- List item 1
- List item 2
```
}

FUNCTION add {
    parameters: "a, b",
    description: "Adds two numbers",
    
    VARIABLE result {
        value: "a + b"
    }
    
    RETURN {
        value: "result"
    }
}

AGENT calculator {
    name: "Calculator Agent",
    description: "An agent that performs calculations",
    
    FUNCTION multiply {
        parameters: "a, b",
        description: "Multiplies two numbers",
        
        RETURN {
```
return a * b;
```
        }
    }
    
    FUNCTION divide {
        parameters: "a, b",
        description: "Divides two numbers",
        
        IF {
            condition: "b == 0",
            
            RETURN {
                value: "Error: Division by zero"
            }
        }
        
        RETURN {
            value: "a / b"
        }
    }
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
