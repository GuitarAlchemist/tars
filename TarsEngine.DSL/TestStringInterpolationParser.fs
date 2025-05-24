namespace TarsEngine.DSL

open System
open Ast

/// Module for testing the string interpolation parser
module TestStringInterpolationParser =
    /// Test the string interpolation parser with a simple TARS program
    let testStringInterpolationParser() =
        let sampleCode = """
CONFIG {
    name: "String Interpolation Test",
    version: "1.0"
}

VARIABLE x {
    value: "Hello, ${name}!"
}

VARIABLE y {
    value: "The answer is ${answer}."
}

FUNCTION greet {
    parameters: "name",
    
    RETURN {
        value: "Hello, ${name}!"
    }
}

VARIABLE z {
    value: """This is a raw string literal.
It can contain multiple lines of text.
It can also contain special characters like { and }.
It can even contain quotes like " without escaping.
"""
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
            
    /// Test the string interpolation parser with a more complex TARS program
    let testComplexStringInterpolationParser() =
        let complexCode = """
// This is a comment
CONFIG {
    name: "Complex String Interpolation Test",
    version: "2.0",
    description: "A more complex configuration"
}

/* This is a multi-line comment
   that spans multiple lines */
VARIABLE x {
    value: "Hello, ${name}!",
    description: "A greeting with interpolation"
}

VARIABLE y {
    value: "The answer is ${answer}.",
    description: "A statement with interpolation"
}

FUNCTION greet {
    parameters: "name",
    description: "Greets a person",
    
    VARIABLE greeting {
        value: "Hello, ${name}!"
    }
    
    RETURN {
        value: "greeting"
    }
}

VARIABLE z {
    value: """This is a raw string literal.
It can contain multiple lines of text.
It can also contain special characters like { and }.
It can even contain quotes like " without escaping.
It can also contain interpolation like ${name}.
""",
    description: "A raw string literal with interpolation"
}

AGENT greeter {
    name: "Greeter Agent",
    description: "An agent that greets people",
    
    FUNCTION greet {
        parameters: "name",
        description: "Greets a person",
        
        RETURN {
            value: "Hello, ${name}!"
        }
    }
    
    FUNCTION farewell {
        parameters: "name",
        description: "Says goodbye to a person",
        
        RETURN {
            value: "Goodbye, ${name}!"
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
