namespace TarsEngine.DSL

open System
open Ast

/// Module for testing custom block types
module TestCustomBlockTypes =
    /// Test custom block types
    let testCustomBlockTypes() =
        let sampleCode = """
// Register a custom block type
CUSTOM_BLOCK {
    name: "Custom Block",
    description: "A custom block type"
}

// Use the custom block type
CUSTOM_BLOCK custom_block {
    value: 42,
    description: "A custom block instance"
}

// Use the custom block type with content
CUSTOM_BLOCK {
```
This is a content block in a custom block type.
It can contain multiple lines of text.
It can also contain special characters like { and }.
```
}

// Use another custom block type
ANOTHER_CUSTOM_BLOCK {
    name: "Another Custom Block",
    description: "Another custom block type"
}
"""
        
        try
            // Clear any existing custom block types
            FParsecParser.clearCustomBlockTypes()
            
            printfn "Parsing with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse sampleCode
            
            printfn "Parsing with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse sampleCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Print the custom block types
            printfn "Custom block types:"
            for (name, blockType) in FParsecParser.getAllCustomBlockTypes() do
                printfn "  %s: %A" name blockType
            
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
            
    /// Test custom block types with nested blocks
    let testCustomBlockTypesWithNestedBlocks() =
        let complexCode = """
// Register a custom block type
CUSTOM_BLOCK {
    name: "Custom Block",
    description: "A custom block type",
    
    // Nested block
    NESTED_BLOCK {
        name: "Nested Block",
        description: "A nested block"
    }
}

// Use the custom block type with nested blocks
CUSTOM_BLOCK custom_block {
    value: 42,
    description: "A custom block instance",
    
    // Nested block
    NESTED_BLOCK {
        name: "Nested Block",
        description: "A nested block"
    }
    
    // Another nested block
    ANOTHER_NESTED_BLOCK {
        name: "Another Nested Block",
        description: "Another nested block"
    }
}

// Use another custom block type
ANOTHER_CUSTOM_BLOCK {
    name: "Another Custom Block",
    description: "Another custom block type",
    
    // Nested custom block
    CUSTOM_BLOCK {
        name: "Nested Custom Block",
        description: "A nested custom block"
    }
}
"""
        
        try
            // Clear any existing custom block types
            FParsecParser.clearCustomBlockTypes()
            
            printfn "Parsing complex program with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parse complexCode
            
            printfn "Parsing complex program with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parse complexCode
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Print the custom block types
            printfn "Custom block types:"
            for (name, blockType) in FParsecParser.getAllCustomBlockTypes() do
                printfn "  %s: %A" name blockType
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
