namespace TarsEngine.DSL

open System
open System.IO
open Ast

/// Module for testing the import and include parsers
module TestImportIncludeParser =
    /// Create test files for imports and includes
    let createTestFiles() =
        // Create a temporary directory for test files
        let tempDir = Path.Combine(Path.GetTempPath(), "TarsEngine.DSL.Tests")
        Directory.CreateDirectory(tempDir) |> ignore
        
        // Create a main file
        let mainFilePath = Path.Combine(tempDir, "main.tars")
        let mainFileContent = """
// Main file
CONFIG {
    name: "Import Include Test",
    version: "1.0"
}

IMPORT {
    "imported.tars"
}

INCLUDE {
    "included.tars"
}

VARIABLE z {
    value: @x + @y
}
"""
        File.WriteAllText(mainFilePath, mainFileContent)
        
        // Create an imported file
        let importedFilePath = Path.Combine(tempDir, "imported.tars")
        let importedFileContent = """
// Imported file
VARIABLE x {
    value: 42,
    description: "Imported variable"
}

FUNCTION add {
    parameters: "a, b",
    
    RETURN {
        value: @a + @b
    }
}
"""
        File.WriteAllText(importedFilePath, importedFileContent)
        
        // Create an included file
        let includedFilePath = Path.Combine(tempDir, "included.tars")
        let includedFileContent = """
// Included file
VARIABLE y {
    value: 10,
    description: "Included variable"
}

FUNCTION multiply {
    parameters: "a, b",
    
    RETURN {
        value: @a * @b
    }
}
"""
        File.WriteAllText(includedFilePath, includedFileContent)
        
        // Return the paths
        (tempDir, mainFilePath, importedFilePath, includedFilePath)
    
    /// Test the import and include parsers with a simple TARS program
    let testImportIncludeParser() =
        let (tempDir, mainFilePath, importedFilePath, includedFilePath) = createTestFiles()
        
        try
            printfn "Parsing with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parseFile mainFilePath
            
            printfn "Parsing with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parseFile mainFilePath
            
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
            
            // Clean up
            Directory.Delete(tempDir, true)
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Clean up
            Directory.Delete(tempDir, true)
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
            
    /// Test the import and include parsers with a more complex TARS program
    let testComplexImportIncludeParser() =
        let (tempDir, mainFilePath, importedFilePath, includedFilePath) = createTestFiles()
        
        // Create a nested import file
        let nestedImportFilePath = Path.Combine(tempDir, "nested_import.tars")
        let nestedImportFileContent = """
// Nested import file
VARIABLE a {
    value: 100,
    description: "Nested imported variable"
}

IMPORT {
    "nested_include.tars"
}
"""
        File.WriteAllText(nestedImportFilePath, nestedImportFileContent)
        
        // Create a nested include file
        let nestedIncludeFilePath = Path.Combine(tempDir, "nested_include.tars")
        let nestedIncludeFileContent = """
// Nested include file
VARIABLE b {
    value: 200,
    description: "Nested included variable"
}

FUNCTION divide {
    parameters: "a, b",
    
    RETURN {
        value: @a / @b
    }
}
"""
        File.WriteAllText(nestedIncludeFilePath, nestedIncludeFileContent)
        
        // Update the main file to import the nested import file
        let mainFileContent = """
// Main file
CONFIG {
    name: "Complex Import Include Test",
    version: "2.0"
}

IMPORT {
    "imported.tars"
}

INCLUDE {
    "included.tars"
}

IMPORT {
    "nested_import.tars"
}

VARIABLE z {
    value: @x + @y + @a + @b
}

FUNCTION calculate {
    parameters: "a, b, c, d",
    
    RETURN {
        value: @add(@a, @b) + @multiply(@c, @d) + @divide(@a, @d)
    }
}
"""
        File.WriteAllText(mainFilePath, mainFileContent)
        
        try
            printfn "Parsing complex program with original parser..."
            // Parse with the original parser
            let originalResult = Parser.parseFile mainFilePath
            
            printfn "Parsing complex program with FParsec-based parser..."
            // Parse with the FParsec-based parser
            let fparsecResult = FParsecParser.parseFile mainFilePath
            
            // Print the results
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Clean up
            Directory.Delete(tempDir, true)
            
            // Return the results
            (originalResult, fparsecResult)
        with
        | ex ->
            printfn "Error: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            
            // Clean up
            Directory.Delete(tempDir, true)
            
            // Return empty results
            let emptyProgram = { Blocks = [] }
            (emptyProgram, emptyProgram)
