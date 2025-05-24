namespace TarsEngine.DSL

open System
open System.Diagnostics
open System.Text
open Ast

/// Module for testing the optimized string parsing
module TestOptimizedStringParsing =
    /// Generate a large content block
    let generateLargeContentBlock (size: int) =
        let sb = StringBuilder()
        
        sb.AppendLine("PROMPT {") |> ignore
        sb.AppendLine("```") |> ignore
        
        for i in 1 .. size do
            sb.AppendLine(sprintf "This is line %d of the content block." i) |> ignore
        
        sb.AppendLine("```") |> ignore
        sb.AppendLine("}") |> ignore
        
        sb.ToString()
    
    /// Test the optimized string parsing with a large content block
    let testOptimizedStringParsing() =
        let sizes = [100; 1000; 10000]
        
        printfn "Testing optimized string parsing with large content blocks..."
        printfn "Size\tOriginal (ms)\tOptimized (ms)\tRatio (Optimized/Original)"
        
        for size in sizes do
            let code = generateLargeContentBlock size
            
            // Test the original parser
            let originalStopwatch = Stopwatch()
            originalStopwatch.Start()
            let originalResult = Parser.parse code
            originalStopwatch.Stop()
            let originalTime = float originalStopwatch.ElapsedMilliseconds
            
            // Test the FParsec-based parser
            let fparsecStopwatch = Stopwatch()
            fparsecStopwatch.Start()
            let fparsecResult = FParsecParser.parse code
            fparsecStopwatch.Stop()
            let fparsecTime = float fparsecStopwatch.ElapsedMilliseconds
            
            // Calculate the ratio
            let ratio = fparsecTime / originalTime
            
            // Print the results
            printfn "%d\t%.2f\t\t%.2f\t\t%.2f" size originalTime fparsecTime ratio
            
            // Print the number of blocks
            printfn "Original parser blocks: %d" originalResult.Blocks.Length
            printfn "FParsec parser blocks: %d" fparsecResult.Blocks.Length
            
            // Print the content length
            if originalResult.Blocks.Length > 0 && fparsecResult.Blocks.Length > 0 then
                printfn "Original content length: %d" originalResult.Blocks.[0].Content.Length
                printfn "FParsec content length: %d" fparsecResult.Blocks.[0].Content.Length
        
        // Return unit
        ()
    
    /// Test the string cache
    let testStringCache() =
        let code = """
CONFIG {
    name: "String Cache Test",
    version: "1.0",
    description: "A test for the string cache"
}

VARIABLE x {
    value: "This is a string that should be cached",
    description: "Variable x"
}

VARIABLE y {
    value: "This is a string that should be cached",
    description: "Variable y"
}

VARIABLE z {
    value: "This is a different string",
    description: "Variable z"
}
"""
        
        // Clear the string cache
        FParsecParser.clearStringCache()
        
        // Parse the code
        let result = FParsecParser.parse code
        
        // Print the results
        printfn "Blocks: %d" result.Blocks.Length
        
        // Return the result
        result
