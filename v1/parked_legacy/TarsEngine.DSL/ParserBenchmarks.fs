namespace TarsEngine.DSL

open System
open System.Diagnostics
open System.IO
open System.Text
open Ast

/// <summary>
/// Module containing benchmarks for the TARS DSL parsers
/// </summary>
module ParserBenchmarks =
    /// <summary>
    /// Generate a TARS program of the specified size
    /// </summary>
    /// <param name="size">The size of the program to generate</param>
    /// <returns>The generated TARS program</returns>
    let generateProgram (size: int) =
        let sb = StringBuilder()
        
        // Add a CONFIG block
        sb.AppendLine("CONFIG {") |> ignore
        sb.AppendLine("    name: \"Generated Program\",") |> ignore
        sb.AppendLine("    version: \"1.0\",") |> ignore
        sb.AppendLine("    description: \"A generated TARS program\"") |> ignore
        sb.AppendLine("}") |> ignore
        sb.AppendLine() |> ignore
        
        // Add VARIABLE blocks
        for i in 1 .. size do
            sb.AppendLine($"VARIABLE x{i} {{") |> ignore
            sb.AppendLine($"    value: {i},") |> ignore
            sb.AppendLine($"    description: \"Variable {i}\"") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        
        // Add FUNCTION blocks
        for i in 1 .. size do
            sb.AppendLine($"FUNCTION add{i} {{") |> ignore
            sb.AppendLine($"    parameters: \"a, b\",") |> ignore
            sb.AppendLine($"    description: \"Adds two numbers\"") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine($"    VARIABLE result {{") |> ignore
            sb.AppendLine($"        value: @a + @b,") |> ignore
            sb.AppendLine($"        description: \"The result of adding a and b\"") |> ignore
            sb.AppendLine($"    }}") |> ignore
            sb.AppendLine() |> ignore
            sb.AppendLine($"    RETURN {{") |> ignore
            sb.AppendLine($"        value: @result") |> ignore
            sb.AppendLine($"    }}") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        
        // Add PROMPT blocks
        for i in 1 .. size do
            sb.AppendLine($"PROMPT prompt{i} {{") |> ignore
            sb.AppendLine("```") |> ignore
            sb.AppendLine($"This is prompt {i}.") |> ignore
            sb.AppendLine($"It can contain multiple lines of text.") |> ignore
            sb.AppendLine($"It can also contain special characters like {{ and }}.") |> ignore
            sb.AppendLine("```") |> ignore
            sb.AppendLine("}") |> ignore
            sb.AppendLine() |> ignore
        
        sb.ToString()
    
    /// <summary>
    /// Benchmark the original parser
    /// </summary>
    /// <param name="code">The TARS program to parse</param>
    /// <param name="iterations">The number of iterations to run</param>
    /// <returns>The average time in milliseconds</returns>
    let benchmarkOriginalParser (code: string) (iterations: int) =
        let stopwatch = Stopwatch()
        
        // Warm up
        let _ = Parser.parse code
        
        // Benchmark
        stopwatch.Start()
        for _ in 1 .. iterations do
            let _ = Parser.parse code
        stopwatch.Stop()
        
        // Return the average time in milliseconds
        float stopwatch.ElapsedMilliseconds / float iterations
    
    /// <summary>
    /// Benchmark the FParsec-based parser
    /// </summary>
    /// <param name="code">The TARS program to parse</param>
    /// <param name="iterations">The number of iterations to run</param>
    /// <returns>The average time in milliseconds</returns>
    let benchmarkFParsecParser (code: string) (iterations: int) =
        let stopwatch = Stopwatch()
        
        // Warm up
        let _ = FParsecParser.parse code
        
        // Benchmark
        stopwatch.Start()
        for _ in 1 .. iterations do
            let _ = FParsecParser.parse code
        stopwatch.Stop()
        
        // Return the average time in milliseconds
        float stopwatch.ElapsedMilliseconds / float iterations
    
    /// <summary>
    /// Run benchmarks for the TARS DSL parsers
    /// </summary>
    /// <param name="sizes">The sizes of the programs to benchmark</param>
    /// <param name="iterations">The number of iterations to run for each size</param>
    let runBenchmarks (sizes: int list) (iterations: int) =
        printfn "Running benchmarks for TARS DSL parsers..."
        printfn "Size\tOriginal (ms)\tFParsec (ms)\tRatio (FParsec/Original)"
        
        for size in sizes do
            // Generate a program of the specified size
            let code = generateProgram size
            
            // Benchmark the original parser
            let originalTime = benchmarkOriginalParser code iterations
            
            // Benchmark the FParsec-based parser
            let fparsecTime = benchmarkFParsecParser code iterations
            
            // Calculate the ratio
            let ratio = fparsecTime / originalTime
            
            // Print the results
            printfn "%d\t%.2f\t\t%.2f\t\t%.2f" size originalTime fparsecTime ratio
    
    /// <summary>
    /// Run benchmarks for the TARS DSL parsers with default parameters
    /// </summary>
    let runDefaultBenchmarks() =
        runBenchmarks [1; 10; 100; 1000] 10
