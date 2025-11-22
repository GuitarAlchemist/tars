#!/usr/bin/env dotnet fsi

// TARS Fractal Grammar CLI Tool
// Comprehensive command-line interface for fractal grammar operations

#r "nuget: System.CommandLine, 2.0.0-beta4.22272.1"
#r "../Tars.Engine.Grammar/bin/Debug/net8.0/Tars.Engine.Grammar.dll"

open System
open System.IO
open System.CommandLine
open Tars.Engine.Grammar.FractalGrammar
open Tars.Engine.Grammar.FractalGrammarParser
open Tars.Engine.Grammar.FractalGrammarIntegration

/// CLI Application for Fractal Grammars
module FractalGrammarCLI =

    /// Print colored output to console
    let printColored color text =
        let originalColor = Console.ForegroundColor
        Console.ForegroundColor <- color
        Console.WriteLine(text)
        Console.ForegroundColor <- originalColor

    /// Print success message
    let printSuccess text = printColored ConsoleColor.Green text

    /// Print error message
    let printError text = printColored ConsoleColor.Red text

    /// Print warning message
    let printWarning text = printColored ConsoleColor.Yellow text

    /// Print info message
    let printInfo text = printColored ConsoleColor.Cyan text

    /// Generate fractal grammar command
    let generateCommand =
        let inputOption = Option<string>([|"-i"; "--input"|], "Input fractal grammar file")
        let outputOption = Option<string>([|"-o"; "--output"|], "Output file path")
        let formatOption = Option<string>([|"-f"; "--format"|], "Output format (EBNF, ANTLR, JSON, XML, GraphViz, SVG)")
        let depthOption = Option<int>([|"-d"; "--depth"|], "Maximum recursion depth")
        let visualizeOption = Option<bool>([|"-v"; "--visualize"|], "Generate visualization")

        inputOption.IsRequired <- true
        outputOption.IsRequired <- true
        formatOption.SetDefaultValue("EBNF")
        depthOption.SetDefaultValue(5)
        visualizeOption.SetDefaultValue(false)

        let command = Command("generate", "Generate fractal grammar from specification")
        command.AddOption(inputOption)
        command.AddOption(outputOption)
        command.AddOption(formatOption)
        command.AddOption(depthOption)
        command.AddOption(visualizeOption)

        command.SetHandler(fun input output format depth visualize ->
            try
                printInfo $"🔄 Generating fractal grammar from: {input}"
                
                let manager = FractalGrammarManager()
                let parseResult = manager.ParseFractalGrammarFile(input)
                
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        // Update recursion depth if specified
                        let updatedGrammar = 
                            if depth <> 5 then
                                let updatedRules = 
                                    fractalGrammar.FractalRules
                                    |> List.map (fun rule -> 
                                        { rule with Properties = { rule.Properties with RecursionLimit = depth } })
                                { fractalGrammar with FractalRules = updatedRules }
                            else
                                fractalGrammar
                        
                        // Parse output format
                        let outputFormat = 
                            match format.ToUpperInvariant() with
                            | "EBNF" -> EBNF
                            | "ANTLR" -> ANTLR
                            | "JSON" -> JSON
                            | "XML" -> XML
                            | "GRAPHVIZ" -> GraphViz
                            | "SVG" -> SVG
                            | _ -> 
                                printWarning $"Unknown format '{format}', using EBNF"
                                EBNF
                        
                        let context = { 
                            manager.CreateDefaultContext() with 
                                OutputFormat = outputFormat
                                EnableVisualization = visualize
                                MaxIterations = depth * 100
                        }
                        
                        let result = manager.ExecuteFractalGrammar(updatedGrammar, context)
                        
                        if result.Success then
                            File.WriteAllText(output, result.GeneratedGrammar)
                            printSuccess $"✅ Fractal grammar generated: {output}"
                            
                            printInfo $"📊 Generation Statistics:"
                            printInfo $"   Execution Time: {result.ExecutionTime.TotalMilliseconds:F1}ms"
                            printInfo $"   Memory Used: {result.MemoryUsed / 1024L} KB"
                            printInfo $"   Iterations: {result.IterationsCompleted}"
                            printInfo $"   Fractal Dimension: {result.FractalDimension:F3}"
                            
                            if visualize && result.VisualizationData.IsSome then
                                let vizFile = Path.ChangeExtension(output, ".viz.svg")
                                File.WriteAllText(vizFile, result.VisualizationData.Value)
                                printSuccess $"🎨 Visualization saved: {vizFile}"
                            
                            if not result.Warnings.IsEmpty then
                                printWarning "⚠️  Warnings:"
                                result.Warnings |> List.iter (fun w -> printWarning $"   {w}")
                        else
                            printError "❌ Generation failed:"
                            result.ErrorMessages |> List.iter (fun e -> printError $"   {e}")
                    | None ->
                        printError "❌ No fractal grammar parsed"
                else
                    printError "❌ Parse errors:"
                    parseResult.ErrorMessages |> List.iter (fun e -> printError $"   {e}")
            with
            | ex ->
                printError $"❌ Error: {ex.Message}"
        , inputOption, outputOption, formatOption, depthOption, visualizeOption)

        command

    /// Parse fractal grammar command
    let parseCommand =
        let inputOption = Option<string>([|"-i"; "--input"|], "Input fractal grammar file")
        let verboseOption = Option<bool>([|"-v"; "--verbose"|], "Verbose output")

        inputOption.IsRequired <- true
        verboseOption.SetDefaultValue(false)

        let command = Command("parse", "Parse and validate fractal grammar file")
        command.AddOption(inputOption)
        command.AddOption(verboseOption)

        command.SetHandler(fun input verbose ->
            try
                printInfo $"🔍 Parsing fractal grammar: {input}"
                
                let manager = FractalGrammarManager()
                let result = manager.ParseFractalGrammarFile(input)
                
                if result.Success then
                    printSuccess "✅ Parse successful!"
                    printInfo $"📊 Parse Statistics:"
                    printInfo $"   Rules found: {result.ParsedRules.Length}"
                    printInfo $"   Parse time: {result.ParseTime.TotalMilliseconds:F1}ms"
                    
                    if verbose then
                        printInfo "\n📋 Parsed Rules:"
                        result.ParsedRules |> List.iteri (fun i rule ->
                            printInfo $"   {i+1}. {rule.Name}"
                            printInfo $"      Pattern: {rule.BasePattern}"
                            printInfo $"      Dimension: {rule.Properties.Dimension:F3}"
                            printInfo $"      Depth: {rule.Properties.RecursionLimit}"
                            printInfo $"      Transformations: {rule.Transformations.Length}")
                    
                    if not result.Warnings.IsEmpty then
                        printWarning "\n⚠️  Warnings:"
                        result.Warnings |> List.iter (fun w -> printWarning $"   {w}")
                else
                    printError "❌ Parse failed:"
                    result.ErrorMessages |> List.iter (fun e -> printError $"   {e}")
            with
            | ex ->
                printError $"❌ Error: {ex.Message}"
        , inputOption, verboseOption)

        command

    /// Analyze fractal grammar command
    let analyzeCommand =
        let inputOption = Option<string>([|"-i"; "--input"|], "Input fractal grammar file")
        let detailedOption = Option<bool>([|"-d"; "--detailed"|], "Detailed analysis")

        inputOption.IsRequired <- true
        detailedOption.SetDefaultValue(false)

        let command = Command("analyze", "Analyze fractal grammar complexity and properties")
        command.AddOption(inputOption)
        command.AddOption(detailedOption)

        command.SetHandler(fun input detailed ->
            try
                printInfo $"📊 Analyzing fractal grammar: {input}"
                
                let manager = FractalGrammarManager()
                let parseResult = manager.ParseFractalGrammarFile(input)
                
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        let context = manager.CreateDefaultContext()
                        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
                        
                        printSuccess "✅ Analysis complete!"
                        printInfo "\n📊 FRACTAL GRAMMAR ANALYSIS"
                        printInfo "============================"
                        printInfo $"Grammar Name: {fractalGrammar.Name}"
                        printInfo $"Version: {fractalGrammar.Version}"
                        printInfo $"Rules Count: {fractalGrammar.FractalRules.Length}"
                        printInfo $"Success: {result.Success}"
                        printInfo $"Execution Time: {result.ExecutionTime.TotalMilliseconds:F1}ms"
                        printInfo $"Memory Used: {result.MemoryUsed / 1024L} KB"
                        printInfo $"Iterations: {result.IterationsCompleted}"
                        printInfo $"Fractal Dimension: {result.FractalDimension:F6}"
                        
                        printInfo "\n🔢 Complexity Metrics:"
                        result.ComplexityMetrics |> Map.iter (fun key value ->
                            printInfo $"   {key}: {value}")
                        
                        if detailed then
                            printInfo "\n📋 Rule Details:"
                            fractalGrammar.FractalRules |> List.iteri (fun i rule ->
                                printInfo $"\n   Rule {i+1}: {rule.Name}"
                                printInfo $"      Base Pattern: {rule.BasePattern}"
                                printInfo $"      Recursive: {rule.RecursivePattern |> Option.defaultValue "None"}"
                                printInfo $"      Termination: {rule.TerminationCondition}"
                                printInfo $"      Dimension: {rule.Properties.Dimension:F6}"
                                printInfo $"      Scaling Factor: {rule.Properties.ScalingFactor:F6}"
                                printInfo $"      Max Depth: {rule.Properties.RecursionLimit}"
                                printInfo $"      Transformations: {rule.Transformations.Length}")
                        
                        if not result.Warnings.IsEmpty then
                            printWarning "\n⚠️  Warnings:"
                            result.Warnings |> List.iter (fun w -> printWarning $"   {w}")
                    | None ->
                        printError "❌ No fractal grammar parsed"
                else
                    printError "❌ Parse failed"
            with
            | ex ->
                printError $"❌ Error: {ex.Message}"
        , inputOption, detailedOption)

        command

    /// Visualize fractal grammar command
    let visualizeCommand =
        let inputOption = Option<string>([|"-i"; "--input"|], "Input fractal grammar file")
        let outputOption = Option<string>([|"-o"; "--output"|], "Output visualization file")
        let formatOption = Option<string>([|"-f"; "--format"|], "Visualization format (SVG, GraphViz)")

        inputOption.IsRequired <- true
        outputOption.IsRequired <- true
        formatOption.SetDefaultValue("SVG")

        let command = Command("visualize", "Generate visualization of fractal grammar")
        command.AddOption(inputOption)
        command.AddOption(outputOption)
        command.AddOption(formatOption)

        command.SetHandler(fun input output format ->
            try
                printInfo $"🎨 Generating visualization: {input} -> {output}"
                
                let manager = FractalGrammarManager()
                let parseResult = manager.ParseFractalGrammarFile(input)
                
                if parseResult.Success then
                    match parseResult.FractalGrammar with
                    | Some fractalGrammar ->
                        let outputFormat = 
                            match format.ToUpperInvariant() with
                            | "SVG" -> SVG
                            | "GRAPHVIZ" -> GraphViz
                            | _ -> SVG
                        
                        let context = { 
                            manager.CreateDefaultContext() with 
                                OutputFormat = outputFormat
                                EnableVisualization = true
                        }
                        
                        let result = manager.ExecuteFractalGrammar(fractalGrammar, context)
                        
                        if result.Success then
                            match result.VisualizationData with
                            | Some vizData ->
                                File.WriteAllText(output, vizData)
                                printSuccess $"✅ Visualization saved: {output}"
                                printInfo $"📊 Fractal Dimension: {result.FractalDimension:F3}"
                                printInfo $"⏱️  Generation Time: {result.ExecutionTime.TotalMilliseconds:F1}ms"
                            | None ->
                                printError "❌ No visualization data generated"
                        else
                            printError "❌ Generation failed"
                    | None ->
                        printError "❌ No fractal grammar parsed"
                else
                    printError "❌ Parse failed"
            with
            | ex ->
                printError $"❌ Error: {ex.Message}"
        , inputOption, outputOption, formatOption)

        command

    /// Examples command
    let examplesCommand =
        let command = Command("examples", "Show fractal grammar examples")
        
        command.SetHandler(fun () ->
            printInfo "🔄 FRACTAL GRAMMAR EXAMPLES"
            printInfo "============================"
            printInfo ""
            
            printInfo "1. Sierpinski Triangle:"
            printInfo "   fractal sierpinski {"
            printInfo "     pattern = \"triangle\""
            printInfo "     recursive = \"triangle triangle triangle\""
            printInfo "     dimension = 1.585"
            printInfo "     depth = 6"
            printInfo "     transform scale 0.5"
            printInfo "   }"
            printInfo ""
            
            printInfo "2. Koch Snowflake:"
            printInfo "   fractal koch_curve {"
            printInfo "     pattern = \"line\""
            printInfo "     recursive = \"line turn60 line turn-120 line turn60 line\""
            printInfo "     dimension = 1.261"
            printInfo "     depth = 7"
            printInfo "     transform scale 0.333"
            printInfo "   }"
            printInfo ""
            
            printInfo "3. Dragon Curve:"
            printInfo "   fractal dragon {"
            printInfo "     pattern = \"F\""
            printInfo "     recursive = \"F+G+\""
            printInfo "     dimension = 2.0"
            printInfo "     depth = 12"
            printInfo "     transform recursive 12 scale 0.707"
            printInfo "   }"
            printInfo ""
            
            printInfo "Available example files:"
            printInfo "   .tars/grammars/SierpinskiTriangle.fractal"
            printInfo "   .tars/grammars/KochSnowflake.fractal"
            printInfo "   .tars/grammars/DragonCurve.fractal"
        )

        command

    /// Main CLI application
    let createRootCommand () =
        let rootCommand = RootCommand("TARS Fractal Grammar CLI - Advanced fractal grammar operations")
        
        rootCommand.AddCommand(generateCommand)
        rootCommand.AddCommand(parseCommand)
        rootCommand.AddCommand(analyzeCommand)
        rootCommand.AddCommand(visualizeCommand)
        rootCommand.AddCommand(examplesCommand)
        
        rootCommand

    /// Entry point
    [<EntryPoint>]
    let main args =
        try
            printInfo "🌀 TARS Fractal Grammar CLI v1.0"
            printInfo "================================="
            printInfo ""
            
            let rootCommand = createRootCommand()
            rootCommand.Invoke(args)
        with
        | ex ->
            printError $"❌ Fatal error: {ex.Message}"
            1

// Run the CLI if this script is executed directly
if fsi.CommandLineArgs.Length > 1 then
    FractalGrammarCLI.main (fsi.CommandLineArgs.[1..])
else
    printfn "Usage: dotnet fsi fractal_grammar_cli.fsx <command> [options]"
    printfn "Commands: generate, parse, analyze, visualize, examples"
    printfn "Use 'dotnet fsi fractal_grammar_cli.fsx examples' to see examples"
