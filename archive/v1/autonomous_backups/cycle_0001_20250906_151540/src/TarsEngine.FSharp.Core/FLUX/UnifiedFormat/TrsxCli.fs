namespace TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat

open System
open System.IO
open TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat.UnifiedTrsxInterpreter
open TarsEngine.FSharp.FLUX.Standalone.UnifiedFormat.TrsxMigrationTool

/// CLI interface for unified TRSX operations
module TrsxCli =

    /// CLI command types
    type TrsxCommand =
        | Execute of filePath: string
        | Migrate of sourcePath: string * outputPath: string option
        | MigrateDirectory of directoryPath: string * pattern: string
        | Analyze of filePath: string
        | Validate of filePath: string
        | Help

    /// CLI result
    type CliResult = {
        Success: bool
        Message: string
        Details: string list
        ExecutionTime: TimeSpan
    }

    /// TRSX CLI processor
    type TrsxCliProcessor() =
        
        /// Process CLI command
        member this.ProcessCommand(command: TrsxCommand) : CliResult =
            let startTime = DateTime.Now
            
            try
                match command with
                | Execute filePath -> this.ExecuteTrsxFile(filePath)
                | Migrate (sourcePath, outputPath) -> this.MigrateSingleFile(sourcePath, outputPath)
                | MigrateDirectory (directoryPath, pattern) -> this.MigrateDirectory(directoryPath, pattern)
                | Analyze filePath -> this.AnalyzeTrsxFile(filePath)
                | Validate filePath -> this.ValidateTrsxFile(filePath)
                | Help -> this.ShowHelp()
                |> fun result -> { result with ExecutionTime = DateTime.Now - startTime }
            
            with
            | ex ->
                {
                    Success = false
                    Message = sprintf "Error: %s" ex.Message
                    Details = [ex.StackTrace]
                    ExecutionTime = DateTime.Now - startTime
                }

        /// Execute TRSX file
        member private this.ExecuteTrsxFile(filePath: string) : CliResult =
            if not (File.Exists(filePath)) then
                {
                    Success = false
                    Message = sprintf "File not found: %s" filePath
                    Details = []
                    ExecutionTime = TimeSpan.Zero
                }
            else
                let interpreter = UnifiedTrsxInterpreter()
                let result = interpreter.ExecuteTrsxFile(filePath)
                
                {
                    Success = result.Success
                    Message = sprintf "Executed %s (Tier %A)" (Path.GetFileName(filePath)) result.Tier
                    Details = [
                        sprintf "Self-similarity: %.3f" result.SelfSimilarityScore
                        sprintf "Entropy: %.3f" result.EntropyScore
                        sprintf "Results: %s" (String.Join("; ", result.Results))
                        match result.NextTierSuggestion with
                        | Some tier -> sprintf "Next tier suggestion: %A" tier
                        | None -> "No tier transition suggested"
                    ]
                    ExecutionTime = TimeSpan.Zero
                }

        /// Migrate single file
        member private this.MigrateSingleFile(sourcePath: string, outputPath: string option) : CliResult =
            let config = MigrationUtilities.createDefaultConfig()
            let configWithOutput = 
                match outputPath with
                | Some path -> { config with OutputDirectory = Some (Path.GetDirectoryName(path)) }
                | None -> config
            
            let migrator = FluxToTrsxMigrator(configWithOutput)
            let result = migrator.MigrateFluxFile(sourcePath)
            
            {
                Success = result.Success
                Message = 
                    if result.Success then
                        sprintf "Migrated %s to %s" (Path.GetFileName(result.SourceFile)) (Path.GetFileName(result.TargetFile))
                    else
                        sprintf "Migration failed for %s" (Path.GetFileName(result.SourceFile))
                Details = [
                    sprintf "Warnings: %d" result.WarningsCount
                    sprintf "Migration time: %A" result.MigrationTime
                    match result.EntropyImprovement with
                    | Some improvement -> sprintf "Entropy improvement: %.3f" improvement
                    | None -> "No entropy analysis"
                    yield! result.Warnings
                ]
                ExecutionTime = TimeSpan.Zero
            }

        /// Migrate directory
        member private this.MigrateDirectory(directoryPath: string, pattern: string) : CliResult =
            if not (Directory.Exists(directoryPath)) then
                {
                    Success = false
                    Message = sprintf "Directory not found: %s" directoryPath
                    Details = []
                    ExecutionTime = TimeSpan.Zero
                }
            else
                let config = MigrationUtilities.createDefaultConfig()
                let migrator = FluxToTrsxMigrator(config)
                let results = migrator.MigrateDirectory(directoryPath, pattern)
                
                let successful = results |> List.filter (fun r -> r.Success) |> List.length
                let total = results.Length
                
                {
                    Success = successful > 0
                    Message = sprintf "Migrated %d/%d files from %s" successful total directoryPath
                    Details = [
                        sprintf "Pattern: %s" pattern
                        sprintf "Successful migrations: %d" successful
                        sprintf "Failed migrations: %d" (total - successful)
                        yield! results 
                               |> List.filter (fun r -> not r.Success) 
                               |> List.map (fun r -> sprintf "Failed: %s - %s" (Path.GetFileName(r.SourceFile)) (String.Join("; ", r.Warnings)))
                    ]
                    ExecutionTime = TimeSpan.Zero
                }

        /// Analyze TRSX file
        member private this.AnalyzeTrsxFile(filePath: string) : CliResult =
            if not (File.Exists(filePath)) then
                {
                    Success = false
                    Message = sprintf "File not found: %s" filePath
                    Details = []
                    ExecutionTime = TimeSpan.Zero
                }
            else
                let parser = UnifiedTrsxParser()
                let document = parser.ParseTrsxFile(filePath)
                
                {
                    Success = true
                    Message = sprintf "Analysis of %s" (Path.GetFileName(filePath))
                    Details = [
                        sprintf "Title: %s" document.Metadata.Title
                        sprintf "Version: %s" document.Metadata.Version
                        sprintf "Tier: %A" document.Metadata.Tier
                        sprintf "Blocks: %d" document.Program.Blocks.Length
                        sprintf "Functions: %d" document.Program.Functions.Length
                        sprintf "Average entropy: %.3f" document.Reflection.EntropyAnalysis.AverageEntropy
                        sprintf "Overall similarity: %.3f" document.Reflection.SelfSimilarity.OverallSimilarity
                        sprintf "Insights: %d" document.Reflection.Insights.Length
                        match document.Evolution with
                        | Some evo -> sprintf "Evolution suggestions: %d" evo.MutationSuggestions.Length
                        | None -> "No evolution data"
                    ]
                    ExecutionTime = TimeSpan.Zero
                }

        /// Validate TRSX file
        member private this.ValidateTrsxFile(filePath: string) : CliResult =
            if not (File.Exists(filePath)) then
                {
                    Success = false
                    Message = sprintf "File not found: %s" filePath
                    Details = []
                    ExecutionTime = TimeSpan.Zero
                }
            else
                try
                    let parser = UnifiedTrsxParser()
                    let document = parser.ParseTrsxFile(filePath)
                    let validationResults = this.ValidateDocument(document)
                    
                    {
                        Success = validationResults |> List.forall (fun (isValid, _) -> isValid)
                        Message = sprintf "Validation of %s" (Path.GetFileName(filePath))
                        Details = validationResults |> List.map snd
                        ExecutionTime = TimeSpan.Zero
                    }
                with
                | ex ->
                    {
                        Success = false
                        Message = sprintf "Validation failed: %s" ex.Message
                        Details = [ex.Message]
                        ExecutionTime = TimeSpan.Zero
                    }

        /// Validate TRSX document
        member private this.ValidateDocument(document: UnifiedTrsxDocument) : (bool * string) list =
            let validations = ResizeArray<(bool * string)>()
            
            // Validate metadata
            let hasTitle = not (String.IsNullOrWhiteSpace(document.Metadata.Title))
            validations.Add((hasTitle, sprintf "Title present: %b" hasTitle))
            
            let hasVersion = not (String.IsNullOrWhiteSpace(document.Metadata.Version))
            validations.Add((hasVersion, sprintf "Version present: %b" hasVersion))
            
            // Validate program
            let hasBlocks = document.Program.Blocks.Length > 0
            validations.Add((hasBlocks, sprintf "Has blocks: %b (%d blocks)" hasBlocks document.Program.Blocks.Length))
            
            // Validate block structure
            for block in document.Program.Blocks do
                let hasId = not (String.IsNullOrWhiteSpace(block.Id))
                validations.Add((hasId, sprintf "Block %s has ID: %b" block.BlockType hasId))
                
                let hasPurpose = not (String.IsNullOrWhiteSpace(block.Purpose))
                validations.Add((hasPurpose, sprintf "Block %s has purpose: %b" block.BlockType hasPurpose))
            
            // Validate reflection
            let hasInsights = document.Reflection.Insights.Length > 0
            validations.Add((hasInsights, sprintf "Has insights: %b (%d insights)" hasInsights document.Reflection.Insights.Length))
            
            let validEntropy = document.Reflection.EntropyAnalysis.AverageEntropy >= 0.0
            validations.Add((validEntropy, sprintf "Valid entropy: %b (%.3f)" validEntropy document.Reflection.EntropyAnalysis.AverageEntropy))
            
            let validSimilarity = document.Reflection.SelfSimilarity.OverallSimilarity >= 0.0 && document.Reflection.SelfSimilarity.OverallSimilarity <= 1.0
            validations.Add((validSimilarity, sprintf "Valid similarity: %b (%.3f)" validSimilarity document.Reflection.SelfSimilarity.OverallSimilarity))
            
            validations |> Seq.toList

        /// Show help
        member private this.ShowHelp() : CliResult =
            {
                Success = true
                Message = "TRSX CLI Help"
                Details = [
                    "Available commands:"
                    ""
                    "Execute TRSX file:"
                    "  trsx execute <file.trsx>"
                    ""
                    "Migrate FLUX to TRSX:"
                    "  trsx migrate <file.flux> [output.trsx]"
                    ""
                    "Migrate directory:"
                    "  trsx migrate-dir <directory> <pattern>"
                    ""
                    "Analyze TRSX file:"
                    "  trsx analyze <file.trsx>"
                    ""
                    "Validate TRSX file:"
                    "  trsx validate <file.trsx>"
                    ""
                    "Show help:"
                    "  trsx help"
                    ""
                    "Examples:"
                    "  trsx execute flux_tier2_unified_example.trsx"
                    "  trsx migrate old_script.flux new_script.trsx"
                    "  trsx migrate-dir ./scripts *.flux"
                    "  trsx analyze my_script.trsx"
                    "  trsx validate my_script.trsx"
                ]
                ExecutionTime = TimeSpan.Zero
            }

    /// Command line argument parser
    type TrsxArgumentParser() =
        
        /// Parse command line arguments
        member this.ParseArguments(args: string[]) : TrsxCommand =
            match args with
            | [| "execute"; filePath |] -> Execute filePath
            | [| "migrate"; sourcePath |] -> Migrate (sourcePath, None)
            | [| "migrate"; sourcePath; outputPath |] -> Migrate (sourcePath, Some outputPath)
            | [| "migrate-dir"; directoryPath; pattern |] -> MigrateDirectory (directoryPath, pattern)
            | [| "analyze"; filePath |] -> Analyze filePath
            | [| "validate"; filePath |] -> Validate filePath
            | [| "help" |] | [||] -> Help
            | _ -> Help

    /// Main CLI entry point
    module TrsxMain =
        
        /// Run TRSX CLI
        let runCli (args: string[]) : int =
            let parser = TrsxArgumentParser()
            let processor = TrsxCliProcessor()
            
            let command = parser.ParseArguments(args)
            let result = processor.ProcessCommand(command)
            
            // Print result
            if result.Success then
                printfn "✅ %s" result.Message
            else
                printfn "❌ %s" result.Message
            
            if result.Details.Length > 0 then
                printfn ""
                for detail in result.Details do
                    printfn "   %s" detail
            
            if result.ExecutionTime > TimeSpan.Zero then
                printfn ""
                printfn "⏱️  Execution time: %A" result.ExecutionTime
            
            if result.Success then 0 else 1

        /// Demo function
        let runDemo() =
            printfn "🌀 TRSX Unified Format Demo"
            printfn "=========================="
            printfn ""
            
            // Demo execution
            let exampleFile = ".tars/Janus/flux_tier2_unified_example.trsx"
            if File.Exists(exampleFile) then
                printfn "📄 Executing example TRSX file..."
                let result = runCli [| "execute"; exampleFile |]
                printfn ""
            
            // Demo analysis
            if File.Exists(exampleFile) then
                printfn "🔍 Analyzing example TRSX file..."
                let result = runCli [| "analyze"; exampleFile |]
                printfn ""
            
            // Demo validation
            if File.Exists(exampleFile) then
                printfn "✅ Validating example TRSX file..."
                let result = runCli [| "validate"; exampleFile |]
                printfn ""
            
            printfn "🎯 Demo completed! Use 'trsx help' for more commands."
