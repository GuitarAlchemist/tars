namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Core

/// Command for generating knowledge mind maps
type MindMapCommand(logger: ILogger<MindMapCommand>, learningMemoryService: LearningMemoryService) =

    interface ICommand with
        member _.Name = "mindmap"
        member _.Description = "Generate ASCII and Markdown mind maps of TARS knowledge base"
        member _.Usage = "mindmap [ascii|markdown|both|stats|explore] [options]"
        member _.Examples = [
            "tars mindmap ascii"
            "tars mindmap markdown F#"
            "tars mindmap both programming"
            "tars mindmap stats"
            "tars mindmap explore"
        ]
        member _.ValidateOptions(_) = true

        member this.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    let args = options.Arguments |> List.toArray
                    if args.Length = 0 then
                        // Default: show interactive explorer
                        let! result = this.InteractiveMindMapExplorer()
                        match result with
                        | Ok () -> return { Success = true; ExitCode = 0; Message = "Mind map explorer completed successfully" }
                        | Error err ->
                            Console.WriteLine(sprintf "❌ Error: %s" err)
                            return { Success = false; ExitCode = 1; Message = err }
                    else
                        let subCommand = args.[0].ToLowerInvariant()
                        let remainingArgs = if args.Length > 1 then args.[1..] else [||]

                        match subCommand with
                        | "ascii" ->
                            let topic = if remainingArgs.Length > 0 then Some (String.Join(" ", remainingArgs)) else None
                            let! result = this.GenerateAsciiMindMap(topic, 3, 20)
                            match result with
                            | Ok () -> return { Success = true; ExitCode = 0; Message = "ASCII mind map generated successfully" }
                            | Error err ->
                                Console.WriteLine(sprintf "❌ Error: %s" err)
                                return { Success = false; ExitCode = 1; Message = err }

                        | "markdown" ->
                            let topic = if remainingArgs.Length > 0 then Some (String.Join(" ", remainingArgs)) else None
                            let! result = this.GenerateMarkdownMindMap(topic, None, true, true)
                            match result with
                            | Ok file -> return { Success = true; ExitCode = 0; Message = sprintf "Markdown mind map saved to: %s" file }
                            | Error err ->
                                Console.WriteLine(sprintf "❌ Error: %s" err)
                                return { Success = false; ExitCode = 1; Message = err }

                        | "both" ->
                            let topic = if remainingArgs.Length > 0 then Some (String.Join(" ", remainingArgs)) else None
                            let! result = this.GenerateCompleteMindMap(topic, None, 3, 20, true, true)
                            match result with
                            | Ok file -> return { Success = true; ExitCode = 0; Message = sprintf "Complete mind map generated: %s" file }
                            | Error err ->
                                Console.WriteLine(sprintf "❌ Error: %s" err)
                                return { Success = false; ExitCode = 1; Message = err }

                        | "stats" ->
                            let! result = this.ShowMindMapStats()
                            match result with
                            | Ok () -> return { Success = true; ExitCode = 0; Message = "Knowledge statistics displayed" }
                            | Error err ->
                                Console.WriteLine(sprintf "❌ Error: %s" err)
                                return { Success = false; ExitCode = 1; Message = err }

                        | "explore" ->
                            let! result = this.InteractiveMindMapExplorer()
                            match result with
                            | Ok () -> return { Success = true; ExitCode = 0; Message = "Interactive explorer completed" }
                            | Error err ->
                                Console.WriteLine(sprintf "❌ Error: %s" err)
                                return { Success = false; ExitCode = 1; Message = err }

                        | _ ->
                            Console.WriteLine("❓ Unknown subcommand. Available: ascii, markdown, both, stats, explore")
                            Console.WriteLine("Usage: mindmap [ascii|markdown|both|stats|explore] [topic]")
                            return { Success = false; ExitCode = 1; Message = "Unknown subcommand" }

                with
                | ex ->
                    logger.LogError(ex, "❌ MIND MAP: Command execution failed")
                    Console.WriteLine(sprintf "❌ Unexpected error: %s" ex.Message)
                    return { Success = false; ExitCode = 1; Message = ex.Message }
            }
    
    /// Generate ASCII mind map for CLI display
    member this.GenerateAsciiMindMap(centralTopic: string option, maxDepth: int, maxNodes: int) =
        async {
            try
                logger.LogInformation("🧠 MIND MAP: Generating ASCII mind map")
                
                let! asciiMap = learningMemoryService.GenerateAsciiMindMap(centralTopic, maxDepth, maxNodes)
                
                // Display in console
                Console.WriteLine()
                Console.WriteLine(asciiMap)
                Console.WriteLine()
                
                logger.LogInformation("✅ MIND MAP: ASCII mind map generated successfully")
                return Ok()
                
            with
            | ex ->
                logger.LogError(ex, "❌ MIND MAP: Failed to generate ASCII mind map")
                return Error ex.Message
        }
    
    /// Generate Markdown mind map and save to file
    member this.GenerateMarkdownMindMap(centralTopic: string option, outputPath: string option, includeContent: bool, includeMermaid: bool) =
        async {
            try
                logger.LogInformation("📝 MIND MAP: Generating Markdown mind map")
                
                let! markdownContent = learningMemoryService.GenerateMarkdownMindMap(centralTopic, includeContent, includeMermaid)
                
                // Determine output path
                let outputFile = 
                    match outputPath with
                    | Some path -> path
                    | None -> 
                        let timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss")
                        let topicSuffix = 
                            match centralTopic with
                            | Some topic -> sprintf "_%s" (topic.Replace(" ", "_").Replace("/", "_"))
                            | None -> ""
                        sprintf "tars_mindmap%s_%s.md" topicSuffix timestamp
                
                // Ensure output directory exists
                let outputDir = Path.GetDirectoryName(outputFile)
                if not (String.IsNullOrEmpty(outputDir)) && not (Directory.Exists(outputDir)) then
                    Directory.CreateDirectory(outputDir) |> ignore
                
                // Write to file
                File.WriteAllText(outputFile, markdownContent)
                
                Console.WriteLine()
                Console.WriteLine("📝 Markdown Mind Map Generated!")
                Console.WriteLine(sprintf "📁 File: %s" (Path.GetFullPath(outputFile)))
                Console.WriteLine(sprintf "📊 Size: %d KB" (FileInfo(outputFile).Length / 1024L))
                
                if includeMermaid then
                    Console.WriteLine("🎨 Includes Mermaid diagrams for visualization")
                
                if includeContent then
                    Console.WriteLine("📚 Includes detailed knowledge content")
                
                Console.WriteLine()
                
                logger.LogInformation("✅ MIND MAP: Markdown mind map saved to {OutputFile}", outputFile)
                return Ok outputFile
                
            with
            | ex ->
                logger.LogError(ex, "❌ MIND MAP: Failed to generate Markdown mind map")
                return Error ex.Message
        }
    
    /// Generate both ASCII and Markdown mind maps
    member this.GenerateCompleteMindMap(centralTopic: string option, outputPath: string option, maxDepth: int, maxNodes: int, includeContent: bool, includeMermaid: bool) =
        async {
            try
                logger.LogInformation("🧠 MIND MAP: Generating complete mind map (ASCII + Markdown)")
                
                // Generate ASCII mind map first
                let! asciiResult = this.GenerateAsciiMindMap(centralTopic, maxDepth, maxNodes)
                match asciiResult with
                | Error err -> return Error err
                | Ok () ->
                    // Generate Markdown mind map
                    let! markdownResult = this.GenerateMarkdownMindMap(centralTopic, outputPath, includeContent, includeMermaid)
                    match markdownResult with
                    | Error err -> return Error err
                    | Ok outputFile ->
                        Console.WriteLine("🎉 Complete mind map generation successful!")
                        Console.WriteLine("   ✅ ASCII mind map displayed above")
                        Console.WriteLine(sprintf "   ✅ Markdown mind map saved to: %s" outputFile)
                        Console.WriteLine()
                        
                        logger.LogInformation("✅ MIND MAP: Complete mind map generation successful")
                        return Ok outputFile
                
            with
            | ex ->
                logger.LogError(ex, "❌ MIND MAP: Failed to generate complete mind map")
                return Error ex.Message
        }
    
    /// Show mind map statistics
    member this.ShowMindMapStats() =
        async {
            try
                logger.LogInformation("📊 MIND MAP: Generating knowledge statistics")
                
                let stats = learningMemoryService.GetMemoryStats()
                
                Console.WriteLine()
                Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════════╗")
                Console.WriteLine("║                        📊 TARS KNOWLEDGE STATISTICS 📊                       ║")
                Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝")
                Console.WriteLine()
                Console.WriteLine(sprintf "🧠 Total Knowledge Entries: %d" stats.TotalKnowledge)
                Console.WriteLine(sprintf "📅 Recent Learning (7 days): %d" stats.RecentLearning)
                Console.WriteLine(sprintf "💾 Cache Size: %d entries" stats.CacheSize)
                Console.WriteLine(sprintf "📏 Estimated Size: %.2f MB" stats.StorageMetrics.EstimatedSizeMB)
                Console.WriteLine(sprintf "🎯 Average Confidence: %.1f%%" (stats.StorageMetrics.AverageConfidence * 100.0))
                Console.WriteLine(sprintf "⭐ High Confidence Entries: %d" stats.StorageMetrics.HighConfidenceEntries)
                Console.WriteLine()
                
                Console.WriteLine("🏆 Top Knowledge Topics:")
                for (i, (topic, count)) in stats.TopTopics |> List.take (min 10 stats.TopTopics.Length) |> List.mapi (fun i x -> (i+1, x)) do
                    Console.WriteLine(sprintf "   %2d. %-40s (%d entries)" i topic count)
                
                Console.WriteLine()
                Console.WriteLine("📊 Knowledge Sources:")
                for (source, count) in stats.SourceDistribution |> List.take (min 5 stats.SourceDistribution.Length) do
                    Console.WriteLine(sprintf "   • %-30s: %d entries" source count)
                
                Console.WriteLine()
                Console.WriteLine("🔧 Storage Capabilities:")
                Console.WriteLine(sprintf "   • In-Memory Cache: %s" (if stats.IndexingCapabilities.InMemoryCache then "✅ Active" else "❌ Inactive"))
                Console.WriteLine(sprintf "   • RDF Triple Store: %s" (if stats.IndexingCapabilities.RDFTripleStore then "✅ Active" else "❌ Inactive"))
                Console.WriteLine(sprintf "   • Tag-Based Indexing: %s" (if stats.IndexingCapabilities.TagBasedIndexing then "✅ Active" else "❌ Inactive"))
                Console.WriteLine(sprintf "   • Confidence Filtering: %s" (if stats.IndexingCapabilities.ConfidenceFiltering then "✅ Active" else "❌ Inactive"))
                Console.WriteLine()
                
                logger.LogInformation("✅ MIND MAP: Knowledge statistics displayed")
                return Ok()
                
            with
            | ex ->
                logger.LogError(ex, "❌ MIND MAP: Failed to show statistics")
                return Error ex.Message
        }
    
    /// Interactive mind map explorer
    member this.InteractiveMindMapExplorer() =
        async {
            try
                logger.LogInformation("🔍 MIND MAP: Starting interactive explorer")
                
                Console.WriteLine()
                Console.WriteLine("╔══════════════════════════════════════════════════════════════════════════════╗")
                Console.WriteLine("║                    🔍 INTERACTIVE MIND MAP EXPLORER 🔍                       ║")
                Console.WriteLine("╚══════════════════════════════════════════════════════════════════════════════╝")
                Console.WriteLine()
                Console.WriteLine("Commands:")
                Console.WriteLine("  ascii [topic]     - Generate ASCII mind map (optional center topic)")
                Console.WriteLine("  markdown [topic]  - Generate Markdown mind map with diagrams")
                Console.WriteLine("  both [topic]      - Generate both ASCII and Markdown")
                Console.WriteLine("  stats             - Show knowledge statistics")
                Console.WriteLine("  help              - Show this help")
                Console.WriteLine("  quit              - Exit explorer")
                Console.WriteLine()
                
                let mutable continueExploring = true
                
                while continueExploring do
                    Console.Write("🧠 mind-map> ")
                    let input = Console.ReadLine()
                    
                    if String.IsNullOrWhiteSpace(input) then
                        ()
                    else
                        let parts = input.Trim().Split([|' '|], StringSplitOptions.RemoveEmptyEntries)
                        let command = parts.[0].ToLowerInvariant()
                        let topic = if parts.Length > 1 then Some (String.Join(" ", parts.[1..])) else None
                        
                        match command with
                        | "ascii" ->
                            let! result = this.GenerateAsciiMindMap(topic, 3, 20)
                            match result with
                            | Ok () -> ()
                            | Error err -> Console.WriteLine(sprintf "❌ Error: %s" err)
                        
                        | "markdown" ->
                            let! result = this.GenerateMarkdownMindMap(topic, None, true, true)
                            match result with
                            | Ok file -> Console.WriteLine(sprintf "✅ Saved to: %s" file)
                            | Error err -> Console.WriteLine(sprintf "❌ Error: %s" err)
                        
                        | "both" ->
                            let! result = this.GenerateCompleteMindMap(topic, None, 3, 20, true, true)
                            match result with
                            | Ok file -> Console.WriteLine(sprintf "✅ Complete mind map generated: %s" file)
                            | Error err -> Console.WriteLine(sprintf "❌ Error: %s" err)
                        
                        | "stats" ->
                            let! result = this.ShowMindMapStats()
                            match result with
                            | Ok () -> ()
                            | Error err -> Console.WriteLine(sprintf "❌ Error: %s" err)
                        
                        | "help" ->
                            Console.WriteLine("Available commands:")
                            Console.WriteLine("  ascii [topic]     - Generate ASCII mind map")
                            Console.WriteLine("  markdown [topic]  - Generate Markdown mind map")
                            Console.WriteLine("  both [topic]      - Generate both formats")
                            Console.WriteLine("  stats             - Show statistics")
                            Console.WriteLine("  quit              - Exit")
                        
                        | "quit" | "exit" ->
                            continueExploring <- false
                            Console.WriteLine("👋 Goodbye!")
                        
                        | _ ->
                            Console.WriteLine("❓ Unknown command. Type 'help' for available commands.")
                
                logger.LogInformation("✅ MIND MAP: Interactive explorer session ended")
                return Ok()
                
            with
            | ex ->
                logger.LogError(ex, "❌ MIND MAP: Interactive explorer failed")
                return Error ex.Message
        }
