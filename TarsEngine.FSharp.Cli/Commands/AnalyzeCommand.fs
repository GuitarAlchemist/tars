namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks

/// <summary>
/// Command for code analysis - self-contained.
/// </summary>
type AnalyzeCommand() =
    interface ICommand with
        member _.Name = "analyze"
        
        member _.Description = "Analyze code quality and generate insights"
        
        member self.Usage = "tars analyze [target] [options]"
        
        member self.Examples = [
            "tars analyze ."
            "tars analyze --detailed"
            "tars analyze src/ --output report.json"
        ]
        
        member self.ValidateOptions(options) = true
        
        member self.ExecuteAsync(options) =
            Task.Run(fun () ->
                try
                    let target =
                        match options.Arguments with
                        | arg :: _ -> arg
                        | [] -> "."

                    let detailed = options.Options.ContainsKey("detailed")
                    let output = options.Options.TryFind("output")
                    let selfAnalysis = options.Options.ContainsKey("self")

                    Console.WriteLine(sprintf "🔍 TARS: Starting autonomous code analysis...")
                    Console.WriteLine(sprintf "Target: %s" target)

                    if selfAnalysis then
                        Console.WriteLine("🤖 TARS: Performing self-analysis...")
                        performTarsSelfAnalysis target output detailed
                    else
                        Console.WriteLine("📊 TARS: Analyzing external codebase...")
                        performCodebaseAnalysis target output detailed

                with
                | ex ->
                    CommandResult.failure(sprintf "Analysis failed: %s" ex.Message)
            )

    // Real TARS self-analysis implementation
    member private _.performTarsSelfAnalysis(targetPath: string) (outputPath: string option) (detailed: bool) =
        Console.WriteLine("\n🤖 TARS Self-Analysis Starting...")
        Console.WriteLine("==================================")

        // Phase 1: Scan TARS codebase structure
        let codebaseStructure = self.scanTarsCodebase targetPath
        Console.WriteLine(sprintf "📁 Found %d F# files, %d projects" codebaseStructure.FSharpFiles codebaseStructure.ProjectFiles)

        // Phase 2: Analyze architecture
        let architectureAnalysis = self.analyzeTarsArchitecture targetPath
        Console.WriteLine(sprintf "🏗️ Architecture: %s" architectureAnalysis.Pattern)

        // Phase 3: Code quality assessment
        let codeQuality = self.assessTarsCodeQuality targetPath
        Console.WriteLine(sprintf "📊 Code Quality Score: %d/100" codeQuality.OverallScore)

        // Phase 4: Security audit
        let securityAudit = self.auditTarsSecurity targetPath
        Console.WriteLine(sprintf "🔒 Security Issues: %d critical, %d high" securityAudit.Critical securityAudit.High)

        // Phase 5: Performance analysis
        let performanceAnalysis = self.analyzeTarsPerformance targetPath
        Console.WriteLine(sprintf "⚡ Performance Score: %d/100" performanceAnalysis.Score)

        // Phase 6: Generate report
        let report = self.generateSelfAnalysisReport codebaseStructure architectureAnalysis codeQuality securityAudit performanceAnalysis

        let outputFile =
            match outputPath with
            | Some path -> path
            | None -> System.IO.Path.Combine(targetPath, ".tars", "projects", "tars", "TARS-AUTONOMOUS-SELF-ANALYSIS.md")

        System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(outputFile)) |> ignore
        System.IO.File.WriteAllText(outputFile, report)

        Console.WriteLine(sprintf "✅ TARS self-analysis complete! Report saved: %s" outputFile)
        CommandResult.success("TARS self-analysis completed successfully")
