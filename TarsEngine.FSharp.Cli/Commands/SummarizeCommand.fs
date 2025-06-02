namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open TarsEngine.FSharp.Core.AI
open TarsEngine.FSharp.Core.Metascript
open Spectre.Console

/// <summary>
/// CLI command for LLM response summarization
/// </summary>
type SummarizeCommand() =
    
    let summarizer = ResponseSummarizer()
    let blockParser = SummarizeBlockParser()
    
    /// Display summarization result
    member private this.DisplaySummary(result: SummarizationResult) =
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader($"📄 {result.Level} Summary")
        panel.Border <- BoxBorder.Rounded
        
        let content = Markup()
        content.AddLine($"[bold]Summary:[/]")
        content.AddLine($"[dim]{result.Summary}[/]")
        content.AddLine("")
        content.AddLine($"[bold]Compression:[/] {result.CompressionRatio:P1}")
        content.AddLine($"[bold]Confidence:[/] {result.ConfidenceScore:P1}")
        content.AddLine($"[bold]Processing Time:[/] {result.ProcessingTime.TotalMilliseconds:F0}ms")
        
        if result.CorrectionsMade.Length > 0 then
            content.AddLine("")
            content.AddLine($"[bold]Corrections Made:[/]")
            for correction in result.CorrectionsMade do
                content.AddLine($"  • {correction}")
        
        if result.ExpertConsensus.IsSome then
            content.AddLine("")
            content.AddLine($"[bold]Expert Consensus:[/]")
            for opinion in result.ExpertConsensus.Value do
                content.AddLine($"  • {opinion.ExpertType}: {opinion.ConfidenceScore:P1} - {opinion.Reasoning}")
        
        panel.Content <- content
        AnsiConsole.Write(panel)
    
    /// Display multi-level summary
    member private this.DisplayMultiLevelSummary(multiLevel: MultiLevelSummary) =
        AnsiConsole.MarkupLine($"[bold cyan]📊 Multi-Level Summary Results[/]")
        AnsiConsole.MarkupLine($"[dim]Original Length: {multiLevel.OriginalLength} characters[/]")
        AnsiConsole.MarkupLine($"[dim]Overall Quality: {multiLevel.OverallQuality:P1}[/]")
        AnsiConsole.MarkupLine($"[dim]Processing Time: {multiLevel.ProcessingTime.TotalMilliseconds:F0}ms[/]")
        AnsiConsole.WriteLine()
        
        for kvp in multiLevel.Summaries do
            this.DisplaySummary(kvp.Value)
            AnsiConsole.WriteLine()
    
    /// Summarize text with single level
    member this.SummarizeSingle(text: string, level: SummarizationLevel, ?moeConsensus: bool, ?autoCorrect: bool) =
        let moeConsensus = defaultArg moeConsensus true
        let autoCorrect = defaultArg autoCorrect true
        
        AnsiConsole.MarkupLine($"[bold blue]📄 Summarizing at {level} level...[/]")
        AnsiConsole.WriteLine()
        
        let config = {
            Levels = [level]
            MoeConsensus = moeConsensus
            AutoCorrect = autoCorrect
            PreserveFacts = true
            TargetAudience = "general"
            ExpertWeights = Map.ofList [
                (ClarityExpert, 0.8)
                (AccuracyExpert, 0.9)
                (BrevityExpert, 0.7)
                (StructureExpert, 0.8)
                (DomainExpert, 0.8)
            ]
            MaxIterations = 3
            QualityThreshold = 0.8
        }
        
        let result = summarizer.SummarizeLevel(text, level, config)
        this.DisplaySummary(result)
    
    /// Summarize text with multiple levels
    member this.SummarizeMultiLevel(text: string, ?levels: SummarizationLevel list, ?moeConsensus: bool) =
        let levels = defaultArg levels [SummarizationLevel.Executive; SummarizationLevel.Tactical; SummarizationLevel.Operational]
        let moeConsensus = defaultArg moeConsensus true
        
        AnsiConsole.MarkupLine($"[bold green]📊 Multi-level summarization...[/]")
        AnsiConsole.WriteLine()
        
        let config = {
            Levels = levels
            MoeConsensus = moeConsensus
            AutoCorrect = true
            PreserveFacts = true
            TargetAudience = "general"
            ExpertWeights = Map.ofList [
                (ClarityExpert, 0.8)
                (AccuracyExpert, 0.9)
                (BrevityExpert, 0.7)
                (StructureExpert, 0.8)
                (DomainExpert, 0.8)
            ]
            MaxIterations = 3
            QualityThreshold = 0.8
        }
        
        let result = summarizer.SummarizeMultiLevel(text, config)
        this.DisplayMultiLevelSummary(result)
    
    /// Compare two summaries
    member this.CompareSummaries(text: string, level: SummarizationLevel) =
        AnsiConsole.MarkupLine($"[bold magenta]⚖️ Comparing summarization approaches...[/]")
        AnsiConsole.WriteLine()
        
        // Create two different configurations
        let config1 = {
            Levels = [level]
            MoeConsensus = true
            AutoCorrect = true
            PreserveFacts = true
            TargetAudience = "general"
            ExpertWeights = Map.ofList [
                (ClarityExpert, 0.9)
                (AccuracyExpert, 0.8)
                (BrevityExpert, 0.6)
            ]
            MaxIterations = 3
            QualityThreshold = 0.8
        }
        
        let config2 = {
            config1 with
                ExpertWeights = Map.ofList [
                    (BrevityExpert, 0.9)
                    (StructureExpert, 0.8)
                    (DomainExpert, 0.7)
                ]
        }
        
        let result1 = summarizer.SummarizeLevel(text, level, config1)
        let result2 = summarizer.SummarizeLevel(text, level, config2)
        
        AnsiConsole.MarkupLine("[bold]Approach 1: Clarity-Focused[/]")
        this.DisplaySummary(result1)
        AnsiConsole.WriteLine()
        
        AnsiConsole.MarkupLine("[bold]Approach 2: Brevity-Focused[/]")
        this.DisplaySummary(result2)
        AnsiConsole.WriteLine()
        
        let comparison = summarizer.CompareSummaries(result1, result2)
        
        let table = Table()
        table.AddColumn("Metric") |> ignore
        table.AddColumn("Approach 1") |> ignore
        table.AddColumn("Approach 2") |> ignore
        
        table.AddRow("Quality Score", $"{comparison.Summary1Quality:P1}", $"{comparison.Summary2Quality:P1}") |> ignore
        table.AddRow("Compression", $"{comparison.CompressionComparison.Summary1Compression:P1}", $"{comparison.CompressionComparison.Summary2Compression:P1}") |> ignore
        table.AddRow("Length", $"{result1.Summary.Length} chars", $"{result2.Summary.Length} chars") |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        
        let recommendationColor = 
            match comparison.BetterSummary with
            | 1 -> "green"
            | 2 -> "blue"
            | _ -> "yellow"
        
        AnsiConsole.MarkupLine($"[{recommendationColor}]Recommendation: {comparison.Recommendation}[/]")
    
    /// Interactive summarization session
    member this.Interactive() =
        AnsiConsole.Clear()
        AnsiConsole.Write(
            FigletText("TARS Summarizer")
                .LeftJustified()
                .Color(Color.Cyan1)
        )
        
        AnsiConsole.MarkupLine("[dim]Multi-level LLM response summarization with MoE consensus[/]")
        AnsiConsole.WriteLine()
        
        let mutable continueSession = true
        
        while continueSession do
            let action = AnsiConsole.Prompt(
                SelectionPrompt<string>()
                    .Title("What would you like to do?")
                    .AddChoices([
                        "Single-level summary"
                        "Multi-level summary"
                        "Compare approaches"
                        "Batch process file"
                        "Test DSL block"
                        "View system stats"
                        "Exit"
                    ])
            )
            
            AnsiConsole.WriteLine()
            
            match action with
            | "Single-level summary" ->
                let text = AnsiConsole.Ask<string>("Enter the text to summarize:")
                let levelChoice = AnsiConsole.Prompt(
                    SelectionPrompt<string>()
                        .Title("Choose summarization level:")
                        .AddChoices(["Executive", "Tactical", "Operational", "Comprehensive", "Detailed"])
                )
                
                let level = 
                    match levelChoice with
                    | "Executive" -> SummarizationLevel.Executive
                    | "Tactical" -> SummarizationLevel.Tactical
                    | "Operational" -> SummarizationLevel.Operational
                    | "Comprehensive" -> SummarizationLevel.Comprehensive
                    | "Detailed" -> SummarizationLevel.Detailed
                    | _ -> SummarizationLevel.Tactical
                
                AnsiConsole.WriteLine()
                this.SummarizeSingle(text, level)
            
            | "Multi-level summary" ->
                let text = AnsiConsole.Ask<string>("Enter the text to summarize:")
                AnsiConsole.WriteLine()
                this.SummarizeMultiLevel(text)
            
            | "Compare approaches" ->
                let text = AnsiConsole.Ask<string>("Enter the text to summarize:")
                let levelChoice = AnsiConsole.Prompt(
                    SelectionPrompt<string>()
                        .Title("Choose level for comparison:")
                        .AddChoices(["Executive", "Tactical", "Operational"])
                )
                
                let level = 
                    match levelChoice with
                    | "Executive" -> SummarizationLevel.Executive
                    | "Tactical" -> SummarizationLevel.Tactical
                    | "Operational" -> SummarizationLevel.Operational
                    | _ -> SummarizationLevel.Tactical
                
                AnsiConsole.WriteLine()
                this.CompareSummaries(text, level)
            
            | "Batch process file" ->
                let inputFile = AnsiConsole.Ask<string>("Enter input file path:")
                let outputFile = AnsiConsole.Ask<string>("Enter output file path:")
                AnsiConsole.WriteLine()
                this.BatchProcess(inputFile, outputFile)
            
            | "Test DSL block" ->
                this.TestDslBlock()
            
            | "View system stats" ->
                this.ShowSystemStats()
            
            | "Exit" ->
                continueSession <- false
            
            | _ -> ()
            
            if continueSession then
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[dim]Press any key to continue...[/]")
                Console.ReadKey() |> ignore
                AnsiConsole.Clear()
    
    /// Batch process file
    member this.BatchProcess(inputFile: string, outputFile: string) =
        if not (File.Exists(inputFile)) then
            AnsiConsole.MarkupLine($"[red]Error: Input file '{inputFile}' not found[/]")
        else
            try
                let texts = File.ReadAllLines(inputFile)
                let results = ResizeArray<string>()
                
                AnsiConsole.MarkupLine($"[blue]Processing {texts.Length} texts...[/]")
                
                AnsiConsole.Progress()
                    .Start(fun ctx ->
                        let task = ctx.AddTask("[green]Summarizing texts[/]", true, texts.Length)
                        
                        for text in texts do
                            if not (String.IsNullOrWhiteSpace(text)) then
                                let multiLevel = summarizer.SummarizeMultiLevel(text)
                                
                                results.Add($"Original: {text}")
                                results.Add($"Length: {text.Length} characters")
                                
                                for kvp in multiLevel.Summaries do
                                    results.Add($"{kvp.Key} Summary: {kvp.Value.Summary}")
                                    results.Add($"Compression: {kvp.Value.CompressionRatio:P1}")
                                    results.Add($"Confidence: {kvp.Value.ConfidenceScore:P1}")
                                
                                results.Add("---")
                            
                            task.Increment(1.0)
                    )
                
                File.WriteAllLines(outputFile, results)
                AnsiConsole.MarkupLine($"[green]✅ Batch processing complete! Results saved to '{outputFile}'[/]")
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]Error during batch processing: {ex.Message}[/]")
    
    /// Test DSL block functionality
    member this.TestDslBlock() =
        AnsiConsole.MarkupLine("[bold cyan]🧪 Testing SUMMARIZE DSL Block[/]")
        AnsiConsole.WriteLine()
        
        let sampleText = "This is a sample text for testing the SUMMARIZE DSL block functionality. It contains multiple sentences to demonstrate the summarization capabilities. The system should be able to process this text and generate summaries at different levels of detail."
        
        let blockConfig = Map.ofList [
            ("source", box "sample_text")
            ("levels", box [1; 2; 3])
            ("output", box "test_summary")
            ("configuration", box (Map.ofList [
                ("moe_consensus", box true)
                ("auto_correct", box true)
                ("preserve_facts", box true)
                ("target_audience", box "technical")
            ]))
            ("experts", box (Map.ofList [
                ("clarity_expert", box 0.8)
                ("accuracy_expert", box 0.9)
                ("brevity_expert", box 0.7)
            ]))
        ]
        
        let variables = Map.ofList [("sample_text", box sampleText)]
        
        match blockParser.ParseSummarizeBlock(blockConfig, variables) with
        | Some block ->
            let result = blockParser.ExecuteSummarizeBlock(block)
            
            if result.Success then
                AnsiConsole.MarkupLine("[green]✅ DSL block executed successfully![/]")
                AnsiConsole.WriteLine()
                
                match result.Output with
                | Some summary -> this.DisplayMultiLevelSummary(summary)
                | None -> AnsiConsole.MarkupLine("[yellow]No output generated[/]")
            else
                AnsiConsole.MarkupLine("[red]❌ DSL block execution failed[/]")
                for error in result.Errors do
                    AnsiConsole.MarkupLine($"[red]  • {error}[/]")
        | None ->
            AnsiConsole.MarkupLine("[red]❌ Failed to parse DSL block[/]")
    
    /// Show system statistics
    member this.ShowSystemStats() =
        let stats = summarizer.GetSystemStats()
        
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader("📊 Summarization System Statistics")
        panel.Border <- BoxBorder.Rounded
        
        let content = Markup()
        content.AddLine($"[bold]Supported Levels:[/] {stats.SupportedLevels}")
        content.AddLine($"[bold]Expert Types:[/] {stats.ExpertTypes}")
        content.AddLine("")
        content.AddLine($"[bold]Default Compression Ratios:[/]")
        for kvp in stats.DefaultCompressionRatios do
            content.AddLine($"  • {kvp.Key}: {kvp.Value:P1}")
        content.AddLine("")
        content.AddLine($"[bold]Features:[/]")
        for feature in stats.Features do
            content.AddLine($"  • {feature}")
        
        panel.Content <- content
        AnsiConsole.Write(panel)
    
    /// Show DSL syntax help
    member this.ShowDslSyntax() =
        let syntaxHelp = blockParser.GetSyntaxHelp()
        
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader("📝 SUMMARIZE DSL Block Syntax")
        panel.Border <- BoxBorder.Rounded
        panel.Content <- Text(syntaxHelp)
        
        AnsiConsole.Write(panel)
