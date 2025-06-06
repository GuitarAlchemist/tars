namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open TarsEngine.FSharp.Core.AI
open Spectre.Console

/// <summary>
/// CLI command for prompt improvement operations
/// </summary>
type PromptCommand() =
    
    let optimizer = PromptOptimizer()
    
    /// Display prompt analysis results
    member private self.DisplayAnalysis(analysis: PromptAnalysis) =
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader("🔍 Prompt Analysis Results")
        panel.Border <- BoxBorder.Rounded
        
        let content = Text()
        content.AddLine($"Confidence Score: {analysis.ConfidenceScore:P1}")
        content.AddLine($"Estimated Improvement: {analysis.EstimatedImprovement:P1}")
        content.AddLine("")
        
        if analysis.Issues.Length > 0 then
            content.AddLine("[red]Issues Found:[/]")
            for issue in analysis.Issues do
                content.AddLine($"  • {issue}")
            content.AddLine("")
        
        if analysis.Suggestions.Length > 0 then
            content.AddLine("[yellow]Suggestions:[/]")
            for suggestion in analysis.Suggestions do
                content.AddLine($"  • {suggestion}")
        
        panel.Content <- content
        AnsiConsole.Write(panel)
    
    /// Display improved prompt
    member private self.DisplayImprovement(improvement: ImprovedPrompt) =
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader("✨ Prompt Improvement")
        panel.Border <- BoxBorder.Rounded
        
        let content = Markup()
        content.AddLine($"[bold]Strategy:[/] {improvement.Strategy}")
        content.AddLine($"[bold]Confidence:[/] {improvement.ConfidenceScore:P1}")
        content.AddLine("")
        content.AddLine($"[bold]Reasoning:[/] {improvement.Reasoning}")
        content.AddLine("")
        content.AddLine($"[bold]Expected Benefit:[/] {improvement.ExpectedBenefit}")
        content.AddLine("")
        content.AddLine("[bold]Improved Prompt:[/]")
        content.AddLine($"[dim]{improvement.Improved}[/]")
        
        panel.Content <- content
        AnsiConsole.Write(panel)
    
    /// Analyze a prompt
    member self.Analyze(prompt: string) =
        AnsiConsole.MarkupLine("[bold blue]🔍 Analyzing prompt...[/]")
        AnsiConsole.WriteLine()
        
        let analysis = optimizer.AnalyzePrompt(prompt)
        self.DisplayAnalysis(analysis)
        
        if analysis.Issues.Length > 0 then
            AnsiConsole.WriteLine()
            if AnsiConsole.Confirm("Would you like to see improvement suggestions?") then
                self.Improve(prompt)
    
    /// Improve a prompt
    member self.Improve(prompt: string) =
        AnsiConsole.MarkupLine("[bold green]✨ Improving prompt...[/]")
        AnsiConsole.WriteLine()
        
        // Get best strategy
        let bestStrategy = optimizer.GetBestStrategy(prompt)
        AnsiConsole.MarkupLine($"[dim]Using strategy: {bestStrategy}[/]")
        AnsiConsole.WriteLine()
        
        let improvement = optimizer.ImprovePrompt(prompt, bestStrategy)
        self.DisplayImprovement(improvement)
        
        AnsiConsole.WriteLine()
        if AnsiConsole.Confirm("Would you like to try a different strategy?") then
            self.ChooseStrategy(prompt)
    
    /// Auto-improve a prompt
    member self.AutoImprove(prompt: string) =
        AnsiConsole.MarkupLine("[bold cyan]🚀 Auto-improving prompt...[/]")
        AnsiConsole.WriteLine()
        
        let improvement = optimizer.AutoImprove(prompt)
        self.DisplayImprovement(improvement)
        
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[green]✅ Auto-improvement complete![/]")
    
    /// Choose improvement strategy
    member self.ChooseStrategy(prompt: string) =
        let strategies = [
            ("Clarity Enhancement", PromptImprovementStrategy.ClarityEnhancement)
            ("Context Enrichment", PromptImprovementStrategy.ContextEnrichment)
            ("Example Addition", PromptImprovementStrategy.ExampleAddition)
            ("Constraint Specification", PromptImprovementStrategy.ConstraintSpecification)
            ("Format Standardization", PromptImprovementStrategy.FormatStandardization)
            ("Performance Optimization", PromptImprovementStrategy.PerformanceOptimization)
            ("Error Reduction", PromptImprovementStrategy.ErrorReduction)
            ("User Experience Improvement", PromptImprovementStrategy.UserExperienceImprovement)
        ]
        
        let choice = AnsiConsole.Prompt(
            SelectionPrompt<string>()
                .Title("Choose improvement strategy:")
                .AddChoices(strategies |> List.map fst)
        )
        
        let selectedStrategy = strategies |> List.find (fun (name, _) -> name = choice) |> snd
        
        AnsiConsole.WriteLine()
        let improvement = optimizer.ImprovePrompt(prompt, selectedStrategy)
        self.DisplayImprovement(improvement)
    
    /// Compare two prompts
    member self.Compare(prompt1: string, prompt2: string) =
        AnsiConsole.MarkupLine("[bold magenta]⚖️ Comparing prompts...[/]")
        AnsiConsole.WriteLine()
        
        let comparison = optimizer.ComparePrompts(prompt1, prompt2)
        
        let table = Table()
        table.AddColumn("Aspect") |> ignore
        table.AddColumn("Prompt 1") |> ignore
        table.AddColumn("Prompt 2") |> ignore
        
        table.AddRow("Quality Score", $"{comparison.Prompt1.Score:P1}", $"{comparison.Prompt2.Score:P1}") |> ignore
        table.AddRow("Issues Count", $"{comparison.Prompt1.Issues.Length}", $"{comparison.Prompt2.Issues.Length}") |> ignore
        table.AddRow("Length", $"{prompt1.Length} chars", $"{prompt2.Length} chars") |> ignore
        
        AnsiConsole.Write(table)
        AnsiConsole.WriteLine()
        
        let recommendationColor = 
            match comparison.BetterPrompt with
            | 1 -> "green"
            | 2 -> "blue"
            | _ -> "yellow"
        
        AnsiConsole.MarkupLine($"[{recommendationColor}]Recommendation: {comparison.Recommendation}[/]")
        
        if comparison.ScoreDifference > 0.1 then
            AnsiConsole.MarkupLine($"[dim]Score difference: {comparison.ScoreDifference:P1}[/]")
    
    /// Interactive prompt improvement session
    member self.Interactive() =
        AnsiConsole.Clear()
        AnsiConsole.Write(
            FigletText("TARS Prompt Optimizer")
                .LeftJustified()
                .Color(Color.Cyan1)
        )
        
        AnsiConsole.MarkupLine("[dim]Universal prompt improvement for TARS operations[/]")
        AnsiConsole.WriteLine()
        
        let mutable continueSession = true
        
        while continueSession do
            let action = AnsiConsole.Prompt(
                SelectionPrompt<string>()
                    .Title("What would you like to do?")
                    .AddChoices([
                        "Analyze a prompt"
                        "Improve a prompt"
                        "Auto-improve a prompt"
                        "Compare two prompts"
                        "View statistics"
                        "Exit"
                    ])
            )
            
            AnsiConsole.WriteLine()
            
            match action with
            | "Analyze a prompt" ->
                let prompt = AnsiConsole.Ask<string>("Enter the prompt to analyze:")
                AnsiConsole.WriteLine()
                self.Analyze(prompt)
            
            | "Improve a prompt" ->
                let prompt = AnsiConsole.Ask<string>("Enter the prompt to improve:")
                AnsiConsole.WriteLine()
                self.Improve(prompt)
            
            | "Auto-improve a prompt" ->
                let prompt = AnsiConsole.Ask<string>("Enter the prompt to auto-improve:")
                AnsiConsole.WriteLine()
                self.AutoImprove(prompt)
            
            | "Compare two prompts" ->
                let prompt1 = AnsiConsole.Ask<string>("Enter the first prompt:")
                let prompt2 = AnsiConsole.Ask<string>("Enter the second prompt:")
                AnsiConsole.WriteLine()
                self.Compare(prompt1, prompt2)
            
            | "View statistics" ->
                self.ShowStatistics()
            
            | "Exit" ->
                continueSession <- false
            
            | _ -> ()
            
            if continueSession then
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[dim]Press any key to continue...[/]")
                Console.ReadKey() |> ignore
                AnsiConsole.Clear()
    
    /// Show improvement statistics
    member self.ShowStatistics() =
        let stats = optimizer.GetImprovementStats()
        
        let panel = Panel(Align.Left)
        panel.Header <- PanelHeader("📊 Prompt Improvement Statistics")
        panel.Border <- BoxBorder.Rounded
        
        let content = Markup()
        content.AddLine($"[bold]Total Prompts Tracked:[/] {stats.TotalPromptsTracked}")
        content.AddLine($"[bold]Total Improvements:[/] {stats.TotalImprovements}")
        content.AddLine($"[bold]Average Success Rate:[/] {stats.AverageSuccessRate:P1}")
        content.AddLine($"[bold]Average Response Time:[/] {stats.AverageResponseTime:F0}ms")
        content.AddLine($"[bold]Average User Satisfaction:[/] {stats.AverageUserSatisfaction:F1}/5.0")
        content.AddLine($"[bold]Most Used Strategy:[/] {stats.MostUsedStrategy}")
        content.AddLine($"[bold]Prompts Needing Improvement:[/] {stats.PromptsNeedingImprovement}")
        
        panel.Content <- content
        AnsiConsole.Write(panel)
    
    /// Batch improve prompts from file
    member self.BatchImprove(inputFile: string, outputFile: string) =
        if not (File.Exists(inputFile)) then
            AnsiConsole.MarkupLine($"[red]Error: Input file '{inputFile}' not found[/]")
            1
        else
            try
                let prompts = File.ReadAllLines(inputFile)
                let improvements = ResizeArray<string>()
                
                AnsiConsole.MarkupLine($"[blue]Processing {prompts.Length} prompts...[/]")
                
                AnsiConsole.Progress()
                    .Start(fun ctx ->
                        let task = ctx.AddTask("[green]Improving prompts[/]", true, prompts.Length)
                        
                        for prompt in prompts do
                            if not (String.IsNullOrWhiteSpace(prompt)) then
                                let improvement = optimizer.AutoImprove(prompt)
                                improvements.Add($"Original: {prompt}")
                                improvements.Add($"Improved: {improvement.Improved}")
                                improvements.Add($"Strategy: {improvement.Strategy}")
                                improvements.Add($"Reasoning: {improvement.Reasoning}")
                                improvements.Add("---")
                            
                            task.Increment(1.0)
                    )
                
                File.WriteAllLines(outputFile, improvements)
                AnsiConsole.MarkupLine($"[green]✅ Batch improvement complete! Results saved to '{outputFile}'[/]")
                0
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]Error during batch processing: {ex.Message}[/]")
                1
    
    /// Record prompt performance
    member self.RecordPerformance(promptId: string, responseTime: float, success: bool, userSatisfaction: float option) =
        optimizer.RecordPerformance(promptId, responseTime, success, userSatisfaction)
        AnsiConsole.MarkupLine($"[green]✅ Performance recorded for prompt '{promptId}'[/]")
    
    /// Get prompt performance
    member self.GetPerformance(promptId: string) =
        match optimizer.GetPerformance(promptId) with
        | Some performance ->
            let panel = Panel(Align.Left)
            panel.Header <- PanelHeader($"📊 Performance for '{promptId}'")
            panel.Border <- BoxBorder.Rounded
            
            let content = Markup()
            content.AddLine($"[bold]Usage Count:[/] {performance.UsageCount}")
            content.AddLine($"[bold]Success Rate:[/] {performance.SuccessRate:P1}")
            content.AddLine($"[bold]Average Response Time:[/] {performance.AverageResponseTime:F0}ms")
            content.AddLine($"[bold]User Satisfaction:[/] {performance.UserSatisfactionScore:F1}/5.0")
            content.AddLine($"[bold]Error Count:[/] {performance.ErrorCount}")
            content.AddLine($"[bold]Last Used:[/] {performance.LastUsed:yyyy-MM-dd HH:mm}")
            
            panel.Content <- content
            AnsiConsole.Write(panel)
        | None ->
            AnsiConsole.MarkupLine($"[yellow]No performance data found for prompt '{promptId}'[/]")
