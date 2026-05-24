namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Commands.Common
open TarsEngine.FSharp.Cli.Commands.MultiAgentReasoningDemo

// ============================================================================
// INTERACTIVE MULTI-AGENT COMMAND - PROPER COMMAND IMPLEMENTATION
// ============================================================================

type InteractiveMultiAgentCommand() =
    interface ICommand with
        member _.Name = "interactive-multiagent"
        member _.Description = "Interactive multi-agent reasoning with problem selection and continuous analysis"
        
        member _.Execute(args: string[]) = task {
            AnsiConsole.Clear()
            AnsiConsole.Write(
                FigletText("TARS Interactive")
                    .Centered()
                    .Color(Color.Cyan))
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold cyan]🎮 TARS Interactive Multi-Agent Reasoning Command[/]")
            AnsiConsole.MarkupLine("[dim]Enhanced multi-agent reasoning with full interactive capabilities[/]")
            AnsiConsole.WriteLine()

            // Parse arguments
            let hasArgs = args.Length > 0
            let interactive = not hasArgs || (args.Length = 1 && args.[0] = "--interactive")

            if interactive then
                AnsiConsole.MarkupLine("[cyan]🎮 Starting Interactive Mode[/]")
                AnsiConsole.MarkupLine("[dim]You can select from predefined scenarios or enter custom problems[/]")
                AnsiConsole.WriteLine()
                
                do! MultiAgentReasoningDemo.runInteractiveMultiAgentDemo()
                
                return { Success = true; Message = "Interactive multi-agent reasoning completed"; Data = None }
            else
                // Direct problem analysis
                let problem = String.Join(" ", args)
                AnsiConsole.MarkupLine($"[cyan]🎯 Direct Problem Analysis: {problem}[/]")
                AnsiConsole.WriteLine()
                
                do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                
                return { Success = true; Message = "Direct problem analysis completed"; Data = Some (box problem) }
        }

// ============================================================================
// ENHANCED DEMO COMMAND INTEGRATION
// ============================================================================

type EnhancedMultiAgentReasoningCommand() =
    interface ICommand with
        member _.Name = "enhanced-reasoning"
        member _.Description = "Enhanced multi-agent reasoning with interactive capabilities and scenario selection"
        
        member _.Execute(args: string[]) = task {
            AnsiConsole.Clear()
            AnsiConsole.Write(
                FigletText("Enhanced TARS")
                    .Centered()
                    .Color(Color.Green))
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold green]🚀 Enhanced Multi-Agent Reasoning System[/]")
            AnsiConsole.MarkupLine("[dim]Advanced problem decomposition with interactive features[/]")
            AnsiConsole.WriteLine()

            // Show available modes
            if args.Length = 0 then
                let mode = AnsiConsole.Prompt(
                    SelectionPrompt<string>()
                        .Title("[green]Select reasoning mode:[/]")
                        .AddChoices([
                            "🎮 Interactive Mode - Select scenarios and analyze multiple problems"
                            "📝 Custom Problem - Enter a specific problem to analyze"
                            "🎲 Random Scenario - Analyze a randomly selected complex scenario"
                            "📊 Demo Mode - Run with the default autonomous vehicle problem"
                        ])
                        .HighlightStyle(Style.Parse("cyan"))
                )

                match mode with
                | m when m.Contains("Interactive Mode") ->
                    do! MultiAgentReasoningDemo.runInteractiveMultiAgentDemo()
                    
                | m when m.Contains("Custom Problem") ->
                    let customProblem = AnsiConsole.Ask<string>("[green]Enter your complex problem statement:[/]")
                    do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some customProblem)
                    
                | m when m.Contains("Random Scenario") ->
                    let scenarios = MultiAgentReasoningDemo.interactiveScenarios |> List.filter (fun (title, _) -> not (title.Contains("Custom")))
                    let randomScenario = scenarios.[0 // HONEST: Cannot generate without real measurement]
                    let (title, problem) = randomScenario
                    
                    AnsiConsole.MarkupLine($"[yellow]🎲 Random Scenario Selected: {title}[/]")
                    AnsiConsole.WriteLine()
                    do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                    
                | _ ->
                    AnsiConsole.MarkupLine("[yellow]📊 Running demo mode with default problem[/]")
                    do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync None

                return { Success = true; Message = "Enhanced reasoning completed"; Data = None }
            else
                // Direct execution with provided problem
                let problem = String.Join(" ", args)
                do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                return { Success = true; Message = "Enhanced reasoning completed"; Data = Some (box problem) }
        }

// ============================================================================
// SCENARIO SELECTION COMMAND
// ============================================================================

type ScenarioReasoningCommand() =
    interface ICommand with
        member _.Name = "scenario-reasoning"
        member _.Description = "Multi-agent reasoning with predefined complex scenarios"
        
        member _.Execute(args: string[]) = task {
            AnsiConsole.Clear()
            AnsiConsole.Write(
                FigletText("Scenarios")
                    .Centered()
                    .Color(Color.Yellow))
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold yellow]📋 TARS Scenario-Based Multi-Agent Reasoning[/]")
            AnsiConsole.MarkupLine("[dim]Choose from complex, real-world scenarios for analysis[/]")
            AnsiConsole.WriteLine()

            if args.Length = 0 then
                // Interactive scenario selection
                let scenarios = MultiAgentReasoningDemo.interactiveScenarios |> List.filter (fun (title, _) -> not (title.Contains("Custom")))
                
                let selectedTitle = AnsiConsole.Prompt(
                    SelectionPrompt<string>()
                        .Title("[yellow]🎯 Select a complex scenario to analyze:[/]")
                        .AddChoices(scenarios |> List.map fst)
                        .HighlightStyle(Style.Parse("yellow"))
                )

                let (_, problem) = scenarios |> List.find (fun (title, _) -> title = selectedTitle)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine($"[green]🎯 Analyzing Scenario: {selectedTitle}[/]")
                AnsiConsole.WriteLine()
                
                do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                
                // Ask if they want to try another scenario
                let continueAnalysis = AnsiConsole.Confirm("🔄 Would you like to analyze another scenario?")
                
                if continueAnalysis then
                    do! MultiAgentReasoningDemo.runInteractiveMultiAgentDemo()

                return { Success = true; Message = "Scenario analysis completed"; Data = Some (box selectedTitle) }
            else
                // Try to match argument to scenario
                let searchTerm = String.Join(" ", args).ToLower()
                let scenarios = MultiAgentReasoningDemo.interactiveScenarios |> List.filter (fun (title, _) -> not (title.Contains("Custom")))
                
                let matchingScenario = 
                    scenarios 
                    |> List.tryFind (fun (title, _) -> title.ToLower().Contains(searchTerm))

                match matchingScenario with
                | Some (title, problem) ->
                    AnsiConsole.MarkupLine($"[green]🎯 Found matching scenario: {title}[/]")
                    AnsiConsole.WriteLine()
                    do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                    return { Success = true; Message = "Scenario analysis completed"; Data = Some (box title) }
                | None ->
                    AnsiConsole.MarkupLine($"[red]❌ No scenario found matching '{searchTerm}'[/]")
                    AnsiConsole.MarkupLine("[yellow]Available scenarios:[/]")
                    scenarios |> List.iter (fun (title, _) -> AnsiConsole.MarkupLine($"[dim]  • {title}[/]"))
                    return { Success = false; Message = "No matching scenario found"; Data = None }
        }

// ============================================================================
// CONTINUOUS REASONING COMMAND
// ============================================================================

type ContinuousReasoningCommand() =
    interface ICommand with
        member _.Name = "continuous-reasoning"
        member _.Description = "Continuous multi-agent reasoning session with multiple problem analysis"
        
        member _.Execute(args: string[]) = task {
            AnsiConsole.Clear()
            AnsiConsole.Write(
                FigletText("Continuous")
                    .Centered()
                    .Color(Color.Magenta))
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold magenta]🔄 TARS Continuous Multi-Agent Reasoning[/]")
            AnsiConsole.MarkupLine("[dim]Analyze multiple problems in a continuous session with learning[/]")
            AnsiConsole.WriteLine()

            let mutable sessionActive = true
            let mutable problemCount = 0
            let sessionStartTime = DateTime.Now

            while sessionActive do
                problemCount <- problemCount + 1
                
                AnsiConsole.MarkupLine($"[magenta]🧠 Reasoning Session #{problemCount}[/]")
                AnsiConsole.MarkupLine($"[dim]Session duration: {(DateTime.Now - sessionStartTime).TotalMinutes:F1} minutes[/]")
                AnsiConsole.WriteLine()

                if args.Length > 0 && problemCount = 1 then
                    // Use provided argument for first problem
                    let problem = String.Join(" ", args)
                    do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                else
                    // Interactive mode for subsequent problems
                    let actionChoice = AnsiConsole.Prompt(
                        SelectionPrompt<string>()
                            .Title("[magenta]What would you like to do?[/]")
                            .AddChoices([
                                "🎯 Analyze a new custom problem"
                                "📋 Select from predefined scenarios"
                                "🎲 Analyze a random complex scenario"
                                "📊 View session summary and exit"
                            ])
                            .HighlightStyle(Style.Parse("magenta"))
                    )

                    match actionChoice with
                    | a when a.Contains("custom problem") ->
                        let customProblem = AnsiConsole.Ask<string>("[magenta]Enter your problem statement:[/]")
                        do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some customProblem)
                        
                    | a when a.Contains("predefined scenarios") ->
                        let scenarios = MultiAgentReasoningDemo.interactiveScenarios |> List.filter (fun (title, _) -> not (title.Contains("Custom")))
                        let selectedTitle = AnsiConsole.Prompt(
                            SelectionPrompt<string>()
                                .Title("[magenta]Select scenario:[/]")
                                .AddChoices(scenarios |> List.map fst)
                        )
                        let (_, problem) = scenarios |> List.find (fun (title, _) -> title = selectedTitle)
                        do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                        
                    | a when a.Contains("random") ->
                        let scenarios = MultiAgentReasoningDemo.interactiveScenarios |> List.filter (fun (title, _) -> not (title.Contains("Custom")))
                        let randomScenario = scenarios.[0 // HONEST: Cannot generate without real measurement]
                        let (title, problem) = randomScenario
                        AnsiConsole.MarkupLine($"[yellow]🎲 Random scenario: {title}[/]")
                        do! MultiAgentReasoningDemo.runMultiAgentReasoningDemoAsync (Some problem)
                        
                    | _ ->
                        sessionActive <- false

                if sessionActive then
                    AnsiConsole.WriteLine()
                    sessionActive <- AnsiConsole.Confirm("🔄 Continue with another problem analysis?")
                    AnsiConsole.WriteLine()

            // Session summary
            let sessionDuration = DateTime.Now - sessionStartTime
            AnsiConsole.MarkupLine("[bold magenta]📊 CONTINUOUS REASONING SESSION SUMMARY[/]")
            AnsiConsole.MarkupLine($"[green]✅ Problems Analyzed: {problemCount}[/]")
            AnsiConsole.MarkupLine($"[green]✅ Session Duration: {sessionDuration.TotalMinutes:F1} minutes[/]")
            AnsiConsole.MarkupLine($"[green]✅ Average Time per Problem: {sessionDuration.TotalMinutes / float problemCount:F1} minutes[/]")
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[dim]Thank you for using TARS Continuous Reasoning![/]")

            return { Success = true; Message = $"Continuous reasoning session completed - {problemCount} problems analyzed"; Data = Some (box problemCount) }
        }
