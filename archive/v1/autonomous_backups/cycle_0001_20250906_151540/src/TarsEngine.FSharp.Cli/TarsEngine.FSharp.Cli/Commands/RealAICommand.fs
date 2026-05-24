namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Core

/// Real AI Command - Demonstrates genuine AI capabilities with honest reporting
type RealAICommand(logger: ILogger<RealAICommand>, mixtralService: MixtralService) =

    member private self.ShowAIHeader() =
        AnsiConsole.Clear()
        
        let figlet = FigletText("REAL AI")
        figlet.Color <- Color.Green
        AnsiConsole.Write(figlet)
        
        let rule = Rule("[bold green]GENUINE AI CAPABILITIES - NO SIMULATIONS[/]")
        rule.Style <- Style.Parse("green")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    member private self.TestOllamaConnection() =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔧 Testing Ollama Connection...[/]")
            
            try
                // Test with a simple query
                let! result = mixtralService.QueryAsync("Hello, can you respond with just 'AI is working'?")
                
                match result with
                | Ok response ->
                    AnsiConsole.MarkupLine("[bold green]✅ Ollama Connection: WORKING[/]")
                    AnsiConsole.MarkupLine($"[green]Response: {response.Content.Substring(0, Math.Min(100, response.Content.Length))}...[/]")
                    AnsiConsole.MarkupLine($"[dim]Tokens used: {response.TokensUsed}, Response time: {response.ResponseTime.TotalMilliseconds:F0}ms[/]")
                    return true
                | Error error ->
                    AnsiConsole.MarkupLine("[bold red]❌ Ollama Connection: FAILED[/]")
                    AnsiConsole.MarkupLine($"[red]Error: {error}[/]")
                    return false
            with
            | ex ->
                AnsiConsole.MarkupLine("[bold red]❌ Ollama Connection: FAILED[/]")
                AnsiConsole.MarkupLine($"[red]Exception: {ex.Message}[/]")
                return false
        }

    member private self.TestExpertRouting() =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔧 Testing Expert Routing...[/]")
            
            try
                // Test code generation expert
                let! codeResult = mixtralService.QueryAsync("Write a simple F# function that adds two numbers", CodeGeneration)
                
                match codeResult with
                | Ok response ->
                    AnsiConsole.MarkupLine("[bold green]✅ Code Generation Expert: WORKING[/]")
                    AnsiConsole.MarkupLine($"[green]Expert: {response.UsedExperts.[0].Name}[/]")
                    AnsiConsole.MarkupLine($"[dim]Confidence: {response.Confidence:F2}, Tokens: {response.TokensUsed}[/]")
                    
                    // Show a snippet of the response
                    let snippet = response.Content.Substring(0, Math.Min(200, response.Content.Length))
                    AnsiConsole.MarkupLine($"[yellow]Response snippet: {snippet}...[/]")
                    return true
                | Error error ->
                    AnsiConsole.MarkupLine("[bold red]❌ Code Generation Expert: FAILED[/]")
                    AnsiConsole.MarkupLine($"[red]Error: {error}[/]")
                    return false
            with
            | ex ->
                AnsiConsole.MarkupLine("[bold red]❌ Expert Routing: FAILED[/]")
                AnsiConsole.MarkupLine($"[red]Exception: {ex.Message}[/]")
                return false
        }

    member private self.TestEnsembleMode() =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔧 Testing Ensemble Mode...[/]")
            
            try
                // Test ensemble with multiple experts
                let! ensembleResult = mixtralService.QueryAsync("Explain the benefits of functional programming", useEnsemble = true)
                
                match ensembleResult with
                | Ok response ->
                    AnsiConsole.MarkupLine("[bold green]✅ Ensemble Mode: WORKING[/]")
                    AnsiConsole.MarkupLine($"[green]Experts used: {response.UsedExperts.Length}[/]")
                    AnsiConsole.MarkupLine($"[dim]Total tokens: {response.TokensUsed}, Response time: {response.ResponseTime.TotalMilliseconds:F0}ms[/]")
                    return true
                | Error error ->
                    AnsiConsole.MarkupLine("[bold red]❌ Ensemble Mode: FAILED[/]")
                    AnsiConsole.MarkupLine($"[red]Error: {error}[/]")
                    return false
            with
            | ex ->
                AnsiConsole.MarkupLine("[bold red]❌ Ensemble Mode: FAILED[/]")
                AnsiConsole.MarkupLine($"[red]Exception: {ex.Message}[/]")
                return false
        }

    member private self.RunInteractiveAI() =
        task {
            AnsiConsole.MarkupLine("[bold green]🤖 Interactive AI Mode - Type 'exit' to quit[/]")
            AnsiConsole.WriteLine()
            
            let mutable continueChat = true
            
            while continueChat do
                let userInput = AnsiConsole.Ask<string>("[bold cyan]You:[/] ")
                
                if userInput.ToLower() = "exit" then
                    continueChat <- false
                    AnsiConsole.MarkupLine("[bold yellow]👋 Goodbye![/]")
                else
                    AnsiConsole.MarkupLine("[bold green]AI:[/] Processing...")
                    
                    try
                        let! result = mixtralService.QueryAsync(userInput)
                        
                        match result with
                        | Ok response ->
                            AnsiConsole.MarkupLine($"[bold green]AI:[/] {response.Content}")
                            AnsiConsole.MarkupLine($"[dim]({response.TokensUsed} tokens, {response.ResponseTime.TotalMilliseconds:F0}ms)[/]")
                        | Error error ->
                            AnsiConsole.MarkupLine($"[bold red]AI Error:[/] {error}")
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[bold red]Exception:[/] {ex.Message}")
                    
                    AnsiConsole.WriteLine()
        }

    member private self.ShowRealAICapabilities() =
        task {
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold yellow]🎯 REAL AI CAPABILITIES:[/]")
            
            let table = Table()
            table.Border <- TableBorder.Rounded
            table.BorderStyle <- Style.Parse("green")
            
            table.AddColumn("[bold green]Feature[/]") |> ignore
            table.AddColumn("[bold green]Status[/]") |> ignore
            table.AddColumn("[bold green]Description[/]") |> ignore
            
            table.AddRow("Ollama Integration", "✅ Real", "Genuine HTTP calls to local Ollama server") |> ignore
            table.AddRow("Expert Routing", "✅ Real", "Intelligent routing to specialized AI experts") |> ignore
            table.AddRow("Ensemble Mode", "✅ Real", "Multiple experts working together") |> ignore
            table.AddRow("Code Generation", "✅ Real", "Actual code generation using AI") |> ignore
            table.AddRow("Code Analysis", "✅ Real", "Real static analysis and code review") |> ignore
            table.AddRow("Interactive Chat", "✅ Real", "Live conversation with AI") |> ignore
            
            AnsiConsole.Write(table)
            
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("[bold cyan]💡 Requirements:[/]")
            AnsiConsole.MarkupLine("• Ollama must be running on localhost:11434")
            AnsiConsole.MarkupLine("• Mixtral model must be available in Ollama")
            AnsiConsole.MarkupLine("• Install with: [yellow]ollama pull mixtral[/]")
        }

    interface ICommand with
        member _.Name = "ai"
        member _.Description = "Real AI capabilities - genuine LLM integration with honest reporting"
        member self.Usage = "tars ai [test|chat|capabilities]"
        member self.Examples = [
            "tars ai test"
            "tars ai chat"
            "tars ai capabilities"
        ]
        member self.ValidateOptions(options) = true

        member self.ExecuteAsync(options) =
            task {
                try
                    self.ShowAIHeader()
                    
                    match options.Arguments with
                    | "test" :: _ ->
                        AnsiConsole.MarkupLine("[bold cyan]🧪 Running Real AI Tests...[/]")
                        AnsiConsole.WriteLine()
                        
                        let! ollamaWorking = self.TestOllamaConnection()
                        let! expertWorking = if ollamaWorking then self.TestExpertRouting() else task { return false }
                        let! ensembleWorking = if ollamaWorking then self.TestEnsembleMode() else task { return false }
                        
                        AnsiConsole.WriteLine()
                        if ollamaWorking && expertWorking && ensembleWorking then
                            AnsiConsole.MarkupLine("[bold green]🎉 ALL AI TESTS PASSED - Real AI is working![/]")
                            return CommandResult.success("All AI tests passed")
                        else
                            AnsiConsole.MarkupLine("[bold red]❌ Some AI tests failed - Check Ollama setup[/]")
                            return CommandResult.failure("AI tests failed")
                    
                    | "chat" :: _ ->
                        do! self.RunInteractiveAI()
                        return CommandResult.success("Interactive AI session completed")
                    
                    | "capabilities" :: _ ->
                        do! self.ShowRealAICapabilities()
                        return CommandResult.success("AI capabilities displayed")
                    
                    | [] ->
                        do! self.ShowRealAICapabilities()
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold yellow]Use 'tars ai test' to test AI functionality[/]")
                        AnsiConsole.MarkupLine("[bold yellow]Use 'tars ai chat' for interactive AI conversation[/]")
                        return CommandResult.success("AI command completed")
                    
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown AI command: {unknown}[/]")
                        return CommandResult.failure($"Unknown command: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in AI command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
