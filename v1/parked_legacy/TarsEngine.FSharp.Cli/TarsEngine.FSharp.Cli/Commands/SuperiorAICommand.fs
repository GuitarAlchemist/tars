namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Core

/// Superior AI Command - Uses state-of-the-art LLMs with intelligent routing
type SuperiorAICommand(logger: ILogger<SuperiorAICommand>, llmService: AdvancedLLMService) =

    member private self.ShowSuperiorAIHeader() =
        AnsiConsole.Clear()
        
        let figlet = FigletText("SUPERIOR AI")
        figlet.Color <- Color.Magenta1
        AnsiConsole.Write(figlet)
        
        let rule = Rule("[bold magenta1]STATE-OF-THE-ART LLM INTEGRATION[/]")
        rule.Style <- Style.Parse("magenta1")
        AnsiConsole.Write(rule)
        AnsiConsole.WriteLine()

    member private self.ShowAvailableModels() =
        AnsiConsole.MarkupLine("[bold magenta]🤖 AVAILABLE SUPERIOR MODELS:[/]")
        
        let table = Table()
        table.Border <- TableBorder.Rounded
        table.BorderStyle <- Style.Parse("magenta")
        
        table.AddColumn("[bold magenta]Model[/]") |> ignore
        table.AddColumn("[bold magenta]Provider[/]") |> ignore
        table.AddColumn("[bold magenta]Reasoning[/]") |> ignore
        table.AddColumn("[bold magenta]Code[/]") |> ignore
        table.AddColumn("[bold magenta]Vision[/]") |> ignore
        table.AddColumn("[bold magenta]Cost[/]") |> ignore
        
        // GPT-4 family
        table.AddRow("GPT-4o (Latest)", "OpenAI", "✅ Excellent", "✅ Advanced", "✅ Yes", "💰 Paid") |> ignore
        table.AddRow("GPT-4 Turbo", "OpenAI", "✅ Excellent", "✅ Advanced", "✅ Yes", "💰 Paid") |> ignore
        
        // Claude 3 family
        table.AddRow("Claude 3 Opus", "Anthropic", "✅ Superior", "✅ Excellent", "✅ Yes", "💰 Paid") |> ignore
        table.AddRow("Claude 3.5 Sonnet", "Anthropic", "✅ Excellent", "✅ Excellent", "✅ Yes", "💰 Paid") |> ignore
        
        // Gemini family
        table.AddRow("Gemini 1.5 Pro", "Google", "✅ Very Good", "✅ Good", "✅ Yes", "💰 Paid") |> ignore
        
        // Local models
        table.AddRow("Qwen2 72B", "Ollama (Local)", "✅ Very Good", "✅ Good", "❌ No", "🆓 Free") |> ignore
        table.AddRow("Llama 3 70B", "Ollama (Local)", "✅ Good", "✅ Good", "❌ No", "🆓 Free") |> ignore
        table.AddRow("Code Llama 34B", "Ollama (Local)", "⚠️ Basic", "✅ Excellent", "❌ No", "🆓 Free") |> ignore
        table.AddRow("Mixtral 8x7B", "Ollama (Local)", "✅ Good", "✅ Good", "❌ No", "🆓 Free") |> ignore
        
        AnsiConsole.Write(table)
        
        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[bold cyan]💡 Model Selection:[/]")
        AnsiConsole.MarkupLine("• [yellow]Reasoning tasks[/]: Claude 3 Opus > GPT-4o > Qwen2 72B")
        AnsiConsole.MarkupLine("• [yellow]Code generation[/]: GPT-4o > Claude 3.5 Sonnet > Code Llama")
        AnsiConsole.MarkupLine("• [yellow]Vision tasks[/]: GPT-4o > Gemini Pro > Claude 3 Opus")
        AnsiConsole.MarkupLine("• [yellow]Local/Private[/]: Qwen2 72B > Llama 3 70B > Mixtral")

    member private self.TestAllModels() =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🧪 Testing All Available Models...[/]")
            AnsiConsole.WriteLine()
            
            let! results = llmService.TestAllModelsAsync()
            
            let table = Table()
            table.Border <- TableBorder.Rounded
            table.BorderStyle <- Style.Parse("cyan")
            
            table.AddColumn("[bold cyan]Model[/]") |> ignore
            table.AddColumn("[bold cyan]Status[/]") |> ignore
            table.AddColumn("[bold cyan]Details[/]") |> ignore
            
            for (modelName, success, details) in results do
                let statusColor = if success then "green" else "red"
                let statusText = if success then "✅ Working" else "❌ Failed"
                table.AddRow(modelName, $"[{statusColor}]{statusText}[/]", details) |> ignore
            
            AnsiConsole.Write(table)
            
            let workingCount = results |> List.filter (fun (_, success, _) -> success) |> List.length
            let totalCount = results.Length
            
            AnsiConsole.WriteLine()
            if workingCount > 0 then
                AnsiConsole.MarkupLine($"[bold green]🎉 {workingCount}/{totalCount} models are working![/]")
            else
                AnsiConsole.MarkupLine($"[bold red]❌ No models are currently available. Check API keys and Ollama setup.[/]")
        }

    member private self.RunInteractiveChat(?preferredModel: LLMProvider) =
        task {
            AnsiConsole.MarkupLine("[bold green]🤖 Superior AI Chat - Type 'exit' to quit, 'models' to switch[/]")
            AnsiConsole.WriteLine()
            
            let mutable continueChat = true
            let mutable currentModel = preferredModel
            
            while continueChat do
                let userInput = AnsiConsole.Ask<string>("[bold cyan]You:[/] ")
                
                match userInput.ToLower() with
                | "exit" ->
                    continueChat <- false
                    AnsiConsole.MarkupLine("[bold yellow]👋 Goodbye![/]")
                
                | "models" ->
                    AnsiConsole.MarkupLine("[bold magenta]Available models:[/]")
                    AnsiConsole.MarkupLine("1. [yellow]gpt4o[/] - GPT-4o (OpenAI)")
                    AnsiConsole.MarkupLine("2. [yellow]claude[/] - Claude 3.5 Sonnet (Anthropic)")
                    AnsiConsole.MarkupLine("3. [yellow]gemini[/] - Gemini 1.5 Pro (Google)")
                    AnsiConsole.MarkupLine("4. [yellow]qwen[/] - Qwen2 72B (Local)")
                    AnsiConsole.MarkupLine("5. [yellow]llama[/] - Llama 3 70B (Local)")
                    AnsiConsole.MarkupLine("6. [yellow]auto[/] - Auto-select best model")
                    
                    let modelChoice = AnsiConsole.Ask<string>("[bold cyan]Choose model:[/] ")
                    currentModel <- 
                        match modelChoice.ToLower() with
                        | "gpt4o" | "1" -> Some OpenAI_GPT4o
                        | "claude" | "2" -> Some Anthropic_Claude3_Sonnet
                        | "gemini" | "3" -> Some Google_Gemini_Pro
                        | "qwen" | "4" -> Some Ollama_Qwen2_72B
                        | "llama" | "5" -> Some Ollama_Llama3_70B
                        | "auto" | "6" -> None
                        | _ -> currentModel
                    
                    let modelName = 
                        match currentModel with
                        | Some model -> model.ToString()
                        | None -> "Auto-select"
                    AnsiConsole.MarkupLine($"[green]✅ Switched to: {modelName}[/]")
                
                | _ ->
                    AnsiConsole.MarkupLine("[bold green]AI:[/] [dim]Thinking...[/]")
                    
                    try
                        let! result = 
                            match currentModel with
                            | Some model -> llmService.QueryAsync(userInput, preferredProvider = model)
                            | None -> llmService.QueryAsync(userInput, taskType = "conversation")
                        
                        match result with
                        | Ok response ->
                            AnsiConsole.MarkupLine($"[bold green]AI ({response.Model}):[/] {response.Content}")
                            AnsiConsole.MarkupLine($"[dim]({response.TokensUsed} tokens, {response.ResponseTime.TotalMilliseconds:F0}ms, confidence: {response.Confidence:F2})[/]")
                        | Error error ->
                            AnsiConsole.MarkupLine($"[bold red]AI Error:[/] {error}")
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[bold red]Exception:[/] {ex.Message}")
                    
                    AnsiConsole.WriteLine()
        }

    member private self.ShowSetupInstructions() =
        AnsiConsole.MarkupLine("[bold yellow]🔧 SETUP INSTRUCTIONS:[/]")
        AnsiConsole.WriteLine()
        
        let panel = Panel("""
[bold cyan]API Keys (for cloud models):[/]
• OpenAI: Set OPENAI_API_KEY environment variable
• Anthropic: Set ANTHROPIC_API_KEY environment variable  
• Google: Set GOOGLE_API_KEY environment variable

[bold cyan]Local Models (Ollama):[/]
• Install Ollama: https://ollama.ai
• Pull models: ollama pull qwen2:72b
• Pull models: ollama pull llama3:70b
• Pull models: ollama pull codellama:34b
• Pull models: ollama pull mixtral:8x7b

[bold cyan]Recommended Setup:[/]
1. Start with local models (free, private)
2. Add API keys for advanced capabilities
3. Use 'tars superior test' to verify setup
        """)
        panel.Header <- PanelHeader("[bold yellow]Setup Guide[/]")
        panel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(panel)

    interface ICommand with
        member _.Name = "superior"
        member _.Description = "Superior AI with state-of-the-art LLMs (GPT-4, Claude 3, Gemini, Qwen2, etc.)"
        member self.Usage = "tars superior [models|test|chat|setup] [query]"
        member self.Examples = [
            "tars superior models"
            "tars superior test"
            "tars superior chat"
            "tars superior setup"
        ]
        member self.ValidateOptions(options) = true

        member self.ExecuteAsync(options) =
            task {
                try
                    self.ShowSuperiorAIHeader()
                    
                    match options.Arguments with
                    | "models" :: _ ->
                        self.ShowAvailableModels()
                        return CommandResult.success("Models displayed")
                    
                    | "test" :: _ ->
                        do! self.TestAllModels()
                        return CommandResult.success("Model testing completed")
                    
                    | "chat" :: _ ->
                        do! self.RunInteractiveChat()
                        return CommandResult.success("Chat session completed")
                    
                    | "setup" :: _ ->
                        self.ShowSetupInstructions()
                        return CommandResult.success("Setup instructions displayed")
                    
                    | query :: _ when not (query.StartsWith("-")) ->
                        AnsiConsole.MarkupLine($"[bold cyan]Query:[/] {query}")
                        AnsiConsole.MarkupLine("[bold green]AI:[/] [dim]Processing with best available model...[/]")
                        
                        let! result = llmService.QueryAsync(query, taskType = "general", requiresLocal = true)
                        match result with
                        | Ok response ->
                            let escapedContent = response.Content.Replace("[", "[[").Replace("]", "]]")
                            let panel = Panel(escapedContent)
                            panel.Header <- PanelHeader($"[bold green]{response.Model} Response[/]")
                            panel.Border <- BoxBorder.Rounded
                            AnsiConsole.Write(panel)
                            AnsiConsole.MarkupLine($"[dim]({response.TokensUsed} tokens, {response.ResponseTime.TotalMilliseconds:F0}ms, confidence: {response.Confidence:F2})[/]")
                            return CommandResult.success("Query processed")
                        | Error error ->
                            AnsiConsole.MarkupLine($"[bold red]❌ Error: {error}[/]")
                            return CommandResult.failure(error)
                    
                    | [] ->
                        self.ShowAvailableModels()
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold yellow]💡 Quick Start:[/]")
                        AnsiConsole.MarkupLine("• [cyan]tars superior test[/] - Test all models")
                        AnsiConsole.MarkupLine("• [cyan]tars superior chat[/] - Interactive chat")
                        AnsiConsole.MarkupLine("• [cyan]tars superior setup[/] - Setup instructions")
                        AnsiConsole.MarkupLine("• [cyan]tars superior \"your question\"[/] - Direct query")
                        return CommandResult.success("Superior AI command completed")
                    
                    | unknown :: _ ->
                        AnsiConsole.MarkupLine($"[red]❌ Unknown command: {unknown}[/]")
                        AnsiConsole.MarkupLine("[yellow]Valid commands: models, test, chat, setup[/]")
                        return CommandResult.failure($"Unknown command: {unknown}")
                with
                | ex ->
                    logger.LogError(ex, "Error in superior AI command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
