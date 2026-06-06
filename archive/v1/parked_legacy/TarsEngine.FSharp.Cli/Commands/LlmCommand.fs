namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Services

/// Generic LLM Command that works with any model provider
type LlmCommand(logger: ILogger<LlmCommand>, llmService: GenericLlmService) =
    
    interface ICommand with
        member _.Name = "llm"
        member _.Description = "Generic LLM interface supporting multiple models and providers"
        member _.Usage = "tars llm [chat|models|test] [model] [prompt]"
        member _.Examples = [
            "tars llm models"
            "tars llm test llama3:latest"
            "tars llm chat llama3:latest Hello from TARS!"
            "tars llm chat mistral Explain quantum computing"
        ]

        member _.ValidateOptions(options: CommandOptions) = true  // Basic validation

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match options.Arguments with
                    | [] | "help" :: _ ->
                        LlmCommand.showHelp()
                        return CommandResult.success("Help displayed")

                    | "models" :: _ ->
                        return! LlmCommand.listModels(llmService, logger)

                    | "test" :: model :: _ ->
                        return! LlmCommand.testModel(llmService, logger, model)

                    | "chat" :: model :: promptParts ->
                        let prompt = String.concat " " promptParts
                        return! LlmCommand.chatWithModel(llmService, logger, model, prompt, None)

                    | _ ->
                        AnsiConsole.MarkupLine("[red]❌ Invalid command. Use 'tars llm help' for usage.[/]")
                        AnsiConsole.WriteLine(sprintf "Received arguments: %A" options.Arguments)
                        return CommandResult.failure("Invalid command")
                        
                with
                | ex ->
                    logger.LogError(ex, "LLM command error")
                    AnsiConsole.MarkupLine(sprintf "[red]❌ Error: %s[/]" ex.Message)
                    return CommandResult.failure(ex.Message)
            }
    
    static member showHelp() =
        let helpText =
            "[bold yellow]🤖 TARS Generic LLM Interface[/]\n\n" +
            "[bold]Commands:[/]\n" +
            "  models                                    List available models\n" +
            "  test --model <name>                       Test a specific model\n" +
            "  chat --model <name> --prompt <text>      Chat with a model\n\n" +
            "[bold]Supported Providers:[/]\n" +
            "  • Ollama (local models): llama3.1, mistral, codellama, phi3, etc.\n" +
            "  • OpenAI (future): gpt-4, gpt-3.5-turbo\n" +
            "  • Anthropic (future): claude-3, claude-2\n\n" +
            "[bold]Examples:[/]\n" +
            "  tars llm models\n" +
            "  tars llm test --model llama3.1\n" +
            "  tars llm chat --model mistral --prompt \"Explain quantum computing\""

        let helpPanel = Panel(helpText)
        helpPanel.Header <- PanelHeader("Generic LLM Help")
        helpPanel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(helpPanel)
    
    static member listModels(llmService: GenericLlmService, logger: ILogger<LlmCommand>) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🤖 Available Models[/]")
            AnsiConsole.WriteLine()
            
            let! result = llmService.ListAvailableModels() |> Async.StartAsTask
            
            match result with
            | Ok models ->
                if models.Length > 0 then
                    let modelsTable = Table()
                    modelsTable.AddColumn("Model Name") |> ignore
                    modelsTable.AddColumn("Status") |> ignore
                    
                    for model in models do
                        modelsTable.AddRow(model, "[green]✅ Available[/]") |> ignore
                    
                    AnsiConsole.Write(modelsTable)
                    AnsiConsole.WriteLine()
                    
                    AnsiConsole.MarkupLine("[bold green]📋 Recommended Models:[/]")
                    let recommendedTable = Table()
                    recommendedTable.AddColumn("Model") |> ignore
                    recommendedTable.AddColumn("Description") |> ignore
                    
                    for (model, description) in llmService.GetRecommendedModels() do
                        let status = if Array.contains model models then "[green]✅[/]" else "[yellow]📥[/]"
                        recommendedTable.AddRow($"{status} {model}", description) |> ignore
                    
                    AnsiConsole.Write(recommendedTable)
                else
                    AnsiConsole.MarkupLine("[yellow]⚠️ No models found. Try pulling a model first:[/]")
                    AnsiConsole.MarkupLine("[dim]ollama pull llama3.1[/]")
                
                return CommandResult.success("Models listed")
                
            | Error error ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Error listing models: %s[/]" error)
                AnsiConsole.MarkupLine("[dim]Make sure Ollama is running: ollama serve[/]")
                return CommandResult.failure(error)
        }
    
    static member testModel(llmService: GenericLlmService, logger: ILogger<LlmCommand>, model: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🧪 Testing model: %s[/]" model)
            AnsiConsole.WriteLine()
            
            let testRequest = {
                Model = model
                Prompt = "Hello! Please respond with a brief greeting and confirm you are working correctly."
                SystemPrompt = Some "You are a helpful AI assistant. Keep responses concise."
                Temperature = Some 0.7
                MaxTokens = Some 100
                Context = None
            }
            
            let! response = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("green"))
                    .StartAsync("Testing model...", fun ctx ->
                        task {
                            ctx.Status <- sprintf "Sending test prompt to %s..." model
                            return! llmService.SendRequest(testRequest) |> Async.StartAsTask
                        })
            
            if response.Success then
                AnsiConsole.MarkupLine("[green]✅ Model test successful![/]")
                AnsiConsole.WriteLine()
                
                let responsePanel = Panel(response.Content)
                responsePanel.Header <- PanelHeader(sprintf "Response from %s" model)
                responsePanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(responsePanel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]Response time: %s[/]" (response.ResponseTime.ToString(@"mm\:ss\.fff")))
                
                return CommandResult.success("Model test completed")
            else
                AnsiConsole.MarkupLine("[red]❌ Model test failed![/]")
                match response.Error with
                | Some error -> AnsiConsole.MarkupLine(sprintf "[red]Error: %s[/]" error)
                | None -> ()
                
                return CommandResult.failure("Model test failed")
        }
    
    static member chatWithModel(llmService: GenericLlmService, logger: ILogger<LlmCommand>, model: string, prompt: string, systemPrompt: string option) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]💬 Chatting with %s[/]" model)
            AnsiConsole.WriteLine()
            
            let chatRequest = {
                Model = model
                Prompt = prompt
                SystemPrompt = systemPrompt
                Temperature = Some 0.7
                MaxTokens = Some 1000
                Context = None
            }
            
            let! response = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("green"))
                    .StartAsync("Generating response...", fun ctx ->
                        task {
                            ctx.Status <- sprintf "Thinking with %s..." model
                            return! llmService.SendRequest(chatRequest) |> Async.StartAsTask
                        })
            
            if response.Success then
                AnsiConsole.MarkupLine("[green]✅ Response generated![/]")
                AnsiConsole.WriteLine()
                
                // Show the prompt
                let promptPanel = Panel(prompt)
                promptPanel.Header <- PanelHeader("Your Prompt")
                promptPanel.Border <- BoxBorder.Rounded
                promptPanel.BorderStyle <- Style.Parse("blue")
                AnsiConsole.Write(promptPanel)
                
                AnsiConsole.WriteLine()
                
                // Show the response
                let responsePanel = Panel(response.Content)
                responsePanel.Header <- PanelHeader(sprintf "%s Response" model)
                responsePanel.Border <- BoxBorder.Rounded
                responsePanel.BorderStyle <- Style.Parse("green")
                AnsiConsole.Write(responsePanel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]Model: %s | Response time: %s[/]" response.Model (response.ResponseTime.ToString(@"mm\:ss\.fff")))
                
                return CommandResult.success("Chat completed")
            else
                AnsiConsole.MarkupLine("[red]❌ Chat failed![/]")
                match response.Error with
                | Some error -> 
                    AnsiConsole.MarkupLine(sprintf "[red]Error: %s[/]" error)
                    if error.Contains("not available") then
                        AnsiConsole.MarkupLine(sprintf "[yellow]💡 Try: ollama pull %s[/]" model)
                | None -> ()
                
                return CommandResult.failure("Chat failed")
        }
