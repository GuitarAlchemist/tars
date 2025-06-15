namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Services

/// TARS-aware LLM Command with integrated knowledge base and context
type TarsLlmCommand(logger: ILogger<TarsLlmCommand>, knowledgeService: TarsKnowledgeService) =
    
    interface ICommand with
        member _.Name = "tars-llm"
        member _.Description = "TARS-aware LLM interface with integrated knowledge base and context"
        member _.Usage = "tars tars-llm [init|status|chat|ask|search] [options]"
        member _.Examples = [
            "tars tars-llm init"
            "tars tars-llm status"
            "tars tars-llm chat llama3:latest What are your capabilities?"
            "tars tars-llm ask llama3:latest How do I create a metascript?"
            "tars tars-llm search vector store"
        ]
        
        member _.ValidateOptions(options: CommandOptions) = true
        
        member _.ExecuteAsync(options: CommandOptions) =
            task {
                try
                    match options.Arguments with
                    | [] | "help" :: _ ->
                        TarsLlmCommand.showHelp()
                        return CommandResult.success("Help displayed")
                        
                    | "init" :: _ ->
                        return! TarsLlmCommand.initializeKnowledgeBase(knowledgeService, logger)
                        
                    | "status" :: _ ->
                        return! TarsLlmCommand.showSystemStatus(knowledgeService, logger)
                        
                    | "chat" :: model :: promptParts ->
                        let prompt = String.concat " " promptParts
                        return! TarsLlmCommand.chatWithTars(knowledgeService, logger, model, prompt)
                        
                    | "ask" :: model :: promptParts ->
                        let prompt = String.concat " " promptParts
                        return! TarsLlmCommand.askTarsQuestion(knowledgeService, logger, model, prompt)
                        
                    | "search" :: queryParts ->
                        let query = String.concat " " queryParts
                        return! TarsLlmCommand.searchKnowledge(knowledgeService, logger, query)
                        
                    | _ ->
                        AnsiConsole.MarkupLine("[red]❌ Invalid command. Use 'tars tars-llm help' for usage.[/]")
                        return CommandResult.failure("Invalid command")
                        
                with
                | ex ->
                    logger.LogError(ex, "TARS LLM command error")
                    AnsiConsole.MarkupLine(sprintf "[red]❌ Error: %s[/]" ex.Message)
                    return CommandResult.failure(ex.Message)
            }
    
    static member showHelp() =
        let helpText = 
            "[bold yellow]🧠 TARS-Aware LLM Interface[/]\n\n" +
            "[bold]Commands:[/]\n" +
            "  init                             Initialize TARS knowledge base\n" +
            "  status                           Show TARS system status\n" +
            "  chat <model> <prompt>            Chat with TARS-aware AI\n" +
            "  ask <model> <question>           Ask TARS-specific questions\n" +
            "  search <query>                   Search TARS knowledge base\n\n" +
            "[bold]Features:[/]\n" +
            "  • Context-aware responses using CUDA vector store\n" +
            "  • Deep knowledge of TARS architecture and capabilities\n" +
            "  • Real-time code and documentation search\n" +
            "  • Cryptographically verified execution context\n\n" +
            "[bold]Examples:[/]\n" +
            "  tars tars-llm init\n" +
            "  tars tars-llm chat llama3:latest What are your capabilities?\n" +
            "  tars tars-llm ask mistral How do I create a metascript?\n" +
            "  tars tars-llm search cryptographic proof"
        
        let helpPanel = Panel(helpText)
        helpPanel.Header <- PanelHeader("TARS-Aware LLM Help")
        helpPanel.Border <- BoxBorder.Rounded
        AnsiConsole.Write(helpPanel)
    
    static member initializeKnowledgeBase(knowledgeService: TarsKnowledgeService, logger: ILogger<TarsLlmCommand>) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🧠 Initializing TARS Knowledge Base[/]")
            AnsiConsole.WriteLine()
            
            let! result = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("cyan"))
                    .StartAsync("Ingesting TARS codebase into vector store...", fun ctx ->
                        task {
                            ctx.Status <- "Scanning files and generating embeddings..."
                            return! knowledgeService.InitializeKnowledgeBase()
                        })
            
            match result with
            | Ok metrics ->
                AnsiConsole.MarkupLine("[green]✅ TARS Knowledge Base initialized successfully![/]")
                AnsiConsole.WriteLine()
                
                let statsText =
                    "[bold green]📊 Initialization Complete[/]\n\n" +
                    $"[cyan]Files Processed:[/] {metrics.FilesProcessed:N0}\n" +
                    $"[cyan]Embeddings Generated:[/] {metrics.EmbeddingsGenerated:N0}\n" +
                    $"[cyan]Total Size:[/] {float metrics.TotalSizeBytes / (1024.0 * 1024.0):F2} MB\n" +
                    $"[cyan]Processing Time:[/] {float metrics.IngestionTimeMs / 1000.0:F2} seconds\n" +
                    $"[cyan]Processing Rate:[/] {metrics.FilesPerSecond:F1} files/sec\n\n" +
                    "The TARS AI is now context-aware and ready for intelligent conversations!"

                let statsPanel = Panel(statsText)
                statsPanel.Header <- PanelHeader("Knowledge Base Ready")
                statsPanel.Border <- BoxBorder.Rounded
                AnsiConsole.Write(statsPanel)
                
                return CommandResult.success("Knowledge base initialized")
                
            | Error error ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Failed to initialize knowledge base: %s[/]" error)
                return CommandResult.failure(error)
        }
    
    static member showSystemStatus(knowledgeService: TarsKnowledgeService, logger: ILogger<TarsLlmCommand>) =
        task {
            AnsiConsole.MarkupLine("[bold cyan]🔍 TARS System Status[/]")
            AnsiConsole.WriteLine()
            
            let status = knowledgeService.GetSystemStatus()
            
            let statusPanel = Panel(status)
            statusPanel.Header <- PanelHeader("TARS System Overview")
            statusPanel.Border <- BoxBorder.Double
            statusPanel.BorderStyle <- Style.Parse("green")
            AnsiConsole.Write(statusPanel)
            
            return CommandResult.success("System status displayed")
        }
    
    static member chatWithTars(knowledgeService: TarsKnowledgeService, logger: ILogger<TarsLlmCommand>, model: string, prompt: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🧠 Chatting with TARS using %s[/]" model)
            AnsiConsole.WriteLine()
            
            let request = knowledgeService.CreateTarsAwareRequest(model, prompt, None)
            
            let! response = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("cyan"))
                    .StartAsync("TARS is thinking with full context...", fun ctx ->
                        task {
                            ctx.Status <- sprintf "Processing with TARS knowledge base..."
                            return! knowledgeService.SendContextualRequest(request) |> Async.StartAsTask
                        })
            
            if response.Success then
                AnsiConsole.MarkupLine("[green]✅ TARS response generated![/]")
                AnsiConsole.WriteLine()
                
                // Show the prompt
                let promptPanel = Panel(prompt)
                promptPanel.Header <- PanelHeader("Your Question to TARS")
                promptPanel.Border <- BoxBorder.Rounded
                promptPanel.BorderStyle <- Style.Parse("blue")
                AnsiConsole.Write(promptPanel)
                
                AnsiConsole.WriteLine()
                
                // Show the response (escape markup to prevent parsing errors)
                let escapedContent = response.Content.Replace("[", "[[").Replace("]", "]]")
                let responsePanel = Panel(escapedContent)
                responsePanel.Header <- PanelHeader(sprintf "TARS Response (via %s)" model)
                responsePanel.Border <- BoxBorder.Rounded
                responsePanel.BorderStyle <- Style.Parse("green")
                AnsiConsole.Write(responsePanel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]🧠 Context-aware response | Model: %s | Time: %s[/]" response.Model (response.ResponseTime.ToString(@"mm\:ss\.fff")))
                
                return CommandResult.success("TARS chat completed")
            else
                AnsiConsole.MarkupLine("[red]❌ TARS chat failed![/]")
                match response.Error with
                | Some error -> 
                    AnsiConsole.MarkupLine(sprintf "[red]Error: %s[/]" error)
                | None -> ()
                
                return CommandResult.failure("TARS chat failed")
        }
    
    static member askTarsQuestion(knowledgeService: TarsKnowledgeService, logger: ILogger<TarsLlmCommand>, model: string, question: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]❓ Asking TARS: %s[/]" (if question.Length > 50 then question.Substring(0, 50) + "..." else question))
            AnsiConsole.WriteLine()
            
            let systemPrompt = "You are TARS. Answer this question with specific, actionable information about your capabilities and how to use them. Include relevant commands and examples."
            let request = knowledgeService.CreateTarsAwareRequest(model, question, Some systemPrompt)
            
            let! response = 
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("yellow"))
                    .StartAsync("TARS is analyzing the question...", fun ctx ->
                        task {
                            return! knowledgeService.SendContextualRequest(request) |> Async.StartAsTask
                        })
            
            if response.Success then
                AnsiConsole.MarkupLine("[green]✅ TARS has an answer![/]")
                AnsiConsole.WriteLine()
                
                let escapedAnswer = response.Content.Replace("[", "[[").Replace("]", "]]")
                let answerPanel = Panel(escapedAnswer)
                answerPanel.Header <- PanelHeader("TARS Answer")
                answerPanel.Border <- BoxBorder.Double
                answerPanel.BorderStyle <- Style.Parse("yellow")
                AnsiConsole.Write(answerPanel)
                
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]💡 Knowledge-based answer | Time: %s[/]" (response.ResponseTime.ToString(@"mm\:ss\.fff")))
                
                return CommandResult.success("Question answered")
            else
                AnsiConsole.MarkupLine("[red]❌ Failed to get answer from TARS![/]")
                return CommandResult.failure("Question failed")
        }
    
    static member searchKnowledge(knowledgeService: TarsKnowledgeService, logger: ILogger<TarsLlmCommand>, query: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🔍 Searching TARS knowledge: %s[/]" query)
            AnsiConsole.WriteLine()
            
            let! result = knowledgeService.SearchKnowledge(query, 10) |> Async.StartAsTask
            
            match result with
            | Ok (documents: Document list) ->
                if documents.Length > 0 then
                    let resultsTable = Table()
                    resultsTable.AddColumn("File") |> ignore
                    resultsTable.AddColumn("Type") |> ignore
                    resultsTable.AddColumn("Size") |> ignore
                    resultsTable.AddColumn("Preview") |> ignore
                    
                    for (doc: Document) in documents do
                        let preview =
                            if doc.Content.Length > 100 then
                                doc.Content.Substring(0, 100).Replace("\n", " ") + "..."
                            else
                                doc.Content.Replace("\n", " ")

                        resultsTable.AddRow(
                            Path.GetFileName(doc.Path),
                            doc.FileType,
                            $"{doc.Size} bytes",
                            preview
                        ) |> ignore
                    
                    AnsiConsole.Write(resultsTable)
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine(sprintf "[green]✅ Found %d relevant documents[/]" documents.Length)
                else
                    AnsiConsole.MarkupLine("[yellow]⚠️ No documents found matching the query[/]")
                
                return CommandResult.success("Search completed")
                
            | Error error ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Search failed: %s[/]" error)
                return CommandResult.failure(error)
        }
