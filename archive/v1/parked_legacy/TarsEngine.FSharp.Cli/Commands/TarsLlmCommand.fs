namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Commands

/// TARS-aware LLM Command with integrated knowledge base and context
type TarsLlmCommand(logger: ILogger<TarsLlmCommand>, knowledgeService: TarsKnowledgeService, enhancedKnowledgeService: EnhancedKnowledgeService, chatSessionService: ChatSessionService) =
    
    interface ICommand with
        member _.Name = "tars-llm"
        member _.Description = "TARS-aware LLM interface with integrated knowledge base and context"
        member _.Usage = "tars tars-llm [init|status|chat|ask|search] [options]"
        member _.Examples = [
            "tars tars-llm init"
            "tars tars-llm status"
            "tars tars-llm chat llama3:latest What are your capabilities?"
            "tars tars-llm interactive llama3:latest"
            "tars tars-llm session abc123"
            "tars tars-llm save-transcript abc123"
            "tars tars-llm enhanced-chat llama3:latest What is MCP protocol?"
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
                        return! TarsLlmCommand.chatWithTarsEnhanced(enhancedKnowledgeService, logger, model, prompt)

                    | "interactive" :: model :: _ ->
                        return! TarsLlmCommand.startInteractiveSession(chatSessionService, enhancedKnowledgeService, logger, model)

                    | "session" :: sessionId :: _ ->
                        return! TarsLlmCommand.resumeSession(chatSessionService, enhancedKnowledgeService, logger, sessionId)

                    | "save-transcript" :: sessionId :: _ ->
                        return! TarsLlmCommand.saveTranscript(chatSessionService, logger, sessionId)

                    | "enhanced-chat" :: model :: promptParts ->
                        let prompt = String.concat " " promptParts
                        return! TarsLlmCommand.chatWithTarsEnhanced(enhancedKnowledgeService, logger, model, prompt)

                    | "legacy-chat" :: model :: promptParts ->
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
            "  chat <model> <prompt>            Chat with enhanced knowledge (web search + gap detection)\n" +
            "  interactive <model>              Start interactive chat session with memory\n" +
            "  session <session-id>             Resume existing chat session\n" +
            "  save-transcript <session-id>     Save session transcript to Markdown file\n" +
            "  enhanced-chat <model> <prompt>   Explicit enhanced chat with web search\n" +
            "  legacy-chat <model> <prompt>     Original TARS-only knowledge chat\n" +
            "  ask <model> <question>           Ask TARS-specific questions\n" +
            "  search <query>                   Search TARS knowledge base\n\n" +
            "[bold]Enhanced Features:[/]\n" +
            "  • Interactive chat sessions with conversation history\n" +
            "  • Short-term memory that persists during the session\n" +
            "  • Transcript saving to Markdown files for documentation\n" +
            "  • Knowledge gap detection - TARS knows when it doesn't know\n" +
            "  • Web search for unknown topics using DuckDuckGo API\n" +
            "  • Honest responses about knowledge limitations\n" +
            "  • Context-aware responses using CUDA vector store\n" +
            "  • Deep knowledge of TARS architecture and capabilities\n" +
            "  • Real-time code and documentation search\n\n" +
            "[bold]Examples:[/]\n" +
            "  tars tars-llm init\n" +
            "  tars tars-llm interactive llama3:latest\n" +
            "  tars tars-llm chat llama3:latest What is MCP protocol?\n" +
            "  tars tars-llm enhanced-chat mistral Tell me about recent AI developments\n" +
            "  tars tars-llm session abc123\n" +
            "  tars tars-llm save-transcript abc123\n" +
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
                
                let totalSizeMB = (float metrics.TotalSizeBytes / (1024.0 * 1024.0)).ToString("F2")
                let processingTimeSec = (float metrics.IngestionTimeMs / 1000.0).ToString("F2")
                let processingRateStr = metrics.FilesPerSecond.ToString("F1")
                let statsText =
                    "[bold green]📊 Initialization Complete[/]\n\n" +
                    $"[cyan]Files Processed:[/] {metrics.FilesProcessed:N0}\n" +
                    $"[cyan]Embeddings Generated:[/] {metrics.EmbeddingsGenerated:N0}\n" +
                    $"[cyan]Total Size:[/] {totalSizeMB} MB\n" +
                    $"[cyan]Processing Time:[/] {processingTimeSec} seconds\n" +
                    $"[cyan]Processing Rate:[/] {processingRateStr} files/sec\n\n" +
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

    static member chatWithTarsEnhanced(enhancedKnowledgeService: EnhancedKnowledgeService, logger: ILogger<TarsLlmCommand>, model: string, prompt: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]🧠 Chatting with TARS using %s (Enhanced Knowledge)[/]" model)
            AnsiConsole.WriteLine()

            let request = {
                Model = model
                Prompt = prompt
                SystemPrompt = None
                Temperature = Some 0.7
                MaxTokens = Some 2000
                Context = None
            }

            let! response =
                AnsiConsole.Status()
                    .Spinner(Spinner.Known.Dots)
                    .SpinnerStyle(Style.Parse("cyan"))
                    .StartAsync("TARS is thinking with enhanced knowledge...", fun ctx ->
                        task {
                            ctx.Status <- sprintf "Processing with knowledge gap detection and web search..."
                            return! enhancedKnowledgeService.CreateKnowledgeAwareResponse(request) |> Async.StartAsTask
                        })

            if response.Success then
                AnsiConsole.MarkupLine("[green]✅ TARS response generated![/]")
                AnsiConsole.WriteLine()

                // Display the user's question
                let escapedQuestion = prompt.Replace("[", "[[").Replace("]", "]]")
                let questionPanel = Panel(escapedQuestion)
                questionPanel.Header <- PanelHeader("Your Question to TARS")
                questionPanel.Border <- BoxBorder.Rounded
                questionPanel.BorderStyle <- Style.Parse("blue")
                AnsiConsole.Write(questionPanel)

                AnsiConsole.WriteLine()

                // Display TARS response
                let escapedResponse = response.Content.Replace("[", "[[").Replace("]", "]]")
                let responsePanel = Panel(escapedResponse)
                responsePanel.Header <- PanelHeader(sprintf "TARS Response (via %s)" model)
                responsePanel.Border <- BoxBorder.Double
                responsePanel.BorderStyle <- Style.Parse("green")
                AnsiConsole.Write(responsePanel)

                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine(sprintf "[dim]🧠 Enhanced knowledge-aware response | Model: %s | Time: %s[/]" model (response.ResponseTime.ToString(@"mm\:ss\.fff")))
                AnsiConsole.MarkupLine("[dim]TARS chat completed[/]")

                return CommandResult.success("TARS enhanced chat completed")
            else
                AnsiConsole.MarkupLine("[red]❌ TARS enhanced chat failed![/]")
                match response.Error with
                | Some error ->
                    AnsiConsole.MarkupLine(sprintf "[red]Error: %s[/]" error)
                | None -> ()

                return CommandResult.failure("TARS enhanced chat failed")
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

    static member startInteractiveSession(chatSessionService: ChatSessionService, enhancedKnowledgeService: EnhancedKnowledgeService, logger: ILogger<TarsLlmCommand>, model: string) =
        task {
            // Create new session
            let session = chatSessionService.CreateSession(model)

            AnsiConsole.MarkupLine($"[bold green]🎯 Started interactive TARS session: {session.SessionId}[/]")
            AnsiConsole.MarkupLine($"[cyan]Model: {model}[/]")
            AnsiConsole.MarkupLine("[dim]Type 'quit', 'exit', or 'bye' to end the session[/]")
            AnsiConsole.MarkupLine("[dim]Type '/help' for session commands[/]")
            AnsiConsole.WriteLine()

            return! TarsLlmCommand.runInteractiveLoop(chatSessionService, enhancedKnowledgeService, logger, session.SessionId)
        }

    static member resumeSession(chatSessionService: ChatSessionService, enhancedKnowledgeService: EnhancedKnowledgeService, logger: ILogger<TarsLlmCommand>, sessionId: string) =
        task {
            match chatSessionService.GetSession(sessionId) with
            | Some session ->
                AnsiConsole.MarkupLine($"[bold green]🔄 Resumed TARS session: {sessionId}[/]")
                AnsiConsole.MarkupLine($"[cyan]Model: {session.Model}[/]")

                // Show session stats
                match chatSessionService.GetSessionStats(sessionId) with
                | Some stats ->
                    let durationStr = stats.Duration.ToString(@"hh\:mm\:ss")
                    AnsiConsole.MarkupLine($"[dim]Duration: {durationStr} | Messages: {stats.MessageCount} | Memory items: {stats.MemoryItems}[/]")
                | None -> ()

                AnsiConsole.WriteLine()
                return! TarsLlmCommand.runInteractiveLoop(chatSessionService, enhancedKnowledgeService, logger, sessionId)
            | None ->
                AnsiConsole.MarkupLine($"[red]❌ Session {sessionId} not found or expired[/]")
                return CommandResult.failure("Session not found")
        }

    static member saveTranscript(chatSessionService: ChatSessionService, logger: ILogger<TarsLlmCommand>, sessionId: string) =
        task {
            match chatSessionService.SaveTranscript(sessionId) with
            | Ok fileName ->
                AnsiConsole.MarkupLine($"[green]💾 Transcript saved successfully![/]")
                AnsiConsole.MarkupLine($"[cyan]📄 File: {fileName}[/]")

                // Show session stats
                match chatSessionService.GetSessionStats(sessionId) with
                | Some stats ->
                    AnsiConsole.MarkupLine($"[dim]Session contained {stats.MessageCount} messages over {stats.Duration}[/]")
                | None -> ()

                return CommandResult.success($"Transcript saved to {fileName}")
            | Error errorMsg ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to save transcript: {errorMsg}[/]")
                return CommandResult.failure(errorMsg)
        }

    static member runInteractiveLoop(chatSessionService: ChatSessionService, enhancedKnowledgeService: EnhancedKnowledgeService, logger: ILogger<TarsLlmCommand>, sessionId: string) =
        task {
            let mutable continueLoop = true
            let mutable messageCount = 0

            while continueLoop do
                // Show prompt
                AnsiConsole.Markup($"[bold cyan]TARS[[{sessionId}]]>[/] ")
                let userInput = Console.ReadLine()

                match userInput with
                | null | "" -> () // Skip empty input
                | input when (let s = input.ToLowerInvariant() in s = "quit" || s = "exit" || s = "bye") ->
                    continueLoop <- false
                    AnsiConsole.MarkupLine("[yellow]👋 Ending TARS session. Goodbye![/]")
                    chatSessionService.CloseSession(sessionId) |> ignore
                | input when input.StartsWith("/") ->
                    // Handle session commands
                    TarsLlmCommand.handleSessionCommand(chatSessionService, sessionId, input)
                | input ->
                    // Process user message
                    messageCount <- messageCount + 1

                    // Add user message to session
                    let userMessage = {
                        Id = Guid.NewGuid().ToString()
                        Role = User
                        Content = input
                        Timestamp = DateTime.UtcNow
                        Model = None
                        ResponseTime = None
                        TokensUsed = None
                        Metadata = Map.empty
                    }

                    chatSessionService.AddMessage(sessionId, userMessage) |> ignore

                    // Get session for model info
                    match chatSessionService.GetSession(sessionId) with
                    | Some session ->
                        // Create LLM request
                        let request = {
                            Model = session.Model
                            Prompt = input
                            SystemPrompt = None
                            Temperature = Some 0.7
                            MaxTokens = Some 2000
                            Context = None
                        }

                        // Get response with session context
                        let! response = enhancedKnowledgeService.CreateKnowledgeAwareResponse(request, sessionId) |> Async.StartAsTask

                        if response.Success then
                            // Display response
                            AnsiConsole.WriteLine()
                            let escapedResponse = response.Content.Replace("[", "[[").Replace("]", "]]")
                            AnsiConsole.MarkupLine($"[green]🤖 TARS:[/] {escapedResponse}")
                            AnsiConsole.WriteLine()

                            // Add assistant message to session
                            let assistantMessage = {
                                Id = Guid.NewGuid().ToString()
                                Role = Assistant
                                Content = response.Content
                                Timestamp = DateTime.UtcNow
                                Model = Some response.Model
                                ResponseTime = Some response.ResponseTime
                                TokensUsed = response.TokensUsed
                                Metadata = Map.empty
                            }

                            chatSessionService.AddMessage(sessionId, assistantMessage) |> ignore

                            // Extract and store facts from the conversation
                            TarsLlmCommand.extractAndStoreSessionFacts(chatSessionService, sessionId, input, response.Content)
                        else
                            let errorMsg = response.Error |> Option.defaultValue "Unknown error"
                            AnsiConsole.MarkupLine($"[red]❌ Error: {errorMsg}[/]")
                    | None ->
                        AnsiConsole.MarkupLine("[red]❌ Session expired or not found[/]")
                        continueLoop <- false

            return CommandResult.success($"Interactive session completed with {messageCount} messages")
        }

    static member handleSessionCommand(chatSessionService: ChatSessionService, sessionId: string, command: string) =
        match command.ToLowerInvariant() with
        | "/help" ->
            AnsiConsole.MarkupLine("[bold yellow]Session Commands:[/]")
            AnsiConsole.MarkupLine("  /help     - Show this help")
            AnsiConsole.MarkupLine("  /stats    - Show session statistics")
            AnsiConsole.MarkupLine("  /memory   - Show session memory")
            AnsiConsole.MarkupLine("  /history  - Show conversation history")
            AnsiConsole.MarkupLine("  /storage  - Show TARS data storage status")
            AnsiConsole.MarkupLine("  /save     - Save transcript to Markdown file")
            AnsiConsole.MarkupLine("  /persist  - Persist session memory to long-term learning")
            AnsiConsole.MarkupLine("  /clear    - Clear session memory")
            AnsiConsole.MarkupLine("  quit/exit/bye - End session")
        | "/stats" ->
            match chatSessionService.GetSessionStats(sessionId) with
            | Some stats ->
                AnsiConsole.MarkupLine($"[cyan]📊 Session Statistics:[/]")
                let durationStr = stats.Duration.ToString(@"hh\:mm\:ss")
                AnsiConsole.MarkupLine($"  Duration: {durationStr}")
                AnsiConsole.MarkupLine($"  Messages: {stats.MessageCount} (User: {stats.UserMessages}, Assistant: {stats.AssistantMessages})")
                AnsiConsole.MarkupLine($"  Memory Items: {stats.MemoryItems}")
                match stats.TotalTokens with
                | Some tokens -> AnsiConsole.MarkupLine($"  Total Tokens: {tokens}")
                | None -> ()
                match stats.AverageResponseTime with
                | Some avgTime -> AnsiConsole.MarkupLine($"  Avg Response Time: {avgTime.TotalMilliseconds:F0}ms")
                | None -> ()
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Could not get session statistics[/]")
        | "/memory" ->
            match chatSessionService.GetSession(sessionId) with
            | Some session ->
                AnsiConsole.MarkupLine("[cyan]🧠 Session Memory:[/]")
                if session.Memory.Facts.IsEmpty && session.Memory.UserPreferences.IsEmpty then
                    AnsiConsole.MarkupLine("  No memory items stored yet")
                else
                    if not session.Memory.Facts.IsEmpty then
                        AnsiConsole.MarkupLine("  [bold]Facts:[/]")
                        for (key, item) in session.Memory.Facts |> Map.toList do
                            AnsiConsole.MarkupLine($"    • {key}: {item.Value}")
                    if not session.Memory.UserPreferences.IsEmpty then
                        AnsiConsole.MarkupLine("  [bold]Preferences:[/]")
                        for (key, item) in session.Memory.UserPreferences |> Map.toList do
                            AnsiConsole.MarkupLine($"    • {key}: {item.Value}")
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Session not found[/]")
        | "/history" ->
            match chatSessionService.GetSession(sessionId) with
            | Some session ->
                AnsiConsole.MarkupLine("[cyan]📜 Conversation History:[/]")
                let recentMessages = session.Messages |> List.take (min 10 session.Messages.Length) |> List.rev
                for msg in recentMessages do
                    let roleColor = if msg.Role = User then "blue" else "green"
                    let roleText = if msg.Role = User then "You" else "TARS"
                    let contentPreview = msg.Content.Substring(0, min 100 msg.Content.Length)
                    let ellipsis = if msg.Content.Length > 100 then "..." else ""
                    AnsiConsole.MarkupLine($"[{roleColor}]{roleText}:[/] {contentPreview}{ellipsis}")
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Session not found[/]")
        | "/storage" ->
            // Display simplified session storage status
            match chatSessionService.GetSession(sessionId) with
            | Some session ->
                AnsiConsole.MarkupLine("[bold blue]📊 Session Storage Status[/]")
                AnsiConsole.WriteLine()

                let table = Table()
                table.AddColumn("[bold]Metric[/]") |> ignore
                table.AddColumn("[bold]Value[/]") |> ignore

                table.AddRow("[cyan]Session ID[/]", $"[green]{sessionId}[/]") |> ignore
                table.AddRow("[cyan]Messages[/]", $"[green]{session.Messages.Length}[/]") |> ignore
                table.AddRow("[cyan]Memory Facts[/]", $"[green]{session.Memory.Facts.Count}[/]") |> ignore
                table.AddRow("[cyan]User Preferences[/]", $"[green]{session.Memory.UserPreferences.Count}[/]") |> ignore
                table.AddRow("[cyan]Context Variables[/]", $"[green]{session.Memory.ContextVariables.Count}[/]") |> ignore

                let totalContentSize = session.Messages |> List.sumBy (fun m -> m.Content.Length)
                let estimatedSizeMB = float totalContentSize / (1024.0 * 1024.0)
                let estimatedTokens = totalContentSize / 4

                table.AddRow("[cyan]Content Size[/]", $"[green]{estimatedSizeMB:F2} MB[/]") |> ignore
                table.AddRow("[cyan]Estimated Tokens[/]", $"[green]{estimatedTokens:N0}[/]") |> ignore
                table.AddRow("[cyan]Model[/]", $"[green]{session.Model}[/]") |> ignore
                let startTimeStr = session.StartTime.ToString("yyyy-MM-dd HH:mm:ss")
                let lastActivityStr = session.LastActivity.ToString("yyyy-MM-dd HH:mm:ss")
                table.AddRow("[cyan]Start Time[/]", $"[green]{startTimeStr}[/]") |> ignore
                table.AddRow("[cyan]Last Activity[/]", $"[green]{lastActivityStr}[/]") |> ignore

                AnsiConsole.Write(table)
                AnsiConsole.WriteLine()
                AnsiConsole.MarkupLine("[dim]💡 Use 'tars storage-status' command for comprehensive TARS data storage details[/]")
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Session not found[/]")
        | "/save" ->
            match chatSessionService.SaveTranscript(sessionId) with
            | Ok fileName ->
                AnsiConsole.MarkupLine($"[green]💾 Transcript saved to: {fileName}[/]")
            | Error errorMsg ->
                AnsiConsole.MarkupLine($"[red]❌ Failed to save transcript: {errorMsg}[/]")
        | "/persist" ->
            task {
                let! result = chatSessionService.PersistSessionMemoryToLearning(sessionId)
                match result with
                | Ok message ->
                    AnsiConsole.MarkupLine($"[green]💾 {message}[/]")
                | Error errorMsg ->
                    AnsiConsole.MarkupLine($"[red]❌ Failed to persist memory: {errorMsg}[/]")
            } |> Async.AwaitTask |> Async.RunSynchronously
        | "/clear" ->
            chatSessionService.UpdateMemory(sessionId, fun memory ->
                { memory with Facts = Map.empty; UserPreferences = Map.empty; ContextVariables = Map.empty }
            ) |> ignore
            AnsiConsole.MarkupLine("[yellow]🧹 Session memory cleared[/]")
        | _ ->
            AnsiConsole.MarkupLine($"[red]❌ Unknown command: {command}[/]")
            AnsiConsole.MarkupLine("[dim]Type /help for available commands[/]")

    static member extractAndStoreSessionFacts(chatSessionService: ChatSessionService, sessionId: string, userInput: string, assistantResponse: string) =
        // Simple fact extraction - could be enhanced with NLP
        let userInputLower = userInput.ToLowerInvariant()

        // Extract user preferences
        if userInputLower.Contains("i prefer") || userInputLower.Contains("i like") then
            let preference = userInput.Substring(userInput.IndexOf("prefer") + 6).Trim()
            chatSessionService.AddPreference(sessionId, "user_preference", preference) |> ignore

        // Extract facts from assistant response
        if assistantResponse.Contains("Model Context Protocol") || assistantResponse.Contains("MCP") then
            chatSessionService.AddFact(sessionId, "mcp_knowledge", "User asked about Model Context Protocol", "conversation") |> ignore

