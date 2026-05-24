namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Core.UnifiedCache
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Acceleration.UnifiedCudaEngine
open TarsEngine.FSharp.Cli.AI.UnifiedLLMEngine
open TarsEngine.FSharp.Cli.Commands.Types

/// Unified AI Chat Command - Intelligent chat using local LLM with unified architecture
module UnifiedAIChatCommand =
    
    /// AI chat session state
    type AIChatSession = {
        SessionId: string
        StartTime: DateTime
        MessageCount: int
        LLMEngine: UnifiedLLMEngine
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
        ConversationHistory: (string * string) list // (user, assistant) pairs
    }
    
    /// Create AI chat session
    let createAIChatSession (logger: ITarsLogger) =
        task {
            try
                let sessionId = generateCorrelationId()
                let configManager = createConfigurationManager logger
                let proofGenerator = createProofGenerator logger
                let cacheManager = new UnifiedCacheManager(logger, configManager, proofGenerator)
                
                let! _ = configManager.InitializeAsync(CancellationToken.None)
                
                // Try to create CUDA engine (optional)
                let cudaEngine = 
                    try
                        Some (createCudaEngine logger)
                    with
                    | _ -> None
                
                let llmEngine = new UnifiedLLMEngine(logger, configManager, proofGenerator, cacheManager, cudaEngine)
                
                // Check if LLM is available
                let! availability = llmEngine.IsAvailableAsync()
                match availability with
                | Success (true, _) ->
                    logger.LogInformation(sessionId, "LLM engine initialized successfully")
                | Success (false, _) ->
                    logger.LogWarning(sessionId, "LLM engine not available - check Ollama installation")
                | Failure (error, _) ->
                    logger.LogError(sessionId, error, Exception("LLM initialization failed"))
                
                return Success ({
                    SessionId = sessionId
                    StartTime = DateTime.UtcNow
                    MessageCount = 0
                    LLMEngine = llmEngine
                    ConfigManager = configManager
                    ProofGenerator = proofGenerator
                    Logger = logger
                    ConversationHistory = []
                }, Map [("sessionId", box sessionId)])
            
            with
            | ex ->
                let error = ExecutionError ($"Failed to create AI chat session: {ex.Message}", Some ex)
                return Failure (error, generateCorrelationId())
        }
    
    /// Generate system prompt for TARS
    let generateSystemPrompt (session: AIChatSession) : string =
        $"""You are TARS, an advanced AI system with a unified architecture. You have the following capabilities:

CORE SYSTEMS:
- Unified Core Foundation with error handling and correlation tracking
- Unified Configuration Management with centralized settings
- Unified Proof Generation with cryptographic evidence
- Unified Caching System with multi-level caching
- Unified Monitoring System with real-time health tracking
- Unified CUDA Engine with GPU acceleration
- Unified Agent Coordination with intelligent orchestration

PERSONALITY:
- Professional but approachable
- Technically accurate and detailed
- Helpful and solution-oriented
- Honest about limitations
- Focused on practical value

CONTEXT:
- Session ID: {session.SessionId.Substring(0, 8)}...
- Messages in conversation: {session.MessageCount}
- Session duration: {DateTime.UtcNow - session.StartTime.ToString(@"hh\:mm\:ss")}
- All operations generate cryptographic proofs for verification

Respond helpfully and accurately. If asked about your capabilities, reference your unified architecture systems."""
    
    /// Process AI chat input
    let processAIChatInput (session: AIChatSession) (input: string) =
        task {
            try
                let correlationId = generateCorrelationId()
                let updatedSession = { session with MessageCount = session.MessageCount + 1 }
                
                session.Logger.LogInformation(correlationId, $"Processing AI chat input: {input.Substring(0, Math.Min(50, input.Length))}...")
                
                match input.ToLower().Trim() with
                | "help" | "?" ->
                    AnsiConsole.MarkupLine("[bold cyan]🤖 TARS AI Chat Help[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold yellow]AI Commands:[/]")
                    AnsiConsole.MarkupLine("  [cyan]Ask anything[/]           Natural language conversation")
                    AnsiConsole.MarkupLine("  [cyan]explain <topic>[/]        Get detailed explanations")
                    AnsiConsole.MarkupLine("  [cyan]analyze <data>[/]         Analyze data or problems")
                    AnsiConsole.MarkupLine("  [cyan]generate <type>[/]        Generate code, text, or ideas")
                    AnsiConsole.MarkupLine("  [cyan]help[/]                   Show this help")
                    AnsiConsole.MarkupLine("  [cyan]status[/]                 Show AI system status")
                    AnsiConsole.MarkupLine("  [cyan]models[/]                 List available models")
                    AnsiConsole.MarkupLine("  [cyan]metrics[/]                Show AI performance metrics")
                    AnsiConsole.MarkupLine("  [cyan]history[/]                Show conversation history")
                    AnsiConsole.MarkupLine("  [cyan]clear[/]                  Clear conversation history")
                    AnsiConsole.MarkupLine("  [cyan]exit[/]                   Exit AI chat")
                    AnsiConsole.WriteLine()
                    return updatedSession
                
                | "status" ->
                    AnsiConsole.MarkupLine("[bold cyan]🤖 AI System Status[/]")
                    
                    let! availability = session.LLMEngine.IsAvailableAsync()
                    match availability with
                    | Success (true, metadata) ->
                        AnsiConsole.MarkupLine("  LLM Engine: [green]✅ Available[/]")
                        if metadata.ContainsKey("endpoint") then
                            AnsiConsole.MarkupLine(sprintf "  Endpoint: [cyan]%s[/]" (metadata.["endpoint"].ToString()))
                    | Success (false, _) ->
                        AnsiConsole.MarkupLine("  LLM Engine: [red]❌ Not Available[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine(sprintf "  LLM Engine: [red]❌ Error: %s[/]" (TarsError.toString error))
                    
                    let capabilities = session.LLMEngine.GetCapabilities()
                    AnsiConsole.MarkupLine("  Capabilities:")
                    for capability in capabilities |> List.take (Math.Min(3, capabilities.Length)) do
                        AnsiConsole.MarkupLine(sprintf "    • [dim]%s[/]" capability)
                    
                    return updatedSession
                
                | "models" ->
                    AnsiConsole.MarkupLine("[bold cyan]🤖 Available Models[/]")
                    
                    let! modelsResult = session.LLMEngine.GetModelsAsync()
                    match modelsResult with
                    | Success (models, _) ->
                        for model in models do
                            let statusIcon = if model.IsLoaded then "✅" else "⏳"
                            AnsiConsole.MarkupLine(sprintf "  %s [yellow]%s[/] ([cyan]%s[/])" statusIcon model.Name model.Size)
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine(sprintf "  [red]❌ Failed to fetch models: %s[/]" (TarsError.toString error))
                    
                    return updatedSession
                
                | "metrics" ->
                    AnsiConsole.MarkupLine("[bold cyan]📊 AI Performance Metrics[/]")
                    
                    let metrics = session.LLMEngine.GetMetrics()
                    AnsiConsole.MarkupLine(sprintf "  Total Requests: [green]%d[/]" metrics.TotalRequests)
                    AnsiConsole.MarkupLine(sprintf "  Successful: [green]%d[/]" metrics.SuccessfulRequests)
                    AnsiConsole.MarkupLine(sprintf "  Failed: [red]%d[/]" metrics.FailedRequests)
                    AnsiConsole.MarkupLine(sprintf "  Average Response Time: [yellow]%sms[/]" (metrics.AverageInferenceTime.TotalMilliseconds.ToString("F0")))
                    AnsiConsole.MarkupLine(sprintf "  Average Tokens/sec: [cyan]%s[/]" (metrics.AverageTokensPerSecond.ToString("F1")))
                    AnsiConsole.MarkupLine(sprintf "  Cache Hit Ratio: [magenta]%s[/]" (metrics.CacheHitRatio.ToString("P1")))
                    AnsiConsole.MarkupLine(sprintf "  Total Tokens Generated: [blue]%s[/]" (metrics.TotalTokensGenerated.ToString("N0")))
                    
                    return updatedSession
                
                | "history" ->
                    AnsiConsole.MarkupLine("[bold cyan]💬 Conversation History[/]")
                    
                    if session.ConversationHistory.IsEmpty then
                        AnsiConsole.MarkupLine("  [dim]No conversation history yet[/]")
                    else
                        for i, (user, assistant) in session.ConversationHistory |> List.rev |> List.indexed do
                            AnsiConsole.MarkupLine(sprintf "  [bold yellow]%d. User:[/] %s..." (i + 1) (user.Substring(0, Math.Min(50, user.Length))))
                            AnsiConsole.MarkupLine(sprintf "     [bold cyan]TARS:[/] %s..." (assistant.Substring(0, Math.Min(50, assistant.Length))))
                            if i < 2 then AnsiConsole.WriteLine() // Show spacing for recent messages
                    
                    return updatedSession
                
                | "clear" ->
                    AnsiConsole.MarkupLine("[yellow]🧹 Conversation history cleared[/]")
                    return { updatedSession with ConversationHistory = [] }
                
                | _ ->
                    // AI inference for general conversation
                    AnsiConsole.MarkupLine("[dim]🤖 TARS is thinking...[/]")
                    
                    let systemPrompt = generateSystemPrompt session
                    let! inferenceResult = session.LLMEngine.InferAsync(input, systemPrompt, 0.7, 1024)
                    
                    match inferenceResult with
                    | Success (response, metadata) ->
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold cyan]🤖 TARS:[/]")
                        AnsiConsole.MarkupLine(sprintf "[white]%s[/]" response.Response)
                        AnsiConsole.WriteLine()
                        
                        // Show inference details
                        let cacheIcon = if response.CacheHit then "💾" else "🧠"
                        let proofIcon = if response.ProofId.IsSome then "🔐" else ""
                        AnsiConsole.MarkupLine(sprintf "[dim]%s %d tokens • %sms • %s tok/s %s[/]" cacheIcon response.TokensGenerated (response.InferenceTime.TotalMilliseconds.ToString("F0")) (response.TokensPerSecond.ToString("F1")) proofIcon)
                        
                        if response.ProofId.IsSome then
                            AnsiConsole.MarkupLine(sprintf "[dim]🔐 Proof: %s...[/]" (response.ProofId.Value.Substring(0, 8)))
                        
                        // Update conversation history
                        let newHistory = (input, response.Response) :: session.ConversationHistory
                        let trimmedHistory = newHistory |> List.take (Math.Min(10, newHistory.Length)) // Keep last 10 exchanges
                        
                        return { updatedSession with ConversationHistory = trimmedHistory }
                    
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine(sprintf "[red]❌ AI inference failed: %s[/]" (TarsError.toString error))
                        AnsiConsole.MarkupLine("[dim]💡 Try: Check if Ollama is running with 'ollama serve'[/]")
                        return updatedSession
            
            with
            | ex ->
                session.Logger.LogError(generateCorrelationId(), TarsError.create "AIChatError" "AI chat processing failed" (Some ex), ex)
                AnsiConsole.MarkupLine(sprintf "[red]❌ Error processing input: %s[/]" ex.Message)
                return session
        }
    
    /// Run AI chat session
    let runAIChatSession (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🤖 TARS AI Chat[/]")
                AnsiConsole.MarkupLine("[dim]Intelligent conversation powered by local LLM and unified architecture[/]")
                AnsiConsole.WriteLine()
                
                let! sessionResult = createAIChatSession logger
                
                match sessionResult with
                | Success (session, _) ->
                    AnsiConsole.MarkupLine(sprintf "[green]✅ AI chat session started[/] [dim](%s...)[/]" (session.SessionId.Substring(0, 8)))
                    AnsiConsole.MarkupLine("[dim]Type 'help' for commands or just chat naturally. Type 'exit' to quit.[/]")
                    AnsiConsole.WriteLine()
                    
                    let mutable currentSession = session
                    let mutable continueChat = true
                    
                    while continueChat do
                        AnsiConsole.Write("[bold cyan]You>[/] ")
                        let input = Console.ReadLine()
                        
                        if String.IsNullOrWhiteSpace(input) then
                            () // Skip empty input
                        elif input.ToLower().Trim() = "exit" then
                            continueChat <- false
                            AnsiConsole.MarkupLine("[yellow]👋 Goodbye! AI chat session ended.[/]")
                            
                            // Generate session summary proof
                            let! sessionProof =
                                ProofExtensions.generateExecutionProof
                                    currentSession.ProofGenerator
                                    (sprintf "AIChatSessionCompleted_%d_messages" currentSession.MessageCount)
                                    (generateCorrelationId())
                            
                            match sessionProof with
                            | Success (proof, _) ->
                                AnsiConsole.MarkupLine(sprintf "[green]🔐 Session proof: %s...[/]" (proof.ProofId.Substring(0, 8)))
                            | Failure _ -> ()
                        else
                            let! updatedSession = processAIChatInput currentSession input
                            currentSession <- updatedSession
                            AnsiConsole.WriteLine()
                    
                    // Cleanup
                    currentSession.ConfigManager.Dispose()
                    currentSession.ProofGenerator.Dispose()
                    (currentSession.LLMEngine :> IDisposable).Dispose()
                    
                    return 0
                
                | Failure (error, _) ->
                    AnsiConsole.MarkupLine(sprintf "[red]❌ Failed to start AI chat session: %s[/]" (TarsError.toString error))
                    AnsiConsole.MarkupLine("[dim]💡 Make sure Ollama is installed and running: 'ollama serve'[/]")
                    return 1
            
            with
            | ex ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ AI chat session failed: %s[/]" ex.Message)
                return 1
        }
    
    /// Unified AI Chat Command implementation
    type UnifiedAIChatCommand() =
        interface ICommand with
            member _.Name = "ai"
            member _.Description = "Intelligent AI chat using local LLM with unified architecture"
            member _.Usage = "tars ai [--chat] [--status] [--models]"
            member _.Examples = [
                "tars ai --chat              # Start interactive AI chat"
                "tars ai --status            # Show AI system status"
                "tars ai --models            # List available models"
                "tars ai                     # Show AI overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedAIChatCommand"
                        
                        let isChatMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--chat")
                        
                        let isStatusMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--status")
                        
                        let isModelsMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--models")
                        
                        if isChatMode then
                            let! result = runAIChatSession logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        elif isStatusMode then
                            // Quick status check
                            let configManager = createConfigurationManager logger
                            let proofGenerator = createProofGenerator logger
                            let cacheManager = new UnifiedCacheManager(logger, configManager, proofGenerator)
                            let llmEngine = new UnifiedLLMEngine(logger, configManager, proofGenerator, cacheManager, None)
                            
                            let! _ = configManager.InitializeAsync(CancellationToken.None)
                            let! availability = llmEngine.IsAvailableAsync()
                            
                            AnsiConsole.MarkupLine("[bold cyan]🤖 TARS AI System Status[/]")
                            match availability with
                            | Success (true, _) ->
                                AnsiConsole.MarkupLine("  Status: [green]✅ Available and ready[/]")
                                let metrics = llmEngine.GetMetrics()
                                AnsiConsole.MarkupLine(sprintf "  Total Requests: [yellow]%d[/]" metrics.TotalRequests)
                                AnsiConsole.MarkupLine(sprintf "  Success Rate: [green]%s%%[/]" (if metrics.TotalRequests > 0L then (float metrics.SuccessfulRequests / float metrics.TotalRequests * 100.0).ToString("F1") else "0.0"))
                            | Success (false, _) ->
                                AnsiConsole.MarkupLine("  Status: [red]❌ Not available[/]")
                                AnsiConsole.MarkupLine("  [dim]💡 Start Ollama with: ollama serve[/]")
                            | Failure (error, _) ->
                                AnsiConsole.MarkupLine(sprintf "  Status: [red]❌ Error: %s[/]" (TarsError.toString error))
                            
                            configManager.Dispose()
                            proofGenerator.Dispose()
                            (llmEngine :> IDisposable).Dispose()
                            return { Message = ""; ExitCode = 0; Success = true }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🤖 TARS AI System[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Intelligent AI chat powered by local LLM and TARS unified architecture.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Features:[/]")
                            AnsiConsole.MarkupLine("  🧠 [cyan]Local LLM Integration[/] - Private AI with Ollama")
                            AnsiConsole.MarkupLine("  ⚡ [green]CUDA Acceleration[/] - GPU-optimized inference")
                            AnsiConsole.MarkupLine("  💾 [blue]Intelligent Caching[/] - Fast response caching")
                            AnsiConsole.MarkupLine("  🔐 [magenta]Proof Generation[/] - Cryptographic evidence for AI operations")
                            AnsiConsole.MarkupLine("  📊 [yellow]Performance Monitoring[/] - Real-time AI metrics")
                            AnsiConsole.MarkupLine("  🎯 [red]Context Awareness[/] - Unified architecture integration")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--chat[/]     Start interactive AI chat")
                            AnsiConsole.MarkupLine("  [yellow]--status[/]   Show AI system status")
                            AnsiConsole.MarkupLine("  [yellow]--models[/]   List available models")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Example: [dim]tars ai --chat[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[dim]💡 Requires Ollama to be installed and running[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine(sprintf "[red]❌ AI command failed: %s[/]" ex.Message)
                        return { Message = ""; ExitCode = 1; Success = false }
                }

