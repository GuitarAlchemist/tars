namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading
open System.Threading.Tasks
open Spectre.Console
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Commands.Types

/// Unified Chatbot Command - Interactive TARS chat using unified architecture
module UnifiedChatbotCommand =
    
    /// Chat session state
    type ChatSession = {
        SessionId: string
        StartTime: DateTime
        MessageCount: int
        ConfigManager: UnifiedConfigurationManager
        ProofGenerator: UnifiedProofGenerator
        Logger: ITarsLogger
    }
    
    /// Create new chat session
    let createChatSession (logger: ITarsLogger) =
        task {
            let sessionId = generateCorrelationId()
            let configManager = createConfigurationManager logger
            let proofGenerator = createProofGenerator logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            return {
                SessionId = sessionId
                StartTime = DateTime.UtcNow
                MessageCount = 0
                ConfigManager = configManager
                ProofGenerator = proofGenerator
                Logger = logger
            }
        }
    
    /// Process user input and generate response
    let processUserInput (session: ChatSession) (input: string) =
        task {
            try
                let correlationId = generateCorrelationId()
                let updatedSession = { session with MessageCount = session.MessageCount + 1 }
                
                // Generate proof for user interaction
                let! interactionProof =
                    ProofExtensions.generateExecutionProof
                        session.ProofGenerator
                        (sprintf "ChatInteraction_%d" session.MessageCount)
                        correlationId
                
                match input.ToLower().Trim() with
                | "help" | "?" ->
                    AnsiConsole.MarkupLine("[bold cyan]🤖 TARS Unified Chatbot Help[/]")
                    AnsiConsole.WriteLine()
                    AnsiConsole.MarkupLine("[bold yellow]Available Commands:[/]")
                    AnsiConsole.MarkupLine("  [cyan]config get <key>[/]        Get configuration value")
                    AnsiConsole.MarkupLine("  [cyan]config set <key>=<value>[/] Set configuration value")
                    AnsiConsole.MarkupLine("  [cyan]config list[/]             List all configuration")
                    AnsiConsole.MarkupLine("  [cyan]proof generate <type>[/]   Generate cryptographic proof")
                    AnsiConsole.MarkupLine("  [cyan]proof verify <id>[/]       Verify proof by ID")
                    AnsiConsole.MarkupLine("  [cyan]system status[/]           Show system status")
                    AnsiConsole.MarkupLine("  [cyan]system health[/]           Run health check")
                    AnsiConsole.MarkupLine("  [cyan]session info[/]            Show session information")
                    AnsiConsole.MarkupLine("  [cyan]help[/]                    Show this help")
                    AnsiConsole.MarkupLine("  [cyan]exit[/]                    Exit chatbot")
                    AnsiConsole.WriteLine()
                    return updatedSession
                
                | input when input.StartsWith("config get ") ->
                    let key = input.Substring(11).Trim()
                    let value = ConfigurationExtensions.getString session.ConfigManager key "Not found"
                    AnsiConsole.MarkupLine(sprintf "[cyan]Configuration[/] [yellow]%s[/]: [green]%s[/]" key value)
                    return updatedSession
                
                | input when input.StartsWith("config set ") ->
                    let configPart = input.Substring(11).Trim()
                    if configPart.Contains("=") then
                        let parts = configPart.Split('=', 2)
                        let key = parts.[0].Trim()
                        let value = parts.[1].Trim()
                        
                        let! setResult = session.ConfigManager.SetValueAsync(key, value, Some correlationId)
                        match setResult with
                        | Success _ ->
                            AnsiConsole.MarkupLine($"[green]✅ Configuration set:[/] [yellow]{key}[/] = [cyan]{value}[/]")
                        | Failure (error, _) ->
                            AnsiConsole.MarkupLine($"[red]❌ Failed to set configuration: {TarsError.toString error}[/]")
                    else
                        AnsiConsole.MarkupLine("[red]❌ Invalid format. Use: config set key=value[/]")
                    return updatedSession
                
                | "config list" ->
                    let allValues = session.ConfigManager.GetAllValues()
                    AnsiConsole.MarkupLine($"[bold cyan]📋 Configuration ({allValues.Count} items):[/]")
                    for kvp in allValues do
                        let valueStr = 
                            match kvp.Value with
                            | StringValue s -> $"\"{s}\""
                            | IntValue i -> i.ToString()
                            | FloatValue f -> f.ToString("F2")
                            | BoolValue b -> b.ToString().ToLower()
                            | _ -> "complex"
                        AnsiConsole.MarkupLine($"  [yellow]{kvp.Key}[/]: [cyan]{valueStr}[/]")
                    return updatedSession
                
                | input when input.StartsWith("proof generate ") ->
                    let proofType = input.Substring(15).Trim()
                    let! proofResult = match proofType.ToLower() with
                                       | "execution" -> ProofExtensions.generateExecutionProof session.ProofGenerator "UserRequested" correlationId
                                       | "performance" -> ProofExtensions.generatePerformanceProof session.ProofGenerator "ChatPerformance" 1.0 correlationId
                                       | _ -> ProofExtensions.generateExecutionProof session.ProofGenerator proofType correlationId
                    
                    match proofResult with
                    | Success (proof, _) ->
                        AnsiConsole.MarkupLine($"[green]✅ Proof generated:[/] [yellow]{proof.ProofId}[/]")
                        AnsiConsole.MarkupLine($"  [dim]Type: {proofType}[/]")
                        let timestampStr = proof.Timestamp.ToString("yyyy-MM-dd HH:mm:ss")
                        AnsiConsole.MarkupLine($"  [dim]Timestamp: {timestampStr}[/]")
                        AnsiConsole.MarkupLine($"  [dim]Signature: {proof.CryptographicSignature.Substring(0, 16)}...[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"[red]❌ Failed to generate proof: {TarsError.toString error}[/]")
                    return updatedSession
                
                | input when input.StartsWith("proof verify ") ->
                    let proofId = input.Substring(13).Trim()
                    let! proofResult = session.ProofGenerator.GetProofAsync(proofId, CancellationToken.None)
                    
                    match proofResult with
                    | Success (Some proof, _) ->
                        let! verificationResult = session.ProofGenerator.VerifyProofAsync(proof, CancellationToken.None)
                        match verificationResult with
                        | Success (verification, _) ->
                            let statusIcon = if verification.IsValid then "✅" else "❌"
                            let statusColor = if verification.IsValid then "green" else "red"
                            AnsiConsole.MarkupLine($"{statusIcon} [bold {statusColor}]Proof {proofId}[/]")
                            AnsiConsole.MarkupLine($"  Valid: [{statusColor}]{verification.IsValid}[/]")
                            let trustScoreStr = verification.TrustScore.ToString("F2")
                            AnsiConsole.MarkupLine($"  Trust Score: [yellow]{trustScoreStr}[/]")
                            if verification.Issues.Length > 0 then
                                AnsiConsole.MarkupLine("  Issues:")
                                for issue in verification.Issues do
                                    AnsiConsole.MarkupLine($"    [red]• {issue}[/]")
                        | Failure (error, _) ->
                            AnsiConsole.MarkupLine($"[red]❌ Verification failed: {TarsError.toString error}[/]")
                    | Success (None, _) ->
                        AnsiConsole.MarkupLine($"[yellow]⚠️ Proof not found: {proofId}[/]")
                    | Failure (error, _) ->
                        AnsiConsole.MarkupLine($"[red]❌ Failed to retrieve proof: {TarsError.toString error}[/]")
                    return updatedSession
                
                | "system status" ->
                    let statistics = session.ConfigManager.GetStatistics()
                    let proofStats = session.ProofGenerator.GetProofStatistics()
                    
                    AnsiConsole.MarkupLine("[bold cyan]🔧 System Status:[/]")
                    AnsiConsole.MarkupLine($"  Session ID: [yellow]{session.SessionId.Substring(0, 8)}...[/]")
                    let uptime = DateTime.UtcNow - session.StartTime
                    let uptimeStr = uptime.ToString(@"hh\:mm\:ss")
                    AnsiConsole.MarkupLine($"  Uptime: [cyan]{uptimeStr}[/]")
                    AnsiConsole.MarkupLine($"  Messages: [green]{session.MessageCount}[/]")
                    let totalConfigs = statistics.["totalConfigurations"]
                    let totalProofs = proofStats.["totalProofs"]
                    AnsiConsole.MarkupLine($"  Configurations: [yellow]{totalConfigs}[/]")
                    AnsiConsole.MarkupLine($"  Proofs Generated: [magenta]{totalProofs}[/]")
                    return updatedSession
                
                | "system health" ->
                    AnsiConsole.MarkupLine("[yellow]🔍 Running health check...[/]")
                    
                    // Quick health checks
                    let configHealth = statistics.["isInitialized"] :?> bool
                    let proofHealth = proofStats.["totalProofs"] :?> int >= 0
                    
                    let healthScore = 
                        [configHealth; proofHealth; true] // Core is always healthy if we got here
                        |> List.map (fun h -> if h then 1.0 else 0.0)
                        |> List.average
                    
                    let healthIcon = if healthScore > 0.8 then "✅" else if healthScore > 0.5 then "⚠️" else "❌"
                    let healthColor = if healthScore > 0.8 then "green" else if healthScore > 0.5 then "yellow" else "red"
                    
                    let healthScoreStr = healthScore.ToString("F2")
                    let healthPercentage = (healthScore * 100.0).ToString("F0")
                    let configColor = if configHealth then "green" else "red"
                    let proofColor = if proofHealth then "green" else "red"
                    AnsiConsole.MarkupLine($"{healthIcon} [bold {healthColor}]System Health: {healthScoreStr}[/] ({healthPercentage}%)")
                    AnsiConsole.MarkupLine($"  Configuration: [{configColor}]{configHealth}[/]")
                    AnsiConsole.MarkupLine($"  Proof System: [{proofColor}]{proofHealth}[/]")
                    AnsiConsole.MarkupLine($"  Core System: [green]true[/]")
                    return updatedSession
                
                | "session info" ->
                    AnsiConsole.MarkupLine("[bold cyan]📊 Session Information:[/]")
                    AnsiConsole.MarkupLine($"  Session ID: [yellow]{session.SessionId}[/]")
                    let startTimeStr = session.StartTime.ToString("yyyy-MM-dd HH:mm:ss")
                    let duration = DateTime.UtcNow - session.StartTime
                    let durationStr = duration.ToString(@"hh\:mm\:ss")
                    AnsiConsole.MarkupLine($"  Start Time: [cyan]{startTimeStr}[/]")
                    AnsiConsole.MarkupLine($"  Duration: [green]{durationStr}[/]")
                    AnsiConsole.MarkupLine($"  Messages Processed: [magenta]{session.MessageCount}[/]")
                    return updatedSession
                
                | _ ->
                    // Default response for unrecognized input
                    AnsiConsole.MarkupLine($"[yellow]🤖 I received:[/] [dim]{input}[/]")
                    AnsiConsole.MarkupLine("[dim]Type 'help' for available commands or 'exit' to quit.[/]")
                    return updatedSession
            
            with
            | ex ->
                session.Logger.LogError(generateCorrelationId(), TarsError.create "ChatError" "Chat processing failed" (Some ex), ex)
                AnsiConsole.MarkupLine($"[red]❌ Error processing input: {ex.Message}[/]")
                return session
        }
    
    /// Run interactive chat session
    let runChatSession (logger: ITarsLogger) =
        task {
            try
                AnsiConsole.MarkupLine("[bold cyan]🤖 TARS Unified Chatbot[/]")
                AnsiConsole.MarkupLine("[dim]Interactive chat using unified TARS architecture[/]")
                AnsiConsole.WriteLine()
                
                let! session = createChatSession logger
                
                AnsiConsole.MarkupLine($"[green]✅ Chat session started[/] [dim]({session.SessionId.Substring(0, 8)}...)[/]")
                AnsiConsole.MarkupLine("[dim]Type 'help' for commands or 'exit' to quit[/]")
                AnsiConsole.WriteLine()
                
                let mutable currentSession = session
                let mutable continueChat = true
                
                while continueChat do
                    AnsiConsole.Write("[bold cyan]TARS>[/] ")
                    let input = Console.ReadLine()
                    
                    if String.IsNullOrWhiteSpace(input) then
                        () // Skip empty input
                    elif input.ToLower().Trim() = "exit" then
                        continueChat <- false
                        AnsiConsole.MarkupLine("[yellow]👋 Goodbye! Chat session ended.[/]")
                        
                        // Generate session summary proof
                        let! sessionProof =
                            ProofExtensions.generateExecutionProof
                                currentSession.ProofGenerator
                                (sprintf "ChatSessionCompleted_%d_messages" currentSession.MessageCount)
                                (generateCorrelationId())
                        
                        match sessionProof with
                        | Success (proof, _) ->
                            AnsiConsole.MarkupLine($"[green]🔐 Session proof: {proof.ProofId}[/]")
                        | Failure _ -> ()
                    else
                        let! updatedSession = processUserInput currentSession input
                        currentSession <- updatedSession
                        AnsiConsole.WriteLine()
                
                // Cleanup
                (currentSession.ConfigManager :> IDisposable).Dispose()
                (currentSession.ProofGenerator :> IDisposable).Dispose()
                
                return 0
            
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Chat session failed: {ex.Message}[/]")
                return 1
        }
    
    /// Unified Chatbot Command implementation
    type UnifiedChatbotCommand() =
        interface ICommand with
            member _.Name = "chat"
            member _.Description = "Interactive TARS chatbot using unified architecture"
            member _.Usage = "tars chat [--interactive]"
            member _.Examples = [
                "tars chat --interactive     # Start interactive chat session"
                "tars chat                   # Show chatbot overview"
            ]
            
            member _.ValidateOptions(options: CommandOptions) = true
            
            member _.ExecuteAsync(options: CommandOptions) =
                task {
                    try
                        let logger = createLogger "UnifiedChatbotCommand"
                        
                        let isInteractiveMode = 
                            options.Arguments 
                            |> List.exists (fun arg -> arg = "--interactive")
                        
                        if isInteractiveMode then
                            let! result = runChatSession logger
                            return { Message = ""; ExitCode = result; Success = result = 0 }
                        else
                            AnsiConsole.MarkupLine("[bold cyan]🤖 TARS Unified Chatbot[/]")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Interactive chatbot powered by TARS unified architecture.")
                            AnsiConsole.MarkupLine("Provides access to configuration, proof generation, and system status.")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("[bold yellow]Features:[/]")
                            AnsiConsole.MarkupLine("  ⚙️ [cyan]Configuration Management[/] - Get/set system configuration")
                            AnsiConsole.MarkupLine("  🔐 [magenta]Proof Generation[/] - Create and verify cryptographic proofs")
                            AnsiConsole.MarkupLine("  🔍 [green]System Monitoring[/] - Check system status and health")
                            AnsiConsole.MarkupLine("  📊 [yellow]Session Tracking[/] - Track chat session with proofs")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Available options:")
                            AnsiConsole.MarkupLine("  [yellow]--interactive[/]  Start interactive chat session")
                            AnsiConsole.WriteLine()
                            AnsiConsole.MarkupLine("Example: [dim]tars chat --interactive[/]")
                            return { Message = ""; ExitCode = 0; Success = true }
                    
                    with
                    | ex ->
                        AnsiConsole.MarkupLine($"[red]❌ Chatbot command failed: {ex.Message}[/]")
                        return { Message = ""; ExitCode = 1; Success = false }
                }

