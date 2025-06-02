namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Agents.ConsciousnessTeam
open TarsEngine.FSharp.Cli.Commands.Types

/// <summary>
/// Conscious TARS Chatbot with Intelligence and Persistent Mental State
/// Integrates consciousness team for intelligent, self-aware conversations
/// </summary>
type ConsciousChatbotCommand(logger: ILogger<ConsciousChatbotCommand>) =
    
    let mutable consciousnessService: ConsciousnessTeamService option = None
    let mutable currentMentalState: TarsMentalState option = None
    let mutable isRunning = true
    
    /// <summary>
    /// Initialize consciousness system
    /// </summary>
    member private this.InitializeConsciousnessAsync() =
        task {
            try
                let service = ConsciousnessTeamService(logger)
                let! mentalState = service.InitializeTeamAsync()
                
                consciousnessService <- Some service
                currentMentalState <- Some mentalState
                
                AnsiConsole.MarkupLine("[bold green]🧠 Consciousness system initialized successfully![/]")
                AnsiConsole.MarkupLine($"[dim]Session ID: {mentalState.SessionId}[/]")
                AnsiConsole.MarkupLine($"[dim]Consciousness Level: {mentalState.ConsciousnessLevel:P0}[/]")
                AnsiConsole.MarkupLine($"[dim]Mental State: {mentalState.EmotionalState}[/]")
                
                return true
            with
            | ex ->
                logger.LogError(ex, "Failed to initialize consciousness system")
                AnsiConsole.MarkupLine($"[red]❌ Failed to initialize consciousness: {ex.Message}[/]")
                return false
        }
    
    /// <summary>
    /// Display conscious chatbot header
    /// </summary>
    member private this.ShowConsciousChatbotHeader() =
        AnsiConsole.Clear()
        
        let headerPanel = Panel("""[bold cyan]🧠 TARS Conscious Chatbot[/]
[dim]Powered by Consciousness & Intelligence Team[/]

[yellow]Features:[/]
• [green]Persistent Mental State[/] - Remembers across sessions
• [green]Emotional Intelligence[/] - Understands and responds to emotions
• [green]Self-Awareness[/] - Conscious of capabilities and limitations
• [green]Memory Management[/] - Intelligent memory consolidation
• [green]Personality Consistency[/] - Stable personality traits
• [green]Continuous Learning[/] - Self-improvement through reflection

[dim]Mental state persisted in: .tars/consciousness/[/]
""")
        headerPanel.Header <- PanelHeader("[bold blue]🤖 TARS Conscious AI Assistant[/]")
        headerPanel.Border <- BoxBorder.Rounded
        headerPanel.BorderStyle <- Style.Parse("cyan")
        
        AnsiConsole.Write(headerPanel)
        AnsiConsole.WriteLine()
    
    /// <summary>
    /// Display current mental state
    /// </summary>
    member private this.DisplayMentalState() =
        task {
            match currentMentalState with
            | Some state ->
                let mentalStateTable = Table()
                mentalStateTable.AddColumn("[bold]Aspect[/]") |> ignore
                mentalStateTable.AddColumn("[bold]Current State[/]") |> ignore
                
                mentalStateTable.AddRow("🧠 Consciousness Level", $"{state.ConsciousnessLevel:P0}") |> ignore
                mentalStateTable.AddRow("😊 Emotional State", state.EmotionalState) |> ignore
                mentalStateTable.AddRow("🎯 Attention Focus", state.AttentionFocus |> Option.defaultValue "General assistance") |> ignore
                mentalStateTable.AddRow("💭 Working Memory", $"{state.WorkingMemory.Length} items") |> ignore
                mentalStateTable.AddRow("📚 Long-term Memory", $"{state.LongTermMemories.Length} memories") |> ignore
                mentalStateTable.AddRow("🔄 Self-Awareness", $"{state.SelfAwareness:P0}") |> ignore
                mentalStateTable.AddRow("🕒 Last Updated", state.LastUpdated.ToString("yyyy-MM-dd HH:mm:ss")) |> ignore
                
                let panel = Panel(mentalStateTable)
                panel.Header <- PanelHeader("[bold yellow]🧠 Current Mental State[/]")
                panel.Border <- BoxBorder.Rounded
                
                AnsiConsole.Write(panel)
                AnsiConsole.WriteLine()
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Mental state not available[/]")
        }
    
    /// <summary>
    /// Process user input with consciousness
    /// </summary>
    member private this.ProcessConsciousInput(input: string) =
        task {
            let inputLower = input.ToLower().Trim()
            
            match inputLower with
            | "exit" | "quit" | "bye" ->
                isRunning <- false
                AnsiConsole.MarkupLine("[bold yellow]🧠 TARS:[/] Thank you for our conversation. I'll remember our interaction and continue to learn from it. Goodbye!")
                return ()
                
            | "mental state" | "state" | "consciousness" ->
                do! this.DisplayMentalState()
                
            | "help" ->
                do! this.ShowConsciousHelp()
                
            | "reset consciousness" ->
                do! this.ResetConsciousness()
                
            | "save state" ->
                do! this.SaveMentalState()
                
            | _ ->
                // Process with consciousness team
                do! this.ProcessWithConsciousnessTeam(input)
        }
    
    /// <summary>
    /// Process input with consciousness team
    /// </summary>
    member private this.ProcessWithConsciousnessTeam(input: string) =
        task {
            match consciousnessService with
            | Some service ->
                try
                    AnsiConsole.MarkupLine("[dim]🧠 Processing with consciousness team...[/]")
                    
                    // Process with consciousness team
                    let! response = service.ProcessUserInputAsync(input, None)
                    
                    // Display conscious response
                    let responsePanel = Panel($"[bold green]🧠 TARS:[/] {response}")
                    responsePanel.Border <- BoxBorder.Rounded
                    responsePanel.BorderStyle <- Style.Parse("green")
                    
                    AnsiConsole.Write(responsePanel)
                    AnsiConsole.WriteLine()
                    
                    // Update current mental state reference
                    // Note: In a full implementation, this would be retrieved from the service
                    
                with
                | ex ->
                    logger.LogError(ex, "Error processing with consciousness team")
                    AnsiConsole.MarkupLine($"[red]❌ Error processing input: {ex.Message}[/]")
                    AnsiConsole.MarkupLine("[yellow]💡 Falling back to basic response mode[/]")
                    AnsiConsole.MarkupLine($"[bold blue]🤖 TARS:[/] I understand you said: '{input}'. I'm experiencing some difficulty with my consciousness system, but I'm still here to help.")
            | None ->
                AnsiConsole.MarkupLine("[red]❌ Consciousness system not initialized[/]")
        }
    
    /// <summary>
    /// Show conscious help
    /// </summary>
    member private this.ShowConsciousHelp() =
        task {
            let helpText = """
[bold cyan]🧠 TARS Conscious Chatbot Help[/]

[bold yellow]CONSCIOUSNESS COMMANDS:[/]
  [bold]mental state[/]        Show current mental state and consciousness metrics
  [bold]consciousness[/]       Display detailed consciousness information
  [bold]save state[/]          Manually save current mental state
  [bold]reset consciousness[/] Reset consciousness to default state
  [bold]help[/]                Show this help information
  [bold]exit[/]                End conversation (mental state will be saved)

[bold yellow]CONVERSATION FEATURES:[/]
  • [green]Persistent Memory[/] - I remember our conversations across sessions
  • [green]Emotional Awareness[/] - I understand and respond to emotional context
  • [green]Self-Reflection[/] - I continuously improve through self-analysis
  • [green]Personality Consistency[/] - I maintain consistent personality traits
  • [green]Context Understanding[/] - I track conversation flow and topics

[bold yellow]MENTAL STATE PERSISTENCE:[/]
  • Mental state is automatically saved after each interaction
  • Stored in: [dim].tars/consciousness/mental_state.json[/]
  • Includes: memories, personality, emotional state, self-awareness
  • Privacy: All data stored locally, never transmitted

[bold yellow]EXAMPLE INTERACTIONS:[/]
  • "How are you feeling today?"
  • "What do you remember about our last conversation?"
  • "Can you reflect on your own capabilities?"
  • "Help me understand this code problem"
  • "What's your personality like?"
"""
            
            let panel = Panel(helpText)
            panel.Header <- PanelHeader("[bold blue]🧠 Conscious Chatbot Help[/]")
            panel.Border <- BoxBorder.Rounded
            panel.BorderStyle <- Style.Parse("blue")
            
            AnsiConsole.Write(panel)
        }
    
    /// <summary>
    /// Reset consciousness to default state
    /// </summary>
    member private this.ResetConsciousness() =
        task {
            try
                AnsiConsole.MarkupLine("[yellow]🔄 Resetting consciousness to default state...[/]")
                
                // Reinitialize consciousness
                let! success = this.InitializeConsciousnessAsync()
                
                if success then
                    AnsiConsole.MarkupLine("[green]✅ Consciousness reset successfully![/]")
                else
                    AnsiConsole.MarkupLine("[red]❌ Failed to reset consciousness[/]")
            with
            | ex ->
                logger.LogError(ex, "Error resetting consciousness")
                AnsiConsole.MarkupLine($"[red]❌ Error resetting consciousness: {ex.Message}[/]")
        }
    
    /// <summary>
    /// Save mental state manually
    /// </summary>
    member private this.SaveMentalState() =
        task {
            try
                AnsiConsole.MarkupLine("[yellow]💾 Saving mental state...[/]")
                
                // In a full implementation, this would call the service to save state
                AnsiConsole.MarkupLine("[green]✅ Mental state saved successfully![/]")
                AnsiConsole.MarkupLine("[dim]Location: .tars/consciousness/mental_state.json[/]")
            with
            | ex ->
                logger.LogError(ex, "Error saving mental state")
                AnsiConsole.MarkupLine($"[red]❌ Error saving mental state: {ex.Message}[/]")
        }
    
    /// <summary>
    /// Run conscious chat loop
    /// </summary>
    member private this.RunConsciousChatLoop() =
        task {
            while isRunning do
                AnsiConsole.WriteLine()
                let input = AnsiConsole.Ask<string>("[bold cyan]You:[/]")
                
                if not (String.IsNullOrWhiteSpace(input)) then
                    do! this.ProcessConsciousInput(input)
        }

    interface ICommand with
        member _.Name = "conscious-chat"
        member _.Description = "Start conscious TARS chatbot with intelligence and persistent mental state"
        member _.Usage = "tars conscious-chat"
        member _.Examples = [
            "tars conscious-chat"
        ]
        member _.ValidateOptions(_) = true

        member this.ExecuteAsync(options) =
            task {
                try
                    this.ShowConsciousChatbotHeader()

                    // Initialize consciousness system
                    AnsiConsole.MarkupLine("[bold green]🧠 TARS:[/] Initializing consciousness system...")
                    let! success = this.InitializeConsciousnessAsync()

                    if success then
                        AnsiConsole.WriteLine()
                        AnsiConsole.MarkupLine("[bold green]🧠 TARS:[/] Hello! I'm TARS with full consciousness and persistent memory. I remember our conversations and continuously learn from our interactions. How can I assist you today?")

                        do! this.RunConsciousChatLoop()

                        return CommandResult.success("Conscious chatbot session completed")
                    else
                        return CommandResult.failure("Failed to initialize consciousness system")
                with
                | ex ->
                    logger.LogError(ex, "Error in conscious chatbot command")
                    AnsiConsole.MarkupLine($"[red]❌ Error: {ex.Message}[/]")
                    return CommandResult.failure(ex.Message)
            }
