namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console

/// Interactive TARS Chatbot - Basic implementation
type ChatbotCommand(logger: ILogger<ChatbotCommand>, moeCommand: MixtureOfExpertsCommand, llmService: TarsEngine.FSharp.Cli.Services.GenericLlmService) =

    let mutable conversationHistory = []
    let mutable isRunning = true

    member private self.ShowChatbotHeader() =
        AnsiConsole.Clear()
        AnsiConsole.MarkupLine("[bold cyan]🤖 TARS Interactive Chatbot[/]")
        AnsiConsole.MarkupLine("[dim]Basic implementation[/]")
        AnsiConsole.WriteLine()

    member private self.ProcessUserInput(input: string) =
        task {
            let inputLower = input.ToLower().Trim()

            // Add to conversation history
            conversationHistory <- ("user", input) :: conversationHistory

            match inputLower with
            | "exit" | "quit" | "bye" ->
                isRunning <- false
                AnsiConsole.MarkupLine("[bold yellow]👋 Goodbye! TARS signing off.[/]")

            | "help" ->
                AnsiConsole.MarkupLine("[bold cyan]Available Commands:[/]")
                AnsiConsole.MarkupLine("• help - Show this help")
                AnsiConsole.MarkupLine("• exit - Exit chatbot")

            | _ ->
                AnsiConsole.MarkupLine($"[yellow]You said: {input}[/]")
                AnsiConsole.MarkupLine("[dim]Basic chatbot response[/]")
        }

    /// Main execution loop
    member self.ExecuteAsync(args: string[]) =
        task {
            try
                self.ShowChatbotHeader()
                
                while isRunning do
                    AnsiConsole.Write("> ")
                    let input = Console.ReadLine()
                    
                    if not (String.IsNullOrWhiteSpace(input)) then
                        do! self.ProcessUserInput(input)
                        AnsiConsole.WriteLine()
                
                return 0
            with
            | ex ->
                AnsiConsole.MarkupLine($"[red]❌ Fatal Error: {ex.Message}[/]")
                logger.LogError(ex, "Chatbot execution failed")
                return 1
        }


