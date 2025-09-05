namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Cli.Core

/// Command to directly teach TARS new knowledge
type TeachCommand(learningMemoryService: LearningMemoryService, logger: ILogger<TeachCommand>) =

    interface ICommand with
        member _.Name = "teach"
        member _.Description = "Directly teach TARS new knowledge"
        member _.Usage = "tars teach [topic] [content]"
        member _.Examples = [
            "tars teach \"F# basics\" \"F# is a functional programming language with immutable data\""
            "tars teach \"Pattern matching\" \"match expression allows pattern-based control flow\""
        ]

        member _.ValidateOptions(options: CommandOptions) = true

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                match options.Arguments with
                | topic :: content ->
                    let fullContent = String.concat " " content
                    return! TeachCommand.teachKnowledge(learningMemoryService, logger, topic, fullContent)
                | [] ->
                    AnsiConsole.MarkupLine("[red]❌ Error: Please provide a topic and content to teach[/]")
                    AnsiConsole.MarkupLine("Usage: tars teach [topic] [content]")
                    return CommandResult.failure("Missing arguments")
            }
    
    static member teachKnowledge(learningMemoryService: LearningMemoryService, logger: ILogger<TeachCommand>, topic: string, content: string) =
        task {
            AnsiConsole.MarkupLine(sprintf "[bold cyan]📚 Teaching TARS about: %s[/]" topic)
            AnsiConsole.WriteLine()

            // Store the knowledge directly
            let! storeResult =
                learningMemoryService.StoreKnowledge(
                    topic,
                    content,
                    LearningSource.UserInteraction("direct-teaching"),
                    None)
                |> Async.StartAsTask

            match storeResult with
            | Ok knowledgeId ->
                AnsiConsole.MarkupLine("[green]✅ Knowledge stored successfully![/]")
                AnsiConsole.MarkupLine(sprintf "[bold yellow]🧠 TARS has learned about '%s'![/]" topic)
                return CommandResult.success("Knowledge stored successfully")
            | Error error ->
                AnsiConsole.MarkupLine(sprintf "[red]❌ Failed to store knowledge: %s[/]" error)
                return CommandResult.failure(error)
        }
