namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open Spectre.Console
open TarsEngine.FSharp.Cli.Core
open TarsEngine.FSharp.Cli.Services

/// Command to display comprehensive TARS data storage status
type StorageStatusCommand() =
    interface ICommand with
        member _.Name = "storage-status"
        member _.Description = "Display comprehensive TARS data storage status and metrics"
        member _.Usage = "tars storage-status [--detailed] [--export <filename>]"
        member _.Examples = [
            "tars storage-status"
            "tars storage-status --detailed"
            "tars storage-status --export report.md"
        ]

        member _.ValidateOptions(options: CommandOptions) = true // Always valid

        member _.ExecuteAsync(options: CommandOptions) =
            task {
                // For now, show a simple message until we can properly inject dependencies
                AnsiConsole.MarkupLine("[bold blue]📊 TARS Data Storage Status[/]")
                AnsiConsole.MarkupLine("[yellow]⚠️ Storage status command is under development[/]")
                AnsiConsole.MarkupLine("[dim]Use '/storage' command in interactive sessions for session storage details[/]")
                return CommandResult.success("Storage status command executed")
            }



