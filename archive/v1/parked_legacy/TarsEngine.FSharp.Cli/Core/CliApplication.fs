namespace TarsEngine.FSharp.Cli.Core

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open Spectre.Console
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Commands.CommandOptions
open TarsEngine.FSharp.Cli.Commands.CommandResult

/// <summary>
/// Simple CLI application for basic functionality.
/// </summary>
type CliApplication() =

    // Set up minimal dependency injection
    let serviceProvider =
        let services = ServiceCollection()

        // Add logging
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
        ) |> ignore

        // Add our working commands
        services.AddTransient<QACommand>() |> ignore
        services.AddTransient<AutoImprovementDashboardCommand>() |> ignore

        services.BuildServiceProvider()

    let logger = serviceProvider.GetRequiredService<ILogger<CliApplication>>()

    // Simple command registry
    let getCommand (name: string) : TarsEngine.FSharp.Cli.Commands.ICommand option =
        match name with
        | "qa" ->
            try
                let cmd = serviceProvider.GetRequiredService<QACommand>()
                Some (cmd :> TarsEngine.FSharp.Cli.Commands.ICommand)
            with
            | _ -> None
        | "auto-ui"
        | "auto-dashboard"
        | "auto-improve" ->
            try
                let cmd = serviceProvider.GetRequiredService<AutoImprovementDashboardCommand>()
                Some (cmd :> TarsEngine.FSharp.Cli.Commands.ICommand)
            with
            | _ -> None
        | "help" | _ ->
            let helpCommand = {
                new TarsEngine.FSharp.Cli.Commands.ICommand with
                    member _.Name = "help"
                    member _.Description = "Show help information"
                    member _.Usage = "tars help"
                    member _.Examples = ["tars help"]
                    member _.ValidateOptions(options) = true
                    member _.ExecuteAsync options =
                        task {
                            printfn "TARS CLI - Basic Commands Available:"
                            printfn "  qa    - Quality assurance commands"
                            printfn "  auto-ui - Auto-improvement governance dashboard"
                            printfn "  help  - Show this help"
                            return success "Help displayed"
                        }
            }
            Some helpCommand

    let commandNames = ["qa"; "auto-ui"; "help"]
    
    /// <summary>
    /// Runs the CLI application with basic commands.
    /// </summary>
    member _.RunAsync(args: string[]) =
        Task.Run(fun () ->
            try
                if args.Length = 0 then
                    // Show available commands
                    Console.WriteLine("🚀 TARS CLI - Available Commands:")
                    Console.WriteLine("================================")
                    for name in commandNames do
                        Console.WriteLine($"  {name}")
                    Console.WriteLine()
                    Console.WriteLine("Usage: tars <command> [options]")
                    Console.WriteLine("Example: tars qa help")
                    0
                else
                    let commandName = args.[0]
                    let commandArgs = args.[1..]

                    // Simple command execution
                    match getCommand commandName with
                    | Some command ->
                        try
                            let options = { CommandOptions.createDefault() with Arguments = Array.toList commandArgs }
                            let result = command.ExecuteAsync(options).Result
                            if result.Success then 0 else 1
                        with
                        | ex ->
                            Console.WriteLine($"Error executing command: {ex.Message}")
                            1
                    | None ->
                        Console.WriteLine($"Unknown command: {commandName}")
                        Console.WriteLine("Use 'tars help' to see available commands")
                        1
            with
            | ex ->
                Console.WriteLine($"💥 Error: {ex.Message}")
                1
        )
