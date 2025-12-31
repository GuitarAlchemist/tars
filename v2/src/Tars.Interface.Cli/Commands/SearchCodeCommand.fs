namespace Tars.Interface.Cli.Commands

open System
open System.Threading.Tasks
open Spectre.Console
open Tars.Tools.Augment

module SearchCodeCommand =

    type SearchCodeOptions =
        { Query: string option
          Workspace: string option
          ConfirmCost: bool }

    let defaultOptions =
        { Query = None
          Workspace = None
          ConfirmCost = false }

    let parseArgs (args: string array) =
        let mutable options = defaultOptions
        let mutable i = 0

        while i < args.Length do
            match args.[i] with
            | "--query"
            | "-q" when i + 1 < args.Length ->
                options <-
                    { options with
                        Query = Some args.[i + 1] }

                i <- i + 2
            | "--workspace"
            | "-w" when i + 1 < args.Length ->
                options <-
                    { options with
                        Workspace = Some args.[i + 1] }

                i <- i + 2
            | "--confirm-cost"
            | "--yes" ->
                options <- { options with ConfirmCost = true }
                i <- i + 1
            | arg when not (arg.StartsWith("-")) && options.Query.IsNone ->
                // Implicit query for first arg
                options <- { options with Query = Some arg }
                i <- i + 1
            | _ -> i <- i + 1

        options

    let run (rawArgs: string array) =
        task {
            // rawArgs might contain "search-code" as the first element if passed directly from main,
            // but Program.fs usually passes the slice. We'll support both.
            let args =
                if rawArgs.Length > 0 && rawArgs.[0] = "search-code" then
                    rawArgs |> Array.skip 1
                else
                    rawArgs

            let opts = parseArgs args

            match opts.Query with
            | None ->
                AnsiConsole.MarkupLine("[red]Error: Query is required.[/]")
                AnsiConsole.MarkupLine("Usage: tars search-code <query> [-w workspace] [--confirm-cost]")
                return 1
            | Some query ->
                // Cost Warning
                // Cost Warning check
                let shouldProceed =
                    if opts.ConfirmCost then
                        true
                    else
                        AnsiConsole.MarkupLine(
                            "[yellow]⚠️  WARNING: This command uses the Augment Code API which incurs costs.[/]"
                        )

                        AnsiConsole.Confirm("Do you want to proceed?")

                if not shouldProceed then
                    AnsiConsole.MarkupLine("[dim]Aborted.[/]")
                    return 0
                else
                    let workspace = opts.Workspace |> Option.defaultValue Environment.CurrentDirectory


                    AnsiConsole.MarkupLine($"[bold blue]Searching codebase for:[/] {query}")
                    AnsiConsole.MarkupLine($"[dim]Workspace: {workspace}[/]")

                    // Construct JSON args for the tool
                    let jsonArgs =
                        $$"""{ "query": "{{query}}", "workspace": "{{workspace.Replace("\\", "\\\\")}}" }"""

                    let! result = AugmentTools.codebaseSearch jsonArgs

                    AnsiConsole.MarkupLine("\n[bold green]Result:[/]")
                    AnsiConsole.MarkupLine(Markup.Escape(result))

                    // Disconnect to clean up
                    let! _ = AugmentTools.disconnect ""

                    return 0
        }
