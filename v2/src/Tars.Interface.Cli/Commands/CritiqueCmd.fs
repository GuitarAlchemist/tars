namespace Tars.Interface.Cli.Commands

open System
open Spectre.Console
open Tars.Core
open Tars.Interface.Cli

module CritiqueCmd =

    type CritiqueOptions = {
        Command: string
        Category: CritiqueCategory option
        Severity: CritiqueSeverity option
        Summary: string option
        Details: string option
        Id: string option
    }

    let defaultOptions = {
        Command = "list"
        Category = None
        Severity = None
        Summary = None
        Details = None
        Id = None
    }

    let private getService() =
        let tarsHome = 
            Environment.GetEnvironmentVariable("TARS_HOME") 
            |> Option.ofObj 
            |> Option.defaultValue (IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars"))
        let path = IO.Path.Combine(tarsHome, "critiques.json")
        CritiqueService(path)

    let run (args: string array) =
        task {
            let mutable options = defaultOptions
            if args.Length > 0 then
                options <- { options with Command = args.[0].ToLowerInvariant() }
            
            let mutable i = 1
            while i < args.Length do
                match args.[i] with
                | "--category" | "-c" when i + 1 < args.Length ->
                    let cat = 
                        match args.[i+1].ToLowerInvariant() with
                        | "arch" | "architectural" -> ArchCritique
                        | "ops" | "operational" -> OpsCritique
                        | "cog" | "cognitive" -> CogCritique
                        | "sec" | "security" -> SecCritique
                        | _ -> OtherCritique
                    options <- { options with Category = Some cat }
                    i <- i + 2
                | "--severity" | "-s" when i + 1 < args.Length ->
                    let sev = 
                        match args.[i+1].ToLowerInvariant() with
                        | "crit" | "critical" -> CriticalCrit
                        | "high" -> HighCrit
                        | "med" | "medium" -> MediumCrit
                        | _ -> LowCrit
                    options <- { options with Severity = Some sev }
                    i <- i + 2
                | "--summary" when i + 1 < args.Length ->
                    options <- { options with Summary = Some args.[i+1] }
                    i <- i + 2
                | "--details" when i + 1 < args.Length ->
                    options <- { options with Details = Some args.[i+1] }
                    i <- i + 2
                | arg when not (arg.StartsWith("-")) && options.Id.IsNone ->
                    options <- { options with Id = Some arg }
                    i <- i + 1
                | _ -> i <- i + 1

            let service = getService()

            match options.Command with
            | "add" ->
                match options.Category, options.Severity, options.Summary with
                | Some cat, Some sev, Some sum ->
                    let details = options.Details |> Option.defaultValue ""
                    let c = service.AddCritique(cat, sev, sum, details)
                    AnsiConsole.MarkupLine(sprintf "[green]✓ Added critique:[/] %s - %s" (Markup.Escape(c.Id.ToString())) (Markup.Escape(c.Summary)))
                | _ -> 
                    AnsiConsole.MarkupLine("[yellow]Usage: tars critique add --category <cat> --severity <sev> --summary <sum> [--details <det>][/]")
            
            | "list" ->
                let critiques = service.GetAll()
                if critiques.IsEmpty then
                    AnsiConsole.MarkupLine("[grey]No critiques found.[/]")
                else
                    let table = Table().Border(TableBorder.Rounded)
                    table.AddColumn("ID") |> ignore
                    table.AddColumn("Category") |> ignore
                    table.AddColumn("Severity") |> ignore
                    table.AddColumn("Summary") |> ignore
                    table.AddColumn("Status") |> ignore
                    
                    for c in critiques |> List.sortByDescending (fun c -> c.CreatedAt) do
                        let color = 
                            match c.Severity with
                            | CriticalCrit -> "red"
                            | HighCrit -> "orange1"
                            | MediumCrit -> "yellow"
                            | LowCrit -> "blue"
                        
                        let statusColor = 
                            match c.Status with
                            | Open -> "red"
                            | Mitigating -> "yellow"
                            | Resolved -> "green"
                            | WontFix -> "grey"

                        table.AddRow(
                            c.Id.ToString().Substring(0, 8),
                            c.Category.ToString(),
                            sprintf "[%s]%A[/]" color c.Severity,
                            Markup.Escape(c.Summary),
                            sprintf "[%s]%A[/]" statusColor c.Status
                        ) |> ignore
                    
                    AnsiConsole.Write(table)

            | "show" ->
                match options.Id with
                | Some idStr ->
                    let critiques = service.GetAll()
                    let found = critiques |> List.tryFind (fun c -> c.Id.ToString().StartsWith(idStr, StringComparison.OrdinalIgnoreCase))
                    match found with
                    | Some c ->
                        AnsiConsole.MarkupLine(sprintf "[bold blue]Critique:[/] %s" (c.Id.ToString()))
                        AnsiConsole.MarkupLine(sprintf "[bold]Category:[/] %A" c.Category)
                        AnsiConsole.MarkupLine(sprintf "[bold]Severity:[/] %A" c.Severity)
                        AnsiConsole.MarkupLine(sprintf "[bold]Summary:[/] %s" (Markup.Escape(c.Summary)))
                        AnsiConsole.MarkupLine(sprintf "[bold]Status:[/] %A" c.Status)
                        AnsiConsole.MarkupLine(sprintf "[bold]Details:[/] %s" (Markup.Escape(c.Details)))
                        match c.MitigationPlan with
                        | Some p -> AnsiConsole.MarkupLine(sprintf "[bold]Mitigation:[/] %s" (Markup.Escape(p)))
                        | None -> ()
                    | None -> AnsiConsole.MarkupLine(sprintf "[red]Critique %s not found.[/]" (Markup.Escape(idStr)))
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars critique show <id>[/]")

            | "resolve" ->
                match options.Id with
                | Some idStr ->
                    let critiques = service.GetAll()
                    let found = critiques |> List.tryFind (fun c -> c.Id.ToString().StartsWith(idStr, StringComparison.OrdinalIgnoreCase))
                    match found with
                    | Some c ->
                        service.Resolve(c.Id)
                        AnsiConsole.MarkupLine(sprintf "[green]✓ Resolved critique:[/] %s" (c.Id.ToString()))
                    | None -> AnsiConsole.MarkupLine(sprintf "[red]Critique %s not found.[/]" (Markup.Escape(idStr)))
                | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars critique resolve <id>[/]")

            | _ ->
                printfn "TARS Self-Critique System"
                printfn "Usage:"
                printfn "  tars critique list                 List all critiques"
                printfn "  tars critique show <id>            Show critique details"
                printfn "  tars critique add [options]        Add a new critique"
                printfn "  tars critique resolve <id>         Mark a critique as resolved"
                printfn ""
                printfn "Options:"
                printfn "  --category, -c <arch|ops|cog|sec>  Critique category"
                printfn "  --severity, -s <crit|high|med|low> Critique severity"
                printfn """  --summary "..."                    Short summary"""
                printfn """  --details "..."                    Detailed description"""

            return 0
        }