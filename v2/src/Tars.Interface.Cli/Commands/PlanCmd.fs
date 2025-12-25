module Tars.Interface.Cli.Commands.PlanCmd

open System
open System.Threading.Tasks
open Spectre.Console
open Tars.Core
open Tars.Knowledge
open Tars.Interface.Cli

type PlanOptions =
    { Command: string
      Goal: string option
      PlanId: string option
      UsePostgres: bool }

let defaultOptions =
    { Command = "help"
      Goal = None
      PlanId = None
      UsePostgres = false }

let parseArgs (args: string array) =
    let mutable options = defaultOptions
    let mutable i = 0

    if args.Length > 0 then
        options <- { options with Command = args.[0] }
        i <- 1

    while i < args.Length do
        match args.[i] with
        | "--pg"
        | "--postgres" ->
            options <- { options with UsePostgres = true }
            i <- i + 1
        | arg when not (arg.StartsWith "-") ->
            match options.Command with
            | "new" ->
                if options.Goal.IsNone then
                    options <- { options with Goal = Some arg }
            | "list" -> ()
            | "show" ->
                if options.PlanId.IsNone then
                    options <- { options with PlanId = Some arg }
            | _ -> ()

            i <- i + 1
        | _ -> i <- i + 1

    options

let runNew (manager: PlanManager) (goal: string) =
    task {
        // Create a simple plan for now. In reality, we might ask LLM to decompose goal.
        let steps =
            [ { Order = 1
                Description = "Analyze goal"
                EstimatedEffort = None
                Dependencies = []
                Status = StepStatus.NotStarted
                CompletedAt = None
                Notes = [] } ]

        // Find assumptions? For now empty.
        let assumptions = []

        let! result = manager.CreatePlan(goal, steps, assumptions, AgentId.User)

        match result with
        | Result.Ok plan -> AnsiConsole.MarkupLine($"[green]✓ Plan created:[/] {plan.Id} - {plan.Goal}")
        | Result.Error e -> AnsiConsole.MarkupLine($"[red]✗ Failed to create plan:[/] {e}")
    }

let runList (manager: PlanManager) =
    task {
        let! plans = manager.GetActive()

        let table = Table()
        table.AddColumn("ID") |> ignore
        table.AddColumn("Goal") |> ignore
        table.AddColumn("Status") |> ignore
        table.AddColumn("Step") |> ignore

        for plan in plans do
            let currentStep =
                plan.Steps
                |> List.tryFind (fun s ->
                    match s.Status with
                    | StepStatus.InProgress -> true
                    | _ -> false)
                |> Option.map (fun s -> s.Description)
                |> Option.defaultValue "-"

            table.AddRow(
                plan.Id.ToString(),
                Markup.Escape(plan.Goal),
                plan.Status.ToString(),
                Markup.Escape(currentStep)
            )
            |> ignore

        AnsiConsole.Write(table)
    }

let runShow (manager: PlanManager) (planIdStr: string) =
    task {
        // Parse ID (handling "p:" prefix or GUID)
        let idStr =
            if planIdStr.StartsWith("p:") then
                planIdStr.Substring(2)
            else
                planIdStr

        match Guid.TryParse(idStr) with
        | true, g ->
            let planId = PlanId(g)

            match! manager.Get(planId) with
            | Some plan ->
                AnsiConsole.MarkupLine($"[bold blue]Plan:[/] {plan.Id}")
                AnsiConsole.MarkupLine($"[bold]Goal:[/] {Markup.Escape(plan.Goal)}")
                AnsiConsole.MarkupLine($"[bold]Status:[/] {plan.Status}")
                AnsiConsole.MarkupLine("\n[bold]Steps:[/]")

                let table = Table()
                table.AddColumn("Order") |> ignore
                table.AddColumn("Description") |> ignore
                table.AddColumn("Status") |> ignore

                for step in plan.Steps do
                    let color =
                        match step.Status with
                        | StepStatus.Completed -> "green"
                        | StepStatus.InProgress -> "yellow"
                        | StepStatus.Failed _ -> "red"
                        | _ -> "grey"

                    table.AddRow(step.Order.ToString(), Markup.Escape(step.Description), $"[{color}]{step.Status}[/]")
                    |> ignore

                AnsiConsole.Write(table)

            | None -> AnsiConsole.MarkupLine($"[red]Plan {planId} not found[/]")
        | _ -> AnsiConsole.MarkupLine($"[red]Invalid Plan ID format[/]")
    }

let run (config: TarsConfig) (options: PlanOptions) =
    task {
        let ledger = KnowledgeLedger.createInMemory () // TODO: Support Postgres via options.UsePostgres
        // Initialize ledger (load from DB if needed)
        do! ledger.Initialize()

        // Create manager
        let manager = PlanManager.createInMemory (ledger)

        match options.Command.ToLowerInvariant() with
        | "new" ->
            match options.Goal with
            | Some g -> do! runNew manager g
            | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars plan new <goal>[/]")
        | "list" -> do! runList manager
        | "show" ->
            match options.PlanId with
            | Some id -> do! runShow manager id
            | None -> AnsiConsole.MarkupLine("[yellow]Usage: tars plan show <id>[/]")
        | _ -> AnsiConsole.MarkupLine("[yellow]Unknown plan command. Use: new, list, show[/]")

        return 0
    }
