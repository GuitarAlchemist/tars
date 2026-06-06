namespace Tars.Interface.Cli.Commands

open System
open System.IO
open Spectre.Console
open Tars.Core

module KgCmd =

    type KgOptions =
        { Command: string
          RunId: string option }

    let parseArgs (args: string list) =
        let mutable options = { Command = "help"; RunId = None }
        match args with
        | "trace" :: rest ->
            options <- { options with Command = "trace" }
            match rest with
            | id :: _ -> options <- { options with RunId = Some id }
            | [] -> ()
        | _ -> ()
        options

    let private printTrace (graph: Tars.Core.TemporalKnowledgeGraph.TemporalGraph) (runIdStr: string) =
        // Find Run Node
        let allNodes = graph.GetAllNodes()
        let runNode = 
            allNodes 
            |> List.tryPick (fun entity -> 
                match entity with
                | TarsEntity.RunE r when r.Id.ToString() = runIdStr || r.Id.ToString().StartsWith(runIdStr) -> Some r
                | _ -> None
            )

        match runNode with
        | Some run ->
            AnsiConsole.MarkupLine($"[bold blue]Trace for Run:[/] {run.Id}")
            AnsiConsole.MarkupLine($"[dim]Goal:[/] {Markup.Escape(run.Goal)}")
            AnsiConsole.MarkupLine($"[dim]Pattern:[/] {run.Pattern}")
            AnsiConsole.MarkupLine($"[dim]Time:[/] {run.Timestamp}")
            AnsiConsole.WriteLine()

            // Find Steps
            let steps = 
                allNodes 
                |> List.choose (fun entity ->
                    match entity with
                    | TarsEntity.StepE s when s.RunId = run.Id -> Some s
                    | _ -> None
                )
                |> List.sortBy (fun s -> s.Timestamp)

            if steps.IsEmpty then
                 AnsiConsole.MarkupLine("[yellow]No steps found for this run.[/]")
            else
                 let table = Table()
                 table.AddColumn("Step") |> ignore
                 table.AddColumn("Type") |> ignore
                 table.AddColumn("Content") |> ignore
                 table.Border <- TableBorder.Rounded
                 
                 for step in steps do
                     let content = 
                         if step.Content.Length > 80 then step.Content.Substring(0, 80) + "..." 
                         else step.Content
                     table.AddRow(step.StepId, step.NodeType, Markup.Escape(content)) |> ignore
                 
                 AnsiConsole.Write(table)
        | None ->
            AnsiConsole.MarkupLine($"[red]Run not found matches:[/] {runIdStr}")
            
    let private listRuns (graph: Tars.Core.TemporalKnowledgeGraph.TemporalGraph) =
        let runs = 
            graph.GetAllNodes()
            |> List.choose (fun entity ->
                match entity with
                | TarsEntity.RunE r -> Some r
                | _ -> None
            )
            |> List.sortByDescending (fun r -> r.Timestamp)
            
        if runs.IsEmpty then
            AnsiConsole.MarkupLine("[yellow]No runs recorded in Knowledge Graph.[/]")
        else
            AnsiConsole.MarkupLine($"[blue]Recorded Runs ({runs.Length}):[/]")
            let table = Table()
            table.AddColumn("Run ID") |> ignore
            table.AddColumn("Goal") |> ignore
            table.AddColumn("Pattern") |> ignore
            table.AddColumn("Time") |> ignore
            
            for r in runs do
                let goal = 
                    if r.Goal.Length > 50 then r.Goal.Substring(0, 50) + "..."
                    else r.Goal
                table.AddRow(r.Id.ToString(), Markup.Escape(goal), r.Pattern, r.Timestamp.ToString("HH:mm:ss")) |> ignore
                
            AnsiConsole.Write(table)

    let run (config: TarsConfig) (options: KgOptions) =
        task {
            // Path to graph - match AgentHelpers.fs
            let storagePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars", "knowledge", "graph")
            if not (Directory.Exists(Path.GetDirectoryName(storagePath))) then
                 Directory.CreateDirectory(Path.GetDirectoryName(storagePath)) |> ignore
                 
            let kg = InternalGraphService(storagePath)
            
            // Access internal graph for direct query
            let internalGraph = kg.Graph 
            
            match options.Command with
            | "trace" ->
                match options.RunId with
                | Some id -> printTrace internalGraph id
                | None -> listRuns internalGraph
            | _ -> 
                AnsiConsole.MarkupLine("Usage: tars kg trace [run_id]")
            
            return 0
        }
