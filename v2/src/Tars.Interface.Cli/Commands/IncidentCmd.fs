module Tars.Interface.Cli.Commands.IncidentCmd

open System
open Tars.Core
open Tars.Core.Errors
open Spectre.Console

let private getLedger () = FailurePipeline.createDefault()

let listIncidents (status: string option) (limit: int option) =
    task {
        let ledger = getLedger()
        
        // Parse status
        let statusFilter =
            match status with
            | Some s -> 
                match s.ToLowerInvariant() with
                | "new" -> Some IncidentStatus.New
                | "triaging" -> Some IncidentStatus.Triaging
                | "ignored" -> Some IncidentStatus.Ignored
                | "confirmed" -> Some IncidentStatus.Confirmed
                | "inprogress" -> Some IncidentStatus.InProgress
                | "resolved" -> Some IncidentStatus.Resolved
                | "closed" -> Some IncidentStatus.Closed
                | _ ->
                    printfn $"⚠ Unknown status '%s{s}', ignoring filter."
                    None
            | None -> None
            
        let! incidents = ledger.ListAsync(statusFilter, limit)
        
        let table = Table()
        table.AddColumn("ID") |> ignore
        table.AddColumn("Status") |> ignore
        table.AddColumn("Severity") |> ignore
        table.AddColumn("Category") |> ignore
        table.AddColumn("Message") |> ignore
        table.AddColumn("Created") |> ignore
        
        for i in incidents do
            let statusColor = 
                match i.Status with
                | IncidentStatus.New -> "red"
                | IncidentStatus.Triaging -> "yellow"
                | IncidentStatus.Resolved -> "green"
                | IncidentStatus.Ignored -> "grey"
                | _ -> "white"
                
            let sevColor =
                match i.Error.Severity with
                | Tars.Core.ErrorSeverity.Critical -> "bold red"
                | Tars.Core.ErrorSeverity.High -> "red"
                | Tars.Core.ErrorSeverity.Medium -> "yellow"
                | Tars.Core.ErrorSeverity.Low -> "blue"
                
            table.AddRow(
                i.Id.ToString().Substring(0, 8),
                $"[{statusColor}]{i.Status}[/]",
                $"[{sevColor}]{i.Error.Severity}[/]",
                i.Error.Category.ToString().Split(' ').[0],
                Markup.Escape(message i.Error),
                i.CreatedAt.ToLocalTime().ToString("g")
            ) |> ignore
            
        AnsiConsole.Write(table) |> ignore
        AnsiConsole.MarkupLine($"[grey]Total: {incidents.Length}[/]") |> ignore
        return 0
    }

let showIncident (id: string) =
    task {
        let ledger = getLedger()
        let guid = Guid.Parse(id)
        
        match! ledger.GetAsync(guid) with
        | Some i ->
            AnsiConsole.Write(new Rule($"Incident {i.Id}")) |> ignore
            AnsiConsole.WriteLine() |> ignore
            
            let grid = Grid()
            grid.AddColumn() |> ignore
            grid.AddColumn() |> ignore
            
            grid.AddRow("Status:", i.Status.ToString()) |> ignore
            grid.AddRow("Severity:", i.Error.Severity.ToString()) |> ignore
            grid.AddRow("Created:", i.CreatedAt.ToString()) |> ignore
            grid.AddRow("Message:", message i.Error) |> ignore
            
            AnsiConsole.Write(grid) |> ignore
            
            if i.TriageNotes.IsSome then
                AnsiConsole.WriteLine() |> ignore
                AnsiConsole.MarkupLine("[bold]Triage Notes:[/]") |> ignore
                AnsiConsole.WriteLine(i.TriageNotes.Value) |> ignore
                
            if i.Resolution.IsSome then
                AnsiConsole.WriteLine() |> ignore
                AnsiConsole.MarkupLine("[bold green]Resolution:[/]") |> ignore
                AnsiConsole.WriteLine(i.Resolution.Value) |> ignore
                
            if not i.Error.Context.IsEmpty then
                AnsiConsole.WriteLine() |> ignore
                AnsiConsole.MarkupLine("[bold]Context:[/]") |> ignore
                for pair in i.Error.Context do
                    AnsiConsole.MarkupLine($"  {pair.Key}: {pair.Value}") |> ignore
                    
            return 0
        | None ->
            AnsiConsole.MarkupLine($"[red]Incident {id} not found.[/]") |> ignore
            return 1
    }

let reportIncident (msg: string) (severity: string) =
    task {
        let ledger = getLedger()
        
        let sev = 
            match severity.ToLowerInvariant() with
            | "critical" -> Tars.Core.ErrorSeverity.Critical
            | "high" -> Tars.Core.ErrorSeverity.High
            | "low" -> Tars.Core.ErrorSeverity.Low
            | _ -> Tars.Core.ErrorSeverity.Medium
            
        let error = 
            create (InternalError msg) sev false
            |> withContext "ReportedBy" "User"
            
        let! incident = ledger.ReportAsync(error)
        
        AnsiConsole.MarkupLine($"[green]Incident reported: {incident.Id}[/]") |> ignore
        return 0
    }

let updateStatus (id: string) (statusMsg: string) =
    task {
        let ledger = getLedger()
        let guid = Guid.Parse(id) 
        
        let newStatus = 
            match statusMsg.ToLowerInvariant() with
            | "new" -> Some IncidentStatus.New
            | "triaging" -> Some IncidentStatus.Triaging
            | "ignored" -> Some IncidentStatus.Ignored
            | "confirmed" -> Some IncidentStatus.Confirmed
            | "inprogress" -> Some IncidentStatus.InProgress
            | "resolved" -> Some IncidentStatus.Resolved
            | "closed" -> Some IncidentStatus.Closed
            | _ -> None

        match! ledger.GetAsync(guid) with
        | Some incident ->
            match newStatus with
            | Some st ->
                let updated = { incident with Status = st; UpdatedAt = DateTime.UtcNow }
                do! ledger.UpdateAsync(updated)
                AnsiConsole.MarkupLine($"[green]Updated incident {id} to {st}[/]") |> ignore
                return 0
            | None ->
                AnsiConsole.MarkupLine($"[red]Invalid status: {statusMsg}[/]") |> ignore
                return 1
        | None ->
            AnsiConsole.MarkupLine($"[red]Incident not found[/]") |> ignore
            return 1
    }

let run (args: string list) =
    task {
        match args with
        | "list" :: rest ->
            let status = rest |> List.tryFind (fun s -> not (s.StartsWith("-")))
            return! listIncidents status (Some 20)
            
        | "show" :: id :: _ ->
             return! showIncident id
             
        | "report" :: msg :: rest ->
            let sev = rest |> List.tryHead |> Option.defaultValue "Medium"
            return! reportIncident msg sev

        | "status" :: id :: statusMsg :: _ ->
             return! updateStatus id statusMsg
            
        | _ ->
            AnsiConsole.MarkupLine("Usage:") |> ignore
            AnsiConsole.MarkupLine("  tars incident list [status]") |> ignore
            AnsiConsole.MarkupLine("  tars incident show <id>") |> ignore
            AnsiConsole.MarkupLine("  tars incident report <message> [severity]") |> ignore
            AnsiConsole.MarkupLine("  tars incident status <id> <new_status>") |> ignore
            return 1
    }
