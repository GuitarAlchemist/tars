namespace Tars.Interface.Cli.Commands

open System
open System.Threading
open Spectre.Console
open Tars.Core
open Tars.Tools
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Connectors.Redis

module SwarmCommand =

    let private defaultRedis = "localhost:6379"

    let private printHelp () =
        AnsiConsole.MarkupLine("[bold cyan]TARS Swarm[/] - Massively parallel TARS instances")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [bold]tars swarm start [[N]][/]     Start N worker instances (default 3)")
        AnsiConsole.MarkupLine("  [bold]tars swarm status[/]          Show active workers and queue")
        AnsiConsole.MarkupLine("  [bold]tars swarm submit <goal>[/]   Submit a job to the swarm")
        AnsiConsole.MarkupLine("  [bold]tars swarm fan-out <goal>[/]  Fan-out: run goal with all patterns in parallel")
        AnsiConsole.MarkupLine("  [bold]tars swarm shutdown[/]        Gracefully stop all workers")
        AnsiConsole.MarkupLine("  [bold]tars swarm flush[/]           Clear all swarm state")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("  [dim]Options:[/]")
        AnsiConsole.MarkupLine("    --redis <host:port>  Redis/Sider address (default localhost:6379)")
        0

    let private tryConnect (redisAddr: string) =
        try
            let bus = new SwarmBus(redisAddr)
            if bus.Connect() then
                Some bus
            else
                AnsiConsole.MarkupLine("[red]Failed to connect to Redis/Sider at {0}[/]", redisAddr)
                AnsiConsole.MarkupLine("[dim]Start Sider: cargo install sider && sider -p 6379[/]")
                None
        with ex ->
            AnsiConsole.MarkupLine("[red]Connection error: {0}[/]", ex.Message)
            AnsiConsole.MarkupLine("[dim]Start Sider: cargo install sider && sider -p 6379[/]")
            None

    let private startWorkers (bus: SwarmBus) (count: int) =
        let compiler = PatternCompiler.DefaultPatternCompiler() :> IPatternCompiler
        let selector = PatternSelector.HistoryAwareSelector() :> IPatternSelector
        let registry = ToolRegistry() :> IToolRegistry

        let cts = new CancellationTokenSource()

        AnsiConsole.MarkupLine(sprintf "[bold green]Starting %d TARS workers...[/]" count)

        let workerIds =
            [ for _ in 1..count do
                let worker = SwarmWorker(bus, compiler, selector, registry)
                let id = worker.StartBackground(cts.Token)
                AnsiConsole.MarkupLine(sprintf "  [green]+[/] Worker [bold]%s[/] started" id)
                yield id ]

        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine(sprintf "[bold]%d workers running.[/] Press Ctrl+C to stop." workerIds.Length)

        // Wait for Ctrl+C
        let waitHandle = new ManualResetEventSlim(false)
        Console.CancelKeyPress.Add(fun e ->
            e.Cancel <- true
            AnsiConsole.MarkupLine("[yellow]Shutting down workers...[/]")
            bus.SendControl("shutdown")
            cts.Cancel()
            waitHandle.Set())

        waitHandle.Wait()
        Thread.Sleep(1500) // Give workers time to send final heartbeat
        AnsiConsole.MarkupLine("[green]All workers stopped.[/]")

    let private showStatus (bus: SwarmBus) =
        let workers = bus.GetWorkers()
        let queueLen = bus.QueueLength()

        AnsiConsole.MarkupLine("[bold cyan]TARS Swarm Status[/]")
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine(sprintf "  Queue depth: [bold]%d[/] jobs" queueLen)
        AnsiConsole.MarkupLine(sprintf "  Active workers: [bold]%d[/]" workers.Length)
        AnsiConsole.MarkupLine("")

        if workers.Length > 0 then
            let table = Table()
            table.AddColumn("Worker ID") |> ignore
            table.AddColumn("Status") |> ignore
            table.AddColumn("Current Job") |> ignore
            table.AddColumn("Completed") |> ignore
            table.AddColumn("Uptime") |> ignore

            for w in workers do
                let statusColor = match w.Status with "busy" -> "yellow" | "idle" -> "green" | _ -> "red"
                let jobStr = w.CurrentJobId |> Option.defaultValue "-"
                let uptime = TimeSpan.FromMilliseconds(float w.UptimeMs)
                table.AddRow(
                    w.WorkerId,
                    sprintf "[%s]%s[/]" statusColor w.Status,
                    jobStr,
                    string w.CompletedJobs,
                    sprintf "%02d:%02d:%02d" (int uptime.TotalHours) uptime.Minutes uptime.Seconds) |> ignore

            AnsiConsole.Write(table)
        else
            AnsiConsole.MarkupLine("  [dim]No workers online[/]")

    let private submitJob (bus: SwarmBus) (goal: string) (patternHint: string option) =
        let jobId = Guid.NewGuid().ToString("N").Substring(0, 12)
        let job =
            { JobId = jobId
              Goal = goal
              PatternHint = patternHint
              MaxSteps = 5
              Priority = 1
              PostedBy = "cli"
              PostedAt = DateTime.UtcNow }

        bus.PostJob(job)
        AnsiConsole.MarkupLine(sprintf "[green]Job submitted:[/] [bold]%s[/]" jobId)
        AnsiConsole.MarkupLine(sprintf "  Goal: %s" goal)
        AnsiConsole.MarkupLine(sprintf "  Queue depth: [bold]%d[/]" (bus.QueueLength()))

        // Wait for result (with timeout)
        AnsiConsole.MarkupLine("[dim]Waiting for result...[/]")
        let mutable attempts = 0
        let mutable result = None

        while attempts < 30 && result.IsNone do
            Thread.Sleep(1000)
            result <- bus.GetResult(jobId)
            attempts <- attempts + 1

        match result with
        | Some r ->
            let statusColor = if r.Success then "green" else "red"
            AnsiConsole.MarkupLine("")
            AnsiConsole.MarkupLine(sprintf "[bold %s]Result:[/] %s" statusColor (if r.Success then "SUCCESS" else "FAILED"))
            AnsiConsole.MarkupLine(sprintf "  Worker: %s" r.WorkerId)
            AnsiConsole.MarkupLine(sprintf "  Pattern: %s" r.PatternUsed)
            AnsiConsole.MarkupLine(sprintf "  Steps: %d" r.StepCount)
            AnsiConsole.MarkupLine(sprintf "  Duration: %dms" r.DurationMs)
            AnsiConsole.MarkupLine(sprintf "  Output: %s" r.Output)
        | None ->
            AnsiConsole.MarkupLine("[yellow]Timeout waiting for result (30s). Job may still be processing.[/]")

    let private fanOut (bus: SwarmBus) (goal: string) =
        let patterns = [ "cot"; "react"; "got"; "tot" ]
        AnsiConsole.MarkupLine(sprintf "[bold cyan]Fan-out:[/] Submitting goal to %d patterns in parallel" patterns.Length)
        AnsiConsole.MarkupLine(sprintf "  Goal: %s" goal)

        let jobIds =
            patterns
            |> List.map (fun p ->
                let jobId = Guid.NewGuid().ToString("N").Substring(0, 12)
                let job =
                    { JobId = jobId
                      Goal = goal
                      PatternHint = Some p
                      MaxSteps = 5
                      Priority = 1
                      PostedBy = "cli-fanout"
                      PostedAt = DateTime.UtcNow }
                bus.PostJob(job)
                AnsiConsole.MarkupLine(sprintf "  [green]+[/] Job [bold]%s[/] (%s)" jobId p)
                jobId, p)

        // Wait and collect results
        AnsiConsole.MarkupLine("")
        AnsiConsole.MarkupLine("[dim]Collecting results...[/]")
        Thread.Sleep(5000) // Give workers time

        let mutable results = []
        for (jobId, pattern) in jobIds do
            match bus.GetResult(jobId) with
            | Some r -> results <- (pattern, r) :: results
            | None -> AnsiConsole.MarkupLine(sprintf "  [yellow]%s: no result yet[/]" pattern)

        if results.Length > 0 then
            AnsiConsole.MarkupLine("")
            let table = Table()
            table.AddColumn("Pattern") |> ignore
            table.AddColumn("Status") |> ignore
            table.AddColumn("Steps") |> ignore
            table.AddColumn("Duration") |> ignore
            table.AddColumn("Worker") |> ignore

            for (pattern, r) in results |> List.rev do
                let status = if r.Success then "[green]PASS[/]" else "[red]FAIL[/]"
                table.AddRow(pattern, status, string r.StepCount, sprintf "%dms" r.DurationMs, r.WorkerId) |> ignore

            AnsiConsole.Write(table)

            // Find best result
            let best =
                results
                |> List.filter (fun (_, r) -> r.Success)
                |> List.sortBy (fun (_, r) -> r.DurationMs)
                |> List.tryHead

            match best with
            | Some (pattern, r) ->
                AnsiConsole.MarkupLine(sprintf "[bold green]Best result:[/] %s (%dms, %d steps)" pattern r.DurationMs r.StepCount)
            | None ->
                AnsiConsole.MarkupLine("[yellow]No successful results[/]")

    let run (args: string list) =
        let mutable redisAddr = defaultRedis
        let mutable subCmd = ""
        let mutable goalParts = []
        let mutable remaining = args

        // Parse args
        while not remaining.IsEmpty do
            match remaining with
            | "--redis" :: addr :: rest ->
                redisAddr <- addr
                remaining <- rest
            | cmd :: rest when subCmd = "" ->
                subCmd <- cmd
                remaining <- rest
            | word :: rest ->
                goalParts <- goalParts @ [ word ]
                remaining <- rest
            | [] -> ()

        let goal = if goalParts.IsEmpty then "" else String.Join(" ", goalParts)

        match subCmd with
        | "start" ->
            let count = if goal <> "" then (try int goal with _ -> 3) else 3
            match tryConnect redisAddr with
            | Some bus -> startWorkers bus count; 0
            | None -> 1
        | "status" ->
            match tryConnect redisAddr with
            | Some bus -> showStatus bus; 0
            | None -> 1
        | "submit" when goal <> "" ->
            match tryConnect redisAddr with
            | Some bus -> submitJob bus goal None; 0
            | None -> 1
        | "fan-out" when goal <> "" ->
            match tryConnect redisAddr with
            | Some bus -> fanOut bus goal; 0
            | None -> 1
        | "shutdown" ->
            match tryConnect redisAddr with
            | Some bus ->
                bus.SendControl("shutdown")
                AnsiConsole.MarkupLine("[green]Shutdown signal sent to all workers[/]")
                0
            | None -> 1
        | "flush" ->
            match tryConnect redisAddr with
            | Some bus ->
                bus.FlushSwarm()
                AnsiConsole.MarkupLine("[green]Swarm state cleared[/]")
                0
            | None -> 1
        | _ -> printHelp ()
