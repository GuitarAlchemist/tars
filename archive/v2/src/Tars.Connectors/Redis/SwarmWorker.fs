namespace Tars.Connectors.Redis

open System
open System.Diagnostics
open System.Threading
open Tars.Core
open Tars.Cortex
open Tars.Cortex.WoTTypes
open Tars.Cortex.ClaudeCodeBridge

/// A TARS swarm worker that polls the work queue and executes WoT plans.
type SwarmWorker
    (bus: SwarmBus,
     compiler: IPatternCompiler,
     selector: IPatternSelector,
     toolRegistry: IToolRegistry) =

    let workerId =
        let id = Guid.NewGuid().ToString("N").Substring(0, 8)
        sprintf "tars-worker-%s" id
    let startedAt = Stopwatch.StartNew()
    let mutable completedJobs = 0
    let mutable currentJobId: string option = None
    let mutable running = false

    let makeHeartbeat status =
        { WorkerId = workerId
          Status = status
          CurrentJobId = currentJobId
          UptimeMs = startedAt.ElapsedMilliseconds
          CompletedJobs = completedJobs
          Timestamp = DateTime.UtcNow }

    /// Execute a single job: compile plan, walk nodes, complete.
    let executeJob (job: SwarmJob) : SwarmResult =
        let sw = Stopwatch.StartNew()
        try
            // Compile plan
            let patternOverride =
                match job.PatternHint with
                | Some "react" | Some "ReAct" -> Some "react"
                | Some "got" | Some "GraphOfThoughts" -> Some "got"
                | Some "tot" | Some "TreeOfThoughts" -> Some "tot"
                | _ -> None

            let escapedGoal = job.Goal.Replace("\"", "\\\"")
            let inputJson =
                sprintf """{"goal": "%s", "max_steps": %d}""" escapedGoal job.MaxSteps

            let planResult = compilePlan compiler selector toolRegistry inputJson

            match planResult with
            | Result.Error err ->
                { JobId = job.JobId
                  WorkerId = workerId
                  Success = false
                  Output = $"Compilation failed: {err}"
                  PatternUsed = "none"
                  DurationMs = sw.ElapsedMilliseconds
                  StepCount = 0
                  CompletedAt = DateTime.UtcNow }
            | Result.Ok planJson ->
                let doc = System.Text.Json.JsonDocument.Parse(planJson)
                let root = doc.RootElement
                let planId = root.GetProperty("planId").GetString()
                let pattern = root.GetProperty("pattern").GetString()
                let nodes = root.GetProperty("nodes")

                // Execute each node (for Reason nodes, we produce stub output
                // since there's no LLM in worker mode — the plan structure is
                // what matters for tracing and regression checking)
                let mutable stepCount = 0
                let mutable allSuccess = true

                for i in 0 .. nodes.GetArrayLength() - 1 do
                    let node = nodes[i]
                    let nodeId = node.GetProperty("id").GetString()
                    let kind = node.GetProperty("kind").GetString()

                    let stepInput =
                        match kind with
                        | "Reason" ->
                            let prompt =
                                match node.GetProperty("prompt").ValueKind with
                                | System.Text.Json.JsonValueKind.String -> node.GetProperty("prompt").GetString()
                                | _ -> job.Goal
                            sprintf """{"plan_id": "%s", "node_id": "%s", "input": "[Worker %s] Reasoning: %s"}"""
                                planId nodeId workerId (prompt.Substring(0, min 80 prompt.Length))
                        | _ ->
                            sprintf """{"plan_id": "%s", "node_id": "%s", "input": ""}"""
                                planId nodeId

                    let result =
                        executeStep toolRegistry stepInput
                        |> Async.RunSynchronously

                    match result with
                    | Result.Ok _ -> stepCount <- stepCount + 1
                    | Result.Error _ ->
                        stepCount <- stepCount + 1
                        allSuccess <- false

                // Complete the plan
                let completeInput =
                    sprintf """{"plan_id": "%s", "final_output": "Completed by worker %s"}"""
                        planId workerId

                let _ = completePlan completeInput

                { JobId = job.JobId
                  WorkerId = workerId
                  Success = allSuccess
                  Output = $"Executed {stepCount} steps via {pattern}"
                  PatternUsed = pattern
                  DurationMs = sw.ElapsedMilliseconds
                  StepCount = stepCount
                  CompletedAt = DateTime.UtcNow }

        with ex ->
            { JobId = job.JobId
              WorkerId = workerId
              Success = false
              Output = $"Worker error: {ex.Message}"
              PatternUsed = "error"
              DurationMs = sw.ElapsedMilliseconds
              StepCount = 0
              CompletedAt = DateTime.UtcNow }

    /// The worker's main loop.
    member _.Run(ct: CancellationToken) =
        running <- true

        // Listen for control commands
        bus.OnControl(fun cmd ->
            match cmd with
            | "shutdown" -> running <- false
            | _ -> ())

        // Main poll loop
        while running && not ct.IsCancellationRequested do
            try
                // Send heartbeat
                let status = if currentJobId.IsSome then "busy" else "idle"
                bus.SendHeartbeat(makeHeartbeat status)

                // Try to take a job
                match bus.TakeJob(1) with
                | None ->
                    // No work — wait briefly
                    Thread.Sleep(500)
                | Some job ->
                    currentJobId <- Some job.JobId

                    // Execute
                    let result = executeJob job
                    bus.SubmitResult(result)

                    completedJobs <- completedJobs + 1
                    currentJobId <- None
            with ex ->
                // Log but don't crash
                eprintfn "[%s] Error: %s" workerId ex.Message
                Thread.Sleep(1000)

        // Final heartbeat
        bus.SendHeartbeat(makeHeartbeat "stopping")
        running <- false

    /// Start the worker in a background thread.
    member this.StartBackground(ct: CancellationToken) =
        let thread = Thread(fun () -> this.Run(ct))
        thread.IsBackground <- true
        thread.Name <- workerId
        thread.Start()
        workerId

    member _.WorkerId = workerId
    member _.CompletedJobs = completedJobs
    member _.IsRunning = running
