namespace Tars.Connectors.Mcp

open System
open System.Collections.Concurrent
open System.Threading
open System.Threading.Tasks
open Tars.Core

/// Request to spawn a new subagent
type SubagentRequest =
    { Goal: string
      MaxDurationMinutes: int
      AllowTools: string list option
      ParentTaskId: Guid option
      AgentHint: string option } // e.g., "research", "coding", "testing"

/// Result from a completed subagent
type SubagentResult =
    { Id: Guid
      Success: bool
      Output: string
      Artifacts: string list
      Duration: TimeSpan
      Error: string option }

/// Internal state for a running subagent
type private SubagentState =
    { Request: SubagentRequest
      CancellationSource: CancellationTokenSource
      Task: Task<SubagentResult>
      StartedAt: DateTime }

/// Manages the lifecycle of long-running subagents
type SubagentManager
    (runSubagent: SubagentRequest -> CancellationToken -> IProgressObserver option -> Task<SubagentResult>) =
    let activeSubagents = ConcurrentDictionary<Guid, SubagentState>()
    let completedSubagents = ConcurrentDictionary<Guid, SubagentResult>()

    /// Spawn a new subagent for deep research or long-running task
    member this.Spawn(request: SubagentRequest, ?progressObserver: IProgressObserver) : Guid =
        let id = Guid.NewGuid()
        let cts = new CancellationTokenSource()

        // Set timeout if specified
        if request.MaxDurationMinutes > 0 then
            cts.CancelAfter(TimeSpan.FromMinutes(float request.MaxDurationMinutes))

        let task =
            task {
                try
                    return! runSubagent request cts.Token (progressObserver)
                with
                | :? OperationCanceledException ->
                    return
                        { Id = id
                          Success = false
                          Output = "Subagent was cancelled"
                          Artifacts = []
                          Duration = DateTime.UtcNow - DateTime.UtcNow
                          Error = Some "Cancelled" }
                | ex ->
                    return
                        { Id = id
                          Success = false
                          Output = ""
                          Artifacts = []
                          Duration = TimeSpan.Zero
                          Error = Some ex.Message }
            }

        let state =
            { Request = request
              CancellationSource = cts
              Task = task
              StartedAt = DateTime.UtcNow }

        activeSubagents.[id] <- state

        // Clean up when done
        task.ContinueWith(fun (t: Task<SubagentResult>) ->
            match activeSubagents.TryRemove(id) with
            | true, _ ->
                let result =
                    { t.Result with
                        Id = id
                        Duration = DateTime.UtcNow - state.StartedAt }

                completedSubagents.[id] <- result
            | false, _ -> ())
        |> ignore

        id

    /// Cancel a running subagent
    member this.Cancel(id: Guid) : bool =
        match activeSubagents.TryGetValue(id) with
        | true, state ->
            state.CancellationSource.Cancel()
            true
        | false, _ -> false

    /// Get status of a subagent (running or completed)
    member this.GetStatus(id: Guid) : SubagentInfo option =
        match activeSubagents.TryGetValue(id) with
        | true, state ->
            Some
                { Id = id
                  Name = state.Request.AgentHint |> Option.defaultValue "Subagent"
                  Goal = state.Request.Goal
                  Status = "Running"
                  Progress = 0.5 // TODO: Get actual progress from observer
                  StartedAt = state.StartedAt }
        | false, _ ->
            match completedSubagents.TryGetValue(id) with
            | true, result ->
                Some
                    { Id = id
                      Name = "Subagent"
                      Goal = ""
                      Status = if result.Success then "Completed" else "Failed"
                      Progress = 1.0
                      StartedAt = DateTime.UtcNow - result.Duration }
            | false, _ -> None

    /// Get result of a completed subagent
    member this.GetResult(id: Guid) : SubagentResult option =
        match completedSubagents.TryGetValue(id) with
        | true, result -> Some result
        | false, _ -> None

    /// List all active subagents
    member this.ListActive() : SubagentInfo list =
        activeSubagents
        |> Seq.map (fun kv ->
            { Id = kv.Key
              Name = kv.Value.Request.AgentHint |> Option.defaultValue "Subagent"
              Goal = kv.Value.Request.Goal
              Status = "Running"
              Progress = 0.5
              StartedAt = kv.Value.StartedAt })
        |> Seq.toList

    /// Get count of active subagents
    member this.ActiveCount = activeSubagents.Count

    /// Wait for a subagent to complete
    member this.WaitAsync(id: Guid, ?timeout: TimeSpan) : Task<SubagentResult option> =
        task {
            match activeSubagents.TryGetValue(id) with
            | true, state ->
                try
                    match timeout with
                    | Some t ->
                        let! completed = Task.WhenAny(state.Task, Task.Delay(t))

                        if completed = state.Task then
                            return Some state.Task.Result
                        else
                            return None
                    | None ->
                        let! result = state.Task
                        return Some result
                with _ ->
                    return None
            | false, _ ->
                return
                    completedSubagents.TryGetValue(id)
                    |> function
                        | true, r -> Some r
                        | false, _ -> None
        }

    /// Cancel all active subagents
    member this.CancelAll() =
        for kv in activeSubagents do
            kv.Value.CancellationSource.Cancel()
