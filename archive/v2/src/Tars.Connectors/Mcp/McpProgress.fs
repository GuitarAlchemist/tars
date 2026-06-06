namespace Tars.Connectors.Mcp

open System

/// Status of a task step
type StepStatus =
    | Pending
    | InProgress
    | Complete
    | Failed
    | Skipped

/// Represents a single step in a task
type TaskStep =
    { Id: string
      Name: string
      Status: StepStatus
      StartedAt: DateTime option
      CompletedAt: DateTime option
      Output: string option }

/// Information about a running subagent
type SubagentInfo =
    { Id: Guid
      Name: string
      Goal: string
      Status: string
      Progress: float
      StartedAt: DateTime }

/// Artifact produced during task execution
type TaskArtifact =
    { Type: string // "plan", "screenshot", "code", "file"
      Path: string
      Description: string option }

/// Mode of the current task
type TaskMode =
    | Planning
    | Execution
    | Verification
    | Research
    | Idle

module TaskMode =
    let toString =
        function
        | Planning -> "PLANNING"
        | Execution -> "EXECUTION"
        | Verification -> "VERIFICATION"
        | Research -> "RESEARCH"
        | Idle -> "IDLE"

    let fromString (s: string) =
        match s.ToUpperInvariant() with
        | "PLANNING" -> Planning
        | "EXECUTION" -> Execution
        | "VERIFICATION" -> Verification
        | "RESEARCH" -> Research
        | _ -> Idle

/// Progress update sent to IDE via MCP notification
type TaskProgressUpdate =
    { TaskId: Guid
      TaskName: string
      Mode: string
      Status: string
      Progress: float // 0.0 to 1.0
      Steps: TaskStep list
      Artifacts: TaskArtifact list
      Subagents: SubagentInfo list
      UpdatedAt: DateTime }

/// Interface for observing progress events
type IProgressObserver =
    abstract member OnProgress: TaskProgressUpdate -> unit

/// Manages progress reporting for a task
type ProgressReporter(taskId: Guid, taskName: string) =
    let mutable mode = TaskMode.Idle
    let mutable status = "Initializing"
    let mutable progress = 0.0
    let mutable steps: TaskStep list = []
    let mutable artifacts: TaskArtifact list = []
    let mutable subagents: SubagentInfo list = []
    let observers = System.Collections.Generic.List<IProgressObserver>()

    member this.TaskId = taskId
    member this.TaskName = taskName

    member this.AddObserver(observer: IProgressObserver) = observers.Add(observer)

    member this.RemoveObserver(observer: IProgressObserver) = observers.Remove(observer) |> ignore

    member private this.Notify() =
        let update =
            { TaskId = taskId
              TaskName = taskName
              Mode = TaskMode.toString mode
              Status = status
              Progress = progress
              Steps = steps
              Artifacts = artifacts
              Subagents = subagents
              UpdatedAt = DateTime.UtcNow }

        for obs in observers do
            obs.OnProgress(update)

    member this.SetMode(newMode: TaskMode) =
        mode <- newMode
        this.Notify()

    member this.SetStatus(newStatus: string) =
        status <- newStatus
        this.Notify()

    member this.SetProgress(newProgress: float) =
        progress <- Math.Clamp(newProgress, 0.0, 1.0)
        this.Notify()

    member this.AddStep(step: TaskStep) =
        steps <- steps @ [ step ]
        this.Notify()

    member this.UpdateStep(stepId: string, newStatus: StepStatus, ?output: string) =
        steps <-
            steps
            |> List.map (fun s ->
                if s.Id = stepId then
                    { s with
                        Status = newStatus
                        CompletedAt =
                            if newStatus = Complete || newStatus = Failed then
                                Some DateTime.UtcNow
                            else
                                s.CompletedAt
                        Output = output |> Option.orElse s.Output }
                else
                    s)

        this.Notify()

    member this.StartStep(stepId: string) =
        steps <-
            steps
            |> List.map (fun s ->
                if s.Id = stepId then
                    { s with
                        Status = InProgress
                        StartedAt = Some DateTime.UtcNow }
                else
                    s)

        this.Notify()

    member this.AddArtifact(artifact: TaskArtifact) =
        artifacts <- artifacts @ [ artifact ]
        this.Notify()

    member this.AddSubagent(subagent: SubagentInfo) =
        subagents <- subagents @ [ subagent ]
        this.Notify()

    member this.UpdateSubagent(subagentId: Guid, newStatus: string, newProgress: float) =
        subagents <-
            subagents
            |> List.map (fun s ->
                if s.Id = subagentId then
                    { s with
                        Status = newStatus
                        Progress = newProgress }
                else
                    s)

        this.Notify()

    member this.RemoveSubagent(subagentId: Guid) =
        subagents <- subagents |> List.filter (fun s -> s.Id <> subagentId)
        this.Notify()

    member this.GetCurrentUpdate() : TaskProgressUpdate =
        { TaskId = taskId
          TaskName = taskName
          Mode = TaskMode.toString mode
          Status = status
          Progress = progress
          Steps = steps
          Artifacts = artifacts
          Subagents = subagents
          UpdatedAt = DateTime.UtcNow }
