/// Pipeline Executor - Runs project pipelines with persona integration
module Tars.Core.PipelineExecutor

open System
open System.Threading.Tasks
open Tars.Core.Project
open Tars.Core.ProjectRegistry
open Tars.Core.Persona
open Tars.Core.PersonaRegistry

// ============================================================================
// Execution Events
// ============================================================================

/// Events emitted during pipeline execution
type PipelineEvent =
    | PipelineStarted of projectId: string * stage: PipelineStage
    | StageStarted of projectId: string * stage: PipelineStage
    | StageCompleted of projectId: string * stage: PipelineStage * artifacts: Map<string, string>
    | StageFailed of projectId: string * stage: PipelineStage * error: string
    | ApprovalRequired of projectId: string * stage: PipelineStage
    | ApprovalReceived of projectId: string * stage: PipelineStage * approved: bool
    | PipelineCompleted of projectId: string
    | PipelineFailed of projectId: string * error: string

/// Handler for pipeline events
type PipelineEventHandler = PipelineEvent -> unit

// ============================================================================
// Stage Execution Context
// ============================================================================

/// Context for executing a stage
type StageExecutionContext =
    { Project: Project
      Stage: StageConfig
      Personas: Persona list
      InputArtifacts: Map<string, string>
      EventHandler: PipelineEventHandler option }

/// Result of executing a stage
type StageExecutionResult =
    | StageSuccess of outputArtifacts: Map<string, string>
    | StageAwaitingApproval
    | StageError of error: string

// ============================================================================
// Stage Executor
// ============================================================================

/// Execute a single stage with its personas
let executeStage (ctx: StageExecutionContext) : Task<StageExecutionResult> =
    task {
        try
            // Emit stage started event
            ctx.EventHandler
            |> Option.iter (fun h -> h (StageStarted(ctx.Project.Id, ctx.Stage.Stage)))

            // Build RTFD prompts for each persona at this stage
            let prompts =
                ctx.Personas
                |> List.map (fun persona ->
                    let taskDesc =
                        $"Execute {stageName ctx.Stage.Stage} phase for project '{ctx.Project.Name}'"

                    let details =
                        if ctx.InputArtifacts.IsEmpty then
                            None
                        else
                            let artifactNames =
                                ctx.InputArtifacts |> Map.keys |> Seq.toList |> String.concat ", "

                            Some $"Input artifacts: {artifactNames}"

                    buildRtfdPrompt
                        { Persona = persona
                          Task = taskDesc
                          Format = None
                          Details = details })

            // Simulate execution (in real impl, this would call LLM)
            let outputArtifacts =
                ctx.Stage.OutputArtifacts
                |> List.map (fun name -> name, $"/artifacts/{ctx.Project.Id}/{stageName ctx.Stage.Stage}/{name}")
                |> Map.ofList

            // Check if approval is required
            if requiresApproval ctx.Project.ExecutionMode ctx.Stage.Stage then
                ctx.EventHandler
                |> Option.iter (fun h -> h (ApprovalRequired(ctx.Project.Id, ctx.Stage.Stage)))

                return StageAwaitingApproval
            else
                ctx.EventHandler
                |> Option.iter (fun h -> h (StageCompleted(ctx.Project.Id, ctx.Stage.Stage, outputArtifacts)))

                return StageSuccess outputArtifacts
        with ex ->
            ctx.EventHandler
            |> Option.iter (fun h -> h (StageFailed(ctx.Project.Id, ctx.Stage.Stage, ex.Message)))

            return StageError ex.Message
    }

// ============================================================================
// Pipeline Executor
// ============================================================================

type PipelineExecutor(projectRegistry: ProjectRegistry, personaRegistry: PersonaRegistry) =

    let mutable eventHandler: PipelineEventHandler option = None

    let mutable pendingApprovals: Map<string * PipelineStage, Map<string, string>> =
        Map.empty

    /// Set event handler
    member _.SetEventHandler(handler: PipelineEventHandler) = eventHandler <- Some handler

    /// Start or resume pipeline execution
    member this.ExecuteAsync(projectId: string) : Task<Result<unit, string>> =
        task {
            match projectRegistry.Get projectId, projectRegistry.GetState projectId with
            | None, _ -> return Result.Error $"Project '{projectId}' not found"
            | _, None -> return Result.Error $"Project state not found for '{projectId}'"
            | Some project, Some state ->
                let stages = templateStages project.Template

                if stages.IsEmpty then
                    return Result.Error "No stages in pipeline"
                else
                    // Determine starting stage
                    let startStage =
                        match state.CurrentStage with
                        | Some s -> s
                        | None -> stages.Head

                    eventHandler
                    |> Option.iter (fun h -> h (PipelineStarted(projectId, startStage)))

                    // Execute stages sequentially
                    let mutable currentIdx = stages |> List.findIndex ((=) startStage)
                    let mutable error: string option = None
                    let mutable artifacts = Map.empty

                    while currentIdx < stages.Length && error.IsNone do
                        let stage = stages.[currentIdx]

                        let stageConfig =
                            project.Stages
                            |> List.tryFind (fun s -> s.Stage = stage)
                            |> Option.defaultValue
                                { Stage = stage
                                  Personas = []
                                  RequiredArtifacts = []
                                  OutputArtifacts = []
                                  CompletionCriteria = None }

                        // Get personas for this stage
                        let personas = stageConfig.Personas |> List.choose personaRegistry.Get

                        let ctx =
                            { Project = project
                              Stage = stageConfig
                              Personas = personas
                              InputArtifacts = artifacts
                              EventHandler = eventHandler }

                        let! result = executeStage ctx

                        match result with
                        | StageSuccess newArtifacts ->
                            artifacts <- Map.fold (fun acc k v -> Map.add k v acc) artifacts newArtifacts
                            // Update state
                            let updatedState = { state with CurrentStage = Some stage }
                            projectRegistry.UpdateState(projectId, updatedState) |> ignore
                            currentIdx <- currentIdx + 1
                        | StageAwaitingApproval ->
                            pendingApprovals <- pendingApprovals |> Map.add (projectId, stage) artifacts
                            // Don't increment - wait for approval
                            currentIdx <- stages.Length // Exit loop
                        | StageError e ->
                            error <- Some e
                            eventHandler |> Option.iter (fun h -> h (PipelineFailed(projectId, e)))

                    match error with
                    | Some e -> return Result.Error e
                    | None when currentIdx >= stages.Length ->
                        // Pipeline complete
                        let updatedState =
                            { state with
                                CurrentStage = None
                                CompletedAt = Some DateTime.UtcNow }

                        projectRegistry.UpdateState(projectId, updatedState) |> ignore
                        eventHandler |> Option.iter (fun h -> h (PipelineCompleted projectId))
                        return Result.Ok()
                    | None ->
                        // Awaiting approval
                        return Result.Ok()
        }

    /// Approve a pending stage
    member this.ApproveStage(projectId: string, stage: PipelineStage, approved: bool) : Task<Result<unit, string>> =
        task {
            eventHandler
            |> Option.iter (fun h -> h (ApprovalReceived(projectId, stage, approved)))

            if not approved then
                eventHandler
                |> Option.iter (fun h -> h (PipelineFailed(projectId, "Stage rejected by user")))

                return Result.Error "Stage rejected"
            else
                pendingApprovals <- pendingApprovals |> Map.remove (projectId, stage)
                // Resume execution
                return! this.ExecuteAsync(projectId)
        }

    /// Check if a stage is pending approval
    member _.IsPendingApproval(projectId: string, stage: PipelineStage) : bool =
        Map.containsKey (projectId, stage) pendingApprovals

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create executor with default registries
let createExecutor () =
    PipelineExecutor(ProjectRegistry.defaultRegistry, PersonaRegistry.defaultRegistry)

/// Start a new project and execute its pipeline
let startProject id name rootPath template mode =
    match createAndRegister id name rootPath template mode with
    | Result.Error e -> Result.Error e
    | Result.Ok project ->
        let executor = createExecutor ()
        executor.ExecuteAsync(project.Id) |> Async.AwaitTask |> Async.RunSynchronously
