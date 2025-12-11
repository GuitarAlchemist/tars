module Tars.Tests.PipelineExecutorTests

open System
open Xunit
open Tars.Core.Project
open Tars.Core.ProjectRegistry
open Tars.Core.PipelineExecutor

// ============================================================================
// PipelineEvent Tests
// ============================================================================

[<Fact>]
let ``PipelineEvent types are correctly defined`` () =
    let events =
        [ PipelineStarted("proj-1", Vision)
          StageStarted("proj-1", Specification)
          StageCompleted("proj-1", Development, Map.empty)
          StageFailed("proj-1", QualityAssurance, "Test error")
          ApprovalRequired("proj-1", Demo)
          ApprovalReceived("proj-1", Demo, true)
          PipelineCompleted("proj-1")
          PipelineFailed("proj-1", "Pipeline error") ]

    Assert.Equal(8, events.Length)

// ============================================================================
// StageExecutionContext Tests
// ============================================================================

[<Fact>]
let ``StageExecutionContext is correctly created`` () =
    let project = createProject "test" "Test" "/tmp" StandardSDLC Continuous

    let stageConfig =
        { Stage = Vision
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = []
          CompletionCriteria = None }

    let ctx =
        { Project = project
          Stage = stageConfig
          Personas = []
          InputArtifacts = Map.empty
          EventHandler = None }

    Assert.Equal(Vision, ctx.Stage.Stage)
    Assert.True(ctx.Personas.IsEmpty)

// ============================================================================
// executeStage Tests
// ============================================================================

[<Fact>]
let ``executeStage returns Success for Continuous mode`` () =
    let project = createProject "test" "Test" "/tmp" StandardSDLC Continuous

    let stageConfig =
        { Stage = Vision
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = [ "vision.md" ]
          CompletionCriteria = None }

    let ctx =
        { Project = project
          Stage = stageConfig
          Personas = []
          InputArtifacts = Map.empty
          EventHandler = None }

    let result = executeStage ctx |> Async.AwaitTask |> Async.RunSynchronously

    match result with
    | StageSuccess artifacts -> Assert.True(Map.containsKey "vision.md" artifacts)
    | _ -> Assert.Fail("Expected StageSuccess")

[<Fact>]
let ``executeStage returns AwaitingApproval for HumanInLoop mode`` () =
    let project = createProject "test" "Test" "/tmp" StandardSDLC HumanInLoop

    let stageConfig =
        { Stage = Vision
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = []
          CompletionCriteria = None }

    let ctx =
        { Project = project
          Stage = stageConfig
          Personas = []
          InputArtifacts = Map.empty
          EventHandler = None }

    let result = executeStage ctx |> Async.AwaitTask |> Async.RunSynchronously

    match result with
    | StageAwaitingApproval -> Assert.True(true)
    | _ -> Assert.Fail("Expected StageAwaitingApproval")

[<Fact>]
let ``executeStage respects Hybrid mode pause stages`` () =
    let project =
        createProject "test" "Test" "/tmp" StandardSDLC (Hybrid [ QualityAssurance; Demo ])

    // Vision should auto-proceed
    let visionConfig =
        { Stage = Vision
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = []
          CompletionCriteria = None }

    let visionCtx =
        { Project = project
          Stage = visionConfig
          Personas = []
          InputArtifacts = Map.empty
          EventHandler = None }

    let visionResult =
        executeStage visionCtx |> Async.AwaitTask |> Async.RunSynchronously

    Assert.True(
        match visionResult with
        | StageSuccess _ -> true
        | _ -> false
    )

    // QA should require approval
    let qaConfig =
        { Stage = QualityAssurance
          Personas = []
          RequiredArtifacts = []
          OutputArtifacts = []
          CompletionCriteria = None }

    let qaCtx =
        { Project = project
          Stage = qaConfig
          Personas = []
          InputArtifacts = Map.empty
          EventHandler = None }

    let qaResult = executeStage qaCtx |> Async.AwaitTask |> Async.RunSynchronously

    Assert.True(
        match qaResult with
        | StageAwaitingApproval -> true
        | _ -> false
    )

// ============================================================================
// PipelineExecutor Tests
// ============================================================================

[<Fact>]
let ``PipelineExecutor tracks events`` () =
    let registry = ProjectRegistry()
    let personaRegistry = Tars.Core.PersonaRegistry.PersonaRegistry()
    let executor = PipelineExecutor(registry, personaRegistry)

    let mutable events = []
    executor.SetEventHandler(fun e -> events <- e :: events)

    let project =
        createProject "events-test" "Events Test" "/tmp" StandardSDLC Continuous

    registry.Register(project) |> ignore

    let result =
        executor.ExecuteAsync("events-test")
        |> Async.AwaitTask
        |> Async.RunSynchronously

    Assert.True(Result.isOk result)

    Assert.True(
        events
        |> List.exists (function
            | PipelineStarted _ -> true
            | _ -> false)
    )

    Assert.True(
        events
        |> List.exists (function
            | PipelineCompleted _ -> true
            | _ -> false)
    )

[<Fact>]
let ``PipelineExecutor handles unknown project`` () =
    let registry = ProjectRegistry()
    let personaRegistry = Tars.Core.PersonaRegistry.PersonaRegistry()
    let executor = PipelineExecutor(registry, personaRegistry)

    let result =
        executor.ExecuteAsync("nonexistent")
        |> Async.AwaitTask
        |> Async.RunSynchronously

    Assert.True(Result.isError result)

[<Fact>]
let ``PipelineExecutor pauses on HumanInLoop`` () =
    let registry = ProjectRegistry()
    let personaRegistry = Tars.Core.PersonaRegistry.PersonaRegistry()
    let executor = PipelineExecutor(registry, personaRegistry)

    let project = createProject "hitl-test" "HITL Test" "/tmp" StandardSDLC HumanInLoop
    registry.Register(project) |> ignore

    let mutable approvalRequested = false

    executor.SetEventHandler(fun e ->
        match e with
        | ApprovalRequired _ -> approvalRequested <- true
        | _ -> ())

    let result =
        executor.ExecuteAsync("hitl-test") |> Async.AwaitTask |> Async.RunSynchronously

    Assert.True(Result.isOk result)
    Assert.True(approvalRequested)
    Assert.True(executor.IsPendingApproval("hitl-test", Vision))
