module Tars.Tests.ProjectTests

open System
open Xunit
open Tars.Core.Project
open Tars.Core.ProjectRegistry

// ============================================================================
// PipelineStage Tests
// ============================================================================

[<Fact>]
let ``stageName returns correct names`` () =
    Assert.Equal("Vision", stageName Vision)
    Assert.Equal("QA", stageName QualityAssurance)
    Assert.Equal("Sprint", stageName Sprint)

[<Fact>]
let ``templateStages returns correct stages for StandardSDLC`` () =
    let stages = templateStages StandardSDLC
    Assert.Equal(5, stages.Length)
    Assert.Equal(Vision, stages.[0])
    Assert.Equal(Demo, stages.[4])

[<Fact>]
let ``templateStages returns correct stages for AgileSprint`` () =
    let stages = templateStages AgileSprint
    Assert.Equal(4, stages.Length)
    Assert.Equal(Backlog, stages.[0])
    Assert.Equal(Deploy, stages.[3])

[<Fact>]
let ``templateStages returns correct stages for Research`` () =
    let stages = templateStages Research
    Assert.Equal(4, stages.Length)
    Assert.Equal(Hypothesis, stages.[0])
    Assert.Equal(Report, stages.[3])

// ============================================================================
// ExecutionMode Tests
// ============================================================================

[<Fact>]
let ``requiresApproval returns true for HumanInLoop`` () =
    Assert.True(requiresApproval HumanInLoop Vision)
    Assert.True(requiresApproval HumanInLoop Development)

[<Fact>]
let ``requiresApproval returns false for Continuous`` () =
    Assert.False(requiresApproval Continuous Vision)
    Assert.False(requiresApproval Continuous Development)

[<Fact>]
let ``requiresApproval respects Hybrid pauseAt list`` () =
    let mode = Hybrid [ QualityAssurance; Demo ]
    Assert.False(requiresApproval mode Vision)
    Assert.False(requiresApproval mode Development)
    Assert.True(requiresApproval mode QualityAssurance)
    Assert.True(requiresApproval mode Demo)

// ============================================================================
// Project Creation Tests
// ============================================================================

[<Fact>]
let ``createProject creates project with correct fields`` () =
    let project =
        createProject "proj-1" "Test Project" "/tmp/test" StandardSDLC Continuous

    Assert.Equal("proj-1", project.Id)
    Assert.Equal("Test Project", project.Name)
    Assert.Equal("/tmp/test", project.RootPath)
    Assert.Equal(StandardSDLC, project.Template)
    Assert.Equal(Continuous, project.ExecutionMode)
    Assert.Equal("project_proj-1", project.GraphitiNamespace)
    Assert.Equal(5, project.Stages.Length)

[<Fact>]
let ``createProject generates Graphiti namespace from ID`` () =
    let project = createProject "my-app" "My App" "/path" AgileSprint HumanInLoop
    Assert.Equal("project_my-app", project.GraphitiNamespace)

// ============================================================================
// ProjectState Tests
// ============================================================================

[<Fact>]
let ``initProjectState creates state with all stages NotStarted`` () =
    let project = createProject "test" "Test" "/tmp" StandardSDLC Continuous
    let state = initProjectState project

    Assert.Equal(5, state.StageStates.Count)
    Assert.True(state.StageStates |> Map.forall (fun _ s -> s.Status = NotStarted))
    Assert.True(state.CurrentStage.IsNone)
    Assert.True(state.StartedAt.IsNone)

[<Fact>]
let ``nextStage returns correct next stage`` () =
    let stages = [ Vision; Specification; Development ]
    Assert.Equal(Some Specification, nextStage stages Vision)
    Assert.Equal(Some Development, nextStage stages Specification)
    Assert.Equal(None, nextStage stages Development)

// ============================================================================
// ProjectRegistry Tests
// ============================================================================

[<Fact>]
let ``ProjectRegistry registers and retrieves project`` () =
    let registry = ProjectRegistry()

    let project =
        createProject "reg-test" "Registry Test" "/tmp" StandardSDLC Continuous

    let result = registry.Register(project)
    Assert.True(Result.isOk result)

    let retrieved = registry.Get("reg-test")
    Assert.True(retrieved.IsSome)
    Assert.Equal("Registry Test", retrieved.Value.Name)

[<Fact>]
let ``ProjectRegistry rejects duplicate ID`` () =
    let registry = ProjectRegistry()
    let project1 = createProject "dup-id" "Project 1" "/tmp" StandardSDLC Continuous
    let project2 = createProject "dup-id" "Project 2" "/tmp2" AgileSprint HumanInLoop

    registry.Register(project1) |> ignore
    let result = registry.Register(project2)
    Assert.True(Result.isError result)

[<Fact>]
let ``ProjectRegistry removes project`` () =
    let registry = ProjectRegistry()

    let project =
        createProject "remove-test" "Remove Test" "/tmp" StandardSDLC Continuous

    registry.Register(project) |> ignore

    Assert.True(registry.Exists("remove-test"))
    let removed = registry.Remove("remove-test")
    Assert.True(removed)
    Assert.False(registry.Exists("remove-test"))

[<Fact>]
let ``ProjectRegistry lists all projects`` () =
    let registry = ProjectRegistry()

    registry.Register(createProject "p1" "Project 1" "/tmp/1" StandardSDLC Continuous)
    |> ignore

    registry.Register(createProject "p2" "Project 2" "/tmp/2" AgileSprint HumanInLoop)
    |> ignore

    let all = registry.List()
    Assert.Equal(2, all.Length)

[<Fact>]
let ``ProjectRegistry manages project state`` () =
    let registry = ProjectRegistry()
    let project = createProject "state-test" "State Test" "/tmp" StandardSDLC Continuous
    registry.Register(project) |> ignore

    let state = registry.GetState("state-test")
    Assert.True(state.IsSome)
    Assert.True(state.Value.CurrentStage.IsNone)
