module TarsEngine.SelfImprovement.Tests.SpecKitWorkspaceTests

open System
open System.IO
open System.Net.Http
open System.Text.Json
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open TarsEngine.FSharp.SelfImprovement.CrossAgentFeedback
open TarsEngine.SelfImprovement.Tests.AutonomousSpecHarnessTests

let private sampleSpec =
    """
# Feature Specification: Intelligent Vector Store Enhancements

**Feature Branch**: `002-intelligent-vector-store`
**Created**: 2025-03-15
**Status**: Draft

### User Story 1 - Prioritized Write Buffer (Priority: P1)

**Acceptance Scenarios**:
1. Given the store receives concurrent writes, When the buffer is saturated, Then high priority segments persist without loss.

### Edge Cases
- Vector dimensions exceeding configured maximum.
- Buffer pressure during CUDA kernel reset.
"""

let private harnessSpec =
    """
# Feature Specification: Harness Demo

**Feature Branch**: `demo-branch`
**Created**: 2025-03-15
**Status**: Draft

### User Story 1 - Deterministic spawn plan (Priority: P1)

**Acceptance Scenarios**:
1. Given the spec, When the dynamic closure runs, Then at least two agent spawns are recorded.

### Edge Cases
- What happens when the metascript inline grammar is malformed?

```metascript
SPAWN QRE 2 HIERARCHICAL
SPAWN ML 1 FRACTAL
CONNECT leader agent-1 directive
CONNECT agent-1 agent-2 support
METRIC innovation 0.85
METRIC stability 0.72
REPEAT adaptive 3
```

```expectations
rules=7
max_depth=3
spawn_count=2
connection_count=2
pattern=adaptive
metric.innovation=0.85
metric.stability=0.72
```
"""

let private sampleTasks =
    """
## Phase 1: Setup

- [ ] T010 [P] Create vector buffer controller module
- [x] T011 [US1] Document buffer eviction strategy
- [ ] T012 [P1] [US1] Implement adaptive priority queue
"""

[<Fact>]
let ``discoverFeatures returns parsed feature with tasks`` () =
    let root = Path.Combine(Path.GetTempPath(), $"spec-kit-{Guid.NewGuid():N}")
    let featureDir = Path.Combine(root, "002-intelligent-vector-store")
    Directory.CreateDirectory(featureDir) |> ignore

    File.WriteAllText(Path.Combine(featureDir, "spec.md"), sampleSpec.Trim())
    File.WriteAllText(Path.Combine(featureDir, "plan.md"), "# Plan\n\n- Implement CUDA tuning")
    File.WriteAllText(Path.Combine(featureDir, "tasks.md"), sampleTasks.Trim())

    try
        let features = discoverFeatures (Some root)
        let feature = Assert.Single(features)

        Assert.Equal("002-intelligent-vector-store", feature.Id)
        Assert.Equal(featureDir, feature.Directory)
        Assert.True(feature.PlanPath.IsSome)
        Assert.True(feature.TasksPath.IsSome)
        Assert.Equal("Intelligent Vector Store Enhancements", feature.Summary.Title)

        let task =
            feature.Tasks
            |> List.find (fun t -> t.TaskId = Some "T012")

        Assert.Equal("todo", task.Status)
        Assert.Equal(Some "P1", task.Priority)
        Assert.Equal(Some "US1", task.StoryTag)
        Assert.Equal(Some "Phase 1: Setup", task.Phase)
    finally
        if Directory.Exists(root) then
            Directory.Delete(root, true)

[<Fact>]
let ``buildIterationConfig surfaces critic requirement`` () =
    let root = Path.Combine(Path.GetTempPath(), $"spec-kit-critic-{Guid.NewGuid():N}")
    let featureDir = Path.Combine(root, "critic-feature")
    Directory.CreateDirectory(featureDir) |> ignore

    File.WriteAllText(Path.Combine(featureDir, "spec.md"), harnessSpec.Trim())

    try
        let features = discoverFeatures (Some root)
        let feature = Assert.Single(features)

        let critic traces =
            traces
            |> List.collect (fun trace -> trace.Events)
            |> List.choose (fun evt -> evt.Score)
            |> function
                | [] -> CriticVerdict.NeedsReview "no scores"
                | scores when List.average scores >= 0.9 -> CriticVerdict.Accept
                | _ -> CriticVerdict.Reject "confidence below threshold"

        let options =
            { defaultHarnessOptions with
                RequireCriticApproval = true
                ReasoningCritic = Some critic }

        let config = buildIterationConfig feature (Some options)

        Assert.True(config.RequireCriticApproval)
        Assert.True(config.ReasoningCritic.IsSome)
        let verdict = config.ReasoningCritic.Value []
        match verdict with
        | CriticVerdict.NeedsReview _ -> ()
        | _ -> Assert.True(false, sprintf "Expected fallback needs review verdict, got %A" verdict)
    finally
        if Directory.Exists(root) then
            Directory.Delete(root, true)

[<Fact>]
let ``SelfImprovementService exposes Spec Kit config`` () =
    let root = Path.Combine(Path.GetTempPath(), $"spec-kit-service-{Guid.NewGuid():N}")
    let featureDir = Path.Combine(root, "010-service-test")
    Directory.CreateDirectory(featureDir) |> ignore

    File.WriteAllText(Path.Combine(featureDir, "spec.md"), sampleSpec.Trim())
    File.WriteAllText(Path.Combine(featureDir, "tasks.md"), sampleTasks.Trim())

    use loggerFactory = LoggerFactory.Create(fun _ -> ())
    use httpClient = new HttpClient()
    let logger = loggerFactory.CreateLogger<SelfImprovementService>()
    let service = new SelfImprovementService(httpClient, logger)

    try
        let configOpt =
            service.GetSpecKitIterationConfigAsync("010-service-test", baseDirectory = root)
            |> Async.RunSynchronously

        match configOpt with
        | None -> Assert.True(false, "Iteration configuration not found for Spec Kit feature.")
        | Some config ->
            Assert.EndsWith("spec.md", config.SpecPath)
            Assert.True(config.Description.IsSome)
    finally
        if Directory.Exists(root) then
            Directory.Delete(root, true)

[<Fact>]
let ``RunNextSpecKitIterationAsync executes highest priority Spec Kit task`` () =
    let root = Path.Combine(Path.GetTempPath(), $"spec-kit-next-{Guid.NewGuid():N}")
    Directory.CreateDirectory(root) |> ignore

    let writeFeature directoryId priority =
        let dir = Path.Combine(root, directoryId)
        Directory.CreateDirectory(dir) |> ignore

        let specContent =
            harnessSpec.Replace("Harness Demo", $"Spec {directoryId}")
                         .Replace("`demo-branch`", $"`{directoryId}`")

        let specPath = Path.Combine(dir, "spec.md")
        File.WriteAllText(specPath, specContent.Trim())
        Assert.Contains("```metascript", File.ReadAllText(specPath))

        let tasks =
            $"""
## Phase 1
- [ ] T{priority}00 [P{priority}] [US1] Execute priority {priority} work
"""
        File.WriteAllText(Path.Combine(dir, "tasks.md"), tasks.Trim())

    writeFeature "100-low" 3
    writeFeature "101-high" 1

    let originalMemoryPath = Environment.GetEnvironmentVariable("TARS_ADAPTIVE_MEMORY_PATH")
    let memoryPath = Path.Combine(root, "memory.jsonl")
    Environment.SetEnvironmentVariable("TARS_ADAPTIVE_MEMORY_PATH", memoryPath)

    let previousLastWrite = TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.getLastWritePath ()

    try
        use httpClient = new HttpClient()
        let logger = NullLoggerFactory.Instance.CreateLogger<SelfImprovementService>()
        let service = new SelfImprovementService(httpClient, logger)
        let executor = new RecordingExecutor(Map.empty)

        let harnessOptions =
            { SpecKitWorkspace.defaultHarnessOptions with
                PersistAdaptiveMemory = false
                Commands =
                    { SpecKitWorkspace.defaultHarnessCommands with
                        Validation =
                            [ { Name = "dotnet-test"
                                Executable = "dotnet"
                                Arguments = "test"
                                WorkingDirectory = None
                                Timeout = None
                                Environment = Map.empty } ] } }

        let result =
            service.RunNextSpecKitIterationAsync(
                        NullLoggerFactory.Instance,
                        baseDirectory = root,
                        options = harnessOptions,
                        executor = (executor :> ICommandExecutor))
            |> Async.RunSynchronously

        Assert.True(result.IsSome, "Expected Spec Kit iteration to run successfully.")
        let iteration = result.Value
        match iteration.Consensus with
        | Some (ConsensusPassed _) -> ()
        | other -> Assert.True(false, sprintf "Unexpected consensus outcome: %A" other)

        let recorded = executor.Executed
        Assert.Equal(1, recorded.Length)
        Assert.Equal("dotnet-test", recorded.Head.Name)

        Assert.False(File.Exists(memoryPath))
    finally
        Environment.SetEnvironmentVariable("TARS_ADAPTIVE_MEMORY_PATH", originalMemoryPath)
        if Directory.Exists(root) then
            Directory.Delete(root, true)
        TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.setLastWritePath previousLastWrite
