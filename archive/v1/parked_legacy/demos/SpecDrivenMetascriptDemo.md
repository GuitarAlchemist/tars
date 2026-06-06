# Spec-Driven Metascript Demo

This walk-through shows how to run a metascript closure driven by a Spec Kit contract and capture spec-aware metrics via `MetascriptClosureIntegrationService`.

## 1. Prepare a Spec Kit file

```markdown
# Feature Specification: Tiered Dynamic Evolution

**Feature Branch**: `421-dynamic-tier`
**Created**: 2025-03-15
**Status**: Draft

### User Story 1 - Deterministic spawn plan (Priority: P1)

**Acceptance Scenarios**:
1. Given the spec, When the dynamic closure runs, Then at least two agent spawns are recorded.

### Edge Cases
- What happens when the metascript inline grammar is malformed?
```

Save the file as `specs/demos/tiered_dynamic_spec.md`.

## 2. Execute the metascript command

```bash
dotnet build -c Release TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj
dotnet fsi demos/SpecDrivenMetascriptRunner.fsx
```

`SpecDrivenMetascriptRunner.fsx` constructs the command:

```fsharp
let commandLine =
    sprintf
        "CLOSURE_CREATE DYNAMIC_METASCRIPT \"TieredDynamic\" spec=\"%s\""
        specPath

let command =
    match service.ParseClosureCommand(commandLine) with
    | Some cmd -> cmd
    | None -> failwith "Failed to parse spec-driven command."

let result =
    service.ExecuteClosureCommand(command)
    |> Async.RunSynchronously
```

## 3. Inspect the spec-aware results

```text
Output Summary:
Dynamic metascript • name=tiered-dynamic • rules=7 • depth=3 • source=.../tiered_dynamic_spec.md • metrics=[innovation:0.85, stability:0.72]

Evolution Data:
- spec.contract.title = Tiered Dynamic Evolution
- spec.contract.user_story_count = 1
- spec.contract.acceptance_count = 1
- spec.contract.p1_story_count = 1
- spec.contract.status = Draft
- spec.contract.path = .../tiered_dynamic_spec.md

Artifacts:
- Dynamic metascript summary recorded (2 spawn, 2 connections)
- Spec contract "Tiered Dynamic Evolution" captured (1 stories, 1 acceptance scenarios)
```

The runner also prints the enriched next steps:

```text
- Validate acceptance scenarios defined in spec contract.
- Validate dynamic closure grammar against production constraints.
- Run rehearsal execution in sandboxed environment.
```

## 4. Extend or automate

- Add more user stories or edge cases to the Spec Kit file and rerun the script to watch the metrics update.
- Feed the command into the autonomous execution harness to enforce spec compliance before promoting dynamic closures.

## 5. Automate the loop

For a fully autonomous flow, call `AutonomousSpecHarness.runIteration`:

```fsharp
open Microsoft.Extensions.Logging.Abstractions
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness

let config =
    { SpecPath = specPath
      Description = Some "Tiered dynamic iteration"
      PatchCommands = [
        { Name = "apply_patch"
          Executable = "pwsh"
          Arguments = "-File scripts\\apply_patch.ps1"
          WorkingDirectory = None
          Timeout = None
          Environment = Map.empty } ]
      ValidationCommands = [
        { Name = "run_tests"
          Executable = "dotnet"
          Arguments = "test TarsEngine.FSharp.Core/TarsEngine.FSharp.Core.fsproj -c Release"
          WorkingDirectory = None
          Timeout = None
          Environment = Map.empty } ]
      BenchmarkCommands = []
      RollbackCommands = [
        { Name = "rollback"
          Executable = "pwsh"
          Arguments = "-File scripts\\rollback_changes.ps1"
          WorkingDirectory = None
          Timeout = None
          Environment = Map.empty } ]
      StopOnFailure = true
      CaptureLogs = true
      ConsensusRule = None
      AgentResultProvider = None
      RequireConsensusForExecution = false
      ReasoningTraceProvider = None
      ReasoningCritic = None
      RequireCriticApproval = false
      ReasoningFeedbackSink = None }

let result =
    runIteration (NullLoggerFactory.Instance :> _) config
    |> Async.RunSynchronously
```

If any command fails, the harness automatically runs the rollback commands and returns a detailed `HarnessReport` for auditing.

This demo combines the Spec Kit parser, metascript integration service, and dynamic closure factory to keep autonomous evolution aligned with formal specs.
