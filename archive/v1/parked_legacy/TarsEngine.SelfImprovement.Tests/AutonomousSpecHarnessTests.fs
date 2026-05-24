module TarsEngine.SelfImprovement.Tests.AutonomousSpecHarnessTests

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.AdaptivePolicy
open TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory
open TarsEngine.FSharp.SelfImprovement.CrossAgentFeedback
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService
open TarsEngine.FSharp.Core.Services.ReasoningTrace

type RecordingExecutor(responses: Map<string, int>) =
    let executed = ResizeArray<CommandSpec>()

    member _.Executed = executed |> Seq.toList

    interface ICommandExecutor with
        member _.RunCommandAsync(command: CommandSpec) =
            executed.Add(command)
            let exitCode = responses |> Map.tryFind command.Name |> Option.defaultValue 0
            let now = DateTime.UtcNow
            async {
                return { Command = command
                         ExitCode = exitCode
                         Duration = TimeSpan.Zero
                         StandardOutput = ""
                         StandardError = ""
                         StartedAt = now
                         CompletedAt = now }
            }

module private Helpers =

    let specContent =
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

    let invalidSpecContent =
        """
# Feature Specification: Invalid Harness Demo

**Feature Branch**: `demo-invalid`
**Created**: 2025-03-15
**Status**: Draft

```metascript
SPAWN ONLY_TWO_ARGS
```
"""

    let writeTempSpec (content: string) =
        let path = Path.Combine(Path.GetTempPath(), $"spec_{Guid.NewGuid():N}.md")
        File.WriteAllText(path, content.Trim())
        path

    let command name =
        { Name = name
          Executable = name
          Arguments = ""
          WorkingDirectory = None
          Timeout = None
          Environment = Map.empty }

    let agentResult id role outcome confidence =
        { AgentId = id
          Role = role
          Outcome = outcome
          Confidence = confidence
          Notes = None
          ProducedAt = DateTime.UtcNow }

    let consensusRule =
        { MinimumPassCount = 2
          RequiredRoles = [ Reasoner; Reviewer ]
          AllowNeedsReview = false
          MinimumConfidence = Some 0.6
          MaxFailureCount = Some 0 }

    let policySnapshot
        (requireConsensus: bool)
        (requireCritic: bool)
        (stopOnFailure: bool)
        (captureLogs: bool)
        (minimumPass: int option)
        (allowNeedsReview: bool option)
        (minimumConfidence: float option)
        (maxFailures: int option) =
        { requireConsensus = requireConsensus
          requireCritic = requireCritic
          stopOnFailure = stopOnFailure
          captureLogs = captureLogs
          patchCommands = 1
          validationCommands = 1
          benchmarkCommands = 0
          hasAgentProvider = false
          hasTraceProvider = false
          hasFeedbackSink = false
          hasReasoningCritic = requireCritic
          consensusRule =
            if requireConsensus
               || minimumPass.IsSome
               || allowNeedsReview.IsSome
               || minimumConfidence.IsSome
               || maxFailures.IsSome then
                Some
                    { minimumPassCount = minimumPass |> Option.defaultValue 2
                      requiredRoles = [ "Reasoner"; "Reviewer" ]
                      allowNeedsReview = allowNeedsReview |> Option.defaultValue false
                      minimumConfidence = minimumConfidence
                      maxFailureCount = maxFailures }
            else
                None }

    let memoryEntry (snapshot: PolicySnapshot) success criticStatus feedback =
        { runId = Guid.NewGuid()
          specId = "adaptive"
          specPath = "spec.md"
          description = None
          timestamp = DateTime.UtcNow
          consensus =
            Some
                { status = if success then "passed" else "failed"
                  message = None
                  agents = [] }
          critic =
            Some
                { status = criticStatus
                  message = None
                  source = None
                  threshold = None
                  sampleSize = None
                  indicators = [] }
          reasoning = []
          policyBefore = snapshot
          policyAfter = snapshot
          policyChanges = []
          inferenceTelemetry = Dictionary<string, obj>()
          agentFeedback = feedback
          validatorFindings = []
          validatorDisagreements = []
          harness =
            Some
                { status = if success then "passed" else "failed"
                  failureReason = if success then None else Some "failure"
                  failedCommandCount = if success then Some 0 else Some 1 } }

open Helpers

[<Fact>]
let ``discoverPatchArtifactsForTesting detects absolute and relative patches`` () =
    let absolutePatch = Path.Combine(Path.GetTempPath(), $"test_patch_{Guid.NewGuid():N}.patch")
    File.WriteAllText(absolutePatch, "--- dummy patch ---")

    let relativeFile = $"test_patch_{Guid.NewGuid():N}.patch"
    File.WriteAllText(relativeFile, "--- dummy patch ---")

    let artifacts =
        [ $"Generated patch stored at {absolutePatch}"
          $"Patched files -> {relativeFile}"
          "No patch here" ]

    try
        let discovered = discoverPatchArtifactsForTesting artifacts
        Assert.Contains(absolutePatch, discovered)
        let expectedRelative =
            Path.Combine(Environment.CurrentDirectory, relativeFile)
            |> Path.GetFullPath
        Assert.Contains(expectedRelative, discovered)
    finally
        if File.Exists(absolutePatch) then File.Delete(absolutePatch)
        if File.Exists(relativeFile) then File.Delete(relativeFile)

[<Fact>]
let ``consensus evaluation passes when quorum met`` () =
    let results =
        [ agentResult "reasoner" Reasoner ValidationOutcome.Pass (Some 0.8)
          agentResult "reviewer" Reviewer ValidationOutcome.Pass (Some 0.92) ]

    match evaluate consensusRule results with
    | ConsensusPassed _ -> ()
    | outcome -> Assert.True(false, $"Unexpected consensus outcome: %A{outcome}")

[<Fact>]
let ``consensus evaluation fails when required role missing`` () =
    let results =
        [ agentResult "reviewer" Reviewer ValidationOutcome.Pass (Some 0.95) ]

    match evaluate consensusRule results with
    | ConsensusFailed _ -> ()
    | outcome -> Assert.True(false, $"Expected failure, got {outcome}.")

[<Fact>]
let ``runIteration executes harness when closure succeeds`` () =
    let specPath = writeTempSpec specContent
    let config =
        { SpecPath = specPath
          Description = Some "Demo iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = Some consensusRule
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let executor = RecordingExecutor(Map.empty)
    let result =
        runIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.ClosureResult.Success)
        Assert.True(result.HarnessReport.IsSome)
        let executed = executor.Executed |> List.map (fun c -> c.Name)
        Assert.Equal<string list>(["apply_patch"; "run_tests"], executed)
        match result.HarnessReport with
        | Some report ->
            match report.Outcome with
            | AllPassed results -> Assert.Equal(2, results.Length)
            | _ -> Assert.True(false, "Harness should have succeeded.")
        | None -> Assert.True(false, "Harness report missing.")
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``reasoning feedback sink captures traces`` () =
    let specPath = writeTempSpec specContent
    let executor = RecordingExecutor(Map.empty)
    let traces _ : ReasoningTrace list =
        [ { CorrelationId = "run-feedback"
            Summary = Some "Feedback summary"
            Events =
                [ { AgentId = "reasoner"
                    Step = "analysis"
                    Message = "All checks passed."
                    Score = Some 0.95
                    Metadata = Map.empty
                    CreatedAt = DateTime.UtcNow } ] } ]

    let critic (traces: ReasoningTrace list) =
        if traces.IsEmpty then CriticVerdict.NeedsReview "Empty trace."
        else CriticVerdict.Accept

    let recorded = ref ([] : (ReasoningTrace list * CriticVerdict option) list)
    let sink : ReasoningTrace list -> CriticVerdict option -> unit =
        fun traces verdict ->
            recorded := (traces, verdict) :: !recorded

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Feedback iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = Some traces
          ReasoningCritic = Some critic
          RequireCriticApproval = false
          ReasoningFeedbackSink = Some sink
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let result =
        runIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.HarnessReport.IsSome)
        Assert.Equal(1, (!recorded).Length)
        let traces, verdict = (!recorded) |> List.head
        Assert.Equal(1, traces.Length)
        Assert.Equal(Some CriticVerdict.Accept, verdict)
    finally
        if File.Exists(specPath) then File.Delete(specPath)
[<Fact>]
let ``runIteration halts when consensus fails`` () =
    let specPath = writeTempSpec specContent
    let executor = RecordingExecutor(Map.empty)
    let provider (_: MetascriptClosureResult) =
        [ agentResult "reasoner" Reasoner ValidationOutcome.Pass (Some 0.9)
          agentResult "reviewer" Reviewer ValidationOutcome.Fail (Some 0.8) ]

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Consensus-gated iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = Some consensusRule
          AgentResultProvider = Some provider
          RequireConsensusForExecution = true
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let result =
        runIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.ClosureResult.Success)
        Assert.True(result.HarnessReport.IsNone)
        Assert.Empty(executor.Executed)
        match result.Consensus with
        | Some (ConsensusFailed _) -> ()
        | other -> Assert.True(false, $"Expected failed consensus, got {other}.")
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``runIteration triggers rollback when validation fails`` () =
    let specPath = writeTempSpec specContent
    let responses = Map.ofList [ "run_tests", 1 ]
    let executor = RecordingExecutor(responses)

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = None
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = [ command "rollback_changes" ]
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let result =
        runIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.ClosureResult.Success)
        Assert.True(result.HarnessReport.IsSome)
        let executed = executor.Executed |> List.map (fun c -> c.Name)
        Assert.Equal<string list>(["apply_patch"; "run_tests"; "rollback_changes"], executed)
        match result.HarnessReport with
        | Some report ->
            match report.Outcome with
            | Failed(_, reason) ->
                Assert.Contains("run_tests", reason)
            | _ -> Assert.True(false, "Harness should have failed.")
        | None -> Assert.True(false, "Harness report missing.")
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``runIteration returns closure failure without running harness`` () =
    let specPath = writeTempSpec invalidSpecContent
    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = None
          PatchCommands = []
          ValidationCommands = []
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let result =
        runIteration (NullLoggerFactory.Instance :> ILoggerFactory) config
        |> Async.RunSynchronously

    try
        Assert.False(result.ClosureResult.Success)
        Assert.True(result.HarnessReport.IsNone)
        Assert.True(result.ReasoningTraces.IsNone)
        Assert.True(result.CriticVerdict.IsNone)
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``runIteration halts when critic rejects`` () =
    let specPath = writeTempSpec specContent
    let executor = RecordingExecutor(Map.empty)
    let traces _ : ReasoningTrace list =
        [ { CorrelationId = "run-1"
            Summary = Some "Reasoning summary"
            Events =
                [ { AgentId = "critic-agent"
                    Step = "analysis"
                    Message = "Detected specification drift."
                    Score = Some 0.2
                    Metadata = Map.empty
                    CreatedAt = DateTime.UtcNow } ] } ]

    let critic traces =
        match traces with
        | _ -> CriticVerdict.Reject "Spec deviation detected."

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Critic-gated iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = Some traces
          ReasoningCritic = Some critic
          RequireCriticApproval = true
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let result =
        runIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.ClosureResult.Success)
        Assert.True(result.HarnessReport.IsNone)
        Assert.Empty(executor.Executed)
        Assert.True(result.ReasoningTraces.IsSome)
        match result.CriticVerdict with
        | Some (CriticVerdict.Reject _) -> ()
        | other -> Assert.True(false, $"Expected critic rejection, got {other}.")
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``runAdaptiveIteration promotes stricter consensus`` () =
    let specPath = writeTempSpec specContent
    let executor = RecordingExecutor(Map.empty)
    let provider (_: MetascriptClosureResult) =
        [ agentResult "reasoner" Reasoner ValidationOutcome.Pass (Some 0.9)
          agentResult "reviewer" Reviewer ValidationOutcome.Fail (Some 0.8) ]

    let rule = { MinimumPassCount = 2; RequiredRoles = [ Reasoner; Reviewer ]; AllowNeedsReview = false; MinimumConfidence = Some 0.6; MaxFailureCount = Some 0 }

    let rule = { MinimumPassCount = 2; RequiredRoles = [ Reasoner; Reviewer ]; AllowNeedsReview = false; MinimumConfidence = Some 0.6; MaxFailureCount = Some 0 }

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Adaptive consensus iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = Some rule
          AgentResultProvider = Some provider
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let history = [ { Consensus = Some(ConsensusFailed ([], "failure")); CriticVerdict = None } ]

    let result, updated =
        TarsEngine.FSharp.SelfImprovement.AdaptivePolicy.runAdaptiveIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) history config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.Consensus.IsSome)
        Assert.True(updated.RequireConsensusForExecution)
        Assert.True(updated.ConsensusRule.IsSome)
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``runAdaptiveIteration enables critic approval`` () =
    let specPath = writeTempSpec specContent
    let executor = RecordingExecutor(Map.empty)
    let traces (_: MetascriptClosureResult) : ReasoningTrace list =
        [ { CorrelationId = "adaptive"
            Summary = Some "Reject"
            Events =
                [ { AgentId = "critic"
                    Step = "analysis"
                    Message = "Rejecting"
                    Score = Some 0.1
                    Metadata = Map.empty
                    CreatedAt = DateTime.UtcNow } ] } ]

    let critic (traces: ReasoningTrace list) =
        if traces.IsEmpty then CriticVerdict.NeedsReview "empty" else CriticVerdict.Reject "bad"

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Adaptive critic iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = Some traces
          ReasoningCritic = Some critic
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let history = [ { Consensus = None; CriticVerdict = Some (CriticVerdict.Reject "bad") } ]

    let result, updated =
        TarsEngine.FSharp.SelfImprovement.AdaptivePolicy.runAdaptiveIterationWithExecutor (NullLoggerFactory.Instance :> ILoggerFactory) history config (executor :> ICommandExecutor)
        |> Async.RunSynchronously

    try
        Assert.True(result.CriticVerdict.IsSome)
        Assert.True(updated.RequireCriticApproval)
    finally
        if File.Exists(specPath) then File.Delete(specPath)

[<Fact>]
let ``adaptive iteration with history persists feedback`` () =
    let specPath = writeTempSpec specContent
    let historyPath = Path.Combine(Path.GetTempPath(), $"history_{Guid.NewGuid():N}.json")
    let memoryPath = Path.Combine(Directory.GetCurrentDirectory(), "output", $"adaptive_memory_{Guid.NewGuid():N}.jsonl")
    let executor = RecordingExecutor(Map.empty)
    let provider (_: MetascriptClosureResult) =
        [ agentResult "reasoner" Reasoner ValidationOutcome.Pass (Some 0.9)
          agentResult "reviewer" Reviewer ValidationOutcome.Fail (Some 0.8) ]

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Adaptive history iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = Some provider
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = Some memoryPath }

    try
        let result, updated =
            TarsEngine.FSharp.SelfImprovement.AdaptivePolicy.runAdaptiveIterationWithHistoryAndExecutor
                (NullLoggerFactory.Instance :> ILoggerFactory)
                historyPath
                config
                (executor :> ICommandExecutor)
            |> Async.RunSynchronously

        let persistedJson = File.ReadAllText(historyPath)
        Assert.False(String.IsNullOrWhiteSpace(persistedJson))
        use historyDoc = JsonDocument.Parse(persistedJson)
        Assert.True(historyDoc.RootElement.GetArrayLength() >= 1)

        let lastWrite = TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.getLastWritePath()
        if lastWrite.IsNone then failwithf $"Expected adaptive memory write path, got %A{lastWrite}"
        let resolvedMemory = Path.GetFullPath(memoryPath)
        Assert.Equal(resolvedMemory, lastWrite |> Option.map Path.GetFullPath |> Option.defaultValue "")
        Assert.True(File.Exists(resolvedMemory))
        let memoryEntries =
            File.ReadAllLines(resolvedMemory)
            |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
        Assert.NotEmpty(memoryEntries)

        use doc = JsonDocument.Parse(memoryEntries |> Array.last)
        let root = doc.RootElement
        let mutable consensusProp = Unchecked.defaultof<JsonElement>
        let hasConsensus = root.TryGetProperty("consensus", &consensusProp)
        if hasConsensus && consensusProp.ValueKind <> JsonValueKind.Null then
            Assert.Equal("failed", consensusProp.GetProperty("status").GetString())
        let mutable policyProp = Unchecked.defaultof<JsonElement>
        let hasPolicy = root.TryGetProperty("policyChanges", &policyProp)
        Assert.True(hasPolicy)
        Assert.Equal(JsonValueKind.Array, policyProp.ValueKind)

        let promptPath = TarsEngine.FSharp.SelfImprovement.MetaReasoningCritic.defaultPromptPath
        Assert.True(File.Exists(promptPath))
        let promptPayload = File.ReadAllText(promptPath)
        Assert.Contains("scoreThreshold", promptPayload)
        Assert.Contains("adaptive_prompts", Path.GetFileNameWithoutExtension(promptPath).ToLowerInvariant())
    finally
        if File.Exists(specPath) then File.Delete(specPath)
        if File.Exists(historyPath) then File.Delete(historyPath)
        let resolvedMemory = Path.GetFullPath(memoryPath)
        if File.Exists(resolvedMemory) then File.Delete(resolvedMemory)
        let promptPath = TarsEngine.FSharp.SelfImprovement.MetaReasoningCritic.defaultPromptPath
        if File.Exists(promptPath) then File.Delete(promptPath)

[<Fact>]
let ``policy tuning favors genomes with higher pass rate`` () =
    let strictSnapshot =
        policySnapshot
            true
            false
            true
            false
            (Some 2)
            (Some false)
            (Some 0.7)
            (Some 0)

    let lenientSnapshot =
        policySnapshot
            false
            false
            true
            false
            None
            None
            None
            None

    let entries =
        [ memoryEntry strictSnapshot true "accept" []
          memoryEntry strictSnapshot true "accept" []
          memoryEntry lenientSnapshot false "reject" []
          memoryEntry lenientSnapshot false "reject" [] ]

    let baseConfig : SpecDrivenIterationConfig =
        { SpecPath = "adaptive.md"
          Description = None
          PatchCommands = []
          ValidationCommands = []
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = false
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = None
          AdaptiveMemoryPath = None }

    let tuned = TarsEngine.FSharp.SelfImprovement.AdaptivePolicy.Internal.tuneWithEntries entries baseConfig

    Assert.True(tuned.RequireConsensusForExecution)
    Assert.True(tuned.ConsensusRule.IsSome)
    tuned.ConsensusRule
    |> Option.iter (fun rule ->
        Assert.Equal(2, rule.MinimumPassCount)
        Assert.False(rule.AllowNeedsReview))
    Assert.False(tuned.RequireCriticApproval)

[<Fact>]
let ``reviewer escalation forces consensus and critic approval`` () =
    let specPath = writeTempSpec specContent
    let historyPath = Path.Combine(Path.GetTempPath(), $"history_{Guid.NewGuid():N}.json")
    let memoryPath = Path.Combine(Path.GetTempPath(), $"memory_{Guid.NewGuid():N}.jsonl")
    let executor = RecordingExecutor(Map.empty)

    let escalationFeedback : AgentFeedback =
        { AgentId = "auditor"
          Role = AgentRole.Reviewer
          Verdict = FeedbackVerdict.Escalate "policy risk"
          Confidence = Some 0.95
          Notes = Some "Escalating due to unresolved safety concerns"
          SuggestedActions = [ "request_manual_review" ]
          RecordedAt = DateTime.UtcNow }

    let config : SpecDrivenIterationConfig =
        { SpecPath = specPath
          Description = Some "Reviewer escalation iteration"
          PatchCommands = [ command "apply_patch" ]
          ValidationCommands = [ command "run_tests" ]
          BenchmarkCommands = []
          RollbackCommands = []
          StopOnFailure = true
          CaptureLogs = false
          AutoApplyPatchArtifacts = true
          ConsensusRule = None
          AgentResultProvider = None
          RequireConsensusForExecution = false
          ReasoningTraceProvider = None
          ReasoningCritic = None
          RequireCriticApproval = false
          ReasoningFeedbackSink = None
          AgentFeedbackProvider = Some(fun _ -> [ escalationFeedback ])
          AdaptiveMemoryPath = Some memoryPath }

    try
        let result, updated =
            TarsEngine.FSharp.SelfImprovement.AdaptivePolicy.runAdaptiveIterationWithHistoryAndExecutor
                (NullLoggerFactory.Instance :> ILoggerFactory)
                historyPath
                config
                (executor :> ICommandExecutor)
            |> Async.RunSynchronously

        Assert.True(updated.RequireConsensusForExecution)
        Assert.True(updated.RequireCriticApproval)
        Assert.True(updated.CaptureLogs)
        Assert.True(result.CrossAgentFeedback.IsSome)
        Assert.True((result.CrossAgentFeedback |> Option.defaultValue []) |> List.exists(fun fb -> fb.AgentId = "auditor"))

        let memoryEntries =
            File.ReadAllLines(memoryPath)
            |> Array.filter (fun line -> not (String.IsNullOrWhiteSpace(line)))
        Assert.NotEmpty(memoryEntries)

        let aggregates = TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory.summarizeFeedbackFile memoryPath (Some 8)
        Assert.True(aggregates |> List.exists (fun agg -> agg.role = "Reviewer" && agg.escalate > 0))

        let directives = TarsEngine.FSharp.SelfImprovement.AgentPolicyTuning.deriveRoleDirectives aggregates
        let reviewerDirective = directives |> List.find (fun d -> d.Role = "Reviewer")
        Assert.True(reviewerDirective.PromptDirectives |> List.exists (fun directive -> directive.Contains("Escalations detected")))
        Assert.Contains("agentFeedback", memoryEntries |> Array.last)
    finally
        if File.Exists(specPath) then File.Delete(specPath)
        if File.Exists(historyPath) then File.Delete(historyPath)
        if File.Exists(memoryPath) then File.Delete(memoryPath)
        let promptPath = TarsEngine.FSharp.SelfImprovement.MetaReasoningCritic.defaultPromptPath
        if File.Exists(promptPath) then File.Delete(promptPath)






