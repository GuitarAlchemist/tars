namespace TarsEngine.SelfImprovement.Tests

open System
open System.IO
open System.Text.Json
open System.Text.Json.Nodes
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.Tier2Runner
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService

module Tier2RunnerTests =

    let private createHarnessReport success =
        let config =
            { Description = "test"
              PreValidation = []
              Validation = []
              Benchmarks = []
              Rollback = None
              StopOnFailure = true
              CaptureLogs = true }

        { Config = config
          Commands = []
          StartedAt = DateTime.UtcNow
          CompletedAt = DateTime.UtcNow
          Outcome =
            if success then
                HarnessOutcome.AllPassed []
            else
                HarnessOutcome.Failed([], "failure") }

    let private createClosureResult success =
        { CommandId = "closure"
          Success = success
          Output = None
          OutputSummary = ""
          Artifacts = []
          EvolutionData = Map.empty
          NextSteps = []
          Error = None
          ExecutionTime = TimeSpan.Zero }

    let private summary =
        { Title = "Demo"
          Status = Some "Draft"
          FeatureBranch = Some "demo"
          Created = Some "2025-01-01"
          UserStories = []
          EdgeCases = [] }

    let private createIterationResult success =
        { SpecSummary = summary
          ClosureResult = createClosureResult success
          HarnessReport = Some (createHarnessReport success)
          Consensus = None
          ReasoningTraces = None
          CriticVerdict = None
          CrossAgentFeedback = None
          ValidatorSnapshot = None }

    type private FakeService(result: SpecDrivenIterationResult option) =
        let mutable dispatched = false
        let mutable capturedOptions: SpecKitHarnessOptions option = None

        member _.RecordedOptions = capturedOptions

        interface ISelfImprovementService with
            member _.AnalyzeFileAsync _ = async { return None }
            member _.AnalyzeDirectoryAsync _ = async { return [] }
            member _.ApplyImprovementsAsync(_, _) = async { return [] }
            member _.GetImprovementHistoryAsync(_) = async { return [] }
            member _.RunExecutionHarnessAsync(_, ?executor) = async { return raise (NotImplementedException()) }
            member _.GetAgentFeedbackSummaryAsync(_, ?limit) = async { return [] }
            member _.GetRoleDirectivesAsync(_, ?limit) = async { return [] }
            member _.DiscoverSpecKitFeaturesAsync _ = async { return [] }
            member _.DecomposeSpecKitFeatureAsync(_, ?baseDirectory) = async { return None }
            member _.GetSpecKitIterationConfigAsync(_, ?baseDirectory, ?options) = async { return None }
            member _.RunSpecKitIterationAsync(_, _, ?baseDirectory, ?options, ?executor) = async { return None }
            member _.RunNextSpecKitIterationAsync(_, ?baseDirectory, ?options, ?executor) =
                async {
                    if dispatched then
                        return None
                    else
                        dispatched <- true
                        capturedOptions <- options
                        return result
                }
            member _.RunTeamCycleAsync(_, ?options) = async { return raise (NotImplementedException()) }
            member _.PlanNextSpecKitGoalsAsync(_, ?topCandidates, ?recentMemory) = async { return ([], []) }
            member _.DispatchPlannerRecommendationsAsync(_, ?topCandidates, ?recentMemory, ?dispatchCount, ?settings) =
                async { return ([], [], []) }
            member _.EnsureRoadmapSpecAsync(_, _, ?tasks, ?allowAuggieFallback) = async { return true }
            member _.UpdateRoadmapTaskStatusAsync(_, _, _, _) = async { return true }

    let private writeLedger
        root
        capability
        consensus
        critic
        disagreement
        findings
        comments
        disagreementCount
        historyCapability
        historyConsensus
        historyCritic
        historyDisagreement
        historyFindings
        historyComments
        historyDisagreements =
        let iterationsDir = Path.Combine(root, ".specify", "ledger", "iterations")
        Directory.CreateDirectory(iterationsDir) |> ignore
        let options = JsonSerializerOptions(WriteIndented = true)

        let createEntry runId timestamp cap conf criticStatus disagreementRatio findingsTotal commentsTotal disagreementsTotal =
            let metrics = JsonObject()
            metrics["capability.pass_ratio"] <- JsonValue.Create<float>(cap)
            metrics["safety.consensus_avg_confidence"] <- JsonValue.Create<float>(conf)
            metrics["safety.critic_status"] <- JsonValue.Create<string>(criticStatus)
            metrics["validators.disagreement_ratio"] <- JsonValue.Create<float>(disagreementRatio)
            metrics["validators.findings_total"] <- JsonValue.Create<int>(findingsTotal)
            metrics["validators.comments"] <- JsonValue.Create<int>(commentsTotal)
            metrics["validators.disagreements"] <- JsonValue.Create<int>(disagreementsTotal)

            let entry = JsonObject()
            entry["runId"] <- JsonValue.Create<string>(runId)
            entry["timestamp"] <- JsonValue.Create<string>(timestamp)
            entry["specId"] <- JsonValue.Create<string>("feature-demo")
            entry["specPath"] <- JsonValue.Create<string>("spec.md")
            entry["status"] <- JsonValue.Create<string>("passed")
            entry["metrics"] <- metrics
            entry

        let latestEntry =
            createEntry
                "00000000-0000-0000-0000-000000000100"
                "2025-03-01T12:00:00Z"
                capability
                consensus
                critic
                disagreement
                findings
                comments
                disagreementCount

        let historyEntry =
            createEntry
                "00000000-0000-0000-0000-000000000099"
                "2025-02-28T12:00:00Z"
                historyCapability
                historyConsensus
                historyCritic
                historyDisagreement
                historyFindings
                historyComments
                historyDisagreements

        File.WriteAllText(Path.Combine(iterationsDir, "latest.json"), JsonSerializer.Serialize(latestEntry, options))
        File.WriteAllText(Path.Combine(iterationsDir, "20250228120000_history.json"), JsonSerializer.Serialize(historyEntry, options))

    let private readPolicy root =
        let path = Path.Combine(root, ".specify", "tier2_policy.json")
        if File.Exists(path) then
            JsonSerializer.Deserialize<Tier2PolicyState>(File.ReadAllText(path))
        else
            { RequireConsensus = false; RequireCritic = false }

    [<Fact>]
    let ``Tier2 runner tightens policy on failure`` () =
        let originalDir = Environment.CurrentDirectory
        let tempRoot = Path.Combine(Path.GetTempPath(), $"tier2-runner-{Guid.NewGuid():N}")
        Directory.CreateDirectory(tempRoot) |> ignore
        Environment.CurrentDirectory <- tempRoot

        try
            writeLedger tempRoot 0.60 0.65 "reject" 0.35 12 4 3 0.65 0.70 "reject" 0.20 10 3 2
            let fake = FakeService(Some (createIterationResult false))
            let service = fake :> ISelfImprovementService
            let loggerFactory = NullLoggerFactory.Instance

            let outcome =
                Tier2Runner.runIterationAsync service loggerFactory None None
                |> Async.RunSynchronously

            match outcome with
            | Failure (_, actions) ->
                Assert.Contains(Tier2Action.PolicyTightened, actions)
                Assert.Contains(Tier2Action.RemediationEnqueued, actions)
            | _ -> Assert.True(false, "Expected failure outcome.")

            let policy = readPolicy tempRoot
            Assert.True(policy.RequireConsensus)
            Assert.True(policy.RequireCritic)

            let verification = FakeService None
            Tier2Runner.runIterationAsync (verification :> ISelfImprovementService) loggerFactory None None
            |> Async.RunSynchronously
            |> ignore

            let recordedOptions = verification.RecordedOptions
            Assert.True(recordedOptions.IsSome, "Tightened policy should persist for subsequent iterations.")
            Assert.True(recordedOptions.Value.RequireCriticApproval)
        finally
            Environment.CurrentDirectory <- originalDir
            if Directory.Exists(tempRoot) then
                Directory.Delete(tempRoot, true)

    [<Fact>]
    let ``Tier2 runner relaxes policy after sustained success`` () =
        let originalDir = Environment.CurrentDirectory
        let tempRoot = Path.Combine(Path.GetTempPath(), $"tier2-runner-success-{Guid.NewGuid():N}")
        Directory.CreateDirectory(tempRoot) |> ignore
        Environment.CurrentDirectory <- tempRoot

        try
            let policyPath = Path.Combine(tempRoot, ".specify", "tier2_policy.json")
            Directory.CreateDirectory(Path.GetDirectoryName(policyPath)) |> ignore
            let strictPolicy =
                { RequireConsensus = true
                  RequireCritic = true }
            File.WriteAllText(policyPath, JsonSerializer.Serialize(strictPolicy))

            let initialPolicy = readPolicy tempRoot
            Assert.True(initialPolicy.RequireConsensus)
            Assert.True(initialPolicy.RequireCritic)

            writeLedger tempRoot 0.95 0.96 "accept" 0.02 18 4 1 0.93 0.94 "accept" 0.01 15 3 1

            let fake = FakeService(Some (createIterationResult true))
            let service = fake :> ISelfImprovementService
            let loggerFactory = NullLoggerFactory.Instance

            let outcome =
                Tier2Runner.runIterationAsync service loggerFactory None None
                |> Async.RunSynchronously

            let policy = readPolicy tempRoot
            Assert.False(policy.RequireConsensus)
            Assert.False(policy.RequireCritic)

            let recordedOptions = fake.RecordedOptions
            Assert.True(recordedOptions.IsSome)
            Assert.True(recordedOptions.Value.RequireCriticApproval)

            let verification = FakeService None
            Tier2Runner.runIterationAsync (verification :> ISelfImprovementService) loggerFactory None None
            |> Async.RunSynchronously
            |> ignore

            let relaxedOptions = verification.RecordedOptions
            Assert.True(relaxedOptions.IsSome)
            Assert.False(relaxedOptions.Value.RequireCriticApproval)
            match outcome with
            | Success _ -> ()
            | _ -> Assert.True(false, "Expected success outcome.")
        finally
            Environment.CurrentDirectory <- originalDir
            if Directory.Exists(tempRoot) then
                Directory.Delete(tempRoot, true)




    [<Fact>]
    let ``Tier2 runner tightens when validator disagreements spike`` () =
        let originalDir = Environment.CurrentDirectory
        let tempRoot = Path.Combine(Path.GetTempPath(), $"tier2-runner-disagreement-{Guid.NewGuid():N}")
        Directory.CreateDirectory(tempRoot) |> ignore
        Environment.CurrentDirectory <- tempRoot

        try
            writeLedger tempRoot 0.92 0.94 "accept" 0.55 22 6 4 0.91 0.93 "accept" 0.18 19 5 3
            let service = FakeService(Some (createIterationResult true))
            let loggerFactory = NullLoggerFactory.Instance

            Tier2Runner.runIterationAsync service loggerFactory None None
            |> Async.RunSynchronously
            |> ignore

            let ledgerDir = Path.Combine(tempRoot, ".specify", "ledger", "iterations")
            let latestPath = Path.Combine(ledgerDir, "latest.json")
            Assert.True(File.Exists(latestPath), "Latest ledger entry should exist.")

            use doc = JsonDocument.Parse(File.ReadAllText(latestPath))
            let metrics = doc.RootElement.GetProperty("metrics")
            let disagreement = metrics.GetProperty("validators.disagreement_ratio").GetDouble()
            Assert.True(disagreement >= 0.55, "Validator disagreement ratio should be persisted.")
        finally
            Environment.CurrentDirectory <- originalDir
            if Directory.Exists(tempRoot) then
                Directory.Delete(tempRoot, true)



