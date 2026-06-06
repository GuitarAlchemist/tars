namespace TarsEngine.SelfImprovement.Tests

open System
open Microsoft.Extensions.Logging.Abstractions
open Xunit
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.TeamOrchestrator
open TarsEngine.FSharp.SelfImprovement.AutonomousNextStepPlanner
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination
open TarsEngine.FSharp.SelfImprovement.ExecutionHarness
open TarsEngine.FSharp.SelfImprovement.SpecKitWorkspace
open TarsEngine.FSharp.Core.Specs
open TarsEngine.FSharp.Core.Services.CrossAgentValidation

module RT = TarsEngine.FSharp.Core.Services.ReasoningTrace

module TeamOrchestratorTests =

    let private sampleSummary =
        { Title = "GPU Optimisation"
          Status = Some "Draft"
          FeatureBranch = Some "feature/gpu"
          Created = Some "2025-01-01"
          UserStories = []
          EdgeCases = [] }

    let private sampleTask =
        { LineNumber = 1
          Phase = Some "Implementation"
          Status = "todo"
          TaskId = Some "T100"
          Priority = Some "P0"
          StoryTag = None
          Description = "Optimise CUDA kernels for top-k search."
          Raw = "- [ ] T100 [P0] Optimise CUDA kernels for top-k search." }

    let private sampleFeature =
        { Id = "feature-gpu"
          Directory = "."
          SpecPath = "spec.md"
          PlanPath = None
          TasksPath = None
          Summary = sampleSummary
          Tasks = [ sampleTask ] }

    let private plannerRecommendation =
        { Selection =
            { Feature = sampleFeature
              Task = sampleTask
              PriorityRank = 0 }
          Score = 2.5
          SimilarityScore = 0.82
          FailureSignal = 1
          PriorityWeight = 1.2
          Rationale = [ "High impact on GPU throughput"; "Aligns with Tier3 roadmap" ] }

    let private harnessReport =
        { Config =
            { Description = "CUDA validation"
              PreValidation = []
              Validation = []
              Benchmarks = []
              Rollback = None
              StopOnFailure = true
              CaptureLogs = true }
          Commands = []
          StartedAt = DateTime.UtcNow.AddMinutes(-3.0)
          CompletedAt = DateTime.UtcNow
          Outcome = HarnessOutcome.AllPassed [] }

    let private validatorSnapshot =
        let finding role outcome =
            { ValidatorFinding.FindingId = Guid.NewGuid()
              AgentId = $"agent:{role}"
              Role = role
              Outcome = outcome
              Confidence = Some 0.8
              Notes = None
              Target = { SpecId = "feature-gpu"; IterationId = Some(Guid.NewGuid()); Topic = Some "spec-kit" }
              RecordedAt = DateTime.UtcNow }

        { Findings =
            [ finding AgentRole.SafetyGovernor ValidationOutcome.Pass
              finding AgentRole.PerformanceBenchmarker ValidationOutcome.Pass
              finding AgentRole.SpecGuardian ValidationOutcome.Pass ]
          Comments = []
          Disagreements = [] }

    let private sampleIteration =
        { SpecSummary = sampleSummary
          ClosureResult =
            { CommandId = "closure-1"
              Success = true
              Output = None
              OutputSummary = "Patched CUDA kernels."
              Artifacts = [ "patches/cuda.patch" ]
              EvolutionData = Map.empty
              NextSteps = []
              Error = None
              ExecutionTime = TimeSpan.FromSeconds 42.0 }
          HarnessReport = Some harnessReport
          Consensus =
            Some(
                ConsensusPassed
                    [ { AgentId = "agent:reasoner"
                        Role = AgentRole.Reasoner
                        Outcome = ValidationOutcome.Pass
                        Confidence = Some 0.9
                        Notes = Some "Plan executed"
                        ProducedAt = DateTime.UtcNow }
                      { AgentId = "agent:reviewer"
                        Role = AgentRole.Reviewer
                        Outcome = ValidationOutcome.Pass
                        Confidence = Some 0.85
                        Notes = Some "QA validated"
                        ProducedAt = DateTime.UtcNow } ])
          ReasoningTraces = None
          CriticVerdict = Some RT.CriticVerdict.Accept
          CrossAgentFeedback = None
          ValidatorSnapshot = Some validatorSnapshot }

    [<Fact>]
    let ``team orchestrator produces squad verdicts`` () =
        let loggerFactory = NullLoggerFactory.Instance
        let options =
            { TeamCycleOptions.Default with
                PlannerTopCandidates = Some 3
                PlannerRecentMemory = Some 5 }

        let dependencies : TeamCycleDependencies =
            { PlanNext = fun _ _ _ -> async { return ([ plannerRecommendation ], [ plannerRecommendation ]) }
              RunIteration = fun _ _ _ _ -> async { return Some sampleIteration } }

        let result =
            TeamOrchestrator.runCycleAsync loggerFactory dependencies options
            |> Async.RunSynchronously

        Assert.Equal("completed", result.Summary.CycleStatus)
        Assert.Equal(Some "feature-gpu", result.Summary.PlannedFeatureId)
        Assert.Equal(Some "T100", result.Summary.PlannedTaskId)
        Assert.True(result.Summary.SquadVerdicts.Length >= 5)
        Assert.Equal(Some 3, result.Summary.ValidatorFindingCount)
        Assert.Equal(Some 0.0, result.Summary.ValidatorDisagreementRatio)
        Assert.Equal(Some 0, result.Summary.ValidatorDisagreementCount)
        Assert.True(result.Summary.Duration.IsSome)
