namespace TarsEngine.SelfImprovement.Tests

open System
open Xunit
open TarsEngine.FSharp.SelfImprovement
open TarsEngine.FSharp.SelfImprovement.AutonomousSpecHarness
open TarsEngine.FSharp.Core.Services.MetascriptClosureIntegrationService
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.Core.Services.ReasoningTrace
open TarsEngine.FSharp.SelfImprovement.PersistentAdaptiveMemory
open TarsEngine.FSharp.Core.Specs

module SelfImprovementHelpersTests =

    let private specSummary : TarsEngine.FSharp.Core.Specs.SpecKitSummary =
        { Title = "Demo"
          Status = None
          FeatureBranch = None
          Created = None
          UserStories = []
          EdgeCases = [] }

    let private closureResult : MetascriptClosureResult =
        { CommandId = "closure"
          Success = true
          Output = None
          OutputSummary = ""
          Artifacts = []
          EvolutionData = Map.empty
          NextSteps = []
          Error = None
          ExecutionTime = TimeSpan.Zero }

    let private iterationResult : SpecDrivenIterationResult =
        { SpecSummary = specSummary
          ClosureResult = closureResult
          HarnessReport = None
          Consensus = None
          ReasoningTraces = None
          CriticVerdict = Some CriticVerdict.Accept
          CrossAgentFeedback = None
          ValidatorSnapshot = None }

    let private safetyAgent : AgentValidationResult =
        { AgentId = "agent:safety"
          Role = AgentRole.SafetyGovernor
          Outcome = ValidationOutcome.Pass
          Confidence = Some 0.91
          Notes = Some "All safety checks passed."
          ProducedAt = DateTime.UtcNow }

    let private performanceAgent : AgentValidationResult =
        { AgentId = "agent:performance"
          Role = AgentRole.PerformanceBenchmarker
          Outcome = ValidationOutcome.Fail
          Confidence = Some 0.35
          Notes = Some "Benchmark regression detected."
          ProducedAt = DateTime.UtcNow }

    let private criticContext : CriticContext =
        { source = Some "model:128"
          threshold = Some 0.62
          sampleSize = Some 144
          indicators = [ "drift"; "hallucination" ] }

    let private previousCriticContext : CriticContext =
        { source = Some "model:127"
          threshold = Some 0.58
          sampleSize = Some 120
          indicators = [ "drift" ] }

    [<Fact>]
    let ``ledger metrics include agent outcomes and critic telemetry`` () =
        let metrics, _ =
            SelfImprovementHelpers.computeLedgerMetrics
                iterationResult
                []
                [ safetyAgent; performanceAgent ]
                (Some criticContext)
                (Some previousCriticContext)

        Assert.Equal("pass", metrics["agents.safety.outcome"] :?> string)
        Assert.Equal(0.91, metrics["agents.safety.confidence"] :?> float, 3)
        Assert.Equal("fail", metrics["agents.performance.outcome"] :?> string)
        Assert.Equal(0.35, metrics["agents.performance.confidence"] :?> float, 3)
        Assert.Equal("model:128", metrics["safety.critic_source"] :?> string)
        Assert.Equal(0.62, metrics["safety.critic_threshold"] :?> float, 2)
        Assert.Equal(144, metrics["safety.critic_samples"] :?> int)
        Assert.Equal(0.04, metrics["safety.critic_threshold_delta"] :?> float, 2)
        Assert.True(metrics["safety.critic_source_changed"] :?> bool)

    [<Fact>]
    let ``computeLedgerMetrics flattens inference telemetry`` () =
        let telemetry =
            Map.ofList [
                "inference.metrics",
                box (
                    Map.ofList [
                        "token_count", box 42
                        "top_terms", box [| "safety"; "performance" |]
                    ])
                "inference.model_name", box "demo-model"
            ]

        let resultWithTelemetry =
            { iterationResult with
                ClosureResult =
                    { iterationResult.ClosureResult with
                        EvolutionData = telemetry } }

        let metrics, _ =
            SelfImprovementHelpers.computeLedgerMetrics
                resultWithTelemetry
                []
                []
                None
                None

        Assert.Equal(42, metrics["inference.metrics.token_count"] :?> int)
        Assert.Equal("safety,performance", metrics["inference.metrics.top_terms"] :?> string)
        Assert.Equal("demo-model", metrics["inference.model_name"] :?> string)
