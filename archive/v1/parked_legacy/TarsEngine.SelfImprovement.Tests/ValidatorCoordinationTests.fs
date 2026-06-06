namespace TarsEngine.SelfImprovement.Tests

open System
open Xunit
open TarsEngine.FSharp.Core.Services.CrossAgentValidation
open TarsEngine.FSharp.SelfImprovement.ValidatorCoordination

[<Collection("Serial")>]
module ValidatorCoordinationTests =

    let private makeTarget specId iterationId =
        { SpecId = specId
          IterationId = Some iterationId
          Topic = Some "validation" }

    let private makeFinding specId iterationId role outcome confidence recordedAt =
        { FindingId = Guid.NewGuid()
          AgentId = $"{role.ToString().ToLowerInvariant()}-agent"
          Role = role
          Outcome = outcome
          Confidence = confidence
          Notes = None
          Target = makeTarget specId iterationId
          RecordedAt = recordedAt }

    [<Fact>]
    let ``CoordinationBus detects conflicting pass and fail verdicts`` () =
        let bus = CoordinationBus()
        let iterationId = Guid.NewGuid()
        let baseTime = DateTime.UtcNow

        let passFinding =
            makeFinding
                "spec-tier3"
                iterationId
                AgentRole.SafetyGovernor
                ValidationOutcome.Pass
                (Some 0.92)
                baseTime

        let failFinding =
            makeFinding
                "spec-tier3"
                iterationId
                AgentRole.PerformanceBenchmarker
                ValidationOutcome.Fail
                (Some 0.41)
                (baseTime.AddSeconds(2.0))

        bus.PublishFinding passFinding
        bus.PublishFinding failFinding

        let snapshot = bus.Snapshot()
        let disagreement = Assert.Single(snapshot.Disagreements)
        Assert.Equal("conflicting_pass_fail", disagreement.Trigger)
        Assert.Contains(AgentRole.SafetyGovernor, disagreement.Roles)
        Assert.Contains(AgentRole.PerformanceBenchmarker, disagreement.Roles)

        match disagreement.ConfidenceSpread with
        | Some spread -> Assert.Equal(0.51, spread, 2)
        | None -> failwith "Expected confidence spread to be present."

    [<Fact>]
    let ``CoordinationBus normalises spec identifiers`` () =
        let bus = CoordinationBus()
        let iterationId = Guid.NewGuid()
        let recordedAt = DateTime.UtcNow

        let finding =
            makeFinding
                "Spec-Alpha"
                iterationId
                AgentRole.MetaCritic
                ValidationOutcome.NeedsReview
                None
                recordedAt

        bus.PublishFinding finding

        let lower = bus.GetFindingsForSpec "spec-alpha"
        let upper = bus.GetFindingsForSpec "SPEC-ALPHA"

        let lowerFinding = Assert.Single(lower)
        let upperFinding = Assert.Single(upper)
        Assert.Equal(lowerFinding.FindingId, upperFinding.FindingId)
