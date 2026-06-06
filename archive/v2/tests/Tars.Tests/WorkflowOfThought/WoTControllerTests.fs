namespace Tars.Tests.WorkflowOfThought

open Xunit

open Xunit
open Tars.Core.WorkflowOfThought

module WotControllerTests =

    // Helper for fake hypothesis
    let mkScore id score volume contradictions =

        { Id = id
          Label = "test"
          EvidenceCoverage = score
          TemporalFit = score
          MechanismPlausibility = score
          Contradictions = contradictions
          ImpactScore = score
          OverallScore = score
          Reason = "test"
          EvidenceIds = []
          ConflictingEvidenceIds = []
          Volume = volume }

    [<Fact>]
    let ``Router - High score triggers Finalize`` () =
        let h = mkScore "h1" 0.95 2 0
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Finalize id -> Assert.Equal("h1", id)
        | _ -> failwith $"Expected Finalize, got {decision}"

    [<Fact>]
    let ``Router - Low score with contradictions triggers Backtrack`` () =
        let h = mkScore "h2" 0.4 2 1
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Backtrack id -> Assert.Equal("h2", id)
        | _ -> failwith $"Expected Backtrack, got {decision}"

    [<Fact>]
    let ``Router - Low score no contradictions triggers Refine`` () =
        let h = mkScore "h3" 0.4 2 0
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Refine id -> Assert.Equal("h3", id)
        | _ -> failwith $"Expected Refine, got {decision}"

    [<Fact>]
    let ``Router - Good score within volume limit triggers Expand`` () =
        let h = mkScore "h4" 0.8 5 0
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Expand(id, k) ->
            Assert.Equal("h4", id)
            Assert.Equal(3, k)
        | _ -> failwith $"Expected Expand, got {decision}"

    [<Fact>]
    let ``Router - Volume limit exceeded with low score triggers Backtrack`` () =
        let h = mkScore "h5" 0.2 11 0
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Backtrack id -> Assert.Equal("h5", id)
        | _ -> failwith $"Expected Backtrack, got {decision}"

    [<Fact>]
    let ``Router - Volume limit exceeded with medium score triggers Finalize (Cut losses)`` () =
        // If we invested a lot (Volume > Max) and have an okay score (but not great),
        // we might just accept it or finalize it to stop spinning.
        // Current logic: if score < 0.3 Backtrack, else Finalize.
        let h = mkScore "h6" 0.5 11 0
        let decision = WotController.Router.decide h 0.7 10

        match decision with
        | WotController.Finalize id -> Assert.Equal("h6", id)
        | _ -> failwith $"Expected Finalize, got {decision}"
