namespace Tars.Tests.WorkflowOfThought

open Xunit
open System
open Tars.Core.WorkflowOfThought

module EvidenceProvenanceTests =

    [<Fact>]
    let ``FeedbackLoop - Refine transformation updates iterations and evidence chain`` () =
        // Setup
        let runId = Guid.NewGuid()
        let state = FeedbackLoop.create runId

        // 1. Register hypothesis
        let h = FeedbackLoop.score "node1" "Subject" 0.5 0.5 0.5 0 0.5 "Initial" [] [] 0
        let state1 = FeedbackLoop.registerHypothesis h state

        // 2. Add evidence via refinement
        let ev =
            { Id = "ev1"
              Source = ReasonerThought "node2"
              Content = "Improved content"
              Confidence = 1.0
              Weight = 0.5
              IsContradiction = false
              ParentIds = []
              Timestamp = DateTime.UtcNow }

        let state2, result = FeedbackLoop.updateHypothesis "node1" ev state1

        // Assert
        Assert.Single(state2.EvidenceChain) |> ignore
        Assert.Equal(1, state2.Iterations)
        Assert.Equal("ev1", state2.EvidenceChain |> List.head)
        Assert.True(state2.Hypotheses.ContainsKey "node1")
        let h2 = state2.Hypotheses.["node1"]
        Assert.Contains("ev1", h2.EvidenceIds)
        Assert.Equal(0.55, h2.EvidenceCoverage, 2) // 0.5 + (0.5 * 0.1)

    [<Fact>]
    let ``FeedbackLoop - Contradict transformation penalizes score`` () =
        let runId = Guid.NewGuid()
        let state = FeedbackLoop.create runId
        let h = FeedbackLoop.score "node1" "Subject" 0.5 0.5 0.5 0 0.5 "Initial" [] [] 0
        let state1 = FeedbackLoop.registerHypothesis h state

        let ev =
            { Id = "con1"
              Source = ReasonerThought "node3"
              Content = "Counter-argument"
              Confidence = 1.0
              Weight = 1.0
              IsContradiction = true
              ParentIds = []
              Timestamp = DateTime.UtcNow }

        let state2, _ = FeedbackLoop.updateHypothesis "node1" ev state1

        let h2 = state2.Hypotheses.["node1"]
        Assert.Equal(1, h2.Contradictions)
        Assert.Equal(0.4, h2.EvidenceCoverage, 2) // 0.5 - (1.0 * 0.1)
        Assert.True(h2.OverallScore < h.OverallScore)

    [<Fact>]
    let ``FeedbackLoop - getEvidenceTrail reconstructs history`` () =
        let runId = Guid.NewGuid()
        let state = FeedbackLoop.create runId
        let h = FeedbackLoop.score "node1" "Subject" 0.5 0.5 0.5 0 0.5 "Initial" [] [] 0
        let state1 = FeedbackLoop.registerHypothesis h state

        let ev1 =
            { Id = "e1"
              Source = ToolContribution("tool", "s1")
              Content = "c1"
              Confidence = 1.0
              Weight = 0.5
              IsContradiction = false
              ParentIds = []
              Timestamp = DateTime.UtcNow }

        let state2, _ = FeedbackLoop.updateHypothesis "node1" ev1 state1

        let ev2 =
            { Id = "e2"
              Source = ReasonerThought "s2"
              Content = "c2"
              Confidence = 1.0
              Weight = 0.5
              IsContradiction = true
              ParentIds = []
              Timestamp = DateTime.UtcNow.AddSeconds(1.0) }

        let state3, _ = FeedbackLoop.updateHypothesis "node1" ev2 state2

        let trail = FeedbackLoop.getEvidenceTrail "node1" state3
        Assert.Equal(2, trail.Length)
        Assert.Equal("e1", trail.[0].Id)
        Assert.Equal("e2", trail.[1].Id)

    [<Fact>]
    let ``FeedbackLoop - Proof of Work - Multi-stage evidence aggregation`` () =
        let runId = Guid.NewGuid()
        let state = FeedbackLoop.create runId

        // 1. Initial Hypothesis (e.g. from node 'initial')
        let h =
            FeedbackLoop.score
                "initial"
                "Tomato Classification"
                0.6
                0.5
                0.5
                0
                0.4
                "A tomato is a biological fruit."
                []
                []
                0

        let state1 = FeedbackLoop.registerHypothesis h state

        // 2. Supporting Evidence from a Tool (e.g. from 'gather_data')
        let ev1 =
            { Id = "tool_ev_botany"
              Source = ToolContribution("botany_analyzer", "step2")
              Content = "Botanical definition confirmed: seeds present."
              Confidence = 1.0
              Weight = 0.8
              IsContradiction = false
              ParentIds = []
              Timestamp = DateTime.UtcNow }

        let state2, _ = FeedbackLoop.updateHypothesis "initial" ev1 state1

        // 3. Contradicting Evidence from a Skeptic Agent (Culinary view)
        let ev2 =
            { Id = "agent_ev_culinary"
              Source = ReasonerThought "skeptic_agent"
              Content = "Culinary usage: tomatoes are utilized as vegetables in cooking."
              Confidence = 0.9
              Weight = 0.5
              IsContradiction = true
              ParentIds = []
              Timestamp = DateTime.UtcNow.AddSeconds(1.0) }

        let state3, _ = FeedbackLoop.updateHypothesis "initial" ev2 state2

        // 4. Refinement from a QA Agent
        let ev3 =
            { Id = "agent_ev_refine"
              Source = ReasonerThought "qa_agent"
              Content = "Refined view: Botanically fruit, culinary vegetable."
              Confidence = 0.95
              Weight = 0.4
              IsContradiction = false
              ParentIds = []
              Timestamp = DateTime.UtcNow.AddSeconds(2.0) }

        let state4, _ = FeedbackLoop.updateHypothesis "initial" ev3 state3

        // Print proof to stdout
        printfn "\n--- PHASE 17.4 PROOF: EVIDENCE PROVENANCE SUMMARY ---\n%s" (FeedbackLoop.summary state4)

        // Assertions for proof
        let finalH = state4.Hypotheses.["initial"]
        Assert.Equal(3, finalH.EvidenceIds.Length + finalH.ConflictingEvidenceIds.Length)
        Assert.Equal(1, finalH.Contradictions)
        Assert.True(finalH.OverallScore > 0.0)
