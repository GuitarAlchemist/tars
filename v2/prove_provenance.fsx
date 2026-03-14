#r "src/Tars.Core/bin/Debug/net10.0/Tars.Core.dll"

open System
open Tars.Core.WorkflowOfThought

let runId = Guid.NewGuid()
let state = FeedbackLoop.create runId

printfn "--- Initializing Hypothesis ---"

let h =
    FeedbackLoop.score
        "initial-node"
        "Biological Fruit Analysis"
        0.6
        0.5
        0.5
        0
        0.4
        "A tomato is botanically a fruit."
        []
        []

let state1 = FeedbackLoop.registerHypothesis h state

printfn "--- Adding Supporting Evidence (Tool) ---"

let ev1 =
    { Id = "ev-botany-01"
      Source = ToolContribution("botany_analyzer", "step-2")
      Content = "Genetic markers confirm seed development in ovary."
      Confidence = 1.0
      Weight = 0.8
      IsContradiction = false
      Timestamp = DateTime.UtcNow }

let state2, _ = FeedbackLoop.updateHypothesis "initial-node" ev1 state1

printfn "--- Adding Contradicting Evidence (Skeptic) ---"

let ev2 =
    { Id = "ev-culinary-01"
      Source = ReasonerThought "skeptic_agent"
      Content = "Culinary usage: Classified as vegetable for tax/cooking purposes."
      Confidence = 0.95
      Weight = 0.7
      IsContradiction = true
      Timestamp = DateTime.UtcNow.AddSeconds(1.0) }

let state3, _ = FeedbackLoop.updateHypothesis "initial-node" ev2 state2

printfn "--- Aggregating All Evidence ---"
let aggContext = FeedbackLoop.aggregateEvidence [ "initial-node" ] state3
printfn "Aggregated Context:\n%s" aggContext

printfn "--- Distilling Core Claim ---"

let state4 =
    let evDistill =
        { Id = "ev-distill-01"
          Source = ReasonerThought "distiller_agent"
          Content = "Consensus: Botanically fruit, culinary vegetable. Both are true in context."
          Confidence = 1.0
          Weight = 0.9
          IsContradiction = false
          Timestamp = DateTime.UtcNow.AddSeconds(2.0) }

    let s, _ = FeedbackLoop.updateHypothesis "initial-node" evDistill state3
    s

printfn "--- Backtracking from a Failed Hypothesis Branch ---"

let state5 =
    FeedbackLoop.registerHypothesis
        (FeedbackLoop.score "fail-node" "Tomato is a Mineral" 0.1 0.1 0.1 0 0.1 "Wait, what?" [] [])
        state4

let state6 =
    FeedbackLoop.invalidateHypothesis "fail-node" "Category error detected." state5

printfn "\n--- FINAL REASONING SUMMARY (Advanced GoT Proof) ---\n"
printfn "%s" (FeedbackLoop.summary state6)
