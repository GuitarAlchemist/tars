module Tars.Tests.ReasonFeedbackTests

open System
open Xunit
open Tars.Core.WorkflowOfThought

// The feedback policy is now a pure reducer behind a small seam, so it can be
// driven and asserted without running the WoT executor loop.

let private mkStep id inputs : Step =
    { Id = id
      Inputs = inputs
      Outputs = []
      Action = StepAction.Reason(ReasonOperation.Plan "x")
      Agent = None
      Metadata = Map.empty }

[<Fact>]
let ``reduce ToolObserved adds evidence keyed by the step id`` () =
    let s0 = FeedbackLoop.create (Guid.NewGuid())
    let s1 = ReasonFeedbackCore.reduce s0 (ToolObserved(mkStep "t1" [], "echo"))
    Assert.True(s1.EvidenceLibrary.ContainsKey "t1")

[<Fact>]
let ``reduce of an Llm Score registers a hypothesis with the parsed coverage`` () =
    let s0 = FeedbackLoop.create (Guid.NewGuid())

    let s1 =
        ReasonFeedbackCore.reduce
            s0
            (ReasonObserved(mkStep "score1" [], ReasonOperation.Score(NodeId "h1"), "Score: 0.9", false))

    let key = string (NodeId "h1")
    Assert.True(s1.Hypotheses.ContainsKey key)
    Assert.Equal(0.9, s1.Hypotheses.[key].EvidenceCoverage)

[<Fact>]
let ``parseScore reads a Score line and defaults to 0.5`` () =
    Assert.Equal(0.75, ReasonFeedbackCore.parseScore "blah\nScore: 0.75\nmore")
    Assert.Equal(0.5, ReasonFeedbackCore.parseScore "no score here")

[<Fact>]
let ``Decide returns Wait for an unknown node`` () =
    let fb = ReasonFeedback(Guid.NewGuid()) :> IReasonFeedback
    Assert.Equal<WotController.RouterDecision>(WotController.Wait, fb.Decide("nope"))

[<Fact>]
let ``Observe then Decide on the scored node yields a routing decision`` () =
    let fb = ReasonFeedback(Guid.NewGuid()) :> IReasonFeedback
    fb.Observe(ReasonObserved(mkStep "score1" [], ReasonOperation.Score(NodeId "h1"), "Score: 0.9", false))
    // A registered hypothesis routes to something other than Wait.
    Assert.NotEqual<WotController.RouterDecision>(WotController.Wait, fb.Decide(string (NodeId "h1")))
