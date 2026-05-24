module Tars.Tests.ResearchCycleTests

open System
open Xunit
open Tars.Evolution.ResearchTypes
open Tars.Evolution.ResearchWeights
open Tars.Evolution.ResearchCycle

let private makeWeights () : DepartmentWeights =
    { Department = "test-dept"
      HypothesisWeights = Map.ofList [
        "inductive", 0.30; "deductive", 0.20; "abductive", 0.25
        "analogical", 0.15; "combinatorial", 0.10 ]
      TestWeights = Map.ofList [
        "empirical", 0.30; "formal_proof", 0.10; "simulation", 0.20
        "thought_experiment", 0.15; "cross_validation", 0.15; "adversarial", 0.10 ]
      CycleCount = 0
      LastUpdated = DateTime.UtcNow }

[<Fact>]
let ``sampleMethod respects weight distribution`` () =
    let weights = Map.ofList [ "a", 0.8; "b", 0.1; "c", 0.1 ]
    let rng = Random(42)
    let samples = [ for _ in 1..1000 -> sampleMethod weights rng ]
    let aCount = samples |> List.filter ((=) "a") |> List.length
    // "a" has 80% weight, should appear 700-900 times out of 1000
    Assert.True(aCount > 600 && aCount < 950, $"Expected ~800, got {aCount}")

[<Fact>]
let ``updateFromOutcome increases weight for successful method`` () =
    let w = makeWeights ()
    let before = w.HypothesisWeights.["inductive"]
    let updated = updateFromOutcome w "inductive" "empirical" true
    let after = updated.HypothesisWeights.["inductive"]
    Assert.True(after > before, $"Expected increase: {before} → {after}")

[<Fact>]
let ``updateFromOutcome decreases weight for failed method`` () =
    let w = makeWeights ()
    let before = w.HypothesisWeights.["inductive"]
    let updated = updateFromOutcome w "inductive" "empirical" false
    let after = updated.HypothesisWeights.["inductive"]
    Assert.True(after < before, $"Expected decrease: {before} → {after}")

[<Fact>]
let ``weights remain normalized after update`` () =
    let w = makeWeights ()
    let updated = updateFromOutcome w "deductive" "simulation" true
    let hypSum = updated.HypothesisWeights |> Map.toList |> List.sumBy snd
    let testSum = updated.TestWeights |> Map.toList |> List.sumBy snd
    Assert.True(abs(hypSum - 1.0) < 0.01, $"Hypothesis sum: {hypSum}")
    Assert.True(abs(testSum - 1.0) < 0.01, $"Test sum: {testSum}")

[<Fact>]
let ``conclusionToTetraValue maps correctly`` () =
    Assert.Equal("T", conclusionToTetraValue Confirm)
    Assert.Equal("F", conclusionToTetraValue Refute)
    Assert.Equal("U", conclusionToTetraValue Insufficient)
    Assert.Equal("C", conclusionToTetraValue Contradictory)
    Assert.Equal("U", conclusionToTetraValue Revise)
    Assert.Equal("U", conclusionToTetraValue DiscoverQuestion)

[<Fact>]
let ``isAnomaly flags F U C but not T`` () =
    let makeResult bv = {
        CycleId = "test"; Department = "test"; Question = "test"
        Path = { HypothesisMethod = Inductive; TestMethod = Empirical; Conclusion = Confirm; Reflection = NormalProgress }
        Hypothesis = "test"; Evidence = []; BeliefValue = bv; BeliefConfidence = 0.8
        DurationSeconds = 10; Timestamp = DateTime.UtcNow }
    Assert.False(isAnomaly (makeResult "T"))
    Assert.True(isAnomaly (makeResult "F"))
    Assert.True(isAnomaly (makeResult "U"))
    Assert.True(isAnomaly (makeResult "C"))

[<Fact>]
let ``createAnomalyEntry produces valid entry`` () =
    let result = {
        CycleId = "cycle-001"; Department = "music"; Question = "test"
        Path = { HypothesisMethod = Abductive; TestMethod = CrossValidation; Conclusion = Refute; Reflection = AnomalyDetected }
        Hypothesis = "tritone substitution is universal"
        Evidence = ["jazz: works"; "metal: fails"]; BeliefValue = "F"; BeliefConfidence = 0.3
        DurationSeconds = 120; Timestamp = DateTime.UtcNow }
    let anomaly = createAnomalyEntry result
    Assert.Equal("music", anomaly.Department)
    Assert.Equal("cycle-001", anomaly.CycleId)
    Assert.Contains("abductive", anomaly.ProductionPath)
    Assert.Contains("cross_validation", anomaly.ProductionPath)
    Assert.True(anomaly.Severity > 0.5)  // low confidence → high severity

[<Fact>]
let ``sampleProductionPath returns valid path`` () =
    let w = makeWeights ()
    let rng = Random(42)
    let path = sampleProductionPath w rng
    // Should have valid hypothesis and test methods
    let hypName = hypothesisMethodName path.HypothesisMethod
    let testName = testMethodName path.TestMethod
    Assert.True(w.HypothesisWeights.ContainsKey(hypName), $"Unknown hypothesis: {hypName}")
    Assert.True(w.TestWeights.ContainsKey(testName), $"Unknown test: {testName}")
