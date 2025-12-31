module Tars.Tests.SymbolicReflectionTests

open Xunit
open FsUnit
open System
open Tars.Core

// =============================================================================
// PHASE 15: SYMBOLIC REFLECTION TESTS
// =============================================================================

// =============================================================================
// 15.1 Structured Reflection Types Tests
// =============================================================================

[<Fact>]
let ``SymbolicReflection.Create initializes with default values`` () =
    let agentId = AgentId(Guid.NewGuid())
    let trigger = ReflectionTrigger.PeriodicReview(TimeSpan.FromHours(1.0))

    let reflection = SymbolicReflection.Create(agentId, trigger)

    reflection.AgentId |> should equal agentId
    reflection.Trigger |> should equal trigger
    reflection.Observations |> should be Empty
    reflection.BeliefUpdates |> should be Empty
    reflection.Justifications |> should be Empty
    reflection.Confidence |> should equal 0.0
    reflection.ImpactScore |> should equal 0.0

[<Fact>]
let ``ReflectionBelief.Create sets default fields`` () =
    let belief = ReflectionBelief.Create("Test belief", 0.85)

    belief.Statement |> should equal "Test belief"
    belief.Confidence |> should equal 0.85
    belief.Evidence |> should be Empty
    belief.Tags |> should be Empty

[<Fact>]
let ``SymbolicReflection.WithObservation adds observation`` () =
    let agentId = AgentId(Guid.NewGuid())

    let reflection =
        SymbolicReflection.Create(agentId, PeriodicReview(TimeSpan.FromMinutes(30.0)))

    let obs = ReflectionObservation.PatternObserved("DRY violation", 3, 0.9)
    let updated = reflection.WithObservation(obs)

    updated.Observations.Length |> should equal 1
    updated.Observations.Head |> should equal obs

[<Fact>]
let ``SymbolicReflection.CalculateImpact returns correct score`` () =
    let agentId = AgentId(Guid.NewGuid())

    let reflection =
        { SymbolicReflection.Create(agentId, PeriodicReview(TimeSpan.FromMinutes(30.0))) with
            BeliefUpdates =
                [ ReflectionBeliefUpdate.AddBelief(ReflectionBelief.Create("New belief", 0.8), [])
                  ReflectionBeliefUpdate.RevokeBelief(Guid.NewGuid(), "Outdated", None) ] }

    let impact = reflection.CalculateImpact()

    // AddBelief = 0.3, RevokeBelief = 0.8, total = 1.0 (capped)
    impact |> should be (greaterThan 0.9)

[<Fact>]
let ``ReflectionEvidence.Create sets correct fields`` () =
    let source =
        ReflectionEvidenceSource.DirectObservation("Observed pattern", DateTimeOffset.UtcNow)

    let evidence = ReflectionEvidence.Create(source, "Details of observation", 0.95)

    evidence.Source |> should equal source
    evidence.Content |> should equal "Details of observation"
    evidence.Confidence |> should equal 0.95

[<Fact>]
let ``SymbolicReflectionHelpers.fromContradiction creates correct trigger`` () =
    let agentId = AgentId(Guid.NewGuid())

    let reflection =
        SymbolicReflectionHelpers.fromContradiction agentId "Belief A" "Belief B"

    match reflection.Trigger with
    | ContradictionDetected(b1, b2) ->
        b1 |> should equal "Belief A"
        b2 |> should equal "Belief B"
    | _ -> failwith "Wrong trigger type"

// =============================================================================
// 15.2 Belief Revision Engine Tests
// =============================================================================

[<Fact>]
let ``BeliefRevision applyAddBelief adds belief to empty store`` () =
    let store = BeliefRevision.ReflectionBeliefStore.empty
    let belief = ReflectionBelief.Create("Binary search requires sorted input", 0.95)

    let evidence =
        [ ReflectionEvidence.Create(ReflectionEvidenceSource.TestResult("binary_search_test", true), "Test passed", 1.0) ]

    match BeliefRevision.applyAddBelief belief evidence store with
    | FSharp.Core.Ok(newStore, events) ->
        BeliefRevision.ReflectionBeliefStore.count newStore |> should equal 1
        events.Length |> should equal 1
    | FSharp.Core.Error e -> failwithf "Should have succeeded: %A" e

[<Fact>]
let ``BeliefRevision applyRevokeBelief removes belief`` () =
    let belief = ReflectionBelief.Create("Outdated assumption", 0.5)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add belief

    match BeliefRevision.applyRevokeBelief belief.Id "No longer valid" None store with
    | FSharp.Core.Ok(newStore, events) ->
        BeliefRevision.ReflectionBeliefStore.count newStore |> should equal 0

        events
        |> should contain (BeliefRevision.BeliefRevoked(belief.Id, "No longer valid"))
    | FSharp.Core.Error e -> failwithf "Should have succeeded: %A" e

[<Fact>]
let ``BeliefRevision applyRevokeBelief fails for non-existent belief`` () =
    let store = BeliefRevision.ReflectionBeliefStore.empty
    let nonExistentId = Guid.NewGuid()

    match BeliefRevision.applyRevokeBelief nonExistentId "Test" None store with
    | FSharp.Core.Error(BeliefRevision.BeliefNotFound id) -> id |> should equal nonExistentId
    | _ -> failwith "Should have failed with BeliefNotFound"

[<Fact>]
let ``BeliefRevision applyAdjustConfidence updates confidence`` () =
    let belief = ReflectionBelief.Create("Initial belief", 0.7)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add belief

    match BeliefRevision.applyAdjustConfidence belief.Id 0.9 "New evidence supports" store with
    | FSharp.Core.Ok(newStore, events) ->
        let updated = BeliefRevision.ReflectionBeliefStore.get belief.Id newStore
        updated |> Option.map (fun b -> b.Confidence) |> should equal (Some 0.9)
        events.Length |> should be (greaterThan 0)
    | FSharp.Core.Error e -> failwithf "Should have succeeded: %A" e

[<Fact>]
let ``BeliefRevision applyAdjustConfidence fails for invalid confidence`` () =
    let belief = ReflectionBelief.Create("Test belief", 0.5)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add belief

    match BeliefRevision.applyAdjustConfidence belief.Id 1.5 "Invalid" store with
    | FSharp.Core.Error(BeliefRevision.InvalidConfidence value) -> value |> should equal 1.5
    | _ -> failwith "Should have failed with InvalidConfidence"

[<Fact>]
let ``BeliefRevision applyMergeBeliefs combines beliefs`` () =
    let belief1 = ReflectionBelief.Create("Arrays need bounds checking", 0.8)
    let belief2 = ReflectionBelief.Create("Buffer operations need validation", 0.75)
    let merged = ReflectionBelief.Create("All memory operations need validation", 0.85)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add belief1
        |> BeliefRevision.ReflectionBeliefStore.add belief2

    match BeliefRevision.applyMergeBeliefs [ belief1.Id; belief2.Id ] merged store with
    | FSharp.Core.Ok(newStore, events) ->
        BeliefRevision.ReflectionBeliefStore.count newStore |> should equal 1

        (BeliefRevision.ReflectionBeliefStore.get merged.Id newStore).IsSome
        |> should be True
    | FSharp.Core.Error e -> failwithf "Should have succeeded: %A" e

[<Fact>]
let ``BeliefRevision applySplitBelief divides belief`` () =
    let original = ReflectionBelief.Create("General memory safety principle", 0.9)
    let refined1 = ReflectionBelief.Create("Array bounds checking", 0.95)
    let refined2 = ReflectionBelief.Create("Null pointer validation", 0.92)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add original

    match BeliefRevision.applySplitBelief original.Id [ refined1; refined2 ] store with
    | FSharp.Core.Ok(newStore, events) ->
        BeliefRevision.ReflectionBeliefStore.count newStore |> should equal 2

        (BeliefRevision.ReflectionBeliefStore.get original.Id newStore).IsNone
        |> should be True
    | FSharp.Core.Error e -> failwithf "Should have succeeded: %A" e

[<Fact>]
let ``BeliefRevision findStaleBeliefs finds old beliefs`` () =
    let oldBelief =
        { ReflectionBelief.Create("Old belief", 0.8) with
            LastUpdated = DateTimeOffset.UtcNow.AddDays(-30.0) }

    let newBelief = ReflectionBelief.Create("New belief", 0.9)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add oldBelief
        |> BeliefRevision.ReflectionBeliefStore.add newBelief

    let stale = BeliefRevision.findStaleBeliefs store (TimeSpan.FromDays(7.0)) 0.5

    stale.Length |> should equal 1
    stale.Head.Id |> should equal oldBelief.Id

[<Fact>]
let ``BeliefRevision findStaleBeliefs finds low confidence beliefs`` () =
    let lowConfidence = ReflectionBelief.Create("Uncertain belief", 0.2)
    let highConfidence = ReflectionBelief.Create("Confident belief", 0.9)

    let store =
        BeliefRevision.ReflectionBeliefStore.empty
        |> BeliefRevision.ReflectionBeliefStore.add lowConfidence
        |> BeliefRevision.ReflectionBeliefStore.add highConfidence

    let stale = BeliefRevision.findStaleBeliefs store (TimeSpan.FromDays(365.0)) 0.5

    stale.Length |> should equal 1
    stale.Head.Confidence |> should be (lessThan 0.5)

// =============================================================================
// 15.3 Evidence Chains Tests
// =============================================================================

[<Fact>]
let ``EvidenceChains buildChain creates chain with evidence`` () =
    let evidence =
        ReflectionEvidence.Create(
            ReflectionEvidenceSource.TestResult("sorting_test", true),
            "Test verifies sorting works correctly",
            0.95
        )

    let belief =
        { ReflectionBelief.Create("Sorting algorithm is correct", 0.9) with
            Evidence = [ evidence ] }

    let chain = EvidenceChains.buildChain belief Map.empty

    chain.DirectEvidence.Length |> should equal 1
    chain.OverallConfidence |> should equal 0.95

[<Fact>]
let ``EvidenceChains verifyChain returns Ok for valid chain`` () =
    let evidence =
        ReflectionEvidence.Create(
            ReflectionEvidenceSource.ExternalSource("https://docs.example.com", DateTimeOffset.UtcNow),
            "Documentation confirms the behavior",
            0.85
        )

    let belief =
        { ReflectionBelief.Create("API returns JSON", 0.85) with
            Evidence = [ evidence ] }

    let chain = EvidenceChains.buildChain belief Map.empty

    match EvidenceChains.verifyChain chain with
    | FSharp.Core.Ok() -> ()
    | FSharp.Core.Error errors -> failwithf "Should have passed: %A" errors

[<Fact>]
let ``EvidenceChains findWeakestLink identifies weakest evidence`` () =
    let strongEvidence =
        ReflectionEvidence.Create(ReflectionEvidenceSource.TestResult("primary_test", true), "Strong", 0.95)

    let weakEvidence =
        ReflectionEvidence.Create(ReflectionEvidenceSource.UserFeedback("Seems right", None), "Weak", 0.6)

    let belief =
        { ReflectionBelief.Create("Multiple evidence", 0.8) with
            Evidence = [ strongEvidence; weakEvidence ] }

    let chain = EvidenceChains.buildChain belief Map.empty
    let weakest = EvidenceChains.findWeakestLink chain

    weakest.IsSome |> should be True

    match weakest with
    | Some(e, conf) -> conf |> should equal 0.6
    | None -> failwith "Should find weakest"

[<Fact>]
let ``EvidenceChains getCompletenessScore scores complete chains higher`` () =
    let evidence =
        ReflectionEvidence.Create(ReflectionEvidenceSource.TestResult("complete_test", true), "Complete", 0.9)

    let belief =
        { ReflectionBelief.Create("Well-supported belief", 0.9) with
            Evidence = [ evidence ] }

    let chain = EvidenceChains.buildChain belief Map.empty
    let score = EvidenceChains.getCompletenessScore chain

    score |> should be (greaterThan 0.5)

[<Fact>]
let ``EvidenceChains visualize produces readable output`` () =
    let evidence =
        ReflectionEvidence.Create(ReflectionEvidenceSource.TestResult("viz_test", true), "Test passed", 0.9)

    let belief =
        { ReflectionBelief.Create("Visualizable belief", 0.9) with
            Evidence = [ evidence ] }

    let chain = EvidenceChains.buildChain belief Map.empty
    let viz = EvidenceChains.visualize chain

    viz |> should contain "Belief:"
    viz |> should contain "Confidence:"
    viz |> should contain "Direct Evidence:"

// =============================================================================
// 15.4 Proof System Tests
// =============================================================================

[<Fact>]
let ``ProofSystem strengthOf returns 1.0 for Tautology`` () =
    let proof = ReflectionProof.Tautology "P or not P"

    ProofSystem.strengthOf proof |> should equal 1.0

[<Fact>]
let ``ProofSystem strengthOf returns high value for ValidationSuccess`` () =
    let proof = ReflectionProof.ValidationSuccess("unit_test", "All assertions passed")

    ProofSystem.strengthOf proof |> should be (greaterThan 0.9)

[<Fact>]
let ``ProofSystem strengthOf returns low value for Analogy`` () =
    let proof =
        ReflectionProof.LogicalInference(
            [ "A is similar to B"; "A has property X" ],
            "B might have property X",
            ReflectionInferenceRule.Analogy
        )

    ProofSystem.strengthOf proof |> should be (lessThan 0.5)

[<Fact>]
let ``ProofSystem verifyProof returns Ok for valid proof`` () =
    let proof =
        ReflectionProof.LogicalInference(
            [ "All humans are mortal"; "Socrates is human" ],
            "Socrates is mortal",
            ReflectionInferenceRule.Syllogism
        )

    match ProofSystem.verifyProof proof with
    | FSharp.Core.Ok result ->
        result.IsValid |> should be True
        result.Strength |> should be (greaterThan 0.7)
    | FSharp.Core.Error errors -> failwithf "Should have passed: %A" errors

[<Fact>]
let ``ProofSystem verifyProof returns Error for invalid proof`` () =
    let proof =
        ReflectionProof.LogicalInference([], "", ReflectionInferenceRule.ModusPonens)

    match ProofSystem.verifyProof proof with
    | FSharp.Core.Error errors -> errors.Length |> should be (greaterThan 0)
    | FSharp.Core.Ok _ -> failwith "Should have failed"

[<Fact>]
let ``ProofSystem supports checks if proof supports belief`` () =
    let proof = ReflectionProof.ValidationSuccess("sorting_test", "Passed")
    let belief = ReflectionBelief.Create("sorting_test verifies correctness", 0.9)

    ProofSystem.supports proof belief |> should be True

[<Fact>]
let ``ProofSystem combineProofs increases confidence`` () =
    let proof1 = ReflectionProof.ValidationSuccess("test1", "Passed")

    let proof2 =
        ReflectionProof.LogicalInference([ "P" ], "Q", ReflectionInferenceRule.ModusPonens)

    let combined = ProofSystem.combineProofs [ proof1; proof2 ]
    let single = ProofSystem.strengthOf proof1

    combined |> should be (greaterThan single)

[<Fact>]
let ``ProofSystem categorizeStrength returns correct category`` () =
    ProofSystem.categorizeStrength 0.1 |> should equal ProofStrength.VeryWeak
    ProofSystem.categorizeStrength 0.3 |> should equal ProofStrength.Weak
    ProofSystem.categorizeStrength 0.5 |> should equal ProofStrength.Moderate
    ProofSystem.categorizeStrength 0.7 |> should equal ProofStrength.Strong
    ProofSystem.categorizeStrength 0.9 |> should equal ProofStrength.VeryStrong

[<Fact>]
let ``ProofSystem describe produces readable descriptions`` () =
    let proof = ReflectionProof.StatisticalEvidence(100, 0.95, 0.05)

    let desc = ProofSystem.describe proof

    desc |> should contain "Statistical"
    desc |> should contain "100"
    desc |> should contain "95"
