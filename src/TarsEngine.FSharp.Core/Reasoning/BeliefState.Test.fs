namespace TarsEngine.FSharp.Core.Reasoning.Tests

open TarsEngine.FSharp.Core.Reasoning.BeliefState
open System

/// Basic test suite for BeliefState module
module BeliefStateTests =

    let runTests () =
        // Test 1: Create basic belief
        let belief1 = createBelief "The sky is blue" T 0.95
        assert (belief1.proposition = "The sky is blue")
        assert (belief1.truth_value = T)
        assert (belief1.confidence = 0.95)

        // Test 2: Update belief
        let belief2 = updateBelief F 0.7 belief1
        assert (belief2.truth_value = F)
        assert (belief2.confidence = 0.7)

        // Test 3: Add evidence
        let evidence1 = createEvidence "telescope_observation" "Observed blue wavelengths" 0.9
        let belief3 = addSupportingEvidence evidence1 belief1
        assert (belief3.evidence.IsSome)

        // Test 4: Assess confidence
        let evidence2 = createEvidence "expert_report" "Colorimetry confirms blue" 0.85
        let belief4 =
            belief1
            |> addSupportingEvidence evidence1
            |> addSupportingEvidence evidence2
        let assessed = assessConfidence belief4
        assert (assessed > 0.0 && assessed <= 1.0)

        // Test 5: Detect stale beliefs
        let isStale = detectStale 1 belief1  // 1 hour threshold
        assert (not isStale)  // Just created, shouldn't be stale

        // Test 6: Reconcile contradictions
        let evidence3 = createEvidence "report_a" "Sky is blue" 0.8
        let evidence4 = createEvidence "report_b" "Sky is not blue" 0.7
        let belief5 =
            belief1
            |> addSupportingEvidence evidence3
            |> addContradictingEvidence evidence4
            |> reconcileContradictions
        assert (belief5.truth_value = C)  // Should be Contradictory

        // Test 7: Generate compliance report
        let report = generateComplianceReport belief1
        assert (report.ContainsKey "proposition")
        assert (report.ContainsKey "truth_value")
        assert (report.ContainsKey "confidence")

        // Test 8: Filter beliefs by truth value
        let beliefs = [
            createBelief "A" T 0.9
            createBelief "B" F 0.8
            createBelief "C" U 0.5
            createBelief "D" T 0.7
        ]
        let trueBeliefs = filterByTruthValue T beliefs
        assert (trueBeliefs.Length = 2)

        // Test 9: Truth value to/from string
        assert (truthValueToString T = "T")
        assert (stringToTruthValue "F" = F)
        assert (stringToTruthValue "U" = U)
        assert (stringToTruthValue "C" = C)

        // Test 10: Evaluator tracking
        let belief6 = belief1 |> withEvaluator "validator_agent"
        assert (belief6.evaluated_by = Some "validator_agent")

        // All tests passed
        printfn "✓ All BeliefState tests passed"

    // Run tests on module load
    let _ = runTests ()
