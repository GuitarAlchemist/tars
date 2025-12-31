namespace Tars.Core

open System

// =============================================================================
// PHASE 15.4: SYMBOLIC PROOF SYSTEM
// =============================================================================
//
// Formal validation of belief updates through symbolic proofs.
// Provides proof verification, proof search, and strength calculation.
// Reference: docs/3_Roadmap/2_Phases/phase_15_symbolic_reflection.md

/// Error that can occur during proof verification
type ProofError =
    | InvalidPremises of reason: string
    | CircularReasoning of involvedIds: Guid list
    | InsufficientSamples of required: int * actual: int
    | InvalidConfidenceInterval of value: float
    | RuleMismatch of expected: ReflectionInferenceRule * found: ReflectionInferenceRule
    | TautologyFalse of statement: string
    | ContradictionTrue of statement: string
    | UnverifiableAssertion of source: string

/// Strength classification for proofs
type ProofStrength =
    | VeryWeak // 0.0 - 0.2
    | Weak // 0.2 - 0.4
    | Moderate // 0.4 - 0.6
    | Strong // 0.6 - 0.8
    | VeryStrong // 0.8 - 1.0

/// Result of proof verification
type ProofVerificationResult =
    { Proof: ReflectionProof
      IsValid: bool
      Strength: float
      StrengthCategory: ProofStrength
      Errors: ProofError list
      VerifiedAt: DateTimeOffset }

module ProofSystem =

    /// Calculate the strength of a proof (0.0 - 1.0)
    let strengthOf (proof: ReflectionProof) : float =
        match proof with
        | Tautology _ -> 1.0 // Tautologies are maximally strong
        | ProofContradiction _ -> 0.0 // Contradictions have no strength
        | ValidationSuccess(_, _) -> 0.95 // Passing tests are very strong
        | ValidationFailure(_, _) -> 0.1 // Failing tests provide weak negative evidence
        | LogicalInference(premises, _, rule) ->
            // Strength depends on the rule and number of premises
            let ruleWeight =
                match rule with
                | ModusPonens -> 0.9
                | ModusTollens -> 0.85
                | Syllogism -> 0.8
                | Contraposition -> 0.85
                | Generalization -> 0.6 // Inductive, weaker
                | Specialization -> 0.9
                | Abduction -> 0.5 // Speculative
                | Analogy -> 0.4 // Weakest logical form
                | StatisticalInference -> 0.7

            let premiseCount = float premises.Length

            let premisePenalty =
                if premiseCount > 3.0 then
                    0.1 * (premiseCount - 3.0)
                else
                    0.0

            max 0.0 (ruleWeight - premisePenalty)
        | StatisticalEvidence(samples, successRate, confidenceInterval) ->
            // Strength based on sample size and success rate
            let sampleWeight = min 1.0 (float samples / 100.0)
            let rateWeight = successRate
            let ciPenalty = confidenceInterval * 0.1 // Wider CI = weaker
            sampleWeight * rateWeight * (1.0 - ciPenalty)
        | ExpertAssertion(_, credibility) -> credibility * 0.8 // Expert assertions capped at 0.8

    /// Categorize proof strength
    let categorizeStrength (strength: float) : ProofStrength =
        if strength < 0.2 then VeryWeak
        elif strength < 0.4 then Weak
        elif strength < 0.6 then Moderate
        elif strength < 0.8 then Strong
        else VeryStrong

    /// Verify a proof is valid
    let verifyProof (proof: ReflectionProof) : Result<ProofVerificationResult, ProofError list> =
        let errors = ResizeArray<ProofError>()

        match proof with
        | Tautology statement when System.String.IsNullOrWhiteSpace(statement) ->
            errors.Add(TautologyFalse "Empty tautology statement")
        | ProofContradiction statement when System.String.IsNullOrWhiteSpace(statement) ->
            errors.Add(ContradictionTrue "Empty contradiction statement")
        | LogicalInference(premises, conclusion, _) ->
            if premises.IsEmpty then
                errors.Add(InvalidPremises "No premises provided")

            if System.String.IsNullOrWhiteSpace(conclusion) then
                errors.Add(InvalidPremises "Empty conclusion")
        | StatisticalEvidence(samples, successRate, confidenceInterval) ->
            if samples < 1 then
                errors.Add(InsufficientSamples(1, samples))

            if successRate < 0.0 || successRate > 1.0 then
                errors.Add(InvalidConfidenceInterval successRate)

            if confidenceInterval < 0.0 || confidenceInterval > 1.0 then
                errors.Add(InvalidConfidenceInterval confidenceInterval)
        | ExpertAssertion(source, credibility) ->
            if System.String.IsNullOrWhiteSpace(source) then
                errors.Add(UnverifiableAssertion "No source provided")

            if credibility < 0.0 || credibility > 1.0 then
                errors.Add(InvalidConfidenceInterval credibility)
        | _ -> ()

        let strength = strengthOf proof

        let result =
            { Proof = proof
              IsValid = errors.Count = 0
              Strength = strength
              StrengthCategory = categorizeStrength strength
              Errors = errors |> Seq.toList
              VerifiedAt = DateTimeOffset.UtcNow }

        if errors.Count = 0 then
            FSharp.Core.Ok result
        else
            FSharp.Core.Error(errors |> Seq.toList)

    /// Check if a proof supports a belief
    let supports (proof: ReflectionProof) (belief: ReflectionBelief) : bool =
        match proof with
        | Tautology statement -> belief.Statement.ToLowerInvariant().Contains(statement.ToLowerInvariant())
        | ProofContradiction _ -> false // Contradictions never support
        | ValidationSuccess(testName, _) ->
            belief.Statement.ToLowerInvariant().Contains(testName.ToLowerInvariant())
            || belief.Tags
               |> List.exists (fun t -> t.ToLowerInvariant() = testName.ToLowerInvariant())
        | ValidationFailure _ -> false // Failures don't support
        | LogicalInference(_, conclusion, _) ->
            belief.Statement.ToLowerInvariant().Contains(conclusion.ToLowerInvariant())
        | StatisticalEvidence _ -> true // Generic statistical support
        | ExpertAssertion _ -> true // Generic expert support

    /// Find all proofs that support a belief from a collection
    let findProofs (belief: ReflectionBelief) (proofs: ReflectionProof list) : ReflectionProof list =
        proofs |> List.filter (fun p -> supports p belief)

    /// Combine multiple proofs for stronger overall support
    let combineProofs (proofs: ReflectionProof list) : float =
        if proofs.IsEmpty then
            0.0
        else
            // Use Dempster-Shafer style combination (simplified)
            let strengths = proofs |> List.map strengthOf
            let combined = strengths |> List.fold (fun acc s -> acc + s * (1.0 - acc)) 0.0
            min 1.0 combined

    /// Create a logical inference proof
    let createInference (premises: string list) (conclusion: string) (rule: ReflectionInferenceRule) : ReflectionProof =
        LogicalInference(premises, conclusion, rule)

    /// Create a statistical evidence proof
    let createStatistical (samples: int) (successRate: float) (confidence: float) : ReflectionProof =
        StatisticalEvidence(samples, successRate, confidence)

    /// Create a validation proof from test result
    let createFromTest (testName: string) (passed: bool) (details: string) : ReflectionProof =
        if passed then
            ValidationSuccess(testName, details)
        else
            ValidationFailure(testName, details)

    /// Render proof as human-readable string
    let describe (proof: ReflectionProof) : string =
        match proof with
        | Tautology statement -> sprintf "Tautology: \"%s\" is self-evidently true" statement
        | ProofContradiction statement -> sprintf "Contradiction: \"%s\" leads to contradiction" statement
        | ValidationSuccess(test, details) -> sprintf "Test Passed: %s - %s" test details
        | ValidationFailure(test, error) -> sprintf "Test Failed: %s - %s" test error
        | LogicalInference(premises, conclusion, rule) ->
            let ruleStr =
                match rule with
                | ModusPonens -> "Modus Ponens"
                | ModusTollens -> "Modus Tollens"
                | Syllogism -> "Syllogism"
                | Contraposition -> "Contraposition"
                | Generalization -> "Generalization"
                | Specialization -> "Specialization"
                | Abduction -> "Abduction"
                | Analogy -> "Analogy"
                | StatisticalInference -> "Statistical Inference"

            sprintf "Inference (%s): From [%s] → %s" ruleStr (String.concat "; " premises) conclusion
        | StatisticalEvidence(samples, rate, ci) ->
            sprintf "Statistical: %d samples, %.1f%% success rate, ±%.1f%% CI" samples (rate * 100.0) (ci * 100.0)
        | ExpertAssertion(source, credibility) ->
            sprintf "Expert (%s, credibility: %.0f%%)" source (credibility * 100.0)

    /// Assess if proof is sufficient for a high-confidence belief
    let isSufficient (proof: ReflectionProof) (requiredStrength: float) : bool = strengthOf proof >= requiredStrength

    /// Get proof category description
    let categoryDescription (category: ProofStrength) : string =
        match category with
        | VeryWeak -> "Very Weak - Barely supports the claim"
        | Weak -> "Weak - Provides minimal support"
        | Moderate -> "Moderate - Reasonable support"
        | Strong -> "Strong - Good evidentiary support"
        | VeryStrong -> "Very Strong - Excellent evidentiary support"
