namespace Tars.Evolution

open System

/// Domain types for Streeling University's probabilistic research grammars.
/// Maps the universal scientific method EBNF grammar to F# types.
module ResearchTypes =

    type HypothesisMethod = Inductive | Deductive | Abductive | Analogical | Combinatorial

    type TestMethod = Empirical | FormalProof | Simulation | ThoughtExperiment | CrossValidation | Adversarial

    type Conclusion = Confirm | Refute | Insufficient | Contradictory | Revise | DiscoverQuestion

    type ReflectionOutcome = NormalProgress | AnomalyDetected | ParadigmTension | ParadigmShift

    type ProductionPath = {
        HypothesisMethod: HypothesisMethod
        TestMethod: TestMethod
        Conclusion: Conclusion
        Reflection: ReflectionOutcome
    }

    type ResearchCycleResult = {
        CycleId: string
        Department: string
        Question: string
        Path: ProductionPath
        Hypothesis: string
        Evidence: string list
        BeliefValue: string   // "T", "F", "U", "C"
        BeliefConfidence: float
        DurationSeconds: int
        Timestamp: DateTime
    }

    type DepartmentWeights = {
        Department: string
        HypothesisWeights: Map<string, float>
        TestWeights: Map<string, float>
        CycleCount: int
        LastUpdated: DateTime
    }

    type AnomalyEntry = {
        AnomalyId: string
        CycleId: string
        Department: string
        ProductionPath: string list
        Hypothesis: string
        FailureMode: string
        DomainContext: string list
        Severity: float
        ClusterId: string option
        ParadigmState: string
    }

    let hypothesisMethodName = function
        | Inductive -> "inductive" | Deductive -> "deductive"
        | Abductive -> "abductive" | Analogical -> "analogical"
        | Combinatorial -> "combinatorial"

    let testMethodName = function
        | Empirical -> "empirical" | FormalProof -> "formal_proof"
        | Simulation -> "simulation" | ThoughtExperiment -> "thought_experiment"
        | CrossValidation -> "cross_validation" | Adversarial -> "adversarial"

    let conclusionToTetraValue = function
        | Confirm -> "T" | Refute -> "F" | Insufficient -> "U"
        | Contradictory -> "C" | Revise -> "U" | DiscoverQuestion -> "U"
