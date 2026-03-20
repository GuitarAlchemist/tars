namespace Tars.Evolution

open System
open System.IO
open System.Text.Json
open ResearchTypes
open ResearchWeights

/// Executes research cycles following the scientific method grammar.
/// Samples production paths from department weights, evaluates outcomes,
/// updates weights, logs results and anomalies.
module ResearchCycle =

    type CycleConfig = {
        Department: string
        StateDir: string
        MaxDurationSeconds: int
        MaxCycles: int
    }

    let defaultConfig dept stateDir = {
        Department = dept
        StateDir = stateDir
        MaxDurationSeconds = 300
        MaxCycles = 12
    }

    let private jsonOptions =
        let opts = JsonSerializerOptions()
        opts.WriteIndented <- true
        opts

    /// Sample a production path from department weights
    let sampleProductionPath (weights: DepartmentWeights) (rng: Random) : ProductionPath =
        let hypName = sampleMethod weights.HypothesisWeights rng
        let testName = sampleMethod weights.TestWeights rng

        let hypMethod =
            match hypName with
            | "inductive" -> Inductive | "deductive" -> Deductive
            | "abductive" -> Abductive | "analogical" -> Analogical
            | _ -> Combinatorial

        let testMethod =
            match testName with
            | "empirical" -> Empirical | "formal_proof" -> FormalProof
            | "simulation" -> Simulation | "thought_experiment" -> ThoughtExperiment
            | "cross_validation" -> CrossValidation | _ -> Adversarial

        { HypothesisMethod = hypMethod
          TestMethod = testMethod
          Conclusion = Insufficient  // placeholder — set by actual execution
          Reflection = NormalProgress }

    /// Check if a cycle result represents an anomaly
    let isAnomaly (result: ResearchCycleResult) : bool =
        match result.BeliefValue with
        | "T" -> false
        | _ -> true  // F, U, C are all anomalous

    /// Create an anomaly entry from a failed research cycle
    let createAnomalyEntry (result: ResearchCycleResult) : AnomalyEntry =
        let path = result.Path
        { AnomalyId = $"anomaly-{result.Department}-{result.Timestamp:yyyyMMdd}-{Guid.NewGuid().ToString().[..7]}"
          CycleId = result.CycleId
          Department = result.Department
          ProductionPath = [
            hypothesisMethodName path.HypothesisMethod
            testMethodName path.TestMethod
            conclusionToTetraValue path.Conclusion
          ]
          Hypothesis = result.Hypothesis
          FailureMode = $"Concluded with {result.BeliefValue} (confidence {result.BeliefConfidence:F2})"
          DomainContext = []  // populated by caller with domain-specific context
          Severity = 1.0 - result.BeliefConfidence  // lower confidence = higher severity
          ClusterId = None
          ParadigmState = "normal" }

    /// Log a research cycle result to state directory
    let logCycle (stateDir: string) (result: ResearchCycleResult) : Result<unit, string> =
        try
            let dir = Path.Combine(stateDir, "streeling", "research")
            Directory.CreateDirectory(dir) |> ignore
            let dateStr = result.Timestamp.ToString("yyyy-MM-dd")
            let shortId = if result.CycleId.Length > 7 then result.CycleId.[..7] else result.CycleId
            let filename = $"{result.Department}-{dateStr}-{shortId}.cycle.json"
            let path = Path.Combine(dir, filename)
            let json = JsonSerializer.Serialize(result, jsonOptions)
            File.WriteAllText(path, json)
            Result.Ok ()
        with ex ->
            Result.Error $"Failed to log cycle: {ex.Message}"

    /// Log an anomaly entry to state directory
    let logAnomaly (stateDir: string) (entry: AnomalyEntry) : Result<unit, string> =
        try
            let dir = Path.Combine(stateDir, "streeling", "research")
            Directory.CreateDirectory(dir) |> ignore
            let path = Path.Combine(dir, $"{entry.Department}-anomalies.json")

            // Read existing anomalies and append
            let existing =
                if File.Exists(path) then
                    try
                        let json = File.ReadAllText(path)
                        JsonSerializer.Deserialize<AnomalyEntry list>(json, jsonOptions)
                    with _ -> []
                else []

            let updated = existing @ [entry]
            let json = JsonSerializer.Serialize(updated, jsonOptions)
            File.WriteAllText(path, json)
            Result.Ok ()
        with ex ->
            Result.Error $"Failed to log anomaly: {ex.Message}"
