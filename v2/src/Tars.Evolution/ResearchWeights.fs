namespace Tars.Evolution

open System
open System.IO
open System.Text.Json
open ResearchTypes

/// Load, save, and update per-department research weight profiles.
/// Reads from Demerzel's state/streeling/departments/{dept}.weights.json.
module ResearchWeights =

    // DTO for JSON serialization
    type HypothesisWeightsDto = {
        inductive: float
        deductive: float
        abductive: float
        analogical: float
        combinatorial: float
    }

    type TestWeightsDto = {
        empirical: float
        formal_proof: float
        simulation: float
        thought_experiment: float
        cross_validation: float
        adversarial: float
    }

    type MetadataDto = {
        last_updated: string
        cycle_count: int
        total_T: int
        total_F: int
        total_U: int
        total_C: int
    }

    type WeightProfileDto = {
        department: string
        grammar_version: string
        hypothesis_weights: HypothesisWeightsDto
        test_weights: TestWeightsDto
        metadata: MetadataDto
    }

    let private jsonOptions =
        let opts = JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower)
        opts.WriteIndented <- true
        opts

    let private hypothesisDtoToMap (dto: HypothesisWeightsDto) : Map<string, float> =
        Map.ofList [
            "inductive", dto.inductive; "deductive", dto.deductive
            "abductive", dto.abductive; "analogical", dto.analogical
            "combinatorial", dto.combinatorial
        ]

    let private testDtoToMap (dto: TestWeightsDto) : Map<string, float> =
        Map.ofList [
            "empirical", dto.empirical; "formal_proof", dto.formal_proof
            "simulation", dto.simulation; "thought_experiment", dto.thought_experiment
            "cross_validation", dto.cross_validation; "adversarial", dto.adversarial
        ]

    /// Load department weights from Demerzel state directory
    let load (stateDir: string) (department: string) : Result<DepartmentWeights, string> =
        let path = Path.Combine(stateDir, "streeling", "departments", $"{department}.weights.json")
        try
            if File.Exists(path) then
                let json = File.ReadAllText(path)
                let dto = JsonSerializer.Deserialize<WeightProfileDto>(json, jsonOptions)
                Result.Ok {
                    Department = dto.department
                    HypothesisWeights = hypothesisDtoToMap dto.hypothesis_weights
                    TestWeights = testDtoToMap dto.test_weights
                    CycleCount = dto.metadata.cycle_count
                    LastUpdated = try DateTime.Parse(dto.metadata.last_updated) with _ -> DateTime.UtcNow
                }
            else
                Result.Error $"Weight profile not found: {path}"
        with ex ->
            Result.Error $"Failed to load weights: {ex.Message}"

    /// Weighted random selection from a weight map
    let sampleMethod (weights: Map<string, float>) (rng: Random) : string =
        let items = weights |> Map.toList
        let total = items |> List.sumBy snd
        if total < 1e-15 then
            items.[rng.Next(items.Length)] |> fst
        else
            let mutable target = rng.NextDouble() * total
            let mutable selected = None
            for (name, weight) in items do
                if selected.IsNone then
                    target <- target - weight
                    if target <= 0.0 then
                        selected <- Some name
            selected |> Option.defaultValue (items |> List.last |> fst)

    /// Bayesian update: increase weight for successful method, decrease for failed.
    /// Uses Beta-Binomial posterior similar to WeightedGrammar.bayesianUpdate.
    let private bayesianAdjust (currentWeight: float) (success: bool) (decayFactor: float) : float =
        let adjustment = if success then 0.05 else -0.03
        max 0.01 (currentWeight * decayFactor + adjustment)

    /// Update weights after observing a research cycle outcome
    let updateFromOutcome
        (weights: DepartmentWeights)
        (hypothesisMethod: string)
        (testMethod: string)
        (success: bool)
        : DepartmentWeights =
        let decay = 0.95
        // Update hypothesis weights
        let newHyp =
            weights.HypothesisWeights
            |> Map.map (fun name w ->
                if name = hypothesisMethod then bayesianAdjust w success decay
                else w)
        // Normalize hypothesis weights
        let hypTotal = newHyp |> Map.toList |> List.sumBy snd
        let normHyp = newHyp |> Map.map (fun _ w -> w / hypTotal)

        // Update test weights
        let newTest =
            weights.TestWeights
            |> Map.map (fun name w ->
                if name = testMethod then bayesianAdjust w success decay
                else w)
        // Normalize test weights
        let testTotal = newTest |> Map.toList |> List.sumBy snd
        let normTest = newTest |> Map.map (fun _ w -> w / testTotal)

        { weights with
            HypothesisWeights = normHyp
            TestWeights = normTest
            CycleCount = weights.CycleCount + 1
            LastUpdated = DateTime.UtcNow }
