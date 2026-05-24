namespace Tars.Evolution

/// Grammar × ML Bridge: marry TARS grammar distillation with ix ML pipelines.
///
/// Three integration points:
///   1. Feature extraction — TypedProduction → numeric feature vector → ix training data
///   2. Predictive priors — ix_ml_predict → Bayesian prior for new productions
///   3. Grammar breeding — ix_evolution → breed new productions from top performers
///
/// Architecture:
///   TypedProduction → extractFeatures → ix_ml_pipeline (train)
///                                     → ix_ml_predict  (score new productions)
///                                     → ix_evolution   (breed novel productions)
///                   → WeightedGrammar (ix prediction as Bayesian prior)

open System
open System.Text.Json
open Tars.Llm

// ─── Types ───────────────────────────────────────────────────────────────────

/// Numeric feature vector extracted from a TypedProduction.
type ProductionFeatures = {
    /// Number of nodes/slots
    SlotCount: int
    /// Number of edges
    EdgeCount: int
    /// Ratio of reason nodes to total
    ReasonRatio: float
    /// Ratio of work nodes to total
    WorkRatio: float
    /// Number of distinct tools used
    ToolDiversity: int
    /// Number of behavioral constraints
    ConstraintCount: int
    /// Whether types compose (1.0 or 0.0)
    TypeComposable: float
    /// Number of distinct input types
    InputTypeDiversity: int
    /// Number of distinct output types
    OutputTypeDiversity: int
    /// Compression ratio from distillation
    CompressionRatio: float
    /// Current success rate (target variable for training)
    SuccessRate: float
}

/// Result from ix ML prediction.
type MlPrediction = {
    PredictedSuccess: float
    Confidence: float
}

/// A bred production candidate from ix_evolution.
type BredCandidate = {
    /// Features of the bred candidate
    Features: ProductionFeatures
    /// Fitness score from ix
    Fitness: float
    /// Which parent productions contributed
    ParentIds: string list
}

/// Result of a breeding round.
type BreedingResult = {
    Candidates: BredCandidate list
    GenerationsRun: int
    BestFitness: float
}

/// Configuration for ML-informed grammar evolution.
type GrammarMlConfig = {
    /// Success rate threshold for binary classification (good/bad)
    SuccessThreshold: float
    /// Key for persisting the trained model in ix
    PersistKey: string
    /// Whether to breed new candidates via ix_evolution
    Breed: bool
    /// Number of generations for breeding
    Generations: int
}

// ─── Feature extraction ──────────────────────────────────────────────────────

module GrammarMlBridge =

    /// Extract a numeric feature vector from a TypedProduction.
    let extractFeatures (production: TypedProduction) : ProductionFeatures =
        let mutable slotCount = 0
        let mutable edgeCount = 0
        let mutable reasonCount = 0
        let mutable workCount = 0
        let mutable toolDiversity = 0
        let mutable constraintCount = 0
        let mutable typeComposable = 0.0
        let mutable inputTypes = Set.empty
        let mutable outputTypes = Set.empty

        for facet in production.Facets do
            match facet with
            | Structural (slots, edges) ->
                slotCount <- slots.Length
                edgeCount <- edges.Length
                reasonCount <- slots |> List.filter (fun s -> s.Kind = "reason") |> List.length
                workCount <- slots |> List.filter (fun s -> s.Kind = "work") |> List.length
                for slot in slots do
                    inputTypes <- inputTypes.Add(slot.InputType)
                    outputTypes <- outputTypes.Add(slot.OutputType)

            | Typed (_, composable) ->
                typeComposable <- if composable then 1.0 else 0.0

            | Behavioral (conditions, tools) ->
                constraintCount <- conditions.Length
                toolDiversity <- tools.Length

        let total = max 1 slotCount
        { SlotCount = slotCount
          EdgeCount = edgeCount
          ReasonRatio = float reasonCount / float total
          WorkRatio = float workCount / float total
          ToolDiversity = toolDiversity
          ConstraintCount = constraintCount
          TypeComposable = typeComposable
          InputTypeDiversity = inputTypes.Count
          OutputTypeDiversity = outputTypes.Count
          CompressionRatio = production.CompressionRatio
          SuccessRate = production.SuccessRate }

    // =========================================================================
    // ix data formatting
    // =========================================================================

    /// Column headers for ix_ml_pipeline CSV format.
    let csvHeaders =
        "slot_count,edge_count,reason_ratio,work_ratio,tool_diversity,constraint_count,type_composable,input_type_diversity,output_type_diversity,compression_ratio,success_rate"

    /// Convert features to a CSV row.
    let featuresToCsvRow (f: ProductionFeatures) : string =
        $"{f.SlotCount},{f.EdgeCount},{f.ReasonRatio:F3},{f.WorkRatio:F3},{f.ToolDiversity},{f.ConstraintCount},{f.TypeComposable:F1},{f.InputTypeDiversity},{f.OutputTypeDiversity},{f.CompressionRatio:F2},{f.SuccessRate:F3}"

    /// Convert features to a JSON array row (for ix_ml_pipeline inline data).
    let featuresToJsonRow (f: ProductionFeatures) : float list =
        [ float f.SlotCount
          float f.EdgeCount
          f.ReasonRatio
          f.WorkRatio
          float f.ToolDiversity
          float f.ConstraintCount
          f.TypeComposable
          float f.InputTypeDiversity
          float f.OutputTypeDiversity
          f.CompressionRatio
          f.SuccessRate ]

    /// Feature names (excluding target).
    let featureNames =
        [ "slot_count"; "edge_count"; "reason_ratio"; "work_ratio"; "tool_diversity"
          "constraint_count"; "type_composable"; "input_type_diversity"
          "output_type_diversity"; "compression_ratio" ]

    // =========================================================================
    // Integration 1: Train a model from production history
    // =========================================================================

    /// Build ix_ml_pipeline args to train a classifier on production features.
    /// Target: success_rate >= threshold → "good" (1) vs "bad" (0).
    let buildTrainArgs
        (productions: TypedProduction list)
        (successThreshold: float)
        (persistKey: string)
        : string =

        let rows =
            productions
            |> List.map (fun p ->
                let f = extractFeatures p
                let featureRow = featuresToJsonRow f |> List.take 10 // exclude target
                let label = if f.SuccessRate >= successThreshold then 1.0 else 0.0
                featureRow @ [label])

        let dataJson = JsonSerializer.Serialize(rows)
        let colNames = featureNames @ ["label"] |> JsonSerializer.Serialize

        $"""{{ "task": "classify", "model": "random-forest", "data": {dataJson}, "column_names": {colNames}, "target": "label", "persist_key": "{persistKey}" }}"""

    // =========================================================================
    // Integration 2: Predict success for new productions
    // =========================================================================

    /// Build ix_ml_predict args for a new production.
    let buildPredictArgs (production: TypedProduction) (persistKey: string) : string =
        let f = extractFeatures production
        let row = featuresToJsonRow f |> List.take 10
        let dataJson = JsonSerializer.Serialize([row])
        $"""{{ "persist_key": "{persistKey}", "data": {dataJson} }}"""

    /// Parse ix_ml_predict result into a prediction.
    let parsePrediction (ixJson: string) : Result<MlPrediction, string> =
        match JsonParsing.tryParseElement ixJson with
        | Error e -> Error $"Failed to parse ix prediction: {e}"
        | Ok elem ->
            try
                // ix returns: { "predictions": [1], "probabilities": [[0.2, 0.8]] }
                let predictions = elem.GetProperty("predictions")
                let predicted = predictions.[0].GetDouble()

                let confidence =
                    try
                        let probs = elem.GetProperty("probabilities")
                        let probArray = probs.[0]
                        let maxProb = [0 .. probArray.GetArrayLength() - 1]
                                      |> List.map (fun i -> probArray.[i].GetDouble())
                                      |> List.max
                        maxProb
                    with _ -> 0.5

                Ok { PredictedSuccess = predicted; Confidence = confidence }
            with ex ->
                Error $"Failed to extract prediction: {ex.Message}"

    /// Use ix prediction as a Bayesian prior for a new production's weight.
    let applyPredictivePrior
        (prediction: MlPrediction)
        (rule: WeightedGrammar.WeightedRule)
        : WeightedGrammar.WeightedRule =

        // Blend ix prediction with default prior, weighted by ix confidence
        let blendedRate =
            prediction.PredictedSuccess * prediction.Confidence
            + rule.SuccessRate * (1.0 - prediction.Confidence)

        { rule with
            SuccessRate = blendedRate
            Confidence = max rule.Confidence (prediction.Confidence * 0.5)
            Source = WeightedGrammar.Evolved }

    // =========================================================================
    // Integration 3: Breed new productions via ix_evolution
    // =========================================================================

    /// Encode productions as genomes for ix_evolution.
    /// Each genome is the feature vector of a production.
    let encodeGenomes (productions: TypedProduction list) : float list list =
        productions
        |> List.map (fun p -> extractFeatures p |> featuresToJsonRow |> List.take 10)

    /// Build ix_evolution args for breeding new grammar productions.
    let buildBreedArgs
        (productions: TypedProduction list)
        (generations: int)
        (populationSize: int)
        : string =

        let genomes = encodeGenomes productions |> JsonSerializer.Serialize
        $"""{{ "algorithm": "genetic", "function": "custom", "initial_population": {genomes}, "generations": {generations}, "population_size": {populationSize}, "dimension": 10 }}"""

    /// Parse ix_evolution result into bred candidates.
    let parseBreedResult (ixJson: string) (parentIds: string list) : Result<BreedingResult, string> =
        match JsonParsing.tryParseElement ixJson with
        | Error e -> Error $"Failed to parse ix evolution result: {e}"
        | Ok elem ->
            try
                let bestValue =
                    try elem.GetProperty("best_value").GetDouble()
                    with _ -> 0.0

                let generations =
                    try elem.GetProperty("iterations").GetInt32()
                    with _ -> 0

                // Parse best_params as a feature vector
                let bestParams =
                    try
                        let arr = elem.GetProperty("best_params")
                        [0 .. arr.GetArrayLength() - 1]
                        |> List.map (fun i -> arr.[i].GetDouble())
                    with _ -> []

                let candidate =
                    if bestParams.Length >= 10 then
                        let features = {
                            SlotCount = max 1 (int (abs bestParams.[0]))
                            EdgeCount = max 0 (int (abs bestParams.[1]))
                            ReasonRatio = max 0.0 (min 1.0 bestParams.[2])
                            WorkRatio = max 0.0 (min 1.0 bestParams.[3])
                            ToolDiversity = max 0 (int (abs bestParams.[4]))
                            ConstraintCount = max 0 (int (abs bestParams.[5]))
                            TypeComposable = if bestParams.[6] >= 0.5 then 1.0 else 0.0
                            InputTypeDiversity = max 1 (int (abs bestParams.[7]))
                            OutputTypeDiversity = max 1 (int (abs bestParams.[8]))
                            CompressionRatio = max 1.0 bestParams.[9]
                            SuccessRate = 0.0 // unknown — needs execution
                        }
                        [{ Features = features; Fitness = bestValue; ParentIds = parentIds }]
                    else []

                Ok { Candidates = candidate
                     GenerationsRun = generations
                     BestFitness = bestValue }
            with ex ->
                Error $"Failed to parse breed result: {ex.Message}"

    // =========================================================================
    // Orchestration: full ML-informed grammar evolution cycle
    // =========================================================================

    /// Run a full ML-informed grammar evolution cycle:
    /// 1. Extract features from all productions
    /// 2. Train classifier via ix
    /// 3. Predict success for untested productions
    /// 4. Apply predictive priors to WeightedGrammar
    /// 5. Optionally breed new candidates
    let evolveAsync
        (callIx: IxCaller)
        (productions: TypedProduction list)
        (existingWeights: WeightedGrammar.WeightedRule list)
        (config: GrammarMlConfig)
        : Async<Result<WeightedGrammar.WeightedRule list * BreedingResult option, string>> =
        async {
            if productions.Length < 3 then
                return Error "Need at least 3 productions to train a model"
            else

            // Step 1: Train classifier
            let trainArgs = buildTrainArgs productions config.SuccessThreshold config.PersistKey
            let! trainResult = callIx "ix_ml_pipeline" trainArgs

            match trainResult with
            | Error e -> return Error $"Training failed: {e}"
            | Ok _ ->

            // Step 2: Score all productions and apply predictive priors
            let mutable updatedWeights = existingWeights

            for production in productions do
                let predictArgs = buildPredictArgs production config.PersistKey
                let! predResult = callIx "ix_ml_predict" predictArgs

                match predResult with
                | Ok predJson ->
                    match parsePrediction predJson with
                    | Ok prediction ->
                        updatedWeights <-
                            updatedWeights
                            |> List.map (fun w ->
                                if w.PatternId = production.Id then
                                    applyPredictivePrior prediction w
                                else w)
                    | Error _ -> () // skip if parse fails
                | Error _ -> () // skip if ix unavailable

            // Step 3: Optionally breed new candidates
            let! breedResult =
                if config.Breed && productions.Length >= 2 then
                    async {
                        let breedArgs = buildBreedArgs productions config.Generations 20
                        let parentIds = productions |> List.map (fun p -> p.Id)
                        let! ixResult = callIx "ix_evolution" breedArgs
                        match ixResult with
                        | Ok json ->
                            match parseBreedResult json parentIds with
                            | Ok result -> return Some result
                            | Error _ -> return None
                        | Error _ -> return None
                    }
                else async { return None }

            return Ok (updatedWeights, breedResult)
        }
