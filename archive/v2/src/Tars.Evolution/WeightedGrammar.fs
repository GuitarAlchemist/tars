namespace Tars.Evolution

/// Probabilistic grammar layer: attaches statistical weights to grammar rules.
/// Converts the deterministic 0-8 promotion scoring into soft probabilities,
/// enabling MDO-optimized weight evolution and Bayesian updates from execution feedback.
///
/// Architecture (from Compound Engineering notebook):
///   Force 1: Grammar constraints (EBNF, WoT DSL) -- guarantees structure
///   Force 2: PCFG-like weights (this module) -- steers preferences
///   Force 3: Semantic validation (GrammarGovernor) -- catches nonsense
module WeightedGrammar =

    open System
    open System.IO
    open System.Text.Json

    // =========================================================================
    // Types
    // =========================================================================

    /// Source system that produced/evolved a rule's weight
    type RuleSource =
        | Tars
        | GuitarAlchemist
        | MachinDeOuf
        | Evolved
        | Manual

    /// A grammar rule annotated with probabilistic weight metadata
    type WeightedRule = {
        PatternId: string
        PatternName: string
        Level: PromotionLevel
        /// Raw 0-8 criteria score from GrammarGovernor
        RawScore: int
        /// Current probability weight in [0.0, 1.0]
        Weight: float
        /// Bayesian confidence in the weight estimate (higher = more certain)
        Confidence: float
        /// Success rate from execution outcomes
        SuccessRate: float
        /// How many times this rule has been selected for use
        SelectionCount: int
        /// Who/what assigned this weight
        Source: RuleSource
        /// When the weight was last updated
        LastUpdated: DateTime
    }

    /// Configuration for weight computation
    type WeightConfig = {
        /// Softmax temperature: high = uniform, low = winner-take-all
        Temperature: float
        /// Bayesian decay factor for old observations (0.9 = slow decay)
        DecayFactor: float
        /// Floor weight — no rule drops below this
        MinWeight: float
        /// Prior success rate for new rules (Beta prior α/(α+β))
        PriorSuccessRate: float
    }

    let defaultConfig = {
        Temperature = 1.0
        DecayFactor = 0.95
        MinWeight = 0.01
        PriorSuccessRate = 0.5
    }

    // =========================================================================
    // Core math
    // =========================================================================

    /// Convert a 0-8 integer score to a [0.0, 1.0] logit
    let scoreToLogit (score: int) : float =
        float score / 8.0

    /// Softmax over a list of logits with temperature scaling.
    /// Returns probabilities that sum to ~1.0.
    let softmax (temperature: float) (logits: float list) : float list =
        if logits.IsEmpty then []
        else
            let t = max 0.01 temperature // prevent division by zero
            let scaled = logits |> List.map (fun x -> x / t)
            let maxVal = scaled |> List.max
            let exps = scaled |> List.map (fun x -> exp (x - maxVal)) // numerical stability
            let sumExps = exps |> List.sum
            if sumExps < 1e-15 then
                List.replicate logits.Length (1.0 / float logits.Length)
            else
                exps |> List.map (fun e -> e / sumExps)

    /// Bayesian update: Beta-Binomial posterior for success rate.
    /// Returns (updatedSuccessRate, updatedConfidence).
    let bayesianUpdate
        (priorRate: float)
        (priorCount: int)
        (success: bool)
        (decayFactor: float)
        : float * float =
        // Effective prior observations (decayed)
        let effectiveCount = float priorCount * decayFactor
        let alpha = priorRate * effectiveCount + (if success then 1.0 else 0.0)
        let beta = (1.0 - priorRate) * effectiveCount + (if success then 0.0 else 1.0)
        let newRate = alpha / (alpha + beta)
        let newConfidence = min 1.0 ((alpha + beta) / (alpha + beta + 10.0)) // asymptotic to 1
        (newRate, newConfidence)

    // =========================================================================
    // Weight management
    // =========================================================================

    /// Create WeightedRules from RecurrenceRecords with their governance scores.
    let fromRecurrenceRecords
        (config: WeightConfig)
        (records: (RecurrenceRecord * int) list) // (record, criteriaScore)
        : WeightedRule list =
        if records.IsEmpty then []
        else
            let logits = records |> List.map (fun (r, s) ->
                scoreToLogit s + r.AverageScore * 0.5) // blend criteria + execution score
            let weights = softmax config.Temperature logits
            (records, weights)
            ||> List.map2 (fun (r, s) w ->
                { PatternId = r.PatternId
                  PatternName = r.PatternName
                  Level = r.CurrentLevel
                  RawScore = s
                  Weight = max config.MinWeight w
                  Confidence = scoreToLogit s
                  SuccessRate = r.AverageScore
                  SelectionCount = 0
                  Source = Tars
                  LastUpdated = DateTime.UtcNow })

    /// Update a rule's weight after observing an execution outcome.
    let updateWeight
        (config: WeightConfig)
        (rule: WeightedRule)
        (success: bool)
        : WeightedRule =
        let newRate, newConf =
            bayesianUpdate rule.SuccessRate rule.SelectionCount success config.DecayFactor
        { rule with
            SuccessRate = newRate
            Confidence = newConf
            SelectionCount = rule.SelectionCount + 1
            LastUpdated = DateTime.UtcNow }

    /// Weighted random selection using accumulated probability.
    let selectWeighted (rules: WeightedRule list) (rng: Random) : WeightedRule option =
        if rules.IsEmpty then None
        else
            let total = rules |> List.sumBy (fun r -> r.Weight)
            if total < 1e-15 then Some rules.[rng.Next(rules.Length)]
            else
                let mutable target = rng.NextDouble() * total
                let mutable selected = None
                for r in rules do
                    if selected.IsNone then
                        target <- target - r.Weight
                        if target <= 0.0 then
                            selected <- Some r
                // Edge case: floating point — return last
                if selected.IsNone then Some (List.last rules)
                else selected

    /// Re-normalize weights within each promotion level so they sum to ~1.0.
    let normalizeByLevel (rules: WeightedRule list) : WeightedRule list =
        rules
        |> List.groupBy (fun r -> r.Level)
        |> List.collect (fun (_level, group) ->
            let total = group |> List.sumBy (fun r -> r.Weight)
            if total < 1e-15 then group
            else group |> List.map (fun r -> { r with Weight = r.Weight / total }))

    /// Evaluate a promotion candidate with probabilistic weight attachment.
    /// Wraps GrammarGovernor.evaluate and adds the weight.
    let evaluateWithWeight
        (existing: RecurrenceRecord list)
        (weights: WeightedRule list)
        (candidate: PromotionCandidate)
        : GovernanceDecision * float =
        let decision = GrammarGovernor.evaluate existing candidate
        let weight =
            weights
            |> List.tryFind (fun w -> w.PatternId = candidate.Record.PatternId)
            |> Option.map (fun w -> w.Weight)
            |> Option.defaultValue (scoreToLogit (GrammarGovernor.score candidate.Criteria))
        (decision, weight)

    // =========================================================================
    // Persistence (weights.json alongside recurrence.json)
    // =========================================================================

    /// DTO for JSON serialization
    type WeightedRuleDto = {
        PatternId: string
        PatternName: string
        Level: string
        RawScore: int
        Weight: float
        Confidence: float
        SuccessRate: float
        SelectionCount: int
        Source: string
        LastUpdated: string
    }

    let private toDto (r: WeightedRule) : WeightedRuleDto =
        { PatternId = r.PatternId
          PatternName = r.PatternName
          Level = PromotionLevel.label r.Level
          RawScore = r.RawScore
          Weight = r.Weight
          Confidence = r.Confidence
          SuccessRate = r.SuccessRate
          SelectionCount = r.SelectionCount
          Source = match r.Source with
                   | Tars -> "tars" | GuitarAlchemist -> "guitar_alchemist"
                   | MachinDeOuf -> "ix" | Evolved -> "evolved" | Manual -> "manual"
          LastUpdated = r.LastUpdated.ToString("o") }

    let private fromDto (dto: WeightedRuleDto) : WeightedRule =
        let level =
            match dto.Level with
            | "helper" -> Helper | "builder" -> Builder
            | "dsl_clause" -> DslClause | "grammar_rule" -> GrammarRule
            | _ -> Implementation
        let source =
            match dto.Source with
            | "guitar_alchemist" -> GuitarAlchemist | "ix" -> MachinDeOuf
            | "evolved" -> Evolved | "manual" -> Manual | _ -> Tars
        { PatternId = dto.PatternId
          PatternName = dto.PatternName
          Level = level
          RawScore = dto.RawScore
          Weight = dto.Weight
          Confidence = dto.Confidence
          SuccessRate = dto.SuccessRate
          SelectionCount = dto.SelectionCount
          Source = source
          LastUpdated = try DateTime.Parse(dto.LastUpdated) with _ -> DateTime.UtcNow }

    let private weightsDir =
        Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".tars", "promotion")

    let private weightsPath = Path.Combine(weightsDir, "weights.json")

    let private jsonOptions =
        let opts = JsonSerializerOptions()
        opts.WriteIndented <- true
        opts

    /// Save weighted rules to ~/.tars/promotion/weights.json
    let save (rules: WeightedRule list) : unit =
        try
            Directory.CreateDirectory(weightsDir) |> ignore
            let dtos = rules |> List.map toDto
            let json = JsonSerializer.Serialize(dtos, jsonOptions)
            File.WriteAllText(weightsPath, json)
        with _ -> () // graceful degradation

    /// Load weighted rules from ~/.tars/promotion/weights.json
    let load () : WeightedRule list =
        try
            if File.Exists(weightsPath) then
                let json = File.ReadAllText(weightsPath)
                let dtos = JsonSerializer.Deserialize<WeightedRuleDto list>(json, jsonOptions)
                dtos |> List.map fromDto
            else []
        with _ -> []
