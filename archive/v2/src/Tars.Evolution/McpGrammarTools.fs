namespace Tars.Evolution

/// MCP tools for probabilistic grammar operations.
/// Exposes grammar weighting, replicator dynamics, and MCTS search
/// via the TARS MCP server for use by Claude Code and other clients.
module McpGrammarTools =

    open System
    open System.Text.Json
    open System.Text.Json.Serialization
    open Tars.Core

    let private jsonOptions =
        let o = JsonSerializerOptions(WriteIndented = true)
        o.Converters.Add(JsonFSharpConverter())
        o

    // =========================================================================
    // Tool: grammar_weights — View grammar rule weights
    // =========================================================================

    type WeightSummary = {
        PatternId: string
        PatternName: string
        Level: string
        Weight: float
        SuccessRate: float
        Confidence: float
        SelectionCount: int
    }

    type WeightsResponse = {
        RuleCount: int
        Rules: WeightSummary list
        TotalWeight: float
    }

    let private grammarWeights (_input: string) : Result<string, string> =
        try
            let rules = WeightedGrammar.load ()
            let summaries =
                rules |> List.map (fun r ->
                    { PatternId = r.PatternId
                      PatternName = r.PatternName
                      Level = PromotionLevel.label r.Level
                      Weight = r.Weight
                      SuccessRate = r.SuccessRate
                      Confidence = r.Confidence
                      SelectionCount = r.SelectionCount })
            let response = {
                RuleCount = rules.Length
                Rules = summaries
                TotalWeight = rules |> List.sumBy (fun r -> r.Weight)
            }
            Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to load grammar weights: {ex.Message}"

    // =========================================================================
    // Tool: grammar_update — Bayesian update a rule from execution outcome
    // =========================================================================

    type UpdateInput = {
        PatternId: string
        Success: bool
    }

    type UpdateResponse = {
        PatternId: string
        OldWeight: float
        NewWeight: float
        OldSuccessRate: float
        NewSuccessRate: float
        NewConfidence: float
    }

    let private grammarUpdate (input: string) : Result<string, string> =
        try
            let req = JsonSerializer.Deserialize<UpdateInput>(input, jsonOptions)
            let rules = WeightedGrammar.load ()
            match rules |> List.tryFind (fun r -> r.PatternId = req.PatternId) with
            | None -> Result.Error $"Rule not found: {req.PatternId}"
            | Some rule ->
                let updated = WeightedGrammar.updateWeight WeightedGrammar.defaultConfig rule req.Success
                let newRules = rules |> List.map (fun r ->
                    if r.PatternId = req.PatternId then updated else r)
                WeightedGrammar.save newRules
                let response = {
                    PatternId = req.PatternId
                    OldWeight = rule.Weight
                    NewWeight = updated.Weight
                    OldSuccessRate = rule.SuccessRate
                    NewSuccessRate = updated.SuccessRate
                    NewConfidence = updated.Confidence
                }
                Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to update grammar weight: {ex.Message}"

    // =========================================================================
    // Tool: grammar_evolve — Run replicator dynamics on grammar ecosystem
    // =========================================================================

    type EvolveResponse = {
        SpeciesCount: int
        StableCount: int
        PrunedCount: int
        StepsRun: int
        Species: ReplicatorDynamics.GrammarSpecies list
    }

    let private grammarEvolve (_input: string) : Result<string, string> =
        try
            let rules = WeightedGrammar.load ()
            if rules.IsEmpty then
                Result.Error "No grammar rules loaded. Run promotion pipeline first."
            else
                let result = ReplicatorDynamics.evolveEcosystem rules Map.empty
                let response = {
                    SpeciesCount = result.Species.Length
                    StableCount = result.Stable.Length
                    PrunedCount = result.Pruned.Length
                    StepsRun = result.StepsRun
                    Species = result.Species
                }
                Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to run replicator dynamics: {ex.Message}"

    // =========================================================================
    // Tool: grammar_search — MCTS search for WoT derivation
    // =========================================================================

    type SearchInput = {
        MaxIterations: int option
        MaxNodes: int option
    }

    type SearchResponse = {
        ActionCount: int
        AverageReward: float
        Iterations: int
        Actions: string list
    }

    let private grammarSearch (input: string) : Result<string, string> =
        try
            let req =
                try JsonSerializer.Deserialize<SearchInput>(input, jsonOptions)
                with _ -> { MaxIterations = None; MaxNodes = None }

            let maxIter = req.MaxIterations |> Option.defaultValue 200
            let maxNodes = req.MaxNodes |> Option.defaultValue 5

            let meta : Tars.DSL.Wot.DslMeta = {
                Id = "mcp-search"
                Title = "MCP Grammar Search"
                Domain = "grammar"
                Objective = "Find optimal WoT derivation"
                Constraints = []
                SuccessCriteria = []
            }

            let templates = [
                { Tars.DSL.Wot.DslConvert.defaultNode "analyze" Tars.DSL.Wot.NodeKind.Reason with
                    Goal = Some "Analyze the problem"
                    Checks = [ Tars.Core.WorkflowOfThought.WotCheck.NonEmpty "${analysis}" ] }
                { Tars.DSL.Wot.DslConvert.defaultNode "plan" Tars.DSL.Wot.NodeKind.Reason with
                    Goal = Some "Create an execution plan"
                    Checks = [ Tars.Core.WorkflowOfThought.WotCheck.NonEmpty "${plan}" ] }
                { Tars.DSL.Wot.DslConvert.defaultNode "execute" Tars.DSL.Wot.NodeKind.Work with
                    Tool = Some "execute"
                    Checks = [ Tars.Core.WorkflowOfThought.WotCheck.NonEmpty "${result}" ] }
                { Tars.DSL.Wot.DslConvert.defaultNode "verify" Tars.DSL.Wot.NodeKind.Reason with
                    Goal = Some "Verify the result"
                    Checks = [ Tars.Core.WorkflowOfThought.WotCheck.NonEmpty "${verification}" ] }
                { Tars.DSL.Wot.DslConvert.defaultNode "refine" Tars.DSL.Wot.NodeKind.Work with
                    Tool = Some "refine"
                    Checks = [ Tars.Core.WorkflowOfThought.WotCheck.NonEmpty "${refined}" ] }
            ]

            let config = { MctsTypes.defaultMctsConfig with MaxIterations = maxIter; MaxRolloutDepth = 20 }
            let result = WotMctsState.searchDerivation config meta templates maxNodes

            let actionNames = result.BestActions |> List.map (fun a -> a.ToString())

            let response = {
                ActionCount = result.BestActions.Length
                AverageReward = result.AverageReward
                Iterations = result.Iterations
                Actions = actionNames
            }
            Result.Ok (JsonSerializer.Serialize(response, jsonOptions))
        with ex ->
            Result.Error $"Failed to run MCTS search: {ex.Message}"

    // =========================================================================
    // Tool registration
    // =========================================================================

    /// Create all probabilistic grammar MCP tools
    let createTools () : Tool list =
        [ { Name = "grammar_weights"
            Description = "View probabilistic grammar rule weights. Shows Bayesian confidence, success rates, and selection counts for all weighted rules. No input required."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return grammarWeights input } }
          { Name = "grammar_update"
            Description = "Bayesian update a grammar rule weight from execution outcome. Input: {\"PatternId\": \"...\", \"Success\": true/false}. Returns old and new weights."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return grammarUpdate input } }
          { Name = "grammar_evolve"
            Description = "Run replicator dynamics on the grammar rule ecosystem. Identifies evolutionarily stable strategies (ESS) and prunes weak rules. No input required."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return grammarEvolve input } }
          { Name = "grammar_search"
            Description = "MCTS search for optimal WoT derivation. Input: {\"MaxIterations\": 200, \"MaxNodes\": 5} (both optional). Returns best action sequence and reward."
            Version = "1.0"
            ParentVersion = None
            CreatedAt = DateTime.UtcNow
            Execute = fun input -> async { return grammarSearch input } } ]
