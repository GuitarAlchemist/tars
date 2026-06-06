namespace Tars.Tests

open System
open System.IO
open Xunit
open Tars.Llm
open Tars.Evolution
open Tars.Evolution.WeightedGrammar
open Tars.Evolution.ReplicatorDynamics
open Tars.Evolution.MctsTypes
open Tars.Evolution.WotMctsState
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot

/// End-to-end integration tests for the probabilistic grammar pipeline:
/// Grammar loading -> Weighting -> MCTS search -> Replicator dynamics -> Constrained LLM request
module ProbabilisticGrammarIntegrationTests =

    // =========================================================================
    // Helpers
    // =========================================================================

    let private withTempGrammarDir (grammars: (string * string) list) (f: string -> 'a) =
        let dir = Path.Combine(Path.GetTempPath(), $"tars_e2e_{Guid.NewGuid():N}")
        Directory.CreateDirectory(dir) |> ignore
        try
            for (name, content) in grammars do
                File.WriteAllText(Path.Combine(dir, name), content)
            f dir
        finally
            Directory.Delete(dir, true)

    let private mkNode id kind : DslNode =
        { DslConvert.defaultNode id kind with
            Name = id
            Tool = if kind = Work then Some "test_tool" else None
            Goal = if kind = Reason then Some "test goal" else None
            Checks = [ WotCheck.NonEmpty "${output}" ] }

    let private mkMeta () : DslMeta =
        { Id = "e2e-test"
          Title = "E2E Test Workflow"
          Domain = "testing"
          Objective = "Test probabilistic grammar pipeline"
          Constraints = []
          SuccessCriteria = [] }

    let private makeRecurrence name count score level : RecurrenceRecord =
        { PatternId = $"pid_{name}"
          PatternName = name
          FirstSeen = DateTime.UtcNow.AddDays(-10.0)
          LastSeen = DateTime.UtcNow
          OccurrenceCount = count
          TaskIds = List.init count (fun i -> $"task_{i}")
          Contexts = [ "ctx1"; "ctx2" ]
          CurrentLevel = level
          PromotionHistory = [ (Implementation, DateTime.UtcNow.AddDays(-10.0)) ]
          AverageScore = score }

    /// Build outcomes map from (patternId, success) pairs
    let private toOutcomesMap (pairs: (string * bool) list) : Map<string, (bool * int64) list> =
        pairs
        |> List.groupBy fst
        |> List.map (fun (id, group) ->
            id, group |> List.map (fun (_, success) -> (success, 100L)))
        |> Map.ofList

    // =========================================================================
    // E2E: Load grammar -> weight rules -> constrained request
    // =========================================================================

    [<Fact>]
    let ``E2E: grammar load to constrained LLM request`` () =
        let grammar = "root ::= step+\nstep ::= 'analyze' | 'plan' | 'execute'"
        withTempGrammarDir [ ("workflow.ebnf", grammar) ] (fun dir ->
            // 1: Load grammar
            let loaded = ConstrainedDecoding.loadEbnfGrammar dir "workflow"
            Assert.True(Result.isOk loaded)

            // 2: Create weighted rules
            let records = [
                (makeRecurrence "analyze" 5 0.8 Implementation, 7)
                (makeRecurrence "plan" 3 0.6 Implementation, 5)
                (makeRecurrence "execute" 4 0.7 Implementation, 6)
            ]
            let weights = fromRecurrenceRecords WeightedGrammar.defaultConfig records

            Assert.Equal(3, weights.Length)
            for w in weights do
                Assert.True(w.Weight > 0.0, $"Weight for {w.PatternName} should be > 0")
            let total = weights |> List.sumBy (fun w -> w.Weight)
            Assert.InRange(total, 0.95, 1.05)

            // 3: Select best rule by weight
            let selected = selectWeighted weights (Random(42))
            Assert.True(selected.IsSome)

            // 4: Create constrained LLM request
            let grammarText = match loaded with Ok g -> g | Error _ -> ""
            let msgs = [ { Role = Role.User; Content = "Plan a workflow" } ]
            let req = ConstrainedDecoding.ebnfConstrained grammarText msgs

            match req.ResponseFormat with
            | Some (ResponseFormat.Constrained (Grammar.Ebnf g)) ->
                Assert.Contains("step", g)
                Assert.Contains("analyze", g)
            | _ -> Assert.Fail("Expected EBNF constrained format")
        )

    // =========================================================================
    // E2E: Weight -> Replicator dynamics -> evolved weights
    // =========================================================================

    [<Fact>]
    let ``E2E: weighted rules through replicator evolution`` () =
        let records = [
            (makeRecurrence "pattern_a" 10 0.9 Helper, 7)
            (makeRecurrence "pattern_b" 5 0.5 Implementation, 4)
            (makeRecurrence "pattern_c" 8 0.8 Helper, 6)
        ]
        let weights = fromRecurrenceRecords WeightedGrammar.defaultConfig records

        let outcomesMap = toOutcomesMap [
            ("pid_pattern_a", true); ("pid_pattern_a", true)
            ("pid_pattern_b", false); ("pid_pattern_b", false)
            ("pid_pattern_c", true); ("pid_pattern_c", false)
        ]

        let result = evolveEcosystem weights outcomesMap

        // pattern_a (high fitness) should outcompete pattern_b
        let speciesA = result.Species |> List.tryFind (fun s -> s.PatternId = "pid_pattern_a")
        let speciesB = result.Species |> List.tryFind (fun s -> s.PatternId = "pid_pattern_b")

        match speciesA, speciesB with
        | Some a, Some b ->
            Assert.True(a.Proportion > b.Proportion,
                $"pattern_a ({a.Proportion:F3}) should outcompete pattern_b ({b.Proportion:F3})")
        | _ -> ()

        Assert.True(result.Species.Length > 0)

    // =========================================================================
    // E2E: MCTS search produces valid WoT derivation
    // =========================================================================

    [<Fact>]
    let ``E2E: MCTS search to WoT derivation`` () =
        let templates = [
            mkNode "analyze" Reason
            mkNode "execute" Work
            mkNode "verify" Reason
        ]

        let config = { defaultMctsConfig with MaxIterations = 100; MaxRolloutDepth = 10 }
        let result = searchDerivation config (mkMeta ()) templates 5

        Assert.True(result.BestActions.Length > 0)
        Assert.True(result.AverageReward > 0.0)
        Assert.True(result.Iterations > 0)

    // =========================================================================
    // E2E: Full pipeline — grammar -> weights -> MCTS -> replicator -> request
    // =========================================================================

    [<Fact>]
    let ``E2E: full probabilistic grammar pipeline`` () =
        let grammar = "root ::= (reason | work)+\nreason ::= 'analyze' | 'evaluate'\nwork ::= 'build' | 'test'"
        withTempGrammarDir [ ("pipeline.ebnf", grammar) ] (fun dir ->
            // 1. Load grammar
            let loaded = ConstrainedDecoding.loadEbnfGrammar dir "pipeline"
            Assert.True(Result.isOk loaded)

            // 2. Create weighted rules
            let records = [
                (makeRecurrence "analyze" 8 0.85 Helper, 7)
                (makeRecurrence "evaluate" 4 0.6 Implementation, 5)
                (makeRecurrence "build" 6 0.75 Helper, 6)
                (makeRecurrence "test" 10 0.9 Builder, 8)
            ]
            let weights = fromRecurrenceRecords WeightedGrammar.defaultConfig records

            // 3. MCTS search for WoT derivation
            let templates = [
                mkNode "analyze" Reason
                mkNode "build" Work
                mkNode "test" Work
            ]
            let mctsResult = searchDerivation
                                { defaultMctsConfig with MaxIterations = 50 }
                                (mkMeta ()) templates 4

            Assert.True(mctsResult.BestActions.Length > 0)

            // 4. Replicator dynamics
            let outcomesMap = toOutcomesMap [
                ("pid_analyze", true); ("pid_build", true); ("pid_test", true)
                ("pid_evaluate", false)
            ]
            let evoResult = evolveEcosystem weights outcomesMap
            Assert.True(evoResult.Species.Length > 0)

            // 5. Use evolved proportions as weights for selection
            let evolvedWeights =
                weights |> List.map (fun w ->
                    let species = evoResult.Species |> List.tryFind (fun s -> s.PatternId = w.PatternId)
                    match species with
                    | Some s -> { w with Weight = s.Proportion }
                    | None -> { w with Weight = 0.01 })

            let selected = selectWeighted evolvedWeights (Random(42))
            Assert.True(selected.IsSome)

            // 6. Build constrained LLM request
            let grammarText = match loaded with Ok g -> g | Error _ -> ""
            let req = ConstrainedDecoding.ebnfConstrained grammarText
                        [ { Role = Role.User; Content = "Execute the pipeline" } ]

            match req.ResponseFormat with
            | Some (ResponseFormat.Constrained (Grammar.Ebnf _)) -> ()
            | _ -> Assert.Fail("Expected EBNF constrained format")

            // 7. Verify IR schemas
            let intentReq = ConstrainedDecoding.jsonConstrained
                                ConstrainedDecoding.intentPlanSchema
                                [ { Role = Role.User; Content = "Plan" } ]
            match intentReq.ResponseFormat with
            | Some (ResponseFormat.Constrained (Grammar.JsonSchema s)) ->
                Assert.Contains("intent", s)
            | _ -> Assert.Fail("Expected JsonSchema")
        )

    // =========================================================================
    // E2E: Bayesian weight updates converge
    // =========================================================================

    [<Fact>]
    let ``E2E: Bayesian updates converge toward true success rate`` () =
        let initial : WeightedRule =
            { PatternId = "test_rule"; PatternName = "converge_test"
              Level = Implementation; RawScore = 5; Weight = 0.5
              Confidence = 0.0; SuccessRate = 0.5; SelectionCount = 0
              Source = Tars; LastUpdated = DateTime.UtcNow }

        let rng = Random(42)
        let mutable rule = initial
        for _ in 1..100 do
            let success = rng.NextDouble() < 0.8
            rule <- updateWeight WeightedGrammar.defaultConfig rule success

        Assert.InRange(rule.SuccessRate, 0.65, 0.95)
        Assert.True(rule.Confidence > 0.5, $"Confidence {rule.Confidence} should be > 0.5")
        Assert.Equal(100, rule.SelectionCount)

    // =========================================================================
    // E2E: Weighted promotion ranking (classifyWeighted)
    // =========================================================================

    [<Fact>]
    let ``E2E: classifyWeighted ranks candidates by weight`` () =
        let records = [
            makeRecurrence "low_weight" 5 0.3 Implementation
            makeRecurrence "high_weight" 5 0.9 Implementation
            makeRecurrence "mid_weight" 5 0.6 Implementation
        ]

        let weights : WeightedRule list = [
            { PatternId = "pid_low_weight"; PatternName = "low_weight"; Level = Implementation
              RawScore = 3; Weight = 0.1; Confidence = 0.5; SuccessRate = 0.3
              SelectionCount = 10; Source = Tars; LastUpdated = DateTime.UtcNow }
            { PatternId = "pid_high_weight"; PatternName = "high_weight"; Level = Implementation
              RawScore = 7; Weight = 0.7; Confidence = 0.9; SuccessRate = 0.9
              SelectionCount = 20; Source = Tars; LastUpdated = DateTime.UtcNow }
            { PatternId = "pid_mid_weight"; PatternName = "mid_weight"; Level = Implementation
              RawScore = 5; Weight = 0.3; Confidence = 0.7; SuccessRate = 0.6
              SelectionCount = 15; Source = Tars; LastUpdated = DateTime.UtcNow }
        ]

        let ranked = PromotionPipeline.classifyWeighted 3 weights records

        Assert.True(ranked.Length >= 2, $"Should have at least 2 candidates, got {ranked.Length}")
        let first = ranked |> List.head
        Assert.Equal("pid_high_weight", first.Record.PatternId)

    // =========================================================================
    // E2E: All IR schemas produce valid constrained requests
    // =========================================================================

    [<Fact>]
    let ``E2E: all IR schemas produce valid constrained requests`` () =
        let schemas = [
            ("intentPlan", ConstrainedDecoding.intentPlanSchema)
            ("beliefUpdate", ConstrainedDecoding.beliefUpdateSchema)
            ("repairProposal", ConstrainedDecoding.repairProposalSchema)
        ]
        for (name, schema) in schemas do
            let req = ConstrainedDecoding.jsonConstrained schema
                        [ { Role = Role.User; Content = $"Generate {name}" } ]
            match req.ResponseFormat with
            | Some (ResponseFormat.Constrained (Grammar.JsonSchema s)) ->
                let doc = System.Text.Json.JsonDocument.Parse(s)
                Assert.Equal("object", doc.RootElement.GetProperty("type").GetString())
            | _ -> Assert.Fail($"Expected JsonSchema for {name}")
