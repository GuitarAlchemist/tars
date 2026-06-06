namespace Tars.Tests

open Xunit
open Tars.Core.WorkflowOfThought
open Tars.Evolution
open Tars.Llm

module GrammarMlBridgeTests =

    // ── Helpers ──────────────────────────────────────────────────────────────

    let makeEvent stepId kind toolName =
        { StepId = stepId; Kind = kind; ToolName = toolName
          ResolvedArgs = None; Outputs = ["output"]
          Status = StepStatus.Ok; Error = None; Usage = None; Metadata = None }

    let sampleProduction =
        let trace = [
            makeEvent "plan" "reason" None
            makeEvent "search" "work" (Some "search_web")
            makeEvent "analyse" "reason" None
            makeEvent "verify" "work" (Some "fsharp_compile")
            makeEvent "summarise" "reason" None
        ]
        (GrammarDistillation.distillTrace trace "test goal").Value

    let smallProduction =
        let trace = [
            makeEvent "think" "reason" None
            makeEvent "act" "work" (Some "run_command")
        ]
        (GrammarDistillation.distillTrace trace "small goal").Value

    // ── Feature extraction ───────────────────────────────────────────────────

    [<Fact>]
    let ``extractFeatures counts slots and edges`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction

        Assert.Equal(5, f.SlotCount)
        Assert.Equal(4, f.EdgeCount)

    [<Fact>]
    let ``extractFeatures computes reason and work ratios`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction

        Assert.Equal(0.6, f.ReasonRatio, 1) // 3/5
        Assert.Equal(0.4, f.WorkRatio, 1)   // 2/5

    [<Fact>]
    let ``extractFeatures counts tool diversity`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction

        Assert.Equal(2, f.ToolDiversity) // search_web, fsharp_compile

    [<Fact>]
    let ``extractFeatures detects type composability`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction

        Assert.Equal(1.0, f.TypeComposable) // should compose

    [<Fact>]
    let ``extractFeatures captures success rate`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction

        Assert.True(f.SuccessRate > 0.0)

    // ── CSV/JSON formatting ──────────────────────────────────────────────────

    [<Fact>]
    let ``featuresToCsvRow produces correct column count`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction
        let row = GrammarMlBridge.featuresToCsvRow f
        let cols = row.Split(',')

        Assert.Equal(11, cols.Length) // 10 features + 1 target

    [<Fact>]
    let ``featuresToJsonRow produces correct element count`` () =
        let f = GrammarMlBridge.extractFeatures sampleProduction
        let row = GrammarMlBridge.featuresToJsonRow f

        Assert.Equal(11, row.Length)

    [<Fact>]
    let ``csvHeaders matches csv row column count`` () =
        let headerCols = GrammarMlBridge.csvHeaders.Split(',')
        let f = GrammarMlBridge.extractFeatures sampleProduction
        let rowCols = (GrammarMlBridge.featuresToCsvRow f).Split(',')

        Assert.Equal(headerCols.Length, rowCols.Length)

    // ── Training args ────────────────────────────────────────────────────────

    [<Fact>]
    let ``buildTrainArgs produces valid JSON`` () =
        let productions = [sampleProduction; smallProduction; sampleProduction]
        let args = GrammarMlBridge.buildTrainArgs productions 0.8 "grammar_model"

        Assert.Contains("classify", args)
        Assert.Contains("random-forest", args)
        Assert.Contains("grammar_model", args)
        Assert.Contains("label", args)
        // Should parse as JSON
        let parsed = JsonParsing.tryParseElement args
        Assert.True(Result.isOk parsed)

    // ── Prediction args ──────────────────────────────────────────────────────

    [<Fact>]
    let ``buildPredictArgs produces valid JSON`` () =
        let args = GrammarMlBridge.buildPredictArgs sampleProduction "grammar_model"

        Assert.Contains("grammar_model", args)
        Assert.Contains("data", args)
        let parsed = JsonParsing.tryParseElement args
        Assert.True(Result.isOk parsed)

    // ── Prediction parsing ───────────────────────────────────────────────────

    [<Fact>]
    let ``parsePrediction extracts success and confidence`` () =
        let ixJson = """{"predictions": [1.0], "probabilities": [[0.15, 0.85]]}"""
        let result = GrammarMlBridge.parsePrediction ixJson

        match result with
        | Ok pred ->
            Assert.Equal(1.0, pred.PredictedSuccess)
            Assert.Equal(0.85, pred.Confidence, 2)
        | Error e -> failwith $"Expected Ok: {e}"

    [<Fact>]
    let ``parsePrediction handles missing probabilities`` () =
        let ixJson = """{"predictions": [0.0]}"""
        let result = GrammarMlBridge.parsePrediction ixJson

        match result with
        | Ok pred ->
            Assert.Equal(0.0, pred.PredictedSuccess)
            Assert.Equal(0.5, pred.Confidence) // default
        | Error e -> failwith $"Expected Ok: {e}"

    [<Fact>]
    let ``parsePrediction fails on invalid JSON`` () =
        let result = GrammarMlBridge.parsePrediction "not json"
        Assert.True(Result.isError result)

    // ── Predictive prior ─────────────────────────────────────────────────────

    [<Fact>]
    let ``applyPredictivePrior blends ix prediction with existing rate`` () =
        let prediction = { PredictedSuccess = 0.9; Confidence = 0.8 }
        let rule = GrammarDistillation.toWeightedRule sampleProduction

        let updated = GrammarMlBridge.applyPredictivePrior prediction rule

        // Blended: 0.9 * 0.8 + original * 0.2
        Assert.True(updated.SuccessRate > rule.SuccessRate * 0.5)
        Assert.Equal(WeightedGrammar.Evolved, updated.Source)

    [<Fact>]
    let ``applyPredictivePrior with low confidence barely changes rate`` () =
        let prediction = { PredictedSuccess = 0.9; Confidence = 0.1 }
        let rule = GrammarDistillation.toWeightedRule sampleProduction
        let originalRate = rule.SuccessRate

        let updated = GrammarMlBridge.applyPredictivePrior prediction rule

        // With 10% confidence, blended rate should be close to original
        Assert.InRange(updated.SuccessRate, originalRate * 0.8, originalRate * 1.2 + 0.1)

    // ── Genome encoding ──────────────────────────────────────────────────────

    [<Fact>]
    let ``encodeGenomes produces 10-element vectors`` () =
        let genomes = GrammarMlBridge.encodeGenomes [sampleProduction; smallProduction]

        Assert.Equal(2, genomes.Length)
        Assert.True(genomes |> List.forall (fun g -> g.Length = 10))

    // ── Breed args ───────────────────────────────────────────────────────────

    [<Fact>]
    let ``buildBreedArgs produces valid JSON with population`` () =
        let args = GrammarMlBridge.buildBreedArgs [sampleProduction; smallProduction] 50 20

        Assert.Contains("genetic", args)
        Assert.Contains("initial_population", args)
        Assert.Contains("50", args) // generations
        let parsed = JsonParsing.tryParseElement args
        Assert.True(Result.isOk parsed)

    // ── Breed result parsing ─────────────────────────────────────────────────

    [<Fact>]
    let ``parseBreedResult extracts candidate from ix_evolution output`` () =
        let ixJson = """{"best_value": 0.92, "best_params": [5.0, 4.0, 0.6, 0.4, 2.0, 3.0, 1.0, 3.0, 3.0, 2.5], "iterations": 100}"""
        let result = GrammarMlBridge.parseBreedResult ixJson ["parent1"; "parent2"]

        match result with
        | Ok br ->
            Assert.Equal(100, br.GenerationsRun)
            Assert.Equal(0.92, br.BestFitness, 2)
            Assert.Equal(1, br.Candidates.Length)
            let c = br.Candidates.[0]
            Assert.Equal(5, c.Features.SlotCount)
            Assert.Equal(2, c.Features.ToolDiversity)
            Assert.Equal(2, c.ParentIds.Length)
        | Error e -> failwith $"Expected Ok: {e}"

    [<Fact>]
    let ``parseBreedResult handles missing best_params gracefully`` () =
        let ixJson = """{"best_value": 0.5, "iterations": 10}"""
        let result = GrammarMlBridge.parseBreedResult ixJson []

        match result with
        | Ok br ->
            Assert.Equal(0, br.Candidates.Length) // no valid candidate
            Assert.Equal(10, br.GenerationsRun)
        | Error e -> failwith $"Expected Ok: {e}"

    // ── Full evolve cycle (with stub ix) ─────────────────────────────────────

    [<Fact>]
    let ``evolveAsync requires minimum 3 productions`` () = async {
        let ix: IxCaller = fun _ _ -> async { return Ok "{}" }
        let config = { SuccessThreshold = 0.8; PersistKey = "test"; Breed = false; Generations = 10 }

        let! result = GrammarMlBridge.evolveAsync ix [sampleProduction; smallProduction] [] config

        match result with
        | Error msg -> Assert.Contains("3", msg)
        | Ok _ -> failwith "Expected error for < 3 productions"
    }

    [<Fact>]
    let ``evolveAsync trains and predicts with stub ix`` () = async {
        let mutable calls = []
        let ix: IxCaller = fun tool args ->
            async {
                calls <- calls @ [tool]
                match tool with
                | "ix_ml_pipeline" -> return Ok """{"accuracy": 0.85, "model": "trained"}"""
                | "ix_ml_predict" -> return Ok """{"predictions": [1.0], "probabilities": [[0.1, 0.9]]}"""
                | _ -> return Error "unknown"
            }

        let thirdProd = { sampleProduction with Id = "third"; Name = "third goal" }
        let productions = [sampleProduction; smallProduction; thirdProd]
        let weights = productions |> List.map GrammarDistillation.toWeightedRule
        let config = { SuccessThreshold = 0.8; PersistKey = "test"; Breed = false; Generations = 10 }

        let! result = GrammarMlBridge.evolveAsync ix productions weights config

        match result with
        | Ok (updatedWeights, breedResult) ->
            Assert.Contains("ix_ml_pipeline", calls)
            Assert.Contains("ix_ml_predict", calls)
            Assert.True(updatedWeights.Length > 0)
            Assert.True(breedResult.IsNone) // breeding disabled
        | Error e -> failwith $"Expected Ok: {e}"
    }

    [<Fact>]
    let ``evolveAsync breeds when enabled`` () = async {
        let ix: IxCaller = fun tool _ ->
            async {
                match tool with
                | "ix_ml_pipeline" -> return Ok """{"accuracy": 0.85}"""
                | "ix_ml_predict" -> return Ok """{"predictions": [1.0], "probabilities": [[0.1, 0.9]]}"""
                | "ix_evolution" -> return Ok """{"best_value": 0.88, "best_params": [4.0, 3.0, 0.7, 0.3, 1.0, 2.0, 1.0, 2.0, 2.0, 1.5], "iterations": 50}"""
                | _ -> return Error "unknown"
            }

        let thirdProd = { sampleProduction with Id = "third"; Name = "third" }
        let productions = [sampleProduction; smallProduction; thirdProd]
        let weights = productions |> List.map GrammarDistillation.toWeightedRule
        let config = { SuccessThreshold = 0.8; PersistKey = "test"; Breed = true; Generations = 50 }

        let! result = GrammarMlBridge.evolveAsync ix productions weights config

        match result with
        | Ok (_, breedResult) ->
            Assert.True(breedResult.IsSome)
            Assert.Equal(50, breedResult.Value.GenerationsRun)
            Assert.True(breedResult.Value.Candidates.Length > 0)
        | Error e -> failwith $"Expected Ok: {e}"
    }
