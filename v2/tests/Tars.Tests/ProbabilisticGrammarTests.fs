namespace Tars.Tests

open System
open Xunit
open Tars.Evolution
open Tars.Evolution.WeightedGrammar
open Tars.Evolution.MctsTypes
open Tars.Evolution.MctsSolver
open Tars.Evolution.WotMctsState
open Tars.Evolution.MctsBridge
open Tars.Core.WorkflowOfThought
open Tars.DSL.Wot

/// Tests for probabilistic grammar weights and MCTS derivation search.
module ProbabilisticGrammarTests =

    // =========================================================================
    // Helper constructors
    // =========================================================================

    let private mkRecurrence id name level score count : RecurrenceRecord * int =
        { PatternId = id
          PatternName = name
          FirstSeen = DateTime.UtcNow.AddDays(-10.0)
          LastSeen = DateTime.UtcNow
          OccurrenceCount = count
          TaskIds = [ "t1"; "t2"; "t3" ]
          Contexts = [ "ctx1" ]
          CurrentLevel = level
          PromotionHistory = []
          AverageScore = float score / 8.0 }, score

    let private mkWeightedRule id name weight : WeightedRule =
        { PatternId = id
          PatternName = name
          Level = Helper
          RawScore = 6
          Weight = weight
          Confidence = 0.75
          SuccessRate = 0.8
          SelectionCount = 10
          Source = Tars
          LastUpdated = DateTime.UtcNow }

    let private mkTemplate id kind : DslNode =
        { DslConvert.defaultNode id kind with
            Name = id
            Tool = if kind = Work then Some "test_tool" else None
            Goal = if kind = Reason then Some "test goal" else None
            Checks = [ WotCheck.NonEmpty "${output}" ] }

    let private mkMeta () : DslMeta =
        { Id = "test-wot"
          Title = "Test Workflow"
          Domain = "testing"
          Objective = "Find optimal derivation"
          Constraints = []
          SuccessCriteria = [] }

    // =========================================================================
    // WeightedGrammar: scoreToLogit
    // =========================================================================

    [<Fact>]
    let ``scoreToLogit maps 0 to 0.0`` () =
        Assert.Equal(0.0, scoreToLogit 0)

    [<Fact>]
    let ``scoreToLogit maps 8 to 1.0`` () =
        Assert.Equal(1.0, scoreToLogit 8)

    [<Fact>]
    let ``scoreToLogit maps 4 to 0.5`` () =
        Assert.Equal(0.5, scoreToLogit 4)

    // =========================================================================
    // WeightedGrammar: softmax
    // =========================================================================

    [<Fact>]
    let ``softmax with uniform scores produces equal weights`` () =
        let result = softmax 1.0 [ 0.5; 0.5; 0.5 ]
        for w in result do
            Assert.True(abs (w - 1.0/3.0) < 0.01, sprintf "Expected ~0.333, got %.4f" w)

    [<Fact>]
    let ``softmax with high temperature approaches uniform`` () =
        let result = softmax 100.0 [ 0.1; 0.5; 0.9 ]
        let spread = (List.max result) - (List.min result)
        Assert.True(spread < 0.05, sprintf "Expected near-uniform, spread was %.4f" spread)

    [<Fact>]
    let ``softmax with low temperature concentrates on max`` () =
        let result = softmax 0.01 [ 0.1; 0.5; 0.9 ]
        let maxWeight = List.max result
        Assert.True(maxWeight > 0.95, sprintf "Expected max > 0.95, got %.4f" maxWeight)

    [<Fact>]
    let ``softmax sums to approximately 1.0`` () =
        let result = softmax 1.0 [ 0.2; 0.4; 0.6; 0.8 ]
        let total = result |> List.sum
        Assert.True(abs (total - 1.0) < 0.001, sprintf "Sum was %.6f" total)

    [<Fact>]
    let ``softmax handles empty list`` () =
        let result = softmax 1.0 []
        Assert.Empty(result)

    // =========================================================================
    // WeightedGrammar: Bayesian update
    // =========================================================================

    [<Fact>]
    let ``bayesianUpdate increases rate on success`` () =
        let newRate, _ = bayesianUpdate 0.5 10 true 0.95
        Assert.True(newRate > 0.5, sprintf "Expected > 0.5, got %.4f" newRate)

    [<Fact>]
    let ``bayesianUpdate decreases rate on failure`` () =
        let newRate, _ = bayesianUpdate 0.5 10 false 0.95
        Assert.True(newRate < 0.5, sprintf "Expected < 0.5, got %.4f" newRate)

    [<Fact>]
    let ``bayesianUpdate confidence increases with more observations`` () =
        let _, conf1 = bayesianUpdate 0.5 5 true 0.95
        let _, conf2 = bayesianUpdate 0.5 50 true 0.95
        Assert.True(conf2 > conf1, sprintf "Expected higher conf with more obs: %.4f vs %.4f" conf2 conf1)

    // =========================================================================
    // WeightedGrammar: fromRecurrenceRecords
    // =========================================================================

    [<Fact>]
    let ``fromRecurrenceRecords produces valid weights`` () =
        let records = [
            mkRecurrence "p1" "CoT" Helper 7 5
            mkRecurrence "p2" "ReAct" Helper 3 4
        ]
        let rules = fromRecurrenceRecords defaultConfig records
        Assert.Equal(2, rules.Length)
        let total = rules |> List.sumBy (fun r -> r.Weight)
        Assert.True(abs (total - 1.0) < 0.1, sprintf "Weights should sum near 1.0, got %.4f" total)
        // Higher score should get higher weight
        let p1 = rules |> List.find (fun r -> r.PatternId = "p1")
        let p2 = rules |> List.find (fun r -> r.PatternId = "p2")
        Assert.True(p1.Weight > p2.Weight, "Higher scored pattern should have higher weight")

    [<Fact>]
    let ``fromRecurrenceRecords handles empty list`` () =
        let rules = fromRecurrenceRecords defaultConfig []
        Assert.Empty(rules)

    // =========================================================================
    // WeightedGrammar: updateWeight
    // =========================================================================

    [<Fact>]
    let ``updateWeight increases weight on success`` () =
        let rule = mkWeightedRule "p1" "test" 0.5
        let updated = updateWeight defaultConfig rule true
        Assert.True(updated.SuccessRate > rule.SuccessRate,
            sprintf "Success rate should increase: %.4f -> %.4f" rule.SuccessRate updated.SuccessRate)
        Assert.Equal(rule.SelectionCount + 1, updated.SelectionCount)

    [<Fact>]
    let ``updateWeight decreases weight on failure`` () =
        let rule = mkWeightedRule "p1" "test" 0.5
        let updated = updateWeight defaultConfig rule false
        Assert.True(updated.SuccessRate < rule.SuccessRate,
            sprintf "Success rate should decrease: %.4f -> %.4f" rule.SuccessRate updated.SuccessRate)

    // =========================================================================
    // WeightedGrammar: selectWeighted
    // =========================================================================

    [<Fact>]
    let ``selectWeighted returns None for empty list`` () =
        let result = selectWeighted [] (Random(42))
        Assert.True(result.IsNone)

    [<Fact>]
    let ``selectWeighted respects probability distribution`` () =
        let rules = [
            mkWeightedRule "high" "high" 0.9
            mkWeightedRule "low" "low" 0.1
        ]
        let rng = Random(42)
        let mutable highCount = 0
        for _ in 1..1000 do
            match selectWeighted rules rng with
            | Some r when r.PatternId = "high" -> highCount <- highCount + 1
            | _ -> ()
        // High-weight rule should be selected ~90% of the time
        Assert.True(highCount > 800, sprintf "Expected >800 high selections, got %d" highCount)

    // =========================================================================
    // WeightedGrammar: normalizeByLevel
    // =========================================================================

    [<Fact>]
    let ``normalizeByLevel sums to 1.0 within each level`` () =
        let rules = [
            { mkWeightedRule "p1" "a" 0.3 with Level = Helper }
            { mkWeightedRule "p2" "b" 0.7 with Level = Helper }
            { mkWeightedRule "p3" "c" 0.4 with Level = Builder }
            { mkWeightedRule "p4" "d" 0.6 with Level = Builder }
        ]
        let normalized = normalizeByLevel rules
        let helperSum = normalized |> List.filter (fun r -> r.Level = Helper) |> List.sumBy (fun r -> r.Weight)
        let builderSum = normalized |> List.filter (fun r -> r.Level = Builder) |> List.sumBy (fun r -> r.Weight)
        Assert.True(abs (helperSum - 1.0) < 0.001, sprintf "Helper sum: %.6f" helperSum)
        Assert.True(abs (builderSum - 1.0) < 0.001, sprintf "Builder sum: %.6f" builderSum)

    // =========================================================================
    // WeightedGrammar: evaluateWithWeight
    // =========================================================================

    [<Fact>]
    let ``evaluateWithWeight returns governance decision with weight`` () =
        let record : RecurrenceRecord =
            { PatternId = "p1"; PatternName = "TestPattern"
              FirstSeen = DateTime.UtcNow; LastSeen = DateTime.UtcNow
              OccurrenceCount = 5; TaskIds = ["t1";"t2";"t3";"t4";"t5"]
              Contexts = ["ctx"]; CurrentLevel = Implementation
              PromotionHistory = []; AverageScore = 0.8 }
        let candidate : PromotionCandidate =
            { Record = record; ProposedLevel = Helper
              Criteria = { PromotionCriteria.empty with
                            MinOccurrences = true; RemovesComplexity = true
                            MoreReadable = true; StableSemantics = true
                            AutoValidatable = true; NoOverlap = true
                            ComposesCleanly = true; ImprovesPlanning = false }
              Evidence = ["e1"]; PatternTemplate = "template"
              RollbackExpansion = None }
        let weights = [ mkWeightedRule "p1" "TestPattern" 0.85 ]
        let decision, weight = evaluateWithWeight [] weights candidate
        match decision with
        | Approve _ -> ()
        | other -> Assert.Fail($"Expected Approve, got {other}")
        Assert.Equal(0.85, weight)

    // =========================================================================
    // MCTS Core: UCB1
    // =========================================================================

    /// Simple 2-action game for testing: action 0 always gives 0.3, action 1 always gives 0.8
    type SimpleAction = Left | Right

    type SimpleState(depth: int, maxDepth: int) =
        interface IMctsState<SimpleAction> with
            member _.LegalActions() =
                if depth >= maxDepth then [] else [ Left; Right ]
            member _.Apply(action) =
                SimpleState(depth + 1, maxDepth) :> IMctsState<SimpleAction>
            member _.IsTerminal = depth >= maxDepth
            member _.Reward() =
                if depth >= maxDepth then 0.8 else 0.5

    [<Fact>]
    let ``ucb1 favors unexplored nodes`` () =
        let root = createNode (SimpleState(0, 2) :> IMctsState<SimpleAction>) None None
        root.Visits <- 10
        let child1 = createNode (SimpleState(1, 2) :> IMctsState<SimpleAction>) (Some Left) (Some root)
        child1.Visits <- 5; child1.TotalReward <- 2.0
        let child2 = createNode (SimpleState(1, 2) :> IMctsState<SimpleAction>) (Some Right) (Some root)
        child2.Visits <- 0; child2.TotalReward <- 0.0
        let score1 = ucb1 (sqrt 2.0) root child1
        let score2 = ucb1 (sqrt 2.0) root child2
        Assert.True(score2 > score1, "Unexplored node should have higher UCB1")

    [<Fact>]
    let ``ucb1 favors high-reward nodes with many visits`` () =
        let root = createNode (SimpleState(0, 2) :> IMctsState<SimpleAction>) None None
        root.Visits <- 100
        let child1 = createNode (SimpleState(1, 2) :> IMctsState<SimpleAction>) (Some Left) (Some root)
        child1.Visits <- 50; child1.TotalReward <- 40.0 // avg 0.8
        let child2 = createNode (SimpleState(1, 2) :> IMctsState<SimpleAction>) (Some Right) (Some root)
        child2.Visits <- 50; child2.TotalReward <- 15.0 // avg 0.3
        let score1 = ucb1 (sqrt 2.0) root child1
        let score2 = ucb1 (sqrt 2.0) root child2
        Assert.True(score1 > score2, "High reward node should have higher UCB1")

    // =========================================================================
    // MCTS Core: backpropagate
    // =========================================================================

    [<Fact>]
    let ``backpropagate increments visits up to root`` () =
        let root = createNode (SimpleState(0, 2) :> IMctsState<SimpleAction>) None None
        let child = createNode (SimpleState(1, 2) :> IMctsState<SimpleAction>) (Some Left) (Some root)
        root.Children <- [ child ]
        backpropagate child 0.7
        Assert.Equal(1, child.Visits)
        Assert.Equal(1, root.Visits)
        Assert.True(abs (child.TotalReward - 0.7) < 0.001)
        Assert.True(abs (root.TotalReward - 0.7) < 0.001)

    // =========================================================================
    // MCTS Core: search
    // =========================================================================

    [<Fact>]
    let ``search on simple game completes`` () =
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 50; MaxRolloutDepth = 5 }
        let state = SimpleState(0, 2) :> IMctsState<SimpleAction>
        let result = search config state
        Assert.True(result.Iterations > 0, "Should complete at least 1 iteration")
        Assert.True(result.RootVisits > 0)

    [<Fact>]
    let ``search respects MaxIterations`` () =
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 25; MaxRolloutDepth = 3 }
        let state = SimpleState(0, 2) :> IMctsState<SimpleAction>
        let result = search config state
        Assert.True(result.Iterations <= 25, sprintf "Expected <= 25 iterations, got %d" result.Iterations)

    // =========================================================================
    // WoT MCTS State: legalActions
    // =========================================================================

    [<Fact>]
    let ``legalActions returns AddNode when templates available`` () =
        let state : WotDerivationState = {
            Nodes = []; Edges = []; IsComplete = false
            AvailableTemplates = [ mkTemplate "n1" Work; mkTemplate "n2" Reason ]
            Meta = mkMeta (); MaxNodes = 5
        }
        let actions = legalActions state
        let addNodes = actions |> List.filter (function AddNode _ -> true | _ -> false)
        Assert.Equal(2, addNodes.Length)

    [<Fact>]
    let ``legalActions returns Complete when nodes >= 2 and edges exist`` () =
        let n1 = mkTemplate "n1" Work
        let n2 = mkTemplate "n2" Reason
        let state : WotDerivationState = {
            Nodes = [ n1; n2 ]
            Edges = [ { From = "n1"; To = "n2"; Relation = EdgeDependsOn } ]
            IsComplete = false; AvailableTemplates = []
            Meta = mkMeta (); MaxNodes = 5
        }
        let actions = legalActions state
        Assert.Contains(Complete, actions)

    [<Fact>]
    let ``legalActions returns empty when complete`` () =
        let state : WotDerivationState = {
            Nodes = [ mkTemplate "n1" Work ]
            Edges = []; IsComplete = true; AvailableTemplates = []
            Meta = mkMeta (); MaxNodes = 5
        }
        Assert.Empty(legalActions state)

    // =========================================================================
    // WoT MCTS State: applyAction
    // =========================================================================

    [<Fact>]
    let ``applyAction AddNode increases node count`` () =
        let template = mkTemplate "n1" Work
        let state : WotDerivationState = {
            Nodes = []; Edges = []; IsComplete = false
            AvailableTemplates = [ template ]
            Meta = mkMeta (); MaxNodes = 5
        }
        let newState = applyAction state (AddNode template)
        Assert.Equal(1, newState.Nodes.Length)
        Assert.Empty(newState.AvailableTemplates)

    [<Fact>]
    let ``applyAction Complete makes state terminal`` () =
        let state : WotDerivationState = {
            Nodes = [ mkTemplate "n1" Work; mkTemplate "n2" Reason ]
            Edges = [ { From = "n1"; To = "n2"; Relation = EdgeDependsOn } ]
            IsComplete = false; AvailableTemplates = []
            Meta = mkMeta (); MaxNodes = 5
        }
        let newState = applyAction state Complete
        Assert.True(newState.IsComplete)

    // =========================================================================
    // WoT MCTS State: computeReward
    // =========================================================================

    [<Fact>]
    let ``computeReward returns 0.0 for empty graph`` () =
        let state : WotDerivationState = {
            Nodes = []; Edges = []; IsComplete = true; AvailableTemplates = []
            Meta = mkMeta (); MaxNodes = 5
        }
        Assert.Equal(0.0, computeReward state)

    [<Fact>]
    let ``computeReward returns positive for valid chain`` () =
        let n1 = mkTemplate "n1" Work
        let n2 = mkTemplate "n2" Reason
        let state : WotDerivationState = {
            Nodes = [ n1; n2 ]
            Edges = [ { From = "n1"; To = "n2"; Relation = EdgeDependsOn } ]
            IsComplete = true; AvailableTemplates = []
            Meta = mkMeta (); MaxNodes = 4
        }
        let reward = computeReward state
        Assert.True(reward > 0.3, sprintf "Expected reward > 0.3, got %.4f" reward)

    [<Fact>]
    let ``computeReward gives bonus for check coverage`` () =
        let noChecks = { mkTemplate "n1" Work with Checks = [] }
        let withChecks = mkTemplate "n2" Work

        let stateNoChecks : WotDerivationState = {
            Nodes = [ noChecks ]; Edges = []; IsComplete = true
            AvailableTemplates = []; Meta = mkMeta (); MaxNodes = 4
        }
        let stateWithChecks : WotDerivationState = {
            Nodes = [ withChecks ]; Edges = []; IsComplete = true
            AvailableTemplates = []; Meta = mkMeta (); MaxNodes = 4
        }
        let r1 = computeReward stateNoChecks
        let r2 = computeReward stateWithChecks
        Assert.True(r2 > r1, sprintf "Checks should increase reward: %.4f vs %.4f" r2 r1)

    // =========================================================================
    // WoT MCTS: full derivation search
    // =========================================================================

    [<Fact>]
    let ``searchDerivation finds non-empty action sequence`` () =
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 100; MaxRolloutDepth = 10 }
        let templates = [
            mkTemplate "analyze" Reason
            mkTemplate "execute" Work
            mkTemplate "verify" Reason
        ]
        let result = searchDerivation config (mkMeta ()) templates 4
        Assert.True(result.BestActions.Length > 0,
            sprintf "Expected non-empty actions, got %d" result.BestActions.Length)
        Assert.True(result.AverageReward > 0.0,
            sprintf "Expected positive reward, got %.4f" result.AverageReward)

    // =========================================================================
    // MctsBridge: parseMctsOutput
    // =========================================================================

    [<Fact>]
    let ``parseMctsOutput parses complete output`` () =
        let output = """
  MCTS:
    Best action:  2
    Iterations:   1000
    Avg reward:   0.75
    Tree size:    3421
"""
        let result = parseMctsOutput output
        Assert.Equal(2, result.BestActionIndex)
        Assert.Equal(1000, result.Iterations)
        Assert.True(abs (result.AverageReward - 0.75) < 0.001)
        Assert.Equal(3421, result.TreeSize)

    [<Fact>]
    let ``parseMctsOutput handles empty output`` () =
        let result = parseMctsOutput "No results"
        Assert.Equal(0, result.BestActionIndex)
        Assert.Equal(0, result.Iterations)

    // =========================================================================
    // MctsBridge: searchWotDerivation
    // =========================================================================

    [<Fact>]
    let ``searchWotDerivation falls back to F# solver`` () =
        let templates = [
            mkTemplate "step1" Work
            mkTemplate "step2" Reason
        ]
        let config = { MctsTypes.defaultMctsConfig with MaxIterations = 50; MaxRolloutDepth = 10 }
        let actions, usedMachin = searchWotDerivation None config (mkMeta ()) templates 3
        Assert.False(usedMachin, "Should not use ix when config is None")
        Assert.True(actions.Length > 0, "Should find some actions")
