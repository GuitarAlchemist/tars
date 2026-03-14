namespace Tars.Tests

open System
open Xunit
open Tars.Evolution
open Tars.Evolution.WeightedGrammar
open Tars.Evolution.ReplicatorDynamics

/// Tests for replicator dynamics — grammar rules competing as species.
module ReplicatorDynamicsTests =

    // =========================================================================
    // Helpers
    // =========================================================================

    let private mkRule id name weight successRate selections : WeightedRule =
        { PatternId = id
          PatternName = name
          Level = Helper
          RawScore = 6
          Weight = weight
          Confidence = 0.75
          SuccessRate = successRate
          SelectionCount = selections
          Source = Tars
          LastUpdated = DateTime.UtcNow }

    let private mkSpecies id name proportion fitness : GrammarSpecies =
        { PatternId = id
          PatternName = name
          Level = Helper
          Proportion = proportion
          Fitness = fitness
          IsStable = false }

    // =========================================================================
    // computeSpeciesFitness
    // =========================================================================

    [<Fact>]
    let ``computeSpeciesFitness returns 0.5 for empty outcomes`` () =
        let fitness = computeSpeciesFitness []
        Assert.Equal(0.5, fitness)

    [<Fact>]
    let ``computeSpeciesFitness returns high for all successes`` () =
        let outcomes = [ for _ in 1..10 -> (true, 500L) ]
        let fitness = computeSpeciesFitness outcomes
        Assert.True(fitness > 0.9, sprintf "Expected > 0.9, got %.4f" fitness)

    [<Fact>]
    let ``computeSpeciesFitness returns low for all failures`` () =
        let outcomes = [ for _ in 1..10 -> (false, 500L) ]
        let fitness = computeSpeciesFitness outcomes
        Assert.True(fitness < 0.1, sprintf "Expected < 0.1, got %.4f" fitness)

    [<Fact>]
    let ``computeSpeciesFitness penalizes slow executions`` () =
        let fast = [ for _ in 1..10 -> (true, 100L) ]
        let slow = [ for _ in 1..10 -> (true, 20000L) ]
        let fitnessFast = computeSpeciesFitness fast
        let fitnessSlow = computeSpeciesFitness slow
        Assert.True(fitnessFast > fitnessSlow,
            sprintf "Fast (%.4f) should beat slow (%.4f)" fitnessFast fitnessSlow)

    // =========================================================================
    // buildSpecies
    // =========================================================================

    [<Fact>]
    let ``buildSpecies produces proportions summing to 1.0`` () =
        let rules = [
            mkRule "p1" "CoT" 0.6 0.8 10
            mkRule "p2" "ReAct" 0.3 0.5 10
            mkRule "p3" "ToT" 0.1 0.9 10
        ]
        let outcomes = Map.empty
        let species = buildSpecies rules outcomes
        let total = species |> List.sumBy (fun s -> s.Proportion)
        Assert.True(abs (total - 1.0) < 0.001, sprintf "Proportions sum to %.6f" total)

    [<Fact>]
    let ``buildSpecies handles empty rules`` () =
        Assert.Empty(buildSpecies [] Map.empty)

    [<Fact>]
    let ``buildSpecies uses outcomes for fitness`` () =
        let rules = [ mkRule "p1" "CoT" 0.5 0.5 10 ]
        let outcomes = Map.ofList [ ("p1", [ (true, 100L); (true, 100L); (false, 100L) ]) ]
        let species = buildSpecies rules outcomes
        let s = species.[0]
        // 2/3 success rate = 0.667, minus tiny duration penalty
        Assert.True(s.Fitness > 0.6, sprintf "Expected fitness > 0.6, got %.4f" s.Fitness)

    // =========================================================================
    // step (single replicator step)
    // =========================================================================

    [<Fact>]
    let ``step increases proportion of fitter species`` () =
        let species = [
            mkSpecies "strong" "Strong" 0.5 0.9
            mkSpecies "weak" "Weak" 0.5 0.1
        ]
        let after = step 0.1 0.01 species
        let strong = after |> List.find (fun s -> s.PatternId = "strong")
        let weak = after |> List.find (fun s -> s.PatternId = "weak")
        Assert.True(strong.Proportion > 0.5,
            sprintf "Strong should grow: %.4f" strong.Proportion)
        Assert.True(weak.Proportion < 0.5,
            sprintf "Weak should shrink: %.4f" weak.Proportion)

    [<Fact>]
    let ``step preserves total proportion at 1.0`` () =
        let species = [
            mkSpecies "a" "A" 0.4 0.8
            mkSpecies "b" "B" 0.35 0.3
            mkSpecies "c" "C" 0.25 0.6
        ]
        let after = step 0.1 0.01 species
        let total = after |> List.sumBy (fun s -> s.Proportion)
        Assert.True(abs (total - 1.0) < 0.001, sprintf "Total: %.6f" total)

    [<Fact>]
    let ``step respects smoothing floor`` () =
        let species = [
            mkSpecies "dominant" "Dom" 0.99 1.0
            mkSpecies "tiny" "Tiny" 0.01 0.0
        ]
        let after = step 0.5 0.005 species
        let tiny = after |> List.find (fun s -> s.PatternId = "tiny")
        // Even with 0 fitness, floor prevents complete extinction
        Assert.True(tiny.Proportion > 0.0, "Floor should prevent extinction")

    [<Fact>]
    let ``step with equal fitness preserves proportions`` () =
        let species = [
            mkSpecies "a" "A" 0.6 0.5
            mkSpecies "b" "B" 0.4 0.5
        ]
        let after = step 0.1 0.001 species
        let a = after |> List.find (fun s -> s.PatternId = "a")
        // Equal fitness = no change (delta = x * (f - f_avg) = x * 0 = 0)
        Assert.True(abs (a.Proportion - 0.6) < 0.01,
            sprintf "Should stay near 0.6, got %.4f" a.Proportion)

    [<Fact>]
    let ``step handles empty species`` () =
        Assert.Empty(step 0.1 0.01 [])

    // =========================================================================
    // detectESS
    // =========================================================================

    [<Fact>]
    let ``detectESS marks highest fitness species as stable`` () =
        let species = [
            mkSpecies "best" "Best" 0.5 0.95
            mkSpecies "ok" "OK" 0.3 0.6
            mkSpecies "weak" "Weak" 0.2 0.3
        ]
        let result = detectESS species
        let best = result |> List.find (fun s -> s.PatternId = "best")
        let weak = result |> List.find (fun s -> s.PatternId = "weak")
        Assert.True(best.IsStable, "Best should be ESS")
        Assert.False(weak.IsStable, "Weak should not be ESS")

    [<Fact>]
    let ``detectESS handles ties`` () =
        let species = [
            mkSpecies "a" "A" 0.5 0.8
            mkSpecies "b" "B" 0.5 0.8
        ]
        let result = detectESS species
        Assert.True(result |> List.forall (fun s -> s.IsStable),
            "Tied species should both be ESS")

    [<Fact>]
    let ``detectESS handles empty list`` () =
        Assert.Empty(detectESS [])

    // =========================================================================
    // simulate (full replicator dynamics)
    // =========================================================================

    [<Fact>]
    let ``simulate converges strong species to dominance`` () =
        let initial = [
            mkSpecies "strong" "Strong" 0.33 0.9
            mkSpecies "medium" "Medium" 0.34 0.5
            mkSpecies "weak" "Weak" 0.33 0.1
        ]
        let config = { defaultConfig with Steps = 100 }
        let result = simulate config initial
        let strong = result.Species |> List.find (fun s -> s.PatternId = "strong")
        Assert.True(strong.Proportion > 0.7,
            sprintf "Strong should dominate: %.4f" strong.Proportion)
        Assert.True(strong.IsStable, "Strong should be ESS")

    [<Fact>]
    let ``simulate prunes near-extinct species`` () =
        let initial = [
            mkSpecies "dominant" "Dom" 0.95 0.99
            mkSpecies "doomed" "Doomed" 0.05 0.01
        ]
        let config = { defaultConfig with Steps = 200; PruneThreshold = 0.01 }
        let result = simulate config initial
        // Doomed should be pruned or near-zero
        let dominated = result.Pruned |> List.exists (fun s -> s.PatternId = "doomed")
        let survived = result.Species |> List.tryFind (fun s -> s.PatternId = "doomed")
        let isPruned = dominated || (survived.IsSome && survived.Value.Proportion < 0.02)
        Assert.True(isPruned, "Doomed species should be pruned or near-extinct")

    [<Fact>]
    let ``simulate records trajectory`` () =
        let initial = [
            mkSpecies "a" "A" 0.5 0.8
            mkSpecies "b" "B" 0.5 0.3
        ]
        let config = { defaultConfig with Steps = 20 }
        let result = simulate config initial
        Assert.Equal(2, result.Trajectory.Length)
        for (_, history) in result.Trajectory do
            // Initial value + 20 steps = 21 entries
            Assert.Equal(21, history.Length)

    [<Fact>]
    let ``simulate reports correct steps run`` () =
        let initial = [ mkSpecies "a" "A" 1.0 0.5 ]
        let config = { defaultConfig with Steps = 42 }
        let result = simulate config initial
        Assert.Equal(42, result.StepsRun)

    [<Fact>]
    let ``simulate handles single species`` () =
        let initial = [ mkSpecies "solo" "Solo" 1.0 0.7 ]
        let result = simulate defaultConfig initial
        Assert.Single(result.Species) |> ignore
        let solo = result.Species.[0]
        Assert.True(abs (solo.Proportion - 1.0) < 0.001)
        Assert.True(solo.IsStable)

    [<Fact>]
    let ``simulate handles empty initial`` () =
        let result = simulate defaultConfig []
        Assert.Empty(result.Species)
        Assert.Equal(0, result.StepsRun)

    // =========================================================================
    // evolveEcosystem (convenience function)
    // =========================================================================

    [<Fact>]
    let ``evolveEcosystem produces valid result from weighted rules`` () =
        let rules = [
            mkRule "p1" "CoT" 0.5 0.9 20
            mkRule "p2" "ReAct" 0.3 0.4 15
            mkRule "p3" "ToT" 0.2 0.7 10
        ]
        let outcomes = Map.ofList [
            ("p1", [ for _ in 1..18 -> (true, 800L) ] @ [ for _ in 1..2 -> (false, 2000L) ])
            ("p2", [ for _ in 1..6 -> (true, 1500L) ] @ [ for _ in 1..9 -> (false, 3000L) ])
            ("p3", [ for _ in 1..7 -> (true, 600L) ] @ [ for _ in 1..3 -> (false, 1000L) ])
        ]
        let result = evolveEcosystem rules outcomes
        Assert.True(result.Species.Length >= 2, "Should have surviving species")
        Assert.True(result.StepsRun > 0)
        // CoT (90% success) should outcompete ReAct (40% success)
        let cot = result.Species |> List.tryFind (fun s -> s.PatternId = "p1")
        let react = result.Species |> List.tryFind (fun s -> s.PatternId = "p2")
        match cot, react with
        | Some c, Some r ->
            Assert.True(c.Proportion > r.Proportion,
                sprintf "CoT (%.4f) should beat ReAct (%.4f)" c.Proportion r.Proportion)
        | Some _, None -> () // ReAct was pruned — even better
        | _ -> Assert.Fail("CoT should exist in results")
