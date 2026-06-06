namespace Tars.Tests

open System
open Xunit
open Tars.Evolution
open Tars.Evolution.MachinBridge
open Tars.Evolution.EvolutionaryPatternBreeder
open Tars.Cortex
open Tars.Cortex.WoTTypes

/// Tests for MachinBridge (ix bridge) and EvolutionaryPatternBreeder.
module EvolutionaryBreederTests =

    /// Helper to construct PatternOutcome without ambiguity with PatternOutcomeDto.
    let private mkOutcome (kind: PatternKind) (goal: string) (success: bool) (durationMs: int64) : PatternOutcomeStore.PatternOutcome =
        { PatternKind = kind; Goal = goal; Success = success; DurationMs = durationMs; Timestamp = DateTime.UtcNow }

    // =========================================================================
    // MachinBridge.FallbackGA tests
    // =========================================================================

    [<Fact>]
    let ``FallbackGA minimizes sphere function`` () =
        let sphere (x: float array) =
            x |> Array.sumBy (fun v -> v * v)

        let config =
            { FallbackGA.defaultGAConfig with
                PopulationSize = 20
                Generations = 50
                Bounds = (-5.0, 5.0) }

        let result = FallbackGA.minimize config sphere 3
        // Should find something reasonably close to origin
        Assert.True(result.BestValue < 1.0,
            sprintf "Expected best value < 1.0, got %.4f" result.BestValue)

    [<Fact>]
    let ``FallbackGA returns correct iteration count`` () =
        let config =
            { FallbackGA.defaultGAConfig with
                Generations = 25 }
        let result = FallbackGA.minimize config (fun x -> x |> Array.sum) 2
        Assert.Equal(25, result.Iterations)

    [<Fact>]
    let ``FallbackGA respects bounds approximately`` () =
        let config =
            { FallbackGA.defaultGAConfig with
                Bounds = (0.0, 1.0)
                Generations = 10 }
        let result = FallbackGA.minimize config (fun x -> x |> Array.sumBy (fun v -> v * v)) 3
        // Sphere function minimum is at origin — GA should stay near bounds
        for p in result.BestParams do
            Assert.True(p >= -0.5 && p <= 1.5,
                sprintf "Param %.3f far outside bounds" p)

    // =========================================================================
    // MachinBridge output parsing tests
    // =========================================================================

    [<Fact>]
    let ``parseOptimizeOutput parses ix output`` () =
        let output = """
  GeneticAlgorithm:
    Best value:   0.000123
    Best params:  [0.001, -0.002, 0.003]
    Iterations:   500
    Converged:    true
"""
        let result = MachinBridge.parseOptimizeOutput output
        Assert.True(abs (result.BestValue - 0.000123) < 1e-8)
        Assert.Equal<float list>([0.001; -0.002; 0.003], result.BestParams)
        Assert.Equal(500, result.Iterations)
        Assert.True(result.Converged)

    [<Fact>]
    let ``parseOptimizeOutput handles incomplete output`` () =
        let output = "No results"
        let result = MachinBridge.parseOptimizeOutput output
        Assert.Equal(0.0, result.BestValue)
        Assert.Empty(result.BestParams)

    // =========================================================================
    // PatternGenome tests
    // =========================================================================

    [<Fact>]
    let ``genome round-trips through array`` () =
        let g = defaultGenome
        let arr = toArray g
        let g2 = fromArray arr
        Assert.Equal(g.CotWeight, g2.CotWeight)
        Assert.Equal(g.ReactWeight, g2.ReactWeight)
        Assert.Equal(g.TotWeight, g2.TotWeight)
        Assert.Equal(g.GotWeight, g2.GotWeight)
        Assert.Equal(g.StepMultiplier, g2.StepMultiplier)
        Assert.Equal(g.Temperature, g2.Temperature)
        Assert.Equal(g.ConfidenceThreshold, g2.ConfidenceThreshold)
        Assert.Equal(g.BranchingFactor, g2.BranchingFactor)

    [<Fact>]
    let ``fromArray clamps values to valid ranges`` () =
        let arr = [| -5.0; 10.0; -1.0; 2.0; 0.0; 99.0; -1.0; 100.0 |]
        let g = fromArray arr
        Assert.Equal(0.0, g.CotWeight)
        Assert.Equal(1.0, g.ReactWeight)
        Assert.Equal(0.0, g.TotWeight)
        Assert.Equal(1.0, g.GotWeight)
        Assert.Equal(0.5, g.StepMultiplier)
        Assert.Equal(1.5, g.Temperature)
        Assert.Equal(0.3, g.ConfidenceThreshold)
        Assert.Equal(5.0, g.BranchingFactor)

    [<Fact>]
    let ``toArray has correct dimension`` () =
        let arr = toArray defaultGenome
        Assert.Equal(genomeDimension, arr.Length)

    // =========================================================================
    // Fitness function tests
    // =========================================================================

    [<Fact>]
    let ``computeFitness returns 1.0 for empty outcomes`` () =
        let fitness = computeFitness [] (toArray defaultGenome)
        Assert.Equal(1.0, fitness)

    [<Fact>]
    let ``computeFitness rewards high-weight successful patterns`` () =
        let outcomes = [ mkOutcome PatternKind.ChainOfThought "test" true 1000L ]

        // Genome with high CoT weight should score better for CoT successes
        let highCot = { defaultGenome with CotWeight = 0.9 }
        let lowCot = { defaultGenome with CotWeight = 0.1 }

        let fitnessHigh = computeFitness outcomes (toArray highCot)
        let fitnessLow = computeFitness outcomes (toArray lowCot)

        // Lower fitness is better (minimization) — high CoT weight + CoT success = lower penalty
        Assert.True(fitnessHigh < fitnessLow,
            sprintf "High CoT weight (%.3f) should have lower fitness than low (%.3f)" fitnessHigh fitnessLow)

    [<Fact>]
    let ``computeFitness penalizes high-weight failed patterns`` () =
        let outcomes = [ mkOutcome PatternKind.ReAct "test" false 5000L ]

        let highReact = { defaultGenome with ReactWeight = 0.9 }
        let lowReact = { defaultGenome with ReactWeight = 0.1 }

        let fitnessHigh = computeFitness outcomes (toArray highReact)
        let fitnessLow = computeFitness outcomes (toArray lowReact)

        // High ReAct weight + ReAct failure = higher penalty
        Assert.True(fitnessHigh > fitnessLow,
            sprintf "High ReAct weight (%.3f) should have higher fitness than low (%.3f) on failure" fitnessHigh fitnessLow)

    // =========================================================================
    // Breeding tests
    // =========================================================================

    [<Fact>]
    let ``breed produces valid genome`` () =
        let outcomes =
            [ for i in 1..20 do
                mkOutcome PatternKind.ChainOfThought (sprintf "goal %d" i) (i % 3 <> 0) (int64 (i * 100)) ]

        let result = breed None outcomes 10
        Assert.True(result.BestFitness >= 0.0)
        Assert.True(result.Generations > 0)
        Assert.False(result.UsedMachinDeOuf) // UsedMachinDeOuf field name kept for compatibility
        Assert.True(result.Recommendation.Length > 0)

        // Genome should be in valid ranges
        let g = result.BestGenome
        Assert.True(g.CotWeight >= 0.0 && g.CotWeight <= 1.0)
        Assert.True(g.Temperature >= 0.1 && g.Temperature <= 1.5)
        Assert.True(g.StepMultiplier >= 0.5 && g.StepMultiplier <= 3.0)

    // =========================================================================
    // Pattern suggestion tests
    // =========================================================================

    [<Fact>]
    let ``suggestPattern returns valid pattern kind`` () =
        let genome = defaultGenome
        let pattern = suggestPattern genome "Analyze code for bugs"
        let validKinds =
            [ PatternKind.ChainOfThought; PatternKind.ReAct
              PatternKind.TreeOfThoughts; PatternKind.GraphOfThoughts ]
        Assert.Contains(pattern, validKinds)

    [<Fact>]
    let ``suggestPattern boosts ReAct for code goals`` () =
        let genome = { defaultGenome with ReactWeight = 0.4; CotWeight = 0.41 }
        let pattern = suggestPattern genome "Generate code for data-processing pipeline"
        // ReAct should get 1.5x boost (0.4*1.5=0.6) > CoT (0.41)
        Assert.Equal(PatternKind.ReAct, pattern)

    // =========================================================================
    // MachinBridge config tests
    // =========================================================================

    [<Fact>]
    let ``defaultConfig has sensible defaults`` () =
        let config = MachinBridge.defaultConfig
        Assert.Equal("cargo", config.SkillPath)
        Assert.Equal(TimeSpan.FromSeconds(30.0), config.Timeout)
        Assert.True(config.WorkingDir.IsNone)
