namespace TarsEngine.CustomTransformers

open System
open System.Collections.Generic
open HybridLossFunctions

/// Meta-optimizer for TARS transformer architecture evolution
module MetaOptimizer =

    /// Transformer architecture configuration
    type TransformerConfig = {
        HiddenDim: int
        OutputDim: int
        NumLayers: int
        NumHeads: int
        Dropout: float
        HyperbolicCurvature: float
        LearningRate: float
        BatchSize: int
        LossWeights: HybridLossWeights
    }

    /// Performance metrics for architecture evaluation
    type ArchitectureMetrics = {
        TrainingLoss: float
        ValidationLoss: float
        BeliefAccuracy: float
        ContradictionDetection: float
        EmbeddingCoherence: float
        TrainingTime: TimeSpan
        MemoryUsage: float
        Convergence: float
    }

    /// Evolution strategy parameters
    type EvolutionParams = {
        PopulationSize: int
        EliteCount: int
        MutationRate: float
        CrossoverRate: float
        MaxGenerations: int
        TemperatureDecay: float
        SelectionPressure: float
    }

    let defaultEvolutionParams = {
        PopulationSize = 20
        EliteCount = 5
        MutationRate = 0.1
        CrossoverRate = 0.7
        MaxGenerations = 50
        TemperatureDecay = 0.95
        SelectionPressure = 2.0
    }

    let defaultConfig = {
        HiddenDim = 384
        OutputDim = 128
        NumLayers = 6
        NumHeads = 8
        Dropout = 0.1
        HyperbolicCurvature = 1.0
        LearningRate = 2e-5
        BatchSize = 16
        LossWeights = defaultWeights
    }

    let random = Random()

    /// Mutate a transformer configuration
    let mutateConfig (config: TransformerConfig) (mutationRate: float) : TransformerConfig =
        let mutate value minVal maxVal =
            if random.NextDouble() < mutationRate then
                let change = (random.NextDouble() - 0.5) * 0.2  // ±10% change
                max minVal (min maxVal (value + change * value))
            else value

        let mutateInt value minVal maxVal =
            if random.NextDouble() < mutationRate then
                let change = random.Next(-2, 3)  // ±2 change
                max minVal (min maxVal (value + change))
            else value

        {
            HiddenDim = mutateInt config.HiddenDim 128 1024
            OutputDim = mutateInt config.OutputDim 64 512
            NumLayers = mutateInt config.NumLayers 2 12
            NumHeads = mutateInt config.NumHeads 4 16
            Dropout = mutate config.Dropout 0.0 0.5
            HyperbolicCurvature = mutate config.HyperbolicCurvature 0.1 5.0
            LearningRate = mutate config.LearningRate 1e-6 1e-3
            BatchSize = mutateInt config.BatchSize 4 64
            LossWeights = {
                config.LossWeights with
                    Euclidean = mutate config.LossWeights.Euclidean 0.1 3.0
                    Hyperbolic = mutate config.LossWeights.Hyperbolic 0.1 3.0
                    BeliefAlignment = mutate config.LossWeights.BeliefAlignment 0.5 5.0
            }
        }

    /// Crossover two transformer configurations
    let crossoverConfigs (parent1: TransformerConfig) (parent2: TransformerConfig) : TransformerConfig =
        let selectFrom p1 p2 = if random.NextDouble() < 0.5 then p1 else p2
        let average p1 p2 = (p1 + p2) / 2.0
        let averageInt p1 p2 = (p1 + p2) / 2

        {
            HiddenDim = selectFrom parent1.HiddenDim parent2.HiddenDim
            OutputDim = selectFrom parent1.OutputDim parent2.OutputDim
            NumLayers = selectFrom parent1.NumLayers parent2.NumLayers
            NumHeads = selectFrom parent1.NumHeads parent2.NumHeads
            Dropout = average parent1.Dropout parent2.Dropout
            HyperbolicCurvature = average parent1.HyperbolicCurvature parent2.HyperbolicCurvature
            LearningRate = average parent1.LearningRate parent2.LearningRate
            BatchSize = selectFrom parent1.BatchSize parent2.BatchSize
            LossWeights = {
                Euclidean = average parent1.LossWeights.Euclidean parent2.LossWeights.Euclidean
                Hyperbolic = average parent1.LossWeights.Hyperbolic parent2.LossWeights.Hyperbolic
                Projective = average parent1.LossWeights.Projective parent2.LossWeights.Projective
                DualQuaternion = average parent1.LossWeights.DualQuaternion parent2.LossWeights.DualQuaternion
                BeliefAlignment = average parent1.LossWeights.BeliefAlignment parent2.LossWeights.BeliefAlignment
                Entropy = average parent1.LossWeights.Entropy parent2.LossWeights.Entropy
                Contrastive = average parent1.LossWeights.Contrastive parent2.LossWeights.Contrastive
            }
        }

    /// Evaluate architecture fitness based on multiple criteria
    let evaluateFitness (metrics: ArchitectureMetrics) : float =
        let weights = {|
            TrainingLoss = -2.0      // Lower is better
            ValidationLoss = -3.0    // Lower is better (most important)
            BeliefAccuracy = 2.0     // Higher is better
            ContradictionDetection = 1.5  // Higher is better
            EmbeddingCoherence = 1.0 // Higher is better
            TrainingTime = -0.5      // Lower is better
            MemoryUsage = -0.3       // Lower is better
            Convergence = 1.0        // Higher is better
        |}

        let normalizedMetrics = {|
            TrainingLoss = 1.0 / (1.0 + metrics.TrainingLoss)
            ValidationLoss = 1.0 / (1.0 + metrics.ValidationLoss)
            BeliefAccuracy = metrics.BeliefAccuracy
            ContradictionDetection = metrics.ContradictionDetection
            EmbeddingCoherence = metrics.EmbeddingCoherence
            TrainingTime = 1.0 / (1.0 + metrics.TrainingTime.TotalMinutes)
            MemoryUsage = 1.0 / (1.0 + metrics.MemoryUsage / 1000.0)  // Normalize GB to 0-1
            Convergence = metrics.Convergence
        |}

        weights.TrainingLoss * normalizedMetrics.TrainingLoss +
        weights.ValidationLoss * normalizedMetrics.ValidationLoss +
        weights.BeliefAccuracy * normalizedMetrics.BeliefAccuracy +
        weights.ContradictionDetection * normalizedMetrics.ContradictionDetection +
        weights.EmbeddingCoherence * normalizedMetrics.EmbeddingCoherence +
        weights.TrainingTime * normalizedMetrics.TrainingTime +
        weights.MemoryUsage * normalizedMetrics.MemoryUsage +
        weights.Convergence * normalizedMetrics.Convergence

    /// Simulated annealing for local optimization
    let simulatedAnnealing 
        (initialConfig: TransformerConfig) 
        (evaluateFunc: TransformerConfig -> ArchitectureMetrics)
        (maxIterations: int)
        (initialTemperature: float) : TransformerConfig =
        
        let mutable currentConfig = initialConfig
        let mutable currentMetrics = evaluateFunc currentConfig
        let mutable currentFitness = evaluateFitness currentMetrics
        let mutable bestConfig = currentConfig
        let mutable bestFitness = currentFitness
        let mutable temperature = initialTemperature

        for iteration in 1 .. maxIterations do
            // Generate neighbor configuration
            let neighborConfig = mutateConfig currentConfig 0.1
            let neighborMetrics = evaluateFunc neighborConfig
            let neighborFitness = evaluateFitness neighborMetrics

            // Accept or reject based on simulated annealing criteria
            let deltaFitness = neighborFitness - currentFitness
            let acceptanceProbability = 
                if deltaFitness > 0.0 then 1.0
                else Math.Exp(deltaFitness / temperature)

            if random.NextDouble() < acceptanceProbability then
                currentConfig <- neighborConfig
                currentMetrics <- neighborMetrics
                currentFitness <- neighborFitness

                if currentFitness > bestFitness then
                    bestConfig <- currentConfig
                    bestFitness <- currentFitness

            // Cool down temperature
            temperature <- temperature * 0.99

            if iteration % 10 = 0 then
                printfn "SA Iteration %d: Current=%.4f, Best=%.4f, Temp=%.4f" 
                    iteration currentFitness bestFitness temperature

        bestConfig

    /// Tournament selection for genetic algorithm
    let tournamentSelection (population: (TransformerConfig * float)[]) (tournamentSize: int) : TransformerConfig =
        let tournament = 
            Array.init tournamentSize (fun _ -> population.[random.Next(population.Length)])
            |> Array.maxBy snd
        fst tournament

    /// Genetic algorithm evolution
    let geneticAlgorithmEvolution
        (initialPopulation: TransformerConfig[])
        (evaluateFunc: TransformerConfig -> ArchitectureMetrics)
        (evolutionParams: EvolutionParams) : TransformerConfig[] =
        
        let mutable population = initialPopulation
        let mutable generation = 0
        let mutable bestFitnessHistory = []

        while generation < evolutionParams.MaxGenerations do
            printfn "🧬 Generation %d/%d" (generation + 1) evolutionParams.MaxGenerations

            // Evaluate population
            let evaluatedPopulation =
                population
                |> Array.map (fun config ->
                    let metrics = evaluateFunc config
                    let fitness = evaluateFitness metrics
                    (config, fitness))
                |> Array.sortByDescending snd

            let bestFitness = snd evaluatedPopulation.[0]
            bestFitnessHistory <- bestFitness :: bestFitnessHistory
            printfn "   Best fitness: %.4f" bestFitness

            // Select elites
            let elites = evaluatedPopulation |> Array.take evolutionParams.EliteCount |> Array.map fst

            // Generate new population
            let newPopulation = Array.zeroCreate evolutionParams.PopulationSize

            // Keep elites
            Array.blit elites 0 newPopulation 0 evolutionParams.EliteCount

            // Generate offspring
            for i in evolutionParams.EliteCount .. evolutionParams.PopulationSize - 1 do
                if random.NextDouble() < evolutionParams.CrossoverRate then
                    // Crossover
                    let parent1 = tournamentSelection evaluatedPopulation 3
                    let parent2 = tournamentSelection evaluatedPopulation 3
                    let offspring = crossoverConfigs parent1 parent2
                    newPopulation.[i] <- mutateConfig offspring evolutionParams.MutationRate
                else
                    // Mutation only
                    let parent = tournamentSelection evaluatedPopulation 3
                    newPopulation.[i] <- mutateConfig parent evolutionParams.MutationRate

            population <- newPopulation
            generation <- generation + 1

        // Return final population sorted by fitness
        population
        |> Array.map (fun config -> 
            let metrics = evaluateFunc config
            let fitness = evaluateFitness metrics
            (config, fitness))
        |> Array.sortByDescending snd
        |> Array.map fst

    /// Mock evaluation function for testing
    let mockEvaluateConfig (config: TransformerConfig) : ArchitectureMetrics =
        // Simulate training and evaluation
        let complexity = float (config.HiddenDim * config.NumLayers * config.NumHeads) / 10000.0
        let trainingLoss = 0.5 + random.NextDouble() * 0.3 + complexity * 0.1
        let validationLoss = trainingLoss + random.NextDouble() * 0.1
        
        {
            TrainingLoss = trainingLoss
            ValidationLoss = validationLoss
            BeliefAccuracy = 0.7 + random.NextDouble() * 0.25
            ContradictionDetection = 0.6 + random.NextDouble() * 0.3
            EmbeddingCoherence = 0.8 + random.NextDouble() * 0.15
            TrainingTime = TimeSpan.FromMinutes(complexity * 10.0 + random.NextDouble() * 5.0)
            MemoryUsage = complexity * 2.0 + random.NextDouble()
            Convergence = 0.5 + random.NextDouble() * 0.4
        }

    /// Demo function for meta-optimization
    let demoMetaOptimization () =
        printfn "🧬 TARS Meta-Optimizer Demo"
        printfn "=========================="
        
        // Create initial population
        let initialPopulation = 
            Array.init 10 (fun _ -> mutateConfig defaultConfig 0.2)
        
        printfn "🔬 Running Genetic Algorithm Evolution..."
        let evolvedPopulation = geneticAlgorithmEvolution initialPopulation mockEvaluateConfig defaultEvolutionParams
        
        printfn ""
        printfn "📊 Top 3 Evolved Configurations:"
        for i in 0 .. min 2 (evolvedPopulation.Length - 1) do
            let config = evolvedPopulation.[i]
            let metrics = mockEvaluateConfig config
            let fitness = evaluateFitness metrics
            printfn "   %d. Fitness: %.4f" (i + 1) fitness
            printfn "      HiddenDim: %d, Layers: %d, Heads: %d" config.HiddenDim config.NumLayers config.NumHeads
            printfn "      Dropout: %.3f, Curvature: %.3f" config.Dropout config.HyperbolicCurvature
            printfn "      Training Loss: %.4f, Validation Loss: %.4f" metrics.TrainingLoss metrics.ValidationLoss
        
        printfn ""
        printfn "🔥 Running Simulated Annealing on Best Configuration..."
        let bestConfig = evolvedPopulation.[0]
        let optimizedConfig = simulatedAnnealing bestConfig mockEvaluateConfig 50 1.0
        let finalMetrics = mockEvaluateConfig optimizedConfig
        let finalFitness = evaluateFitness finalMetrics
        
        printfn ""
        printfn "🎯 Final Optimized Configuration:"
        printfn "   Fitness: %.4f" finalFitness
        printfn "   HiddenDim: %d, Layers: %d, Heads: %d" optimizedConfig.HiddenDim optimizedConfig.NumLayers optimizedConfig.NumHeads
        printfn "   Training Loss: %.4f, Validation Loss: %.4f" finalMetrics.TrainingLoss finalMetrics.ValidationLoss
        
        printfn ""
        printfn "✅ Meta-optimization demo complete!"
        
        optimizedConfig
