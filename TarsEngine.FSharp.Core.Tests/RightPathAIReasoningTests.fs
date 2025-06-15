namespace TarsEngine.FSharp.Core.Tests

open System
open Microsoft.Extensions.Logging
open Xunit
open TarsEngine.FSharp.Core.RightPathAIReasoningIntegration
open TarsEngine.FSharp.Core.BSPReasoningEngine

module RightPathAIReasoningTests =

    let createTestLogger<'T>() =
        LoggerFactory.Create(fun builder -> builder.AddConsole() |> ignore).CreateLogger<'T>()

    [<Fact>]
    let ``Right Path AI Reasoning Engine initializes correctly`` () =
        // Arrange
        let logger = createTestLogger<RightPathAIReasoningEngine>()
        
        // Act
        let engine = RightPathAIReasoningEngine(logger)
        let status = engine.GetRightPathStatus()
        
        // Assert
        Assert.Equal(0, status.TotalDiffusions)
        Assert.Equal(0.0, status.AverageFinalLoss)
        Assert.Equal(1.0, status.AveragePerformanceGain)

    [<Fact>]
    let ``Belief diffusion executes with CPU fallback`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let config = {
                NumAgents = 5
                BeliefDimension = 8
                MaxIterations = 20
                ConvergenceThreshold = 0.1
                LearningRate = 0.1
                UseNashEquilibrium = false
                UseFractalTopology = false
                EnableCudaAcceleration = false // Force CPU
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result = engine.ExecuteRightPathReasoning(config, "Test belief diffusion")
            
            // Assert
            Assert.True(result.Success)
            Assert.True(result.IterationsCompleted > 0)
            Assert.True(result.FinalLoss >= 0.0)
            Assert.False(result.CudaAccelerated) // Should use CPU
        }

    [<Fact>]
    let ``Fractal topology creates correct neighbor connections`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let config = {
                NumAgents = 8
                BeliefDimension = 4
                MaxIterations = 10
                ConvergenceThreshold = 0.1
                LearningRate = 0.1
                UseNashEquilibrium = false
                UseFractalTopology = true // Enable fractal topology
                EnableCudaAcceleration = false
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result = engine.ExecuteRightPathReasoning(config, "Test fractal topology")
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal("fractal", result.FinalNetwork.Topology)
            Assert.True(result.FractalComplexity > 0.0)
            
            // Check that agents have fractal connections
            for agent in result.FinalNetwork.Agents do
                Assert.True(agent.Neighbors.Length > 0)
                Assert.True(agent.Neighbors.Length <= result.FinalNetwork.MaxNeighbors)
        }

    [<Fact>]
    let ``Nash equilibrium detection works correctly`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let config = {
                NumAgents = 6
                BeliefDimension = 4
                MaxIterations = 50
                ConvergenceThreshold = 0.05
                LearningRate = 0.2
                UseNashEquilibrium = true // Enable Nash equilibrium
                UseFractalTopology = true
                EnableCudaAcceleration = false
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result = engine.ExecuteRightPathReasoning(config, "Test Nash equilibrium")
            
            // Assert
            Assert.True(result.Success)
            
            // If converged, Nash equilibrium should be detected
            if result.ConvergenceAchieved then
                Assert.True(result.NashEquilibriumReached)
        }

    [<Fact>]
    let ``Cross-entropy loss decreases over iterations`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let config = {
                NumAgents = 4
                BeliefDimension = 6
                MaxIterations = 30
                ConvergenceThreshold = 0.01
                LearningRate = 0.15
                UseNashEquilibrium = false
                UseFractalTopology = false
                EnableCudaAcceleration = false
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result = engine.ExecuteRightPathReasoning(config, "Test loss reduction")
            
            // Assert
            Assert.True(result.Success)
            Assert.True(result.FinalLoss < 1.0) // Should be less than initial loss
            Assert.True(result.PerformanceGain >= 1.0) // Should show improvement
        }

    [<Fact>]
    let ``BSP integration with Right Path AI works`` () =
        async {
            // Arrange
            let logger = createTestLogger<BSPReasoningEngine>()
            let bspEngine = BSPReasoningEngine(logger)
            
            let problem = {
                ProblemId = "TEST_001"
                Description = "Test integrated BSP + Right Path AI reasoning"
                InitialBeliefs = Map.ofList [("belief1", 0.3); ("belief2", 0.7)]
                TargetBeliefs = Map.ofList [("belief1", 0.8); ("belief2", 0.9)]
                MaxReasoningDepth = 20
                RequiredConfidence = 0.8
                Context = Map.empty
            }
            
            // Act
            let! solution = bspEngine.SolveWithIntegratedRightPathReasoning(problem)
            
            // Assert
            Assert.True(solution.SolutionQuality > 0.0)
            Assert.True(solution.MetaReasoningInsights.Length > 0)
            
            // Check for Right Path AI integration insights
            let hasRightPathInsights = 
                solution.MetaReasoningInsights 
                |> Array.exists (fun insight -> insight.Contains("Right Path AI"))
            Assert.True(hasRightPathInsights)
        }

    [<Fact>]
    let ``Performance metrics are tracked correctly`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let config = {
                NumAgents = 3
                BeliefDimension = 4
                MaxIterations = 15
                ConvergenceThreshold = 0.1
                LearningRate = 0.1
                UseNashEquilibrium = false
                UseFractalTopology = false
                EnableCudaAcceleration = false
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result1 = engine.ExecuteRightPathReasoning(config, "Test 1")
            let! result2 = engine.ExecuteRightPathReasoning(config, "Test 2")
            let status = engine.GetRightPathStatus()
            
            // Assert
            Assert.Equal(2, status.TotalDiffusions)
            Assert.True(status.AveragePerformanceGain >= 1.0)
            Assert.True(status.SystemHealth > 0.0)
            Assert.True(status.RevolutionaryInsights.Length > 0)
        }

    [<Fact>]
    let ``CUDA acceleration flag is respected`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let configCuda = {
                NumAgents = 4
                BeliefDimension = 8
                MaxIterations = 10
                ConvergenceThreshold = 0.1
                LearningRate = 0.1
                UseNashEquilibrium = false
                UseFractalTopology = false
                EnableCudaAcceleration = true // Request CUDA
                CrossEntropyWeight = 1.0
            }
            
            let configCpu = { configCuda with EnableCudaAcceleration = false }
            
            // Act
            let! resultCuda = engine.ExecuteRightPathReasoning(configCuda, "CUDA test")
            let! resultCpu = engine.ExecuteRightPathReasoning(configCpu, "CPU test")
            
            // Assert
            Assert.True(resultCuda.Success)
            Assert.True(resultCpu.Success)
            Assert.False(resultCpu.CudaAccelerated) // CPU should never use CUDA
            // CUDA result may or may not use CUDA depending on availability
        }

    [<Fact>]
    let ``Belief dimensions are preserved throughout diffusion`` () =
        async {
            // Arrange
            let logger = createTestLogger<RightPathAIReasoningEngine>()
            let engine = RightPathAIReasoningEngine(logger)
            
            let beliefDim = 12
            let config = {
                NumAgents = 5
                BeliefDimension = beliefDim
                MaxIterations = 10
                ConvergenceThreshold = 0.1
                LearningRate = 0.1
                UseNashEquilibrium = false
                UseFractalTopology = false
                EnableCudaAcceleration = false
                CrossEntropyWeight = 1.0
            }
            
            // Act
            let! result = engine.ExecuteRightPathReasoning(config, "Dimension test")
            
            // Assert
            Assert.True(result.Success)
            Assert.Equal(beliefDim, result.FinalNetwork.BeliefDimension)
            
            for agent in result.FinalNetwork.Agents do
                Assert.Equal(beliefDim, agent.Belief.Length)
        }
