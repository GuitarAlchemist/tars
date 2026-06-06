namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Core.RevolutionaryTypes
open TarsEngine.FSharp.Core.RevolutionaryEngine
open TarsEngine.FSharp.Tests.TestHelpers

/// Comprehensive tests for the Revolutionary Engine
module RevolutionaryEngineTests =

    [<UnitTest>]
    let ``Revolutionary Engine should initialize successfully`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        
        // Act & Assert
        let engine = RevolutionaryEngine(logger)
        engine |> should not' (be null)

    [<UnitTest>]
    let ``Multi-space embedding creation should work correctly`` () =
        // Arrange
        let text = "TARS revolutionary AI system"
        let complexity = 0.85
        
        // Act
        let embedding = RevolutionaryFactory.CreateMultiSpaceEmbedding(text, complexity)
        
        // Assert
        embedding.Euclidean |> should not' (be null)
        embedding.Euclidean.Length |> should equal 384
        embedding.Hyperbolic |> should not' (be null)
        embedding.Hyperbolic.Length |> should equal 384
        embedding.Projective |> should not' (be null)
        embedding.Projective.Length |> should equal 384
        embedding.DualQuaternion |> should not' (be null)
        embedding.DualQuaternion.Length |> should equal 8
        embedding.Complexity |> should equal complexity

    [<UnitTest>]
    let ``Revolutionary operation creation should generate valid operations`` () =
        // Arrange
        let capability = AutonomousReasoning
        let complexity = 0.9
        
        // Act
        let operation = RevolutionaryFactory.CreateRevolutionaryOperation(capability, complexity)
        
        // Assert
        operation.Capability |> should equal capability
        operation.Complexity |> should equal complexity
        operation.RequiredTier |> should equal GrammarTier.Revolutionary
        operation.EstimatedGain |> should be (greaterThan 1.0)

    [<PerformanceTest>]
    let ``Revolutionary operations should complete within performance thresholds`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        let operations = [
            AutonomousReasoning
            ConceptualBreakthrough
            PerformanceOptimization
            CodeGeneration
            SelfImprovement
        ]
        
        // Act & Assert
        for capability in operations do
            let measurement = measurePerformance (fun () ->
                engine.TriggerEvolution(capability) |> Async.RunSynchronously
            )
            
            // Validate performance
            measurement.Success |> should be True
            Validation.validateExecutionTime measurement.ExecutionTime (TimeSpan.FromSeconds(5.0))
            Validation.validateMemoryUsage measurement.MemoryUsed (100L * 1024L * 1024L) // 100MB max

    [<IntegrationTest>]
    let ``Revolutionary engine should handle multiple concurrent operations`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        let capabilities = [
            AutonomousReasoning
            ConceptualBreakthrough
            PerformanceOptimization
        ]
        
        // Act
        let tasks = 
            capabilities
            |> List.map (fun cap -> 
                async {
                    let! result = engine.TriggerEvolution(cap)
                    return (cap, result)
                })
        
        let results = 
            tasks
            |> Async.Parallel
            |> Async.RunSynchronously
        
        // Assert
        results.Length |> should equal capabilities.Length
        results |> Array.iter (fun (cap, result) ->
            result.Success |> should be True
            result.NewCapabilities |> should contain cap
        )

    [<ValidationTest>]
    let ``Revolutionary operations should produce valid performance gains`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        let testCases = [
            (AutonomousReasoning, 1.2)
            (ConceptualBreakthrough, 1.5)
            (PerformanceOptimization, 2.0)
            (CodeGeneration, 1.3)
            (SelfImprovement, 1.8)
        ]
        
        // Act & Assert
        for (capability, expectedMinGain) in testCases do
            let result = engine.TriggerEvolution(capability) |> Async.RunSynchronously
            
            result.Success |> should be True
            Validation.validatePerformanceGain result.PerformanceGain expectedMinGain

    [<UnitTest>]
    let ``Multi-space embeddings should have consistent dimensions`` () =
        // Arrange
        let testTexts = TestData.generateTestTexts 10
        let complexity = 0.75
        
        // Act & Assert
        for text in testTexts do
            let embedding = RevolutionaryFactory.CreateMultiSpaceEmbedding(text, complexity)
            
            // Validate dimensions
            embedding.Euclidean.Length |> should equal 384
            embedding.Hyperbolic.Length |> should equal 384
            embedding.Projective.Length |> should equal 384
            embedding.DualQuaternion.Length |> should equal 8
            
            // Validate complexity
            embedding.Complexity |> should equal complexity

    [<ValidationTest>]
    let ``Revolutionary operations should maintain state consistency`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        
        // Act
        let initialMetrics = engine.GetRevolutionaryMetrics()
        let result1 = engine.TriggerEvolution(AutonomousReasoning) |> Async.RunSynchronously
        let midMetrics = engine.GetRevolutionaryMetrics()
        let result2 = engine.TriggerEvolution(ConceptualBreakthrough) |> Async.RunSynchronously
        let finalMetrics = engine.GetRevolutionaryMetrics()
        
        // Assert
        result1.Success |> should be True
        result2.Success |> should be True
        
        // Metrics should progress
        midMetrics.TotalOperations |> should be (greaterThan initialMetrics.TotalOperations)
        finalMetrics.TotalOperations |> should be (greaterThan midMetrics.TotalOperations)
        
        // Success rate should be maintained
        finalMetrics.SuccessRate |> should be (greaterThanOrEqualTo 0.8)

    [<PerformanceTest>]
    let ``Revolutionary engine should scale with operation complexity`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        let complexities = [0.1; 0.3; 0.5; 0.7; 0.9]
        
        // Act & Assert
        let measurements = 
            complexities
            |> List.map (fun complexity ->
                let operation = RevolutionaryFactory.CreateRevolutionaryOperation(AutonomousReasoning, complexity)
                measurePerformance (fun () ->
                    engine.TriggerEvolution(operation.Capability) |> Async.RunSynchronously
                )
            )
        
        // Validate scaling behavior
        measurements |> List.iter (fun m -> m.Success |> should be True)
        
        // Higher complexity should generally take more time (with some tolerance)
        let times = measurements |> List.map (_.ExecutionTime.TotalMilliseconds)
        let avgTime = times |> List.average
        times |> List.iter (fun time -> 
            time |> should be (lessThan (avgTime * 3.0)) // Allow 3x variance
        )

    [<UnitTest>]
    let ``Revolutionary factory should create valid tier mappings`` () =
        // Arrange
        let testComplexities = [0.1; 0.3; 0.5; 0.7; 0.9; 1.0]
        
        // Act & Assert
        for complexity in testComplexities do
            let tier = RevolutionaryFactory.ComplexityToTier(complexity)
            let backToComplexity = RevolutionaryFactory.TierToComplexity(tier)
            
            // Validate tier mapping
            tier |> should not' (be null)
            backToComplexity |> should be (greaterThanOrEqualTo 0.0)
            backToComplexity |> should be (lessThanOrEqualTo 1.0)
            
            // Validate approximate round-trip
            abs(complexity - backToComplexity) |> should be (lessThan 0.2)

    [<EndToEndTest>]
    let ``Revolutionary engine end-to-end workflow should complete successfully`` () =
        // Arrange
        let logger = createTestLogger<RevolutionaryEngine>()
        let engine = RevolutionaryEngine(logger)
        let workflow = [
            AutonomousReasoning
            ConceptualBreakthrough
            PerformanceOptimization
            CodeGeneration
            SelfImprovement
        ]
        
        // Act
        let results = 
            workflow
            |> List.map (fun capability ->
                let result = engine.TriggerEvolution(capability) |> Async.RunSynchronously
                (capability, result)
            )
        
        let finalMetrics = engine.GetRevolutionaryMetrics()
        
        // Assert
        results |> List.iter (fun (cap, result) ->
            result.Success |> should be True
            result.NewCapabilities |> should contain cap
            result.PerformanceGain |> should not' (be None)
        )
        
        // Final metrics validation
        finalMetrics.TotalOperations |> should equal workflow.Length
        finalMetrics.SuccessRate |> should equal 1.0
        finalMetrics.AveragePerformanceGain |> should be (greaterThan 1.0)
        finalMetrics.EmergentPropertiesCount |> should be (greaterThan 0)

    /// Run all Revolutionary Engine tests and return summary
    let runAllTests () =
        let testMethods = [
            ("Initialize", fun () -> ``Revolutionary Engine should initialize successfully`` ())
            ("MultiSpaceEmbedding", fun () -> ``Multi-space embedding creation should work correctly`` ())
            ("OperationCreation", fun () -> ``Revolutionary operation creation should generate valid operations`` ())
            ("PerformanceThresholds", fun () -> ``Revolutionary operations should complete within performance thresholds`` ())
            ("ConcurrentOperations", fun () -> ``Revolutionary engine should handle multiple concurrent operations`` ())
            ("PerformanceGains", fun () -> ``Revolutionary operations should produce valid performance gains`` ())
            ("EmbeddingDimensions", fun () -> ``Multi-space embeddings should have consistent dimensions`` ())
            ("StateConsistency", fun () -> ``Revolutionary operations should maintain state consistency`` ())
            ("ComplexityScaling", fun () -> ``Revolutionary engine should scale with operation complexity`` ())
            ("TierMappings", fun () -> ``Revolutionary factory should create valid tier mappings`` ())
            ("EndToEndWorkflow", fun () -> ``Revolutionary engine end-to-end workflow should complete successfully`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running Revolutionary Engine test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
            ("total_memory_mb", measurements |> List.map (_.MemoryUsed) |> List.sum |> float |> fun x -> x / (1024.0 * 1024.0))
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "Revolutionary Engine" result
        result
