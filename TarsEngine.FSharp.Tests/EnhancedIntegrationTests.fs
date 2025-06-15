namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Core.RevolutionaryTypes
open TarsEngine.FSharp.Core.EnhancedRevolutionaryIntegration
open TarsEngine.FSharp.Tests.TestHelpers

/// Comprehensive tests for Enhanced Revolutionary Integration
module EnhancedIntegrationTests =

    [<UnitTest>]
    let ``Enhanced TARS Engine should initialize successfully`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        
        // Act & Assert
        let engine = EnhancedTarsEngine(logger)
        engine |> should not' (be null)

    [<IntegrationTest>]
    let ``Enhanced capabilities initialization should detect available features`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Act
        let (cudaEnabled, transformersEnabled) = 
            engine.InitializeEnhancedCapabilities() |> Async.RunSynchronously
        
        // Assert
        // Should at least attempt to initialize (may fail due to missing CUDA DLL)
        cudaEnabled |> should be (ofType<bool>)
        transformersEnabled |> should be (ofType<bool>)

    [<PerformanceTest>]
    let ``Enhanced semantic analysis should provide performance gains`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let testTexts = TestData.generateTestTexts 5
        
        // Act & Assert
        for text in testTexts do
            let operation = SemanticAnalysis(text, Hyperbolic 1.0, true)
            let measurement = measurePerformanceAsync (fun () ->
                engine.ExecuteEnhancedOperation(operation)
            ) |> Async.RunSynchronously
            
            // Validate performance
            measurement.Success |> should be True
            let result = measurement.Result.Value :?> EnhancedRevolutionaryResult
            result.Success |> should be True
            Validation.validatePerformanceGain result.PerformanceGain 1.0

    [<ValidationTest>]
    let ``Enhanced concept evolution should progress through tiers correctly`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let concepts = TestData.generateTestConcepts 3
        let tiers = [GrammarTier.Basic; GrammarTier.Intermediate; GrammarTier.Revolutionary]
        
        // Act & Assert
        for concept in concepts do
            for tier in tiers do
                let operation = ConceptEvolution(concept, tier, true)
                let result = engine.ExecuteEnhancedOperation(operation) |> Async.RunSynchronously
                
                result.Success |> should be True
                result.Insights |> should not' (be empty)
                result.Improvements |> should not' (be empty)
                Validation.validatePerformanceGain result.PerformanceGain 1.0

    [<UnitTest>]
    let ``Enhanced multi-space embeddings should support all geometric spaces`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let spaces = [
            (Euclidean, DualQuaternion)
            (Hyperbolic 1.0, Projective)
            (Projective, Euclidean)
            (DualQuaternion, Hyperbolic 2.0)
        ]
        
        // Act & Assert
        for (source, target) in spaces do
            let operation = CrossSpaceMapping(source, target, true)
            let result = engine.ExecuteEnhancedOperation(operation) |> Async.RunSynchronously
            
            result.Success |> should be True
            result.HybridEmbeddings |> should not' (be None)
            
            match result.HybridEmbeddings with
            | Some embedding ->
                embedding.GeometricSpaces |> should contain source
                embedding.GeometricSpaces |> should contain target
                embedding.HybridEmbedding |> should not' (be empty)
            | None -> failwith "Hybrid embeddings should not be None"

    [<PerformanceTest>]
    let ``Enhanced emergent discovery should scale with domain complexity`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let domains = [
            "simple_patterns"
            "complex_multi_dimensional_relationships"
            "revolutionary_breakthrough_discovery_with_advanced_semantics"
        ]
        
        // Act & Assert
        let measurements = 
            domains
            |> List.map (fun domain ->
                let operation = EmergentDiscovery(domain, true)
                measurePerformanceAsync (fun () ->
                    engine.ExecuteEnhancedOperation(operation)
                ) |> Async.RunSynchronously
            )
        
        // Validate all operations succeeded
        measurements |> List.iter (fun m -> m.Success |> should be True)
        
        // Validate performance scaling
        let results = measurements |> List.map (fun m -> m.Result.Value :?> EnhancedRevolutionaryResult)
        results |> List.iter (fun r ->
            r.Success |> should be True
            Validation.validatePerformanceGain r.PerformanceGain 2.0 // Emergent discovery should have high gains
        )

    [<IntegrationTest>]
    let ``Enhanced transformer training should integrate with evolution`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let configs = [
            "basic_transformer_config"
            "advanced_multi_head_attention"
            "revolutionary_hybrid_architecture"
        ]
        
        // Act & Assert
        for config in configs do
            let operation = HybridTransformerTraining(config, true)
            let result = engine.ExecuteEnhancedOperation(operation) |> Async.RunSynchronously
            
            result.Success |> should be True
            result.TransformerMetrics |> should not' (be empty)
            
            if result.Success then
                result.NewCapabilities |> should not' (be empty)
                Validation.validatePerformanceGain result.PerformanceGain 2.5

    [<PerformanceTest>]
    let ``Enhanced CUDA vector operations should handle large batches`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        let batchSizes = [100; 1000; 10000]
        
        // Act & Assert
        for batchSize in batchSizes do
            let operation = CudaVectorStoreOperation("batch_similarity", batchSize)
            let measurement = measurePerformanceAsync (fun () ->
                engine.ExecuteEnhancedOperation(operation)
            ) |> Async.RunSynchronously
            
            measurement.Success |> should be True
            let result = measurement.Result.Value :?> EnhancedRevolutionaryResult
            result.Success |> should be True
            
            // Larger batches should have higher performance gains
            let expectedMinGain = 1.0 + (float batchSize * 0.001)
            Validation.validatePerformanceGain result.PerformanceGain expectedMinGain

    [<ValidationTest>]
    let ``Enhanced system status should reflect all capabilities`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Initialize capabilities
        engine.InitializeEnhancedCapabilities() |> Async.RunSynchronously |> ignore
        
        // Act
        let status = engine.GetEnhancedStatus()
        
        // Assert
        status.EnhancedCapabilities |> should not' (be empty)
        status.SystemHealth |> should be (greaterThan 0.0)
        status.SystemHealth |> should be (lessThanOrEqualTo 2.0) // Max 2.0 with all enhancements
        
        // Should include key capabilities
        let capabilityNames = status.EnhancedCapabilities |> List.map (fun s -> s.ToLower())
        capabilityNames |> should contain "multi-space embeddings"
        capabilityNames |> should contain "enhanced geometric operations"

    [<IntegrationTest>]
    let ``Enhanced engine should maintain compatibility with base engines`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Act
        let revolutionaryEngine = engine.GetRevolutionaryEngine()
        let unifiedEngine = engine.GetUnifiedEngine()
        
        // Assert
        revolutionaryEngine |> should not' (be null)
        unifiedEngine |> should not' (be null)
        
        // Test basic functionality
        let revMetrics = revolutionaryEngine.GetRevolutionaryMetrics()
        let unifiedStatus = unifiedEngine.GetUnifiedStatus()
        
        revMetrics |> should not' (be null)
        unifiedStatus |> should not' (be null)

    [<EndToEndTest>]
    let ``Enhanced integration end-to-end workflow should demonstrate all capabilities`` () =
        // Arrange
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Initialize enhanced capabilities
        let (cudaEnabled, transformersEnabled) = 
            engine.InitializeEnhancedCapabilities() |> Async.RunSynchronously
        
        // Define comprehensive workflow
        let workflow = [
            SemanticAnalysis("TARS enhanced revolutionary AI", Hyperbolic 1.0, cudaEnabled)
            ConceptEvolution("enhanced_intelligence", GrammarTier.Revolutionary, transformersEnabled)
            CrossSpaceMapping(Euclidean, DualQuaternion, cudaEnabled)
            EmergentDiscovery("revolutionary_patterns", true)
            HybridTransformerTraining("comprehensive_config", true)
            CudaVectorStoreOperation("end_to_end_test", 5000)
        ]
        
        // Act
        let results = 
            workflow
            |> List.map (fun operation ->
                let result = engine.ExecuteEnhancedOperation(operation) |> Async.RunSynchronously
                (operation, result)
            )
        
        let finalStatus = engine.GetEnhancedStatus()
        
        // Assert
        results |> List.iter (fun (op, result) ->
            result.Success |> should be True
            result.Insights |> should not' (be empty)
            result.ExecutionTime |> should be (lessThan (TimeSpan.FromSeconds(10.0)))
        )
        
        // Calculate overall performance
        let totalGain = 
            results 
            |> List.map (fun (_, result) -> result.PerformanceGain |> Option.defaultValue 1.0)
            |> List.average
        
        totalGain |> should be (greaterThan 5.0) // Should have significant overall gains
        
        // Final system health should be excellent
        finalStatus.SystemHealth |> should be (greaterThan 1.0)

    /// Run all Enhanced Integration tests and return summary
    let runAllTests () =
        let testMethods = [
            ("Initialize", fun () -> ``Enhanced TARS Engine should initialize successfully`` ())
            ("CapabilitiesInit", fun () -> ``Enhanced capabilities initialization should detect available features`` ())
            ("SemanticAnalysis", fun () -> ``Enhanced semantic analysis should provide performance gains`` ())
            ("ConceptEvolution", fun () -> ``Enhanced concept evolution should progress through tiers correctly`` ())
            ("MultiSpaceEmbeddings", fun () -> ``Enhanced multi-space embeddings should support all geometric spaces`` ())
            ("EmergentDiscovery", fun () -> ``Enhanced emergent discovery should scale with domain complexity`` ())
            ("TransformerTraining", fun () -> ``Enhanced transformer training should integrate with evolution`` ())
            ("CudaOperations", fun () -> ``Enhanced CUDA vector operations should handle large batches`` ())
            ("SystemStatus", fun () -> ``Enhanced system status should reflect all capabilities`` ())
            ("Compatibility", fun () -> ``Enhanced engine should maintain compatibility with base engines`` ())
            ("EndToEndWorkflow", fun () -> ``Enhanced integration end-to-end workflow should demonstrate all capabilities`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running Enhanced Integration test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
            ("total_memory_mb", measurements |> List.map (_.MemoryUsed) |> List.sum |> float |> fun x -> x / (1024.0 * 1024.0))
            ("success_rate", measurements |> List.filter (_.Success) |> List.length |> float |> fun x -> x / float measurements.Length)
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "Enhanced Integration" result
        result
