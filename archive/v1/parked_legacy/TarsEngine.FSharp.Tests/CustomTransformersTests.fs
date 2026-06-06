namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.CustomTransformers.CudaHybridOperations
open TarsEngine.FSharp.Tests.TestHelpers

/// Comprehensive tests for CustomTransformers CUDA operations
module CustomTransformersTests =

    [<UnitTest>]
    let ``Möbius addition should produce valid results`` () =
        // Arrange
        let x = [| 0.1; 0.2; 0.3 |]
        let y = [| 0.4; 0.5; 0.6 |]
        let curvature = 1.0
        
        // Act
        let result = mobiusAdd x y curvature
        
        // Assert
        result |> should not' (be null)
        result.Length |> should equal x.Length
        result |> Array.iter (fun r -> r |> should not' (be nan))
        result |> Array.iter (fun r -> r |> should not' (be infinityOrNaN))

    [<UnitTest>]
    let ``Hyperbolic distance should be non-negative`` () =
        // Arrange
        let u = [| 0.2; 0.3; 0.1 |]
        let v = [| 0.5; 0.1; 0.4 |]
        let curvature = 1.0
        
        // Act
        let distance = hyperbolicDistance u v curvature
        
        // Assert
        distance |> should be (greaterThanOrEqualTo 0.0)
        distance |> should not' (be nan)
        distance |> should not' (be infinityOrNaN)

    [<UnitTest>]
    let ``Projective normalization should preserve vector count`` () =
        // Arrange
        let vectors = [| 
            [| 1.0; 2.0; 3.0 |]
            [| 4.0; 5.0; 6.0 |]
            [| 7.0; 8.0; 9.0 |]
        |]
        
        // Act
        let normalized = projectiveNormalize vectors
        
        // Assert
        normalized |> should not' (be null)
        normalized.Length |> should equal vectors.Length
        normalized |> Array.iter (fun vec ->
            vec |> should not' (be null)
            vec.Length |> should equal vectors.[0].Length
            vec |> Array.iter (fun v -> v |> should not' (be nan))
        )

    [<UnitTest>]
    let ``Dual quaternion norm should be positive for non-zero quaternions`` () =
        // Arrange
        let real = [| 1.0; 0.0; 0.0; 0.0 |]
        let dual = [| 0.0; 1.0; 0.0; 0.0 |]
        
        // Act
        let norm = dualQuaternionNorm real dual
        
        // Assert
        norm |> should be (greaterThan 0.0)
        norm |> should not' (be nan)
        norm |> should not' (be infinityOrNaN)

    [<ValidationTest>]
    let ``Hybrid embedding creation should include all specified components`` () =
        // Arrange
        let euclidean = Some [| 1.0; 0.0; 0.0 |]
        let hyperbolic = Some [| 0.3; 0.2; 0.1 |]
        let projective = Some [| 0.577; 0.577; 0.577 |]
        let dualQuaternion = Some [| 1.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0 |]
        let metadata = Map.ofList [("source", box "test")]
        
        // Act
        let embedding = createHybridEmbedding euclidean hyperbolic projective dualQuaternion metadata
        
        // Assert
        embedding.Euclidean |> should equal euclidean
        embedding.Hyperbolic |> should equal hyperbolic
        embedding.Projective |> should equal projective
        embedding.DualQuaternion |> should equal dualQuaternion
        embedding.Metadata |> should equal metadata

    [<PerformanceTest>]
    let ``CUDA operations should complete within reasonable time`` () =
        // Arrange & Act
        let measurement = measurePerformance (fun () -> testCudaOperations())
        
        // Assert
        measurement.Success |> should be True
        let result = measurement.Result.Value :?> bool
        result |> should be True // Operations should succeed (even with CPU fallback)
        
        // Performance validation
        Validation.validateExecutionTime measurement.ExecutionTime (TimeSpan.FromSeconds(10.0))

    [<ValidationTest>]
    let ``Similarity calculations should work for all geometric spaces`` () =
        // Arrange
        let embedding1 = createHybridEmbedding 
            (Some [| 1.0; 0.0; 0.0 |]) 
            (Some [| 0.3; 0.2; 0.1 |]) 
            (Some [| 0.577; 0.577; 0.577 |]) 
            (Some [| 1.0; 0.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0 |]) 
            (Map.ofList [("source", box "test1")])

        let embedding2 = createHybridEmbedding 
            (Some [| 0.8; 0.6; 0.0 |]) 
            (Some [| 0.4; 0.3; 0.2 |]) 
            (Some [| 0.707; 0.707; 0.0 |]) 
            (Some [| 0.9; 0.1; 0.0; 0.0; 0.1; 0.9; 0.0; 0.0 |]) 
            (Map.ofList [("source", box "test2")])
        
        let spaces = [
            Euclidean
            Hyperbolic 1.0
            Projective
            DualQuaternion
        ]
        
        // Act & Assert
        for space in spaces do
            let similarity = calculateSimilarity space embedding1 embedding2
            
            match similarity with
            | Some sim ->
                sim |> should be (greaterThanOrEqualTo 0.0)
                sim |> should be (lessThanOrEqualTo 1.0)
                sim |> should not' (be nan)
            | None ->
                failwithf "Similarity calculation failed for space: %A" space

    [<UnitTest>]
    let ``Möbius addition should be commutative`` () =
        // Arrange
        let x = [| 0.1; 0.2; 0.3 |]
        let y = [| 0.4; 0.5; 0.6 |]
        let curvature = 1.0
        
        // Act
        let result1 = mobiusAdd x y curvature
        let result2 = mobiusAdd y x curvature
        
        // Assert
        Validation.validateArraySimilarity result1 result2 1e-10

    [<UnitTest>]
    let ``Hyperbolic distance should be symmetric`` () =
        // Arrange
        let u = [| 0.2; 0.3; 0.1 |]
        let v = [| 0.5; 0.1; 0.4 |]
        let curvature = 1.0
        
        // Act
        let distance1 = hyperbolicDistance u v curvature
        let distance2 = hyperbolicDistance v u curvature
        
        // Assert
        abs(distance1 - distance2) |> should be (lessThan 1e-10)

    [<PerformanceTest>]
    let ``Batch operations should scale efficiently`` () =
        // Arrange
        let batchSizes = [10; 100; 1000]
        
        // Act & Assert
        for batchSize in batchSizes do
            let vectors = Array.init batchSize (fun i -> 
                TestData.generateFloatArray 3 (0.0, 1.0)
            )
            
            let measurement = measurePerformance (fun () ->
                projectiveNormalize vectors
            )
            
            measurement.Success |> should be True
            let result = measurement.Result.Value :?> float[][]
            result.Length |> should equal batchSize
            
            // Validate reasonable performance scaling
            let timePerVector = measurement.ExecutionTime.TotalMilliseconds / float batchSize
            timePerVector |> should be (lessThan 10.0) // Max 10ms per vector

    [<ValidationTest>]
    let ``Projective normalization should produce unit vectors`` () =
        // Arrange
        let vectors = [| 
            [| 3.0; 4.0; 0.0 |]  // Should normalize to [0.6, 0.8, 0.0]
            [| 1.0; 1.0; 1.0 |]  // Should normalize to [0.577, 0.577, 0.577]
        |]
        
        // Act
        let normalized = projectiveNormalize vectors
        
        // Assert
        for vec in normalized do
            let norm = vec |> Array.map (fun x -> x * x) |> Array.sum |> sqrt
            abs(norm - 1.0) |> should be (lessThan 1e-6) // Should be unit vectors

    [<IntegrationTest>]
    let ``Hybrid embeddings demo should complete successfully`` () =
        // Arrange & Act
        let measurement = measurePerformance (fun () -> demoHybridEmbeddings())
        
        // Assert
        measurement.Success |> should be True
        Validation.validateExecutionTime measurement.ExecutionTime (TimeSpan.FromSeconds(30.0))

    [<ValidationTest>]
    let ``CUDA operations should handle edge cases gracefully`` () =
        // Arrange
        let edgeCases = [
            // Zero vectors
            ([| 0.0; 0.0; 0.0 |], [| 0.0; 0.0; 0.0 |])
            // Very small vectors
            ([| 1e-10; 1e-10; 1e-10 |], [| 1e-10; 1e-10; 1e-10 |])
            // Large vectors
            ([| 1000.0; 1000.0; 1000.0 |], [| 1000.0; 1000.0; 1000.0 |])
        ]
        
        // Act & Assert
        for (x, y) in edgeCases do
            // Test Möbius addition
            let mobiusResult = mobiusAdd x y 1.0
            mobiusResult |> Array.iter (fun r -> r |> should not' (be nan))
            
            // Test hyperbolic distance
            let distance = hyperbolicDistance x y 1.0
            distance |> should not' (be nan)
            distance |> should be (greaterThanOrEqualTo 0.0)

    [<EndToEndTest>]
    let ``CustomTransformers end-to-end workflow should demonstrate all operations`` () =
        // Arrange
        let testVectors = [
            [| 0.1; 0.2; 0.3 |]
            [| 0.4; 0.5; 0.6 |]
            [| 0.7; 0.8; 0.9 |]
        ]
        
        // Act - Test complete workflow
        let workflow = [
            ("CUDA Test", fun () -> testCudaOperations())
            ("Möbius Addition", fun () -> 
                testVectors |> List.pairwise |> List.map (fun (x, y) -> mobiusAdd x y 1.0) |> ignore
                true)
            ("Hyperbolic Distance", fun () ->
                testVectors |> List.pairwise |> List.map (fun (x, y) -> hyperbolicDistance x y 1.0) |> ignore
                true)
            ("Projective Normalize", fun () ->
                projectiveNormalize (testVectors |> Array.ofList) |> ignore
                true)
            ("Hybrid Demo", fun () -> demoHybridEmbeddings(); true)
        ]
        
        let results = 
            workflow
            |> List.map (fun (name, operation) ->
                let measurement = measurePerformance operation
                (name, measurement)
            )
        
        // Assert
        results |> List.iter (fun (name, measurement) ->
            measurement.Success |> should be True
            let result = measurement.Result.Value :?> bool
            result |> should be True
        )
        
        // Calculate overall performance
        let totalTime = results |> List.map (fun (_, m) -> m.ExecutionTime) |> List.fold (+) TimeSpan.Zero
        totalTime |> should be (lessThan (TimeSpan.FromMinutes(2.0)))

    /// Run all CustomTransformers tests and return summary
    let runAllTests () =
        let testMethods = [
            ("MobiusAddition", fun () -> ``Möbius addition should produce valid results`` ())
            ("HyperbolicDistance", fun () -> ``Hyperbolic distance should be non-negative`` ())
            ("ProjectiveNormalization", fun () -> ``Projective normalization should preserve vector count`` ())
            ("DualQuaternionNorm", fun () -> ``Dual quaternion norm should be positive for non-zero quaternions`` ())
            ("HybridEmbedding", fun () -> ``Hybrid embedding creation should include all specified components`` ())
            ("CudaPerformance", fun () -> ``CUDA operations should complete within reasonable time`` ())
            ("SimilarityCalculations", fun () -> ``Similarity calculations should work for all geometric spaces`` ())
            ("MobiusCommutative", fun () -> ``Möbius addition should be commutative`` ())
            ("HyperbolicSymmetric", fun () -> ``Hyperbolic distance should be symmetric`` ())
            ("BatchScaling", fun () -> ``Batch operations should scale efficiently`` ())
            ("ProjectiveUnitVectors", fun () -> ``Projective normalization should produce unit vectors`` ())
            ("HybridDemo", fun () -> ``Hybrid embeddings demo should complete successfully`` ())
            ("EdgeCases", fun () -> ``CUDA operations should handle edge cases gracefully`` ())
            ("EndToEndWorkflow", fun () -> ``CustomTransformers end-to-end workflow should demonstrate all operations`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running CustomTransformers test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
            ("total_memory_mb", measurements |> List.map (_.MemoryUsed) |> List.sum |> float |> fun x -> x / (1024.0 * 1024.0))
            ("success_rate", measurements |> List.filter (_.Success) |> List.length |> float |> fun x -> x / float measurements.Length)
            ("cuda_fallback_rate", 1.0) // All operations fall back to CPU in test environment
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "CustomTransformers" result
        result
