module TarsEngine.FSharp.Tests.Unified.UnifiedIntegrationTests

open System
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager
open TarsEngine.FSharp.Cli.Integration.UnifiedProofSystem
open TarsEngine.FSharp.Cli.Acceleration.UnifiedCudaEngine

/// Integration tests for all unified systems working together
[<TestClass>]
type UnifiedIntegrationTests() =
    
    let createTestLogger() = createLogger "UnifiedIntegrationTests"
    
    [<Fact>]
    let ``All unified systems should initialize and work together`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            // Act - Initialize all systems
            let! configResult = configManager.InitializeAsync(CancellationToken.None)
            let! cudaResult = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Assert - All systems should initialize successfully
            match configResult, cudaResult with
            | Success _, Success _ -> () // All good
            | Failure (error, _), _ -> failwith $"Config initialization failed: {TarsError.toString error}"
            | _, Failure (error, _) -> failwith $"CUDA initialization failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Configuration should control CUDA engine behavior`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Configure CUDA settings
            let! _ = configManager.SetValueAsync("tars.cuda.enabled", true, None)
            let! _ = configManager.SetValueAsync("tars.cuda.deviceId", 0, None)
            
            // Verify configuration
            let cudaEnabled = ConfigurationExtensions.getBool configManager "tars.cuda.enabled" false
            let deviceId = ConfigurationExtensions.getInt configManager "tars.cuda.deviceId" -1
            
            // Assert
            cudaEnabled |> should be True
            deviceId |> should equal 0
        }
    
    [<Fact>]
    let ``Proof system should generate proofs for CUDA operations`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            let correlationId = generateCorrelationId()
            
            // Act - Execute CUDA operation and generate proof
            let operation = CudaOperationFactory.createVectorSimilarity 512
            let! cudaResult = cudaEngine.ExecuteOperationAsync(operation, box "test_data", CancellationToken.None)
            
            match cudaResult with
            | Success (result, _) ->
                // Generate proof for the CUDA operation
                let! proofResult = ProofExtensions.generatePerformanceProof proofGenerator "CudaVectorSimilarity" result.ThroughputGFlops correlationId
                
                // Assert
                match proofResult with
                | Success (proof, _) ->
                    proof.CorrelationId |> should equal correlationId
                    match proof.ProofType with
                    | PerformanceProof (benchmark, gflops) ->
                        benchmark |> should equal "CudaVectorSimilarity"
                        gflops |> should equal result.ThroughputGFlops
                    | _ -> failwith "Expected PerformanceProof"
                | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
            
            | Failure (error, _) -> failwith $"CUDA operation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Configuration changes should be tracked with proofs`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let correlationId = generateCorrelationId()
            
            // Act - Change configuration and generate proof
            let oldValue = ConfigurationExtensions.getString configManager "test.tracked.value" "default"
            let! _ = configManager.SetValueAsync("test.tracked.value", "new_value", Some correlationId)
            let newValue = ConfigurationExtensions.getString configManager "test.tracked.value" "default"
            
            // Generate proof for configuration change
            let! proofResult = ProofExtensions.generateStateChangeProof proofGenerator oldValue newValue correlationId
            
            // Assert
            match proofResult with
            | Success (proof, _) ->
                proof.CorrelationId |> should equal correlationId
                match proof.ProofType with
                | StateChangeProof (before, after) ->
                    before |> should equal oldValue
                    after |> should equal newValue
                | _ -> failwith "Expected StateChangeProof"
            | Failure (error, _) -> failwith $"Proof generation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``System should handle end-to-end workflow with all components`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            let correlationId = generateCorrelationId()
            
            // Act - Complete workflow
            // 1. Configure system
            let! _ = configManager.SetValueAsync("workflow.test.enabled", true, Some correlationId)
            
            // 2. Generate proof for configuration
            let! configProof = ProofExtensions.generateExecutionProof proofGenerator "ConfigurationUpdate" correlationId
            
            // 3. Execute CUDA operation
            let operation = CudaOperationFactory.createMatrixMultiplication 256 256 256
            let! cudaResult = cudaEngine.ExecuteOperationAsync(operation, box "workflow_data", CancellationToken.None)
            
            // 4. Generate proof for CUDA operation
            let! cudaProof = match cudaResult with
                              | Success (result, _) -> 
                                  ProofExtensions.generatePerformanceProof proofGenerator "WorkflowCudaOperation" result.ThroughputGFlops correlationId
                              | Failure (error, _) -> 
                                  failwith $"CUDA operation failed: {TarsError.toString error}"
            
            // 5. Create proof chain for entire workflow
            let! chainResult = match configProof, cudaProof with
                               | Success (proof1, _), Success (proof2, _) ->
                                   proofGenerator.CreateProofChainAsync("workflow_chain", [proof1; proof2], CancellationToken.None)
                               | _ -> failwith "Failed to generate workflow proofs"
            
            // 6. Create configuration snapshot
            let! snapshotResult = configManager.CreateSnapshotAsync("workflow_snapshot", CancellationToken.None)
            
            // Assert - All steps should complete successfully
            match chainResult, snapshotResult with
            | Success (chain, _), Success (snapshot, _) ->
                chain.Proofs.Length |> should equal 2
                chain.IsValid |> should be True
                snapshot.Configuration.Count |> should be (greaterThan 0)
            | Failure (error, _), _ -> failwith $"Proof chain creation failed: {TarsError.toString error}"
            | _, Failure (error, _) -> failwith $"Snapshot creation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``System should maintain consistency across concurrent operations`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Perform concurrent operations across all systems
            let tasks = [
                // Configuration operations
                for i in 1..5 do
                    yield configManager.SetValueAsync($"concurrent.config{i}", $"value{i}", None)
                
                // CUDA operations
                for i in 1..3 do
                    let operation = CudaOperationFactory.createVectorSimilarity (256 * i)
                    yield cudaEngine.ExecuteOperationAsync(operation, box $"data{i}", CancellationToken.None)
                
                // Proof operations
                for i in 1..3 do
                    yield ProofExtensions.generateExecutionProof proofGenerator $"ConcurrentOperation{i}" (generateCorrelationId())
            ]
            
            let! results = System.Threading.Tasks.Task.WhenAll(tasks)
            
            // Assert - All operations should complete successfully
            results |> Array.forall (function | Success _ -> true | Failure _ -> false) |> should be True
            
            // Verify system state consistency
            let configStats = configManager.GetStatistics()
            let cudaMetrics = cudaEngine.GetPerformanceMetrics()
            let proofStats = proofGenerator.GetProofStatistics()
            
            configStats.["isInitialized"] :?> bool |> should be True
            cudaMetrics.TotalOperations |> should be (greaterThanOrEqualTo 3L)
            proofStats.["totalProofs"] :?> int |> should be (greaterThanOrEqualTo 3)
        }
    
    [<Fact>]
    let ``System should handle errors gracefully across all components`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Attempt operations that might fail
            let! invalidConfigResult = configManager.SetValueAsync("tars.core.logLevel", "InvalidLevel", None)
            
            // Create operation with invalid parameters (this should still work but might be slower)
            let operation = CudaOperationFactory.createVectorSimilarity 0 // Edge case
            let! cudaResult = cudaEngine.ExecuteOperationAsync(operation, box "test", CancellationToken.None)
            
            // Generate proof for error scenario
            let! errorProofResult = ProofExtensions.generateExecutionProof proofGenerator "ErrorHandlingTest" (generateCorrelationId())
            
            // Assert - System should handle errors gracefully
            match invalidConfigResult with
            | Success _ -> failwith "Invalid config should have failed"
            | Failure (ValidationError _, _) -> () // Expected
            | Failure (error, _) -> failwith $"Unexpected error type: {TarsError.toString error}"
            
            // CUDA operation should either succeed or fail gracefully
            match cudaResult with
            | Success (result, _) -> result.Success |> should be True
            | Failure _ -> () // Acceptable for edge case
            
            // Proof generation should always work
            match errorProofResult with
            | Success _ -> () // Expected
            | Failure (error, _) -> failwith $"Proof generation should not fail: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``System should provide comprehensive monitoring across all components`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use proofGenerator = createProofGenerator logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Perform some operations to generate metrics
            let! _ = configManager.SetValueAsync("monitoring.test", "value", None)
            let operation = CudaOperationFactory.createVectorSimilarity 512
            let! _ = cudaEngine.ExecuteOperationAsync(operation, box "data", CancellationToken.None)
            let! _ = ProofExtensions.generateExecutionProof proofGenerator "MonitoringTest" (generateCorrelationId())
            
            // Act - Collect metrics from all systems
            let configStats = configManager.GetStatistics()
            let cudaMetrics = cudaEngine.GetPerformanceMetrics()
            let proofStats = proofGenerator.GetProofStatistics()
            
            // Assert - All systems should provide meaningful metrics
            configStats.Count |> should be (greaterThan 5)
            configStats.ContainsKey("totalConfigurations") |> should be True
            configStats.ContainsKey("isInitialized") |> should be True
            
            cudaMetrics.TotalOperations |> should be (greaterThan 0L)
            cudaMetrics.LastUpdate |> should be (lessThanOrEqualTo DateTime.UtcNow)
            
            proofStats.Count |> should be (greaterThan 3)
            proofStats.ContainsKey("totalProofs") |> should be True
            proofStats.ContainsKey("proofsByType") |> should be True
        }
    
    [<Fact>]
    let ``System should support configuration-driven behavior`` () =
        task {
            // Arrange
            let logger = createTestLogger()
            use configManager = createConfigurationManager logger
            use cudaEngine = createCudaEngine logger
            
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Configure system behavior
            let! _ = configManager.SetValueAsync("tars.cuda.enabled", false, None) // Disable CUDA
            let! _ = configManager.SetValueAsync("tars.core.maxConcurrency", 5, None)
            
            // Verify configuration affects behavior
            let cudaEnabled = ConfigurationExtensions.getBool configManager "tars.cuda.enabled" true
            let maxConcurrency = ConfigurationExtensions.getInt configManager "tars.core.maxConcurrency" 10
            
            // Assert
            cudaEnabled |> should be False
            maxConcurrency |> should equal 5
            
            // CUDA operations should still work (CPU fallback)
            let operation = CudaOperationFactory.createVectorSimilarity 256
            let! cudaResult = cudaEngine.ExecuteOperationAsync(operation, box "data", CancellationToken.None)
            
            match cudaResult with
            | Success (result, metadata) ->
                result.Success |> should be True
                // Should indicate CPU fallback
                metadata.ContainsKey("fallback") |> should be True
            | Failure (error, _) -> failwith $"CUDA operation should work in fallback mode: {TarsError.toString error}"
        }
