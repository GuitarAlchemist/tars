module TarsEngine.FSharp.Tests.Unified.UnifiedCudaEngineTests

open System
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Acceleration.UnifiedCudaEngine

/// Tests for the Unified CUDA Engine
[<TestClass>]
type UnifiedCudaEngineTests() =
    
    let createTestLogger() = createLogger "UnifiedCudaEngineTests"
    
    [<Fact>]
    let ``CUDA engine should initialize successfully`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            
            // Act
            let! result = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Assert
            match result with
            | Success (_, metadata) ->
                metadata.ContainsKey("deviceCount") |> should be True
                // Should work in either GPU or CPU fallback mode
                let deviceCount = metadata.["deviceCount"] :?> int
                deviceCount |> should be (greaterThanOrEqualTo 0)
            | Failure (error, _) -> failwith $"CUDA initialization failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should execute vector similarity operation`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            let operation = CudaOperationFactory.createVectorSimilarity 512
            let testData = box "vector_data_512"
            
            // Act
            let! result = cudaEngine.ExecuteOperationAsync(operation, testData, CancellationToken.None)
            
            // Assert
            match result with
            | Success (cudaResult, metadata) ->
                cudaResult.Success |> should be True
                cudaResult.OperationId |> should equal operation.OperationId
                cudaResult.ExecutionTime.TotalMilliseconds |> should be (greaterThan 0.0)
                cudaResult.MemoryUsed |> should be (greaterThan 0L)
                cudaResult.ThroughputGFlops |> should be (greaterThanOrEqualTo 0.0)
                metadata.ContainsKey("executionTime") |> should be True
            | Failure (error, _) -> failwith $"CUDA operation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should execute matrix multiplication operation`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            let operation = CudaOperationFactory.createMatrixMultiplication 256 256 256
            let testData = box "matrix_data_256"
            
            // Act
            let! result = cudaEngine.ExecuteOperationAsync(operation, testData, CancellationToken.None)
            
            // Assert
            match result with
            | Success (cudaResult, metadata) ->
                cudaResult.Success |> should be True
                cudaResult.OperationId |> should equal operation.OperationId
                cudaResult.ThroughputGFlops |> should be (greaterThan 0.0)
                
                // Matrix multiplication should use significant memory
                cudaResult.MemoryUsed |> should be (greaterThan 1000000L) // > 1MB
            | Failure (error, _) -> failwith $"CUDA operation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should execute reasoning kernel operation`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            let operation = CudaOperationFactory.createReasoningKernel 500
            let testData = box "reasoning_data_medium"
            
            // Act
            let! result = cudaEngine.ExecuteOperationAsync(operation, testData, CancellationToken.None)
            
            // Assert
            match result with
            | Success (cudaResult, metadata) ->
                cudaResult.Success |> should be True
                cudaResult.OperationId |> should equal operation.OperationId
                
                // Reasoning operations should take some time
                cudaResult.ExecutionTime.TotalMilliseconds |> should be (greaterThan 100.0)
            | Failure (error, _) -> failwith $"CUDA operation failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should provide performance metrics`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Execute some operations to generate metrics
            let operation1 = CudaOperationFactory.createVectorSimilarity 256
            let operation2 = CudaOperationFactory.createMatrixMultiplication 128 128 128
            
            let! _ = cudaEngine.ExecuteOperationAsync(operation1, box "data1", CancellationToken.None)
            let! _ = cudaEngine.ExecuteOperationAsync(operation2, box "data2", CancellationToken.None)
            
            // Act
            let metrics = cudaEngine.GetPerformanceMetrics()
            
            // Assert
            metrics.TotalOperations |> should be (greaterThanOrEqualTo 2L)
            metrics.SuccessfulOperations |> should be (greaterThanOrEqualTo 2L)
            metrics.FailedOperations |> should be (greaterThanOrEqualTo 0L)
            metrics.TotalGpuTime.TotalMilliseconds |> should be (greaterThan 0.0)
            metrics.LastUpdate |> should be (lessThanOrEqualTo DateTime.UtcNow)
        }
    
    [<Fact>]
    let ``Should handle concurrent operations`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Execute multiple operations concurrently
            let operations = [
                for i in 1..5 do
                    let operation = CudaOperationFactory.createVectorSimilarity (256 * i)
                    yield cudaEngine.ExecuteOperationAsync(operation, box $"data{i}", CancellationToken.None)
            ]
            
            let! results = System.Threading.Tasks.Task.WhenAll(operations)
            
            // Assert
            results |> Array.forall (function | Success _ -> true | Failure _ -> false) |> should be True
            
            // All operations should have unique IDs
            let operationIds = 
                results 
                |> Array.choose (function | Success (result, _) -> Some result.OperationId | Failure _ -> None)
            
            let uniqueIds = operationIds |> Array.distinct
            uniqueIds.Length |> should equal operationIds.Length
        }
    
    [<Fact>]
    let ``Should track active operations`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Check active operations before and after execution
            let activeOpsBefore = cudaEngine.GetActiveOperations()
            
            let operation = CudaOperationFactory.createVectorSimilarity 512
            let! _ = cudaEngine.ExecuteOperationAsync(operation, box "data", CancellationToken.None)
            
            let activeOpsAfter = cudaEngine.GetActiveOperations()
            
            // Assert
            activeOpsBefore.Length |> should equal 0
            activeOpsAfter.Length |> should equal 0 // Should be empty after completion
        }
    
    [<Fact>]
    let ``Should detect GPU availability`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act
            let isGpuAvailable = cudaEngine.IsGpuAvailable()
            let availableDevices = cudaEngine.GetAvailableDevices()
            
            // Assert
            // Should work in either GPU or CPU fallback mode
            if isGpuAvailable then
                availableDevices.Length |> should be (greaterThan 0)
            else
                // CPU fallback mode
                availableDevices.Length |> should equal 0
        }
    
    [<Fact>]
    let ``Should provide device information when GPU available`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act
            let devices = cudaEngine.GetAvailableDevices()
            
            // Assert
            if devices.Length > 0 then
                let device = devices.[0]
                device.DeviceId |> should be (greaterThanOrEqualTo 0)
                device.Name.Length |> should be (greaterThan 0)
                device.TotalMemory |> should be (greaterThan 0L)
                device.ComputeCapability |> should be (greaterThan 0.0)
                device.IsAvailable |> should be True
        }
    
    [<Fact>]
    let ``Should handle operation timeouts gracefully`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            let operation = CudaOperationFactory.createReasoningKernel 1000 // High complexity
            
            // Act with short timeout
            use cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(1))
            let! result = cudaEngine.ExecuteOperationAsync(operation, box "data", cts.Token)
            
            // Assert - Should either complete quickly or handle cancellation gracefully
            match result with
            | Success (cudaResult, _) ->
                // Operation completed within timeout
                cudaResult.Success |> should be True
            | Failure (error, _) ->
                // Operation was cancelled or timed out - this is acceptable
                TarsError.toString error |> should contain "cancel" |> should be False // Don't require specific error message
        }
    
    [<Fact>]
    let ``Should cleanup resources properly`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Execute some operations
            let operation = CudaOperationFactory.createVectorSimilarity 256
            let! _ = cudaEngine.ExecuteOperationAsync(operation, box "data", CancellationToken.None)
            
            // Act
            let! cleanupResult = cudaEngine.CleanupAsync(CancellationToken.None)
            
            // Assert
            match cleanupResult with
            | Success _ -> () // Cleanup succeeded
            | Failure (error, _) -> failwith $"Cleanup failed: {TarsError.toString error}"
        }
    
    [<Fact>]
    let ``Should handle different operation types`` () =
        task {
            // Arrange
            use cudaEngine = createCudaEngine (createTestLogger())
            let! _ = cudaEngine.InitializeAsync(CancellationToken.None)
            
            // Act - Test different operation types
            let vectorOp = CudaOperationFactory.createVectorSimilarity 512
            let matrixOp = CudaOperationFactory.createMatrixMultiplication 256 256 256
            let reasoningOp = CudaOperationFactory.createReasoningKernel 300
            
            let! vectorResult = cudaEngine.ExecuteOperationAsync(vectorOp, box "vector_data", CancellationToken.None)
            let! matrixResult = cudaEngine.ExecuteOperationAsync(matrixOp, box "matrix_data", CancellationToken.None)
            let! reasoningResult = cudaEngine.ExecuteOperationAsync(reasoningOp, box "reasoning_data", CancellationToken.None)
            
            // Assert
            match vectorResult, matrixResult, reasoningResult with
            | Success (vResult, _), Success (mResult, _), Success (rResult, _) ->
                vResult.Success |> should be True
                mResult.Success |> should be True
                rResult.Success |> should be True
                
                // Different operations should have different characteristics
                vResult.OperationId |> should not' (equal mResult.OperationId)
                mResult.OperationId |> should not' (equal rResult.OperationId)
                
            | _ -> failwith "Not all operations completed successfully"
        }
    
    [<Fact>]
    let ``Should provide operation estimates`` () =
        // Arrange
        let vectorOp = CudaOperationFactory.createVectorSimilarity 1024
        let matrixOp = CudaOperationFactory.createMatrixMultiplication 512 512 512
        let reasoningOp = CudaOperationFactory.createReasoningKernel 1000
        
        // Assert - Operations should have reasonable estimates
        vectorOp.EstimatedTime.TotalMilliseconds |> should be (greaterThan 0.0)
        vectorOp.MemoryRequired |> should be (greaterThan 0L)
        
        matrixOp.EstimatedTime.TotalMilliseconds |> should be (greaterThan vectorOp.EstimatedTime.TotalMilliseconds)
        matrixOp.MemoryRequired |> should be (greaterThan vectorOp.MemoryRequired)
        
        reasoningOp.EstimatedTime.TotalMilliseconds |> should be (greaterThan 0.0)
        reasoningOp.MemoryRequired |> should be (greaterThan 0L)
    
    [<Fact>]
    let ``Should handle operation priorities`` () =
        // Arrange
        let lowPriorityOp = CudaOperationFactory.createVectorSimilarity 256
        let highPriorityOp = CudaOperationFactory.createReasoningKernel 500
        
        // Assert - Different operations should have different priorities
        lowPriorityOp.Priority |> should be (greaterThan 0)
        highPriorityOp.Priority |> should be (greaterThan 0)
        
        // Reasoning operations typically have higher priority
        highPriorityOp.Priority |> should be (greaterThanOrEqualTo lowPriorityOp.Priority)
    
    [<Fact>]
    let ``Should track operation creation time`` () =
        // Arrange
        let startTime = DateTime.UtcNow
        
        // Act
        let operation = CudaOperationFactory.createVectorSimilarity 512
        
        // Assert
        operation.CreatedAt |> should be (greaterThanOrEqualTo startTime)
        operation.CreatedAt |> should be (lessThanOrEqualTo DateTime.UtcNow)
