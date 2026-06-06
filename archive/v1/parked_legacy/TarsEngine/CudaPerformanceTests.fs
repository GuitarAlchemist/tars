namespace TarsEngine

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.CudaInterop

/// CUDA Performance Benchmark Tests - Real GPU performance measurement
module CudaPerformanceTests =
    
    /// Performance test result
    type PerformanceTestResult = {
        TestName: string
        Success: bool
        ExecutionTimeMs: float
        ThroughputOpsPerSec: float
        GFlops: float option
        MemoryBandwidthGBPerSec: float option
        ErrorMessage: string option
        PerformanceMetrics: Map<string, float>
    }
    
    /// CUDA Performance Test Suite
    type CudaPerformanceTestSuite(logger: ILogger<CudaPerformanceTestSuite>) =
        
        /// Benchmark matrix multiplication performance across different sizes
        member _.BenchmarkMatrixMultiplication() = async {
            logger.LogInformation("🧪 Benchmarking CUDA Matrix Multiplication Performance - REAL GPU EXECUTION...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Matrix Multiplication Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Matrix Multiplication Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Test different matrix sizes (must be multiples of 16 for Tensor Cores)
                let testSizes = [
                    ("Small", 512, 512, 512)
                    ("Medium", 1024, 1024, 1024)
                    ("Large", 2048, 2048, 2048)
                    ("XLarge", 4096, 4096, 4096)
                ]
                
                let mutable allResults = []
                let mutable totalGFlops = 0.0
                let mutable totalBandwidth = 0.0
                let mutable successfulTests = 0
                
                for (sizeName, M, N, K) in testSizes do
                    logger.LogInformation($"🔧 Benchmarking {sizeName} matrices: {M}x{N}x{K}")
                    
                    // Calculate memory requirements
                    let elementSize = 2 // FP16 = 2 bytes
                    let sizeA = unativeint (M * K * elementSize)
                    let sizeB = unativeint (K * N * elementSize)
                    let sizeC = unativeint (M * N * elementSize)
                    let totalMemory = int64 sizeA + int64 sizeB + int64 sizeC
                    
                    logger.LogInformation($"💾 Memory required: {totalMemory / 1024L / 1024L}MB")
                    
                    // Allocate GPU memory
                    let mutable ptrA = nativeint 0
                    let mutable ptrB = nativeint 0
                    let mutable ptrC = nativeint 0
                    
                    let allocA = tars_cuda_malloc(&ptrA, sizeA)
                    let allocB = tars_cuda_malloc(&ptrB, sizeB)
                    let allocC = tars_cuda_malloc(&ptrC, sizeC)
                    
                    if allocA = TarsCudaError.Success && allocB = TarsCudaError.Success && allocC = TarsCudaError.Success then
                        // Warm-up run
                        let warmupResult = tars_gemm_tensor_core(ptrA, ptrB, ptrC, M, N, K, 1.0f, 0.0f)
                        if warmupResult = TarsCudaError.Success then
                            let syncResult = tars_synchronize_device()
                            if syncResult = TarsCudaError.Success then
                                logger.LogInformation("🔥 Warm-up completed")
                        
                        // Benchmark multiple runs for accuracy
                        let numRuns = 5
                        let mutable runTimes = []
                        
                        for run in 1 .. numRuns do
                            let startTime = DateTime.UtcNow
                            let gemmResult = tars_gemm_tensor_core(ptrA, ptrB, ptrC, M, N, K, 1.0f, 0.0f)
                            let syncResult = tars_synchronize_device()
                            let endTime = DateTime.UtcNow
                            
                            if gemmResult = TarsCudaError.Success && syncResult = TarsCudaError.Success then
                                let runTime = (endTime - startTime).TotalMilliseconds
                                runTimes <- runTime :: runTimes
                                logger.LogInformation($"  Run {run}: {runTime:F2}ms")
                            else
                                logger.LogError($"❌ Run {run} failed: GEMM={gemmResult}, Sync={syncResult}")
                        
                        // Calculate performance metrics
                        if runTimes.Length > 0 then
                            let avgTime = runTimes |> List.average
                            let minTime = runTimes |> List.min
                            let maxTime = runTimes |> List.max
                            let stdDev = 
                                let variance = runTimes |> List.map (fun t -> (t - avgTime) ** 2.0) |> List.average
                                sqrt variance
                            
                            let totalOps = 2L * int64 M * int64 N * int64 K // 2 ops per multiply-add
                            let gflops = float totalOps / (avgTime / 1000.0) / 1e9
                            let memoryBandwidth = float totalMemory / (avgTime / 1000.0) / 1e9
                            
                            totalGFlops <- totalGFlops + gflops
                            totalBandwidth <- totalBandwidth + memoryBandwidth
                            successfulTests <- successfulTests + 1
                            
                            logger.LogInformation($"✅ {sizeName} Results:")
                            logger.LogInformation($"  Average: {avgTime:F2}ms ± {stdDev:F2}ms")
                            logger.LogInformation($"  Performance: {gflops:F1} GFLOPS")
                            logger.LogInformation($"  Bandwidth: {memoryBandwidth:F1} GB/s")
                            
                            allResults <- (sizeName, avgTime, gflops, memoryBandwidth) :: allResults
                        else
                            logger.LogError($"❌ No successful runs for {sizeName}")
                    else
                        logger.LogError($"❌ Memory allocation failed for {sizeName}: A={allocA}, B={allocB}, C={allocC}")
                    
                    // Cleanup memory
                    if ptrA <> nativeint 0 then tars_cuda_free(ptrA) |> ignore
                    if ptrB <> nativeint 0 then tars_cuda_free(ptrB) |> ignore
                    if ptrC <> nativeint 0 then tars_cuda_free(ptrC) |> ignore
                
                // Cleanup CUDA
                let cleanupResult = tars_cuda_cleanup()
                if cleanupResult <> TarsCudaError.Success then
                    logger.LogWarning($"⚠️ CUDA cleanup warning: {cleanupResult}")
                
                if successfulTests > 0 then
                    let avgGFlops = totalGFlops / float successfulTests
                    let avgBandwidth = totalBandwidth / float successfulTests
                    let totalTime = allResults |> List.sumBy (fun (_, time, _, _) -> time)
                    
                    logger.LogInformation($"📊 Benchmark Summary:")
                    logger.LogInformation($"  Tests completed: {successfulTests}/{testSizes.Length}")
                    logger.LogInformation($"  Average performance: {avgGFlops:F1} GFLOPS")
                    logger.LogInformation($"  Average bandwidth: {avgBandwidth:F1} GB/s")
                    
                    return {
                        TestName = "Matrix Multiplication Benchmark"
                        Success = true
                        ExecutionTimeMs = totalTime
                        ThroughputOpsPerSec = avgGFlops * 1e9
                        GFlops = Some avgGFlops
                        MemoryBandwidthGBPerSec = Some avgBandwidth
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("avg_gflops", avgGFlops)
                            ("avg_bandwidth_gb_s", avgBandwidth)
                            ("successful_tests", float successfulTests)
                            ("total_tests", float testSizes.Length)
                            ("total_time_ms", totalTime)
                        ]
                    }
                else
                    return {
                        TestName = "Matrix Multiplication Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some "No successful benchmark runs"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ Matrix multiplication benchmark failed: {ex.Message}")
                return {
                    TestName = "Matrix Multiplication Benchmark"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = 0.0
                    GFlops = None
                    MemoryBandwidthGBPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Benchmark GELU activation performance across different sizes
        member _.BenchmarkGeluActivation() = async {
            logger.LogInformation("🧪 Benchmarking CUDA GELU Activation Performance - REAL GPU EXECUTION...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "GELU Activation Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "GELU Activation Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Test different vector sizes
                let testSizes = [
                    ("Small", 1024 * 1024)      // 1M elements
                    ("Medium", 10 * 1024 * 1024) // 10M elements
                    ("Large", 100 * 1024 * 1024) // 100M elements
                    ("XLarge", 500 * 1024 * 1024) // 500M elements
                ]
                
                let mutable allResults = []
                let mutable totalThroughput = 0.0
                let mutable totalBandwidth = 0.0
                let mutable successfulTests = 0
                
                for (sizeName, numElements) in testSizes do
                    logger.LogInformation($"🔧 Benchmarking {sizeName} GELU: {numElements:N0} elements")
                    
                    // Calculate memory requirements
                    let elementSize = 4 // FP32 = 4 bytes
                    let bufferSize = unativeint (numElements * elementSize)
                    let totalMemory = int64 bufferSize * 2L // Input + Output
                    
                    logger.LogInformation($"💾 Memory required: {totalMemory / 1024L / 1024L}MB")
                    
                    // Allocate GPU memory
                    let mutable inputPtr = nativeint 0
                    let mutable outputPtr = nativeint 0
                    
                    let allocInput = tars_cuda_malloc(&inputPtr, bufferSize)
                    let allocOutput = tars_cuda_malloc(&outputPtr, bufferSize)
                    
                    if allocInput = TarsCudaError.Success && allocOutput = TarsCudaError.Success then
                        // Warm-up run
                        let warmupResult = tars_gelu_forward(inputPtr, outputPtr, numElements, nativeint 0)
                        if warmupResult = TarsCudaError.Success then
                            let syncResult = tars_synchronize_device()
                            if syncResult = TarsCudaError.Success then
                                logger.LogInformation("🔥 Warm-up completed")
                        
                        // Benchmark multiple runs
                        let numRuns = 10
                        let mutable runTimes = []
                        
                        for run in 1 .. numRuns do
                            let startTime = DateTime.UtcNow
                            let geluResult = tars_gelu_forward(inputPtr, outputPtr, numElements, nativeint 0)
                            let syncResult = tars_synchronize_device()
                            let endTime = DateTime.UtcNow
                            
                            if geluResult = TarsCudaError.Success && syncResult = TarsCudaError.Success then
                                let runTime = (endTime - startTime).TotalMilliseconds
                                runTimes <- runTime :: runTimes
                            else
                                logger.LogError($"❌ Run {run} failed: GELU={geluResult}, Sync={syncResult}")
                        
                        // Calculate performance metrics
                        if runTimes.Length > 0 then
                            let avgTime = runTimes |> List.average
                            let minTime = runTimes |> List.min
                            let maxTime = runTimes |> List.max
                            
                            let throughput = float numElements / (avgTime / 1000.0)
                            let memoryBandwidth = float totalMemory / (avgTime / 1000.0) / 1e9
                            
                            totalThroughput <- totalThroughput + throughput
                            totalBandwidth <- totalBandwidth + memoryBandwidth
                            successfulTests <- successfulTests + 1
                            
                            logger.LogInformation($"✅ {sizeName} Results:")
                            logger.LogInformation($"  Average: {avgTime:F2}ms (range: {minTime:F2}-{maxTime:F2}ms)")
                            logger.LogInformation($"  Throughput: {throughput / 1e6:F1}M elements/sec")
                            logger.LogInformation($"  Bandwidth: {memoryBandwidth:F1} GB/s")
                            
                            allResults <- (sizeName, avgTime, throughput, memoryBandwidth) :: allResults
                        else
                            logger.LogError($"❌ No successful runs for {sizeName}")
                    else
                        logger.LogError($"❌ Memory allocation failed for {sizeName}: Input={allocInput}, Output={allocOutput}")
                    
                    // Cleanup memory
                    if inputPtr <> nativeint 0 then tars_cuda_free(inputPtr) |> ignore
                    if outputPtr <> nativeint 0 then tars_cuda_free(outputPtr) |> ignore
                
                // Cleanup CUDA
                let cleanupResult = tars_cuda_cleanup()
                if cleanupResult <> TarsCudaError.Success then
                    logger.LogWarning($"⚠️ CUDA cleanup warning: {cleanupResult}")
                
                if successfulTests > 0 then
                    let avgThroughput = totalThroughput / float successfulTests
                    let avgBandwidth = totalBandwidth / float successfulTests
                    let totalTime = allResults |> List.sumBy (fun (_, time, _, _) -> time)
                    
                    logger.LogInformation($"📊 GELU Benchmark Summary:")
                    logger.LogInformation($"  Tests completed: {successfulTests}/{testSizes.Length}")
                    logger.LogInformation($"  Average throughput: {avgThroughput / 1e6:F1}M elements/sec")
                    logger.LogInformation($"  Average bandwidth: {avgBandwidth:F1} GB/s")
                    
                    return {
                        TestName = "GELU Activation Benchmark"
                        Success = true
                        ExecutionTimeMs = totalTime
                        ThroughputOpsPerSec = avgThroughput
                        GFlops = None
                        MemoryBandwidthGBPerSec = Some avgBandwidth
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("avg_throughput_elements_per_sec", avgThroughput)
                            ("avg_bandwidth_gb_s", avgBandwidth)
                            ("successful_tests", float successfulTests)
                            ("total_tests", float testSizes.Length)
                            ("total_time_ms", totalTime)
                        ]
                    }
                else
                    return {
                        TestName = "GELU Activation Benchmark"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = 0.0
                        GFlops = None
                        MemoryBandwidthGBPerSec = None
                        ErrorMessage = Some "No successful benchmark runs"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ GELU activation benchmark failed: {ex.Message}")
                return {
                    TestName = "GELU Activation Benchmark"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = 0.0
                    GFlops = None
                    MemoryBandwidthGBPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
