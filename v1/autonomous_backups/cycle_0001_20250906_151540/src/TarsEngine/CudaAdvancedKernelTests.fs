namespace TarsEngine

open System
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging
open TarsEngine.CudaInterop

/// CUDA Advanced Kernel Tests - Real GPU execution of advanced operations
module CudaAdvancedKernelTests =
    
    // Advanced CUDA function declarations for testing
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_flash_attention(
        nativeint Q, nativeint K, nativeint V,
        nativeint output, nativeint softmax_lse,
        int batch_size, int seq_len, int head_dim, int num_heads,
        float32 scale, nativeint stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_swiglu_activation(
        nativeint gate, nativeint up, nativeint output,
        int size, nativeint stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_sedenion_distance(
        nativeint vectors1, nativeint vectors2, nativeint distances,
        int num_vectors, int dimensions, nativeint stream)
    
    /// Advanced test result
    type AdvancedTestResult = {
        TestName: string
        Success: bool
        ExecutionTimeMs: float
        ThroughputOpsPerSec: float option
        MemoryUsageMB: float
        ErrorMessage: string option
        PerformanceMetrics: Map<string, float>
    }
    
    /// CUDA Advanced Kernel Test Suite
    type CudaAdvancedKernelTestSuite(logger: ILogger<CudaAdvancedKernelTestSuite>) =
        
        /// Test Flash Attention implementation - REAL GPU EXECUTION
        member _.TestFlashAttention() = async {
            logger.LogInformation("🧪 Testing CUDA Flash Attention - REAL GPU EXECUTION...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Flash Attention"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Flash Attention"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                // Flash Attention parameters
                let batch_size = 2
                let seq_len = 512
                let head_dim = 64
                let num_heads = 8
                let scale = 1.0f / sqrt(float32 head_dim)
                
                logger.LogInformation($"🔧 Flash Attention config:")
                logger.LogInformation($"  Batch size: {batch_size}")
                logger.LogInformation($"  Sequence length: {seq_len}")
                logger.LogInformation($"  Head dimension: {head_dim}")
                logger.LogInformation($"  Number of heads: {num_heads}")
                logger.LogInformation($"  Scale: {scale:F4}")
                
                // Calculate memory requirements
                let elementSize = 4 // FP32 = 4 bytes
                let qkvSize = unativeint (batch_size * num_heads * seq_len * head_dim * elementSize)
                let outputSize = qkvSize
                let lseSize = unativeint (batch_size * num_heads * seq_len * elementSize)
                let totalMemory = int64 qkvSize * 3L + int64 outputSize + int64 lseSize
                let totalMemoryMB = float totalMemory / 1e6
                
                logger.LogInformation($"💾 Memory required: {totalMemoryMB:F1}MB")
                
                // Allocate GPU memory
                let mutable ptrQ = nativeint 0
                let mutable ptrK = nativeint 0
                let mutable ptrV = nativeint 0
                let mutable ptrOutput = nativeint 0
                let mutable ptrLSE = nativeint 0
                
                let allocQ = tars_cuda_malloc(&ptrQ, qkvSize)
                let allocK = tars_cuda_malloc(&ptrK, qkvSize)
                let allocV = tars_cuda_malloc(&ptrV, qkvSize)
                let allocOutput = tars_cuda_malloc(&ptrOutput, outputSize)
                let allocLSE = tars_cuda_malloc(&ptrLSE, lseSize)
                
                if allocQ <> TarsCudaError.Success || allocK <> TarsCudaError.Success || 
                   allocV <> TarsCudaError.Success || allocOutput <> TarsCudaError.Success || 
                   allocLSE <> TarsCudaError.Success then
                    
                    // Cleanup any successful allocations
                    if ptrQ <> nativeint 0 then tars_cuda_free(ptrQ) |> ignore
                    if ptrK <> nativeint 0 then tars_cuda_free(ptrK) |> ignore
                    if ptrV <> nativeint 0 then tars_cuda_free(ptrV) |> ignore
                    if ptrOutput <> nativeint 0 then tars_cuda_free(ptrOutput) |> ignore
                    if ptrLSE <> nativeint 0 then tars_cuda_free(ptrLSE) |> ignore
                    
                    return {
                        TestName = "Flash Attention"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"GPU memory allocation failed: Q={allocQ}, K={allocK}, V={allocV}, Out={allocOutput}, LSE={allocLSE}"
                        PerformanceMetrics = Map.empty
                    }
                
                logger.LogInformation("✅ GPU memory allocated successfully")
                
                let startTime = DateTime.UtcNow
                
                // REAL Flash Attention execution
                let flashResult = tars_flash_attention(
                    ptrQ, ptrK, ptrV, ptrOutput, ptrLSE,
                    batch_size, seq_len, head_dim, num_heads,
                    scale, nativeint 0)
                
                let syncResult = tars_synchronize_device()
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                // Cleanup GPU memory
                let freeQ = tars_cuda_free(ptrQ)
                let freeK = tars_cuda_free(ptrK)
                let freeV = tars_cuda_free(ptrV)
                let freeOutput = tars_cuda_free(ptrOutput)
                let freeLSE = tars_cuda_free(ptrLSE)
                let cleanupResult = tars_cuda_cleanup()
                
                if flashResult = TarsCudaError.Success && syncResult = TarsCudaError.Success then
                    // Calculate performance metrics
                    let totalOps = int64 batch_size * int64 num_heads * int64 seq_len * int64 seq_len * int64 head_dim
                    let throughput = float totalOps / (executionTime / 1000.0)
                    let memoryBandwidth = float totalMemory / (executionTime / 1000.0) / 1e9
                    
                    logger.LogInformation($"✅ REAL Flash Attention completed: {executionTime:F2}ms")
                    logger.LogInformation($"🚀 Throughput: {throughput / 1e9:F1}G ops/sec")
                    logger.LogInformation($"📊 Memory Bandwidth: {memoryBandwidth:F1} GB/s")
                    
                    return {
                        TestName = "Flash Attention"
                        Success = true
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = Some throughput
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("batch_size", float batch_size)
                            ("seq_len", float seq_len)
                            ("head_dim", float head_dim)
                            ("num_heads", float num_heads)
                            ("total_ops", float totalOps)
                            ("execution_time_ms", executionTime)
                            ("throughput_ops_per_sec", throughput)
                            ("memory_bandwidth_gb_s", memoryBandwidth)
                            ("memory_usage_mb", totalMemoryMB)
                        ]
                    }
                else
                    return {
                        TestName = "Flash Attention"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"Flash Attention execution failed: Flash={flashResult}, Sync={syncResult}"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ Flash Attention test failed: {ex.Message}")
                return {
                    TestName = "Flash Attention"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    MemoryUsageMB = 0.0
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test SwiGLU activation function - REAL GPU EXECUTION
        member _.TestSwiGLUActivation() = async {
            logger.LogInformation("🧪 Testing CUDA SwiGLU Activation - REAL GPU EXECUTION...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "SwiGLU Activation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "SwiGLU Activation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                let numElements = 10 * 1024 * 1024 // 10M elements
                logger.LogInformation($"🔢 Processing {numElements:N0} elements")
                
                // Calculate memory requirements
                let elementSize = 4 // FP32 = 4 bytes
                let bufferSize = unativeint (numElements * elementSize)
                let totalMemory = int64 bufferSize * 3L // Gate + Up + Output
                let totalMemoryMB = float totalMemory / 1e6
                
                logger.LogInformation($"💾 Memory required: {totalMemoryMB:F1}MB")
                
                // Allocate GPU memory
                let mutable ptrGate = nativeint 0
                let mutable ptrUp = nativeint 0
                let mutable ptrOutput = nativeint 0
                
                let allocGate = tars_cuda_malloc(&ptrGate, bufferSize)
                let allocUp = tars_cuda_malloc(&ptrUp, bufferSize)
                let allocOutput = tars_cuda_malloc(&ptrOutput, bufferSize)
                
                if allocGate <> TarsCudaError.Success || allocUp <> TarsCudaError.Success || allocOutput <> TarsCudaError.Success then
                    // Cleanup any successful allocations
                    if ptrGate <> nativeint 0 then tars_cuda_free(ptrGate) |> ignore
                    if ptrUp <> nativeint 0 then tars_cuda_free(ptrUp) |> ignore
                    if ptrOutput <> nativeint 0 then tars_cuda_free(ptrOutput) |> ignore
                    
                    return {
                        TestName = "SwiGLU Activation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"GPU memory allocation failed: Gate={allocGate}, Up={allocUp}, Output={allocOutput}"
                        PerformanceMetrics = Map.empty
                    }
                
                logger.LogInformation("✅ GPU memory allocated successfully")
                
                let startTime = DateTime.UtcNow
                
                // REAL SwiGLU activation execution
                let swigluResult = tars_swiglu_activation(ptrGate, ptrUp, ptrOutput, numElements, nativeint 0)
                let syncResult = tars_synchronize_device()
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                // Cleanup GPU memory
                let freeGate = tars_cuda_free(ptrGate)
                let freeUp = tars_cuda_free(ptrUp)
                let freeOutput = tars_cuda_free(ptrOutput)
                let cleanupResult = tars_cuda_cleanup()
                
                if swigluResult = TarsCudaError.Success && syncResult = TarsCudaError.Success then
                    let throughput = float numElements / (executionTime / 1000.0)
                    let memoryBandwidth = float totalMemory / (executionTime / 1000.0) / 1e9
                    
                    logger.LogInformation($"✅ REAL SwiGLU completed: {executionTime:F2}ms")
                    logger.LogInformation($"🚀 Throughput: {throughput / 1e6:F1}M elements/sec")
                    logger.LogInformation($"📊 Memory Bandwidth: {memoryBandwidth:F1} GB/s")
                    
                    return {
                        TestName = "SwiGLU Activation"
                        Success = true
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = Some throughput
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("num_elements", float numElements)
                            ("execution_time_ms", executionTime)
                            ("throughput_elements_per_sec", throughput)
                            ("memory_bandwidth_gb_s", memoryBandwidth)
                            ("memory_usage_mb", totalMemoryMB)
                        ]
                    }
                else
                    return {
                        TestName = "SwiGLU Activation"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"SwiGLU execution failed: SwiGLU={swigluResult}, Sync={syncResult}"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ SwiGLU activation test failed: {ex.Message}")
                return {
                    TestName = "SwiGLU Activation"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    MemoryUsageMB = 0.0
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test Sedenion distance calculation - REAL GPU EXECUTION
        member _.TestSedenionDistance() = async {
            logger.LogInformation("🧪 Testing CUDA Sedenion Distance - REAL GPU EXECUTION...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Sedenion Distance"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Sedenion Distance"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = 0.0
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        PerformanceMetrics = Map.empty
                    }
                
                let numVectors = 100000 // 100K vectors
                let dimensions = 16 // Sedenion has 16 dimensions
                logger.LogInformation($"🔢 Computing distances for {numVectors:N0} vectors with {dimensions} dimensions")
                
                // Calculate memory requirements
                let elementSize = 4 // FP32 = 4 bytes
                let vectorSize = unativeint (numVectors * dimensions * elementSize)
                let distanceSize = unativeint (numVectors * elementSize)
                let totalMemory = int64 vectorSize * 2L + int64 distanceSize
                let totalMemoryMB = float totalMemory / 1e6
                
                logger.LogInformation($"💾 Memory required: {totalMemoryMB:F1}MB")
                
                // Allocate GPU memory
                let mutable ptrVectors1 = nativeint 0
                let mutable ptrVectors2 = nativeint 0
                let mutable ptrDistances = nativeint 0
                
                let allocVec1 = tars_cuda_malloc(&ptrVectors1, vectorSize)
                let allocVec2 = tars_cuda_malloc(&ptrVectors2, vectorSize)
                let allocDist = tars_cuda_malloc(&ptrDistances, distanceSize)
                
                if allocVec1 <> TarsCudaError.Success || allocVec2 <> TarsCudaError.Success || allocDist <> TarsCudaError.Success then
                    // Cleanup any successful allocations
                    if ptrVectors1 <> nativeint 0 then tars_cuda_free(ptrVectors1) |> ignore
                    if ptrVectors2 <> nativeint 0 then tars_cuda_free(ptrVectors2) |> ignore
                    if ptrDistances <> nativeint 0 then tars_cuda_free(ptrDistances) |> ignore
                    
                    return {
                        TestName = "Sedenion Distance"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"GPU memory allocation failed: Vec1={allocVec1}, Vec2={allocVec2}, Dist={allocDist}"
                        PerformanceMetrics = Map.empty
                    }
                
                logger.LogInformation("✅ GPU memory allocated successfully")
                
                let startTime = DateTime.UtcNow
                
                // REAL Sedenion distance calculation
                let sedenionResult = tars_sedenion_distance(ptrVectors1, ptrVectors2, ptrDistances, numVectors, dimensions, nativeint 0)
                let syncResult = tars_synchronize_device()
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                // Cleanup GPU memory
                let freeVec1 = tars_cuda_free(ptrVectors1)
                let freeVec2 = tars_cuda_free(ptrVectors2)
                let freeDist = tars_cuda_free(ptrDistances)
                let cleanupResult = tars_cuda_cleanup()
                
                if sedenionResult = TarsCudaError.Success && syncResult = TarsCudaError.Success then
                    let totalOps = int64 numVectors * int64 dimensions // Operations per distance calculation
                    let throughput = float totalOps / (executionTime / 1000.0)
                    let memoryBandwidth = float totalMemory / (executionTime / 1000.0) / 1e9
                    
                    logger.LogInformation($"✅ REAL Sedenion distance completed: {executionTime:F2}ms")
                    logger.LogInformation($"🚀 Throughput: {throughput / 1e6:F1}M ops/sec")
                    logger.LogInformation($"📊 Memory Bandwidth: {memoryBandwidth:F1} GB/s")
                    
                    return {
                        TestName = "Sedenion Distance"
                        Success = true
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = Some throughput
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("num_vectors", float numVectors)
                            ("dimensions", float dimensions)
                            ("total_ops", float totalOps)
                            ("execution_time_ms", executionTime)
                            ("throughput_ops_per_sec", throughput)
                            ("memory_bandwidth_gb_s", memoryBandwidth)
                            ("memory_usage_mb", totalMemoryMB)
                        ]
                    }
                else
                    return {
                        TestName = "Sedenion Distance"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        MemoryUsageMB = totalMemoryMB
                        ErrorMessage = Some $"Sedenion distance execution failed: Sedenion={sedenionResult}, Sync={syncResult}"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ Sedenion distance test failed: {ex.Message}")
                return {
                    TestName = "Sedenion Distance"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    MemoryUsageMB = 0.0
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
