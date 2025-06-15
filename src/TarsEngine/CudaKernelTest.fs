namespace TarsEngine

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.CudaInterop

/// CUDA Kernel Test Program - Real testing of compiled CUDA kernels
module CudaKernelTest =
    
    /// Test result for CUDA operations
    type CudaTestResult = {
        TestName: string
        Success: bool
        ExecutionTimeMs: float
        ThroughputOpsPerSec: float option
        ErrorMessage: string option
        PerformanceMetrics: Map<string, float>
    }
    
    /// CUDA Kernel Test Suite
    type CudaKernelTestSuite(logger: ILogger<CudaKernelTestSuite>) =
        
        /// Test CUDA device detection and initialization
        member _.TestDeviceInitialization() = async {
            logger.LogInformation("🧪 Testing CUDA device detection and initialization...")
            
            try
                let! devices = getAvailableCudaDevices()
                
                if devices.Length = 0 then
                    return {
                        TestName = "Device Detection"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "No CUDA devices found"
                        PerformanceMetrics = Map.empty
                    }
                else
                    logger.LogInformation($"✅ Found {devices.Length} CUDA device(s):")
                    for device in devices do
                        logger.LogInformation($"  Device {device.DeviceId}: {device.Name}")
                        logger.LogInformation($"    Memory: {device.TotalMemory / (1024UL * 1024UL * 1024UL)}GB")
                        logger.LogInformation($"    Compute Capability: {device.ComputeCapability / 10}.{device.ComputeCapability % 10}")
                    
                    // Test initialization on first device
                    use cudaContext = new CudaContext(0, logger)
                    let startTime = DateTime.UtcNow
                    let! initSuccess = cudaContext.Initialize()
                    let endTime = DateTime.UtcNow
                    let executionTime = (endTime - startTime).TotalMilliseconds
                    
                    if initSuccess then
                        let deviceInfo = cudaContext.GetDeviceInfo()
                        match deviceInfo with
                        | Some info ->
                            return {
                                TestName = "Device Initialization"
                                Success = true
                                ExecutionTimeMs = executionTime
                                ThroughputOpsPerSec = None
                                ErrorMessage = None
                                PerformanceMetrics = Map [
                                    ("total_memory_gb", float info.TotalMemory / (1024.0 * 1024.0 * 1024.0))
                                    ("compute_capability", float info.ComputeCapability / 10.0)
                                    ("initialization_time_ms", executionTime)
                                ]
                            }
                        | None ->
                            return {
                                TestName = "Device Initialization"
                                Success = false
                                ExecutionTimeMs = executionTime
                                ThroughputOpsPerSec = None
                                ErrorMessage = Some "Failed to get device info"
                                PerformanceMetrics = Map.empty
                            }
                    else
                        return {
                            TestName = "Device Initialization"
                            Success = false
                            ExecutionTimeMs = executionTime
                            ThroughputOpsPerSec = None
                            ErrorMessage = Some "CUDA initialization failed"
                            PerformanceMetrics = Map.empty
                        }
            with
            | ex ->
                logger.LogError($"❌ Device initialization test failed: {ex.Message}")
                return {
                    TestName = "Device Initialization"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test tensor creation and memory management
        member _.TestTensorOperations() = async {
            logger.LogInformation("🧪 Testing CUDA tensor operations...")
            
            try
                use cudaContext = new CudaContext(0, logger)
                let! initSuccess = cudaContext.Initialize()
                
                if not initSuccess then
                    return {
                        TestName = "Tensor Operations"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "CUDA initialization failed"
                        PerformanceMetrics = Map.empty
                    }
                
                let startTime = DateTime.UtcNow
                
                // Test tensor creation
                let shape = [| 1024; 1024 |]  // 1M elements
                let! tensorOpt = cudaContext.CreateTensor(shape, "float16")
                
                match tensorOpt with
                | Some tensor ->
                    logger.LogInformation($"✅ Created tensor: {shape.[0]}x{shape.[1]} (float16)")
                    
                    // Test tensor destruction
                    let! destroySuccess = cudaContext.DestroyTensor(tensor)
                    
                    let endTime = DateTime.UtcNow
                    let executionTime = (endTime - startTime).TotalMilliseconds
                    
                    if destroySuccess then
                        logger.LogInformation("✅ Tensor destroyed successfully")
                        
                        let totalElements = shape |> Array.fold (*) 1
                        let throughput = float totalElements / (executionTime / 1000.0)
                        
                        return {
                            TestName = "Tensor Operations"
                            Success = true
                            ExecutionTimeMs = executionTime
                            ThroughputOpsPerSec = Some throughput
                            ErrorMessage = None
                            PerformanceMetrics = Map [
                                ("tensor_elements", float totalElements)
                                ("memory_allocation_time_ms", executionTime)
                                ("elements_per_second", throughput)
                            ]
                        }
                    else
                        return {
                            TestName = "Tensor Operations"
                            Success = false
                            ExecutionTimeMs = executionTime
                            ThroughputOpsPerSec = None
                            ErrorMessage = Some "Tensor destruction failed"
                            PerformanceMetrics = Map.empty
                        }
                | None ->
                    let endTime = DateTime.UtcNow
                    let executionTime = (endTime - startTime).TotalMilliseconds
                    
                    return {
                        TestName = "Tensor Operations"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "Tensor creation failed"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ Tensor operations test failed: {ex.Message}")
                return {
                    TestName = "Tensor Operations"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test matrix multiplication with Tensor Cores - REAL GPU EXECUTION
        member _.TestMatrixMultiplication() = async {
            logger.LogInformation("🧪 Testing CUDA matrix multiplication with Tensor Cores - REAL GPU EXECUTION...")

            try
                use cudaContext = new CudaContext(0, logger)
                let! initSuccess = cudaContext.Initialize()

                if not initSuccess then
                    return {
                        TestName = "Matrix Multiplication"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "CUDA initialization failed"
                        PerformanceMetrics = Map.empty
                    }

                // Test matrix dimensions (must be multiples of 16 for Tensor Cores)
                let M, N, K = 1024, 1024, 1024
                logger.LogInformation($"🔢 Matrix dimensions: {M}x{N}x{K}")

                // REAL GPU memory allocation
                let elementSize = 2 // FP16 = 2 bytes per element
                let sizeA = unativeint (M * K * elementSize)
                let sizeB = unativeint (K * N * elementSize)
                let sizeC = unativeint (M * N * elementSize)

                logger.LogInformation($"💾 Allocating GPU memory:")
                logger.LogInformation($"  Matrix A: {sizeA / 1024UL / 1024UL}MB")
                logger.LogInformation($"  Matrix B: {sizeB / 1024UL / 1024UL}MB")
                logger.LogInformation($"  Matrix C: {sizeC / 1024UL / 1024UL}MB")

                // Allocate GPU memory
                let mutable ptrA = nativeint 0
                let mutable ptrB = nativeint 0
                let mutable ptrC = nativeint 0

                let allocA = tars_cuda_malloc(&ptrA, sizeA)
                let allocB = tars_cuda_malloc(&ptrB, sizeB)
                let allocC = tars_cuda_malloc(&ptrC, sizeC)

                if allocA <> TarsCudaError.Success || allocB <> TarsCudaError.Success || allocC <> TarsCudaError.Success then
                    // Cleanup any successful allocations
                    if ptrA <> nativeint 0 then tars_cuda_free(ptrA) |> ignore
                    if ptrB <> nativeint 0 then tars_cuda_free(ptrB) |> ignore
                    if ptrC <> nativeint 0 then tars_cuda_free(ptrC) |> ignore

                    return {
                        TestName = "Matrix Multiplication"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some $"GPU memory allocation failed: A={allocA}, B={allocB}, C={allocC}"
                        PerformanceMetrics = Map.empty
                    }

                logger.LogInformation("✅ GPU memory allocated successfully")

                let startTime = DateTime.UtcNow

                // REAL CUDA GEMM operation with Tensor Cores
                let! gemmSuccess = cudaContext.RunGemmTensorCore(ptrA, ptrB, ptrC, M, N, K, 1.0f, 0.0f)

                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds

                // Cleanup GPU memory
                let freeA = tars_cuda_free(ptrA)
                let freeB = tars_cuda_free(ptrB)
                let freeC = tars_cuda_free(ptrC)

                if freeA <> TarsCudaError.Success || freeB <> TarsCudaError.Success || freeC <> TarsCudaError.Success then
                    logger.LogWarning($"⚠️ GPU memory cleanup warnings: A={freeA}, B={freeB}, C={freeC}")

                if gemmSuccess then
                    // Calculate REAL performance metrics
                    let totalOps = 2L * int64 M * int64 N * int64 K // 2 ops per multiply-add
                    let gflops = float totalOps / (executionTime / 1000.0) / 1e9
                    let totalMemoryBytes = int64 sizeA + int64 sizeB + int64 sizeC
                    let memoryBandwidth = float totalMemoryBytes / (executionTime / 1000.0) / 1e9

                    logger.LogInformation($"✅ REAL GEMM completed: {executionTime:F2}ms")
                    logger.LogInformation($"🚀 REAL Performance: {gflops:F1} GFLOPS")
                    logger.LogInformation($"📊 Memory Bandwidth: {memoryBandwidth:F1} GB/s")

                    return {
                        TestName = "Matrix Multiplication"
                        Success = true
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = Some (float totalOps / (executionTime / 1000.0))
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("matrix_size", float (M * N * K))
                            ("execution_time_ms", executionTime)
                            ("gflops", gflops)
                            ("memory_bandwidth_gb_s", memoryBandwidth)
                            ("gpu_memory_mb", float totalMemoryBytes / 1e6)
                        ]
                    }
                else
                    return {
                        TestName = "Matrix Multiplication"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "REAL GEMM operation failed on GPU"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ Matrix multiplication test failed: {ex.Message}")
                return {
                    TestName = "Matrix Multiplication"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Test GELU activation function - REAL GPU EXECUTION
        member _.TestGeluActivation() = async {
            logger.LogInformation("🧪 Testing CUDA GELU activation function - REAL GPU EXECUTION...")

            try
                use cudaContext = new CudaContext(0, logger)
                let! initSuccess = cudaContext.Initialize()

                if not initSuccess then
                    return {
                        TestName = "GELU Activation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "CUDA initialization failed"
                        PerformanceMetrics = Map.empty
                    }

                let numElements = 1024 * 1024 // 1M elements
                logger.LogInformation($"🔢 Processing {numElements:N0} elements")

                // REAL GPU memory allocation for input and output
                let elementSize = 4 // FP32 = 4 bytes per element
                let bufferSize = unativeint (numElements * elementSize)

                logger.LogInformation($"💾 Allocating GPU memory: {bufferSize / 1024UL / 1024UL}MB per buffer")

                let mutable inputPtr = nativeint 0
                let mutable outputPtr = nativeint 0

                let allocInput = tars_cuda_malloc(&inputPtr, bufferSize)
                let allocOutput = tars_cuda_malloc(&outputPtr, bufferSize)

                if allocInput <> TarsCudaError.Success || allocOutput <> TarsCudaError.Success then
                    // Cleanup any successful allocations
                    if inputPtr <> nativeint 0 then tars_cuda_free(inputPtr) |> ignore
                    if outputPtr <> nativeint 0 then tars_cuda_free(outputPtr) |> ignore

                    return {
                        TestName = "GELU Activation"
                        Success = false
                        ExecutionTimeMs = 0.0
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some $"GPU memory allocation failed: Input={allocInput}, Output={allocOutput}"
                        PerformanceMetrics = Map.empty
                    }

                logger.LogInformation("✅ GPU memory allocated successfully")

                let startTime = DateTime.UtcNow

                // REAL CUDA GELU operation
                let! geluSuccess = cudaContext.RunGeluActivation(inputPtr, outputPtr, numElements)

                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds

                // Cleanup GPU memory
                let freeInput = tars_cuda_free(inputPtr)
                let freeOutput = tars_cuda_free(outputPtr)

                if freeInput <> TarsCudaError.Success || freeOutput <> TarsCudaError.Success then
                    logger.LogWarning($"⚠️ GPU memory cleanup warnings: Input={freeInput}, Output={freeOutput}")

                if geluSuccess then
                    let throughput = float numElements / (executionTime / 1000.0)
                    let totalMemoryBytes = int64 bufferSize * 2L // Input + Output
                    let memoryBandwidth = float totalMemoryBytes / (executionTime / 1000.0) / 1e9

                    logger.LogInformation($"✅ REAL GELU completed: {executionTime:F2}ms")
                    logger.LogInformation($"🚀 REAL Throughput: {throughput / 1e6:F1}M elements/sec")
                    logger.LogInformation($"📊 Memory Bandwidth: {memoryBandwidth:F1} GB/s")

                    return {
                        TestName = "GELU Activation"
                        Success = true
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = Some throughput
                        ErrorMessage = None
                        PerformanceMetrics = Map [
                            ("num_elements", float numElements)
                            ("execution_time_ms", executionTime)
                            ("elements_per_second", throughput)
                            ("memory_bandwidth_gb_s", memoryBandwidth)
                            ("gpu_memory_mb", float totalMemoryBytes / 1e6)
                        ]
                    }
                else
                    return {
                        TestName = "GELU Activation"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ThroughputOpsPerSec = None
                        ErrorMessage = Some "REAL GELU operation failed on GPU"
                        PerformanceMetrics = Map.empty
                    }
            with
            | ex ->
                logger.LogError($"❌ GELU activation test failed: {ex.Message}")
                return {
                    TestName = "GELU Activation"
                    Success = false
                    ExecutionTimeMs = 0.0
                    ThroughputOpsPerSec = None
                    ErrorMessage = Some ex.Message
                    PerformanceMetrics = Map.empty
                }
        }
        
        /// Run complete test suite
        member this.RunCompleteTestSuite() = async {
            logger.LogInformation("🧪 Running complete CUDA kernel test suite...")
            
            let tests = [
                this.TestDeviceInitialization
                this.TestTensorOperations
                this.TestMatrixMultiplication
                this.TestGeluActivation
            ]
            
            let mutable results = []
            let mutable totalTests = 0
            let mutable passedTests = 0
            
            for test in tests do
                let! result = test()
                results <- result :: results
                totalTests <- totalTests + 1
                
                if result.Success then
                    passedTests <- passedTests + 1
                    logger.LogInformation($"✅ {result.TestName}: PASSED ({result.ExecutionTimeMs:F2}ms)")
                else
                    let errorMsg = result.ErrorMessage |> Option.defaultValue "Unknown error"
                    logger.LogError($"❌ {result.TestName}: FAILED - {errorMsg}")
            
            let successRate = float passedTests / float totalTests * 100.0
            
            logger.LogInformation($"📊 Test Suite Complete: {passedTests}/{totalTests} tests passed ({successRate:F1}%)")
            
            return (results |> List.rev, successRate)
        }
