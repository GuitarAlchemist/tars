namespace TarsEngine

open System
open System.Runtime.InteropServices
open Microsoft.Extensions.Logging

/// Real CUDA Test - Testing actual compiled CUDA library
module RealCudaTest =
    
    // ============================================================================
    // REAL P/INVOKE TO COMPILED CUDA LIBRARY
    // ============================================================================
    
    [<Struct>]
    type TarsCudaError =
        | Success = 0
        | InvalidDevice = 1
        | OutOfMemory = 2
        | InvalidValue = 3
        | KernelLaunch = 4
        | CublasError = 5
    
    // Real P/Invoke declarations to our compiled library
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_init(int deviceId)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_cleanup()
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int tars_cuda_device_count()
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_get_device_info(
        int deviceId, 
        [<MarshalAs(UnmanagedType.LPStr)>] System.Text.StringBuilder name, 
        unativeint nameLen,
        unativeint& totalMemory,
        int& computeCapability)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_malloc(nativeint& ptr, unativeint size)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_cuda_free(nativeint ptr)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_gemm_tensor_core(
        nativeint A, nativeint B, nativeint C,
        int M, int N, int K,
        float32 alpha, float32 beta, nativeint stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_gelu_forward(
        nativeint input, nativeint output, int size, nativeint stream)
    
    [<DllImport("libTarsCudaKernels.so", CallingConvention = CallingConvention.Cdecl)>]
    extern TarsCudaError tars_synchronize_device()
    
    // ============================================================================
    // REAL CUDA TEST IMPLEMENTATION
    // ============================================================================
    
    type RealCudaTestResult = {
        TestName: string
        Success: bool
        ExecutionTimeMs: float
        ErrorMessage: string option
        ActualResults: Map<string, obj>
    }
    
    type RealCudaTester(logger: ILogger<RealCudaTester>) =
        
        /// Test real CUDA device detection
        member _.TestRealDeviceDetection() = async {
            logger.LogInformation("🧪 Testing REAL CUDA device detection...")
            
            let startTime = DateTime.UtcNow
            
            try
                // Call real CUDA function
                let deviceCount = tars_cuda_device_count()
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                logger.LogInformation($"📊 Real CUDA devices found: {deviceCount}")
                
                let results = Map [
                    ("device_count", deviceCount :> obj)
                    ("execution_time_ms", executionTime :> obj)
                ]
                
                return {
                    TestName = "Real Device Detection"
                    Success = true
                    ExecutionTimeMs = executionTime
                    ErrorMessage = None
                    ActualResults = results
                }
            with
            | ex ->
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                logger.LogError($"❌ Real device detection failed: {ex.Message}")
                
                return {
                    TestName = "Real Device Detection"
                    Success = false
                    ExecutionTimeMs = executionTime
                    ErrorMessage = Some ex.Message
                    ActualResults = Map.empty
                }
        }
        
        /// Test real CUDA initialization
        member _.TestRealCudaInitialization() = async {
            logger.LogInformation("🧪 Testing REAL CUDA initialization...")
            
            let startTime = DateTime.UtcNow
            
            try
                // Try to initialize CUDA on device 0
                let result = tars_cuda_init(0)
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                if result = TarsCudaError.Success then
                    logger.LogInformation("✅ Real CUDA initialization successful")
                    
                    // Test cleanup
                    let cleanupResult = tars_cuda_cleanup()
                    
                    let results = Map [
                        ("init_result", result :> obj)
                        ("cleanup_result", cleanupResult :> obj)
                        ("execution_time_ms", executionTime :> obj)
                    ]
                    
                    return {
                        TestName = "Real CUDA Initialization"
                        Success = cleanupResult = TarsCudaError.Success
                        ExecutionTimeMs = executionTime
                        ErrorMessage = None
                        ActualResults = results
                    }
                else
                    logger.LogWarning($"⚠️ Real CUDA initialization returned: {result}")
                    
                    let results = Map [
                        ("init_result", result :> obj)
                        ("execution_time_ms", executionTime :> obj)
                    ]
                    
                    return {
                        TestName = "Real CUDA Initialization"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ErrorMessage = Some $"CUDA init failed: {result}"
                        ActualResults = results
                    }
            with
            | ex ->
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                logger.LogError($"❌ Real CUDA initialization exception: {ex.Message}")
                
                return {
                    TestName = "Real CUDA Initialization"
                    Success = false
                    ExecutionTimeMs = executionTime
                    ErrorMessage = Some ex.Message
                    ActualResults = Map.empty
                }
        }
        
        /// Test real GPU memory allocation
        member _.TestRealGpuMemoryAllocation() = async {
            logger.LogInformation("🧪 Testing REAL GPU memory allocation...")
            
            let startTime = DateTime.UtcNow
            
            try
                // Initialize CUDA first
                let initResult = tars_cuda_init(0)
                
                if initResult = TarsCudaError.Success then
                    // Allocate 1MB of GPU memory
                    let size = 1024UL * 1024UL // 1MB
                    let mutable ptr = nativeint 0
                    
                    let allocResult = tars_cuda_malloc(&ptr, unativeint size)
                    
                    if allocResult = TarsCudaError.Success && ptr <> nativeint 0 then
                        logger.LogInformation($"✅ Real GPU memory allocated: {size} bytes at {ptr}")
                        
                        // Free the memory
                        let freeResult = tars_cuda_free(ptr)
                        
                        // Cleanup CUDA
                        let cleanupResult = tars_cuda_cleanup()
                        
                        let endTime = DateTime.UtcNow
                        let executionTime = (endTime - startTime).TotalMilliseconds
                        
                        let results = Map [
                            ("allocated_bytes", size :> obj)
                            ("gpu_pointer", ptr.ToString() :> obj)
                            ("alloc_result", allocResult :> obj)
                            ("free_result", freeResult :> obj)
                            ("execution_time_ms", executionTime :> obj)
                        ]
                        
                        return {
                            TestName = "Real GPU Memory Allocation"
                            Success = freeResult = TarsCudaError.Success
                            ExecutionTimeMs = executionTime
                            ErrorMessage = None
                            ActualResults = results
                        }
                    else
                        tars_cuda_cleanup() |> ignore
                        
                        let endTime = DateTime.UtcNow
                        let executionTime = (endTime - startTime).TotalMilliseconds
                        
                        return {
                            TestName = "Real GPU Memory Allocation"
                            Success = false
                            ExecutionTimeMs = executionTime
                            ErrorMessage = Some $"Memory allocation failed: {allocResult}"
                            ActualResults = Map [("alloc_result", allocResult :> obj)]
                        }
                else
                    let endTime = DateTime.UtcNow
                    let executionTime = (endTime - startTime).TotalMilliseconds
                    
                    return {
                        TestName = "Real GPU Memory Allocation"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ErrorMessage = Some $"CUDA init failed: {initResult}"
                        ActualResults = Map [("init_result", initResult :> obj)]
                    }
            with
            | ex ->
                tars_cuda_cleanup() |> ignore
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                logger.LogError($"❌ Real GPU memory allocation exception: {ex.Message}")
                
                return {
                    TestName = "Real GPU Memory Allocation"
                    Success = false
                    ExecutionTimeMs = executionTime
                    ErrorMessage = Some ex.Message
                    ActualResults = Map.empty
                }
        }
        
        /// Test real CUDA kernel execution
        member _.TestRealCudaKernelExecution() = async {
            logger.LogInformation("🧪 Testing REAL CUDA kernel execution...")
            
            let startTime = DateTime.UtcNow
            
            try
                // Initialize CUDA
                let initResult = tars_cuda_init(0)
                
                if initResult = TarsCudaError.Success then
                    // Test GELU kernel with small array
                    let size = 1024 // 1K elements
                    let sizeBytes = size * 2 // FP16 = 2 bytes per element
                    
                    let mutable inputPtr = nativeint 0
                    let mutable outputPtr = nativeint 0
                    
                    // Allocate GPU memory
                    let allocResult1 = tars_cuda_malloc(&inputPtr, unativeint sizeBytes)
                    let allocResult2 = tars_cuda_malloc(&outputPtr, unativeint sizeBytes)
                    
                    if allocResult1 = TarsCudaError.Success && allocResult2 = TarsCudaError.Success then
                        logger.LogInformation($"✅ Real GPU memory allocated for kernel test")
                        
                        // Execute GELU kernel
                        let kernelResult = tars_gelu_forward(inputPtr, outputPtr, size, nativeint 0)
                        
                        if kernelResult = TarsCudaError.Success then
                            // Synchronize to ensure kernel completion
                            let syncResult = tars_synchronize_device()
                            
                            logger.LogInformation($"✅ Real CUDA GELU kernel executed successfully")
                            
                            // Cleanup
                            tars_cuda_free(inputPtr) |> ignore
                            tars_cuda_free(outputPtr) |> ignore
                            tars_cuda_cleanup() |> ignore
                            
                            let endTime = DateTime.UtcNow
                            let executionTime = (endTime - startTime).TotalMilliseconds
                            
                            let results = Map [
                                ("kernel_elements", size :> obj)
                                ("kernel_result", kernelResult :> obj)
                                ("sync_result", syncResult :> obj)
                                ("execution_time_ms", executionTime :> obj)
                            ]
                            
                            return {
                                TestName = "Real CUDA Kernel Execution"
                                Success = syncResult = TarsCudaError.Success
                                ExecutionTimeMs = executionTime
                                ErrorMessage = None
                                ActualResults = results
                            }
                        else
                            tars_cuda_free(inputPtr) |> ignore
                            tars_cuda_free(outputPtr) |> ignore
                            tars_cuda_cleanup() |> ignore
                            
                            let endTime = DateTime.UtcNow
                            let executionTime = (endTime - startTime).TotalMilliseconds
                            
                            return {
                                TestName = "Real CUDA Kernel Execution"
                                Success = false
                                ExecutionTimeMs = executionTime
                                ErrorMessage = Some $"Kernel execution failed: {kernelResult}"
                                ActualResults = Map [("kernel_result", kernelResult :> obj)]
                            }
                    else
                        tars_cuda_cleanup() |> ignore
                        
                        let endTime = DateTime.UtcNow
                        let executionTime = (endTime - startTime).TotalMilliseconds
                        
                        return {
                            TestName = "Real CUDA Kernel Execution"
                            Success = false
                            ExecutionTimeMs = executionTime
                            ErrorMessage = Some "GPU memory allocation failed for kernel test"
                            ActualResults = Map.empty
                        }
                else
                    let endTime = DateTime.UtcNow
                    let executionTime = (endTime - startTime).TotalMilliseconds
                    
                    return {
                        TestName = "Real CUDA Kernel Execution"
                        Success = false
                        ExecutionTimeMs = executionTime
                        ErrorMessage = Some $"CUDA init failed: {initResult}"
                        ActualResults = Map [("init_result", initResult :> obj)]
                    }
            with
            | ex ->
                tars_cuda_cleanup() |> ignore
                
                let endTime = DateTime.UtcNow
                let executionTime = (endTime - startTime).TotalMilliseconds
                
                logger.LogError($"❌ Real CUDA kernel execution exception: {ex.Message}")
                
                return {
                    TestName = "Real CUDA Kernel Execution"
                    Success = false
                    ExecutionTimeMs = executionTime
                    ErrorMessage = Some ex.Message
                    ActualResults = Map.empty
                }
        }
        
        /// Run complete real CUDA test suite
        member this.RunRealCudaTestSuite() = async {
            logger.LogInformation("🚀 Running REAL CUDA Test Suite - No Simulations!")
            logger.LogInformation("=======================================================")
            
            let tests = [
                this.TestRealDeviceDetection
                this.TestRealCudaInitialization
                this.TestRealGpuMemoryAllocation
                this.TestRealCudaKernelExecution
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
                    
                    // Log actual results
                    for kvp in result.ActualResults do
                        logger.LogInformation($"   📊 {kvp.Key}: {kvp.Value}")
                else
                    let errorMsg = result.ErrorMessage |> Option.defaultValue "Unknown error"
                    logger.LogError($"❌ {result.TestName}: FAILED - {errorMsg}")
            
            let successRate = float passedTests / float totalTests * 100.0
            
            logger.LogInformation("")
            logger.LogInformation($"📊 REAL CUDA Test Suite Complete:")
            logger.LogInformation($"   Tests Passed: {passedTests}/{totalTests}")
            logger.LogInformation($"   Success Rate: {successRate:F1}%%")
            logger.LogInformation($"   Library: libTarsCudaKernels.so")
            logger.LogInformation($"   Platform: WSL2 with NVIDIA GPU")
            
            return (results |> List.rev, successRate, passedTests > 0)
        }
