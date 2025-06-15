namespace TarsEngine

open System
open Microsoft.Extensions.Logging
open TarsEngine.CudaInterop

/// CUDA Error Handling and Edge Case Tests - Real GPU error scenarios
module CudaErrorHandlingTests =
    
    /// Error test result
    type ErrorTestResult = {
        TestName: string
        Success: bool
        ExpectedError: TarsCudaError option
        ActualError: TarsCudaError option
        ErrorMessage: string option
        TestDetails: Map<string, string>
    }
    
    /// CUDA Error Handling Test Suite
    type CudaErrorHandlingTestSuite(logger: ILogger<CudaErrorHandlingTestSuite>) =
        
        /// Test invalid device ID handling
        member _.TestInvalidDeviceId() = async {
            logger.LogInformation("🧪 Testing invalid device ID error handling...")
            
            try
                let deviceCount = tars_cuda_device_count()
                logger.LogInformation($"📊 Available devices: {deviceCount}")
                
                // Try to initialize with invalid device ID
                let invalidDeviceId = deviceCount + 10 // Way beyond available devices
                logger.LogInformation($"🔧 Attempting to initialize device {invalidDeviceId} (invalid)")
                
                let result = tars_cuda_init(invalidDeviceId)
                
                if result = TarsCudaError.InvalidDevice then
                    logger.LogInformation("✅ Correctly detected invalid device ID")
                    return {
                        TestName = "Invalid Device ID"
                        Success = true
                        ExpectedError = Some TarsCudaError.InvalidDevice
                        ActualError = Some result
                        ErrorMessage = None
                        TestDetails = Map [
                            ("device_count", string deviceCount)
                            ("invalid_device_id", string invalidDeviceId)
                            ("error_code", string result)
                        ]
                    }
                else
                    logger.LogError($"❌ Expected InvalidDevice error, got: {result}")
                    return {
                        TestName = "Invalid Device ID"
                        Success = false
                        ExpectedError = Some TarsCudaError.InvalidDevice
                        ActualError = Some result
                        ErrorMessage = Some $"Expected InvalidDevice error, got {result}"
                        TestDetails = Map [
                            ("device_count", string deviceCount)
                            ("invalid_device_id", string invalidDeviceId)
                            ("error_code", string result)
                        ]
                    }
            with
            | ex ->
                logger.LogError($"❌ Invalid device ID test failed: {ex.Message}")
                return {
                    TestName = "Invalid Device ID"
                    Success = false
                    ExpectedError = Some TarsCudaError.InvalidDevice
                    ActualError = None
                    ErrorMessage = Some ex.Message
                    TestDetails = Map.empty
                }
        }
        
        /// Test out of memory scenarios
        member _.TestOutOfMemory() = async {
            logger.LogInformation("🧪 Testing out of memory error handling...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Out of Memory"
                        Success = false
                        ExpectedError = Some TarsCudaError.OutOfMemory
                        ActualError = None
                        ErrorMessage = Some "No CUDA devices found"
                        TestDetails = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Out of Memory"
                        Success = false
                        ExpectedError = Some TarsCudaError.OutOfMemory
                        ActualError = Some initResult
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        TestDetails = Map.empty
                    }
                
                // Try to allocate an impossibly large amount of memory
                let impossibleSize = unativeint (1024UL * 1024UL * 1024UL * 1024UL) // 1TB
                logger.LogInformation($"🔧 Attempting to allocate {impossibleSize / 1024UL / 1024UL / 1024UL}GB (should fail)")
                
                let mutable ptr = nativeint 0
                let result = tars_cuda_malloc(&ptr, impossibleSize)
                
                // Cleanup CUDA regardless of result
                let cleanupResult = tars_cuda_cleanup()
                
                if result = TarsCudaError.OutOfMemory then
                    logger.LogInformation("✅ Correctly detected out of memory condition")
                    return {
                        TestName = "Out of Memory"
                        Success = true
                        ExpectedError = Some TarsCudaError.OutOfMemory
                        ActualError = Some result
                        ErrorMessage = None
                        TestDetails = Map [
                            ("requested_size_gb", string (impossibleSize / 1024UL / 1024UL / 1024UL))
                            ("error_code", string result)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
                else
                    // If allocation somehow succeeded, free the memory
                    if ptr <> nativeint 0 then
                        let freeResult = tars_cuda_free(ptr)
                        logger.LogWarning($"⚠️ Unexpected allocation success, freed memory: {freeResult}")
                    
                    logger.LogError($"❌ Expected OutOfMemory error, got: {result}")
                    return {
                        TestName = "Out of Memory"
                        Success = false
                        ExpectedError = Some TarsCudaError.OutOfMemory
                        ActualError = Some result
                        ErrorMessage = Some $"Expected OutOfMemory error, got {result}"
                        TestDetails = Map [
                            ("requested_size_gb", string (impossibleSize / 1024UL / 1024UL / 1024UL))
                            ("error_code", string result)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
            with
            | ex ->
                logger.LogError($"❌ Out of memory test failed: {ex.Message}")
                return {
                    TestName = "Out of Memory"
                    Success = false
                    ExpectedError = Some TarsCudaError.OutOfMemory
                    ActualError = None
                    ErrorMessage = Some ex.Message
                    TestDetails = Map.empty
                }
        }
        
        /// Test double initialization scenarios
        member _.TestDoubleInitialization() = async {
            logger.LogInformation("🧪 Testing double initialization handling...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Double Initialization"
                        Success = false
                        ExpectedError = None
                        ActualError = None
                        ErrorMessage = Some "No CUDA devices found"
                        TestDetails = Map.empty
                    }
                
                // First initialization (should succeed)
                logger.LogInformation("🔧 First CUDA initialization...")
                let firstResult = tars_cuda_init(0)
                
                if firstResult <> TarsCudaError.Success then
                    return {
                        TestName = "Double Initialization"
                        Success = false
                        ExpectedError = None
                        ActualError = Some firstResult
                        ErrorMessage = Some $"First CUDA initialization failed: {firstResult}"
                        TestDetails = Map [
                            ("first_init_result", string firstResult)
                        ]
                    }
                
                logger.LogInformation("✅ First initialization succeeded")
                
                // Second initialization (behavior depends on implementation)
                logger.LogInformation("🔧 Second CUDA initialization (should handle gracefully)...")
                let secondResult = tars_cuda_init(0)
                
                // Cleanup
                let cleanupResult = tars_cuda_cleanup()
                
                // Both success or specific error handling are acceptable
                if secondResult = TarsCudaError.Success || secondResult = TarsCudaError.InvalidValue then
                    logger.LogInformation($"✅ Double initialization handled correctly: {secondResult}")
                    return {
                        TestName = "Double Initialization"
                        Success = true
                        ExpectedError = None
                        ActualError = Some secondResult
                        ErrorMessage = None
                        TestDetails = Map [
                            ("first_init_result", string firstResult)
                            ("second_init_result", string secondResult)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
                else
                    logger.LogError($"❌ Unexpected double initialization result: {secondResult}")
                    return {
                        TestName = "Double Initialization"
                        Success = false
                        ExpectedError = None
                        ActualError = Some secondResult
                        ErrorMessage = Some $"Unexpected double initialization result: {secondResult}"
                        TestDetails = Map [
                            ("first_init_result", string firstResult)
                            ("second_init_result", string secondResult)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
            with
            | ex ->
                logger.LogError($"❌ Double initialization test failed: {ex.Message}")
                return {
                    TestName = "Double Initialization"
                    Success = false
                    ExpectedError = None
                    ActualError = None
                    ErrorMessage = Some ex.Message
                    TestDetails = Map.empty
                }
        }
        
        /// Test invalid pointer operations
        member _.TestInvalidPointerOperations() = async {
            logger.LogInformation("🧪 Testing invalid pointer operations...")
            
            try
                let deviceCount = tars_cuda_device_count()
                if deviceCount <= 0 then
                    return {
                        TestName = "Invalid Pointer Operations"
                        Success = false
                        ExpectedError = Some TarsCudaError.InvalidValue
                        ActualError = None
                        ErrorMessage = Some "No CUDA devices found"
                        TestDetails = Map.empty
                    }
                
                let initResult = tars_cuda_init(0)
                if initResult <> TarsCudaError.Success then
                    return {
                        TestName = "Invalid Pointer Operations"
                        Success = false
                        ExpectedError = Some TarsCudaError.InvalidValue
                        ActualError = Some initResult
                        ErrorMessage = Some $"CUDA initialization failed: {initResult}"
                        TestDetails = Map.empty
                    }
                
                // Test freeing null pointer (should handle gracefully)
                logger.LogInformation("🔧 Testing free of null pointer...")
                let nullFreeResult = tars_cuda_free(nativeint 0)
                
                // Test freeing invalid pointer
                logger.LogInformation("🔧 Testing free of invalid pointer...")
                let invalidPtr = nativeint 0x12345678 // Arbitrary invalid address
                let invalidFreeResult = tars_cuda_free(invalidPtr)
                
                // Test operations with invalid parameters
                logger.LogInformation("🔧 Testing GEMM with invalid parameters...")
                let invalidGemmResult = tars_gemm_tensor_core(
                    nativeint 0, nativeint 0, nativeint 0, // Null pointers
                    -1, -1, -1, // Invalid dimensions
                    1.0f, 0.0f, nativeint 0)
                
                // Cleanup
                let cleanupResult = tars_cuda_cleanup()
                
                // Check if errors were handled appropriately
                let errorsHandledCorrectly = 
                    (nullFreeResult = TarsCudaError.Success || nullFreeResult = TarsCudaError.InvalidValue) &&
                    (invalidFreeResult = TarsCudaError.InvalidValue || invalidFreeResult = TarsCudaError.Success) &&
                    (invalidGemmResult = TarsCudaError.InvalidValue || invalidGemmResult = TarsCudaError.KernelLaunch)
                
                if errorsHandledCorrectly then
                    logger.LogInformation("✅ Invalid pointer operations handled correctly")
                    return {
                        TestName = "Invalid Pointer Operations"
                        Success = true
                        ExpectedError = Some TarsCudaError.InvalidValue
                        ActualError = Some invalidGemmResult
                        ErrorMessage = None
                        TestDetails = Map [
                            ("null_free_result", string nullFreeResult)
                            ("invalid_free_result", string invalidFreeResult)
                            ("invalid_gemm_result", string invalidGemmResult)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
                else
                    logger.LogError("❌ Invalid pointer operations not handled correctly")
                    return {
                        TestName = "Invalid Pointer Operations"
                        Success = false
                        ExpectedError = Some TarsCudaError.InvalidValue
                        ActualError = Some invalidGemmResult
                        ErrorMessage = Some "Invalid pointer operations not handled correctly"
                        TestDetails = Map [
                            ("null_free_result", string nullFreeResult)
                            ("invalid_free_result", string invalidFreeResult)
                            ("invalid_gemm_result", string invalidGemmResult)
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
            with
            | ex ->
                logger.LogError($"❌ Invalid pointer operations test failed: {ex.Message}")
                return {
                    TestName = "Invalid Pointer Operations"
                    Success = false
                    ExpectedError = Some TarsCudaError.InvalidValue
                    ActualError = None
                    ErrorMessage = Some ex.Message
                    TestDetails = Map.empty
                }
        }
        
        /// Test cleanup without initialization
        member _.TestCleanupWithoutInit() = async {
            logger.LogInformation("🧪 Testing cleanup without initialization...")
            
            try
                // Try to cleanup without initializing first
                logger.LogInformation("🔧 Attempting cleanup without initialization...")
                let cleanupResult = tars_cuda_cleanup()
                
                // This should either succeed (no-op) or return a specific error
                if cleanupResult = TarsCudaError.Success || cleanupResult = TarsCudaError.InvalidValue then
                    logger.LogInformation($"✅ Cleanup without init handled correctly: {cleanupResult}")
                    return {
                        TestName = "Cleanup Without Init"
                        Success = true
                        ExpectedError = None
                        ActualError = Some cleanupResult
                        ErrorMessage = None
                        TestDetails = Map [
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
                else
                    logger.LogError($"❌ Unexpected cleanup result: {cleanupResult}")
                    return {
                        TestName = "Cleanup Without Init"
                        Success = false
                        ExpectedError = None
                        ActualError = Some cleanupResult
                        ErrorMessage = Some $"Unexpected cleanup result: {cleanupResult}"
                        TestDetails = Map [
                            ("cleanup_result", string cleanupResult)
                        ]
                    }
            with
            | ex ->
                logger.LogError($"❌ Cleanup without init test failed: {ex.Message}")
                return {
                    TestName = "Cleanup Without Init"
                    Success = false
                    ExpectedError = None
                    ActualError = None
                    ErrorMessage = Some ex.Message
                    TestDetails = Map.empty
                }
        }
        
        /// Run all error handling tests
        member this.RunAllErrorTests() = async {
            logger.LogInformation("🧪 Running all CUDA error handling tests...")
            
            let tests = [
                ("Invalid Device ID", this.TestInvalidDeviceId)
                ("Out of Memory", this.TestOutOfMemory)
                ("Double Initialization", this.TestDoubleInitialization)
                ("Invalid Pointer Operations", this.TestInvalidPointerOperations)
                ("Cleanup Without Init", this.TestCleanupWithoutInit)
            ]
            
            let mutable results = []
            let mutable totalTests = 0
            let mutable passedTests = 0
            
            for (testName, testFunc) in tests do
                logger.LogInformation($"🔧 Running {testName}...")
                let! result = testFunc()
                results <- result :: results
                totalTests <- totalTests + 1
                
                if result.Success then
                    passedTests <- passedTests + 1
                    logger.LogInformation($"✅ {testName}: PASSED")
                else
                    let errorMsg = result.ErrorMessage |> Option.defaultValue "Unknown error"
                    logger.LogError($"❌ {testName}: FAILED - {errorMsg}")
            
            let successRate = float passedTests / float totalTests * 100.0
            
            logger.LogInformation($"📊 Error Handling Tests Complete: {passedTests}/{totalTests} tests passed ({successRate:F1}%)")
            
            return (results |> List.rev, successRate)
        }
