namespace TarsEngine

open System
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.CudaKernelTest
open TarsEngine.CudaMemoryTests
open TarsEngine.CudaPerformanceTests
open TarsEngine.CudaAdvancedKernelTests
open TarsEngine.CudaErrorHandlingTests

/// Comprehensive CUDA Test Runner - Real GPU testing with full coverage
module CudaComprehensiveTestRunner =
    
    /// Overall test suite result
    type TestSuiteResult = {
        SuiteName: string
        TotalTests: int
        PassedTests: int
        FailedTests: int
        SuccessRate: float
        TotalExecutionTimeMs: float
        Results: obj list
    }
    
    /// Comprehensive test report
    type ComprehensiveTestReport = {
        TestStartTime: DateTime
        TestEndTime: DateTime
        TotalExecutionTimeMs: float
        TotalTests: int
        TotalPassed: int
        TotalFailed: int
        OverallSuccessRate: float
        SuiteResults: TestSuiteResult list
        SystemInfo: Map<string, string>
        PerformanceSummary: Map<string, float>
    }
    
    /// CUDA Comprehensive Test Runner
    type CudaComprehensiveTestRunner(logger: ILogger<CudaComprehensiveTestRunner>) =
        
        /// Get system information for the test report
        member private _.GetSystemInfo() = async {
            try
                let deviceCount = tars_cuda_device_count()
                let mutable systemInfo = Map [
                    ("test_timestamp", DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))
                    ("cuda_device_count", string deviceCount)
                    ("os_platform", System.Runtime.InteropServices.RuntimeInformation.OSDescription)
                    ("framework_version", System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription)
                ]
                
                if deviceCount > 0 then
                    // Try to get device info for first device
                    let initResult = tars_cuda_init(0)
                    if initResult = TarsCudaError.Success then
                        let name = System.Text.StringBuilder(256)
                        let mutable totalMemory = 0UL
                        let mutable computeCapability = 0
                        
                        let infoResult = tars_cuda_get_device_info(0, name, 256UL, &totalMemory, &computeCapability)
                        if infoResult = TarsCudaError.Success then
                            systemInfo <- systemInfo
                                |> Map.add "gpu_name" (name.ToString())
                                |> Map.add "gpu_memory_gb" (string (totalMemory / (1024UL * 1024UL * 1024UL)))
                                |> Map.add "compute_capability" $"{computeCapability / 10}.{computeCapability % 10}"
                        
                        tars_cuda_cleanup() |> ignore
                
                return systemInfo
            with
            | ex ->
                logger.LogWarning($"⚠️ Could not gather complete system info: {ex.Message}")
                return Map [
                    ("test_timestamp", DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))
                    ("error", ex.Message)
                ]
        }
        
        /// Run basic kernel tests
        member _.RunBasicKernelTests() = async {
            logger.LogInformation("🧪 Running Basic CUDA Kernel Tests...")
            
            try
                let testSuite = CudaKernelTestSuite(logger)
                let! (results, successRate) = testSuite.RunCompleteTestSuite()
                
                let totalTests = results.Length
                let passedTests = results |> List.filter (fun r -> r.Success) |> List.length
                let failedTests = totalTests - passedTests
                let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
                
                return {
                    SuiteName = "Basic Kernel Tests"
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SuccessRate = successRate
                    TotalExecutionTimeMs = totalTime
                    Results = results |> List.map box
                }
            with
            | ex ->
                logger.LogError($"❌ Basic kernel tests failed: {ex.Message}")
                return {
                    SuiteName = "Basic Kernel Tests"
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 1
                    SuccessRate = 0.0
                    TotalExecutionTimeMs = 0.0
                    Results = []
                }
        }
        
        /// Run memory tests
        member _.RunMemoryTests() = async {
            logger.LogInformation("🧪 Running CUDA Memory Tests...")
            
            try
                let testSuite = CudaMemoryTestSuite(logger)
                let! basicMemResult = testSuite.TestBasicMemoryAllocation()
                let! transferResult = testSuite.TestMemoryTransfer()
                
                let results = [basicMemResult; transferResult]
                let totalTests = results.Length
                let passedTests = results |> List.filter (fun r -> r.Success) |> List.length
                let failedTests = totalTests - passedTests
                let successRate = float passedTests / float totalTests * 100.0
                let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
                
                return {
                    SuiteName = "Memory Tests"
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SuccessRate = successRate
                    TotalExecutionTimeMs = totalTime
                    Results = results |> List.map box
                }
            with
            | ex ->
                logger.LogError($"❌ Memory tests failed: {ex.Message}")
                return {
                    SuiteName = "Memory Tests"
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 1
                    SuccessRate = 0.0
                    TotalExecutionTimeMs = 0.0
                    Results = []
                }
        }
        
        /// Run performance tests
        member _.RunPerformanceTests() = async {
            logger.LogInformation("🧪 Running CUDA Performance Tests...")
            
            try
                let testSuite = CudaPerformanceTestSuite(logger)
                let! gemmBenchmark = testSuite.BenchmarkMatrixMultiplication()
                let! geluBenchmark = testSuite.BenchmarkGeluActivation()
                
                let results = [gemmBenchmark; geluBenchmark]
                let totalTests = results.Length
                let passedTests = results |> List.filter (fun r -> r.Success) |> List.length
                let failedTests = totalTests - passedTests
                let successRate = float passedTests / float totalTests * 100.0
                let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
                
                return {
                    SuiteName = "Performance Tests"
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SuccessRate = successRate
                    TotalExecutionTimeMs = totalTime
                    Results = results |> List.map box
                }
            with
            | ex ->
                logger.LogError($"❌ Performance tests failed: {ex.Message}")
                return {
                    SuiteName = "Performance Tests"
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 1
                    SuccessRate = 0.0
                    TotalExecutionTimeMs = 0.0
                    Results = []
                }
        }
        
        /// Run advanced kernel tests
        member _.RunAdvancedKernelTests() = async {
            logger.LogInformation("🧪 Running Advanced CUDA Kernel Tests...")
            
            try
                let testSuite = CudaAdvancedKernelTestSuite(logger)
                let! flashAttentionResult = testSuite.TestFlashAttention()
                let! swigluResult = testSuite.TestSwiGLUActivation()
                let! sedenionResult = testSuite.TestSedenionDistance()
                
                let results = [flashAttentionResult; swigluResult; sedenionResult]
                let totalTests = results.Length
                let passedTests = results |> List.filter (fun r -> r.Success) |> List.length
                let failedTests = totalTests - passedTests
                let successRate = float passedTests / float totalTests * 100.0
                let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
                
                return {
                    SuiteName = "Advanced Kernel Tests"
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SuccessRate = successRate
                    TotalExecutionTimeMs = totalTime
                    Results = results |> List.map box
                }
            with
            | ex ->
                logger.LogError($"❌ Advanced kernel tests failed: {ex.Message}")
                return {
                    SuiteName = "Advanced Kernel Tests"
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 1
                    SuccessRate = 0.0
                    TotalExecutionTimeMs = 0.0
                    Results = []
                }
        }
        
        /// Run error handling tests
        member _.RunErrorHandlingTests() = async {
            logger.LogInformation("🧪 Running CUDA Error Handling Tests...")
            
            try
                let testSuite = CudaErrorHandlingTestSuite(logger)
                let! (results, successRate) = testSuite.RunAllErrorTests()
                
                let totalTests = results.Length
                let passedTests = results |> List.filter (fun r -> r.Success) |> List.length
                let failedTests = totalTests - passedTests
                let totalTime = 0.0 // Error tests don't track execution time
                
                return {
                    SuiteName = "Error Handling Tests"
                    TotalTests = totalTests
                    PassedTests = passedTests
                    FailedTests = failedTests
                    SuccessRate = successRate
                    TotalExecutionTimeMs = totalTime
                    Results = results |> List.map box
                }
            with
            | ex ->
                logger.LogError($"❌ Error handling tests failed: {ex.Message}")
                return {
                    SuiteName = "Error Handling Tests"
                    TotalTests = 0
                    PassedTests = 0
                    FailedTests = 1
                    SuccessRate = 0.0
                    TotalExecutionTimeMs = 0.0
                    Results = []
                }
        }
        
        /// Run complete comprehensive test suite
        member this.RunComprehensiveTestSuite() = async {
            let startTime = DateTime.UtcNow
            logger.LogInformation("🚀 Starting TARS CUDA Comprehensive Test Suite - REAL GPU EXECUTION")
            logger.LogInformation("=" + String.replicate 80 "=")
            
            // Get system information
            let! systemInfo = this.GetSystemInfo()
            
            // Run all test suites
            let! basicResults = this.RunBasicKernelTests()
            let! memoryResults = this.RunMemoryTests()
            let! performanceResults = this.RunPerformanceTests()
            let! advancedResults = this.RunAdvancedKernelTests()
            let! errorResults = this.RunErrorHandlingTests()
            
            let endTime = DateTime.UtcNow
            let totalExecutionTime = (endTime - startTime).TotalMilliseconds
            
            let allSuites = [basicResults; memoryResults; performanceResults; advancedResults; errorResults]
            let totalTests = allSuites |> List.sumBy (fun s -> s.TotalTests)
            let totalPassed = allSuites |> List.sumBy (fun s -> s.PassedTests)
            let totalFailed = allSuites |> List.sumBy (fun s -> s.FailedTests)
            let overallSuccessRate = if totalTests > 0 then float totalPassed / float totalTests * 100.0 else 0.0
            
            // Extract performance metrics
            let performanceSummary = Map [
                ("total_execution_time_ms", totalExecutionTime)
                ("total_tests", float totalTests)
                ("success_rate_percent", overallSuccessRate)
                ("basic_kernel_success_rate", basicResults.SuccessRate)
                ("memory_success_rate", memoryResults.SuccessRate)
                ("performance_success_rate", performanceResults.SuccessRate)
                ("advanced_kernel_success_rate", advancedResults.SuccessRate)
                ("error_handling_success_rate", errorResults.SuccessRate)
            ]
            
            let report = {
                TestStartTime = startTime
                TestEndTime = endTime
                TotalExecutionTimeMs = totalExecutionTime
                TotalTests = totalTests
                TotalPassed = totalPassed
                TotalFailed = totalFailed
                OverallSuccessRate = overallSuccessRate
                SuiteResults = allSuites
                SystemInfo = systemInfo
                PerformanceSummary = performanceSummary
            }
            
            // Print comprehensive summary
            logger.LogInformation("=" + String.replicate 80 "=")
            logger.LogInformation("📊 TARS CUDA COMPREHENSIVE TEST REPORT")
            logger.LogInformation("=" + String.replicate 80 "=")
            logger.LogInformation($"🕐 Test Duration: {totalExecutionTime:F1}ms ({totalExecutionTime / 1000.0:F1}s)")
            logger.LogInformation($"📈 Overall Results: {totalPassed}/{totalTests} tests passed ({overallSuccessRate:F1}%)")
            logger.LogInformation("")
            
            logger.LogInformation("🖥️ System Information:")
            for kvp in systemInfo do
                logger.LogInformation($"  {kvp.Key}: {kvp.Value}")
            logger.LogInformation("")
            
            logger.LogInformation("📋 Test Suite Breakdown:")
            for suite in allSuites do
                let status = if suite.SuccessRate >= 100.0 then "✅" elif suite.SuccessRate >= 80.0 then "⚠️" else "❌"
                logger.LogInformation($"  {status} {suite.SuiteName}: {suite.PassedTests}/{suite.TotalTests} ({suite.SuccessRate:F1}%)")
            
            logger.LogInformation("")
            if overallSuccessRate >= 90.0 then
                logger.LogInformation("🎉 EXCELLENT: CUDA implementation is working correctly!")
            elif overallSuccessRate >= 70.0 then
                logger.LogInformation("⚠️ GOOD: CUDA implementation mostly working, some issues detected")
            else
                logger.LogInformation("❌ ISSUES: CUDA implementation has significant problems")
            
            logger.LogInformation("=" + String.replicate 80 "=")
            
            return report
        }
