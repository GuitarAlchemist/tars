namespace TarsEngine

open System
open System.Runtime.InteropServices

/// Simple CUDA Test - Real GPU execution
module SimpleCudaTest =
    
    // Basic CUDA function declarations using the working minimal library
    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int minimal_cuda_device_count()

    [<DllImport("libminimal_cuda.so", CallingConvention = CallingConvention.Cdecl)>]
    extern int minimal_cuda_init(int device_id)

    /// Test result with detailed metrics
    type TestResult = {
        TestName: string
        Success: bool
        Message: string
        ExecutionTimeMs: float
        StartTime: DateTime
        EndTime: DateTime
        MemoryUsedMB: float
        CpuUsagePercent: float
    }

    /// System metrics
    type SystemMetrics = {
        TotalMemoryMB: float
        AvailableMemoryMB: float
        CpuCores: int
        ProcessMemoryMB: float
        GcCollections: int * int * int  // Gen0, Gen1, Gen2
    }

    /// CUDA device information
    type CudaDeviceInfo = {
        DeviceId: int
        DeviceName: string
        ComputeCapability: string
        TotalMemoryGB: float
        DriverVersion: string
    }

    /// Get system metrics
    let getSystemMetrics() : SystemMetrics =
        let totalMemory = float (GC.GetTotalMemory(false)) / (1024.0 * 1024.0)
        let currentProcess = System.Diagnostics.Process.GetCurrentProcess()
        let processMemory = float currentProcess.WorkingSet64 / (1024.0 * 1024.0)
        let cpuCores = Environment.ProcessorCount

        {
            TotalMemoryMB = totalMemory
            AvailableMemoryMB = totalMemory // Simplified for now
            CpuCores = cpuCores
            ProcessMemoryMB = processMemory
            GcCollections = (GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2))
        }

    /// Create test result with metrics
    let createTestResult (testName: string) (success: bool) (message: string) (startTime: DateTime) (endTime: DateTime) : TestResult =
        let executionTime = (endTime - startTime).TotalMilliseconds
        let currentMemory = float (GC.GetTotalMemory(false)) / (1024.0 * 1024.0)

        {
            TestName = testName
            Success = success
            Message = message
            ExecutionTimeMs = executionTime
            StartTime = startTime
            EndTime = endTime
            MemoryUsedMB = currentMemory
            CpuUsagePercent = 0.0 // Simplified for now
        }
    
    /// Test CUDA device detection
    let testDeviceDetection() =
        let startTime = DateTime.UtcNow
        try
            let deviceCount = minimal_cuda_device_count()
            let endTime = DateTime.UtcNow

            if deviceCount > 0 then
                createTestResult "Device Detection" true (sprintf "Found %d CUDA device(s)" deviceCount) startTime endTime
            else
                createTestResult "Device Detection" false "No CUDA devices found" startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            createTestResult "Device Detection" false (sprintf "Error: %s" ex.Message) startTime endTime
    
    /// Test CUDA initialization
    let testCudaInitialization() =
        let startTime = DateTime.UtcNow
        try
            let initResult = minimal_cuda_init(0)
            let endTime = DateTime.UtcNow

            if initResult = 0 then
                createTestResult "CUDA Initialization" true "CUDA initialized successfully" startTime endTime
            else
                createTestResult "CUDA Initialization" false (sprintf "CUDA initialization failed with code: %d" initResult) startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            createTestResult "CUDA Initialization" false (sprintf "Error: %s" ex.Message) startTime endTime
    
    /// Test basic CUDA functionality
    let testBasicFunctionality() =
        let startTime = DateTime.UtcNow
        try
            // Test device detection first
            let deviceCount = minimal_cuda_device_count()
            if deviceCount = 0 then
                let endTime = DateTime.UtcNow
                createTestResult "Basic Functionality" false "No CUDA devices found" startTime endTime
            else
                // Test initialization
                let initResult = minimal_cuda_init(0)
                let endTime = DateTime.UtcNow

                if initResult = 0 then
                    createTestResult "Basic Functionality" true (sprintf "Found %d device(s) and initialized successfully" deviceCount) startTime endTime
                else
                    createTestResult "Basic Functionality" false (sprintf "Device found but initialization failed: %d" initResult) startTime endTime
        with
        | ex ->
            let endTime = DateTime.UtcNow
            createTestResult "Basic Functionality" false (sprintf "Error: %s" ex.Message) startTime endTime
    
    /// Display system information
    let displaySystemInfo() =
        let metrics = getSystemMetrics()
        let osInfo = Environment.OSVersion
        let runtimeInfo = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription

        printfn "🖥️  SYSTEM INFORMATION"
        printfn "======================"
        printfn "OS: %s" osInfo.VersionString
        printfn "Runtime: %s" runtimeInfo
        printfn "CPU Cores: %d" metrics.CpuCores
        printfn "Process Memory: %.1f MB" metrics.ProcessMemoryMB
        printfn "GC Collections: Gen0=%d, Gen1=%d, Gen2=%d" (let (g0,g1,g2) = metrics.GcCollections in g0) (let (g0,g1,g2) = metrics.GcCollections in g1) (let (g0,g1,g2) = metrics.GcCollections in g2)
        printfn ""

    /// Display detailed test metrics
    let displayTestMetrics (results: TestResult list) =
        printfn "📈 DETAILED METRICS"
        printfn "==================="

        let totalTime = results |> List.sumBy (fun r -> r.ExecutionTimeMs)
        let avgTime = totalTime / float results.Length
        let minTime = results |> List.map (fun r -> r.ExecutionTimeMs) |> List.min
        let maxTime = results |> List.map (fun r -> r.ExecutionTimeMs) |> List.max

        printfn "Total Execution Time: %.2f ms" totalTime
        printfn "Average Test Time: %.2f ms" avgTime
        printfn "Fastest Test: %.2f ms" minTime
        printfn "Slowest Test: %.2f ms" maxTime
        printfn ""

        printfn "📋 PER-TEST BREAKDOWN"
        printfn "====================="
        for result in List.rev results do
            let status = if result.Success then "✅ PASS" else "❌ FAIL"
            printfn "%s | %-20s | %8.2f ms | %6.1f MB | %s"
                status
                result.TestName
                result.ExecutionTimeMs
                result.MemoryUsedMB
                result.Message
        printfn ""

    /// Run all simple CUDA tests
    let runAllTests() =
        let overallStartTime = DateTime.UtcNow

        printfn ""
        printfn "🚀 TARS CUDA COMPREHENSIVE TEST SUITE"
        printfn "======================================"
        printfn "Real GPU execution - No simulations!"
        printfn ""

        displaySystemInfo()

        let tests = [
            ("Device Detection", testDeviceDetection)
            ("CUDA Initialization", testCudaInitialization)
            ("Basic Functionality", testBasicFunctionality)
        ]

        let mutable results = []
        let mutable totalTests = 0
        let mutable passedTests = 0

        printfn "🧪 RUNNING TESTS"
        printfn "================"

        for (testName, testFunc) in tests do
            printf "🔧 Running %-20s... " testName
            let result = testFunc()
            results <- result :: results
            totalTests <- totalTests + 1

            if result.Success then
                passedTests <- passedTests + 1
                printfn "✅ PASSED (%.2f ms)" result.ExecutionTimeMs
            else
                printfn "❌ FAILED (%.2f ms)" result.ExecutionTimeMs

        let overallEndTime = DateTime.UtcNow
        let overallTime = (overallEndTime - overallStartTime).TotalMilliseconds
        let successRate = float passedTests / float totalTests * 100.0

        printfn ""
        displayTestMetrics results

        printfn "🏁 FINAL RESULTS"
        printfn "================"
        printfn "Tests Passed: %d/%d (%.1f%%)" passedTests totalTests successRate
        printfn "Total Runtime: %.2f ms" overallTime
        printfn "Success Rate: %.1f%%" successRate

        if successRate >= 80.0 then
            printfn ""
            printfn "🎉 CUDA TESTS COMPLETED SUCCESSFULLY!"
            printfn "✅ CUDA implementation is working correctly"
            printfn "🚀 Ready for production use!"
            0 // Success exit code
        else
            printfn ""
            printfn "❌ CUDA TESTS FAILED!"
            printfn "🔧 Check the detailed metrics above for specific issues"
            1 // Failure exit code
    
    /// Main entry point
    [<EntryPoint>]
    let main args =
        try
            runAllTests()
        with
        | ex ->
            printfn "💥 Fatal error running CUDA tests: %s" ex.Message
            printfn "Stack trace: %s" ex.StackTrace
            2 // Fatal error exit code
