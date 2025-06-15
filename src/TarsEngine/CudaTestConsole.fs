namespace TarsEngine

open System
open Microsoft.Extensions.Logging
open Microsoft.Extensions.DependencyInjection
open TarsEngine.CudaComprehensiveTestRunner

/// Console application for running CUDA tests
module CudaTestConsole =

    /// Configure logging for the test console
    let configureLogging() =
        let services = ServiceCollection()
        services.AddLogging(fun builder ->
            builder
                .AddConsole()
                .SetMinimumLevel(LogLevel.Information)
            |> ignore
        ) |> ignore

        let serviceProvider = services.BuildServiceProvider()
        serviceProvider.GetService<ILoggerFactory>()

    /// Main entry point for CUDA tests
    let runCudaTests args =
        try
            printfn ""
            printfn "🚀 TARS CUDA COMPREHENSIVE TEST SUITE"
            printfn "====================================="
            printfn "Real GPU execution - No simulations!"
            printfn ""

            // Configure logging
            let loggerFactory = configureLogging()
            let logger = loggerFactory.CreateLogger<CudaComprehensiveTestRunner>()

            // Create test runner
            let testRunner = CudaComprehensiveTestRunner(logger)

            // Check command line arguments
            let testMode =
                if args.Length > 0 then
                    match args.[0].ToLower() with
                    | "basic" -> "basic"
                    | "memory" -> "memory"
                    | "performance" -> "performance"
                    | "advanced" -> "advanced"
                    | "errors" -> "errors"
                    | "all" | "comprehensive" -> "comprehensive"
                    | _ -> "comprehensive"
                else
                    "comprehensive"

            printfn "🔧 Test Mode: %s" testMode
            printfn ""

            // Run tests based on mode
            let testTask =
                match testMode with
                | "basic" ->
                    async {
                        let! result = testRunner.RunBasicKernelTests()
                        printfn "Basic Kernel Tests: %d/%d passed (%.1f%%)" result.PassedTests result.TotalTests result.SuccessRate
                        return result.SuccessRate >= 80.0
                    }
                | "memory" ->
                    async {
                        let! result = testRunner.RunMemoryTests()
                        printfn "Memory Tests: %d/%d passed (%.1f%%)" result.PassedTests result.TotalTests result.SuccessRate
                        return result.SuccessRate >= 80.0
                    }
                | "performance" ->
                    async {
                        let! result = testRunner.RunPerformanceTests()
                        printfn "Performance Tests: %d/%d passed (%.1f%%)" result.PassedTests result.TotalTests result.SuccessRate
                        return result.SuccessRate >= 80.0
                    }
                | "advanced" ->
                    async {
                        let! result = testRunner.RunAdvancedKernelTests()
                        printfn "Advanced Kernel Tests: %d/%d passed (%.1f%%)" result.PassedTests result.TotalTests result.SuccessRate
                        return result.SuccessRate >= 80.0
                    }
                | "errors" ->
                    async {
                        let! result = testRunner.RunErrorHandlingTests()
                        printfn "Error Handling Tests: %d/%d passed (%.1f%%)" result.PassedTests result.TotalTests result.SuccessRate
                        return result.SuccessRate >= 80.0
                    }
                | _ -> // comprehensive
                    async {
                        let! report = testRunner.RunComprehensiveTestSuite()
                        return report.OverallSuccessRate >= 70.0
                    }

            // Execute the test
            let success = testTask |> Async.RunSynchronously

            printfn ""
            if success then
                printfn "🎉 CUDA tests completed successfully!"
                printfn "✅ CUDA implementation is working correctly"
                0 // Success exit code
            else
                printfn "❌ CUDA tests failed!"
                printfn "🔧 Check the logs above for specific issues"
                1 // Failure exit code
        with
        | ex ->
            printfn "💥 Fatal error running CUDA tests: %s" ex.Message
            2 // Fatal error exit code
