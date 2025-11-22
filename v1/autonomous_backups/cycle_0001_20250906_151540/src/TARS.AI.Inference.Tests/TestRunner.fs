namespace TARS.AI.Inference.Tests

open System
open System.Diagnostics
open System.Threading.Tasks
open Xunit
open TARS.AI.Inference.Tests.Unit.CudaInteropTests
open TARS.AI.Inference.Tests.Integration.PerformanceBenchmarks
open TARS.AI.Inference.Tests.Integration.OllamaCompatibilityTests

/// Comprehensive test runner for TARS AI Inference Engine
module TestRunner =

    /// Test suite configuration
    type TestConfig = {
        RunUnitTests: bool
        RunIntegrationTests: bool
        RunPerformanceTests: bool
        RunStressTests: bool
        RunValidationTests: bool
        Verbose: bool
        GenerateReport: bool
        ReportPath: string
    }

    let defaultConfig = {
        RunUnitTests = true
        RunIntegrationTests = true
        RunPerformanceTests = true
        RunStressTests = false  // Disabled by default (time-consuming)
        RunValidationTests = true
        Verbose = true
        GenerateReport = true
        ReportPath = "./test-results/"
    }

    /// Test result summary
    type TestResult = {
        TestName: string
        Passed: bool
        Duration: TimeSpan
        ErrorMessage: string option
        Details: Map<string, obj>
    }

    type TestSuiteResult = {
        SuiteName: string
        Results: TestResult list
        TotalTests: int
        PassedTests: int
        FailedTests: int
        TotalDuration: TimeSpan
        SuccessRate: float
    }

    /// Execute a single test with error handling
    let executeTest (testName: string) (testFunc: unit -> Task<unit>) : Task<TestResult> =
        task {
            let stopwatch = Stopwatch.StartNew()
            try
                do! testFunc()
                stopwatch.Stop()
                return {
                    TestName = testName
                    Passed = true
                    Duration = stopwatch.Elapsed
                    ErrorMessage = None
                    Details = Map.empty
                }
            with
            | ex ->
                stopwatch.Stop()
                return {
                    TestName = testName
                    Passed = false
                    Duration = stopwatch.Elapsed
                    ErrorMessage = Some(ex.Message)
                    Details = Map.ofList [("exception_type", box ex.GetType().Name)]
                }
        }

    /// Run unit tests
    let runUnitTests (config: TestConfig) : Task<TestSuiteResult> =
        task {
            if config.Verbose then
                printfn "🧪 Running Unit Tests"
                printfn "===================="
            
            let stopwatch = Stopwatch.StartNew()
            
            let tests = [
                ("CUDA Availability Check", fun () -> task {
                    let result = CudaInteropTests.``CUDA availability check should not throw``()
                    return ()
                })
                ("Device Info Validation", fun () -> task {
                    CudaInteropTests.``Get device info should handle invalid device gracefully``()
                    return ()
                })
                ("Context Initialization", fun () -> task {
                    CudaInteropTests.``CUDA context initialization should be deterministic``()
                    return ()
                })
                ("Error Message Retrieval", fun () -> task {
                    CudaInteropTests.``Error message retrieval should work for all error codes``()
                    return ()
                })
                ("Memory Tests", fun () -> task {
                    CudaMemoryTests.``Tensor creation should validate parameters``()
                    CudaMemoryTests.``Memory allocation size calculations should be correct``()
                    CudaMemoryTests.``Large tensor size calculations should not overflow``()
                    return ()
                })
                ("Error Handling", fun () -> task {
                    CudaErrorHandlingTests.``Error handling should be consistent``()
                    CudaErrorHandlingTests.``Success error code should be zero``()
                    CudaErrorHandlingTests.``Error codes should be positive``()
                    return ()
                })
            ]
            
            let! results = 
                tests 
                |> List.map (fun (name, test) -> executeTest name test)
                |> Task.WhenAll
            
            stopwatch.Stop()
            
            let resultsList = results |> Array.toList
            let passedCount = resultsList |> List.filter (fun r -> r.Passed) |> List.length
            let failedCount = resultsList |> List.filter (fun r -> not r.Passed) |> List.length
            
            if config.Verbose then
                for result in resultsList do
                    let status = if result.Passed then "✅" else "❌"
                    printfn "   %s %s (%.1fms)" status result.TestName result.Duration.TotalMilliseconds
                    match result.ErrorMessage with
                    | Some(msg) -> printfn "      Error: %s" msg
                    | None -> ()
                
                printfn ""
                printfn "Unit Tests Summary: %d passed, %d failed" passedCount failedCount
            
            return {
                SuiteName = "Unit Tests"
                Results = resultsList
                TotalTests = resultsList.Length
                PassedTests = passedCount
                FailedTests = failedCount
                TotalDuration = stopwatch.Elapsed
                SuccessRate = float passedCount / float resultsList.Length
            }
        }

    /// Run integration tests
    let runIntegrationTests (config: TestConfig) : Task<TestSuiteResult> =
        task {
            if config.Verbose then
                printfn "🔗 Running Integration Tests"
                printfn "============================"
            
            let stopwatch = Stopwatch.StartNew()
            
            let tests = [
                ("Ollama Generate Compatibility", fun () -> 
                    OllamaCompatibilityTests.``TARS should handle basic Ollama generate request``())
                ("Ollama Chat Compatibility", fun () -> 
                    OllamaCompatibilityTests.``TARS should handle Ollama chat requests``())
                ("Model Listing", fun () -> 
                    OllamaCompatibilityTests.``TARS should handle model listing``())
                ("Response Format Validation", fun () -> 
                    OllamaCompatibilityTests.``TARS should maintain Ollama response format``())
                ("Performance Consistency", fun () -> 
                    OllamaCompatibilityTests.``TARS should provide consistent performance metrics``())
            ]
            
            let! results = 
                tests 
                |> List.map (fun (name, test) -> executeTest name test)
                |> Task.WhenAll
            
            stopwatch.Stop()
            
            let resultsList = results |> Array.toList
            let passedCount = resultsList |> List.filter (fun r -> r.Passed) |> List.length
            let failedCount = resultsList |> List.filter (fun r -> not r.Passed) |> List.length
            
            if config.Verbose then
                for result in resultsList do
                    let status = if result.Passed then "✅" else "❌"
                    printfn "   %s %s (%.1fms)" status result.TestName result.Duration.TotalMilliseconds
                    match result.ErrorMessage with
                    | Some(msg) -> printfn "      Error: %s" msg
                    | None -> ()
                
                printfn ""
                printfn "Integration Tests Summary: %d passed, %d failed" passedCount failedCount
            
            return {
                SuiteName = "Integration Tests"
                Results = resultsList
                TotalTests = resultsList.Length
                PassedTests = passedCount
                FailedTests = failedCount
                TotalDuration = stopwatch.Elapsed
                SuccessRate = float passedCount / float resultsList.Length
            }
        }

    /// Run performance tests
    let runPerformanceTests (config: TestConfig) : Task<TestSuiteResult> =
        task {
            if config.Verbose then
                printfn "⚡ Running Performance Tests"
                printfn "==========================="
            
            let stopwatch = Stopwatch.StartNew()
            
            let tests = [
                ("Short Prompt Performance", fun () -> 
                    PerformanceBenchmarks.``TARS should outperform Ollama on short prompts``())
                ("Long Prompt Performance", fun () -> 
                    PerformanceBenchmarks.``TARS should outperform Ollama on long prompts``())
                ("CUDA vs CPU Performance", fun () -> 
                    PerformanceBenchmarks.``TARS CUDA should outperform TARS CPU``())
                ("Performance Consistency", fun () -> 
                    PerformanceBenchmarks.``Performance should be consistent across multiple runs``())
                ("Throughput Scaling", fun () -> 
                    PerformanceBenchmarks.``Throughput should scale with concurrency``())
                ("Memory Usage", fun () -> 
                    PerformanceBenchmarks.``Memory usage should be reasonable``())
            ]
            
            let! results = 
                tests 
                |> List.map (fun (name, test) -> executeTest name test)
                |> Task.WhenAll
            
            stopwatch.Stop()
            
            let resultsList = results |> Array.toList
            let passedCount = resultsList |> List.filter (fun r -> r.Passed) |> List.length
            let failedCount = resultsList |> List.filter (fun r -> not r.Passed) |> List.length
            
            if config.Verbose then
                for result in resultsList do
                    let status = if result.Passed then "✅" else "❌"
                    printfn "   %s %s (%.1fms)" status result.TestName result.Duration.TotalMilliseconds
                    match result.ErrorMessage with
                    | Some(msg) -> printfn "      Error: %s" msg
                    | None -> ()
                
                printfn ""
                printfn "Performance Tests Summary: %d passed, %d failed" passedCount failedCount
            
            return {
                SuiteName = "Performance Tests"
                Results = resultsList
                TotalTests = resultsList.Length
                PassedTests = passedCount
                FailedTests = failedCount
                TotalDuration = stopwatch.Elapsed
                SuccessRate = float passedCount / float resultsList.Length
            }
        }

    /// Generate test report
    let generateTestReport (suiteResults: TestSuiteResult list) (config: TestConfig) : Task<unit> =
        task {
            if not config.GenerateReport then
                return ()
            
            let totalTests = suiteResults |> List.sumBy (fun s -> s.TotalTests)
            let totalPassed = suiteResults |> List.sumBy (fun s -> s.PassedTests)
            let totalFailed = suiteResults |> List.sumBy (fun s -> s.FailedTests)
            let overallSuccessRate = float totalPassed / float totalTests
            
            let reportContent = sprintf """# TARS AI Inference Engine - Test Report

**Generated:** %s
**Total Tests:** %d
**Passed:** %d
**Failed:** %d
**Success Rate:** %.1f%%

## Test Suite Results

%s

## Summary

%s

---
*Generated by TARS AI Inference Engine Test Suite*
""" 
                (DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC"))
                totalTests totalPassed totalFailed (overallSuccessRate * 100.0)
                (suiteResults |> List.map (fun s -> 
                    sprintf "### %s\n- Tests: %d\n- Passed: %d\n- Failed: %d\n- Success Rate: %.1f%%\n- Duration: %.1fs\n" 
                        s.SuiteName s.TotalTests s.PassedTests s.FailedTests (s.SuccessRate * 100.0) s.TotalDuration.TotalSeconds
                ) |> String.concat "\n")
                (if overallSuccessRate >= 0.95 then 
                    "✅ **EXCELLENT** - Test suite passed with high confidence"
                elif overallSuccessRate >= 0.80 then 
                    "⚠️ **GOOD** - Test suite passed with some issues to investigate"
                else 
                    "❌ **NEEDS ATTENTION** - Significant test failures require investigation")
            
            let reportPath = System.IO.Path.Combine(config.ReportPath, $"test-report-{DateTime.UtcNow:yyyyMMdd-HHmmss}.md")
            System.IO.Directory.CreateDirectory(config.ReportPath) |> ignore
            do! System.IO.File.WriteAllTextAsync(reportPath, reportContent)
            
            if config.Verbose then
                printfn "📄 Test report generated: %s" reportPath
        }

    /// Run comprehensive test suite
    let runComprehensiveTests (config: TestConfig) : Task<int> =
        task {
            try
                printfn "🚀 TARS AI INFERENCE ENGINE - COMPREHENSIVE TESTING"
                printfn "=================================================="
                printfn "Starting comprehensive test suite execution..."
                printfn ""
                
                let overallStopwatch = Stopwatch.StartNew()
                let mutable suiteResults = []
                
                // Run test suites based on configuration
                if config.RunUnitTests then
                    let! unitResults = runUnitTests config
                    suiteResults <- unitResults :: suiteResults
                
                if config.RunIntegrationTests then
                    let! integrationResults = runIntegrationTests config
                    suiteResults <- integrationResults :: suiteResults
                
                if config.RunPerformanceTests then
                    let! performanceResults = runPerformanceTests config
                    suiteResults <- performanceResults :: suiteResults
                
                overallStopwatch.Stop()
                
                // Generate report
                do! generateTestReport suiteResults config
                
                // Final summary
                let totalTests = suiteResults |> List.sumBy (fun s -> s.TotalTests)
                let totalPassed = suiteResults |> List.sumBy (fun s -> s.PassedTests)
                let totalFailed = suiteResults |> List.sumBy (fun s -> s.FailedTests)
                let overallSuccessRate = float totalPassed / float totalTests
                
                printfn ""
                printfn "🎉 COMPREHENSIVE TESTING COMPLETE!"
                printfn "=================================="
                printfn "Total Execution Time: %.1fs" overallStopwatch.Elapsed.TotalSeconds
                printfn "Total Tests: %d" totalTests
                printfn "Passed: %d" totalPassed
                printfn "Failed: %d" totalFailed
                printfn "Success Rate: %.1f%%" (overallSuccessRate * 100.0)
                
                if overallSuccessRate >= 0.95 then
                    printfn "✅ EXCELLENT - TARS AI Inference Engine is ready for production!"
                    return 0
                elif overallSuccessRate >= 0.80 then
                    printfn "⚠️ GOOD - TARS AI Inference Engine is mostly ready, investigate failures"
                    return 1
                else
                    printfn "❌ NEEDS ATTENTION - Significant issues found, not ready for production"
                    return 2
                
            with
            | ex ->
                printfn "💥 TEST SUITE ERROR: %s" ex.Message
                return 3
        }

    /// Entry point for test runner
    [<EntryPoint>]
    let main args =
        let config = 
            match args with
            | [| "quick" |] -> { defaultConfig with RunStressTests = false; RunPerformanceTests = false }
            | [| "performance" |] -> { defaultConfig with RunUnitTests = false; RunIntegrationTests = false }
            | [| "unit" |] -> { defaultConfig with RunIntegrationTests = false; RunPerformanceTests = false }
            | [| "integration" |] -> { defaultConfig with RunUnitTests = false; RunPerformanceTests = false }
            | _ -> defaultConfig
        
        let result = runComprehensiveTests config
        result.Result
