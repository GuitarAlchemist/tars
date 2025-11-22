// TARS Comprehensive Test Suite Execution
// Runs all tests and generates detailed reports

#load "tests/TarsEngine.Tests/TarsTestRunner.fs"
#load "tests/TarsEngine.Tests/CudaTranspilationTests.fs"
#load "tests/TarsEngine.Tests/GrammarTierTests.fs"
#load "tests/TarsEngine.Tests/Tier5CudafyTests.fs"
#load "tests/TarsEngine.Tests/IntegrationTests.fs"

open System
open System.IO
open TarsEngine.Tests.TarsTestRunner
open TarsEngine.Tests.CudaTranspilationTests
open TarsEngine.Tests.GrammarTierTests
open TarsEngine.Tests.Tier5CudafyTests
open TarsEngine.Tests.IntegrationTests

printfn "🚀 TARS COMPREHENSIVE TEST SUITE"
printfn "================================="
printfn "Running complete test suite with 80%% coverage target"
printfn ""

// ============================================================================
// TEST SUITE CONFIGURATION
// ============================================================================

let allTestSuites = [
    cudaTranspilationTestSuite
    grammarTierTestSuite
    tier5CudafyTestSuite
    integrationTestSuite
]

printfn "📋 TEST SUITE OVERVIEW:"
printfn "======================="
for suite in allTestSuites do
    printfn "🧪 %s (%s)" suite.SuiteName suite.Category
    printfn "   Tests: %d" suite.Tests.Length
    printfn ""

let totalTests = allTestSuites |> List.sumBy (fun s -> s.Tests.Length)
printfn "📊 TOTAL TESTS TO EXECUTE: %d" totalTests
printfn ""

// ============================================================================
// EXECUTE ALL TESTS
// ============================================================================

let startTime = DateTime.UtcNow
printfn "⏰ Test execution started at: %s" (startTime.ToString("yyyy-MM-dd HH:mm:ss"))
printfn ""

let testReport = runAllTests allTestSuites true

let endTime = DateTime.UtcNow
let totalDuration = endTime - startTime

printfn ""
printfn "⏰ Test execution completed at: %s" (endTime.ToString("yyyy-MM-dd HH:mm:ss"))
printfn "⏱️ Total execution time: %.2f seconds" totalDuration.TotalSeconds
printfn ""

// ============================================================================
// DETAILED ANALYSIS
// ============================================================================

printfn "🔍 DETAILED TEST ANALYSIS"
printfn "========================="

// Analyze results by category
let resultsByCategory = testReport.TestResults |> List.groupBy (fun r -> r.TestCategory)

for (category, results) in resultsByCategory do
    let passed = results |> List.filter (fun r -> r.Success) |> List.length
    let failed = results |> List.filter (fun r -> not r.Success) |> List.length
    let avgTime = results |> List.averageBy (fun r -> r.ExecutionTime.TotalMilliseconds)
    
    printfn ""
    printfn "📂 %s Category:" category
    printfn "   Passed: %d/%d (%.1f%%)" passed results.Length (float passed / float results.Length * 100.0)
    printfn "   Failed: %d" failed
    printfn "   Average Execution Time: %.2f ms" avgTime
    
    if failed > 0 then
        printfn "   Failed Tests:"
        for result in results |> List.filter (fun r -> not r.Success) do
            printfn "     ❌ %s: %s" result.TestName (result.ErrorMessage |> Option.defaultValue "Unknown error")

// Performance analysis
printfn ""
printfn "⚡ PERFORMANCE ANALYSIS"
printfn "======================"

if not testReport.PerformanceBenchmarks.IsEmpty then
    let sortedBenchmarks = 
        testReport.PerformanceBenchmarks 
        |> Map.toList 
        |> List.sortByDescending snd
    
    printfn "🏆 Top 10 Performance Metrics:"
    for i, (metric, value) in sortedBenchmarks |> List.take (min 10 sortedBenchmarks.Length) |> List.indexed do
        printfn "   %d. %s: %.2f ms" (i + 1) metric value
    
    // Identify performance bottlenecks
    let slowOperations = sortedBenchmarks |> List.filter (fun (_, time) -> time > 100.0)
    if not slowOperations.IsEmpty then
        printfn ""
        printfn "⚠️ Performance Bottlenecks (>100ms):"
        for (operation, time) in slowOperations do
            printfn "   🐌 %s: %.2f ms" operation time

// Code coverage analysis
printfn ""
printfn "📊 CODE COVERAGE ANALYSIS"
printfn "========================="
printfn "Estimated Code Coverage: %.1f%%" testReport.CodeCoverage

let coverageStatus = 
    if testReport.CodeCoverage >= 80.0 then "✅ EXCELLENT"
    elif testReport.CodeCoverage >= 70.0 then "✅ GOOD"
    elif testReport.CodeCoverage >= 60.0 then "⚠️ ACCEPTABLE"
    else "❌ NEEDS IMPROVEMENT"

printfn "Coverage Status: %s" coverageStatus

if testReport.CodeCoverage < 80.0 then
    printfn ""
    printfn "📈 COVERAGE IMPROVEMENT RECOMMENDATIONS:"
    printfn "   • Add more unit tests for edge cases"
    printfn "   • Increase integration test scenarios"
    printfn "   • Add performance stress tests"
    printfn "   • Test error handling paths"

// Quality metrics
printfn ""
printfn "🎯 QUALITY METRICS"
printfn "=================="

let successRate = float testReport.PassedTests / float testReport.TotalTests * 100.0
let qualityGrade = 
    if successRate >= 95.0 then "A+"
    elif successRate >= 90.0 then "A"
    elif successRate >= 85.0 then "B+"
    elif successRate >= 80.0 then "B"
    elif successRate >= 75.0 then "C+"
    else "C"

printfn "Success Rate: %.1f%% (Grade: %s)" successRate qualityGrade
printfn "Test Reliability: %s" (if testReport.FailedTests = 0 then "EXCELLENT" else "NEEDS ATTENTION")
printfn "Performance: %s" (if testReport.TotalExecutionTime.TotalSeconds < 30.0 then "FAST" else "ACCEPTABLE")

// Component health assessment
printfn ""
printfn "🏥 COMPONENT HEALTH ASSESSMENT"
printfn "=============================="

let componentHealth = [
    ("CUDA Transpilation", resultsByCategory |> List.tryFind (fun (cat, _) -> cat = "CudaTranspilation"))
    ("Grammar Tiers", resultsByCategory |> List.tryFind (fun (cat, _) -> cat = "GrammarTier"))
    ("Tier 5 Cudafy", resultsByCategory |> List.tryFind (fun (cat, _) -> cat = "Tier5Cudafy"))
    ("Integration", resultsByCategory |> List.tryFind (fun (cat, _) -> cat = "Integration"))
]

for (componentName, categoryResults) in componentHealth do
    match categoryResults with
    | Some (_, results) ->
        let passed = results |> List.filter (fun r -> r.Success) |> List.length
        let total = results.Length
        let healthPercentage = float passed / float total * 100.0
        
        let healthStatus = 
            if healthPercentage = 100.0 then "🟢 HEALTHY"
            elif healthPercentage >= 80.0 then "🟡 GOOD"
            elif healthPercentage >= 60.0 then "🟠 FAIR"
            else "🔴 CRITICAL"
        
        printfn "%s: %s (%.1f%%)" componentName healthStatus healthPercentage
    | None ->
        printfn "%s: ❓ NOT TESTED" componentName

// ============================================================================
// RECOMMENDATIONS
// ============================================================================

printfn ""
printfn "💡 RECOMMENDATIONS"
printfn "=================="

if testReport.FailedTests > 0 then
    printfn "🔧 IMMEDIATE ACTIONS REQUIRED:"
    printfn "   • Fix %d failing test(s)" testReport.FailedTests
    printfn "   • Review error messages and root causes"
    printfn "   • Ensure all dependencies are properly installed"

if testReport.CodeCoverage < 80.0 then
    printfn "📈 COVERAGE IMPROVEMENTS:"
    printfn "   • Target: 80%% code coverage (currently %.1f%%)" testReport.CodeCoverage
    printfn "   • Add tests for untested code paths"
    printfn "   • Increase edge case testing"

if testReport.TotalExecutionTime.TotalSeconds > 60.0 then
    printfn "⚡ PERFORMANCE OPTIMIZATIONS:"
    printfn "   • Optimize slow-running tests"
    printfn "   • Consider parallel test execution"
    printfn "   • Review performance bottlenecks"

printfn ""
printfn "🌟 STRENGTHS:"
if testReport.PassedTests > 0 then
    printfn "   • %d tests passing successfully" testReport.PassedTests
if testReport.CodeCoverage >= 70.0 then
    printfn "   • Good code coverage (%.1f%%)" testReport.CodeCoverage
if testReport.TotalExecutionTime.TotalSeconds < 30.0 then
    printfn "   • Fast test execution"

// ============================================================================
// FINAL SUMMARY
// ============================================================================

printfn ""
printfn "🎉 FINAL TEST SUMMARY"
printfn "===================="

let overallStatus = 
    if testReport.FailedTests = 0 && testReport.CodeCoverage >= 80.0 then
        "🎯 EXCELLENT - All tests passing with high coverage!"
    elif testReport.FailedTests = 0 then
        "✅ GOOD - All tests passing, coverage could be improved"
    elif float testReport.PassedTests / float testReport.TotalTests >= 0.8 then
        "⚠️ ACCEPTABLE - Most tests passing, some issues to address"
    else
        "❌ NEEDS WORK - Multiple test failures require attention"

printfn "Overall Status: %s" overallStatus
printfn ""
printfn "📊 Summary Statistics:"
printfn "   Total Tests: %d" testReport.TotalTests
printfn "   Passed: %d (%.1f%%)" testReport.PassedTests (float testReport.PassedTests / float testReport.TotalTests * 100.0)
printfn "   Failed: %d (%.1f%%)" testReport.FailedTests (float testReport.FailedTests / float testReport.TotalTests * 100.0)
printfn "   Code Coverage: %.1f%%" testReport.CodeCoverage
printfn "   Execution Time: %.2f seconds" testReport.TotalExecutionTime.TotalSeconds
printfn "   Performance Metrics: %d" testReport.PerformanceBenchmarks.Count

printfn ""
printfn "📄 Detailed report saved to: ./test_reports/"
printfn ""

if testReport.FailedTests = 0 then
    printfn "🎉 ALL TESTS PASSED! TARS system is ready for production!"
else
    printfn "⚠️ %d test(s) failed. Please review and fix before deployment." testReport.FailedTests

printfn ""
printfn "🚀 TARS COMPREHENSIVE TEST SUITE COMPLETED!"
printfn "============================================"
