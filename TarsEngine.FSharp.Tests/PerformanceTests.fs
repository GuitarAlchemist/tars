namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Core.EnhancedRevolutionaryIntegration
open TarsEngine.FSharp.Tests.TestHelpers

/// Performance and load testing
module PerformanceTests =

    [<PerformanceTest>]
    let ``System should handle concurrent operations efficiently`` () =
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        let operations = [
            SemanticAnalysis("test1", Euclidean, false)
            ConceptEvolution("test2", RevolutionaryTypes.GrammarTier.Advanced, false)
            EmergentDiscovery("test3", false)
        ]
        
        let measurement = measurePerformanceAsync (fun () ->
            operations
            |> List.map (engine.ExecuteEnhancedOperation)
            |> Async.Parallel
        )
        
        let results = measurement |> Async.RunSynchronously
        results.Success |> should be True

    [<PerformanceTest>]
    let ``Memory usage should remain stable under load`` () =
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        let iterations = 100
        let mutable maxMemory = 0L
        
        for i in 1..iterations do
            let operation = SemanticAnalysis($"test_{i}", Euclidean, false)
            let result = engine.ExecuteEnhancedOperation(operation) |> Async.RunSynchronously
            result.Success |> should be True
            
            let currentMemory = GC.GetTotalMemory(false)
            maxMemory <- max maxMemory currentMemory
        
        // Memory should not grow excessively
        maxMemory |> should be (lessThan (500L * 1024L * 1024L)) // 500MB max

    /// Run all Performance tests
    let runAllTests () =
        let testMethods = [
            ("ConcurrentOperations", fun () -> ``System should handle concurrent operations efficiently`` ())
            ("MemoryStability", fun () -> ``Memory usage should remain stable under load`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running Performance test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "Performance" result
        result
