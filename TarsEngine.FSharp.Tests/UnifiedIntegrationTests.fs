namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Core.UnifiedIntegration
open TarsEngine.FSharp.Tests.TestHelpers

/// Tests for Unified Integration System
module UnifiedIntegrationTests =

    [<UnitTest>]
    let ``Unified TARS Engine should initialize successfully`` () =
        let logger = createTestLogger<UnifiedTarsEngine>()
        let engine = UnifiedTarsEngine(logger)
        engine |> should not' (be null)

    [<IntegrationTest>]
    let ``Unified integration test should pass all components`` () =
        let logger = createTestLogger<UnifiedTarsEngine>()
        let engine = UnifiedTarsEngine(logger)
        let result = engine.TestUnifiedIntegration() |> Async.RunSynchronously
        result.SuccessRate |> should be (greaterThanOrEqualTo 0.8)

    [<ValidationTest>]
    let ``Unified status should show healthy integration`` () =
        let logger = createTestLogger<UnifiedTarsEngine>()
        let engine = UnifiedTarsEngine(logger)
        let status = engine.GetUnifiedStatus()
        status.IntegrationHealth |> should be (greaterThan 0.7)

    /// Run all Unified Integration tests
    let runAllTests () =
        let testMethods = [
            ("Initialize", fun () -> ``Unified TARS Engine should initialize successfully`` ())
            ("IntegrationTest", fun () -> ``Unified integration test should pass all components`` ())
            ("StatusCheck", fun () -> ``Unified status should show healthy integration`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running Unified Integration test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "Unified Integration" result
        result
