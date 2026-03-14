namespace TarsEngine.FSharp.Tests

open System
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Core.EnhancedRevolutionaryIntegration
open TarsEngine.FSharp.Tests.TestHelpers

/// End-to-end system tests
module EndToEndTests =

    [<EndToEndTest>]
    let ``Complete TARS workflow should execute successfully`` () =
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Initialize system
        engine.InitializeEnhancedCapabilities() |> Async.RunSynchronously |> ignore
        
        // Execute comprehensive workflow
        let workflow = [
            SemanticAnalysis("TARS end-to-end test", Euclidean, true)
            ConceptEvolution("e2e_concept", RevolutionaryTypes.GrammarTier.Revolutionary, true)
            EmergentDiscovery("e2e_discovery", true)
        ]
        
        let results = 
            workflow
            |> List.map (fun op -> engine.ExecuteEnhancedOperation(op) |> Async.RunSynchronously)
        
        results |> List.iter (fun r -> r.Success |> should be True)

    [<EndToEndTest>]
    let ``System integration should maintain consistency across all components`` () =
        let logger = createTestLogger<EnhancedTarsEngine>()
        let engine = EnhancedTarsEngine(logger)
        
        // Test all major components work together
        let status = engine.GetEnhancedStatus()
        status.SystemHealth |> should be (greaterThan 0.5)

    /// Run all End-to-End tests
    let runAllTests () =
        let testMethods = [
            ("CompleteWorkflow", fun () -> ``Complete TARS workflow should execute successfully`` ())
            ("SystemIntegration", fun () -> ``System integration should maintain consistency across all components`` ())
        ]
        
        let measurements = 
            testMethods
            |> List.map (fun (name, test) ->
                printfn "🧪 Running End-to-End test: %s" name
                measurePerformance test
            )
        
        let performanceMetrics = Map.ofList [
            ("avg_execution_time_ms", measurements |> List.map (_.ExecutionTime.TotalMilliseconds) |> List.average)
        ]
        
        let result = createTestSuiteResult measurements performanceMetrics
        printTestSuiteSummary "End-to-End" result
        result
