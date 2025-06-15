namespace TarsEngine.FSharp.FLUX.Tests.Performance

open System
open System.Diagnostics
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open NBomber.FSharp
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Performance tests for FLUX using NBomber
module FluxPerformanceTests =
    
    [<Fact>]
    let ``FLUX Engine executes simple script within performance bounds`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let simpleScript = """
META {
    title: "Performance Test"
    version: "1.0.0"
}

FSHARP {
    let x = 42
    let y = x * 2
    printfn "Result: %d" y
}
"""
            let maxExecutionTime = TimeSpan.FromSeconds(5.0)
            
            // Act
            let (result, executionTime) = TestHelpers.measureExecutionTime (fun () ->
                task {
                    return! engine.ExecuteString(simpleScript)
                } |> Async.AwaitTask |> Async.RunSynchronously
            )
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            executionTime |> should be (lessThan maxExecutionTime)
            printfn "Simple script execution time: %A" executionTime
        }
    
    [<Fact>]
    let ``FLUX Engine handles multiple blocks efficiently`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let blockCount = 10
            let script = TestHelpers.createPerformanceTestData blockCount
            let maxExecutionTime = TimeSpan.FromSeconds(10.0)
            
            // Act
            let (result, executionTime) = TestHelpers.measureExecutionTime (fun () ->
                task {
                    return! TestHelpers.executeFluxScript script
                } |> Async.AwaitTask |> Async.RunSynchronously
            )
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal (blockCount + 1) // +1 for META block
            executionTime |> should be (lessThan maxExecutionTime)
            printfn "Multiple blocks (%d) execution time: %A" blockCount executionTime
        }
    
    [<Fact>]
    let ``FLUX Engine parsing performance is acceptable`` () =
        // Arrange
        let engine = FluxEngine()
        let largeScript = String.replicate 100 """
FSHARP {
    let x = 42
    printfn "Value: %d" x
}
"""
        let maxParseTime = TimeSpan.FromSeconds(2.0)
        
        // Act
        let (result, parseTime) = TestHelpers.measureExecutionTime (fun () ->
            engine.ParseScript(largeScript)
        )
        
        // Assert
        match result with
        | Ok script -> script.Blocks |> should not' (be Empty)
        | Error msg -> failwith (sprintf "Parse failed: %s" msg)
        
        parseTime |> should be (lessThan maxParseTime)
        printfn "Large script parsing time: %A" parseTime
    
    [<Fact>]
    let ``FLUX Engine memory usage is reasonable`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = TestHelpers.createTestFluxFileContent()
            
            // Measure initial memory
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            let initialMemory = GC.GetTotalMemory(false)
            
            // Act - Execute script multiple times
            for i in 1..10 do
                let! _ = engine.ExecuteString(script)
                ()
            
            // Measure final memory
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            let finalMemory = GC.GetTotalMemory(false)
            
            // Assert
            let memoryIncrease = finalMemory - initialMemory
            let maxMemoryIncrease = 50L * 1024L * 1024L // 50 MB
            
            memoryIncrease |> should be (lessThan maxMemoryIncrease)
            printfn "Memory increase: %d bytes (%.2f MB)" memoryIncrease (float memoryIncrease / (1024.0 * 1024.0))
        }
    
    [<Fact>]
    let ``FLUX Engine concurrent execution performance`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = TestHelpers.createTestFluxFileContent()
            let concurrentTasks = 5
            let maxTotalTime = TimeSpan.FromSeconds(15.0)
            
            // Act
            let (results, totalTime) = TestHelpers.measureExecutionTime (fun () ->
                let tasks = 
                    [1..concurrentTasks] 
                    |> List.map (fun _ -> engine.ExecuteString(script))
                    |> Array.ofList
                
                Task.WhenAll(tasks) |> Async.AwaitTask |> Async.RunSynchronously
            )
            
            // Assert
            results |> Array.iter TestHelpers.assertExecutionSuccess
            totalTime |> should be (lessThan maxTotalTime)
            printfn "Concurrent execution (%d tasks) time: %A" concurrentTasks totalTime
        }
    
    /// NBomber load test for FLUX execution
    [<Fact(Skip = "Long running performance test - enable manually")>]
    let ``FLUX Engine load test with NBomber`` () =
        // Arrange
        let engine = FluxEngine()
        let testScript = TestHelpers.createTestFluxFileContent()
        
        let scenario = 
            Scenario.create "flux_execution" (fun context ->
                task {
                    try
                        let! result = engine.ExecuteString(testScript)
                        return if result.Success then Response.ok() else Response.fail()
                    with
                    | ex -> 
                        return Response.fail(ex.Message)
                }
            )
            |> Scenario.withLoadSimulations [
                InjectPerSec(rate = 10, during = TimeSpan.FromMinutes(1.0));
                KeepConstant(copies = 5, during = TimeSpan.FromMinutes(2.0))
            ]
        
        // Act & Assert
        let stats =
            NBomberRunner
                .registerScenarios([scenario])
                .run()
        
        // Verify performance metrics
        let fluxScenarioStats = stats.AllScenarioStats |> Array.find (fun s -> s.ScenarioName = "flux_execution")
        fluxScenarioStats.Ok.Request.Mean |> should be (lessThan 1000.0) // Less than 1 second mean response time
        fluxScenarioStats.Fail.Request.Count |> should be (lessThan 10) // Less than 10 failures
    
    /// Benchmark different script sizes
    [<Fact>]
    let ``FLUX Engine performance scales linearly with script size`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let scriptSizes = [1; 5; 10; 20]
            let results = ResizeArray<int * TimeSpan>()
            
            // Act
            for size in scriptSizes do
                let script = TestHelpers.createPerformanceTestData size
                let (result, executionTime) = TestHelpers.measureExecutionTime (fun () ->
                    task {
                        return! TestHelpers.executeFluxScript script
                    } |> Async.AwaitTask |> Async.RunSynchronously
                )
                
                TestHelpers.assertExecutionSuccess result
                results.Add(size, executionTime)
                printfn "Script size %d blocks: %A" size executionTime
            
            // Assert - Check that execution time doesn't grow exponentially
            let timings = results |> Seq.map snd |> Seq.toList
            let maxTime = timings |> List.max
            let minTime = timings |> List.min
            let ratio = maxTime.TotalMilliseconds / minTime.TotalMilliseconds
            
            ratio |> should be (lessThan 10.0) // Should not be more than 10x slower for larger scripts
        }
    
    /// Memory leak detection test
    [<Fact>]
    let ``FLUX Engine does not leak memory over multiple executions`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = TestHelpers.createTestFluxFileContent()
            let executionCount = 20
            
            // Baseline memory measurement
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            let baselineMemory = GC.GetTotalMemory(false)
            
            // Act - Execute script multiple times
            for i in 1..executionCount do
                let! result = engine.ExecuteString(script)
                TestHelpers.assertExecutionSuccess result
                
                // Force garbage collection every 5 executions
                if i % 5 = 0 then
                    GC.Collect()
                    GC.WaitForPendingFinalizers()
            
            // Final memory measurement
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            let finalMemory = GC.GetTotalMemory(false)
            
            // Assert
            let memoryGrowth = finalMemory - baselineMemory
            let maxAcceptableGrowth = 20L * 1024L * 1024L // 20 MB
            
            memoryGrowth |> should be (lessThan maxAcceptableGrowth)
            printfn "Memory growth after %d executions: %d bytes (%.2f MB)" 
                executionCount memoryGrowth (float memoryGrowth / (1024.0 * 1024.0))
        }
    
    /// CPU usage monitoring test
    [<Fact>]
    let ``FLUX Engine CPU usage is reasonable`` () =
        task {
            // Arrange
            let engine = FluxEngine()
            let script = TestHelpers.createTestFluxFileContent()
            let currentProcess = Process.GetCurrentProcess()

            // Measure CPU time before
            let startCpuTime = currentProcess.TotalProcessorTime
            let startTime = DateTime.UtcNow

            // Act - Execute script multiple times
            for i in 1..10 do
                let! result = engine.ExecuteString(script)
                TestHelpers.assertExecutionSuccess result

            // Measure CPU time after
            let endCpuTime = currentProcess.TotalProcessorTime
            let endTime = DateTime.UtcNow

            // Calculate CPU usage percentage
            let cpuUsed = endCpuTime - startCpuTime
            let totalTime = endTime - startTime
            let cpuUsagePercent = (cpuUsed.TotalMilliseconds / totalTime.TotalMilliseconds) * 100.0
            
            // Assert
            cpuUsagePercent |> should be (lessThan 80.0) // Should not use more than 80% CPU
            printfn "CPU usage: %.2f%%" cpuUsagePercent
        }

printfn "🚀 FLUX Performance Tests Loaded"
printfn "================================="
printfn "✅ Execution time benchmarks"
printfn "✅ Memory usage tests"
printfn "✅ Concurrent execution tests"
printfn "✅ Load testing with NBomber"
printfn "✅ Scalability tests"
printfn "✅ Memory leak detection"
printfn "✅ CPU usage monitoring"
