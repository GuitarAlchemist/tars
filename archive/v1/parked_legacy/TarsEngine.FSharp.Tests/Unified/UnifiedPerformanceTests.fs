module TarsEngine.FSharp.Tests.Unified.UnifiedPerformanceTests

open System
open System.Diagnostics
open System.Threading
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.Cli.Core.UnifiedCore
open TarsEngine.FSharp.Cli.Core.UnifiedLogger
open TarsEngine.FSharp.Cli.Configuration.UnifiedConfigurationManager

/// Performance tests for unified systems
[<TestClass>]
type UnifiedPerformanceTests() =
    
    let createTestLogger() = createLogger "UnifiedPerformanceTests"
    
    [<Fact>]
    let ``Configuration access should be fast`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            let! _ = configManager.SetValueAsync("perf.test", "value", None)
            
            // Act - Measure configuration access performance
            let stopwatch = Stopwatch.StartNew()
            let iterations = 1000
            
            for i in 1..iterations do
                let _ = ConfigurationExtensions.getString configManager "perf.test" "default"
                ()
            
            stopwatch.Stop()
            
            // Assert - Should be very fast (< 1ms per operation)
            let avgTimeMs = stopwatch.ElapsedMilliseconds / int64 iterations
            avgTimeMs |> should be (lessThan 1L)
        }
    
    [<Fact>]
    let ``Correlation ID generation should be fast`` () =
        // Act - Measure correlation ID generation performance
        let stopwatch = Stopwatch.StartNew()
        let iterations = 10000
        
        for i in 1..iterations do
            let _ = generateCorrelationId()
            ()
        
        stopwatch.Stop()
        
        // Assert - Should be very fast
        let avgTimeMicroseconds = (stopwatch.ElapsedTicks * 1000000L) / (int64 iterations * Stopwatch.Frequency)
        avgTimeMicroseconds |> should be (lessThan 100L) // Less than 100 microseconds per ID
    
    [<Fact>]
    let ``Error handling should not impact performance significantly`` () =
        // Arrange
        let iterations = 1000
        
        // Act - Measure error creation performance
        let stopwatch = Stopwatch.StartNew()
        
        for i in 1..iterations do
            let _ = TarsError.create "TestError" $"Error {i}" None
            ()
        
        stopwatch.Stop()
        
        // Assert - Should be reasonably fast
        let avgTimeMs = stopwatch.ElapsedMilliseconds / int64 iterations
        avgTimeMs |> should be (lessThan 1L)
    
    [<Fact>]
    let ``Concurrent operations should scale well`` () =
        task {
            // Arrange
            use configManager = createConfigurationManager (createTestLogger())
            let! _ = configManager.InitializeAsync(CancellationToken.None)
            
            // Act - Measure concurrent configuration access
            let stopwatch = Stopwatch.StartNew()
            let concurrentTasks = 10
            let operationsPerTask = 100
            
            let tasks = [
                for i in 1..concurrentTasks do
                    yield task {
                        for j in 1..operationsPerTask do
                            let _ = ConfigurationExtensions.getString configManager "test.concurrent" "default"
                            ()
                    }
            ]
            
            do! Task.WhenAll(tasks)
            stopwatch.Stop()
            
            // Assert - Concurrent access should not be significantly slower
            let totalOperations = concurrentTasks * operationsPerTask
            let avgTimeMs = stopwatch.ElapsedMilliseconds / int64 totalOperations
            avgTimeMs |> should be (lessThan 2L) // Allow some overhead for concurrency
        }
