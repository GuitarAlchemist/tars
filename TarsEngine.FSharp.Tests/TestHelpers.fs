namespace TarsEngine.FSharp.Tests

open System
open System.Diagnostics
open Microsoft.Extensions.Logging
open Xunit
open FsUnit.Xunit

/// Test utilities and helpers for TARS testing
module TestHelpers =

    /// Create a test logger
    let createTestLogger<'T>() =
        LoggerFactory.Create(fun builder -> 
            builder.AddConsole().SetMinimumLevel(LogLevel.Information) |> ignore
        ).CreateLogger<'T>()

    /// Performance measurement helper
    type PerformanceMeasurement = {
        ExecutionTime: TimeSpan
        MemoryBefore: int64
        MemoryAfter: int64
        MemoryUsed: int64
        Success: bool
        Result: obj option
    }

    /// Measure performance of an operation
    let measurePerformance<'T> (operation: unit -> 'T) : PerformanceMeasurement =
        GC.Collect()
        GC.WaitForPendingFinalizers()
        GC.Collect()
        
        let memoryBefore = GC.GetTotalMemory(false)
        let stopwatch = Stopwatch.StartNew()
        
        try
            let result = operation()
            stopwatch.Stop()
            let memoryAfter = GC.GetTotalMemory(false)
            
            {
                ExecutionTime = stopwatch.Elapsed
                MemoryBefore = memoryBefore
                MemoryAfter = memoryAfter
                MemoryUsed = memoryAfter - memoryBefore
                Success = true
                Result = Some (box result)
            }
        with
        | ex ->
            stopwatch.Stop()
            let memoryAfter = GC.GetTotalMemory(false)
            
            {
                ExecutionTime = stopwatch.Elapsed
                MemoryBefore = memoryBefore
                MemoryAfter = memoryAfter
                MemoryUsed = memoryAfter - memoryBefore
                Success = false
                Result = None
            }

    /// Async performance measurement
    let measurePerformanceAsync<'T> (operation: unit -> Async<'T>) : Async<PerformanceMeasurement> =
        async {
            GC.Collect()
            GC.WaitForPendingFinalizers()
            GC.Collect()
            
            let memoryBefore = GC.GetTotalMemory(false)
            let stopwatch = Stopwatch.StartNew()
            
            try
                let! result = operation()
                stopwatch.Stop()
                let memoryAfter = GC.GetTotalMemory(false)
                
                return {
                    ExecutionTime = stopwatch.Elapsed
                    MemoryBefore = memoryBefore
                    MemoryAfter = memoryAfter
                    MemoryUsed = memoryAfter - memoryBefore
                    Success = true
                    Result = Some (box result)
                }
            with
            | ex ->
                stopwatch.Stop()
                let memoryAfter = GC.GetTotalMemory(false)
                
                return {
                    ExecutionTime = stopwatch.Elapsed
                    MemoryBefore = memoryBefore
                    MemoryAfter = memoryAfter
                    MemoryUsed = memoryAfter - memoryBefore
                    Success = false
                    Result = None
                }
        }

    /// Test data generators
    module TestData =
        
        /// Generate random float array
        let generateFloatArray (size: int) (range: float * float) : float[] =
            let random = Random()
            let (min, max) = range
            Array.init size (fun _ -> min + random.NextDouble() * (max - min))
        
        /// Generate test text samples
        let generateTestTexts (count: int) : string[] =
            [|
                "TARS revolutionary AI system with autonomous capabilities"
                "Multi-space geometric embeddings for enhanced understanding"
                "CUDA-accelerated vector operations with hybrid transformers"
                "Fractal grammar progression through complexity tiers"
                "Emergent discovery patterns in non-Euclidean spaces"
                "Consciousness integration with self-aware reasoning"
                "Dynamic UI generation with autonomous interface evolution"
                "Scientific research applications with Janus cosmology"
                "Drug discovery optimization with financial risk modeling"
                "Real-time performance monitoring with comprehensive diagnostics"
            |]
            |> Array.take (min count 10)
            |> fun arr -> if count > 10 then Array.append arr (Array.replicate (count - 10) arr.[0]) else arr

        /// Generate test concepts for evolution
        let generateTestConcepts (count: int) : string[] =
            [|
                "autonomous_reasoning"
                "multi_space_embeddings"
                "revolutionary_capabilities"
                "emergent_intelligence"
                "consciousness_integration"
                "dynamic_adaptation"
                "performance_optimization"
                "scientific_discovery"
                "pattern_recognition"
                "conceptual_breakthrough"
            |]
            |> Array.take (min count 10)
            |> fun arr -> if count > 10 then Array.append arr (Array.replicate (count - 10) arr.[0]) else arr

    /// Validation helpers
    module Validation =
        
        /// Validate performance gain
        let validatePerformanceGain (gain: float option) (expectedMin: float) =
            match gain with
            | Some g -> g |> should be (greaterThan expectedMin)
            | None -> failwith "Performance gain should not be None"
        
        /// Validate execution time
        let validateExecutionTime (time: TimeSpan) (maxTime: TimeSpan) =
            time |> should be (lessThan maxTime)
        
        /// Validate memory usage
        let validateMemoryUsage (memoryUsed: int64) (maxMemory: int64) =
            memoryUsed |> should be (lessThan maxMemory)
        
        /// Validate success rate
        let validateSuccessRate (successCount: int) (totalCount: int) (minRate: float) =
            let rate = float successCount / float totalCount
            rate |> should be (greaterThanOrEqualTo minRate)
        
        /// Validate array similarity
        let validateArraySimilarity (arr1: float[]) (arr2: float[]) (tolerance: float) =
            arr1.Length |> should equal arr2.Length
            Array.zip arr1 arr2
            |> Array.iter (fun (a, b) -> abs(a - b) |> should be (lessThan tolerance))

    /// Test categories and attributes
    [<AttributeUsage(AttributeTargets.Method)>]
    type UnitTestAttribute() = inherit FactAttribute()
    
    [<AttributeUsage(AttributeTargets.Method)>]
    type IntegrationTestAttribute() = inherit FactAttribute()
    
    [<AttributeUsage(AttributeTargets.Method)>]
    type PerformanceTestAttribute() = inherit FactAttribute()
    
    [<AttributeUsage(AttributeTargets.Method)>]
    type ValidationTestAttribute() = inherit FactAttribute()
    
    [<AttributeUsage(AttributeTargets.Method)>]
    type EndToEndTestAttribute() = inherit FactAttribute()

    /// Test result aggregation
    type TestSuiteResult = {
        TotalTests: int
        PassedTests: int
        FailedTests: int
        SkippedTests: int
        TotalExecutionTime: TimeSpan
        AverageExecutionTime: TimeSpan
        SuccessRate: float
        PerformanceMetrics: Map<string, float>
    }

    /// Create test suite result
    let createTestSuiteResult (results: PerformanceMeasurement list) (performanceMetrics: Map<string, float>) =
        let totalTests = results.Length
        let passedTests = results |> List.filter (_.Success) |> List.length
        let failedTests = totalTests - passedTests
        let totalTime = results |> List.map (_.ExecutionTime) |> List.fold (+) TimeSpan.Zero
        let avgTime = if totalTests > 0 then TimeSpan.FromTicks(totalTime.Ticks / int64 totalTests) else TimeSpan.Zero
        let successRate = if totalTests > 0 then float passedTests / float totalTests else 0.0
        
        {
            TotalTests = totalTests
            PassedTests = passedTests
            FailedTests = failedTests
            SkippedTests = 0
            TotalExecutionTime = totalTime
            AverageExecutionTime = avgTime
            SuccessRate = successRate
            PerformanceMetrics = performanceMetrics
        }

    /// Print test suite summary
    let printTestSuiteSummary (suiteName: string) (result: TestSuiteResult) =
        printfn ""
        printfn "🧪 %s Test Suite Results:" suiteName
        printfn "================================"
        printfn "📊 Total Tests: %d" result.TotalTests
        printfn "✅ Passed: %d" result.PassedTests
        printfn "❌ Failed: %d" result.FailedTests
        printfn "⏱️  Total Time: %A" result.TotalExecutionTime
        printfn "⚡ Average Time: %A" result.AverageExecutionTime
        printfn "📈 Success Rate: %.1f%%" (result.SuccessRate * 100.0)
        
        if not result.PerformanceMetrics.IsEmpty then
            printfn "🚀 Performance Metrics:"
            result.PerformanceMetrics
            |> Map.iter (fun key value -> printfn "   - %s: %.2fx" key value)
        
        printfn ""
