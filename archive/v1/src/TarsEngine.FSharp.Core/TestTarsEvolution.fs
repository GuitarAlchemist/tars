namespace TarsEngine.FSharp.Core

open System
open System.IO
open Microsoft.Extensions.Logging
open Microsoft.Extensions.Logging.Console
open TarsEngine.FSharp.Core.TarsEvolutionEngine
open TarsEngine.FSharp.Core.TarsPrimePattern
open TarsEngine.FSharp.Core.TarsPrimeCuda
open TarsEngine.FSharp.Core.TarsHurwitzQuaternions
open TarsEngine.FSharp.Core.Tests.TarsHurwitzQuaternionsTests
open TarsEngine.FSharp.Core.TarsRsxDiff
open TarsEngine.FSharp.Core.TarsRsxGraph
open TarsEngine.FSharp.Core.TarsSedenionPartitioner
open TarsEngine.FSharp.Core.TarsAutoReflection
open TarsEngine.FSharp.Core.TarsAdvancedFlux
open TarsEngine.FSharp.Core.TarsCliIntegration
open TarsEngine.FSharp.Core.TarsBeliefDriftVisualization
open TarsEngine.FSharp.Core.TarsExtendedPrimePatterns
open TarsEngine.FSharp.Core.TarsMetaCognitiveLoops
open TarsEngine.FSharp.Core.TarsEnhancedEvolution
open TarsEngine.FSharp.Core.TarsAutonomousCodeAnalysis
open TarsEngine.FSharp.Core.TarsAutonomousCodeModification
open TarsEngine.FSharp.Core.TarsPerformanceDrivenEvolution

/// Simple test program to verify TARS auto-improvement functionality
module TestTarsEvolution =

    /// Create a simple console logger
    let createLogger() =
        let loggerFactory = LoggerFactory.Create(fun builder -> 
            builder.AddConsole().SetMinimumLevel(LogLevel.Information) |> ignore)
        loggerFactory.CreateLogger<TarsEvolutionEngineService>()

    /// Test the evolution system
    let testEvolution() = async {
        let logger = createLogger()
        
        printfn "🚀 Testing TARS Auto-Improvement System"
        printfn "========================================"
        
        try
            // Test 1: Quick Evolution Check
            printfn "\n🔍 Test 1: Quick Evolution Check"
            let! ready = EvolutionHelpers.isEvolutionReady Environment.CurrentDirectory logger
            printfn $"Evolution Ready: {ready}"
            
            if ready then
                // Test 2: Run Quick Evolution
                printfn "\n⚡ Test 2: Running Quick Evolution Session"
                let! evolutionResult = EvolutionHelpers.quickEvolution Environment.CurrentDirectory logger
                
                printfn $"Evolution Session ID: {evolutionResult.SessionId}"
                printfn $"Overall Success: {evolutionResult.OverallSuccess}"
                printfn $"Projects Analyzed: {evolutionResult.ProjectsAnalyzed}"
                printfn $"Improvements Applied: {evolutionResult.ImprovementsApplied}"
                printfn $"Duration: {evolutionResult.TotalDurationMs}ms"
                
                if evolutionResult.PerformanceGain.IsSome then
                    printfn $"Performance Gain: {evolutionResult.PerformanceGain.Value}%%"
                
                printfn "\nRecommended Next Steps:"
                for step in evolutionResult.RecommendedNextSteps do
                    printfn $"  • {step}"
                
                printfn "\nEvolution Steps:"
                for step in evolutionResult.Steps do
                    let status = if step.Success then "✅" else "❌"
                    printfn $"  {status} {step.StepName} ({step.ExecutionTimeMs}ms)"
                    if not step.Success && step.ErrorMessage.IsSome then
                        printfn $"      Error: {step.ErrorMessage.Value}"
            else
                printfn "❌ Evolution system not ready. Cannot run full test."
            
            // Test 3: Prime Pattern Integration
            printfn "\n🔢 Test 3: Prime Pattern Integration"
            let primeTestResult = testPrimeSystem logger
            let primeStatus = if primeTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Prime Pattern Test: {primeStatus}"

            // Test 4: CUDA Integration (if available)
            printfn "\n🚀 Test 4: CUDA Integration Test"
            let cudaTestResult = testCudaIntegration logger
            let cudaStatus = if cudaTestResult then "✅ PASSED" else "⚠️ SKIPPED/FAILED"
            printfn $"CUDA Integration Test: {cudaStatus}"

            // Test 5: Hurwitz Quaternions
            printfn "\n🔢 Test 5: Hurwitz Quaternions Test"
            let quaternionTestResult = runAllTests logger
            let quaternionStatus = if quaternionTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Hurwitz Quaternions Test: {quaternionStatus}"

            // Test 6: TRSX Diff Engine
            printfn "\n🔄 Test 6: TRSX Diff Engine Test"
            let trsxDiffTestResult = testTrsxDiff logger
            let trsxDiffStatus = if trsxDiffTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"TRSX Diff Engine Test: {trsxDiffStatus}"

            // Test 7: TRSX Hypergraph
            printfn "\n🕸️ Test 7: TRSX Hypergraph Test"
            let trsxGraphTestResult = testTrsxGraph logger
            let trsxGraphStatus = if trsxGraphTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"TRSX Hypergraph Test: {trsxGraphStatus}"

            // Test 8: Sedenion Partitioner
            printfn "\n🌌 Test 8: Sedenion Partitioner Test"
            let sedenionTestResult = testSedenionPartitioning logger
            let sedenionStatus = if sedenionTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Sedenion Partitioner Test: {sedenionStatus}"

            // Test 9: Auto-Reflection System
            printfn "\n🧠 Test 9: Auto-Reflection System Test"
            let reflectionTestResult = testAutoReflection logger
            let reflectionStatus = if reflectionTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Auto-Reflection System Test: {reflectionStatus}"

            // Test 10: Advanced FLUX Integration
            printfn "\n🌊 Test 10: Advanced FLUX Integration Test"
            let fluxTestResult = testAdvancedFlux logger
            let fluxStatus = if fluxTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Advanced FLUX Integration Test: {fluxStatus}"

            // Test 11: CLI Integration
            printfn "\n💻 Test 11: CLI Integration Test"
            let cliTestResult = testCliIntegration logger
            let cliStatus = if cliTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"CLI Integration Test: {cliStatus}"

            // Test 12: Belief Drift Visualization
            printfn "\n📊 Test 12: Belief Drift Visualization Test"
            let beliefDriftTestResult = testBeliefDriftVisualization logger
            let beliefDriftStatus = if beliefDriftTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Belief Drift Visualization Test: {beliefDriftStatus}"

            // Test 13: Extended Prime Patterns
            printfn "\n🔢 Test 13: Extended Prime Patterns Test"
            let extendedPrimeTestResult = testExtendedPrimePatterns logger
            let extendedPrimeStatus = if extendedPrimeTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Extended Prime Patterns Test: {extendedPrimeStatus}"

            // Test 14: Meta-Cognitive Loops
            printfn "\n🧠 Test 14: Meta-Cognitive Loops Test"
            let metaCognitiveTestResult = testMetaCognitiveLoops logger
            let metaCognitiveStatus = if metaCognitiveTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Meta-Cognitive Loops Test: {metaCognitiveStatus}"

            // Test 15: Enhanced Evolution Integration
            printfn "\n🚀 Test 15: Enhanced Evolution Integration Test"
            let enhancedEvolutionTestResult = testEnhancedEvolution logger
            let enhancedEvolutionStatus = if enhancedEvolutionTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Enhanced Evolution Integration Test: {enhancedEvolutionStatus}"

            // Test 16: Autonomous Code Analysis
            printfn "\n🧠 Test 16: Autonomous Code Analysis Test"
            let codeAnalysisTestResult = testAutonomousCodeAnalysis logger
            let codeAnalysisStatus = if codeAnalysisTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Autonomous Code Analysis Test: {codeAnalysisStatus}"

            // Test 17: Autonomous Code Modification
            printfn "\n🔧 Test 17: Autonomous Code Modification Test"
            let codeModificationTestResult = testAutonomousCodeModification logger
            let codeModificationStatus = if codeModificationTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Autonomous Code Modification Test: {codeModificationStatus}"

            // Test 18: Performance-Driven Evolution
            printfn "\n📊 Test 18: Performance-Driven Evolution Test"
            let performanceEvolutionTestResult = testPerformanceDrivenEvolution logger
            let performanceEvolutionStatus = if performanceEvolutionTestResult then "✅ PASSED" else "❌ FAILED"
            printfn $"Performance-Driven Evolution Test: {performanceEvolutionStatus}"

            // REAL AUTONOMOUS EVOLUTION DEMONSTRATION
            printfn "\n🌟 REAL AUTONOMOUS TARS EVOLUTION DEMONSTRATION"
            printfn "=============================================="

            if codeAnalysisTestResult && codeModificationTestResult && performanceEvolutionTestResult then
                printfn "🚀 Running REAL autonomous TARS evolution with actual code modifications..."

                let evolutionResult = executePerformanceDrivenEvolution logger

                printfn "\n🎉 REAL AUTONOMOUS EVOLUTION RESULTS:"
                printfn $"   Session ID: {evolutionResult.SessionId}"
                printfn $"   Total Improvements Identified: {evolutionResult.TotalImprovements}"
                printfn $"   Successful Code Modifications: {evolutionResult.SuccessfulModifications}"
                printfn $"   Failed Modifications: {evolutionResult.FailedModifications}"
                printfn $"   Overall Performance Gain: {evolutionResult.OverallPerformanceGain:F2}%%"
                printfn $"   Evolution Duration: {evolutionResult.ExecutionTimeMs:F0}ms"

                printfn "\n📊 Performance Benchmark Results:"
                for benchmark in evolutionResult.BenchmarkResults do
                    match benchmark.CurrentMeasurement with
                    | Some current ->
                        printfn $"   🏁 {benchmark.Name}:"
                        printfn $"      Baseline: {benchmark.BaselineMeasurement.ExecutionTimeMs:F2}ms"
                        printfn $"      Current:  {current.ExecutionTimeMs:F2}ms"
                        printfn $"      Improvement: {benchmark.ImprovementPercentage:F1}%%"
                    | None ->
                        printfn $"   🏁 {benchmark.Name}: Baseline only ({benchmark.BaselineMeasurement.ExecutionTimeMs:F2}ms)"

                printfn "\n💡 Evolution Insights:"
                for step in evolutionResult.RecommendedNextSteps |> List.take 3 do
                    printfn $"   • {step}"

                if evolutionResult.SuccessfulModifications > 0 then
                    printfn "\n🌟 TARS has successfully modified its own code autonomously!"
                    printfn "   Real autonomous evolution achieved with measurable improvements."
                else
                    printfn "\n⚠️ TARS evolution completed but no code modifications were applied."
            else
                printfn "❌ Autonomous evolution prerequisites not met - skipping real evolution"

            // Enhanced Evolution Demonstration (Previous)
            printfn "\n🌟 COMPREHENSIVE TARS AUTO-EVOLUTION DEMONSTRATION"
            printfn "=================================================="

            if enhancedEvolutionTestResult then
                printfn "🚀 Running full TARS auto-evolution with all cognitive capabilities..."

                let config = createDefaultEnhancedConfig()
                let enhancedConfig = { config with MaxEvolutionCycles = 2; PrimeAnalysisLimit = 25000L }

                let evolutionResult = runEnhancedEvolution enhancedConfig logger

                printfn "\n🎉 TARS AUTO-EVOLUTION RESULTS:"
                printfn $"   Overall Success: {evolutionResult.BaseResult.OverallSuccess}"
                printfn $"   Cognitive Growth: {evolutionResult.OverallCognitiveGrowth:F2}"
                printfn $"   Prime Pattern Insights: {evolutionResult.PrimePatternInsights.Length}"
                printfn $"   Meta-Cognitive Insights: {evolutionResult.MetaCognitiveInsights.Length}"
                printfn $"   Emergent Patterns: {evolutionResult.EmergentPatterns.Length}"

                if evolutionResult.BeliefDriftAnalysis.IsSome then
                    let timeline = evolutionResult.BeliefDriftAnalysis.Value.Timeline
                    printfn $"   Belief States Tracked: {timeline.States.Length}"
                    printfn $"   Belief Drift Magnitude: {timeline.TotalDriftMagnitude:F3}"
                    printfn $"   Overall Direction: {timeline.OverallDirection}"

                printfn "\n💡 Key Insights Generated:"
                for insight in evolutionResult.PrimePatternInsights |> List.take (min 3 evolutionResult.PrimePatternInsights.Length) do
                    printfn $"   🔢 {insight}"

                for insight in evolutionResult.MetaCognitiveInsights |> List.take (min 3 evolutionResult.MetaCognitiveInsights.Length) do
                    printfn $"   🧠 {insight}"

                printfn "\n📋 Recommended Next Steps:"
                for step in evolutionResult.RecommendedNextSteps |> List.take 3 do
                    printfn $"   • {step}"

                if evolutionResult.BaseResult.OverallSuccess then
                    printfn "\n🌟 TARS has successfully demonstrated comprehensive auto-evolution!"
                    printfn "   All cognitive systems are integrated and functioning optimally."
                else
                    printfn "\n⚠️ TARS evolution completed with some limitations."
            else
                printfn "❌ Enhanced evolution test failed - skipping comprehensive demonstration"

            printfn "\n🎉 TARS Auto-Improvement Test Completed!"

        with
        | ex ->
            printfn $"❌ Test failed: {ex.Message}"
            printfn $"Stack trace: {ex.StackTrace}"
    }

    /// Entry point for testing
    let runTest() =
        testEvolution() |> Async.RunSynchronously
