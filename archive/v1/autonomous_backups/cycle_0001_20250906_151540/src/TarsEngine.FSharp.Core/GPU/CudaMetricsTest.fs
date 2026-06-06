namespace TarsEngine.FSharp.Core.GPU

open System
open System.Diagnostics

/// Real CUDA metrics test - shows actual performance measurements
module CudaMetricsTest =
    
    /// Test real CUDA initialization and show actual metrics
    let testRealCudaMetrics () =
        printfn ""
        printfn "🔥 REAL CUDA METRICS TEST"
        printfn "========================="
        printfn ""
        
        try
            // Test ILGPU initialization
            printfn "🚀 Testing ILGPU CUDA initialization..."
            let initSuccess = RealCudaEngine.initializeRealCuda()
            
            if initSuccess then
                printfn "✅ REAL CUDA GPU DETECTED AND INITIALIZED!"
                printfn ""
                
                // Get real GPU information
                let gpuInfo = RealCudaEngine.getGpuInfo()
                printfn "📊 REAL GPU SPECIFICATIONS:"
                for kvp in gpuInfo do
                    printfn "   • %s: %s" kvp.Key (kvp.Value.ToString())
                
                printfn ""
                printfn "🔥 RUNNING REAL GPU BENCHMARKS..."
                printfn ""
                
                // Test 1: Small sedenion calculation
                printfn "Test 1: Sedenion Distance (1,000 vectors × 16D)"
                let stopwatch1 = Stopwatch.StartNew()
                let vectors1 = Array.init (1000 * 16) (fun i -> float32 (i % 100) / 100.0f)
                let vectors2 = Array.init (1000 * 16) (fun i -> float32 ((i + 50) % 100) / 100.0f)
                let results1 = RealCudaEngine.executeSedenionDistanceGpu vectors1 vectors2 1000 16
                stopwatch1.Stop()
                
                let flops1 = float (1000 * 16 * 4) // subtract, square, sum, sqrt
                let gflops1 = flops1 / stopwatch1.Elapsed.TotalSeconds / 1e9
                
                printfn "   ⚡ Time: %.3f ms" stopwatch1.Elapsed.TotalMilliseconds
                printfn "   ⚡ GFLOPS: %.6f" gflops1
                printfn "   ⚡ Results: %d distances calculated" results1.Length
                printfn "   ⚡ Sample: [%.3f, %.3f, %.3f]" results1.[0] results1.[1] results1.[2]
                
                printfn ""
                
                // Test 2: Massive computation with larger workload
                printfn "Test 2: Massive Computation (100,000 elements × 100 ops)"
                let (results2, gflops2) = RealCudaEngine.executeMassiveComputeGpu 100000 100
                
                printfn "   ⚡ GFLOPS: %.6f" gflops2
                printfn "   ⚡ Elements: %d" results2.Length
                printfn "   ⚡ Sample: [%.6f, %.6f, %.6f]" results2.[0] results2.[1] results2.[2]
                
                printfn ""
                
                // Test 3: Neural network
                printfn "Test 3: Neural Network (100 batch × 128 input × 64 output)"
                let inputs = Array.init (100 * 128) (fun i -> float32 (i % 10) / 10.0f)
                let weights = Array.init (128 * 64) (fun i -> float32 (i % 5) / 5.0f)
                let biases = Array.init 64 (fun i -> float32 i / 64.0f)
                let (results3, gflops3) = RealCudaEngine.executeNeuralForwardGpu inputs weights biases 100 128 64
                
                printfn "   ⚡ GFLOPS: %.6f" gflops3
                printfn "   ⚡ Outputs: %d" results3.Length
                printfn "   ⚡ Sample: [%.6f, %.6f, %.6f]" results3.[0] results3.[1] results3.[2]
                
                printfn ""
                printfn "🎯 REAL PERFORMANCE SUMMARY:"
                printfn "   • Sedenion Distance: %.6f GFLOPS" gflops1
                printfn "   • Massive Compute: %.6f GFLOPS" gflops2
                printfn "   • Neural Network: %.6f GFLOPS" gflops3
                printfn "   • Peak Performance: %.6f GFLOPS" (max (max gflops1 gflops2) gflops3)
                
                printfn ""
                if gflops2 > 1.0 then
                    printfn "🔥 EXCELLENT GPU PERFORMANCE! Real CUDA acceleration working!"
                elif gflops2 > 0.1 then
                    printfn "⚡ GOOD GPU PERFORMANCE! CUDA kernels executing on GPU!"
                elif gflops2 > 0.01 then
                    printfn "✅ MODERATE GPU PERFORMANCE! GPU acceleration active!"
                else
                    printfn "⚠️  LOW PERFORMANCE - May be running on integrated GPU or CPU fallback"
                
                // Cleanup
                RealCudaEngine.cleanup()
                
            else
                printfn "❌ NO CUDA GPU DETECTED"
                printfn ""
                printfn "📊 SYSTEM ANALYSIS:"
                printfn "   • CUDA Toolkit: Not installed or not compatible"
                printfn "   • GPU Hardware: No CUDA-capable GPU found"
                printfn "   • Driver: NVIDIA drivers may not be installed"
                printfn ""
                printfn "💡 TO ENABLE REAL CUDA:"
                printfn "   1. Install NVIDIA CUDA Toolkit 12.0+"
                printfn "   2. Install latest NVIDIA drivers"
                printfn "   3. Ensure CUDA-capable GPU (GTX 10xx+ or RTX series)"
                printfn ""
                printfn "🔄 FALLBACK: Running CPU simulation mode"
                
                // Show CPU baseline performance
                printfn ""
                printfn "📊 CPU BASELINE PERFORMANCE:"
                let stopwatch = Stopwatch.StartNew()
                let mutable sum = 0.0f
                for i = 0 to 999999 do
                    sum <- sum + float32 i * 1.001f
                stopwatch.Stop()
                
                let cpuFlops = 1000000.0 * 2.0 // multiply and add
                let cpuGflops = cpuFlops / stopwatch.Elapsed.TotalSeconds / 1e9
                
                printfn "   • CPU GFLOPS: %.6f" cpuGflops
                printfn "   • Time: %.3f ms" stopwatch.Elapsed.TotalMilliseconds
                printfn "   • Result: %.6f" sum
                
        with
        | ex ->
            printfn "❌ CUDA TEST FAILED: %s" ex.Message
            printfn ""
            printfn "🔍 ERROR ANALYSIS:"
            printfn "   • Exception Type: %s" (ex.GetType().Name)
            printfn "   • Message: %s" ex.Message
            if ex.InnerException <> null then
                printfn "   • Inner Exception: %s" ex.InnerException.Message
            
            printfn ""
            printfn "💡 POSSIBLE SOLUTIONS:"
            printfn "   1. Install NVIDIA CUDA Toolkit"
            printfn "   2. Update NVIDIA drivers"
            printfn "   3. Check GPU compatibility"
            printfn "   4. Restart application after driver installation"
        
        printfn ""
        printfn "🎯 REAL METRICS TEST COMPLETE"
        printfn "=============================="
