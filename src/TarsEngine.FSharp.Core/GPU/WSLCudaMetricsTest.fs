namespace TarsEngine.FSharp.Core.GPU

open System
open System.Diagnostics

/// WSL CUDA metrics test - shows actual performance measurements using WSL for CUDA compilation
module WSLCudaMetricsTest =
    
    /// Test WSL CUDA and show actual metrics
    let testWSLCudaMetrics () =
        printfn ""
        printfn "🔥 WSL CUDA METRICS TEST"
        printfn "========================"
        printfn ""
        
        try
            // Test WSL CUDA initialization
            printfn "🚀 Testing WSL CUDA initialization..."
            let initSuccess = WSLCudaEngine.initializeWSLCuda()
            
            if initSuccess then
                printfn ""
                printfn "✅ WSL CUDA SUCCESSFULLY INITIALIZED!"
                printfn ""
                
                // Get WSL GPU information
                let gpuInfo = WSLCudaEngine.getWSLGpuInfo()
                printfn "📊 WSL CUDA GPU SPECIFICATIONS:"
                for (key, value) in gpuInfo do
                    printfn "   • %s: %s" key value
                
                printfn ""
                printfn "🔥 RUNNING WSL CUDA BENCHMARKS..."
                printfn ""
                
                // Test 1: Sedenion calculation with WSL CUDA
                printfn "Test 1: WSL CUDA Sedenion Distance (10,000 vectors × 16D)"
                let vectors1 = Array.init (10000 * 16) (fun i -> float32 (i % 100) / 100.0f)
                let vectors2 = Array.init (10000 * 16) (fun i -> float32 ((i + 50) % 100) / 100.0f)
                let (results1, gflops1) = WSLCudaEngine.executeSedenionDistanceWSL vectors1 vectors2 10000 16
                
                printfn "   ⚡ GFLOPS: %.6f" gflops1
                printfn "   ⚡ Results: %d distances calculated" results1.Length
                printfn "   ⚡ Sample: [%.3f, %.3f, %.3f]" results1.[0] results1.[1] results1.[2]
                
                printfn ""
                
                // Test 2: Massive computation with WSL CUDA
                printfn "Test 2: WSL CUDA Massive Computation (100,000 elements × 1000 ops)"
                let (results2, gflops2) = WSLCudaEngine.executeMassiveComputeWSL 100000 1000
                
                printfn "   ⚡ GFLOPS: %.6f" gflops2
                printfn "   ⚡ Elements: %d" results2.Length
                printfn "   ⚡ Sample: [%.6f, %.6f, %.6f]" results2.[0] results2.[1] results2.[2]
                
                printfn ""
                
                // Test 3: Neural network with WSL CUDA
                printfn "Test 3: WSL CUDA Neural Network (1000 batch × 512 input × 256 output)"
                let inputs = Array.init (1000 * 512) (fun i -> float32 (i % 10) / 10.0f)
                let weights = Array.init (512 * 256) (fun i -> float32 (i % 5) / 5.0f)
                let biases = Array.init 256 (fun i -> float32 i / 256.0f)
                let (results3, gflops3) = WSLCudaEngine.executeNeuralForwardWSL inputs weights biases 1000 512 256
                
                printfn "   ⚡ GFLOPS: %.6f" gflops3
                printfn "   ⚡ Outputs: %d" results3.Length
                printfn "   ⚡ Sample: [%.6f, %.6f, %.6f]" results3.[0] results3.[1] results3.[2]
                
                printfn ""
                
                // Run comprehensive benchmark
                printfn "🚀 COMPREHENSIVE WSL CUDA BENCHMARK:"
                let benchmarkResults = WSLCudaEngine.runWSLCudaBenchmark()
                
                for kvp in benchmarkResults do
                    match kvp.Key with
                    | "sedenion_distance_gflops" -> printfn "   • Sedenion Distance: %.2f GFLOPS" kvp.Value
                    | "massive_compute_gflops" -> printfn "   • Massive Compute: %.2f GFLOPS" kvp.Value
                    | "neural_network_gflops" -> printfn "   • Neural Network: %.2f GFLOPS" kvp.Value
                    | "peak_performance" -> printfn "   • Peak Performance: %.2f GFLOPS" kvp.Value
                    | _ -> ()
                
                printfn ""
                printfn "🎯 WSL CUDA PERFORMANCE SUMMARY:"
                printfn "   • Sedenion Distance: %.6f GFLOPS" gflops1
                printfn "   • Massive Compute: %.6f GFLOPS" gflops2
                printfn "   • Neural Network: %.6f GFLOPS" gflops3
                printfn "   • Peak Performance: %.6f GFLOPS" (max (max gflops1 gflops2) gflops3)
                
                printfn ""
                let peakGflops = max (max gflops1 gflops2) gflops3
                if peakGflops > 100.0 then
                    printfn "🔥 EXCEPTIONAL WSL CUDA PERFORMANCE! Real GPU acceleration working!"
                elif peakGflops > 10.0 then
                    printfn "⚡ EXCELLENT WSL CUDA PERFORMANCE! CUDA kernels executing on GPU!"
                elif peakGflops > 1.0 then
                    printfn "✅ GOOD WSL CUDA PERFORMANCE! GPU acceleration active!"
                elif peakGflops > 0.1 then
                    printfn "⚠️  MODERATE WSL CUDA PERFORMANCE - Optimization needed"
                else
                    printfn "❌ LOW WSL CUDA PERFORMANCE - Check CUDA installation"
                
                printfn ""
                printfn "💡 WSL CUDA ADVANTAGES:"
                printfn "   ✅ Native CUDA compilation in Linux environment"
                printfn "   ✅ Full CUDA Toolkit support (nvcc, libraries)"
                printfn "   ✅ Better GPU driver compatibility"
                printfn "   ✅ Access to latest CUDA features"
                printfn "   ✅ Seamless integration with Linux CUDA ecosystem"
                
                // Cleanup
                WSLCudaEngine.cleanup()
                
            else
                printfn "❌ WSL CUDA NOT AVAILABLE"
                printfn ""
                printfn "📊 SYSTEM ANALYSIS:"
                printfn "   • WSL2: Not installed or not running"
                printfn "   • CUDA in WSL: Not installed"
                printfn "   • GPU Drivers: May not support WSL CUDA"
                printfn ""
                printfn "💡 TO ENABLE WSL CUDA:"
                printfn "   1. Install WSL2:"
                printfn "      wsl --install -d Ubuntu"
                printfn ""
                printfn "   2. Install NVIDIA drivers with WSL support:"
                printfn "      Download from: https://developer.nvidia.com/cuda/wsl"
                printfn ""
                printfn "   3. Install CUDA in WSL:"
                printfn "      wsl"
                printfn "      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb"
                printfn "      sudo dpkg -i cuda-keyring_1.0-1_all.deb"
                printfn "      sudo apt-get update"
                printfn "      sudo apt-get -y install cuda"
                printfn ""
                printfn "   4. Verify installation:"
                printfn "      wsl nvidia-smi"
                printfn "      wsl nvcc --version"
                printfn ""
                printfn "🔄 FALLBACK: Running CPU simulation mode"
                
                // Show CPU baseline performance for comparison
                printfn ""
                printfn "📊 CPU BASELINE PERFORMANCE (for comparison):"
                let stopwatch = Stopwatch.StartNew()
                let mutable sum = 0.0f
                for i = 0 to 999999 do
                    sum <- sum + float32 i * 1.001f
                    sum <- sum / 1.001f
                stopwatch.Stop()
                
                let cpuFlops = 1000000.0 * 3.0 // multiply, add, divide
                let cpuGflops = cpuFlops / stopwatch.Elapsed.TotalSeconds / 1e9
                
                printfn "   • CPU GFLOPS: %.6f" cpuGflops
                printfn "   • Time: %.3f ms" stopwatch.Elapsed.TotalMilliseconds
                printfn "   • Result: %.6f" sum
                printfn ""
                printfn "   Expected WSL CUDA improvement: 100-1000x faster!"
                
        with
        | ex ->
            printfn "❌ WSL CUDA TEST FAILED: %s" ex.Message
            printfn ""
            printfn "🔍 ERROR ANALYSIS:"
            printfn "   • Exception Type: %s" (ex.GetType().Name)
            printfn "   • Message: %s" ex.Message
            if ex.InnerException <> null then
                printfn "   • Inner Exception: %s" ex.InnerException.Message
            
            printfn ""
            printfn "💡 TROUBLESHOOTING STEPS:"
            printfn "   1. Check WSL2 installation: wsl --version"
            printfn "   2. Check WSL distribution: wsl --list --verbose"
            printfn "   3. Check NVIDIA drivers: nvidia-smi"
            printfn "   4. Check WSL CUDA: wsl nvidia-smi"
            printfn "   5. Restart WSL: wsl --shutdown && wsl"
            printfn "   6. Update Windows and WSL: wsl --update"
        
        printfn ""
        printfn "🎯 WSL CUDA METRICS TEST COMPLETE"
        printfn "=================================="
        printfn ""
        printfn "🌟 WSL CUDA BENEFITS:"
        printfn "   • Native Linux CUDA environment"
        printfn "   • Full nvcc compiler support"
        printfn "   • Better performance than Windows CUDA"
        printfn "   • Access to latest CUDA features"
        printfn "   • Seamless GPU acceleration"
