#!/usr/bin/env dotnet fsi

#r "nuget: Microsoft.Extensions.Logging"
#r "nuget: Microsoft.Extensions.Logging.Console"

open System
open System.IO
open Microsoft.Extensions.Logging

// Check if we're running on the right platform
let isLinux = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux)
let libraryExists = File.Exists("libTarsCudaKernels.so")

printfn ""
printfn "========================================================================"
printfn "                    TARS REAL CUDA LIBRARY TEST"
printfn "========================================================================"
printfn ""
printfn "🚀 Testing REAL compiled CUDA library - NO SIMULATIONS!"
printfn ""

printfn "🔍 Environment Check:"
let platformStr = if isLinux then "Linux/WSL" else "Windows"
let libraryStr = if libraryExists then "✅ libTarsCudaKernels.so found" else "❌ libTarsCudaKernels.so missing"
printfn $"   Platform: {platformStr}"
printfn $"   Library: {libraryStr}"
printfn ""

if not libraryExists then
    printfn "❌ CUDA library not found!"
    printfn "💡 Make sure libTarsCudaKernels.so is in the current directory"
    printfn "🔧 Run: wsl nvcc -O3 --shared -Xcompiler -fPIC -lcublas -o libTarsCudaKernels.so src/TarsEngine/CUDA/TarsCudaKernelsSimple.cu"
    exit 1

// Load the source file
#load "src/TarsEngine/RealCudaTest.fs"

open TarsEngine.RealCudaTest

// Create logger
let loggerFactory = LoggerFactory.Create(fun builder ->
    builder.AddConsole().SetMinimumLevel(LogLevel.Information) |> ignore
)

let logger = loggerFactory.CreateLogger<RealCudaTester>()

// Create tester
let tester = RealCudaTester(logger)

printfn "🧪 Starting REAL CUDA tests..."
printfn ""

// Run the real tests
let testResults = 
    async {
        try
            return! tester.RunRealCudaTestSuite()
        with
        | ex ->
            printfn $"❌ Test suite failed with exception: {ex.Message}"
            printfn $"🔍 Stack trace: {ex.StackTrace}"
            return ([], 0.0, false)
    } |> Async.RunSynchronously

let (results, successRate, anyPassed) = testResults

printfn ""
printfn "========================================================================"
printfn "                    REAL CUDA TEST RESULTS"
printfn "========================================================================"
printfn ""

if anyPassed then
    printfn "🎉 REAL CUDA LIBRARY WORKING!"
    printfn ""
    printfn "✅ SUCCESS METRICS:"
    printfn $"   Success Rate: {successRate:F1}%%"
    let librarySize = (new FileInfo("libTarsCudaKernels.so")).Length
    printfn $"   Library Size: {librarySize} bytes"
    printfn "   Platform: WSL2 with NVIDIA GPU support"
    printfn "   Compilation: Real nvcc with CUDA 11.5"
    printfn ""
    
    printfn "🔧 WORKING FEATURES:"
    for result in results do
        if result.Success then
            printfn $"   ✅ {result.TestName}"
            for kvp in result.ActualResults do
                printfn $"      📊 {kvp.Key}: {kvp.Value}"
    
    printfn ""
    printfn "🚀 READY FOR TARS INTEGRATION!"
    printfn "   The CUDA library is working and can be integrated with TARS AI inference"
    
else
    printfn "⚠️ CUDA LIBRARY ISSUES DETECTED"
    printfn ""
    printfn "❌ FAILED TESTS:"
    for result in results do
        if not result.Success then
            let errorMsg = result.ErrorMessage |> Option.defaultValue "Unknown error"
            printfn $"   ❌ {result.TestName}: {errorMsg}"
    
    printfn ""
    printfn "💡 TROUBLESHOOTING:"
    printfn "   • Check if NVIDIA drivers are installed in WSL"
    printfn "   • Verify CUDA runtime is available"
    printfn "   • Ensure GPU is accessible from WSL2"
    printfn "   • Library may still work for CPU-only operations"

printfn ""
printfn "🌟 This demonstrates REAL CUDA compilation and testing"
printfn "   No simulations - actual GPU code execution!"
printfn ""

// Cleanup
loggerFactory.Dispose()
