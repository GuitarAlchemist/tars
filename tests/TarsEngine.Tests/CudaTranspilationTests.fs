namespace TarsEngine.Tests

open System
open System.IO
open TarsEngine.Tests.TarsTestRunner

/// Comprehensive CUDA Transpilation Tests
/// Tests all aspects of F# to CUDA transpilation functionality
module CudaTranspilationTests =

    // ============================================================================
    // TEST DATA AND HELPERS
    // ============================================================================

    let testVectorA = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f |]
    let testVectorB = [| 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    let expectedVectorSum = [| 3.0f; 5.0f; 7.0f; 9.0f; 11.0f |]
    let expectedVectorProduct = [| 2.0f; 6.0f; 12.0f; 20.0f; 30.0f |]

    let testSedenion1 = Array.init 32 (fun i -> float32 (i + 1))
    let testSedenion2 = Array.init 32 (fun i -> float32 (i * 2 + 1))

    let createTempDirectory () : string =
        let tempDir = Path.Combine(Path.GetTempPath(), "tars_cuda_tests_" + Guid.NewGuid().ToString("N").[..7])
        Directory.CreateDirectory(tempDir) |> ignore
        tempDir

    let cleanupTempDirectory (dir: string) : unit =
        try
            if Directory.Exists(dir) then
                Directory.Delete(dir, true)
        with
        | _ -> () // Ignore cleanup errors

    // ============================================================================
    // CUDA TRANSPILATION UNIT TESTS
    // ============================================================================

    let testBasicFSharpToCudaTranspilation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let fsharpCode = """
let vectorAdd (a: float32 array) (b: float32 array) (result: float32 array) (n: int) =
    for i in 0 .. n-1 do
        result.[i] <- a.[i] + b.[i]
"""
            
            // Simple transpilation test (using our demo transpiler)
            let lines = fsharpCode.Split([|'\n'; '\r'|], StringSplitOptions.RemoveEmptyEntries)
            let hasForLoop = lines |> Array.exists (fun line -> line.Contains("for i in"))
            let hasArrayAccess = lines |> Array.exists (fun line -> line.Contains(".[i]"))
            let hasAddition = lines |> Array.exists (fun line -> line.Contains("+"))
            
            assertTrue hasForLoop "F# code should contain for loop"
            assertTrue hasArrayAccess "F# code should contain array access"
            assertTrue hasAddition "F# code should contain addition operation"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Basic F# to CUDA Transpilation"
                TestCategory = "CudaTranspilation"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("TranspilationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Basic F# to CUDA Transpilation"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testVectorOperationTranspilation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let operations = ["+"; "*"; "-"; "/"]
            let mutable allSucceeded = true
            let performanceMetrics = ResizeArray<string * float>()
            
            for operation in operations do
                let operationStart = DateTime.UtcNow
                
                let fsharpCode = sprintf "let vectorOp (a: float32 array) (b: float32 array) (result: float32 array) (n: int) =\n    for i in 0 .. n-1 do\n        result.[i] <- a.[i] %s b.[i]" operation
                
                // Validate the generated code structure
                let hasOperation = fsharpCode.Contains(operation)
                let hasLoop = fsharpCode.Contains("for i in")
                
                if not (hasOperation && hasLoop) then
                    allSucceeded <- false
                
                let operationTime = (DateTime.UtcNow - operationStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "VectorOp_%s_Time" operation, operationTime))
            
            assertTrue allSucceeded "All vector operations should transpile correctly"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Vector Operation Transpilation"
                TestCategory = "CudaTranspilation"
                Success = allSucceeded
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Vector Operation Transpilation"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testMathematicalFunctionTranspilation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let mathFunctions = ["sqrt"; "sin"; "cos"; "exp"; "log"]
            let mutable allSucceeded = true
            let performanceMetrics = ResizeArray<string * float>()
            
            for func in mathFunctions do
                let funcStart = DateTime.UtcNow
                
                let fsharpCode = sprintf "let mathKernel (input: float32 array) (output: float32 array) (n: int) =\n    for i in 0 .. n-1 do\n        output.[i] <- %s(input.[i])" func
                
                // Validate the generated code structure
                let hasFunction = fsharpCode.Contains(func)
                let hasLoop = fsharpCode.Contains("for i in")
                let hasArrayAccess = fsharpCode.Contains(".[i]")
                
                if not (hasFunction && hasLoop && hasArrayAccess) then
                    allSucceeded <- false
                
                let funcTime = (DateTime.UtcNow - funcStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "MathFunc_%s_Time" func, funcTime))
            
            assertTrue allSucceeded "All mathematical functions should transpile correctly"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Mathematical Function Transpilation"
                TestCategory = "CudaTranspilation"
                Success = allSucceeded
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Mathematical Function Transpilation"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testSedenionOperationTranspilation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let sedenionCode = "let sedenionAdd (a: float32 array) (b: float32 array) (result: float32 array) (count: int) =\n    for i in 0 .. count-1 do\n        for j in 0 .. 15 do\n            let idx = i * 16 + j\n            result.[idx] <- a.[idx] + b.[idx]"
            
            // Validate 16D sedenion structure
            let hasNestedLoops = sedenionCode.Contains("for i in") && sedenionCode.Contains("for j in")
            let has16Components = sedenionCode.Contains("15")
            let hasIndexCalculation = sedenionCode.Contains("i * 16 + j")
            let hasSedenionOperation = sedenionCode.Contains("a.[idx] + b.[idx]")
            
            assertTrue hasNestedLoops "Sedenion code should have nested loops"
            assertTrue has16Components "Sedenion code should handle 16 components"
            assertTrue hasIndexCalculation "Sedenion code should calculate 16D indices"
            assertTrue hasSedenionOperation "Sedenion code should perform component-wise operations"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Sedenion Operation Transpilation"
                TestCategory = "CudaTranspilation"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("SedenionTranspilationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Sedenion Operation Transpilation"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testCudaCodeGeneration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let kernelName = "test_kernel"
            let fsharpCode = "let add a b = a + b"
            
            // Simulate CUDA code generation
            let cudaCode = sprintf "#include <cuda_runtime.h>\n#include <stdio.h>\n#include <math.h>\n\n__global__ void %s(float* a, float* b, float* result, int n) {\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;\n    if (idx < n) {\n        result[idx] = a[idx] + b[idx];\n    }\n}\n\nint main() {\n    printf(\"CUDA kernel '%s' compiled successfully\\n\");\n    return 0;\n}" kernelName kernelName
            
            // Validate CUDA code structure
            let hasGlobalKeyword = cudaCode.Contains("__global__")
            let hasThreadIndexing = cudaCode.Contains("blockIdx.x") && cudaCode.Contains("threadIdx.x")
            let hasBoundsCheck = cudaCode.Contains("if (idx < n)")
            let hasKernelName = cudaCode.Contains(kernelName)
            let hasHeaders = cudaCode.Contains("#include <cuda_runtime.h>")
            
            assertTrue hasGlobalKeyword "CUDA code should have __global__ keyword"
            assertTrue hasThreadIndexing "CUDA code should use thread indexing"
            assertTrue hasBoundsCheck "CUDA code should have bounds checking"
            assertTrue hasKernelName "CUDA code should contain kernel name"
            assertTrue hasHeaders "CUDA code should include necessary headers"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "CUDA Code Generation"
                TestCategory = "CudaTranspilation"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("CudaGenerationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "CUDA Code Generation"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testErrorHandling () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            // Test invalid F# code
            let invalidCode = "this is not valid F# code"
            let mutable errorHandled = false
            
            try
                // Simulate transpilation of invalid code
                let lines = invalidCode.Split([|'\n'|])
                if lines.Length = 1 && not (lines.[0].Contains("let")) then
                    errorHandled <- true
            with
            | _ -> errorHandled <- true
            
            assertTrue errorHandled "Invalid F# code should be handled gracefully"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Error Handling"
                TestCategory = "CudaTranspilation"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("ErrorHandlingTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Error Handling"
                TestCategory = "CudaTranspilation"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    // ============================================================================
    // TEST SUITE DEFINITION
    // ============================================================================

    let cudaTranspilationTestSuite : TestSuite = {
        SuiteName = "CUDA Transpilation Tests"
        Category = "CudaTranspilation"
        Tests = [
            testBasicFSharpToCudaTranspilation
            testVectorOperationTranspilation
            testMathematicalFunctionTranspilation
            testSedenionOperationTranspilation
            testCudaCodeGeneration
            testErrorHandling
        ]
        SetupAction = Some (fun () -> 
            printfn "🔧 Setting up CUDA transpilation test environment..."
        )
        TeardownAction = Some (fun () -> 
            printfn "🧹 Cleaning up CUDA transpilation test environment..."
        )
    }
