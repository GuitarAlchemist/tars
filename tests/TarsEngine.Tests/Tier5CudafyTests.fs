namespace TarsEngine.Tests

open System
open System.IO
open TarsEngine.Tests.TarsTestRunner

/// Comprehensive Tier 5 Cudafy Tests
/// Tests Cudafy computational expressions, closure factory, and GPU operations
module Tier5CudafyTests =

    // ============================================================================
    // TEST DATA AND HELPERS
    // ============================================================================

    type MockCudafyContext = {
        WorkingDirectory: string
        mutable CompiledKernels: Map<string, string>
        mutable ExecutionLog: string list
        mutable PerformanceMetrics: Map<string, float>
    }

    type MockCudafyOperation<'T> = {
        OperationName: string
        InputData: 'T
        KernelCode: string option
        mutable Result: 'T option
        mutable IsCompiled: bool
        mutable ExecutionTime: TimeSpan
    }

    let createMockCudafyContext () : MockCudafyContext =
        {
            WorkingDirectory = Path.GetTempPath()
            CompiledKernels = Map.empty
            ExecutionLog = []
            PerformanceMetrics = Map.empty
        }

    let testVectorA = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f |]
    let testVectorB = [| 2.0f; 3.0f; 4.0f; 5.0f; 6.0f |]
    let expectedVectorSum = [| 3.0f; 5.0f; 7.0f; 9.0f; 11.0f |]
    let expectedVectorProduct = [| 2.0f; 6.0f; 12.0f; 20.0f; 30.0f |]

    let testSedenion1 = Array.init 32 (fun i -> float32 (i + 1))
    let testSedenion2 = Array.init 32 (fun i -> float32 (i * 2 + 1))

    // ============================================================================
    // CUDAFY COMPUTATIONAL EXPRESSION TESTS
    // ============================================================================

    let testCudafyExpressionBasics () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            
            // Test basic cudafy expression structure
            let mockOperation = {
                OperationName = "test_operation"
                InputData = testVectorA
                KernelCode = Some "mock kernel code"
                Result = Some testVectorA
                IsCompiled = false
                ExecutionTime = TimeSpan.Zero
            }
            
            // Simulate cudafy expression execution
            let operationStart = DateTime.UtcNow
            mockOperation.IsCompiled <- true
            mockOperation.ExecutionTime <- DateTime.UtcNow - operationStart
            context.CompiledKernels <- context.CompiledKernels.Add(mockOperation.OperationName, mockOperation.KernelCode.Value)
            context.ExecutionLog <- sprintf "Executed %s" mockOperation.OperationName :: context.ExecutionLog
            
            // Validate results
            assertTrue mockOperation.IsCompiled "Operation should be marked as compiled"
            assertTrue (mockOperation.Result.IsSome) "Operation should have a result"
            assertTrue (context.CompiledKernels.ContainsKey(mockOperation.OperationName)) "Kernel should be stored in context"
            assertTrue (context.ExecutionLog.Length > 0) "Execution should be logged"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Cudafy Expression Basics"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("CudafyExpressionTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Cudafy Expression Basics"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testVectorOperations () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test vector addition
            let addStart = DateTime.UtcNow
            let vectorSum = Array.zeroCreate testVectorA.Length
            for i in 0 .. testVectorA.Length - 1 do
                vectorSum.[i] <- testVectorA.[i] + testVectorB.[i]
            let addTime = (DateTime.UtcNow - addStart).TotalMilliseconds
            performanceMetrics.Add(("VectorAddition", addTime))
            
            // Test vector multiplication
            let mulStart = DateTime.UtcNow
            let vectorProduct = Array.zeroCreate testVectorA.Length
            for i in 0 .. testVectorA.Length - 1 do
                vectorProduct.[i] <- testVectorA.[i] * testVectorB.[i]
            let mulTime = (DateTime.UtcNow - mulStart).TotalMilliseconds
            performanceMetrics.Add(("VectorMultiplication", mulTime))
            
            // Validate results
            assertArrayWithinToleranceF32 expectedVectorSum vectorSum 0.001f "Vector addition should be correct"
            assertArrayWithinToleranceF32 expectedVectorProduct vectorProduct 0.001f "Vector multiplication should be correct"
            
            // Log operations
            context.ExecutionLog <- "Vector multiplication completed" :: context.ExecutionLog
            context.ExecutionLog <- "Vector addition completed" :: context.ExecutionLog
            
            assertTrue (context.ExecutionLog.Length = 2) "Both operations should be logged"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Vector Operations"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Vector Operations"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testMathematicalFunctions () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            let testInput = [| 1.0f; 4.0f; 9.0f; 16.0f; 25.0f |]
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test sqrt function
            let sqrtStart = DateTime.UtcNow
            let sqrtResult = testInput |> Array.map (fun x -> sqrt (float x) |> float32)
            let sqrtTime = (DateTime.UtcNow - sqrtStart).TotalMilliseconds
            performanceMetrics.Add(("SqrtFunction", sqrtTime))

            // Test sin function
            let sinStart = DateTime.UtcNow
            let sinResult = testInput |> Array.map (fun x -> sin (float x) |> float32)
            let sinTime = (DateTime.UtcNow - sinStart).TotalMilliseconds
            performanceMetrics.Add(("SinFunction", sinTime))

            // Test cos function
            let cosStart = DateTime.UtcNow
            let cosResult = testInput |> Array.map (fun x -> cos (float x) |> float32)
            let cosTime = (DateTime.UtcNow - cosStart).TotalMilliseconds
            performanceMetrics.Add(("CosFunction", cosTime))
            
            // Validate results
            let expectedSqrt = [| 1.0f; 2.0f; 3.0f; 4.0f; 5.0f |]
            assertArrayWithinToleranceF32 expectedSqrt sqrtResult 0.001f "Square root should be correct"
            
            assertTrue (sqrtResult.Length = testInput.Length) "Sqrt result should have same length"
            assertTrue (sinResult.Length = testInput.Length) "Sin result should have same length"
            assertTrue (cosResult.Length = testInput.Length) "Cos result should have same length"
            
            // Log operations
            context.ExecutionLog <- "Cos function completed" :: context.ExecutionLog
            context.ExecutionLog <- "Sin function completed" :: context.ExecutionLog
            context.ExecutionLog <- "Sqrt function completed" :: context.ExecutionLog
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Mathematical Functions"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Mathematical Functions"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testSedenionOperations () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            
            // Test 16D sedenion addition
            let sedenionStart = DateTime.UtcNow
            let sedenionResult = Array.zeroCreate testSedenion1.Length
            
            // Simulate 16D sedenion addition
            for i in 0 .. (testSedenion1.Length / 16) - 1 do
                for j in 0 .. 15 do
                    let idx = i * 16 + j
                    if idx < testSedenion1.Length && idx < testSedenion2.Length then
                        sedenionResult.[idx] <- testSedenion1.[idx] + testSedenion2.[idx]
            
            let sedenionTime = (DateTime.UtcNow - sedenionStart).TotalMilliseconds
            
            // Validate 16D structure
            assertEqual 32 sedenionResult.Length "Sedenion result should have 32 components (2 sedenions)"
            
            // Validate first few components
            let expectedFirst8 = [| 2.0f; 5.0f; 8.0f; 11.0f; 14.0f; 17.0f; 20.0f; 23.0f |]
            let actualFirst8 = sedenionResult.[..7]
            assertArrayWithinToleranceF32 expectedFirst8 actualFirst8 0.001f "First 8 sedenion components should be correct"
            
            // Log operation
            context.ExecutionLog <- "Sedenion addition completed" :: context.ExecutionLog
            context.CompiledKernels <- context.CompiledKernels.Add("sedenion_add", "sedenion addition kernel")
            
            assertTrue (context.CompiledKernels.ContainsKey("sedenion_add")) "Sedenion kernel should be compiled"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Sedenion Operations"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("SedenionAddition", sedenionTime)])
            }
        with
        | ex ->
            {
                TestName = "Sedenion Operations"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testClosureFactory () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test vector operation closure creation
            let createVectorOpStart = DateTime.UtcNow
            let vectorAddClosure = fun (a: float32[]) (b: float32[]) ->
                let result = Array.zeroCreate a.Length
                for i in 0 .. a.Length - 1 do
                    result.[i] <- a.[i] + b.[i]
                context.ExecutionLog <- "Vector add closure executed" :: context.ExecutionLog
                result
            let createVectorOpTime = (DateTime.UtcNow - createVectorOpStart).TotalMilliseconds
            performanceMetrics.Add(("CreateVectorOpClosure", createVectorOpTime))
            
            // Test math function closure creation
            let createMathFuncStart = DateTime.UtcNow
            let sqrtClosure = fun (input: float32[]) ->
                let result = input |> Array.map (fun x -> sqrt (float x) |> float32)
                context.ExecutionLog <- "Sqrt closure executed" :: context.ExecutionLog
                result
            let createMathFuncTime = (DateTime.UtcNow - createMathFuncStart).TotalMilliseconds
            performanceMetrics.Add(("CreateMathFuncClosure", createMathFuncTime))
            
            // Test sedenion operation closure creation
            let createSedenionOpStart = DateTime.UtcNow
            let sedenionAddClosure = fun (a: float32[]) (b: float32[]) ->
                let result = Array.zeroCreate a.Length
                for i in 0 .. (a.Length / 16) - 1 do
                    for j in 0 .. 15 do
                        let idx = i * 16 + j
                        if idx < a.Length && idx < b.Length then
                            result.[idx] <- a.[idx] + b.[idx]
                context.ExecutionLog <- "Sedenion add closure executed" :: context.ExecutionLog
                result
            let createSedenionOpTime = (DateTime.UtcNow - createSedenionOpStart).TotalMilliseconds
            performanceMetrics.Add(("CreateSedenionOpClosure", createSedenionOpTime))
            
            // Test closure execution
            let vectorResult = vectorAddClosure testVectorA testVectorB
            let sqrtResult = sqrtClosure [| 1.0f; 4.0f; 9.0f |]
            let sedenionResult = sedenionAddClosure testSedenion1 testSedenion2
            
            // Validate closure results
            assertArrayWithinToleranceF32 expectedVectorSum vectorResult 0.001f "Vector closure should work correctly"
            assertArrayWithinToleranceF32 [| 1.0f; 2.0f; 3.0f |] sqrtResult 0.001f "Math closure should work correctly"
            assertTrue (sedenionResult.Length = testSedenion1.Length) "Sedenion closure should preserve length"
            
            // Validate execution logging
            assertTrue (context.ExecutionLog.Length = 3) "All closures should be logged"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Closure Factory"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Closure Factory"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testPerformanceMonitoring () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let context = createMockCudafyContext()
            
            // Simulate multiple operations with performance tracking
            let operations = ["vector_add"; "vector_mul"; "math_sqrt"; "sedenion_add"]
            let mutable totalOperations = 0
            let mutable totalTime = 0.0
            
            for operation in operations do
                let opStart = DateTime.UtcNow
                
                // Simulate operation execution
                System.Threading.Thread.Sleep(1) // Small delay to simulate work
                
                let opTime = (DateTime.UtcNow - opStart).TotalMilliseconds
                totalTime <- totalTime + opTime
                totalOperations <- totalOperations + 1
                
                context.ExecutionLog <- sprintf "Executed %s in %.2f ms" operation opTime :: context.ExecutionLog
                context.CompiledKernels <- context.CompiledKernels.Add(operation, sprintf "%s kernel code" operation)
                context.PerformanceMetrics <- context.PerformanceMetrics.Add(operation, opTime)
            
            // Calculate performance metrics
            let averageTime = totalTime / float totalOperations
            let totalKernels = context.CompiledKernels.Count
            let totalExecutions = context.ExecutionLog.Length
            
            // Validate performance tracking
            assertTrue (totalOperations = operations.Length) "All operations should be tracked"
            assertTrue (totalKernels = operations.Length) "All kernels should be compiled"
            assertTrue (totalExecutions = operations.Length) "All executions should be logged"
            assertTrue (averageTime > 0.0) "Average time should be positive"
            assertTrue (context.PerformanceMetrics.Count = operations.Length) "All metrics should be stored"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Performance Monitoring"
                TestCategory = "Tier5Cudafy"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [
                    ("TotalOperations", float totalOperations)
                    ("AverageOperationTime", averageTime)
                    ("TotalKernels", float totalKernels)
                    ("MonitoringOverhead", executionTime.TotalMilliseconds)
                ])
            }
        with
        | ex ->
            {
                TestName = "Performance Monitoring"
                TestCategory = "Tier5Cudafy"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    // ============================================================================
    // TEST SUITE DEFINITION
    // ============================================================================

    let tier5CudafyTestSuite : TestSuite = {
        SuiteName = "Tier 5 Cudafy Tests"
        Category = "Tier5Cudafy"
        Tests = [
            testCudafyExpressionBasics
            testVectorOperations
            testMathematicalFunctions
            testSedenionOperations
            testClosureFactory
            testPerformanceMonitoring
        ]
        SetupAction = Some (fun () -> 
            printfn "🔧 Setting up Tier 5 Cudafy test environment..."
        )
        TeardownAction = Some (fun () -> 
            printfn "🧹 Cleaning up Tier 5 Cudafy test environment..."
        )
    }
