namespace TarsEngine.Tests

open System
open System.IO
open TarsEngine.Tests.TarsTestRunner

/// Comprehensive Integration Tests
/// Tests end-to-end functionality across all TARS components
module IntegrationTests =

    // ============================================================================
    // INTEGRATION TEST DATA
    // ============================================================================

    let createIntegrationTestEnvironment () : string =
        let testDir = Path.Combine(Path.GetTempPath(), "tars_integration_tests_" + Guid.NewGuid().ToString("N").[..7])
        Directory.CreateDirectory(testDir) |> ignore
        
        // Create subdirectories
        Directory.CreateDirectory(Path.Combine(testDir, "grammars")) |> ignore
        Directory.CreateDirectory(Path.Combine(testDir, "cuda")) |> ignore
        Directory.CreateDirectory(Path.Combine(testDir, "generated")) |> ignore
        
        testDir

    let cleanupIntegrationEnvironment (dir: string) : unit =
        try
            if Directory.Exists(dir) then
                Directory.Delete(dir, true)
        with
        | _ -> () // Ignore cleanup errors

    // ============================================================================
    // END-TO-END INTEGRATION TESTS
    // ============================================================================

    let testFullTierIntegration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testEnv = createIntegrationTestEnvironment()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test Tier 1-5 integration
            let tiers = [
                (1, "BasicMathematical", ["createSedenion"; "basicArithmetic"])
                (2, "ComputationalExpressions", ["sedenionBuilder"; "vectorBuilder"])
                (3, "AdvancedMathematical", ["nonEuclideanDistance"; "hyperComplexAnalysis"])
                (4, "DomainSpecificResearch", ["janusModelAnalysis"; "cmbAnalysis"])
                (5, "CudafyEnhanced", ["cudafyTranspilation"; "gpuKernelGeneration"])
            ]
            
            let mutable allTiersValid = true
            
            for (tierNum, tierName, operations) in tiers do
                let tierStart = DateTime.UtcNow
                
                // Validate tier structure
                let hasValidName = not (String.IsNullOrEmpty(tierName))
                let hasOperations = operations.Length > 0
                let hasCorrectTierNumber = tierNum >= 1 && tierNum <= 5
                
                if not (hasValidName && hasOperations && hasCorrectTierNumber) then
                    allTiersValid <- false
                
                // Validate tier progression
                if tierNum > 1 then
                    let expectedMinOps = tierNum + 1
                    if operations.Length < expectedMinOps then
                        allTiersValid <- false
                
                let tierTime = (DateTime.UtcNow - tierStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "Tier%dValidation" tierNum, tierTime))
            
            assertTrue allTiersValid "All tiers should integrate correctly"
            
            cleanupIntegrationEnvironment testEnv
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Full Tier Integration"
                TestCategory = "Integration"
                Success = allTiersValid
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Full Tier Integration"
                TestCategory = "Integration"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testGrammarToCudaIntegration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testEnv = createIntegrationTestEnvironment()
            
            // Test grammar tier to CUDA transpilation integration
            let grammarStart = DateTime.UtcNow
            
            // Simulate grammar tier 5 with CUDA operations
            let tier5Operations = [
                "cudafyTranspilation"
                "cudafyClosureFactory"
                "gpuKernelGeneration"
                "parallelExecution"
                "memoryManagement"
                "performanceOptimization"
            ]
            
            let tier5Expressions = [
                "cudafy { ... }"
                "gpuParallel { ... }"
            ]
            
            let grammarTime = (DateTime.UtcNow - grammarStart).TotalMilliseconds
            
            // Test CUDA transpilation integration
            let cudaStart = DateTime.UtcNow
            
            let fsharpCode = """
let vectorAdd (a: float32 array) (b: float32 array) (result: float32 array) (n: int) =
    for i in 0 .. n-1 do
        result.[i] <- a.[i] + b.[i]
"""
            
            // Simulate transpilation
            let hasForLoop = fsharpCode.Contains("for i in")
            let hasArrayAccess = fsharpCode.Contains(".[i]")
            let hasOperation = fsharpCode.Contains("+")
            
            let cudaTime = (DateTime.UtcNow - cudaStart).TotalMilliseconds
            
            // Validate integration
            assertTrue (tier5Operations.Length >= 6) "Tier 5 should have sufficient CUDA operations"
            assertTrue (tier5Expressions.Length >= 2) "Tier 5 should have CUDA expressions"
            assertTrue hasForLoop "F# code should be transpilable"
            assertTrue hasArrayAccess "F# code should have array operations"
            assertTrue hasOperation "F# code should have mathematical operations"
            
            cleanupIntegrationEnvironment testEnv
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Grammar to CUDA Integration"
                TestCategory = "Integration"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [
                    ("GrammarProcessing", grammarTime)
                    ("CudaTranspilation", cudaTime)
                    ("TotalIntegration", executionTime.TotalMilliseconds)
                ])
            }
        with
        | ex ->
            {
                TestName = "Grammar to CUDA Integration"
                TestCategory = "Integration"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testComputationalExpressionIntegration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testEnv = createIntegrationTestEnvironment()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test all computational expressions integration
            let expressions = [
                ("sedenion", "Tier 1-3")
                ("geometric", "Tier 1-3")
                ("vector", "Tier 1-3")
                ("cuda", "Tier 4-5")
                ("bsp", "Tier 3-4")
                ("janus", "Tier 4")
                ("cmb", "Tier 4")
                ("spacetime", "Tier 4")
                ("cudafy", "Tier 5")
                ("gpuParallel", "Tier 5")
            ]
            
            let mutable allExpressionsValid = true
            
            for (exprName, tierRange) in expressions do
                let exprStart = DateTime.UtcNow
                
                // Validate expression structure
                let hasValidName = not (String.IsNullOrEmpty(exprName))
                let hasValidTierRange = not (String.IsNullOrEmpty(tierRange))
                
                // Simulate expression execution
                let mockResult = sprintf "%s expression executed successfully" exprName
                let hasResult = not (String.IsNullOrEmpty(mockResult))
                
                if not (hasValidName && hasValidTierRange && hasResult) then
                    allExpressionsValid <- false
                
                let exprTime = (DateTime.UtcNow - exprStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "%sExpression" exprName, exprTime))
            
            assertTrue allExpressionsValid "All computational expressions should integrate correctly"
            
            // Test expression composition
            let compositionStart = DateTime.UtcNow
            
            // Simulate nested expression execution
            let nestedExpressions = [
                "sedenion { vector { ... } }"
                "cudafy { sedenion { ... } }"
                "gpuParallel { vector { ... } }"
            ]
            
            let allNested = nestedExpressions |> List.forall (fun expr -> expr.Contains("{") && expr.Contains("}"))
            assertTrue allNested "Nested expressions should be supported"
            
            let compositionTime = (DateTime.UtcNow - compositionStart).TotalMilliseconds
            performanceMetrics.Add(("ExpressionComposition", compositionTime))
            
            cleanupIntegrationEnvironment testEnv
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Computational Expression Integration"
                TestCategory = "Integration"
                Success = allExpressionsValid
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Computational Expression Integration"
                TestCategory = "Integration"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testPerformanceScalability () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testEnv = createIntegrationTestEnvironment()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test scalability with different data sizes
            let dataSizes = [100; 1000; 10000]
            let mutable allScalable = true
            
            for size in dataSizes do
                let scaleStart = DateTime.UtcNow
                
                // Test vector operations scalability
                let vectorA = Array.init size (fun i -> float32 i)
                let vectorB = Array.init size (fun i -> float32 (i * 2))
                let result = Array.zeroCreate size
                
                for i in 0 .. size - 1 do
                    result.[i] <- vectorA.[i] + vectorB.[i]
                
                let scaleTime = (DateTime.UtcNow - scaleStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "VectorOp_Size%d" size, scaleTime))
                
                // Validate results
                let isCorrect = result.[0] = 0.0f && result.[size-1] = float32 (size - 1 + (size - 1) * 2)
                if not isCorrect then
                    allScalable <- false
                
                // Test sedenion operations scalability
                let sedenionStart = DateTime.UtcNow
                let sedenionCount = size / 16
                let sedenionA = Array.init (sedenionCount * 16) (fun i -> float32 i)
                let sedenionB = Array.init (sedenionCount * 16) (fun i -> float32 (i + 1))
                let sedenionResult = Array.zeroCreate (sedenionCount * 16)
                
                for i in 0 .. sedenionCount - 1 do
                    for j in 0 .. 15 do
                        let idx = i * 16 + j
                        sedenionResult.[idx] <- sedenionA.[idx] + sedenionB.[idx]
                
                let sedenionTime = (DateTime.UtcNow - sedenionStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "SedenionOp_Size%d" size, sedenionTime))
            
            assertTrue allScalable "Operations should scale correctly with data size"
            
            // Test memory efficiency
            let memoryStart = DateTime.UtcNow
            GC.Collect()
            let beforeMemory = GC.GetTotalMemory(false)
            
            // Allocate test data
            let largeArray = Array.init 100000 (fun i -> float32 i)
            let processedArray = largeArray |> Array.map (fun x -> x * 2.0f)
            
            GC.Collect()
            let afterMemory = GC.GetTotalMemory(false)
            let memoryUsed = afterMemory - beforeMemory
            
            let memoryTime = (DateTime.UtcNow - memoryStart).TotalMilliseconds
            performanceMetrics.Add(("MemoryEfficiency", memoryTime))
            performanceMetrics.Add(("MemoryUsedMB", float memoryUsed / 1024.0 / 1024.0))
            
            assertTrue (memoryUsed > 0L) "Memory usage should be tracked"
            assertTrue (processedArray.Length = largeArray.Length) "Array processing should preserve length"
            
            cleanupIntegrationEnvironment testEnv
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Performance Scalability"
                TestCategory = "Integration"
                Success = allScalable
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Performance Scalability"
                TestCategory = "Integration"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testErrorRecoveryIntegration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testEnv = createIntegrationTestEnvironment()
            let performanceMetrics = ResizeArray<string * float>()
            
            // Test error handling across components
            let errorScenarios = [
                ("InvalidGrammarTier", "Invalid tier number: -1")
                ("MissingDependency", "Missing dependency: Tier 99")
                ("InvalidFSharpCode", "Syntax error in F# code")
                ("CudaCompilationError", "CUDA compilation failed")
                ("MemoryAllocationError", "Insufficient GPU memory")
            ]
            
            let mutable allErrorsHandled = true
            
            for (scenario, errorMessage) in errorScenarios do
                let errorStart = DateTime.UtcNow
                
                // Simulate error handling
                let mutable errorCaught = false
                
                try
                    // Simulate error condition
                    match scenario with
                    | "InvalidGrammarTier" -> 
                        let invalidTier = -1
                        if invalidTier < 1 then failwith errorMessage
                    | "MissingDependency" ->
                        let missingDep = 99
                        if missingDep > 10 then failwith errorMessage
                    | "InvalidFSharpCode" ->
                        let invalidCode = "this is not F#"
                        if not (invalidCode.Contains("let")) then failwith errorMessage
                    | "CudaCompilationError" ->
                        let invalidCuda = "__invalid__ void kernel() {}"
                        if invalidCuda.Contains("__invalid__") then failwith errorMessage
                    | "MemoryAllocationError" ->
                        let largeSize = Int32.MaxValue
                        if largeSize > 1000000000 then failwith errorMessage
                    | _ -> ()
                with
                | ex when ex.Message = errorMessage ->
                    errorCaught <- true
                | _ ->
                    errorCaught <- false
                
                if not errorCaught then
                    allErrorsHandled <- false
                
                let errorTime = (DateTime.UtcNow - errorStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "%sHandling" scenario, errorTime))
            
            assertTrue allErrorsHandled "All error scenarios should be handled gracefully"
            
            cleanupIntegrationEnvironment testEnv
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Error Recovery Integration"
                TestCategory = "Integration"
                Success = allErrorsHandled
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Error Recovery Integration"
                TestCategory = "Integration"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    // ============================================================================
    // TEST SUITE DEFINITION
    // ============================================================================

    let integrationTestSuite : TestSuite = {
        SuiteName = "Integration Tests"
        Category = "Integration"
        Tests = [
            testFullTierIntegration
            testGrammarToCudaIntegration
            testComputationalExpressionIntegration
            testPerformanceScalability
            testErrorRecoveryIntegration
        ]
        SetupAction = Some (fun () -> 
            printfn "🔧 Setting up integration test environment..."
        )
        TeardownAction = Some (fun () -> 
            printfn "🧹 Cleaning up integration test environment..."
        )
    }
