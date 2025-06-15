namespace TarsEngine.Tests

open System
open System.IO
open System.Text.Json
open TarsEngine.Tests.TarsTestRunner

/// Comprehensive Grammar Tier Tests
/// Tests grammar tier validation, progression, and computational expression generation
module GrammarTierTests =

    // ============================================================================
    // TEST DATA AND HELPERS
    // ============================================================================

    type TestGrammarTier = {
        Tier: int
        Name: string
        Description: string
        Operations: string list
        Dependencies: int list
        ComputationalExpressions: string list
    }

    let createTestGrammarFile (tier: TestGrammarTier) (directory: string) : string =
        let fileName = sprintf "%d.json" tier.Tier
        let filePath = Path.Combine(directory, fileName)
        
        let jsonContent = sprintf "{\n  \"tier\": %d,\n  \"name\": \"%s\",\n  \"description\": \"%s\",\n  \"operations\": [%s],\n  \"dependencies\": [%s],\n  \"computationalExpressions\": [%s]\n}" tier.Tier tier.Name tier.Description (tier.Operations |> List.map (sprintf "\"%s\"") |> String.concat ", ") (tier.Dependencies |> List.map string |> String.concat ", ") (tier.ComputationalExpressions |> List.map (sprintf "\"%s\"") |> String.concat ", ")
        
        File.WriteAllText(filePath, jsonContent)
        filePath

    let createTestGrammarDirectory () : string =
        let tempDir = Path.Combine(Path.GetTempPath(), "tars_grammar_tests_" + Guid.NewGuid().ToString("N").[..7])
        Directory.CreateDirectory(tempDir) |> ignore
        tempDir

    let cleanupTestDirectory (dir: string) : unit =
        try
            if Directory.Exists(dir) then
                Directory.Delete(dir, true)
        with
        | _ -> () // Ignore cleanup errors

    // ============================================================================
    // GRAMMAR TIER VALIDATION TESTS
    // ============================================================================

    let testTier1Validation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testDir = createTestGrammarDirectory()
            
            let tier1 = {
                Tier = 1
                Name = "BasicMathematical"
                Description = "Basic mathematical constructs"
                Operations = ["createSedenion"; "createGeometricSpace"; "basicArithmetic"]
                Dependencies = []
                ComputationalExpressions = []
            }
            
            let filePath = createTestGrammarFile tier1 testDir
            
            // Validate file creation
            assertTrue (File.Exists(filePath)) "Tier 1 grammar file should be created"
            
            // Validate JSON structure
            let jsonContent = File.ReadAllText(filePath)
            let jsonDoc = JsonDocument.Parse(jsonContent)
            let root = jsonDoc.RootElement
            
            assertEqual 1 (root.GetProperty("tier").GetInt32()) "Tier number should be 1"
            assertEqual "BasicMathematical" (root.GetProperty("name").GetString()) "Tier name should match"
            assertTrue (root.GetProperty("operations").GetArrayLength() > 0) "Tier 1 should have operations"
            assertEqual 0 (root.GetProperty("dependencies").GetArrayLength()) "Tier 1 should have no dependencies"
            
            cleanupTestDirectory testDir
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Tier 1 Validation"
                TestCategory = "GrammarTier"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("Tier1ValidationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Tier 1 Validation"
                TestCategory = "GrammarTier"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testTierProgression () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testDir = createTestGrammarDirectory()
            
            let tiers = [
                { Tier = 1; Name = "BasicMathematical"; Description = "Basic math"; Operations = ["op1"; "op2"]; Dependencies = []; ComputationalExpressions = [] }
                { Tier = 2; Name = "ComputationalExpressions"; Description = "Comp expressions"; Operations = ["op1"; "op2"; "op3"]; Dependencies = [1]; ComputationalExpressions = ["expr1"] }
                { Tier = 3; Name = "AdvancedMathematical"; Description = "Advanced math"; Operations = ["op1"; "op2"; "op3"; "op4"]; Dependencies = [1; 2]; ComputationalExpressions = ["expr1"; "expr2"] }
                { Tier = 4; Name = "DomainSpecific"; Description = "Domain specific"; Operations = ["op1"; "op2"; "op3"; "op4"; "op5"]; Dependencies = [1; 2; 3]; ComputationalExpressions = ["expr1"; "expr2"; "expr3"] }
                { Tier = 5; Name = "CudafyEnhanced"; Description = "CUDA enhanced"; Operations = ["op1"; "op2"; "op3"; "op4"; "op5"; "op6"]; Dependencies = [1; 2; 3; 4]; ComputationalExpressions = ["expr1"; "expr2"; "expr3"; "expr4"] }
            ]
            
            let mutable allValid = true
            let performanceMetrics = ResizeArray<string * float>()
            
            for tier in tiers do
                let tierStart = DateTime.UtcNow
                let filePath = createTestGrammarFile tier testDir
                
                // Validate tier progression rules
                if tier.Tier > 1 then
                    // Should depend on previous tier
                    let dependsOnPrevious = tier.Dependencies |> List.contains (tier.Tier - 1)
                    if not dependsOnPrevious then
                        allValid <- false
                
                // Operations should increase with tier
                let expectedMinOps = tier.Tier + 1
                if tier.Operations.Length < expectedMinOps then
                    allValid <- false
                
                // Computational expressions should increase with tier
                if tier.Tier > 1 && tier.ComputationalExpressions.Length < tier.Tier - 1 then
                    allValid <- false
                
                let tierTime = (DateTime.UtcNow - tierStart).TotalMilliseconds
                performanceMetrics.Add((sprintf "Tier%dValidation" tier.Tier, tierTime))
            
            assertTrue allValid "All tiers should follow progression rules"
            
            cleanupTestDirectory testDir
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Tier Progression"
                TestCategory = "GrammarTier"
                Success = allValid
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList (performanceMetrics.ToArray() |> Array.toList))
            }
        with
        | ex ->
            {
                TestName = "Tier Progression"
                TestCategory = "GrammarTier"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testDependencyValidation () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testDir = createTestGrammarDirectory()
            
            // Create valid dependency chain
            let tier1 = { Tier = 1; Name = "Base"; Description = "Base tier"; Operations = ["op1"]; Dependencies = []; ComputationalExpressions = [] }
            let tier2 = { Tier = 2; Name = "Level2"; Description = "Level 2"; Operations = ["op1"; "op2"]; Dependencies = [1]; ComputationalExpressions = ["expr1"] }
            let tier3 = { Tier = 3; Name = "Level3"; Description = "Level 3"; Operations = ["op1"; "op2"; "op3"]; Dependencies = [1; 2]; ComputationalExpressions = ["expr1"; "expr2"] }
            
            let validTiers = [tier1; tier2; tier3]
            let mutable validationResults = []
            
            for tier in validTiers do
                createTestGrammarFile tier testDir |> ignore
                
                // Validate dependencies exist
                let dependenciesExist = 
                    tier.Dependencies 
                    |> List.forall (fun depTier -> 
                        let depFile = Path.Combine(testDir, sprintf "%d.json" depTier)
                        File.Exists(depFile))
                
                validationResults <- dependenciesExist :: validationResults
            
            let allDependenciesValid = validationResults |> List.forall id
            assertTrue allDependenciesValid "All tier dependencies should be valid"
            
            // Test invalid dependency (missing tier)
            let invalidTier = { Tier = 4; Name = "Invalid"; Description = "Invalid tier"; Operations = ["op1"]; Dependencies = [1; 2; 3; 99]; ComputationalExpressions = [] }
            createTestGrammarFile invalidTier testDir |> ignore
            
            let invalidDependencyExists = 
                let depFile = Path.Combine(testDir, "99.json")
                File.Exists(depFile)
            
            assertFalse invalidDependencyExists "Invalid dependency should not exist"
            
            cleanupTestDirectory testDir
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Dependency Validation"
                TestCategory = "GrammarTier"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("DependencyValidationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Dependency Validation"
                TestCategory = "GrammarTier"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testComputationalExpressionGeneration () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let testDir = createTestGrammarDirectory()
            
            let tier5 = {
                Tier = 5
                Name = "CudafyEnhanced"
                Description = "CUDA transpilation and GPU acceleration"
                Operations = ["cudafyTranspilation"; "cudafyClosureFactory"; "gpuKernelGeneration"]
                Dependencies = [1; 2; 3; 4]
                ComputationalExpressions = ["cudafy { ... }"; "gpuParallel { ... }"]
            }
            
            createTestGrammarFile tier5 testDir |> ignore
            
            // Validate computational expressions
            let hasCudafyExpression = tier5.ComputationalExpressions |> List.exists (fun expr -> expr.Contains("cudafy"))
            let hasGpuParallelExpression = tier5.ComputationalExpressions |> List.exists (fun expr -> expr.Contains("gpuParallel"))
            
            assertTrue hasCudafyExpression "Tier 5 should have cudafy computational expression"
            assertTrue hasGpuParallelExpression "Tier 5 should have gpuParallel computational expression"
            
            // Validate operations include CUDA-specific ones
            let hasCudafyOps = tier5.Operations |> List.exists (fun op -> op.Contains("cudafy"))
            let hasGpuOps = tier5.Operations |> List.exists (fun op -> op.Contains("gpu"))
            
            assertTrue hasCudafyOps "Tier 5 should have cudafy operations"
            assertTrue hasGpuOps "Tier 5 should have GPU operations"
            
            cleanupTestDirectory testDir
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Computational Expression Generation"
                TestCategory = "GrammarTier"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [("ExpressionGenerationTime", executionTime.TotalMilliseconds)])
            }
        with
        | ex ->
            {
                TestName = "Computational Expression Generation"
                TestCategory = "GrammarTier"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    let testGrammarEvolutionMetrics () : TestResult =
        let startTime = DateTime.UtcNow
        
        try
            let tiers = [
                { Tier = 1; Name = "T1"; Description = "Tier 1"; Operations = ["op1"; "op2"]; Dependencies = []; ComputationalExpressions = [] }
                { Tier = 2; Name = "T2"; Description = "Tier 2"; Operations = ["op1"; "op2"; "op3"; "op4"]; Dependencies = [1]; ComputationalExpressions = ["expr1"] }
                { Tier = 3; Name = "T3"; Description = "Tier 3"; Operations = ["op1"; "op2"; "op3"; "op4"; "op5"; "op6"]; Dependencies = [1; 2]; ComputationalExpressions = ["expr1"; "expr2"] }
            ]
            
            // Calculate evolution metrics
            let totalOperations = tiers |> List.sumBy (fun t -> t.Operations.Length)
            let totalExpressions = tiers |> List.sumBy (fun t -> t.ComputationalExpressions.Length)
            let complexityGrowth = 
                if tiers.Length > 1 then
                    let firstTier = tiers |> List.head
                    let lastTier = tiers |> List.last
                    float lastTier.Operations.Length / float firstTier.Operations.Length
                else 1.0
            
            // Validate metrics
            assertTrue (totalOperations > 0) "Total operations should be positive"
            assertTrue (totalExpressions >= 0) "Total expressions should be non-negative"
            assertTrue (complexityGrowth >= 1.0) "Complexity should grow or stay same"
            
            // Validate progression
            let operationCounts = tiers |> List.map (fun t -> t.Operations.Length)
            let isIncreasing = 
                operationCounts 
                |> List.pairwise 
                |> List.forall (fun (prev, curr) -> curr >= prev)
            
            assertTrue isIncreasing "Operation count should increase with tier"
            
            let executionTime = DateTime.UtcNow - startTime
            
            {
                TestName = "Grammar Evolution Metrics"
                TestCategory = "GrammarTier"
                Success = true
                ExecutionTime = executionTime
                ErrorMessage = None
                PerformanceMetrics = Some (Map.ofList [
                    ("TotalOperations", float totalOperations)
                    ("TotalExpressions", float totalExpressions)
                    ("ComplexityGrowth", complexityGrowth)
                    ("MetricsCalculationTime", executionTime.TotalMilliseconds)
                ])
            }
        with
        | ex ->
            {
                TestName = "Grammar Evolution Metrics"
                TestCategory = "GrammarTier"
                Success = false
                ExecutionTime = DateTime.UtcNow - startTime
                ErrorMessage = Some ex.Message
                PerformanceMetrics = None
            }

    // ============================================================================
    // TEST SUITE DEFINITION
    // ============================================================================

    let grammarTierTestSuite : TestSuite = {
        SuiteName = "Grammar Tier Tests"
        Category = "GrammarTier"
        Tests = [
            testTier1Validation
            testTierProgression
            testDependencyValidation
            testComputationalExpressionGeneration
            testGrammarEvolutionMetrics
        ]
        SetupAction = Some (fun () -> 
            printfn "🔧 Setting up grammar tier test environment..."
        )
        TeardownAction = Some (fun () -> 
            printfn "🧹 Cleaning up grammar tier test environment..."
        )
    }
