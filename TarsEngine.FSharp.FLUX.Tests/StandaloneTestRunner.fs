namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.Threading.Tasks
open TarsEngine.FSharp.FLUX.Refinement.CrossEntropyRefinement
open TarsEngine.FSharp.FLUX.VectorStore.SemanticVectorStore

/// Standalone test runner for ChatGPT-Cross-Entropy and Vector Store Semantics
/// This runs independently of the main TARS system to validate our implementations
module StandaloneTestRunner =

    /// Test result
    type TestResult = {
        TestName: string
        Success: bool
        Message: string
        ExecutionTime: TimeSpan
    }

    /// Run a single test with error handling
    let runTest testName testFunc =
        let startTime = DateTime.UtcNow
        try
            testFunc()
            {
                TestName = testName
                Success = true
                Message = "Test passed"
                ExecutionTime = DateTime.UtcNow - startTime
            }
        with
        | ex ->
            {
                TestName = testName
                Success = false
                Message = sprintf "Test failed: %s" ex.Message
                ExecutionTime = DateTime.UtcNow - startTime
            }

    /// Run async test with error handling
    let runAsyncTest testName testFunc =
        task {
            let startTime = DateTime.UtcNow
            try
                do! testFunc()
                return {
                    TestName = testName
                    Success = true
                    Message = "Test passed"
                    ExecutionTime = DateTime.UtcNow - startTime
                }
            with
            | ex ->
                return {
                    TestName = testName
                    Success = false
                    Message = sprintf "Test failed: %s" ex.Message
                    ExecutionTime = DateTime.UtcNow - startTime
                }
        }

    /// Cross-Entropy Refinement Tests
    let testCrossEntropyRefinement() =
        [
            runTest "CrossEntropy - Calculate Loss for Empty Outcomes" (fun () ->
                let engine = CrossEntropyRefinementEngine()
                let loss = engine.CalculateCrossEntropyLoss([])
                if loss <> 0.0 then failwith "Expected 0.0 loss for empty outcomes"
            )

            runTest "CrossEntropy - Calculate Loss for Perfect Match" (fun () ->
                let engine = CrossEntropyRefinementEngine()
                let outcomes = [
                    {
                        Expected = "Hello World"
                        Actual = "Hello World"
                        Success = true
                        ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                        MemoryUsage = 1000L
                        ErrorMessage = None
                    }
                ]
                let loss = engine.CalculateCrossEntropyLoss(outcomes)
                if loss >= 1.0 then failwith $"Expected low loss for perfect match, got {loss}"
            )

            runTest "CrossEntropy - Generate Refinement Suggestions" (fun () ->
                let engine = CrossEntropyRefinementEngine()
                let code = "let x = 1\nprintfn \"Hello\""
                let outcomes = [
                    {
                        Expected = "Hello"
                        Actual = "Error"
                        Success = false
                        ExecutionTime = TimeSpan.FromMilliseconds(2000.0)
                        MemoryUsage = 1000000L
                        ErrorMessage = Some "Compilation error"
                    }
                ]
                let suggestions = engine.GenerateRefinementSuggestions(code, outcomes)
                if suggestions.IsEmpty then failwith "Expected refinement suggestions for poor performance"
            )

            runTest "CrossEntropy - Calculate Metrics" (fun () ->
                let engine = CrossEntropyRefinementEngine()
                let outcomes = [
                    {
                        Expected = "Result 1"
                        Actual = "Result 1"
                        Success = true
                        ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                        MemoryUsage = 1000L
                        ErrorMessage = None
                    }
                    {
                        Expected = "Result 2"
                        Actual = "Result 3"
                        Success = false
                        ExecutionTime = TimeSpan.FromMilliseconds(200.0)
                        MemoryUsage = 2000L
                        ErrorMessage = Some "Error"
                    }
                ]
                let metrics = engine.CalculateMetrics(outcomes)
                if metrics.Accuracy <> 0.5 then failwith $"Expected 50% accuracy, got {metrics.Accuracy}"
                if metrics.Loss < 0.0 then failwith "Loss should be non-negative"
            )

            runTest "CrossEntropy - Refinement Service" (fun () ->
                let service = CrossEntropyRefinementService()
                let code = "printfn \"Hello\""
                let executionHistory = [
                    {
                        Expected = "Hello"
                        Actual = "Hello"
                        Success = true
                        ExecutionTime = TimeSpan.FromMilliseconds(50.0)
                        MemoryUsage = 500L
                        ErrorMessage = None
                    }
                ]
                let (refinedCode, metrics) = service.RefineFluxCode(code, executionHistory)
                if String.IsNullOrEmpty(refinedCode) then failwith "Refined code should not be empty"
                if metrics.Accuracy < 0.0 then failwith "Accuracy should be non-negative"
            )
        ]

    /// Vector Store Semantics Tests
    let testVectorStoreSemantics() =
        [
            runAsyncTest "VectorStore - Simple Embedding Service" (fun () ->
                task {
                    let service = SimpleEmbeddingService() :> IEmbeddingService
                    let text = "Hello World"
                    let! embedding1 = service.GenerateEmbedding(text)
                    let! embedding2 = service.GenerateEmbedding(text)
                    
                    if embedding1.Length <> embedding2.Length then 
                        failwith "Embeddings should have consistent length"
                    if embedding1.Length <> 384 then 
                        failwith "Expected 384-dimensional embeddings"
                    
                    // Check normalization
                    let magnitude = Math.Sqrt(embedding1 |> Array.map (fun x -> x * x) |> Array.sum)
                    if Math.Abs(magnitude - 1.0) > 0.001 then 
                        failwith $"Expected normalized vector, got magnitude {magnitude}"
                }
            )

            runAsyncTest "VectorStore - Add and Retrieve Vectors" (fun () ->
                task {
                    let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
                    let store = SemanticVectorStore(embeddingService)
                    let content = "let x = 1"
                    
                    let! vectorId = store.AddVectorAsync(content, CodeBlock)
                    let vector = store.GetVector(vectorId)
                    
                    if vector.IsNone then failwith "Vector should be retrievable after adding"
                    if vector.Value.Content <> content then failwith "Vector content should match"
                    if vector.Value.SemanticType <> CodeBlock then failwith "Vector type should match"
                }
            )

            runAsyncTest "VectorStore - Search Similar Vectors" (fun () ->
                task {
                    let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
                    let store = SemanticVectorStore(embeddingService)
                    
                    let! id1 = store.AddVectorAsync("let x = 1", CodeBlock)
                    let! id2 = store.AddVectorAsync("let y = 2", CodeBlock)
                    let! id3 = store.AddVectorAsync("This is documentation", Documentation)
                    
                    let! results = store.SearchSimilarAsync("let z = 3", 2, CodeBlock)
                    
                    if results.Length <> 2 then failwith "Should return exactly 2 results"
                    if not (results |> List.forall (fun r -> r.Vector.SemanticType = CodeBlock)) then
                        failwith "All results should be CodeBlock type"
                }
            )

            runAsyncTest "VectorStore - Semantic Clustering" (fun () ->
                task {
                    let embeddingService = SimpleEmbeddingService() :> IEmbeddingService
                    let store = SemanticVectorStore(embeddingService)
                    
                    // Add multiple vectors
                    let! _ = store.AddVectorAsync("let x = 1", CodeBlock)
                    let! _ = store.AddVectorAsync("let y = 2", CodeBlock)
                    let! _ = store.AddVectorAsync("This is documentation", Documentation)
                    let! _ = store.AddVectorAsync("Another documentation", Documentation)
                    let! _ = store.AddVectorAsync("Error occurred", ErrorMessage)
                    let! _ = store.AddVectorAsync("Another error", ErrorMessage)
                    
                    let clusters = store.PerformSemanticClustering(3)
                    
                    if clusters.Length <> 3 then failwith "Should create exactly 3 clusters"
                    if not (clusters |> List.forall (fun c -> c.Coherence >= 0.0 && c.Coherence <= 1.0)) then
                        failwith "Cluster coherence should be between 0 and 1"
                }
            )

            runAsyncTest "VectorStore - Service Integration" (fun () ->
                task {
                    let service = SemanticVectorStoreService()
                    let code = "let result = 42"
                    
                    let! vectorId = service.AddFluxCodeAsync(code)
                    let! results = service.SearchSimilarCodeAsync("let value = 43", 5)
                    let insights = service.GetSemanticInsights()
                    
                    if String.IsNullOrEmpty(vectorId) then failwith "Vector ID should not be empty"
                    if results.IsEmpty then failwith "Should find similar code"
                    if not (insights.ContainsKey("TotalVectors")) then failwith "Insights should contain TotalVectors"
                }
            )
        ]

    /// Integration Tests
    let testIntegration() =
        [
            runAsyncTest "Integration - Cross-Entropy with Vector Store" (fun () ->
                task {
                    let vectorStoreService = SemanticVectorStoreService()
                    let refinementService = CrossEntropyRefinementService()
                    
                    let originalCode = "let x = 1\nprintfn \"Value: %d\" x"
                    let! codeVectorId = vectorStoreService.AddFluxCodeAsync(originalCode)
                    
                    let executionHistory = [
                        {
                            Expected = "Value: 1"
                            Actual = "Value: 1"
                            Success = true
                            ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                            MemoryUsage = 1000L
                            ErrorMessage = None
                        }
                    ]
                    
                    let (refinedCode, metrics) = refinementService.RefineFluxCode(originalCode, executionHistory)
                    let! refinedVectorId = vectorStoreService.AddFluxCodeAsync(refinedCode)
                    let! similarCode = vectorStoreService.SearchSimilarCodeAsync(originalCode, 5)
                    
                    if codeVectorId = refinedVectorId then failwith "Vector IDs should be different"
                    if similarCode.IsEmpty then failwith "Should find similar code"
                    if metrics.Accuracy <= 0.0 then failwith "Accuracy should be positive"
                }
            )
        ]

    /// Run all tests
    let runAllTests() =
        task {
            printfn "🧪 TARS ChatGPT-Cross-Entropy & Vector Store Semantics Tests"
            printfn "============================================================"
            printfn ""
            
            let mutable allResults = []
            
            // Cross-Entropy Tests
            printfn "🔬 Running Cross-Entropy Refinement Tests..."
            let crossEntropyResults = testCrossEntropyRefinement()
            allResults <- allResults @ crossEntropyResults
            
            // Vector Store Tests
            printfn "🗃️  Running Vector Store Semantics Tests..."
            let! vectorStoreResults = 
                testVectorStoreSemantics()
                |> List.map (fun taskResult -> taskResult.Result)
                |> Task.WhenAll
            allResults <- allResults @ (vectorStoreResults |> Array.toList)
            
            // Integration Tests
            printfn "🔗 Running Integration Tests..."
            let! integrationResults = 
                testIntegration()
                |> List.map (fun taskResult -> taskResult.Result)
                |> Task.WhenAll
            allResults <- allResults @ (integrationResults |> Array.toList)
            
            // Report Results
            printfn ""
            printfn "📊 TEST RESULTS"
            printfn "==============="
            
            let passed = allResults |> List.filter (fun r -> r.Success) |> List.length
            let failed = allResults |> List.filter (fun r -> not r.Success) |> List.length
            let totalTime = allResults |> List.map (fun r -> r.ExecutionTime.TotalMilliseconds) |> List.sum
            
            for result in allResults do
                let status = if result.Success then "✅ PASS" else "❌ FAIL"
                let time = sprintf "%.1fms" result.ExecutionTime.TotalMilliseconds
                printfn "%s | %s | %s | %s" status result.TestName time result.Message
            
            printfn ""
            printfn "Summary: %d passed, %d failed, %.1fms total" passed failed totalTime
            printfn ""
            
            if failed = 0 then
                printfn "🎉 ALL TESTS PASSED! ChatGPT-Cross-Entropy & Vector Store Semantics are working correctly!"
            else
                printfn "⚠️  Some tests failed. Please review the implementation."
            
            return (passed, failed)
        }

    /// Entry point for standalone testing
    [<EntryPoint>]
    let main argv =
        task {
            let! (passed, failed) = runAllTests()
            return if failed = 0 then 0 else 1
        } |> Async.AwaitTask |> Async.RunSynchronously
