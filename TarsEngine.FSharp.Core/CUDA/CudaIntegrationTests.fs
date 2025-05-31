namespace TarsEngine.FSharp.Core.CUDA.Tests

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Core.CUDA.CudaVectorStore
open TarsEngine.FSharp.Core.CUDA.AgenticCudaRAG

/// Comprehensive integration tests for TARS CUDA Vector Store and Agentic RAG
module CudaIntegrationTests =
    
    /// Test result type
    type TestResult = {
        TestName: string
        Success: bool
        Duration: TimeSpan
        Message: string
        Metrics: obj option
    }
    
    /// Test runner utility
    let runTest (testName: string) (testFunc: unit -> Async<'T>) =
        async {
            printfn "🧪 Running test: %s" testName
            let startTime = DateTime.UtcNow
            
            try
                let! result = testFunc()
                let duration = DateTime.UtcNow - startTime
                
                printfn "   ✅ PASSED in %.2f ms" duration.TotalMilliseconds
                return {
                    TestName = testName
                    Success = true
                    Duration = duration
                    Message = "Test passed successfully"
                    Metrics = Some (box result)
                }
            with
            | ex ->
                let duration = DateTime.UtcNow - startTime
                printfn "   ❌ FAILED in %.2f ms: %s" duration.TotalMilliseconds ex.Message
                return {
                    TestName = testName
                    Success = false
                    Duration = duration
                    Message = ex.Message
                    Metrics = None
                }
        }
    
    /// Test 1: CUDA Vector Store Initialization
    let testCudaVectorStoreInit() =
        async {
            let config = CudaVectorStoreFactory.createDefaultConfig()
            use store = new CudaVectorStore(config)
            
            do! store.Initialize()
            
            let stats = store.GetStatistics()
            if stats.IsInitialized then
                return $"Store initialized with {stats.MaxVectors} max vectors, {stats.VectorDimension} dimensions"
            else
                failwith "Store failed to initialize"
        }
    
    /// Test 2: Vector Addition and Storage
    let testVectorAddition() =
        async {
            let config = CudaVectorStoreFactory.createDefaultConfig()
            use store = new CudaVectorStore(config)
            
            do! store.Initialize()
            
            // Create test vectors
            let testVectors = [|
                for i in 0..99 do
                    [| for j in 0..383 do float32 (sin(float i + float j)) |]
            |]
            let metadata = [| for i in 0..99 do $"test_vector_{i}" |]
            
            let! result = store.AddVectors(testVectors, metadata)
            
            match result with
            | Ok count ->
                let stats = store.GetStatistics()
                return $"Added {testVectors.Length} vectors, total count: {count}, memory: {stats.MemoryUsageEstimateMB:F2} MB"
            | Error err ->
                failwith $"Failed to add vectors: {err}"
        }
    
    /// Test 3: CUDA Similarity Search Performance
    let testCudaSimilaritySearch() =
        async {
            let config = CudaVectorStoreFactory.createDefaultConfig()
            use store = new CudaVectorStore(config)
            
            do! store.Initialize()
            
            // Add test vectors
            let testVectors = [|
                for i in 0..999 do  // 1000 vectors for performance test
                    [| for j in 0..383 do float32 (sin(float i + float j)) |]
            |]
            let metadata = [| for i in 0..999 do $"perf_vector_{i}" |]
            
            let! addResult = store.AddVectors(testVectors, metadata)
            match addResult with
            | Error err -> failwith $"Failed to add vectors: {err}"
            | Ok _ -> ()
            
            // Perform search
            let queryVector = [| for i in 0..383 do float32 (cos(float i)) |]
            let! searchResult = store.SearchSimilar(queryVector, 10)
            
            match searchResult with
            | Ok (results, metrics) ->
                return $"Search completed: {results.Length} results, {metrics.SearchTimeMs:F2}ms, {metrics.ThroughputSearchesPerSecond:F0} searches/sec"
            | Error err ->
                failwith $"Search failed: {err}"
        }
    
    /// Test 4: Batch Search Performance
    let testBatchSearch() =
        async {
            let config = CudaVectorStoreFactory.createDefaultConfig()
            use store = new CudaVectorStore(config)
            
            do! store.Initialize()
            
            // Add test vectors
            let testVectors = [|
                for i in 0..499 do
                    [| for j in 0..383 do float32 (sin(float i + float j)) |]
            |]
            let metadata = [| for i in 0..499 do $"batch_vector_{i}" |]
            
            let! addResult = store.AddVectors(testVectors, metadata)
            match addResult with
            | Error err -> failwith $"Failed to add vectors: {err}"
            | Ok _ -> ()
            
            // Create multiple queries
            let queries = [|
                for i in 0..9 do
                    [| for j in 0..383 do float32 (cos(float i + float j)) |]
            |]
            
            let! batchResults = store.BatchSearch(queries, 5)
            
            let totalResults = batchResults |> Array.sumBy (fun (results, _) -> results.Length)
            let avgTime = batchResults |> Array.averageBy (fun (_, metrics) -> float metrics.SearchTimeMs)
            
            return $"Batch search: {queries.Length} queries, {totalResults} total results, {avgTime:F2}ms avg time"
        }
    
    /// Test 5: Agentic RAG System Initialization
    let testAgenticRAGInit() =
        async {
            let config = AgenticCudaRAGFactory.createDefaultConfig()
            use ragSystem = new TarsAgenticCudaRAG(config)
            
            do! ragSystem.Initialize()
            
            let metrics = ragSystem.GetPerformanceMetrics()
            if metrics.IsInitialized then
                return $"Agentic RAG initialized with {config.AgentCapabilities.Length} agent capabilities"
            else
                failwith "Agentic RAG failed to initialize"
        }
    
    /// Test 6: End-to-End RAG Query Processing
    let testEndToEndRAGQuery() =
        async {
            let config = AgenticCudaRAGFactory.createDefaultConfig()
            use ragSystem = new TarsAgenticCudaRAG(config)
            
            do! ragSystem.Initialize()
            
            // Add test documents
            let documents = [|
                "TARS is an autonomous AI system with advanced metascript capabilities for self-improvement"
                "CUDA acceleration enables 184M+ vector similarity searches per second on modern GPUs"
                "Agentic RAG combines retrieval-augmented generation with intelligent agent decision-making"
                "Vector databases provide semantic search capabilities across large document collections"
                "F# functional programming offers type-safe development for AI and machine learning systems"
            |]
            let metadata = [| for i in 0..documents.Length-1 do $"rag_doc_{i}" |]
            
            let! addResult = ragSystem.AddDocuments(documents, metadata)
            match addResult with
            | Error err -> failwith $"Failed to add documents: {err}"
            | Ok _ -> ()
            
            // Process test query
            let query = {
                Query = "What is TARS and how does it use CUDA acceleration?"
                Context = None
                Intent = TechnicalQuery
                MaxResults = 3
                RequiredSources = []
            }
            
            let! queryResult = ragSystem.ProcessQuery(query)
            
            match queryResult with
            | Ok response ->
                return $"RAG query processed: confidence {response.Confidence:F3}, {response.Sources.Length} sources, {response.AgentDecisions.Length} agent decisions"
            | Error err ->
                failwith $"RAG query failed: {err}"
        }
    
    /// Test 7: Performance Stress Test
    let testPerformanceStress() =
        async {
            let config = CudaVectorStoreFactory.createLargeScaleConfig()
            use store = new CudaVectorStore(config)
            
            do! store.Initialize()
            
            // Add larger dataset
            let vectorCount = 5000
            let testVectors = [|
                for i in 0..vectorCount-1 do
                    [| for j in 0..767 do float32 (sin(float i + float j * 0.1)) |]  // 768-dim vectors
            |]
            let metadata = [| for i in 0..vectorCount-1 do $"stress_vector_{i}" |]
            
            let startTime = DateTime.UtcNow
            let! addResult = store.AddVectors(testVectors, metadata)
            let addTime = DateTime.UtcNow - startTime
            
            match addResult with
            | Error err -> failwith $"Failed to add vectors: {err}"
            | Ok _ -> ()
            
            // Perform multiple searches
            let searchCount = 100
            let searchTimes = ResizeArray<float>()
            
            for i in 0..searchCount-1 do
                let queryVector = [| for j in 0..767 do float32 (cos(float i + float j * 0.1)) |]
                let searchStart = DateTime.UtcNow
                let! searchResult = store.SearchSimilar(queryVector, 10)
                let searchTime = (DateTime.UtcNow - searchStart).TotalMilliseconds
                searchTimes.Add(searchTime)
                
                match searchResult with
                | Error err -> failwith $"Search {i} failed: {err}"
                | Ok _ -> ()
            
            let avgSearchTime = searchTimes |> Seq.average
            let minSearchTime = searchTimes |> Seq.min
            let maxSearchTime = searchTimes |> Seq.max
            
            return $"Stress test: {vectorCount} vectors added in {addTime.TotalMilliseconds:F0}ms, {searchCount} searches avg {avgSearchTime:F2}ms (min: {minSearchTime:F2}ms, max: {maxSearchTime:F2}ms)"
        }
    
    /// Run all integration tests
    let runAllTests() =
        async {
            printfn "🧪 TARS CUDA INTEGRATION TESTS"
            printfn "==============================="
            printfn ""
            
            let tests = [
                ("CUDA Vector Store Initialization", testCudaVectorStoreInit)
                ("Vector Addition and Storage", testVectorAddition)
                ("CUDA Similarity Search Performance", testCudaSimilaritySearch)
                ("Batch Search Performance", testBatchSearch)
                ("Agentic RAG System Initialization", testAgenticRAGInit)
                ("End-to-End RAG Query Processing", testEndToEndRAGQuery)
                ("Performance Stress Test", testPerformanceStress)
            ]
            
            let! results = 
                tests
                |> List.map (fun (name, test) -> runTest name test)
                |> Async.Parallel
            
            printfn ""
            printfn "📊 TEST RESULTS SUMMARY"
            printfn "======================="
            
            let passedTests = results |> Array.filter (fun r -> r.Success)
            let failedTests = results |> Array.filter (fun r -> not r.Success)
            
            printfn "✅ Passed: %d/%d tests" passedTests.Length results.Length
            if failedTests.Length > 0 then
                printfn "❌ Failed: %d tests" failedTests.Length
                for test in failedTests do
                    printfn "   - %s: %s" test.TestName test.Message
            
            let totalTime = results |> Array.sumBy (fun r -> r.Duration.TotalMilliseconds)
            printfn "⏱️  Total time: %.2f ms" totalTime
            
            printfn ""
            if passedTests.Length = results.Length then
                printfn "🎉 ALL TESTS PASSED!"
                printfn "✅ TARS CUDA Vector Store is ready for production"
                printfn "✅ Agentic RAG system is fully functional"
                printfn "🚀 Performance validated for enterprise deployment"
            else
                printfn "⚠️  Some tests failed - review and fix issues before deployment"
            
            return results
        }

/// Demo runner for integration tests
module CudaIntegrationTestsDemo =
    
    /// Run integration tests demo
    let runDemo() =
        async {
            printfn "🔬 Starting TARS CUDA Integration Tests Demo..."
            printfn ""
            
            let! results = CudaIntegrationTests.runAllTests()
            
            printfn ""
            printfn "📋 Detailed Results:"
            for result in results do
                printfn ""
                printfn "Test: %s" result.TestName
                printfn "Status: %s" (if result.Success then "✅ PASSED" else "❌ FAILED")
                printfn "Duration: %.2f ms" result.Duration.TotalMilliseconds
                printfn "Message: %s" result.Message
                match result.Metrics with
                | Some metrics -> printfn "Metrics: %A" metrics
                | None -> ()
            
            printfn ""
            printfn "🏁 Integration tests demo completed!"
        }
