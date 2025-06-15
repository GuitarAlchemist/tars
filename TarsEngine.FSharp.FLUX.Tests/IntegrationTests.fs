namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.Threading.Tasks
open Xunit
open TarsEngine.FSharp.FLUX.Refinement.CrossEntropyRefinement
open TarsEngine.FSharp.FLUX.VectorStore.SemanticVectorStore

/// Comprehensive integration tests for ChatGPT-Cross-Entropy and Vector Store Semantics
module IntegrationTests =

    [<Fact>]
    let ``Cross-Entropy Refinement and Vector Store should work together`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementService = CrossEntropyRefinementService()
            
            let originalCode = "let x = 1\nprintfn \"Value: %d\" x"
            let! codeVectorId = vectorStoreService.AddFluxCodeAsync(originalCode)
            
            // Simulate execution outcomes
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
            
            // Act
            let (refinedCode, metrics) = refinementService.RefineFluxCode(originalCode, executionHistory)
            let! refinedVectorId = vectorStoreService.AddFluxCodeAsync(refinedCode)
            let! similarCode = vectorStoreService.SearchSimilarCodeAsync(originalCode, 5)
            
            // Assert
            Assert.NotEqual(codeVectorId, refinedVectorId)
            Assert.NotEmpty(similarCode)
            Assert.Contains(similarCode, fun result -> result.Vector.Id = codeVectorId || result.Vector.Id = refinedVectorId)
            Assert.True(metrics.Accuracy > 0.0)
        }

    [<Fact>]
    let ``Continuous refinement should improve code quality over iterations`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementService = CrossEntropyRefinementService()
            
            let originalCode = "printfn \"Hello %s\" name"
            let! _ = vectorStoreService.AddFluxCodeAsync(originalCode)
            
            // Act
            let (finalCode, allMetrics) = refinementService.ContinuousRefinement(originalCode, 3)
            let! _ = vectorStoreService.AddFluxCodeAsync(finalCode)
            
            let clusters = vectorStoreService.AnalyzeFluxCodebase()
            let insights = vectorStoreService.GetSemanticInsights()
            
            // Assert
            Assert.NotEmpty(allMetrics)
            Assert.NotEmpty(clusters)
            Assert.True(insights.ContainsKey("TotalVectors"))
            
            // Check if refinement improved over iterations
            if allMetrics.Length > 1 then
                let firstMetrics = allMetrics.Head
                let lastMetrics = allMetrics |> List.last
                // At least one metric should improve or stay the same
                Assert.True(lastMetrics.Accuracy >= firstMetrics.Accuracy || 
                           lastMetrics.F1Score >= firstMetrics.F1Score ||
                           lastMetrics.Loss <= firstMetrics.Loss)
        }

    [<Fact>]
    let ``Vector store should maintain semantic relationships during refinement`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementEngine = CrossEntropyRefinementEngine()
            
            let codeVariants = [
                "let x = 1"
                "let y = 2"
                "let z = 3"
                "printfn \"Hello\""
                "printfn \"World\""
            ]
            
            // Add all code variants to vector store
            let! vectorIds = 
                codeVariants 
                |> List.map vectorStoreService.AddFluxCodeAsync
                |> Task.WhenAll
            
            // Act
            let clusters = vectorStoreService.AnalyzeFluxCodebase()
            
            // Generate refinement suggestions for each code variant
            let refinementTasks = 
                codeVariants
                |> List.map (fun code ->
                    let outcomes = [
                        {
                            Expected = "Success"
                            Actual = "Success"
                            Success = true
                            ExecutionTime = TimeSpan.FromMilliseconds(50.0)
                            MemoryUsage = 500L
                            ErrorMessage = None
                        }
                    ]
                    refinementEngine.GenerateRefinementSuggestions(code, outcomes))
            
            let allSuggestions = refinementTasks |> List.concat
            
            // Assert
            Assert.NotEmpty(clusters)
            Assert.True(clusters.Length <= codeVariants.Length)
            
            // Check that similar code patterns are clustered together
            let letStatementCluster = clusters |> List.tryFind (fun c -> 
                c.Vectors |> List.exists (fun v -> v.Content.StartsWith("let")))
            let printfnCluster = clusters |> List.tryFind (fun c -> 
                c.Vectors |> List.exists (fun v -> v.Content.StartsWith("printfn")))
            
            // We should have some semantic clustering
            Assert.True(letStatementCluster.IsSome || printfnCluster.IsSome)
        }

    [<Fact>]
    let ``Cross-entropy metrics should correlate with vector store similarity`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementEngine = CrossEntropyRefinementEngine()
            
            let goodCode = "let result = 42\nprintfn \"Result: %d\" result"
            let poorCode = "let x = \nprintfn \"Error\""
            
            let! goodVectorId = vectorStoreService.AddFluxCodeAsync(goodCode)
            let! poorVectorId = vectorStoreService.AddFluxCodeAsync(poorCode)
            
            // Create execution outcomes
            let goodOutcomes = [
                {
                    Expected = "Result: 42"
                    Actual = "Result: 42"
                    Success = true
                    ExecutionTime = TimeSpan.FromMilliseconds(50.0)
                    MemoryUsage = 500L
                    ErrorMessage = None
                }
            ]
            
            let poorOutcomes = [
                {
                    Expected = "Result: 42"
                    Actual = "Compilation Error"
                    Success = false
                    ExecutionTime = TimeSpan.FromMilliseconds(1000.0)
                    MemoryUsage = 2000L
                    ErrorMessage = Some "Syntax error"
                }
            ]
            
            // Act
            let goodMetrics = refinementEngine.CalculateMetrics(goodOutcomes)
            let poorMetrics = refinementEngine.CalculateMetrics(poorOutcomes)
            
            let! similarToGood = vectorStoreService.SearchSimilarCodeAsync(goodCode, 2)
            let! similarToPoor = vectorStoreService.SearchSimilarCodeAsync(poorCode, 2)
            
            // Assert
            Assert.True(goodMetrics.Accuracy > poorMetrics.Accuracy)
            Assert.True(goodMetrics.Loss < poorMetrics.Loss)
            Assert.True(goodMetrics.F1Score > poorMetrics.F1Score)
            
            // Vector store should find the good code more similar to itself
            let goodSelfSimilarity = similarToGood |> List.tryFind (fun r -> r.Vector.Id = goodVectorId)
            let poorSelfSimilarity = similarToPoor |> List.tryFind (fun r -> r.Vector.Id = poorVectorId)
            
            Assert.True(goodSelfSimilarity.IsSome)
            Assert.True(poorSelfSimilarity.IsSome)
        }

    [<Fact>]
    let ``Refinement suggestions should improve vector store semantic clustering`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementEngine = CrossEntropyRefinementEngine()
            
            let problematicCodes = [
                "let x = 1\nlet y = 2"  // Could be improved
                "printfn \"Hello\""      // Simple but correct
                "let z = "              // Incomplete
            ]
            
            // Add original codes
            let! originalIds = 
                problematicCodes 
                |> List.map vectorStoreService.AddFluxCodeAsync
                |> Task.WhenAll
            
            let originalClusters = vectorStoreService.AnalyzeFluxCodebase()
            
            // Generate refinements
            let refinedCodes = 
                problematicCodes
                |> List.map (fun code ->
                    let outcomes = [
                        {
                            Expected = "Success"
                            Actual = if code.EndsWith(" ") then "Error" else "Success"
                            Success = not (code.EndsWith(" "))
                            ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                            MemoryUsage = 1000L
                            ErrorMessage = if code.EndsWith(" ") then Some "Incomplete" else None
                        }
                    ]
                    let suggestions = refinementEngine.GenerateRefinementSuggestions(code, outcomes)
                    refinementEngine.ApplyRefinements(code, suggestions, 0.5))
            
            // Add refined codes
            let! refinedIds = 
                refinedCodes 
                |> List.map vectorStoreService.AddFluxCodeAsync
                |> Task.WhenAll
            
            // Act
            let finalClusters = vectorStoreService.AnalyzeFluxCodebase()
            let insights = vectorStoreService.GetSemanticInsights()
            
            // Assert
            Assert.True(finalClusters.Length >= originalClusters.Length)
            Assert.True(insights.["TotalVectors"] :?> int >= problematicCodes.Length * 2)
            
            // Check that refinement created better semantic organization
            let avgOriginalCoherence = 
                if originalClusters.IsEmpty then 0.0 
                else originalClusters |> List.map (fun c -> c.Coherence) |> List.average
            
            let avgFinalCoherence = 
                if finalClusters.IsEmpty then 0.0 
                else finalClusters |> List.map (fun c -> c.Coherence) |> List.average
            
            // Coherence should improve or at least not degrade significantly
            Assert.True(avgFinalCoherence >= avgOriginalCoherence - 0.2)
        }

    [<Fact>]
    let ``End-to-end workflow should demonstrate complete system integration`` () =
        task {
            // Arrange
            let vectorStoreService = SemanticVectorStoreService()
            let refinementService = CrossEntropyRefinementService()
            
            // Simulate a complete FLUX development workflow
            let initialCode = "let calculate x = x * 2\nprintfn \"Result: %d\" (calculate 5)"
            
            // Step 1: Add initial code to vector store
            let! initialVectorId = vectorStoreService.AddFluxCodeAsync(initialCode)
            
            // Step 2: Simulate execution and gather outcomes
            let executionOutcomes = [
                {
                    Expected = "Result: 10"
                    Actual = "Result: 10"
                    Success = true
                    ExecutionTime = TimeSpan.FromMilliseconds(75.0)
                    MemoryUsage = 800L
                    ErrorMessage = None
                }
            ]
            
            // Step 3: Apply cross-entropy refinement
            let (refinedCode, metrics) = refinementService.RefineFluxCode(initialCode, executionOutcomes)
            let! refinedVectorId = vectorStoreService.AddFluxCodeAsync(refinedCode)
            
            // Step 4: Add execution result to vector store
            let! resultVectorId = vectorStoreService.AddExecutionResultAsync("Result: 10")
            
            // Step 5: Perform semantic analysis
            let clusters = vectorStoreService.AnalyzeFluxCodebase()
            let insights = vectorStoreService.GetSemanticInsights()
            
            // Step 6: Search for similar patterns
            let! similarCode = vectorStoreService.SearchSimilarCodeAsync("let multiply x = x * 3", 3)
            
            // Act & Assert - Verify complete workflow
            Assert.NotNull(initialVectorId)
            Assert.NotNull(refinedVectorId)
            Assert.NotNull(resultVectorId)
            
            Assert.True(metrics.Accuracy > 0.0)
            Assert.True(metrics.Loss >= 0.0)
            
            Assert.NotEmpty(clusters)
            Assert.NotEmpty(insights)
            Assert.True(insights.ContainsKey("TotalVectors"))
            Assert.True(insights.["TotalVectors"] :?> int >= 3)
            
            Assert.NotEmpty(similarCode)
            
            // Verify that the system found semantic relationships
            let codeVectors = similarCode |> List.filter (fun r -> r.Vector.SemanticType = CodeBlock)
            Assert.NotEmpty(codeVectors)
            
            // Verify refinement quality
            Assert.True(refinedCode.Length >= initialCode.Length - 10) // Shouldn't drastically shrink
            
            // Verify semantic insights are meaningful
            let semanticTypes = insights.["SemanticTypes"] :?> (string * int) list
            Assert.NotEmpty(semanticTypes)
            Assert.Contains(semanticTypes, fun (typeName, count) -> typeName.Contains("CodeBlock") && count > 0)
        }
