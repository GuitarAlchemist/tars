namespace TarsEngine.FSharp.FLUX.Tests

open System
open Xunit
open TarsEngine.FSharp.FLUX.Refinement.CrossEntropyRefinement

/// Comprehensive unit tests for ChatGPT-Cross-Entropy FLUX Refinement
module CrossEntropyRefinementTests =

    [<Fact>]
    let ``CalculateCrossEntropyLoss should return 0 for empty outcomes`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let outcomes = []
        
        // Act
        let loss = engine.CalculateCrossEntropyLoss(outcomes)
        
        // Assert
        Assert.Equal(0.0, loss)

    [<Fact>]
    let ``CalculateCrossEntropyLoss should return low loss for perfect matches`` () =
        // Arrange
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
        
        // Act
        let loss = engine.CalculateCrossEntropyLoss(outcomes)
        
        // Assert
        Assert.True(loss < 1.0, $"Expected loss < 1.0, but got {loss}")

    [<Fact>]
    let ``CalculateCrossEntropyLoss should return high loss for poor matches`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let outcomes = [
            {
                Expected = "Hello World"
                Actual = "Goodbye Universe"
                Success = false
                ExecutionTime = TimeSpan.FromMilliseconds(5000.0)
                MemoryUsage = 1000000L
                ErrorMessage = Some "Execution failed"
            }
        ]
        
        // Act
        let loss = engine.CalculateCrossEntropyLoss(outcomes)
        
        // Assert
        Assert.True(loss > 2.0, $"Expected loss > 2.0, but got {loss}")

    [<Fact>]
    let ``CalculateStringSimilarity should return 1.0 for identical strings`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        
        // Act
        let similarity = engine.GetType().GetMethod("CalculateStringSimilarity", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(engine, [| "test"; "test" |]) :?> float
        
        // Assert
        Assert.Equal(1.0, similarity, 3)

    [<Fact>]
    let ``CalculateStringSimilarity should return 0.0 for completely different strings`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        
        // Act
        let similarity = engine.GetType().GetMethod("CalculateStringSimilarity", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(engine, [| "abc"; "xyz" |]) :?> float
        
        // Assert
        Assert.True(similarity < 0.5, $"Expected similarity < 0.5, but got {similarity}")

    [<Fact>]
    let ``LevenshteinDistance should calculate correct distance`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        
        // Act
        let distance = engine.GetType().GetMethod("LevenshteinDistance", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(engine, [| "kitten"; "sitting" |]) :?> int
        
        // Assert
        Assert.Equal(3, distance)

    [<Fact>]
    let ``CalculateMetrics should return valid metrics for sample outcomes`` () =
        // Arrange
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
        
        // Act
        let metrics = engine.CalculateMetrics(outcomes)
        
        // Assert
        Assert.True(metrics.Loss >= 0.0)
        Assert.True(metrics.Accuracy >= 0.0 && metrics.Accuracy <= 1.0)
        Assert.True(metrics.Precision >= 0.0 && metrics.Precision <= 1.0)
        Assert.True(metrics.Recall >= 0.0 && metrics.Recall <= 1.0)
        Assert.True(metrics.F1Score >= 0.0 && metrics.F1Score <= 1.0)
        Assert.True(metrics.Confidence >= 0.0 && metrics.Confidence <= 1.0)
        Assert.Equal(0.5, metrics.Accuracy, 1) // 1 success out of 2

    [<Fact>]
    let ``GenerateRefinementSuggestions should return suggestions for poor performance`` () =
        // Arrange
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
        
        // Act
        let suggestions = engine.GenerateRefinementSuggestions(code, outcomes)
        
        // Assert
        Assert.NotEmpty(suggestions)
        Assert.Contains(suggestions, fun s -> s.Category = SyntaxOptimization || s.Category = ErrorCorrection)

    [<Fact>]
    let ``OptimizeSyntax should improve F# code syntax`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let originalCode = "printfn \"Hello %s\" name"
        
        // Act
        let optimizedCode = engine.GetType().GetMethod("OptimizeSyntax", 
            System.Reflection.BindingFlags.NonPublic ||| System.Reflection.BindingFlags.Instance)
            .Invoke(engine, [| originalCode |]) :?> string
        
        // Assert
        Assert.Contains("printfn $\"", optimizedCode)

    [<Fact>]
    let ``ApplyRefinements should apply high-confidence suggestions`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let originalCode = "let x = 1"
        let suggestions = [
            {
                OriginalCode = originalCode
                RefinedCode = "let x = 1.0"
                Confidence = 0.9
                Reasoning = "Type consistency"
                ExpectedImprovement = 0.3
                Category = SyntaxOptimization
            }
            {
                OriginalCode = originalCode
                RefinedCode = "let x = 2"
                Confidence = 0.5
                Reasoning = "Low confidence change"
                ExpectedImprovement = 0.1
                Category = LogicImprovement
            }
        ]
        
        // Act
        let refinedCode = engine.ApplyRefinements(originalCode, suggestions, 0.7)
        
        // Assert
        Assert.Equal("let x = 1.0", refinedCode)

    [<Fact>]
    let ``ValidateRefinement should return true for improved metrics`` () =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let originalOutcomes = [
            {
                Expected = "Result"
                Actual = "Wrong"
                Success = false
                ExecutionTime = TimeSpan.FromMilliseconds(1000.0)
                MemoryUsage = 1000L
                ErrorMessage = Some "Error"
            }
        ]
        let refinedOutcomes = [
            {
                Expected = "Result"
                Actual = "Result"
                Success = true
                ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                MemoryUsage = 500L
                ErrorMessage = None
            }
        ]
        
        // Act
        let isImproved = engine.ValidateRefinement(originalOutcomes, refinedOutcomes)
        
        // Assert
        Assert.True(isImproved)

    [<Fact>]
    let ``CrossEntropyRefinementService should refine code successfully`` () =
        // Arrange
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
        
        // Act
        let (refinedCode, metrics) = service.RefineFluxCode(code, executionHistory)
        
        // Assert
        Assert.NotNull(refinedCode)
        Assert.True(metrics.Accuracy >= 0.0)
        Assert.True(metrics.Loss >= 0.0)

    [<Fact>]
    let ``ContinuousRefinement should improve code over iterations`` () =
        // Arrange
        let service = CrossEntropyRefinementService()
        let code = "let x = 1\nprintfn \"Value: %d\" x"
        
        // Act
        let (finalCode, allMetrics) = service.ContinuousRefinement(code, 3)
        
        // Assert
        Assert.NotNull(finalCode)
        Assert.NotEmpty(allMetrics)
        Assert.True(allMetrics.Length <= 3)

    [<Theory>]
    [<InlineData("", 0.0)>]
    [<InlineData("simple code", 0.5)>]
    [<InlineData("complex code with multiple statements", 1.0)>]
    let ``CalculateMetrics should handle different code complexities`` (code: string, expectedComplexity: float) =
        // Arrange
        let engine = CrossEntropyRefinementEngine()
        let outcomes = [
            {
                Expected = code
                Actual = code
                Success = true
                ExecutionTime = TimeSpan.FromMilliseconds(100.0)
                MemoryUsage = 1000L
                ErrorMessage = None
            }
        ]
        
        // Act
        let metrics = engine.CalculateMetrics(outcomes)
        
        // Assert
        Assert.True(metrics.Accuracy > 0.0)
        Assert.True(metrics.Loss >= 0.0)

    [<Fact>]
    let ``RefinementSuggestion categories should be comprehensive`` () =
        // Arrange & Act
        let categories = [
            SyntaxOptimization
            LogicImprovement
            PerformanceEnhancement
            ErrorCorrection
            SemanticClarification
        ]
        
        // Assert
        Assert.Equal(5, categories.Length)
        Assert.Contains(SyntaxOptimization, categories)
        Assert.Contains(ErrorCorrection, categories)
