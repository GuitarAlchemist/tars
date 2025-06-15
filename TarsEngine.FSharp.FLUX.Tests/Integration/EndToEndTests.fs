namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.IO
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Unquote
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Parser.FluxParser
open TarsEngine.FSharp.FLUX.Execution.FluxRuntime
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// End-to-end integration tests for FLUX
module EndToEndTests =
    
    [<Fact>]
    let ``FLUX Engine can execute complete metascript`` () =
        task {
            // Arrange
            let fluxContent = TestHelpers.createTestFluxFileContent()
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(fluxContent)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThan 0)
            result.ExecutionTime |> should be (lessThan (TimeSpan.FromSeconds(10.0)))
        }
    
    [<Fact>]
    let ``FLUX Engine can execute script from file`` () =
        task {
            // Arrange
            let fluxContent = TestHelpers.createTestFluxFileContent()
            let tempFile = TestHelpers.createTempFile fluxContent ".flux"
            let engine = FluxEngine.FluxEngine()
            
            try
                // Act
                let! result = engine.ExecuteFile(tempFile)
                
                // Assert
                TestHelpers.assertExecutionSuccess result
                result.BlocksExecuted |> should be (greaterThan 0)
            finally
                TestHelpers.cleanupTempFile tempFile
        }
    
    [<Fact>]
    let ``FLUX Engine handles parsing errors gracefully`` () =
        task {
            // Arrange
            let invalidFluxContent = "INVALID { this is not valid FLUX syntax }"
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(invalidFluxContent)
            
            // Assert
            TestHelpers.assertExecutionFailure result
            result.ErrorMessage |> should not' (equal None)
        }
    
    [<Fact>]
    let ``FLUX Engine can execute multiple language blocks`` () =
        task {
            // Arrange
            let multiLanguageScript = """
META {
    title: "Multi-Language Test"
    version: "1.0.0"
}

FSHARP {
    let fsharpResult = "F# executed successfully"
    printfn "%s" fsharpResult
}

PYTHON {
    python_result = "Python executed successfully"
    print(python_result)
}

JAVASCRIPT {
    const jsResult = "JavaScript executed successfully";
    console.log(jsResult);
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(multiLanguageScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThanOrEqualTo 4) // META + 3 language blocks
        }
    
    [<Fact>]
    let ``FLUX Engine can execute agent orchestration`` () =
        task {
            // Arrange
            let agentScript = """
META {
    title: "Agent Orchestration Test"
    version: "1.0.0"
}

AGENT DataProcessor {
    role: "Data Processing Agent"
    capabilities: ["data_processing", "validation", "transformation"]
    reflection: true
    planning: true
    
    FSHARP {
        let processData data = 
            data |> List.map (fun x -> x * 2)
        let testData = [1; 2; 3; 4; 5]
        let result = processData testData
        printfn "Processed data: %A" result
    }
}

AGENT Validator {
    role: "Validation Agent"
    capabilities: ["validation", "testing"]
    
    FSHARP {
        let validateResult result =
            result |> List.forall (fun x -> x > 0)
        printfn "Validation completed"
    }
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(agentScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            TestHelpers.assertAgentOutput "DataProcessor" result
            TestHelpers.assertAgentOutput "Validator" result
        }
    
    [<Fact>]
    let ``FLUX Engine can execute diagnostic operations`` () =
        task {
            // Arrange
            let diagnosticScript = """
META {
    title: "Diagnostic Test"
    version: "1.0.0"
}

FSHARP {
    let systemStatus = "operational"
    let performanceMetric = 95.5
    printfn "System: %s, Performance: %.1f%%" systemStatus performanceMetric
}

DIAGNOSTIC {
    test: "System functionality test"
    validate: "Performance metrics within acceptable range"
    benchmark: "Response time under 100ms"
    assert: ("systemStatus = 'operational'", "System must be operational")
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(diagnosticScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.DiagnosticResults.Count |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FLUX Engine can execute reflection operations`` () =
        task {
            // Arrange
            let reflectionScript = """
META {
    title: "Reflection Test"
    version: "1.0.0"
}

FSHARP {
    let codeQuality = "good"
    let complexity = "medium"
    printfn "Code quality: %s, Complexity: %s" codeQuality complexity
}

REFLECT {
    analyze: "Code quality and complexity metrics"
    plan: "Optimization strategy for better performance"
    improve: ("performance", "caching and memoization")
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(reflectionScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.ReflectionInsights.Length |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FLUX Engine can execute reasoning blocks`` () =
        task {
            // Arrange
            let reasoningScript = """
META {
    title: "Reasoning Test"
    version: "1.0.0"
}

FSHARP {
    let problem = "optimization"
    let approach = "gradient descent"
    printfn "Problem: %s, Approach: %s" problem approach
}

REASONING {
    The optimization problem requires a systematic approach.
    We will use gradient descent to find the optimal solution.
    This reasoning block demonstrates the system's ability
    to process natural language reasoning alongside code execution.
    
    Key considerations:
    1. Performance requirements
    2. Accuracy constraints
    3. Resource limitations
    4. Scalability needs
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(reasoningScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThanOrEqualTo 3) // META + FSHARP + REASONING
        }
    
    [<Fact>]
    let ``FLUX Engine can handle complex nested scenarios`` () =
        task {
            // Arrange
            let complexScript = """
META {
    title: "Complex Integration Test"
    version: "1.0.0"
    description: "Tests complex nested agent and diagnostic scenarios"
}

AGENT DataAnalyst {
    role: "Data Analysis Specialist"
    capabilities: ["analysis", "statistics", "visualization"]
    reflection: true
    
    FSHARP {
        let analyzeData data =
            let mean = data |> List.average
            let variance = data |> List.map (fun x -> (x - mean) ** 2.0) |> List.average
            (mean, variance)
        
        let testData = [1.0; 2.0; 3.0; 4.0; 5.0]
        let (mean, variance) = analyzeData testData
        printfn "Analysis: Mean=%.2f, Variance=%.2f" mean variance
    }
    
    PYTHON {
        import statistics
        data = [1, 2, 3, 4, 5]
        mean = statistics.mean(data)
        variance = statistics.variance(data)
        print(f"Python Analysis: Mean={mean:.2f}, Variance={variance:.2f}")
    }
}

DIAGNOSTIC {
    test: "Data analysis accuracy"
    validate: "Statistical calculations are correct"
    benchmark: "Analysis completes within time limit"
}

REFLECT {
    analyze: "Multi-language data analysis implementation"
    diff: ("single_language", "multi_language")
    plan: "Optimization strategy for cross-language data processing"
}

REASONING {
    This complex scenario demonstrates the FLUX system's ability
    to orchestrate multiple agents, execute code in different languages,
    perform diagnostic validation, and conduct reflection analysis.
    
    The data analysis is performed in both F# and Python to showcase
    cross-language capabilities and result consistency validation.
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(complexScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThanOrEqualTo 6)
            TestHelpers.assertAgentOutput "DataAnalyst" result
            result.DiagnosticResults.Count |> should be (greaterThan 0)
            result.ReflectionInsights.Length |> should be (greaterThan 0)
            TestHelpers.assertExecutionTime (TimeSpan.FromSeconds(30.0)) result
        }
    
    [<Fact>]
    let ``FLUX Engine maintains execution context across blocks`` () =
        task {
            // Arrange
            let contextScript = """
META {
    title: "Context Persistence Test"
    version: "1.0.0"
}

FSHARP {
    let sharedValue = 42
    let sharedMessage = "Hello from first block"
    printfn "Set shared values: %d, %s" sharedValue sharedMessage
}

FSHARP {
    // This should have access to variables from previous block
    let combinedValue = sharedValue + 8
    let combinedMessage = sharedMessage + " - processed in second block"
    printfn "Combined values: %d, %s" combinedValue combinedMessage
}
"""
            let engine = FluxEngine.FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(contextScript)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 3 // META + 2 FSHARP blocks
        }

printfn "🧪 End-to-End Integration Tests Loaded"
printfn "======================================"
printfn "✅ Complete metascript execution tests"
printfn "✅ Multi-language execution tests"
printfn "✅ Agent orchestration tests"
printfn "✅ Diagnostic operation tests"
printfn "✅ Reflection capability tests"
printfn "✅ Complex nested scenario tests"
printfn "✅ Context persistence tests"
