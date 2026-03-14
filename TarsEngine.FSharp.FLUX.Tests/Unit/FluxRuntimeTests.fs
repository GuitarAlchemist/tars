namespace TarsEngine.FSharp.FLUX.Tests.Unit

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open Unquote
open TarsEngine.FSharp.FLUX.Ast.FluxAst
open TarsEngine.FSharp.FLUX.Execution.FluxRuntime
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Unit tests for FLUX Runtime
module FluxRuntimeTests =
    
    [<Fact>]
    let ``FluxExecutionEngine can be instantiated`` () =
        // Arrange & Act
        let engine = FluxExecutionEngine()
        
        // Assert
        engine |> should not' (equal null)
    
    [<Fact>]
    let ``FluxExecutionEngine can execute simple script`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestMetaBlock [("title", StringValue "Test")]
                TestHelpers.createTestLanguageBlock "FSHARP" "let x = 42"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FluxExecutionEngine handles empty script`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let emptyScript = TestHelpers.createTestFluxScript []
            
            // Act
            let! result = engine.ExecuteScript(emptyScript)
            
            // Assert
            result |> should not' (equal null)
            result.BlocksExecuted |> should equal 0
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes META blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestMetaBlock [
                    ("title", StringValue "Test Script")
                    ("version", StringValue "1.0.0")
                    ("description", StringValue "Test description")
                ]
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes FSHARP blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestLanguageBlock "FSHARP" "let x = 42\nprintfn \"Value: %d\" x"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes PYTHON blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestLanguageBlock "PYTHON" "x = 42\nprint(f'Value: {x}')"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes JAVASCRIPT blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestLanguageBlock "JAVASCRIPT" "const x = 42;\nconsole.log(`Value: ${x}`);"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes AGENT blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let langBlock = {
                Language = "FSHARP"
                Content = "printfn \"Agent executing\""
                LineNumber = 1
                Variables = Map.empty
            }
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestAgentBlock "TestAgent" [Role "Test Role"] [langBlock]
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
            TestHelpers.assertAgentOutput "TestAgent" result
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes DIAGNOSTIC blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestDiagnosticBlock [
                    Test "Basic functionality test"
                    Validate "System is operational"
                    Benchmark "Performance test"
                ]
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
            result.DiagnosticResults.Count |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes REFLECTION blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestReflectionBlock [
                    Analyze "Code quality analysis"
                    Plan "Optimization strategy"
                ]
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
            result.ReflectionInsights.Length |> should be (greaterThan 0)
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes REASONING blocks`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestReasoningBlock "This is a reasoning block that analyzes the problem."
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 1
        }
    
    [<Fact>]
    let ``FluxExecutionEngine executes multiple blocks in sequence`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestMetaBlock [("title", StringValue "Multi-block Test")]
                TestHelpers.createTestLanguageBlock "FSHARP" "let x = 42"
                TestHelpers.createTestLanguageBlock "PYTHON" "y = 24"
                TestHelpers.createTestDiagnosticBlock [Test "Multi-block test"]
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.BlocksExecuted |> should equal 4
        }
    
    [<Fact>]
    let ``FluxExecutionEngine measures execution time`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestLanguageBlock "FSHARP" "System.Threading.Thread.Sleep(100)"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.ExecutionTime |> should be (greaterThan TimeSpan.Zero)
        }
    
    [<Fact>]
    let ``FluxExecutionEngine handles execution errors gracefully`` () =
        task {
            // Arrange
            let engine = FluxExecutionEngine()
            let script = TestHelpers.createTestFluxScript [
                TestHelpers.createTestLanguageBlock "INVALID_LANGUAGE" "This should cause an error"
            ]
            
            // Act
            let! result = engine.ExecuteScript(script)
            
            // Assert
            // The runtime should handle this gracefully
            result |> should not' (equal null)
            result.BlocksExecuted |> should be (greaterThanOrEqualTo 0)
        }

printfn "🧪 FLUX Runtime Unit Tests Loaded"
printfn "================================="
printfn "✅ Engine instantiation tests"
printfn "✅ Script execution tests"
printfn "✅ Block type execution tests"
printfn "✅ Multi-block execution tests"
printfn "✅ Error handling tests"
printfn "✅ Performance measurement tests"
