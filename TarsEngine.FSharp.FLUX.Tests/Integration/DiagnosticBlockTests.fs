namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Integration tests for Diagnostic Block execution
module DiagnosticBlockTests =
    
    [<Fact>]
    let ``Diagnostic block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Diagnostic Test"
    version: "1.0.0"
}

DIAGNOSTIC {
    test: "System functionality test"
    validate: "Performance metrics within range"
    benchmark: "Response time under 100ms"
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.DiagnosticResults.Count |> should be (greaterThan 0)
        }

printfn "🔍 Diagnostic Block Integration Tests Loaded"
