namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Integration tests for Metaprogramming capabilities
module MetaprogrammingTests =
    
    [<Fact>]
    let ``Metaprogramming capabilities are available`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Metaprogramming Test"
    version: "1.0.0"
}

FSHARP {
    // Test quotations and metaprogramming
    let expr = <@ fun x -> x + 1 @>
    printfn "Expression: %A" expr
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
        }

printfn "🧬 Metaprogramming Integration Tests Loaded"
