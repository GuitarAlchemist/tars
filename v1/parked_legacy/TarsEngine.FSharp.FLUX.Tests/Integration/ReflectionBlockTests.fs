namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Integration tests for Reflection Block execution
module ReflectionBlockTests =
    
    [<Fact>]
    let ``Reflection block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Reflection Test"
    version: "1.0.0"
}

REFLECT {
    analyze: "Code quality and complexity metrics"
    plan: "Optimization strategy"
    improve: ("performance", "caching")
}
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            result.ReflectionInsights.Length |> should be (greaterThan 0)
        }

printfn "🪞 Reflection Block Integration Tests Loaded"
