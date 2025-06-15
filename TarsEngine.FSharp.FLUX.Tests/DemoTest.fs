namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine

/// Demo test to showcase FLUX capabilities
module DemoTest =
    
    [<Fact>]
    let ``FLUX can execute comprehensive demo script`` () =
        async {
            // Arrange
            let engine = FluxEngine()
            let demoPath = Path.Combine(__SOURCE_DIRECTORY__, "demo.flux")
            
            // Act
            let! result = engine.ExecuteFile(demoPath) |> Async.AwaitTask
            
            // Assert
            result |> should not' (equal null)
            result.Success |> should equal true
            result.BlocksExecuted |> should be (greaterThan 0)
            result.ExecutionTime |> should be (lessThan (TimeSpan.FromSeconds(30.0)))

            // Note: With simplified parser, we get basic execution
            // Full parser implementation will enable all advanced features
            
            printfn "🎉 FLUX Demo Execution Results:"
            printfn "  ✅ Blocks executed: %d" result.BlocksExecuted
            printfn "  ✅ Execution time: %A" result.ExecutionTime
            printfn "  ✅ Agent outputs: %d" result.AgentOutputs.Count
            printfn "  ✅ Diagnostic results: %d" result.DiagnosticResults.Count
            printfn "  ✅ Reflection insights: %d" result.ReflectionInsights.Length
        }
