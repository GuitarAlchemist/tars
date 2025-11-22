namespace TarsEngine.FSharp.FLUX.Tests

open System
open System.IO
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine

/// Test for the cool AI code generator demo
module CoolDemoTest =
    
    [<Fact>]
    let ``FLUX can execute AI-powered code generator`` () =
        async {
            // Arrange
            let engine = FluxEngine()
            let aiCodeGenPath = Path.Combine(__SOURCE_DIRECTORY__, "ai-code-generator.flux")
            
            printfn "🚀 Executing AI-Powered Code Generator Demo"
            printfn "============================================"
            
            // Act
            let! result = engine.ExecuteFile(aiCodeGenPath) |> Async.AwaitTask
            
            // Assert
            result |> should not' (equal null)
            result.Success |> should equal true
            result.BlocksExecuted |> should be (greaterThan 0)
            result.ExecutionTime |> should be (lessThan (TimeSpan.FromSeconds(60.0)))
            
            printfn ""
            printfn "🎉 AI Code Generator Demo Results:"
            printfn "=================================="
            printfn "  ✅ Success: %b" result.Success
            printfn "  ✅ Blocks executed: %d" result.BlocksExecuted
            printfn "  ✅ Execution time: %A" result.ExecutionTime
            printfn "  ✅ Agent outputs: %d" result.AgentOutputs.Count
            printfn "  ✅ Diagnostic results: %d" result.DiagnosticResults.Count
            printfn "  ✅ Reflection insights: %d" result.ReflectionInsights.Length
            
            if result.AgentOutputs.Count > 0 then
                printfn ""
                printfn "🤖 AI Agent Outputs:"
                result.AgentOutputs |> Map.iter (fun agentName output ->
                    printfn "  - %s: Executed successfully" agentName)
            
            printfn ""
            printfn "🔥 FLUX AI Code Generator Demo Complete!"
            printfn "This demonstrates the revolutionary potential of"
            printfn "AI-powered software development with FLUX!"
        }
