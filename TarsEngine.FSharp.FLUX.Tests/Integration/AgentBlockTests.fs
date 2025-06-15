namespace TarsEngine.FSharp.FLUX.Tests.Integration

open System
open System.Threading.Tasks
open Xunit
open FsUnit.Xunit
open TarsEngine.FSharp.FLUX.FluxEngine
open TarsEngine.FSharp.FLUX.Tests.TestHelpers

/// Integration tests for Agent Block execution
module AgentBlockTests =
    
    [<Fact>]
    let ``Agent block executes successfully`` () =
        task {
            // Arrange
            let script = """
META {
    title: "Agent Test"
    version: "1.0.0"
}

AGENT DataProcessor {
    role: "Data Processing Agent"
    capabilities: ["data_processing", "validation"]
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
"""
            let engine = FluxEngine()
            
            // Act
            let! result = engine.ExecuteString(script)
            
            // Assert
            TestHelpers.assertExecutionSuccess result
            TestHelpers.assertAgentOutput "DataProcessor" result
        }

printfn "🤖 Agent Block Integration Tests Loaded"
