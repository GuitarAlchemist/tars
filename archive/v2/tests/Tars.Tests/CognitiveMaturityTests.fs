namespace Tars.Tests

open System
open Xunit
open Tars.Core
open Tars.Cortex

module CognitiveMaturityTests =

    [<Fact>]
    let ``CognitiveBenchmarker: Evaluates Graph of Thoughts correctly`` () =
        let node1 = { Id = Guid.NewGuid(); Content = "Root"; NodeType = "Hypothesis"; Confidence = 0.9; Metadata = Map.empty; Timestamp = DateTime.UtcNow }
        let node2 = { Id = Guid.NewGuid(); Content = "Branch 1"; NodeType = "Observation"; Confidence = 0.8; Metadata = Map.empty; Timestamp = DateTime.UtcNow }
        let node3 = { Id = Guid.NewGuid(); Content = "Synthesis"; NodeType = "Synthesis"; Confidence = 0.95; Metadata = Map.empty; Timestamp = DateTime.UtcNow }
        
        let graph = {
            Nodes = Map.ofList [ (node1.Id, node1); (node2.Id, node2); (node3.Id, node3) ]
            Edges = [ 
                { SourceId = node1.Id; TargetId = node2.Id; Relation = "Supports"; Weight = 1.0 }
                { SourceId = node2.Id; TargetId = node3.Id; Relation = "Refines"; Weight = 1.0 }
            ]
            ContextId = Guid.NewGuid()
        }
        
        let benchmarker = CognitiveBenchmarker()
        let metrics = benchmarker.EvaluateReasoning(graph)
        
        Assert.Equal(2.0 / 3.0, metrics.BranchingFactor, 2)
        Assert.Equal(1.0 / 3.0, metrics.ConvergenceRate, 2)
        Assert.Equal(0.95, metrics.SynthesisQuality, 2)

    [<Fact>]
    let ``CognitiveAnalyzer: Reports GoT metrics`` () =
        let kernel = { new IAgentRegistry with 
            member _.GetAllAgents() = async { return [] }
            member _.GetAgent _ = async { return None }
            member _.FindAgents _ = async { return [] }
        }
        
        let analyzer = CognitiveAnalyzer(kernel)
        let state = analyzer.Analyze() |> Async.RunSynchronously
        
        Assert.True(state.BranchingFactor >= 1.0)
