namespace Tars.Cortex

open System
open Tars.Core

/// Metrics for evaluating Graph of Thoughts (GoT) reasoning.
type GoTMetrics =
    { Density: float
      BranchingFactor: float
      ConvergenceRate: float
      SynthesisQuality: float }

/// Metrics for evaluating Web of Things (WoT) tool grounding.
type WoTMetrics =
    { DescriptionFidelity: float
      InteractionSuccessRate: float
      GroundingAccuracy: float }

/// Benchmarks cognitive capacities of TARS agents.
type CognitiveBenchmarker() =

    /// Evaluates a ThoughtGraph and returns GoT metrics.
    member _.EvaluateReasoning(graph: ThoughtGraph) =
        let nodeCount = float (max 1 graph.Nodes.Count)
        let edgeCount = float graph.Edges.Length
        
        { Density = edgeCount / (nodeCount * (nodeCount - 1.0) |> max 1.0)
          BranchingFactor = edgeCount / nodeCount
          ConvergenceRate = 
            graph.Nodes.Values 
            |> Seq.filter (fun n -> n.NodeType = "Synthesis") 
            |> Seq.length |> float |> (fun s -> s / nodeCount)
          SynthesisQuality = 
            graph.Nodes.Values 
            |> Seq.filter (fun n -> n.NodeType = "Synthesis")
            |> Seq.averageBy (fun n -> n.Confidence) }

    /// Evaluates tool usage against WoT descriptions.
    member _.EvaluateGrounding(tools: Tool list) =
        let toolsWithTd = tools |> List.filter (fun t -> t.ThingDescription.IsSome)
        let fidelity = float toolsWithTd.Length / float (max 1 tools.Length)
        
        { DescriptionFidelity = fidelity
          InteractionSuccessRate = 0.9 // Placeholder
          GroundingAccuracy = fidelity * 0.9 }

    /// Generates a comprehensive cognitive report.
    member this.GenerateReport(graph: ThoughtGraph, tools: Tool list) =
        let got = this.EvaluateReasoning(graph)
        let wot = this.EvaluateGrounding(tools)
        
        sprintf "COGNITIVE REPORT\n----------------\nGoT Branching: %.2f\nGoT Convergence: %.2f\nWoT Fidelity: %.2f\nOverall Grounding: %.2f"
            got.BranchingFactor got.ConvergenceRate wot.DescriptionFidelity wot.GroundingAccuracy
