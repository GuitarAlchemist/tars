namespace Tars.Cortex

open Tars.Core

/// Metrics for evaluating Graph of Thoughts (GoT) reasoning.
type GoTMetrics =
    { Density: float
      BranchingFactor: float
      ConvergenceRate: float
      SynthesisQuality: float }

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

    /// Generates a comprehensive cognitive report.
    member this.GenerateReport(graph: ThoughtGraph) =
        let got = this.EvaluateReasoning(graph)

        $"COGNITIVE REPORT\n----------------\nGoT Branching: %.2f{got.BranchingFactor}\nGoT Convergence: %.2f{got.ConvergenceRate}"
