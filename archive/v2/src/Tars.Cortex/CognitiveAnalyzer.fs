namespace Tars.Cortex

open System
open Tars.Core

/// <summary>
/// Represents the cognitive mode of the system.
/// </summary>
type CognitiveMode =
    /// <summary>High entropy, seeking new information.</summary>
    | Exploratory
    /// <summary>Low entropy, optimizing and executing.</summary>
    | Convergent
    /// <summary>System under stress or error state.</summary>
    | Critical

/// <summary>
/// Snapshot of the system's cognitive state.
/// </summary>
type CognitiveState =
    {
        Mode: CognitiveMode
        /// <summary>System coherence/stability metric (0.0 - 1.0).</summary>
        Eigenvalue: float
        /// <summary>Information density/disorder (0.0 - 1.0).</summary>
        Entropy: float
        /// <summary>Current attention span in tokens or steps.</summary>
        AttentionSpan: int
        /// <summary>Reasoning complexity (Graph of Thoughts metric).</summary>
        BranchingFactor: float
    }

/// <summary>
/// Analyzes the system's cognitive state based on agent activities and context.
/// </summary>
type CognitiveAnalyzer(kernel: IAgentRegistry, ?thoughtGraph: ThoughtGraph) =

    /// <summary>
    /// Analyzes the current state of the kernel and returns a cognitive assessment.
    /// </summary>
    member this.Analyze() =
        async {
            let! agents = kernel.GetAllAgents()

            let activeAgents =
                agents
                |> List.filter (fun a ->
                    match a.State with
                    | AgentState.Idle -> false
                    | _ -> true)

            let errorAgents =
                agents
                |> List.filter (fun a ->
                    match a.State with
                    | AgentState.Error _ -> true
                    | _ -> false)

            let totalAgents = float (max 1 agents.Length)
            let activeCount = float activeAgents.Length
            let errorCount = float errorAgents.Length

            // Eigenvalue: Stability.
            let activityPenalty = (activeCount / totalAgents) * 0.2
            let errorPenalty = (errorCount / totalAgents) * 0.8
            let eigenvalue = max 0.0 (1.0 - activityPenalty - errorPenalty)

            // Entropy: Information density / Disorder.
            let entropy = min 1.0 (activeCount / totalAgents)

            // GoT Metric: Branching Factor
            let branchingFactor =
                match thoughtGraph with
                | Some g when g.Nodes.Count > 0 ->
                    float g.Edges.Length / float g.Nodes.Count
                | _ -> 1.0

            let mode =
                if errorCount > 0.0 then Critical
                elif entropy > 0.6 then Exploratory
                else Convergent

            return
                { Mode = mode
                  Eigenvalue = eigenvalue
                  Entropy = entropy
                  AttentionSpan = 8192
                  BranchingFactor = branchingFactor }
        }

/// <summary>
/// Monitors the entropy of the context window to trigger compression.
/// </summary>
type EntropyMonitor() =

    /// <summary>
    /// Measures the entropy of a given text context.
    /// Returns a value between 0.0 (repetitive) and 1.0 (diverse).
    /// </summary>
    member this.Measure(context: string) =
        if String.IsNullOrWhiteSpace(context) then
            0.0
        else
            let tokens =
                context.Split([| ' '; '\n'; '\r'; '\t' |], StringSplitOptions.RemoveEmptyEntries)

            if tokens.Length = 0 then
                0.0
            else
                let unique = tokens |> Array.distinct |> Array.length
                float unique / float tokens.Length
