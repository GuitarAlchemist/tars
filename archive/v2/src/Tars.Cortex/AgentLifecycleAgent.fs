namespace Tars.Cortex

open System
open System.Threading.Tasks
open Tars.Core

/// <summary>
/// Agent responsible for the lifecycle of agents in the registry.
/// Manages fitness decay and pruning of low-fitness agents.
/// </summary>
module AgentLifecycleAgent =

    /// <summary>
    /// Runs a fitness decay cycle on all agents.
    /// Applies decay rate and prunes agents below threshold.
    /// </summary>
    /// <param name="registry">The agent registry.</param>
    /// <param name="decayRate">Multiplicative decay factor (e.g., 0.99).</param>
    /// <param name="pruneThreshold">Fitness score below which an agent is removed.</param>
    /// <returns>Number of agents pruned.</returns>
    let runLifecycleCycle (registry: IAgentRegistry) (decayRate: float) (pruneThreshold: float) =
        task {
            let! agents = registry.GetAllAgents() |> Async.StartAsTask

            let mutable prunedCount = 0
            
            for agent in agents do
                // Apply decay
                let newFitness = agent.Fitness * decayRate
                
                if newFitness < pruneThreshold then
                    // Prune
                    do! registry.RemoveAgent(agent.Id) |> Async.StartAsTask
                    prunedCount <- prunedCount + 1
                else
                    // Update if changed (always changed if decayRate != 1.0)
                    let updatedAgent = { agent with Fitness = newFitness }
                    do! registry.UpdateAgent(updatedAgent) |> Async.StartAsTask
                        
            return prunedCount
        }
