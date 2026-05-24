namespace Tars.Interface.Cli

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Tars.Core
open Tars.Cortex

/// <summary>
/// Background service that periodically runs agent fitness decay and pruning.
/// </summary>
type AgentLifecycleBackgroundService(logger: ILogger<AgentLifecycleBackgroundService>, registry: IAgentRegistry) =
    inherit BackgroundService()

    // Interval: Every 30 minutes
    let interval = TimeSpan.FromMinutes(30.0)
    
    // Decay: 1% reduction per cycle
    let decayRate = 0.99
    
    // Pruning: Remove agents below 0.2 fitness (20%)
    let pruneThreshold = 0.2

    override this.ExecuteAsync(cancellationToken: CancellationToken) =
        task {
            logger.LogInformation("🚀 Agent Lifecycle Service started. Interval: {Interval}", interval)
            
            // Initial delay to let the application startup settle
            do! Task.Delay(TimeSpan.FromMinutes(1.0), cancellationToken)

            while not cancellationToken.IsCancellationRequested do
                try
                    logger.LogInformation("🌱 Starting agent lifecycle cycle (Decay: {Decay}, Prune: {Prune})...", decayRate, pruneThreshold)
                    
                    let! count = AgentLifecycleAgent.runLifecycleCycle registry decayRate pruneThreshold
                    
                    if count > 0 then
                        logger.LogInformation("✂️ Agent Lifecycle: Pruned {Count} agents.", count)
                    else
                        logger.LogInformation("✅ Agent Lifecycle complete. No agents pruned.")
                
                with ex ->
                    logger.LogError(ex, "❌ Error during agent lifecycle cycle.")
                
                try
                    // Wait for next cycle
                    do! Task.Delay(interval, cancellationToken)
                with :? TaskCanceledException ->
                    logger.LogInformation("🛑 Agent Lifecycle Service stopping...")
        }
