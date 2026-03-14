namespace Tars.Interface.Cli

open System
open System.Threading
open System.Threading.Tasks
open Microsoft.Extensions.Hosting
open Microsoft.Extensions.Logging
open Tars.Knowledge
open Tars.Llm

/// Background service that periodically runs the ReflectionAgent to maintain the knowledge ledger.
type ReflectionBackgroundService(logger: ILogger<ReflectionBackgroundService>, ledger: KnowledgeLedger, registry: Tars.Core.IAgentRegistry, llm: ILlmService) =
    inherit BackgroundService()

    // Interval for reflection - in a real scenario this might be configurable (e.g., 15 mins or 1 hour)
    let reflectionInterval = TimeSpan.FromMinutes(60.0)

    override this.ExecuteAsync(cancellationToken: CancellationToken) =
        task {
            logger.LogInformation("🚀 Scheduled Reflection Service started. Interval: {Interval}", reflectionInterval)
            
            // Initial delay to let the application startup settle
            do! Task.Delay(TimeSpan.FromSeconds(30.0), cancellationToken)

            while not cancellationToken.IsCancellationRequested do
                try
                    logger.LogInformation("🧠 Starting scheduled knowledge reflection cycle...")
                    
                    // Create agent instance
                    let agent = ReflectionAgent(ledger, Some registry, Some llm)
                    
                    // 1. Run Reflection (Contradiction Detection)
                    // We await the async F# computation
                    do! agent.ReflectAsync() |> Async.StartAsTask
                    
                    // 1.5 Run Architectural Reflection (LLM-based)
                    do! agent.ReflectOnArchitectureAsync() |> Async.StartAsTask
                    
                    // 2. Auto-cleanup low confidence beliefs (e.g., < 20% confidence)
                    // Only remove items if they are severely degraded
                    let! retracted = agent.CleanupAsync(0.2) |> Async.StartAsTask
                    
                    if retracted > 0 then
                        logger.LogInformation("🧹 Reflection cleanup: Retracted {Count} low-confidence beliefs.", retracted)
                    else
                        logger.LogInformation("✅ Reflection cycle complete. No low-confidence beliefs found.")

                with ex ->
                    // Log but don't crash the service
                    logger.LogError(ex, "❌ Error during scheduled reflection cycle.")

                try
                    // Wait for next cycle
                    do! Task.Delay(reflectionInterval, cancellationToken)
                with :? TaskCanceledException ->
                    // Graceful shutdown
                    logger.LogInformation("🛑 Reflection Service stopping...")
        }
