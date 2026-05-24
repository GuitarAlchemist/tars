namespace Tars.Interface.Cli

open System
open System.Threading
open System.Threading.Tasks
open Tars.Core
open Tars.Knowledge
open Serilog

module ReflectionService =

    let startScheduledReflection (logger: ILogger) (ledger: KnowledgeLedger) (interval: TimeSpan) (cancellationToken: CancellationToken) =
        let agent = ReflectionAgent(ledger)
        
        task {
            logger.Information("Scheduled Reflection Service started. Interval: {Interval}", interval)
            
            while not cancellationToken.IsCancellationRequested do
                try
                    logger.Information("Starting scheduled knowledge reflection...")
                    do! agent.ReflectAsync() |> Async.StartAsTask
                    
                    // Auto-cleanup low confidence beliefs (e.g., < 20% confidence)
                    let! retracted = agent.CleanupAsync(0.2) |> Async.StartAsTask
                    if retracted > 0 then
                        logger.Information("Reflection cleanup: Retracted {Count} low-confidence beliefs.", retracted)
                        
                with ex ->
                    logger.Error(ex, "Error during scheduled reflection.")

                try
                    do! Task.Delay(interval, cancellationToken)
                with :? TaskCanceledException ->
                    () // Clean exit
        }
