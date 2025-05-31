namespace TarsEngine.FSharp.Cli.Commands

open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascripts.Services
open TarsEngine.FSharp.Metascripts.Discovery

type RealExecuteCommand(logger: ILogger<RealExecuteCommand>, metascriptService: MetascriptService) =
    interface ICommand with
        member _.Name = "exec"
        member _.Description = "Execute TARS metascripts with REAL execution engine"
        member _.Usage = "tars exec <metascript-name>"
        member _.Examples = ["tars exec hello_world"; "tars exec autonomous_improvement"]
        member _.ValidateOptions(_) = true
        
        member _.ExecuteAsync(options) =
            task {
                try
                    match options.Arguments with
                    | metascriptName :: _ ->
                        printfn "🚀 TARS REAL METASCRIPT EXECUTION"
                        printfn "===================================="
                        printfn "Target: %s" metascriptName
                        printfn ""
                        
                        // Discover metascripts
                        printfn "🔍 Discovering metascripts..."
                        let discovery = MetascriptDiscovery(logger)
                        let! discoveryResult = discovery.DiscoverMetascriptsAsync(".", true)
                        
                        match discoveryResult with
                        | Ok metascripts ->
                            let target = metascripts |> List.tryFind (fun m -> m.Name = metascriptName)
                            
                            match target with
                            | Some metascript ->
                                printfn "✅ Found: %s" metascript.FilePath
                                printfn "📄 Description: %s" (metascript.Description |> Option.defaultValue "No description")
                                printfn ""
                                
                                // Execute with REAL engine
                                printfn "⚡ EXECUTING WITH REAL ENGINE..."
                                let! result = metascriptService.ExecuteMetascriptAsync(metascript.Name)
                                
                                match result with
                                | Ok execResult ->
                                    printfn "✅ REAL EXECUTION COMPLETED!"
                                    printfn "Status: %s" execResult.Status
                                    printfn "Time: %dms" (int execResult.ExecutionTime.TotalMilliseconds)
                                    printfn ""
                                    printfn "📋 OUTPUT:"
                                    printfn "%s" execResult.Output
                                    return CommandResult.success("Real execution completed")
                                | Error error ->
                                    printfn "❌ Execution failed: %s" error.Message
                                    return CommandResult.failure("Execution failed")
                            | None ->
                                printfn "❌ Metascript not found: %s" metascriptName
                                return CommandResult.failure("Metascript not found")
                        | Error error ->
                            printfn "❌ Discovery failed: %s" error
                            return CommandResult.failure("Discovery failed")
                    | [] ->
                        printfn "Usage: tars exec <metascript-name>"
                        return CommandResult.failure("No metascript specified")
                with
                | ex ->
                    logger.LogError(ex, "Real execution error")
                    return CommandResult.failure(ex.Message)
            }
