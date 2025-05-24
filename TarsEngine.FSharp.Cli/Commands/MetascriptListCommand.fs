namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Metascripts.Services

/// <summary>
/// Command for listing metascripts using the fixed metascript engine.
/// </summary>
type MetascriptListCommand(metascriptService: IMetascriptService) =
    interface ICommand with
        member _.Name = "metascript-list"
        
        member _.Description = "List and discover TARS metascripts"
        
        member _.Usage = "tars metascript-list [options]"
        
        member _.Examples = [
            "tars metascript-list"
            "tars metascript-list --discover"
            "tars metascript-list --stats"
        ]
        
        member _.ValidateOptions(options) = true
        
        member _.ExecuteAsync(options) =
            Task.Run(fun () ->
                try
                    let discover = options.Options.ContainsKey("discover")
                    let showStats = options.Options.ContainsKey("stats")
                    
                    if discover then
                        Console.WriteLine("Discovering metascripts...")
                        let discoveryResult = metascriptService.DiscoverMetascriptsAsync(".").Result
                        match discoveryResult with
                        | Ok metascripts ->
                            Console.WriteLine(sprintf "Discovered %d metascripts:" metascripts.Length)
                            metascripts |> List.iter (fun m ->
                                Console.WriteLine(sprintf "  %s - %s" m.Source.Name m.Source.Metadata.Description)
                            )
                            CommandResult.success("Metascript discovery completed")
                        | Error error ->
                            Console.WriteLine(sprintf "Discovery failed: %s" error)
                            CommandResult.failure("Metascript discovery failed")
                    elif showStats then
                        Console.WriteLine("Getting metascript statistics...")
                        let statsResult = metascriptService.GetStatisticsAsync().Result
                        match statsResult with
                        | Ok stats ->
                            Console.WriteLine("Metascript Statistics:")
                            Console.WriteLine(sprintf "  Total Metascripts: %d" stats.TotalMetascripts)
                            Console.WriteLine(sprintf "  Executed Today: %d" stats.ExecutedToday)
                            Console.WriteLine(sprintf "  Success Rate: %.1f%%" (stats.SuccessRate * 100.0))
                            Console.WriteLine(sprintf "  Average Execution Time: %dms" (int stats.AverageExecutionTime.TotalMilliseconds))
                            Console.WriteLine(sprintf "  Most Used Category: %A" stats.MostUsedCategory)
                            CommandResult.success("Metascript statistics displayed")
                        | Error error ->
                            Console.WriteLine(sprintf "Failed to get statistics: %s" error)
                            CommandResult.failure("Failed to get metascript statistics")
                    else
                        Console.WriteLine("Listing registered metascripts...")
                        let listResult = metascriptService.ListMetascriptsAsync().Result
                        match listResult with
                        | Ok metascripts ->
                            if metascripts.IsEmpty then
                                Console.WriteLine("No metascripts registered.")
                                Console.WriteLine("Use 'tars metascript-list --discover' to find metascripts.")
                            else
                                Console.WriteLine(sprintf "Registered metascripts (%d):" metascripts.Length)
                                metascripts |> List.iter (fun m ->
                                    let usageInfo = 
                                        match m.LastUsed with
                                        | Some lastUsed -> sprintf "Last used: %s" (lastUsed.ToString("yyyy-MM-dd HH:mm"))
                                        | None -> "Never used"
                                    Console.WriteLine(sprintf "  %s - %s (%s)" 
                                        m.Source.Name m.Source.Metadata.Description usageInfo)
                                )
                            CommandResult.success("Metascripts listed successfully")
                        | Error error ->
                            Console.WriteLine(sprintf "Failed to list metascripts: %s" error)
                            CommandResult.failure("Failed to list metascripts")
                with
                | ex ->
                    CommandResult.failure(sprintf "Metascript list command failed: %s" ex.Message)
            )
