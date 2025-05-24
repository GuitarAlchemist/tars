namespace TarsEngine.FSharp.Cli.Commands

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Services

/// <summary>
/// Command for executing metascripts - merged functionality.
/// </summary>
type MetascriptCommand(metascriptService: MetascriptService) =
    interface ICommand with
        member _.Name = "metascript"
        
        member _.Description = "Execute TARS metascripts"
        
        member _.Usage = "tars metascript [name] [options]"
        
        member _.Examples = [
            "tars metascript code_analysis"
            "tars metascript --list"
            "tars metascript improve_code --verbose"
        ]
        
        member _.ValidateOptions(options) = true
        
        member _.ExecuteAsync(options) =
            Task.Run(fun () ->
                try
                    let listRequested = options.Options.ContainsKey("list")
                    
                    if listRequested then
                        Console.WriteLine("Listing available metascripts...")
                        let metascriptsResult = metascriptService.ListMetascriptsAsync().Result
                        match metascriptsResult with
                        | Ok metascripts ->
                            if metascripts.IsEmpty then
                                Console.WriteLine("No metascripts found. Use discovery to find metascripts.")
                            else
                                Console.WriteLine("Available metascripts:")
                                metascripts |> List.iter (fun m ->
                                    Console.WriteLine(sprintf "  %s - %s" m.Name m.Metadata.Description)
                                )
                            CommandResult.success("Metascripts listed successfully")
                        | Error error ->
                            Console.WriteLine(sprintf "Error listing metascripts: %s" error.Message)
                            CommandResult.failure("Failed to list metascripts")
                    else
                        match options.Arguments with
                        | metascriptName :: _ ->
                            Console.WriteLine(sprintf "Executing metascript: %s" metascriptName)
                            let executionResult = metascriptService.ExecuteMetascriptAsync(metascriptName).Result
                            match executionResult with
                            | Ok result ->
                                Console.WriteLine("Metascript execution completed")
                                Console.WriteLine(sprintf "Status: %s" result.Status)
                                Console.WriteLine(sprintf "Output:\n%s" result.Output)
                                Console.WriteLine(sprintf "Execution time: %dms" (int result.ExecutionTime.TotalMilliseconds))
                                CommandResult.success("Metascript executed successfully")
                            | Error error ->
                                Console.WriteLine(sprintf "Execution error: %s" error.Message)
                                CommandResult.failure("Metascript execution failed")
                        | [] ->
                            Console.WriteLine("Please specify a metascript name or use --list to see available metascripts")
                            CommandResult.failure("No metascript name provided")
                with
                | ex ->
                    CommandResult.failure(sprintf "Metascript command failed: %s" ex.Message)
            )
