namespace TarsEngine.FSharp.Cli.Commands

/// <summary>
/// Simple command line parser using Commands.CommandOptions.
/// </summary>
type CommandLineParser() =
    
    /// <summary>
    /// Parses command line arguments.
    /// </summary>
    member self.Parse(args: string[]) : string * CommandOptions =
        let argsList = Array.toList args
        
        match argsList with
        | [] -> 
            ("help", CommandOptions.createDefault())
        | commandName :: rest ->
            let arguments, options = 
                rest
                |> List.fold (fun (args, opts) arg ->
                    if arg.StartsWith("--") then
                        let parts = arg.Substring(2).Split([|'='|], 2)
                        if parts.Length = 2 then
                            (args, Map.add parts.[0] parts.[1] opts)
                        else
                            (args, Map.add parts.[0] "true" opts)
                    elif arg.StartsWith("-") then
                        let key = arg.Substring(1)
                        (args, Map.add key "true" opts)
                    else
                        (arg :: args, opts)
                ) ([], Map.empty)
            
            let finalArguments = List.rev arguments
            let isHelp = options.ContainsKey("help") || options.ContainsKey("h")
            
            let commandOptions = 
                CommandOptions.createDefault()
                |> CommandOptions.withArguments finalArguments
                |> CommandOptions.withOptions options
                |> CommandOptions.withHelp isHelp
            
            (commandName, commandOptions)
