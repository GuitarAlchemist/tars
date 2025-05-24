namespace TarsEngine.FSharp.Cli.Core

open System
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Cli.Commands
open TarsEngine.FSharp.Cli.Services
open TarsEngine.FSharp.Metascripts.Core
open TarsEngine.FSharp.Metascripts.Discovery
open TarsEngine.FSharp.Metascripts.Services

/// <summary>
/// CLI application with separate metascript engine.
/// </summary>
type CliApplication() =
    let serviceProvider = 
        let services = ServiceCollection()
        
        // Add logging
        services.AddLogging(fun logging -> 
            logging.AddConsole() |> ignore
        ) |> ignore
        
        // Add metascript engine services
        services.AddSingleton<MetascriptRegistry>() |> ignore
        services.AddSingleton<MetascriptManager>() |> ignore
        services.AddSingleton<MetascriptDiscovery>() |> ignore
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore
        
        // Add CLI services
        services.AddSingleton<IntelligenceService>() |> ignore
        services.AddSingleton<MLService>() |> ignore
        
        services.BuildServiceProvider()
    
    let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
    let intelligenceService = serviceProvider.GetRequiredService<IntelligenceService>()
    let mlService = serviceProvider.GetRequiredService<MLService>()
    let commandRegistry = CommandRegistry(metascriptService, intelligenceService, mlService)
    let commandLineParser = CommandLineParser()
    
    do
        // Register all commands
        commandRegistry.RegisterDefaultCommands()
    
    /// <summary>
    /// Runs the CLI application with separate metascript engine.
    /// </summary>
    member _.RunAsync(args: string[]) =
        Task.Run(fun () ->
            try
                // Parse the command line arguments
                let commandName, options = commandLineParser.Parse(args)
                
                // Get the command
                match commandRegistry.GetCommand(commandName) with
                | Some command ->
                    // Execute the command
                    let result = command.ExecuteAsync(options).Result
                    
                    // Write the result to the console
                    if not (String.IsNullOrEmpty(result.Message)) then
                        Console.WriteLine(result.Message)
                    
                    // Return the exit code
                    result.ExitCode
                | None ->
                    // Command not found
                    Console.WriteLine(sprintf "Command not found: %s" commandName)
                    Console.WriteLine("Use 'tars help' to see available commands.")
                    1
            with
            | ex ->
                // Write the error to the console
                Console.WriteLine(sprintf "Error: %s" ex.Message)
                1
        )
