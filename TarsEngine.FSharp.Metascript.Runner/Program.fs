open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Metascript.Services
// open TarsEngine.FSharp.Core.Metascript.DependencyInjection

/// <summary>
/// Metascript runner program.
/// </summary>
module Program =
    /// <summary>
    /// Configures the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let configureServices (services: IServiceCollection) =
        services
            .AddLogging(fun logging ->
                logging.AddConsole() |> ignore
                logging.SetMinimumLevel(LogLevel.Information) |> ignore
            )
            .AddSingleton<IMetascriptService, MetascriptService>()
    
    /// <summary>
    /// Runs a metascript from a file.
    /// </summary>
    /// <param name="serviceProvider">The service provider.</param>
    /// <param name="filePath">The file path.</param>
    /// <returns>The execution result.</returns>
    let runMetascriptFromFile (serviceProvider: IServiceProvider) (filePath: string) =
        task {
            // Get the metascript service
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            
            // Execute the metascript
            return! metascriptService.ExecuteMetascriptAsync(filePath)
        }
    
    /// <summary>
    /// Main entry point.
    /// </summary>
    /// <param name="args">Command-line arguments.</param>
    /// <returns>The exit code.</returns>
    [<EntryPoint>]
    let main args =
        try
            // Check if a file path was provided
            if args.Length = 0 then
                Console.WriteLine("Please provide a metascript file path.")
                1
            else
                // Get the file path
                let filePath = args.[0]
                
                // Check if the file exists
                if not (File.Exists(filePath)) then
                    Console.WriteLine($"File not found: {filePath}")
                    1
                else
                    // Configure services
                    let services = ServiceCollection()
                    let serviceProvider = configureServices(services).BuildServiceProvider()
                    
                    // Run the metascript
                    let result = runMetascriptFromFile serviceProvider filePath
                    
                    // Wait for the result
                    result.Wait()
                    
                    // Print the result
                    Console.WriteLine("Metascript execution completed.")
                    Console.WriteLine("Status: SUCCESS")
                    Console.WriteLine("Output:")
                    Console.WriteLine(result.Result.ToString())
                    0
        with
        | ex ->
            Console.WriteLine("Error:")
            Console.WriteLine(ex.ToString())
            1
