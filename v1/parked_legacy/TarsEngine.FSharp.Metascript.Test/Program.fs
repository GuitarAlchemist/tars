open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Core.Metascript
open TarsEngine.FSharp.Core.Metascript.Services
open TarsEngine.FSharp.Core.Metascript.BlockHandlers
open TarsEngine.FSharp.Core.Metascript.DependencyInjection

/// <summary>
/// Simple console application to test the Metascript module.
/// </summary>
module Program =
    /// <summary>
    /// Configures the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    let configureServices (services: IServiceCollection) =
        // Add logging
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore
        ) |> ignore
        
        // Add metascript services
        ServiceCollectionExtensions.addTarsEngineFSharpMetascript(services) |> ignore
        
        services
    
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
            return! metascriptService.ExecuteMetascriptFileAsync(filePath)
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
            let filePath = 
                if args.Length > 0 then
                    args.[0]
                else
                    Path.Combine(Directory.GetCurrentDirectory(), "simple_test.meta")
            
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
                Console.WriteLine($"Status: {result.Result.Status}")
                
                if result.Result.Status = MetascriptExecutionStatus.Success then
                    Console.WriteLine("Output:")
                    Console.WriteLine(result.Result.Output)
                    0
                else
                    Console.WriteLine("Error:")
                    Console.WriteLine(result.Result.Error)
                    1
        with
        | ex ->
            Console.WriteLine("Error:")
            Console.WriteLine(ex.ToString())
            1
