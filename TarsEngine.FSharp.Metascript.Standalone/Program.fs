open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.Services
open TarsEngine.FSharp.Metascript.BlockHandlers
open TarsEngine.FSharp.Metascript.DependencyInjection

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
        
        // Add block handlers
        services.AddSingleton<IBlockHandler, FSharpBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, CommandBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, TextBlockHandler>() |> ignore
        
        // Add registry
        services.AddSingleton<BlockHandlerRegistry>() |> ignore
        
        // Add executor and service
        services.AddSingleton<IMetascriptExecutor, MetascriptExecutor>() |> ignore
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore
        
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
                
                // Register block handlers
                let registry = serviceProvider.GetRequiredService<BlockHandlerRegistry>()
                
                for handler in serviceProvider.GetServices<IBlockHandler>() do
                    registry.RegisterHandler(handler)
                
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
