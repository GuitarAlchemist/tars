namespace TarsEngine.FSharp.Core.Metascript.Examples

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
/// Simple program to test the Metascript module.
/// </summary>
type MetascriptTester() =
    /// <summary>
    /// Configures the service collection.
    /// </summary>
    /// <param name="services">The service collection.</param>
    /// <returns>The service collection.</returns>
    static member ConfigureServices(services: IServiceCollection) =
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
    static member RunMetascriptFromFile(serviceProvider: IServiceProvider, filePath: string) =
        task {
            // Get the metascript service
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            
            // Execute the metascript
            return! metascriptService.ExecuteMetascriptFileAsync(filePath)
        }
    
    /// <summary>
    /// Runs a metascript from text.
    /// </summary>
    /// <param name="serviceProvider">The service provider.</param>
    /// <param name="text">The metascript text.</param>
    /// <param name="name">The metascript name.</param>
    /// <returns>The execution result.</returns>
    static member RunMetascriptFromText(serviceProvider: IServiceProvider, text: string, ?name: string) =
        task {
            // Get the metascript service
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            
            // Execute the metascript
            return! metascriptService.ExecuteMetascriptTextAsync(text, ?name = name)
        }
    
    /// <summary>
    /// Runs the simple test.
    /// </summary>
    /// <returns>The execution result.</returns>
    static member RunSimpleTest() =
        // Configure services
        let services = ServiceCollection()
        let serviceProvider = MetascriptTester.ConfigureServices(services).BuildServiceProvider()
        
        // Get the file path
        let filePath = Path.Combine(Directory.GetCurrentDirectory(), "Metascript", "Examples", "simple_test.meta")
        
        // Run the metascript
        let result = MetascriptTester.RunMetascriptFromFile(serviceProvider, filePath)
        
        // Wait for the result
        result.Wait()
        
        // Print the result
        Console.WriteLine("Metascript execution completed.")
        Console.WriteLine($"Status: {result.Result.Status}")
        
        if result.Result.Status = MetascriptExecutionStatus.Success then
            Console.WriteLine("Output:")
            Console.WriteLine(result.Result.Output)
        else
            Console.WriteLine("Error:")
            Console.WriteLine(result.Result.Error)
        
        result.Result
