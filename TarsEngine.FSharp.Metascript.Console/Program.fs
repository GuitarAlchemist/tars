open System
open System.IO
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.Metascript
open TarsEngine.FSharp.Metascript.Services
open TarsEngine.FSharp.Metascript.BlockHandlers

/// <summary>
/// Simple console application to test the Metascript module.
/// </summary>
module Program =
    /// <summary>
    /// Creates a simple metascript.
    /// </summary>
    /// <returns>The metascript.</returns>
    let createSimpleMetascript() =
        let fsharpBlock = {
            Type = MetascriptBlockType.FSharp
            Content = """
let greeting = "Hello, World!"
printfn "%s" greeting
greeting
"""
            LineNumber = 1
            ColumnNumber = 0
            Parameters = []
            Id = Guid.NewGuid().ToString()
            ParentId = None
            Metadata = Map.empty
        }
        
        let commandBlock = {
            Type = MetascriptBlockType.Command
            Content = """echo "Hello from the command line!""""
            LineNumber = 6
            ColumnNumber = 0
            Parameters = []
            Id = Guid.NewGuid().ToString()
            ParentId = None
            Metadata = Map.empty
        }
        
        let textBlock = {
            Type = MetascriptBlockType.Text
            Content = "This is a text block."
            LineNumber = 8
            ColumnNumber = 0
            Parameters = []
            Id = Guid.NewGuid().ToString()
            ParentId = None
            Metadata = Map.empty
        }
        
        {
            Name = "Simple Test"
            Blocks = [fsharpBlock; commandBlock; textBlock]
            FilePath = None
            CreationTime = DateTime.UtcNow
            LastModificationTime = None
            Description = Some "A simple metascript test"
            Author = Some "TARS"
            Version = Some "1.0.0"
            Dependencies = []
            Imports = []
            Metadata = Map.empty
        }
    
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
    /// Runs a metascript.
    /// </summary>
    /// <param name="serviceProvider">The service provider.</param>
    /// <param name="metascript">The metascript to run.</param>
    /// <returns>The execution result.</returns>
    let runMetascript (serviceProvider: IServiceProvider) (metascript: Metascript) =
        task {
            // Get the metascript service
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            
            // Execute the metascript
            return! metascriptService.ExecuteMetascriptAsync(metascript)
        }
    
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
            // Configure services
            let services = ServiceCollection()
            let serviceProvider = configureServices(services).BuildServiceProvider()
            
            // Register block handlers
            let registry = serviceProvider.GetRequiredService<BlockHandlerRegistry>()
            
            for handler in serviceProvider.GetServices<IBlockHandler>() do
                registry.RegisterHandler(handler)
            
            // Create a simple metascript
            let metascript = createSimpleMetascript()
            
            // Run the metascript
            let result = runMetascript serviceProvider metascript
            
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
