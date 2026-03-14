namespace TarsEngine.FSharp.Metascript.ConsoleApp

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
    let createSimpleMetascript () =
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
            Content = "echo \"Hello from the command line!\""
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
            Blocks = [ fsharpBlock; commandBlock; textBlock ]
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
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore) |> ignore

        services.AddSingleton<IBlockHandler, FSharpBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, CommandBlockHandler>() |> ignore
        services.AddSingleton<IBlockHandler, TextBlockHandler>() |> ignore

        services.AddSingleton<BlockHandlerRegistry>() |> ignore
        services.AddSingleton<IMetascriptExecutor, MetascriptExecutor>() |> ignore
        services.AddSingleton<IMetascriptService, MetascriptService>() |> ignore

        services

    /// <summary>
    /// Runs a metascript.
    /// </summary>
    let runMetascript (serviceProvider: IServiceProvider) (metascript: Metascript) =
        task {
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            return! metascriptService.ExecuteMetascriptAsync(metascript)
        }

    /// <summary>
    /// Runs a metascript from a file.
    /// </summary>
    let runMetascriptFromFile (serviceProvider: IServiceProvider) (filePath: string) =
        task {
            let metascriptService = serviceProvider.GetRequiredService<IMetascriptService>()
            return! metascriptService.ExecuteMetascriptFileAsync(filePath)
        }

    /// <summary>
    /// Main entry point.
    /// </summary>
    [<EntryPoint>]
    let main args =
        try
            let services = ServiceCollection()
            let serviceProvider = configureServices services |> fun svc -> svc.BuildServiceProvider()

            let registry = serviceProvider.GetRequiredService<BlockHandlerRegistry>()
            for handler in serviceProvider.GetServices<IBlockHandler>() do
                registry.RegisterHandler(handler)

            let metascript = createSimpleMetascript ()
            let resultTask = runMetascript serviceProvider metascript

            resultTask.Wait()

            printfn "Metascript execution completed."
            printfn $"Status: %A{resultTask.Result.Status}"

            if resultTask.Result.Status = MetascriptExecutionStatus.Success then
                printfn "Output:"
                printfn $"%s{resultTask.Result.Output}"
                0
            else
                printfn "Error:"
                printfn $"%A{resultTask.Result.Error}"
                1
        with
        | ex ->
            printfn "Error:"
            printfn $"{ex}"
            1
