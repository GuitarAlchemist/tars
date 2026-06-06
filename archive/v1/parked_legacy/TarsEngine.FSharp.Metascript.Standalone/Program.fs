namespace TarsEngine.FSharp.Metascript.StandaloneApp

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
/// Standalone metascript runner that executes a metascript file passed on the command line.
/// </summary>
module Program =
    /// <summary>
    /// Configures the service collection.
    /// </summary>
    let configureServices (services: IServiceCollection) =
        services.AddLogging(fun logging ->
            logging.AddConsole() |> ignore
            logging.SetMinimumLevel(LogLevel.Information) |> ignore) |> ignore

        ServiceCollectionExtensions.addTarsEngineFSharpMetascript(services) |> ignore
        services

    /// <summary>
    /// Registers all discovered block handlers with the shared registry.
    /// </summary>
    let registerHandlers (provider: IServiceProvider) =
        let registry = provider.GetRequiredService<BlockHandlerRegistry>()
        for handler in provider.GetServices<IBlockHandler>() do
            registry.RegisterHandler(handler)
        registry

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
            let filePath =
                if args.Length > 0 then
                    args.[0]
                else
                    Path.Combine(Directory.GetCurrentDirectory(), "simple_test.meta")

            if not (File.Exists(filePath)) then
                printfn $"File not found: %s{filePath}"
                1
            else
                let services = ServiceCollection()
                let serviceProvider = configureServices services |> fun svc -> svc.BuildServiceProvider()
                registerHandlers serviceProvider |> ignore

                let resultTask = runMetascriptFromFile serviceProvider filePath
                resultTask.Wait()

                let result = resultTask.Result
                printfn "Metascript execution completed."
                printfn $"Status: %A{result.Status}"

                if result.Status = MetascriptExecutionStatus.Success then
                    printfn "Output:"
                    printfn $"%s{result.Output}"
                    0
                else
                    printfn "Error:"
                    printfn $"%A{result.Error}"
                    1
        with
        | ex ->
            printfn "Error:"
            printfn $"{ex}"
            1
