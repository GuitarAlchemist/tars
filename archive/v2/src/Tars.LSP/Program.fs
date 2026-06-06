module Tars.LSP.Program

open System
open System.Threading.Tasks
open Microsoft.Extensions.DependencyInjection
open Microsoft.Extensions.Logging
open OmniSharp.Extensions.LanguageServer.Server
open Tars.LSP

[<EntryPoint>]
let main _argv =
    let runServer () =
        task {
            let! server =
                LanguageServer.From(fun (options: LanguageServerOptions) ->
                    options
                        .WithInput(Console.OpenStandardInput())
                        .WithOutput(Console.OpenStandardOutput())
                        .WithHandler<WotTextDocumentSyncHandler>()
                        .WithHandler<WotCompletionHandler>()
                        .WithHandler<WotHoverHandler>()
                        .WithHandler<WotDocumentSymbolHandler>()
                        .WithServices(fun services ->
                            services.AddLogging(fun logging ->
                                logging.SetMinimumLevel(LogLevel.Warning) |> ignore
                            ) |> ignore
                        )
                    |> ignore
                )

            do! server.WaitForExit
        }

    runServer().GetAwaiter().GetResult()
    0
