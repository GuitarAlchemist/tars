#r "nuget: Microsoft.Extensions.Logging, 9.0.0"
#r "nuget: Microsoft.Extensions.Logging.Console, 9.0.0"

#r "TarsEngine.FSharp.Core/bin/Release/net9.0/TarsEngine.FSharp.Core.dll"
#r "TarsEngine.FSharp.SelfImprovement/bin/Release/net9.0/TarsEngine.FSharp.SelfImprovement.dll"

open System
open Microsoft.Extensions.Logging
open TarsEngine.FSharp.SelfImprovement

let loggerFactory =
    LoggerFactory.Create(fun builder ->
        builder
            .AddSimpleConsole(fun options ->
                options.SingleLine <- true
                options.TimestampFormat <- "HH:mm:ss "
            )
        |> ignore)

use httpClient = new System.Net.Http.HttpClient()
let service = SelfImprovementService(httpClient, loggerFactory.CreateLogger<SelfImprovementService>())

printfn "Running Spec Kit autonomous iteration..."

let result =
    service.RunNextSpecKitIterationAsync(loggerFactory)
    |> Async.RunSynchronously

printfn "Iteration result: %A" result
