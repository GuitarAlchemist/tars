/// <summary>
/// TARS Code Protection Demo - Standalone Program
/// Demonstrates the TARS protection system functionality
/// Run with: dotnet run --project TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli.fsproj ProtectDemo [command]
/// </summary>

open System
open System.Threading.Tasks
open TarsEngine.FSharp.Cli.Commands

[<EntryPoint>]
let main args =
    task {
        try
            // Run the protection demo
            do! ProtectCommandDemo.runDemo(args)
            return 0
        with ex ->
            printfn "Demo failed: %s" ex.Message
            return 1
    }
    |> Async.AwaitTask
    |> Async.RunSynchronously
