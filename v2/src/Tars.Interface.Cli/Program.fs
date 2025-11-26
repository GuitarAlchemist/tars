module Tars.Interface.Cli.Program

open System
open Serilog
open Tars.Security
open System.Threading.Tasks
open Tars.Interface.Cli.Commands
open Microsoft.Extensions.Configuration

[<EntryPoint>]
let main argv =
    // Initialize Configuration
    let config =
        ConfigurationBuilder().AddUserSecrets<Commands.Demo.DemoAgent>().AddEnvironmentVariables().Build()

    // Register secrets from configuration into CredentialVault
    let email = config["OPENWEBUI_EMAIL"]
    let password = config["OPENWEBUI_PASSWORD"]

    if not (String.IsNullOrEmpty(email)) then
        CredentialVault.registerSecret "OPENWEBUI_EMAIL" email

    if not (String.IsNullOrEmpty(password)) then
        CredentialVault.registerSecret "OPENWEBUI_PASSWORD" password

    Log.Logger <- LoggerConfiguration().WriteTo.Console().CreateLogger()
    let logger = Log.Logger

    task {
        match argv with
        | [| "ask"; prompt |] -> return! Ask.run prompt
        | [| "test-grammar"; file |] -> return TestGrammar.run file
        | [| "memory-add"; coll; id; text |] -> return! Memory.add coll id text
        | [| "memory-search"; coll; text |] -> return! Memory.search coll text
        | [| "demo-ping" |] -> return! Demo.ping logger
        | [| "chat" |] -> return! Chat.run logger
        | [| "evolve" |] -> return! Evolve.run logger
        | _ ->
            printfn "Usage:"
            printfn "  tars chat                        Start the interactive chat mode"
            printfn "  tars ask <prompt>                Ask a question to the AI"
            printfn "  tars test-grammar <file>         Parse a grammar file"
            printfn "  tars memory-add <coll> <id> <text> Add text to vector memory"
            printfn "  tars memory-search <coll> <text> Search vector memory"
            printfn "  tars demo-ping                   Run a demo ping agent"
            printfn "  tars evolve                      Run the evolution engine"
            return 1
    }
    |> Async.AwaitTask
    |> Async.RunSynchronously
