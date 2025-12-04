module Tars.Interface.Cli.Program

open System
open System.Text
open Serilog
open Tars.Security
open System.Threading.Tasks
open Tars.Interface.Cli.Commands
open Tars.Interface.Cli
open Microsoft.Extensions.Configuration

[<EntryPoint>]
let main argv =
    // Ensure Unicode/UTF-8 I/O for CLI output (fixes mojibake on Windows consoles)
    Encoding.RegisterProvider(CodePagesEncodingProvider.Instance)
    let isRedirected = Console.IsOutputRedirected || Console.IsErrorRedirected
    // When redirected (e.g., piping to clip), include BOM so downstream readers detect UTF-8
    let outputEncoding =
        if isRedirected then
            new UTF8Encoding(true)
        else
            new UTF8Encoding(false)

    Console.OutputEncoding <- outputEncoding
    Console.InputEncoding <- new UTF8Encoding(false)

    // Initialize Configuration
    let config =
        ConfigurationBuilder().AddEnvironmentVariables().AddUserSecrets<Commands.Demo.DemoAgent>().Build()

    // Register secrets from configuration into CredentialVault
    let email = config["OPENWEBUI_EMAIL"]
    let password = config["OPENWEBUI_PASSWORD"]
    let ollamaUrl = config["OLLAMA_BASE_URL"]

    if not (String.IsNullOrEmpty(email)) then
        CredentialVault.registerSecret "OPENWEBUI_EMAIL" email

    if not (String.IsNullOrEmpty(password)) then
        CredentialVault.registerSecret "OPENWEBUI_PASSWORD" password

    if not (String.IsNullOrEmpty(ollamaUrl)) then
        CredentialVault.registerSecret "OLLAMA_BASE_URL" ollamaUrl

    Log.Logger <- LoggerConfiguration().WriteTo.Console().CreateLogger()
    let logger = Log.Logger

    // Parse demo-rag options
    let parseRagOptions (args: string array) =
        let mutable options = RagDemo.defaultOptions
        let mutable i = 1 // Skip "demo-rag"

        while i < args.Length do
            match args.[i] with
            | "--quick" -> options <- { options with Quick = true }
            | "--verbose" -> options <- { options with Verbose = true }
            | "--output" when i + 1 < args.Length ->
                i <- i + 1

                options <-
                    { options with
                        OutputFormat = if args.[i] = "json" then RagDemo.Json else RagDemo.Text }
            | "--scenario" when i + 1 < args.Length ->
                i <- i + 1
                let nums = args.[i].Split(',') |> Array.map int |> Array.toList
                options <- { options with Scenarios = nums }
            | "--benchmark" when i + 1 < args.Length ->
                i <- i + 1

                options <-
                    { options with
                        BenchmarkRuns = int args.[i] }
            | "--export" when i + 1 < args.Length ->
                i <- i + 1

                options <-
                    { options with
                        ExportPath = Some args.[i] }
            | "--docs" when i + 1 < args.Length ->
                i <- i + 1

                options <-
                    { options with
                        DocsPath = Some args.[i] }
            | "--live" -> options <- { options with UseLiveLlm = true }
            | "--compare" -> options <- { options with CompareMode = true }
            | "--diag" -> options <- { options with ShowDiagnostics = true }
            | _ -> ()

            i <- i + 1

        options

    task {
        match argv with
        | [| "ask"; prompt |] -> return! Ask.run config prompt
        | [| "guard-output"; path |] ->
            // Defaults: no required fields, citations not required, extra fields not allowed
            return GuardOutputCommand.run config path None false false
        | [| "guard-output"; path; "--fields"; fields |] ->
            return GuardOutputCommand.run config path (Some fields) false false
        | [| "guard-output"; path; "--fields"; fields; "--allow-extra" |] ->
            return GuardOutputCommand.run config path (Some fields) false true
        | [| "guard-output"; path; "--require-citations" |] -> return GuardOutputCommand.run config path None true false
        | [| "guard-output"; path; "--fields"; fields; "--require-citations" |] ->
            return GuardOutputCommand.run config path (Some fields) true false
        | [| "test-grammar"; file |] -> return TestGrammar.run file
        | [| "memory-add"; coll; id; text |] -> return! Memory.add coll id text
        | [| "memory-search"; coll; text |] -> return! Memory.search coll text
        | args when args.Length > 0 && args.[0] = "smem" ->
            return! SemanticMemoryCommand.run config (args |> Array.skip 1)
        | [| "demo-ping" |] -> return! Demo.ping logger
        | [| "diag" |] -> return! Diagnostics.run logger
        | [| "diag"; "--verbose" |] -> return! Diagnostics.runWithVerbose logger true
        | [| "diag"; "--arch" |] -> return! Diagnostics.runWithArch logger
        | args when args.Length > 0 && args.[0] = "demo-rag" ->
            let options = parseRagOptions args
            return! RagDemo.runWithOptions logger options
        | [| "chat" |] ->
            Tui.showSplashScreen ()
            return! Chat.run logger
        | args when args.Length > 0 && args.[0] = "evolve" ->
            let mutable options: Evolve.EvolveOptions =
                { MaxIterations = 5
                  Quiet = false
                  DemoMode = false
                  Model = None
                  Trace = false }

            let mutable i = 1

            while i < args.Length do
                match args.[i] with
                | "--max-iterations" when i + 1 < args.Length ->
                    i <- i + 1
                    let mutable parsedVal = 0

                    if System.Int32.TryParse(args.[i], &parsedVal) then
                        options <-
                            { options with
                                MaxIterations = parsedVal }
                    else
                        printfn "Invalid number for --max-iterations"
                | "--quiet" -> options <- { options with Quiet = true }
                | "--demo" -> options <- { options with DemoMode = true }
                | "--trace" -> options <- { options with Trace = true }
                | "--model" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Model = Some args.[i] }
                | _ -> ()

                i <- i + 1

            if options.Quiet then
                System.Environment.SetEnvironmentVariable("TARS_NO_SPLASH", "1")

            return! Evolve.run logger options
        | [| "experiment" |] -> return! Experiment.run logger
        | [| "run"; script |] -> return! Run.execute logger script
        | args when args.Length > 0 && args.[0] = "knowledge" ->
            let mutable options: Knowledge.KnowledgeOptions =
                { Command = if args.Length > 1 then args.[1] else "help"
                  Query = None
                  Title = None
                  Content = None
                  Category = None
                  Tags = None }

            let mutable i = 2

            while i < args.Length do
                match args.[i] with
                | "--title" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Title = Some args.[i] }
                | "--content" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Content = Some args.[i] }
                | "--category" when i + 1 < args.Length ->
                    i <- i + 1

                    options <-
                        { options with
                            Category = Some args.[i] }
                | "--tags" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Tags = Some args.[i] }
                | arg when not (arg.StartsWith("--")) && options.Query.IsNone ->
                    options <- { options with Query = Some arg }
                | _ -> ()

                i <- i + 1

            Knowledge.run options
            return 0
        | _ ->
            Tui.showSplashScreen ()
            printfn "Usage:"
            printfn "  tars chat                        Start the interactive chat mode"
            printfn "  tars run <script.tars>           Execute a Metascript workflow"
            printfn "  tars ask <prompt>                Ask a question to the AI"
            printfn "  tars test-grammar <file>         Parse a grammar file"
            printfn "  tars memory-add <coll> <id> <text> Add text to vector memory"
            printfn "  tars memory-search <coll> <text> Search vector memory"
            printfn "  tars demo-ping                   Run a demo ping agent"
            printfn "  tars diag [--verbose]            Run system diagnostics"
            printfn "  tars demo-rag [options]          Run the RAG capabilities demo"
            printfn "       --quick                     Skip interactive prompts"
            printfn "       --verbose                   Show detailed internal state"
            printfn "       --scenario 1,5,14           Run specific scenarios"
            printfn "       --output json               Output results as JSON"
            printfn "       --benchmark N               Run each scenario N times"
            printfn "       --export <file>             Export results to JSON file"
            printfn "       --docs <folder>             Use custom documents folder"
            printfn "       --live                      Use live Ollama LLM (requires Ollama)"
            printfn "       --compare                   Compare before/after with configs"
            printfn "       --diag                      Run diagnostics before demo"
            printfn "  tars evolve [options]            Run the evolution engine"
            printfn "       --max-iterations N          Set max generations (default 5)"
            printfn "       --quiet                     Suppress splash screen"
            printfn "  tars knowledge <command>         Manage TARS knowledge base"
            printfn "       list [--category <cat>]     List all entries"
            printfn "       search <query>              Search entries"
            printfn "       show <id>                   Show entry details"
            printfn "       add --title --content       Add new entry"
            printfn "       delete <id>                 Delete entry"
            printfn "  tars experiment                  Run an A/B testing experiment"
            return 1
    }
    |> Async.AwaitTask
    |> Async.RunSynchronously
