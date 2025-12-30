module Tars.Interface.Cli.Program

open System
open System.Text
open Serilog
open Tars.Security
open System.Threading.Tasks
open Tars.Interface.Cli.Commands
open Tars.Interface.Cli.Commands.AgentHelpers
open Tars.Interface.Cli
open Microsoft.Extensions.Configuration
open Spectre.Console

[<EntryPoint>]
let main argv =
    // Ensure Unicode/UTF-8 I/O for CLI output (fixes mojibake on Windows consoles)
    Encoding.RegisterProvider(CodePagesEncodingProvider.Instance)
    let isRedirected = Console.IsOutputRedirected || Console.IsErrorRedirected
    
    // When redirected (e.g., piping to clip), use ASCII-safe output
    if isRedirected then
        // Use Code Page 437 (OEM) which supports box-drawing characters
        // This is compatible with Windows cmd.exe and clip.exe
        Console.OutputEncoding <- Encoding.GetEncoding(437)
        // Disable ANSI escape codes for Spectre.Console when output is redirected
        AnsiConsole.Profile.Capabilities.Ansi <- false
        AnsiConsole.Profile.Capabilities.Links <- false
        AnsiConsole.Profile.Capabilities.Interactive <- false
    else
        Console.OutputEncoding <- new UTF8Encoding(false)
    
    Console.InputEncoding <- new UTF8Encoding(false)

    // Initialize Configuration (read appsettings.json, then env, then user secrets)
    let config =
        ConfigurationBuilder()
            .SetBasePath(AppContext.BaseDirectory)
            .AddJsonFile("appsettings.json", optional = true, reloadOnChange = false)
            .AddEnvironmentVariables()
            .AddUserSecrets<Commands.Demo.DemoAgent>()
            .Build()


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

    let tarsConfig = ConfigurationLoader.load ()

    let isMcpServer =
        match argv with
        | [| "mcp"; "server" |] -> true
        | _ -> false

    if isMcpServer then
        // Redirect all logs to Stderr to keep Stdout clean for JSON-RPC
        Log.Logger <-
            LoggerConfiguration()
                .WriteTo.Console(standardErrorFromLevel = Serilog.Events.LogEventLevel.Verbose)
                .CreateLogger()
    else
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
            | _ -> ()

            i <- i + 1

        options

    // Parse demo-puzzle options
    let parsePuzzleOptions (args: string array) =
        let mutable options = PuzzleDemo.defaultOptions
        let mutable i = 1 // Skip "demo-puzzle"

        while i < args.Length do
            match args.[i] with
            | "--all" -> options <- { options with All = true }
            | "--verbose" -> options <- { options with Verbose = true }
            | "--output" when i + 1 < args.Length ->
                i <- i + 1

                options <-
                    { options with
                        OutputFormat =
                            if args.[i] = "json" then
                                PuzzleDemo.Json
                            else
                                PuzzleDemo.Text }
            | "--difficulty" when i + 1 < args.Length ->
                i <- i + 1
                let mutable d = 0

                if Int32.TryParse(args.[i], &d) then
                    options <- { options with Difficulty = Some d }
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
            | "--internet" -> options <- { options with Internet = true }
            | "--internet-count" when i + 1 < args.Length ->
                i <- i + 1
                let mutable count = 3
                if Int32.TryParse(args.[i], &count) then
                    options <- { options with InternetCount = count }
            | arg when not (arg.StartsWith("--")) && options.PuzzleName.IsNone ->
                options <- { options with PuzzleName = Some arg }
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
        | [| "test-grammar" |] -> return! TestGrammarCommand.run config argv
        | [| "memory-add"; coll; id; text |] -> return! Memory.add coll id text
        | [| "memory-search"; coll; text |] -> return! Memory.search coll text
        | args when args.Length > 0 && args.[0] = "smem" ->
            return! SemanticMemoryCommand.run config (args |> Array.skip 1)
        | [| "demo-ping" |] -> return! Demo.ping logger
        | [| "status" |] -> return! Diagnostics.status logger
        | [| "diag" |] -> return! Diagnostics.run logger
        | [| "diag"; "--verbose" |] -> return! Diagnostics.runWithVerbose logger true
        | [| "diag"; "--full" |] -> return! Diagnostics.runFull logger
        | [| "diag"; "--arch" |] -> return! Diagnostics.runWithArch logger
        | args when args.Length > 1 && args.[0] = "diag" && args.[1] = "reasoning" ->
            let diagArgs = args |> Array.skip 2 |> Array.toList
            return! ReasoningDiag.run logger config tarsConfig diagArgs

        | args when args.Length > 0 && args.[0] = "demo-rag" ->
            let options = parseRagOptions args
            return! RagDemo.runWithOptions logger options

        | [| "macro-demo" |] -> return! Commands.MacroDemo.run logger

        // Escape Room Demo
        | [| "demo-escape" |] -> return! EscapeRoomDemo.run logger 30 false
        | [| "demo-escape"; "--verbose" |] -> return! EscapeRoomDemo.run logger 30 true
        | [| "demo-escape"; "--max-turns"; n |] ->
            match Int32.TryParse(n) with
            | true, turns -> return! EscapeRoomDemo.run logger turns false
            | _ ->
                printfn "Invalid turn count: %s" n
                return 1

        // Benchmark
        | args when args.Length > 0 && args.[0] = "benchmark" -> return! Benchmark.run logger (args |> Array.skip 1)

        // Puzzle Demo - now with benchmark/export support
        | args when args.Length > 0 && args.[0] = "demo-puzzle" ->
            if args.Length = 1 then
                // No arguments - list puzzles
                return PuzzleDemo.listPuzzles ()
            else
                let options = parsePuzzleOptions args
                return! PuzzleDemo.runWithOptions logger options

        // Smoke Test - verify LLM works
        | args when args.Length > 0 && args.[0] = "smoke-test" -> return! SmokeTest.runSmokeTest logger

        // Incident Management
        | args when args.Length > 0 && args.[0] = "incident" ->
            return! IncidentCmd.run (args |> Array.skip 1 |> Array.toList)

        // Config commands
        | args when args.Length > 0 && args.[0] = "config" ->
            let configArgs = args |> Array.skip 1 |> Array.toList
            return! Config.run configArgs

        | args when args.Length > 0 && args.[0] = "chat" ->
            Tui.showSplashScreen ()

            let mutable options: Chat.ChatOptions = { Streaming = false; Model = None }

            let mutable i = 1

            while i < args.Length do
                match args.[i] with
                | "--model" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Model = Some args.[i] }
                | "--stream" -> options <- { options with Streaming = true }
                | _ -> ()

                i <- i + 1

            if options.Streaming then
                return! Chat.runStreaming logger options
            else
                return! Chat.run logger options

        | args when args.Length > 0 && args.[0] = "evolve" ->
            let mutable options: Evolve.EvolveOptions =
                { MaxIterations = 5
                  Quiet = false
                  DemoMode = false
                  Verbose = false
                  Model = None
                  Trace = false
                  Budget = None
                  DisableGraphiti = false
                  PlanPath = None
                  Focus = None
                  ResearchEnhanced = false }

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
                | "--verbose"
                | "-v" -> options <- { options with Verbose = true }
                | "--trace" -> options <- { options with Trace = true }
                | "--model" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Model = Some args.[i] }
                | "--budget" when i + 1 < args.Length ->
                    i <- i + 1
                    let mutable parsedVal = 0.0m

                    if System.Decimal.TryParse(args.[i], &parsedVal) then
                        options <- { options with Budget = Some parsedVal }
                    else
                        printfn "Invalid number for --budget"
                | "--no-graphiti" -> options <- { options with DisableGraphiti = true }
                | "--plan" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with PlanPath = Some args.[i] }
                | "--focus" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Focus = Some args.[i] }
                | "--research" -> options <- { options with ResearchEnhanced = true }
                | _ -> ()

                i <- i + 1

            if options.Quiet then
                System.Environment.SetEnvironmentVariable("TARS_NO_SPLASH", "1")

            return! Evolve.run logger options
        | [| "experiment" |] -> return! Experiment.run logger
        | args when args.Length > 0 && args.[0] = "run" ->
            if args.Length < 2 then
                printfn "Usage: tars run <workflow.json> [--optimize]"
                return 1
            else
                let script = args.[1]

                let shouldOptimize =
                    args |> Array.contains "--optimize" || args |> Array.contains "-o"

                return! RunCommand.run logger script shouldOptimize
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

            // Legacy knowledge command - show deprecation notice
            AnsiConsole.MarkupLine("[yellow]⚠️ 'tars knowledge' is deprecated. Use 'tars know' instead.[/]")
            Knowledge.run options
            return 0

        // TARS Plan - Phase 9.3 Evolving Plans
        | args when args.Length > 0 && args.[0] = "plan" ->
            let planArgs = args |> Array.skip 1
            let options = PlanCmd.parseArgs planArgs
            return! PlanCmd.run tarsConfig options

        // TARS Know - Phase 9 Knowledge Ledger
        | args when args.Length > 0 && args.[0] = "know" ->
            let knowArgs = args |> Array.skip 1
            let options = KnowCmd.parseArgs knowArgs
            return! KnowCmd.run tarsConfig options

        | args when args.Length > 0 && args.[0] = "mcp" ->
            if args.Length > 1 && args.[1] = "server" then
                return! McpServerCommand.run logger args
            else
                let cmd = if args.Length > 1 then args.[1] else "help"
                let arg = if args.Length > 2 then args.[2] else ""
                return! McpCommand.run cmd arg

        | args when args.Length > 0 && args.[0] = "pipeline" -> return PipelineCommand.run (args |> Array.skip 1)
        | args when args.Length > 0 && args.[0] = "skill" ->
            let subCmd = if args.Length > 1 then args.[1] else "help"
            let subArgs = if args.Length > 2 then args.[2..] |> Array.toList else []
            return! SkillCommand.run subCmd subArgs

        | args when args.Length > 0 && args.[0] = "agent" ->
            let mutable options: AgentOptions = defaultOptions
            let mutable subCommand = "help"
            let mutable goalArgs: string list = []
            let mutable i = 1

            // First arg is sub-command
            if args.Length > 1 then
                subCommand <- args.[1]
                i <- 2

            while i < args.Length do
                match args.[i] with
                | "--max-steps" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with MaxSteps = int args.[i] }
                | "--verbose"
                | "-v" -> options <- { options with Verbose = true }
                | "--model" when i + 1 < args.Length ->
                    i <- i + 1
                    options <- { options with Model = Some args.[i] }
                | "--evidence" when i + 1 < args.Length ->
                    i <- i + 1

                    options <-
                        { options with
                            EvidencePath = Some args.[i] }
                | arg when not (arg.StartsWith("--")) -> goalArgs <- goalArgs @ [ arg ]
                | _ -> ()

                i <- i + 1

            return! Agent.run config subCommand goalArgs options
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
            printfn "  tars diag [--verbose|--full]     Run system diagnostics (--full for all checks)"

            printfn
                "  tars diag reasoning <wot|tot|got> <goal> [--ledger|--no-ledger] [--evidence <path>] Run reasoning diag + ledger trace"

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
            printfn "  tars demo-escape [options]       Watch TARS solve an escape room puzzle"
            printfn "       --max-turns N               Maximum turns allowed (default 30)"
            printfn "       --verbose                   Show detailed reasoning"
            printfn "  tars demo-puzzle [options]       Test TARS on classic AI reasoning puzzles"
            printfn "       (no args)                   List available puzzles"
            printfn "       --all                       Run all puzzles"
            printfn "       --difficulty N              Run puzzles up to difficulty N (1-5)"
            printfn "       <name>                      Run a specific puzzle by name"
            printfn "       --verbose                   Show detailed reasoning"
            printfn "       --output json               Output results as JSON"
            printfn "       --benchmark N               Run each puzzle N times"
            printfn "       --export <file>             Export results to JSON file"
            printfn "  tars evolve [options]            Run the evolution engine"
            printfn "       --max-iterations N          Set max generations (default 5)"
            printfn "       --budget USD                Maximum monetary budget in USD"
            printfn "       --quiet                     Suppress splash screen"
            printfn "  tars knowledge <command>         Manage TARS knowledge base"
            printfn "       list [--category <cat>]     List all entries"
            printfn "       search <query>              Search entries"
            printfn "       show <id>                   Show entry details"
            printfn "       add --title --content       Add new entry"
            printfn "       delete <id>                 Delete entry"
            printfn "  tars config [command]            Manage LLM configuration"
            printfn "       show                        Show current configuration"
            printfn "       set <key> <value>           Set configuration value"
            printfn "       test                        Test LLM connection"
            printfn "  tars experiment                  Run an A/B testing experiment"
            printfn "  tars pipeline [command]          Manage project pipelines"
            printfn "       new <id> [-t template]      Create a new project"
            printfn "       list                        List all projects"
            printfn "       status <id>                 Show project status"
            printfn "       run <id>                    Run project pipeline"
            printfn "       demo <id> [-f format]       Generate demo output"
            printfn "  tars skill [command]             Manage MCP skills"
            printfn "       list                        List installed skills"
            printfn "       catalog                     Show available skills"
            printfn "       install <name>              Install a skill"
            printfn "       remove <name>               Remove a skill"
            printfn "  tars agent [command]             Run agentic patterns"
            printfn "       react <goal>                Run ReAct reasoning loop"
            printfn "       cot <input>                 Run Chain of Thought"
            printfn "       got <goal>                  Run Graph of Thoughts"
            printfn "       tot <goal>                  Run Tree of Thoughts"
            printfn "       wot <goal>                  Run Workflow of Thoughts"
            printfn "  tars know <command>              TARS Knowledge Ledger (Phase 9)"
            printfn "       status [--pg]               Show ledger statistics"
            printfn "       assert <s> <p> <o> [--pg]   Add a belief triple"
            printfn "       query [--pg]                Search beliefs"
            printfn "       fetch <topic> [--pg]        Fetch Wikipedia summary"
            printfn "       propose <topic> [--pg]      Extract triples via LLM"
            printfn ""
            return 1
    }
    |> Async.AwaitTask
    |> Async.RunSynchronously
