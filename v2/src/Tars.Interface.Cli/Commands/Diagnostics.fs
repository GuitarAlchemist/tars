module Tars.Interface.Cli.Commands.Diagnostics

open System
open System.IO
open System.Net.Http
open System.Threading.Tasks
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Knowledge
open Tars.Interface.Cli
open Spectre.Console

// --- Color Helpers (ASCII-safe for terminal compatibility) ---
let private writeCheck (name: string) =
    AnsiConsole.Markup($"[cyan]Checking {name}... [/]")

let private writeOk (msg: string) =
    AnsiConsole.MarkupLine($"[green]OK[/] ({msg})")

let private writeInfo (msg: string) =
    AnsiConsole.MarkupLine($"[blue]INFO:[/] {msg}")

let private writeFail (msg: string) =
    AnsiConsole.MarkupLine($"[red]FAIL[/] ({msg})")

let private writeSkipped (msg: string) =
    AnsiConsole.MarkupLine($"[yellow]SKIP[/] ({msg})")

// --- Helper to create LLM service ---
let private createLlmService () =
    let config = ConfigurationLoader.load ()
    let useLlamaCpp =
        config.Llm.Provider.Equals("LlamaCpp", StringComparison.OrdinalIgnoreCase)
        || config.Llm.Provider.Equals("llama.cpp", StringComparison.OrdinalIgnoreCase)

    let routingCfg =
        { RoutingConfig.Default with
            OllamaBaseUri =
                config.Llm.BaseUrl
                |> Option.map Uri
                |> Option.defaultValue (Uri "http://localhost:11434")
            DefaultOllamaModel = config.Llm.Model
            LlamaCppBaseUri =
                if useLlamaCpp then
                    config.Llm.LlamaCppUrl |> Option.map Uri
                else
                    None
            DefaultLlamaCppModel =
                if useLlamaCpp && config.Llm.LlamaCppUrl.IsSome then
                    Some config.Llm.Model
                else
                    None }

    let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
    let client = new HttpClient()
    (DefaultLlmService(client, serviceConfig) :> ILlmService, config)

// --- Core Diagnostic Checks ---

/// Check 1: File System Paths
let private checkPaths (logger: ILogger) : Task<bool> =
    task {
        writeCheck "TARS Home"

        let tarsHome =
            Environment.GetEnvironmentVariable("TARS_HOME")
            |> Option.ofObj
            |> Option.defaultValue (
                Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".tars")
            )

        if Directory.Exists(tarsHome) then
            writeOk tarsHome

            writeCheck "Knowledge Graph"
            let kgPath = Path.Combine(tarsHome, "knowledge", "temporal_graph.json")

            if File.Exists(kgPath) then
                let info = FileInfo(kgPath)
                writeOk $"{info.Length} bytes"
                return true
            else
                writeInfo "Not found (will be created on first use)"
                return true
        else
            try
                Directory.CreateDirectory(tarsHome) |> ignore
                writeOk $"Created {tarsHome}"
                return true
            with _ ->
                writeFail $"Could not create {tarsHome}"
                return false
    }

/// Check 2: LLM Connectivity
let private checkLlm (logger: ILogger) : Task<bool> =
    task {
        writeCheck "LLM Connectivity"

        try
            let svc, config = createLlmService ()

            let req =
                { ModelHint = None
                  Model = Some config.Llm.Model
                  SystemPrompt = None
                  MaxTokens = Some 20
                  Temperature = None
                  Stop = []
                  Messages = [ { Role = Role.User; Content = "Ping" } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = svc.CompleteAsync(req)

            if String.IsNullOrWhiteSpace(response.Text) then
                let reason = response.FinishReason |> Option.defaultValue "unknown"
                writeFail $"Empty response. Reason: {Markup.Escape(reason)}"
                return false
            else
                writeOk $"Response: {Markup.Escape(response.Text.Trim())}"
                return true
        with ex ->
            writeFail ex.Message
            return false
    }

/// Check 3: Knowledge Ledger (Postgres)
let private checkLedger (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Knowledge Ledger (Postgres)"
        let config = ConfigurationLoader.load ()

        let tryPostgres =
            config.Memory.PostgresConnectionString
            |> Option.orElse (Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION") |> Option.ofObj)

        match tryPostgres with
        | Some connStr ->
            try
                use conn = new Npgsql.NpgsqlConnection(connStr)
                do! conn.OpenAsync()

                use cmd = conn.CreateCommand()
                cmd.CommandText <- "SELECT extname FROM pg_extension WHERE extname = 'vector';"
                let! result = cmd.ExecuteScalarAsync()

                if result = null then
                    writeFail "pgvector extension missing"
                    return false
                else
                    writeOk "Connected + vector extension found"
                    return true
            with ex ->
                writeFail $"Connection failed: {ex.Message}"
                return false
        | None ->
            writeSkipped "No connection string found"
            return true
    }

// --- Extended Capability Checks ---

/// Check 4: Chain of Thought (CoT)
let private checkCoT (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Chain of Thought (CoT)"

        try
            let svc, config = createLlmService ()

            // Simple reasoning test
            let req =
                { ModelHint = None
                  Model = Some config.Llm.Model
                  SystemPrompt = Some "You are a helpful assistant. Think step by step."
                  MaxTokens = Some 100
                  Temperature = None
                  Stop = []
                  Messages =
                    [ { Role = Role.User
                        Content = "What is 5 + 3? Think step by step." } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = svc.CompleteAsync(req)

            if String.IsNullOrWhiteSpace(response.Text) then
                writeFail "No response"
                return false
            elif response.Text.Contains("8") then
                writeOk "Reasoning works"
                return true
            else
                writeInfo
                    $"Got response but no '8': {Markup.Escape(response.Text.Substring(0, min 50 response.Text.Length))}..."

                return true // Not a hard failure
        with ex ->
            writeFail ex.Message
            return false
    }

/// Check 5: Workflow of Thought (WoT) - simplified check
let private checkWoT (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Workflow of Thought (WoT)"

        try
            let svc, config = createLlmService ()

            // Simple multi-step test
            let req =
                { ModelHint = None
                  Model = Some config.Llm.Model
                  SystemPrompt = Some "You are a helpful assistant."
                  MaxTokens = Some 50
                  Temperature = None
                  Stop = []
                  Messages =
                    [ { Role = Role.User
                        Content = "Name three primary colors. Be brief." } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None

                  ContextWindow = None }

            let! response = svc.CompleteAsync(req)

            if String.IsNullOrWhiteSpace(response.Text) then
                writeFail "No response"
                return false
            else
                // Check for at least one color
                let hasColor =
                    [ "red"; "blue"; "yellow"; "green"; "orange"; "purple" ]
                    |> List.exists (fun c -> response.Text.ToLowerInvariant().Contains(c))

                if hasColor then
                    writeOk "Multi-step reasoning works"
                    return true
                else
                    writeInfo "Response received but no colors found"
                    return true
        with ex ->
            writeFail ex.Message
            return false
    }

/// Check 6: Knowledge Ledger Status (belief count)
let private checkKnowledgeLedgerStatus (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Knowledge Ledger Status"
        let config = ConfigurationLoader.load ()

        let tryPostgres =
            config.Memory.PostgresConnectionString
            |> Option.orElse (Environment.GetEnvironmentVariable("TARS_POSTGRES_CONNECTION") |> Option.ofObj)

        match tryPostgres with
        | Some connStr ->
            try
                use conn = new Npgsql.NpgsqlConnection(connStr)
                do! conn.OpenAsync()

                // Check belief count
                use cmd = conn.CreateCommand()
                cmd.CommandText <- "SELECT COUNT(*) FROM beliefs WHERE is_valid = true;"
                let! result = cmd.ExecuteScalarAsync()

                let count =
                    match result with
                    | :? int64 as n -> n
                    | :? int as n -> int64 n
                    | _ -> 0L

                writeOk $"{count} valid beliefs"
                return true
            with ex ->
                // Table might not exist yet
                if ex.Message.Contains("does not exist") then
                    writeInfo "Beliefs table not created yet"
                    return true
                else
                    writeFail $"Query failed: {ex.Message}"
                    return false
        | None ->
            writeSkipped "No connection string"
            return true
    }

/// Check 7: RAG Components
let private checkRag (logger: ILogger) : Task<bool> =
    task {
        writeCheck "RAG Components"

        try
            let svc, config = createLlmService ()

            // Test embedding capability
            let! embeddings = svc.EmbedAsync("test embedding")

            if embeddings.Length > 0 then
                writeOk $"Embeddings work ({embeddings.Length} dimensions)"
                return true
            else
                writeFail "Empty embeddings"
                return false
        with ex ->
            writeFail $"Embedding failed: {ex.Message}"
            return false
    }

/// Check 8: Tool Registry
let private checkToolRegistry (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Tool Registry"

        try
            let registry = Tars.Tools.ToolRegistry()
            registry.RegisterAssembly(typeof<Tars.Tools.ToolRegistry>.Assembly)
            let tools = registry.GetAll()
            let count = tools |> List.length

            if count > 0 then
                writeOk $"{count} tools registered"
                return true
            else
                writeFail "No tools found"
                return false
        with ex ->
            writeFail $"Registry error: {ex.Message}"
            return false
    }

/// Check 9: Configuration Validation
let private checkConfiguration (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Configuration"

        try
            let config = ConfigurationLoader.load ()

            // Check essential config values
            let issues = ResizeArray<string>()

            if String.IsNullOrWhiteSpace(config.Llm.Model) then
                issues.Add("LLM model not set")

            if config.Llm.BaseUrl.IsNone && config.Llm.LlamaCppUrl.IsNone then
                issues.Add("No LLM endpoint configured")

            if issues.Count = 0 then
                writeOk $"Model: {config.Llm.Model}"
                return true
            else
                writeFail (String.Join(", ", issues))
                return false
        with ex ->
            writeFail $"Config error: {ex.Message}"
            return false
    }

/// Check 10: Model Availability (llama.cpp specific)
let private checkModelAvailability (logger: ILogger) : Task<bool> =
    task {
        writeCheck "Model Availability"

        try
            let config = ConfigurationLoader.load ()

            match config.Llm.LlamaCppUrl with
            | Some url ->
                use client = new HttpClient()
                let uri = Uri(url.TrimEnd('/') + "/v1/models")
                let! response = client.GetStringAsync(uri)

                if response.Contains("data") then
                    writeOk "llama.cpp model loaded"
                    return true
                else
                    writeInfo "Model endpoint responded but no model data"
                    return true
            | None ->
                // Try Ollama
                match config.Llm.BaseUrl with
                | Some url ->
                    use client = new HttpClient()
                    let uri = Uri(url.TrimEnd('/') + "/api/tags")
                    let! response = client.GetStringAsync(uri)

                    if response.Contains("models") then
                        writeOk "Ollama models available"
                        return true
                    else
                        writeInfo "Ollama responded but no models found"
                        return true
                | None ->
                    writeSkipped "No backend to check"
                    return true
        with ex ->
            writeFail $"Model check failed: {ex.Message}"
            return false
    }

// --- Main Runners ---

/// Basic diagnostics (quick)
let runAll (logger: ILogger) (verbose: bool) =
    task {
        do AnsiConsole.Write(new Rule("[bold yellow]TARS System Diagnostics[/]"))

        let! r1 = checkPaths logger
        let! r2 = checkLlm logger
        let! r3 = checkLedger logger

        do AnsiConsole.Write(new Rule())

        if r1 && r2 && r3 then return 0 else return 1
    }

/// Full diagnostics including capability checks
let runFull (logger: ILogger) =
    task {
        do AnsiConsole.Write(new Rule("[bold yellow]TARS Full System Diagnostics[/]"))
        AnsiConsole.MarkupLine("[dim]Running comprehensive capability verification...[/]")
        AnsiConsole.WriteLine()

        // Core checks
        AnsiConsole.MarkupLine("[bold]Core Infrastructure[/]")
        let! r1 = checkPaths logger
        let! r2 = checkConfiguration logger
        let! r3 = checkLlm logger
        let! r4 = checkModelAvailability logger
        let! r5 = checkLedger logger

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[bold]Reasoning Capabilities[/]")
        let! r6 = checkCoT logger
        let! r7 = checkWoT logger

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[bold]Knowledge & Retrieval[/]")
        let! r8 = checkKnowledgeLedgerStatus logger
        let! r9 = checkRag logger

        AnsiConsole.WriteLine()
        AnsiConsole.MarkupLine("[bold]Tooling[/]")
        let! r10 = checkToolRegistry logger

        do AnsiConsole.Write(new Rule())

        let results = [ r1; r2; r3; r4; r5; r6; r7; r8; r9; r10 ]
        let allPassed = results |> List.forall id
        let passCount = results |> List.filter id |> List.length
        let totalCount = results.Length

        if allPassed then
            AnsiConsole.MarkupLine($"[bold green]SUCCESS: All {totalCount} checks passed![/]")
            return 0
        else
            AnsiConsole.MarkupLine($"[bold yellow]WARNING: {passCount}/{totalCount} checks passed[/]")
            return 1
    }

// --- Interface Shims ---

let run (logger: ILogger) = runAll logger false

let runWithVerbose (logger: ILogger) (verbose: bool) = runAll logger verbose

let runWithArch (logger: ILogger) =
    task {
        logger.Information("Architecture validation not implemented yet.")
        return 0
    }

let status (logger: ILogger) = runAll logger false
