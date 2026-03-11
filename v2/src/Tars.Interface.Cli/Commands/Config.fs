module Tars.Interface.Cli.Commands.Config

open System
open System.IO
open System.Net.Http
open System.Threading.Tasks
open Spectre.Console
open Tars.Security

/// Configuration keys supported by the CLI
let private configKeys =
    [ ("ollama-url", "OLLAMA_BASE_URL", "Ollama/OpenWebUI base URL")
      ("model", "DEFAULT_OLLAMA_MODEL", "Default Ollama model")
      ("webui-email", "OPENWEBUI_EMAIL", "OpenWebUI login email")
      ("webui-password", "OPENWEBUI_PASSWORD", "OpenWebUI login password")
      ("openai-key", "OPENAI_API_KEY", "OpenAI API Key")
      ("google-key", "GOOGLE_API_KEY", "Google Gemini API Key")
      ("deepseek-key", "DEEPSEEK_API_KEY", "DeepSeek API Key (OpenAI Compatible)")
      ("anthropic-key", "ANTHROPIC_API_KEY", "Anthropic API Key")
      ("chroma-url", "CHROMA_URL", "ChromaDB URL (e.gyles, http://localhost:8000)") ]

let private mask (v: string) =
    if String.IsNullOrEmpty v then "****"
    elif v.Length <= 4 then "****"
    else v.Substring(0, 4) + "****"

let private getSecretOrEnv (key: string) =
    match Environment.GetEnvironmentVariable(key) with
    | value when not (isNull value) -> Some("env", value)
    | _ ->
        match CredentialVault.getSecret key with
        | Ok v -> Some("secrets.json", v)
        | Error _ -> None

/// Show current LLM configuration
let show () =
    // Load secrets from disk first
    let secretsPath = "secrets.json"

    match CredentialVault.loadSecretsFromDisk secretsPath with
    | Ok() -> ()
    | Error _ -> ()

    AnsiConsole.MarkupLine("[bold cyan]TARS LLM Configuration[/]")
    AnsiConsole.WriteLine()

    let table = Table()
    table.AddColumn("Setting") |> ignore
    table.AddColumn("Value") |> ignore
    table.AddColumn("Source") |> ignore

    for (cliKey, envKey, description) in configKeys do
        match getSecretOrEnv envKey with
        | Some(source, value) ->
            let displayValue = if cliKey.Contains("password") then mask value else value

            table.AddRow(cliKey, $"[green]{Markup.Escape(displayValue)}[/]", $"[dim]{source}[/]")
            |> ignore
        | None -> table.AddRow(cliKey, "[red]not set[/]", "[dim]n/a[/]") |> ignore

    AnsiConsole.Write(table)
    AnsiConsole.WriteLine()
    AnsiConsole.MarkupLine("[dim]Use 'tars config set <key> <value>' to update configuration[/]")
    Task.FromResult(0)

/// Set a configuration value in secrets.json
let set (key: string) (value: string) =
    task {
        // Find the matching config key
        let matchingKey = configKeys |> List.tryFind (fun (cliKey, _, _) -> cliKey = key)

        match matchingKey with
        | None ->
            AnsiConsole.MarkupLine($"[red]Unknown configuration key: {Markup.Escape(key)}[/]")
            AnsiConsole.MarkupLine("[yellow]Available keys:[/]")

            for (cliKey, _, desc) in configKeys do
                AnsiConsole.MarkupLine($"  [cyan]{cliKey}[/] - {desc}")

            return 1
        | Some(_, envKey, description) ->
            // Clear any existing environment variable so our new value takes effect
            let envValue = Environment.GetEnvironmentVariable(envKey)

            if not (isNull envValue) then
                AnsiConsole.MarkupLine($"[dim]Clearing environment variable {envKey}...[/]")
                Environment.SetEnvironmentVariable(envKey, null)

            // Load existing secrets.json or create new
            let secretsPath = "secrets.json"

            let secrets =
                try
                    if File.Exists(secretsPath) then
                        let json = File.ReadAllText(secretsPath)

                        System.Text.Json.JsonSerializer.Deserialize<
                            System.Collections.Generic.Dictionary<string, string>
                         >(
                            json
                        )
                    else
                        System.Collections.Generic.Dictionary<string, string>()
                with _ ->
                    System.Collections.Generic.Dictionary<string, string>()

            // Update the value
            secrets.[envKey] <- value

            // Write back to file with indentation
            let options = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
            let json = System.Text.Json.JsonSerializer.Serialize(secrets, options)
            File.WriteAllText(secretsPath, json)

            // Also register in memory for immediate use
            CredentialVault.registerSecret envKey value

            let displayValue = if key.Contains("password") then mask value else value
            AnsiConsole.MarkupLine($"[green]✓[/] Set [cyan]{key}[/] = [yellow]{Markup.Escape(displayValue)}[/]")
            AnsiConsole.MarkupLine($"[dim]Saved to {secretsPath}[/]")
            return 0
    }

/// Test the LLM connection
let test () : Task<int> =
    task {
        AnsiConsole.MarkupLine("[bold cyan]Testing LLM Connection...[/]")
        AnsiConsole.WriteLine()

        // Load secrets
        let secretsPath = "secrets.json"
        let _ = CredentialVault.loadSecretsFromDisk secretsPath

        // Get Ollama URL
        let ollamaUrl =
            match getSecretOrEnv "OLLAMA_BASE_URL" with
            | Some(source, url) ->
                AnsiConsole.MarkupLine($"[dim]Using URL from {source}: {url}[/]")
                url
            | None ->
                AnsiConsole.MarkupLine("[yellow]No OLLAMA_BASE_URL configured, using default[/]")
                "http://localhost:11434/"

        try
            use httpClient = new HttpClient()
            httpClient.Timeout <- TimeSpan.FromSeconds(10.0)

            let baseUri = Uri(ollamaUrl)

            // Determine API path - use /ollama/ prefix for OpenWebUI (non-localhost)
            let isRemote = baseUri.Host <> "localhost" && baseUri.Host <> "127.0.0.1"
            let apiPrefix = if isRemote then "ollama/api/" else "api/"

            // Note: Authentication skipped for now - requires fixing async branching issues
            if isRemote then
                AnsiConsole.MarkupLine("[dim]Remote endpoint detected - authentication not yet implemented[/]")

            // Try to get tags/models list
            let tagsUri = Uri(baseUri, apiPrefix + "tags")
            AnsiConsole.MarkupLine($"[dim]Connecting to {tagsUri}...[/]")

            let! response = httpClient.GetAsync(tagsUri)

            if response.IsSuccessStatusCode then
                let! content = response.Content.ReadAsStringAsync()
                AnsiConsole.MarkupLine("[green]✓ Connection successful![/]")

                // Try to parse and show models
                try
                    let doc = System.Text.Json.JsonDocument.Parse(content)

                    match doc.RootElement.TryGetProperty("models") with
                    | true, modelsEl ->
                        let models = modelsEl.EnumerateArray() |> Seq.toList
                        AnsiConsole.MarkupLine($"[cyan]Available models ({models.Length}):[/]")

                        for model in models |> List.truncate 10 do
                            match model.TryGetProperty("name") with
                            | true, nameEl -> AnsiConsole.MarkupLine($"  • {nameEl.GetString()}")
                            | _ -> ()

                        if models.Length > 10 then
                            AnsiConsole.MarkupLine($"  [dim]... and {models.Length - 10} more[/]")
                    | _ ->
                        AnsiConsole.MarkupLine(
                            $"[dim]Response: {content.Substring(0, Math.Min(200, content.Length))}...[/]"
                        )
                with _ ->
                    AnsiConsole.MarkupLine(
                        $"[dim]Response: {content.Substring(0, Math.Min(200, content.Length))}...[/]"
                    )

                return 0
            else
                let! errorContent = response.Content.ReadAsStringAsync()
                AnsiConsole.MarkupLine($"[red]✗ Connection failed: {response.StatusCode}[/]")

                // Show response body preview for debugging
                if errorContent.Length > 0 then
                    let preview = errorContent.Substring(0, Math.Min(100, errorContent.Length))
                    AnsiConsole.MarkupLine($"[dim]Response: {Markup.Escape(preview)}...[/]")

                // Check if it needs authentication
                if
                    response.StatusCode = System.Net.HttpStatusCode.Unauthorized
                    || response.StatusCode = System.Net.HttpStatusCode.Forbidden
                then
                    AnsiConsole.MarkupLine("[yellow]This endpoint may require authentication.[/]")
                    AnsiConsole.MarkupLine("[dim]Set webui-email and webui-password if using OpenWebUI[/]")

                return 1
        with ex ->
            AnsiConsole.MarkupLine($"[red]✗ Connection error: {Markup.Escape(ex.Message)}[/]")
            return 1
    }

/// Run config command based on args
let run (args: string list) =
    task {
        match args with
        | []
        | [ "show" ] -> return! show ()
        | [ "set"; key; value ] -> return! set key value
        | [ "set"; key ] ->
            AnsiConsole.MarkupLine("[red]Missing value. Usage: tars config set <key> <value>[/]")
            return 1
        | [ "test" ] -> return! test ()
        | _ ->
            AnsiConsole.MarkupLine("[bold cyan]TARS Configuration[/]")
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("Usage:")
            AnsiConsole.MarkupLine("  tars config show                  Show current configuration")
            AnsiConsole.MarkupLine("  tars config set <key> <value>     Set a configuration value")
            AnsiConsole.MarkupLine("  tars config test                  Test LLM connection")
            AnsiConsole.WriteLine()
            AnsiConsole.MarkupLine("Available keys:")

            for (cliKey, _, desc) in configKeys do
                AnsiConsole.MarkupLine($"  [cyan]{cliKey}[/] - {desc}")

            return 0
    }
