namespace Tars.Llm

open System
open System.Diagnostics
open System.IO
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm.Routing

/// ILlmService implementation that delegates to Claude Code CLI as a subprocess.
/// Spawns `claude -p --output-format json`, writes the prompt to its stdin and parses the response.
/// This lets any TARS component tap into Claude's reasoning without API keys —
/// it piggybacks on the user's authenticated Claude Code session.
module ClaudeCodeService =

    /// Configuration for the Claude Code subprocess adapter.
    type ClaudeCodeConfig =
        { /// Path to the `claude` executable (default: "claude" on PATH)
          ClaudePath: string
          /// Timeout for subprocess execution
          Timeout: TimeSpan
          /// Whether to pass --verbose flag
          Verbose: bool
          /// Optional model override (e.g. "sonnet", "opus")
          Model: string option }

    let defaultConfig =
        { ClaudePath = "claude"
          Timeout = TimeSpan.FromMinutes(2.0)
          Verbose = false
          Model = None }

    /// The request's system prompt: its SystemPrompt, then its system messages.
    let systemPromptOf (req: LlmRequest) : string =
        [ yield! Option.toList req.SystemPrompt
          for msg in req.Messages do
              match msg.Role with
              | Role.System -> yield msg.Content
              | _ -> () ]
        |> String.concat "\n\n"

    /// Build the prompt string from an LlmRequest: the conversation, without the system
    /// prompt, which goes to its own file (`systemPromptOf`).
    let buildPrompt (req: LlmRequest) : string =
        let sb = StringBuilder()

        for msg in req.Messages do
            match msg.Role with
            | Role.System -> ()
            | Role.User -> sb.AppendLine(msg.Content) |> ignore
            | Role.Assistant -> sb.AppendLine(sprintf "<assistant>%s</assistant>" msg.Content) |> ignore
            | Role.Tool _ -> sb.AppendLine(sprintf "<tool_result>%s</tool_result>" msg.Content) |> ignore
            | Role.AssistantCalling _ -> sb.AppendLine(sprintf "<assistant>%s</assistant>" msg.Content) |> ignore

        sb.ToString().Trim()

    /// The `claude` process for one completion. The prompt goes to its stdin (a command
    /// line holds at most 32,767 characters on Windows) and the system prompt to a file.
    /// The call is text only: no tools, settings, CLAUDE.md, hooks, MCP servers or saved
    /// session, so the answer depends on the request alone (a `language` in the user's
    /// settings would otherwise change it). Auth still comes from the user's login.
    let startInfo (config: ClaudeCodeConfig) (systemPromptFile: string) : ProcessStartInfo =
        let psi = ProcessStartInfo(config.ClaudePath)

        [ "-p"
          "--output-format"
          "json"
          "--safe-mode"
          "--setting-sources"
          ""
          "--tools"
          ""
          "--strict-mcp-config"
          "--no-session-persistence"
          "--system-prompt-file"
          systemPromptFile ]
        @ (config.Model |> Option.map (fun model -> [ "--model"; model ]) |> Option.defaultValue [])
        @ (if config.Verbose then [ "--verbose" ] else [])
        |> List.iter psi.ArgumentList.Add

        psi.UseShellExecute <- false
        psi.RedirectStandardInput <- true
        psi.RedirectStandardOutput <- true
        psi.RedirectStandardError <- true
        psi.StandardInputEncoding <- UTF8Encoding(false)
        psi.StandardOutputEncoding <- Encoding.UTF8
        psi.StandardErrorEncoding <- Encoding.UTF8
        psi.CreateNoWindow <- true

        // Inherit env (picks up ANTHROPIC_API_KEY, session tokens, etc.)
        psi.EnvironmentVariables.["CLAUDE_CODE_ENTRYPOINT"] <- "tars-llm-service"
        psi

    /// Execute claude CLI and capture output.
    let private executeClaudeProcess
        (config: ClaudeCodeConfig)
        (systemPrompt: string)
        (prompt: string)
        : Task<Result<string, string>> =
        task {
            let systemPromptFile = Path.GetTempFileName()

            try
                try
                    // Without its own system prompt, Claude Code would use its coding-agent one.
                    let systemPrompt =
                        if String.IsNullOrWhiteSpace systemPrompt then "Answer the message." else systemPrompt

                    File.WriteAllText(systemPromptFile, systemPrompt)

                    use proc = new Process()
                    proc.StartInfo <- startInfo config systemPromptFile

                    let stdout = StringBuilder()
                    let stderr = StringBuilder()

                    proc.OutputDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            stdout.AppendLine(e.Data) |> ignore)

                    proc.ErrorDataReceived.Add(fun e ->
                        if not (isNull e.Data) then
                            stderr.AppendLine(e.Data) |> ignore)

                    proc.Start() |> ignore
                    proc.BeginOutputReadLine()
                    proc.BeginErrorReadLine()
                    proc.StandardInput.Write(prompt)
                    proc.StandardInput.Close()

                    let! completed =
                        Task.Run(fun () ->
                            proc.WaitForExit(int config.Timeout.TotalMilliseconds))

                    if not completed then
                        try proc.Kill() with _ -> ()
                        return Error (sprintf "Claude Code process timed out after %.0fs" config.Timeout.TotalSeconds)
                    else
                        // The timed wait can return before the output handlers have run.
                        proc.WaitForExit()

                        if proc.ExitCode <> 0 then
                            let errText = stderr.ToString().Trim()
                            return Error (sprintf "Claude Code exited with code %d: %s" proc.ExitCode errText)
                        else
                            return Ok (stdout.ToString().Trim())
                with ex ->
                    return Error (sprintf "Failed to launch Claude Code: %s" ex.Message)
            finally
                try File.Delete systemPromptFile with _ -> ()
        }

    /// Parse Claude Code JSON output into an LlmResponse.
    let parseResponse (json: string) : LlmResponse =
        try
            // Claude Code --output-format json returns: {"type":"result","result":"...","cost_usd":...}
            let doc = JsonDocument.Parse(json)
            let root = doc.RootElement

            let text =
                let mutable p = JsonElement()
                if root.TryGetProperty("result", &p) then
                    p.GetString()
                else
                    // Fallback: try to find the text content
                    json

            let costUsd =
                let mutable p = JsonElement()
                if root.TryGetProperty("cost_usd", &p) then
                    Some (p.GetDouble())
                else
                    None

            // Estimate tokens from cost (rough: $3/1M input, $15/1M output for Sonnet)
            let estimatedTokens =
                costUsd
                |> Option.map (fun c -> int (c * 1_000_000.0 / 15.0))
                |> Option.defaultValue 0

            { Text = text
              FinishReason = Some "stop"
              Usage =
                  Some
                      { PromptTokens = 0
                        CompletionTokens = estimatedTokens
                        TotalTokens = estimatedTokens }
              Raw = Some json }
        with _ ->
            // If JSON parsing fails, treat the entire output as text
            { Text = json
              FinishReason = Some "stop"
              Usage = None
              Raw = Some json }

    /// Create an ILlmService that delegates to Claude Code CLI.
    type ClaudeCodeLlmService(config: ClaudeCodeConfig) =

        new() = ClaudeCodeLlmService(defaultConfig)

        interface ILlmService with

            member _.CompleteAsync(req: LlmRequest) : Task<LlmResponse> =
                task {
                    let! result = executeClaudeProcess config (systemPromptOf req) (buildPrompt req)

                    match result with
                    | Ok output -> return parseResponse output
                    | Error err ->
                        return
                            { Text = sprintf "[ClaudeCode Error] %s" err
                              FinishReason = Some "error"
                              Usage = None
                              Raw = None }
                }

            member _.EmbedAsync(_text: string) : Task<float32[]> =
                // Claude Code doesn't support embeddings — fall back to empty
                Task.FromResult(Array.empty<float32>)

            member this.CompleteStreamAsync(req: LlmRequest, onChunk: string -> unit) : Task<LlmResponse> =
                task {
                    // Claude Code subprocess doesn't stream — execute and deliver as one chunk
                    let! response = (this :> ILlmService).CompleteAsync(req)
                    onChunk response.Text
                    return response
                }

            // The model Claude Code is told to use. Without one it picks its own
            // default, which this service cannot name, so it reports "claude-code".
            member _.RouteAsync(_req: LlmRequest) : Task<RoutedBackend> =
                Task.FromResult(
                    { Backend = Anthropic(config.Model |> Option.defaultValue "claude-code")
                      Endpoint = Uri "https://api.anthropic.com"
                      ApiKey = None })

    /// Detect if Claude Code is available on the system.
    let isAvailable () : bool =
        try
            let psi = ProcessStartInfo()
            psi.FileName <- "claude"
            psi.Arguments <- "--version"
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true

            use proc = Process.Start(psi)
            proc.WaitForExit(5000) |> ignore
            proc.ExitCode = 0
        with _ -> false

    /// Create a service with optional model override.
    let create (model: string option) : ILlmService =
        let config =
            { defaultConfig with
                Model = model }
        ClaudeCodeLlmService(config) :> ILlmService
