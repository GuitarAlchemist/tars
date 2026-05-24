namespace Tars.Llm

open System
open System.Diagnostics
open System.Text
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm.Routing

/// ILlmService implementation that delegates to Claude Code CLI as a subprocess.
/// Spawns `claude -p <prompt> --output-format json` and parses the response.
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

    /// Build the prompt string from an LlmRequest.
    let buildPrompt (req: LlmRequest) : string =
        let sb = StringBuilder()

        // Add system prompt if present
        match req.SystemPrompt with
        | Some sys ->
            sb.AppendLine(sprintf "<system>%s</system>" sys) |> ignore
            sb.AppendLine() |> ignore
        | None -> ()

        // Add conversation messages
        for msg in req.Messages do
            match msg.Role with
            | Role.System -> sb.AppendLine(sprintf "<system>%s</system>" msg.Content) |> ignore
            | Role.User -> sb.AppendLine(msg.Content) |> ignore
            | Role.Assistant -> sb.AppendLine(sprintf "<assistant>%s</assistant>" msg.Content) |> ignore

        sb.ToString().Trim()

    /// Execute claude CLI and capture output.
    let private executeClaudeProcess
        (config: ClaudeCodeConfig)
        (prompt: string)
        : Task<Result<string, string>> =
        task {
            let args = StringBuilder()
            args.Append("-p ") |> ignore

            // Escape the prompt for shell argument
            let escapedPrompt = prompt.Replace("\"", "\\\"")
            args.Append(sprintf "\"%s\"" escapedPrompt) |> ignore

            args.Append(" --output-format json") |> ignore

            match config.Model with
            | Some model -> args.Append(sprintf " --model %s" model) |> ignore
            | None -> ()

            if config.Verbose then
                args.Append(" --verbose") |> ignore

            let psi = ProcessStartInfo()
            psi.FileName <- config.ClaudePath
            psi.Arguments <- args.ToString()
            psi.UseShellExecute <- false
            psi.RedirectStandardOutput <- true
            psi.RedirectStandardError <- true
            psi.CreateNoWindow <- true

            // Inherit env (picks up ANTHROPIC_API_KEY, session tokens, etc.)
            psi.EnvironmentVariables.["CLAUDE_CODE_ENTRYPOINT"] <- "tars-llm-service"

            try
                use proc = new Process()
                proc.StartInfo <- psi

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

                let! completed =
                    Task.Run(fun () ->
                        proc.WaitForExit(int config.Timeout.TotalMilliseconds))

                if not completed then
                    try proc.Kill() with _ -> ()
                    return Error (sprintf "Claude Code process timed out after %.0fs" config.Timeout.TotalSeconds)
                elif proc.ExitCode <> 0 then
                    let errText = stderr.ToString().Trim()
                    return Error (sprintf "Claude Code exited with code %d: %s" proc.ExitCode errText)
                else
                    return Ok (stdout.ToString().Trim())
            with ex ->
                return Error (sprintf "Failed to launch Claude Code: %s" ex.Message)
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
                    let prompt = buildPrompt req
                    let! result = executeClaudeProcess config prompt

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

            member _.RouteAsync(_req: LlmRequest) : Task<RoutedBackend> =
                Task.FromResult(
                    { Backend = Anthropic "claude-code"
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
