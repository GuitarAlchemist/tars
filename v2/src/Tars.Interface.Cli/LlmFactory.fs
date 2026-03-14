namespace Tars.Interface.Cli

open System
open Serilog
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService

module LlmFactory =
    /// Create the default LLM service (Ollama/vLLM/OpenAI routing).
    let create (logger: ILogger) =
        let config = ConfigurationLoader.load ()
        let routingCfg = RoutingConfig.fromTarsConfig config

        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        let client = new System.Net.Http.HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(10.0)
        DefaultLlmService(client, serviceConfig) :> ILlmService

    /// Create a Claude Code subprocess LLM service.
    /// Uses the user's authenticated Claude Code session — no API key needed.
    let createClaudeCode (model: string option) =
        ClaudeCodeService.create model

    /// Create an LLM service with auto-detection:
    /// prefers configured backends, falls back to Claude Code if available.
    let createWithFallback (logger: ILogger) =
        try
            let primary = create logger
            // Quick check: can we reach the configured backend?
            let probe =
                { LlmRequest.Default with
                    Messages = [ { Role = Role.User; Content = "ping" } ]
                    MaxTokens = Some 1 }
            let result = primary.CompleteAsync(probe) |> fun t -> t.Wait(TimeSpan.FromSeconds(5.0))
            if result then primary
            else
                logger.Warning("Primary LLM backend unreachable, trying Claude Code...")
                if ClaudeCodeService.isAvailable () then
                    logger.Information("Using Claude Code as LLM backend")
                    createClaudeCode None
                else
                    logger.Warning("Claude Code not available either, using primary (may fail)")
                    primary
        with _ ->
            if ClaudeCodeService.isAvailable () then
                createClaudeCode None
            else
                create logger
