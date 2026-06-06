namespace Tars.Interface.Cli

open System
open System.Net.Http
open Serilog
open Tars.Core
open Tars.Llm
open Tars.Llm.Routing
open Tars.Llm.LlmService
open Tars.Security

/// Centralized LLM service factory.
/// All CLI commands should use this instead of creating DefaultLlmService directly.
module LlmFactory =

    /// Shared HttpClient singleton — avoids socket exhaustion from repeated new HttpClient().
    let private sharedClient =
        let client = new HttpClient()
        client.Timeout <- TimeSpan.FromMinutes(10.0)
        client

    /// Enrich a RoutingConfig with per-provider API keys from CredentialVault.
    let private enrichKeys (cfg: RoutingConfig) =
        let resolve secretName fallback =
            match CredentialVault.getSecret secretName with
            | Ok key -> Some key
            | _ -> fallback
        { cfg with
            OpenAIKey = resolve "OPENAI_API_KEY" cfg.OpenAIKey
            GoogleGeminiKey = resolve "GOOGLE_API_KEY" cfg.GoogleGeminiKey
            AnthropicKey = resolve "ANTHROPIC_API_KEY" cfg.AnthropicKey }

    /// Load config and build a RoutingConfig with enriched keys.
    let loadConfig () : TarsConfig * RoutingConfig =
        let config = ConfigurationLoader.load ()
        let routingCfg = RoutingConfig.fromTarsConfig config |> enrichKeys
        config, routingCfg

    /// Create the default LLM service from config.
    let create (_logger: ILogger) : ILlmService =
        let _, routingCfg = loadConfig ()
        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        DefaultLlmService(sharedClient, serviceConfig) :> ILlmService

    /// Create LLM service and return the TarsConfig alongside it.
    let createWithConfig (_logger: ILogger) : ILlmService * TarsConfig =
        let config, routingCfg = loadConfig ()
        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        let llm = DefaultLlmService(sharedClient, serviceConfig) :> ILlmService
        llm, config

    /// Create LLM service with a model override (replaces DefaultOllamaModel).
    let createWithModel (_logger: ILogger) (model: string) : ILlmService =
        let _, routingCfg = loadConfig ()
        let routingCfg = { routingCfg with DefaultOllamaModel = model; DefaultVllmModel = model }
        let serviceConfig = { LlmServiceConfig.Routing = routingCfg }
        DefaultLlmService(sharedClient, serviceConfig) :> ILlmService

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
