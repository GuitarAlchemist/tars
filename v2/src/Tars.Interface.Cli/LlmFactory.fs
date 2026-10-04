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

    /// Routing that sends every request to `model`, whatever its hint (reasoning, coding, fast).
    /// A configured LlamaSharp model would take every local route first, and Docker Model Runner
    /// and llama.cpp would take the "docker", "llamacpp", "perf" and "gguf" hints, so they are dropped.
    let pinnedTo (model: string) (cfg: RoutingConfig) : RoutingConfig =
        { cfg with
            DefaultOllamaModel = model
            DefaultVllmModel = model
            ReasoningModel = Some model
            CodingModel = Some model
            FastModel = Some model
            LlamaSharpModelPath = None
            DockerModelRunnerBaseUri = None
            DefaultDockerModelRunnerModel = None
            LlamaCppBaseUri = None
            DefaultLlamaCppModel = None }

    /// Create an LLM service that answers every request with `model`, whatever its hint.
    let createPinnedTo (_logger: ILogger) (model: string) : ILlmService =
        let _, routingCfg = loadConfig ()
        let serviceConfig = { LlmServiceConfig.Routing = pinnedTo model routingCfg }
        DefaultLlmService(sharedClient, serviceConfig) :> ILlmService

    /// The API model `openai:<model>`, `gemini:<model>` or `anthropic:<model>` names, routed to its
    /// provider with that provider's key. `anthropic:` is the paid API; `claude:` is Claude Code.
    /// The provider is named, not guessed: Ollama's `gpt-oss:20b` stays local.
    let apiRoute (cfg: RoutingConfig) (model: string) : RoutedBackend option =
        match model.Split(':', 2, StringSplitOptions.None) with
        | [| "openai"; name |] when name <> "" ->
            Some
                { Backend = OpenAI name
                  Endpoint = cfg.OpenAIBaseUri
                  ApiKey = cfg.OpenAIKey }
        | [| "gemini"; name |] when name <> "" ->
            Some
                { Backend = GoogleGemini name
                  Endpoint = cfg.GoogleGeminiBaseUri
                  ApiKey = cfg.GoogleGeminiKey }
        | [| "anthropic"; name |] when name <> "" ->
            Some
                { Backend = Anthropic name
                  Endpoint = cfg.AnthropicBaseUri
                  ApiKey = cfg.AnthropicKey }
        | _ -> None

    /// An LLM service that sends every completion to `route`, whatever the request's model or hint.
    /// Embeddings come from the configured backend, as with `create`.
    let onRoute (cfg: RoutingConfig) (route: RoutedBackend) : ILlmService =
        let backend = Backends.resolve { LlmServiceConfig.Routing = cfg } sharedClient route

        { new ILlmService with
            member _.CompleteAsync req = backend.Complete(enrichRequest cfg req)
            member _.CompleteStreamAsync(req, onToken) = backend.Stream(enrichRequest cfg req, onToken)
            member _.EmbedAsync text = Embedder.embed sharedClient cfg text
            member _.RouteAsync _ = Threading.Tasks.Task.FromResult route }

    /// The service for `openai:<model>`, `gemini:<model>` or `anthropic:<model>`, billed to that
    /// provider's API key (CredentialVault, the environment or the config). None for other names.
    let createOnApi (_logger: ILogger) (model: string) : ILlmService option =
        let _, routingCfg = loadConfig ()

        apiRoute routingCfg model
        |> Option.map (fun route ->
            if route.ApiKey |> Option.forall String.IsNullOrWhiteSpace then
                let secret =
                    match route.Backend with
                    | OpenAI _ -> "OPENAI_API_KEY"
                    | GoogleGemini _ -> "GOOGLE_API_KEY"
                    | _ -> "ANTHROPIC_API_KEY"

                failwith $"{model} needs an API key: set {secret}."

            onRoute routingCfg route)

    /// The model `--model claude:<model>` asks Claude Code for (`claude:sonnet` -> `sonnet`).
    let claudeCodeModel (model: string) : string option =
        let prefix = "claude:"

        if model.StartsWith(prefix, StringComparison.Ordinal) && model.Length > prefix.Length then
            Some(model.Substring prefix.Length)
        else
            None

    /// `llm` with `embedder`'s embeddings, for a service that has none: Claude Code returns
    /// an empty vector, which the vector stores cannot compare with the others. A caller's
    /// cancellation still reaches whichever of the two can take it.
    let withEmbeddings (embedder: ILlmService) (llm: ILlmService) : ILlmService =
        { new ILlmService with
            member _.CompleteAsync req = llm.CompleteAsync req
            member _.EmbedAsync text = embedder.EmbedAsync text
            member _.CompleteStreamAsync(req, onChunk) = llm.CompleteStreamAsync(req, onChunk)
            member _.RouteAsync req = llm.RouteAsync req
          interface ICancellableLlmService with
            member _.CompleteAsync(req, token) =
                match llm with
                | :? ICancellableLlmService as cancellable -> cancellable.CompleteAsync(req, token)
                | _ -> llm.CompleteAsync req

            member _.EmbedAsync(text, token) =
                match embedder with
                | :? ICancellableLlmService as cancellable -> cancellable.EmbedAsync(text, token)
                | _ -> embedder.EmbedAsync text

            member _.CompleteStreamAsync(req, onChunk, token) =
                match llm with
                | :? ICancellableLlmService as cancellable -> cancellable.CompleteStreamAsync(req, onChunk, token)
                | _ -> llm.CompleteStreamAsync(req, onChunk)

            member _.RouteAsync(req, token) =
                match llm with
                | :? ICancellableLlmService as cancellable -> cancellable.RouteAsync(req, token)
                | _ -> llm.RouteAsync req }

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
                Prompt.ask "ping"
                |> Prompt.withMaxTokens 1
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
