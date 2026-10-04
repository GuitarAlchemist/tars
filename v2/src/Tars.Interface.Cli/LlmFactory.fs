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
    /// provider. `anthropic:` is the paid API; `claude:` is Claude Code. The provider is named, not
    /// guessed: Ollama's `gpt-oss:20b` stays local. The key is that provider's own `secret`
    /// (OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY), never `cfg`'s: `RoutingConfig.fromTarsConfig`
    /// copies `Llm:ApiKey`, which is OPENAI_API_KEY, into every provider's slot.
    let apiRoute (secret: string -> string option) (cfg: RoutingConfig) (model: string) : RoutedBackend option =
        let route backend endpoint secretName =
            Some
                { Backend = backend
                  Endpoint = endpoint
                  ApiKey = secret secretName }

        match model.Split(':', 2, StringSplitOptions.None) with
        | [| "openai"; name |] when name <> "" -> route (OpenAI name) cfg.OpenAIBaseUri "OPENAI_API_KEY"
        | [| "gemini"; name |] when name <> "" -> route (GoogleGemini name) cfg.GoogleGeminiBaseUri "GOOGLE_API_KEY"
        | [| "anthropic"; name |] when name <> "" -> route (Anthropic name) cfg.AnthropicBaseUri "ANTHROPIC_API_KEY"
        | _ -> None

    /// Why TARS cannot use the API model `model` names, if it cannot. OpenAI's reasoning models
    /// (o1, o3, o4, gpt-5) reject the `temperature` and `max_tokens` its client sends, and would
    /// spend the judge's 400-token limit on hidden reasoning.
    let unsupportedApiModel (model: string) : string option =
        match model.Split(':', 2, StringSplitOptions.None) with
        | [| "openai"; name |] when
            System.Text.RegularExpressions.Regex.IsMatch(
                name,
                @"^(o\d|gpt-5)",
                System.Text.RegularExpressions.RegexOptions.IgnoreCase
            )
            ->
            Some
                $"{model} is an OpenAI reasoning model, which rejects the temperature and max_tokens TARS sends: use a chat model such as openai:gpt-4.1."
        | _ -> None

    /// An LLM service that sends every completion to `route`, whatever the request's model or hint.
    /// A constraint the route cannot enforce is reported, as `DefaultLlmService` does. Embeddings
    /// come from the configured backend, as with `create`.
    let onRoute (cfg: RoutingConfig) (route: RoutedBackend) : ILlmService =
        let backend = Backends.resolve { LlmServiceConfig.Routing = cfg } sharedClient route

        let prepared req =
            downgradeOf route.Backend req |> Option.iter ConstraintDowngradeLog.warn
            enrichRequest cfg req

        { new ILlmService with
            member _.CompleteAsync req = backend.Complete(prepared req)
            member _.CompleteStreamAsync(req, onToken) = backend.Stream(prepared req, onToken)
            member _.EmbedAsync text = Embedder.embed sharedClient cfg text
            member _.RouteAsync _ = Threading.Tasks.Task.FromResult route }

    /// The service for `openai:<model>`, `gemini:<model>` or `anthropic:<model>`, billed to that
    /// provider's API key (the environment or secrets.json, through CredentialVault). None for
    /// other names.
    let createOnApi (_logger: ILogger) (model: string) : ILlmService option =
        let _, routingCfg = loadConfig ()

        let secret name =
            match CredentialVault.getSecret name with
            | Ok key -> Some key
            | _ -> None

        apiRoute secret routingCfg model
        |> Option.map (fun route ->
            unsupportedApiModel model |> Option.iter failwith

            if route.ApiKey |> Option.forall String.IsNullOrWhiteSpace then
                let secret =
                    match route.Backend with
                    | OpenAI _ -> "OPENAI_API_KEY"
                    | GoogleGemini _ -> "GOOGLE_API_KEY"
                    | _ -> "ANTHROPIC_API_KEY"

                failwith $"{model} needs an API key: set {secret}."

            onRoute routingCfg route)

    /// A price "IN/OUT": USD per million input tokens and per million output tokens (`2.5/10`).
    let parsePrice (text: string) : (decimal * decimal) option =
        let parse (s: string) =
            match
                Decimal.TryParse(
                    s,
                    Globalization.NumberStyles.AllowDecimalPoint,
                    Globalization.CultureInfo.InvariantCulture
                )
            with
            | true, value -> Some value
            | _ -> None

        match text.Split('/') with
        | [| input; output |] -> Option.map2 (fun i o -> i, o) (parse input) (parse output)
        | _ -> None

    /// `llm`, billed at `price` (USD per million input and output tokens), within `budget`'s money,
    /// which it never exceeds. A call is sent only when its worst case fits in the money left: its
    /// input at one token per UTF-8 byte of everything billed (system prompt, messages, schema or
    /// grammar, tool definitions; plus 8 per part and 64 per call) and its `MaxTokens` of output
    /// (4096 when it sets none). That worst case is reserved before the call, then settled at the
    /// response's usage, or kept whole when the response has none.
    let charged (budget: BudgetGovernor) (inputPrice: decimal, outputPrice: decimal) (llm: ILlmService) : ILlmService =
        let usd (input: int) (output: int) =
            (decimal input * inputPrice + decimal output * outputPrice) / 1_000_000m * 1m<usd>

        let money amount = { Cost.Zero with Money = amount }

        let send (req: LlmRequest) (call: LlmRequest -> Threading.Tasks.Task<LlmResponse>) =
            task {
                let req =
                    { req with
                        MaxTokens = Some(req.MaxTokens |> Option.defaultValue 4096) }

                let prompt =
                    Option.toList req.SystemPrompt @ (req.Messages |> List.map (fun m -> m.Content))

                let grammar =
                    match req.ResponseFormat with
                    | Some(ResponseFormat.Constrained(Grammar.JsonSchema text | Grammar.Ebnf text | Grammar.Regex text)) ->
                        [ text ]
                    | _ -> []

                let tools =
                    req.Tools
                    |> List.map (fun tool ->
                        try
                            System.Text.Json.JsonSerializer.Serialize tool
                        with _ ->
                            string tool)

                let worstInput =
                    64
                    + (prompt @ grammar @ tools
                       |> List.sumBy (fun text -> System.Text.Encoding.UTF8.GetByteCount text + 8))

                let reserved = usd worstInput req.MaxTokens.Value

                match budget.TryConsume(money reserved) with
                | Ok() ->
                    let! response =
                        task {
                            try
                                return! call req
                            with ex ->
                                budget.Consume(money -reserved) |> ignore
                                return raise ex
                        }

                    let cost =
                        match response.Usage with
                        | Some usage -> usd usage.PromptTokens usage.CompletionTokens
                        | None -> reserved

                    budget.Consume(money (cost - reserved)) |> ignore
                    return response
                | _ ->
                    let left = budget.Remaining.MaxMoney |> Option.defaultValue 0m<usd>

                    return
                        failwith
                            $"The USD budget is spent: this call may cost up to {reserved} USD, and {left} USD is left."
            }

        { new ILlmService with
            member _.CompleteAsync req = send req llm.CompleteAsync
            member _.CompleteStreamAsync(req, onToken) = send req (fun req -> llm.CompleteStreamAsync(req, onToken))
            member _.EmbedAsync text = llm.EmbedAsync text
            member _.RouteAsync req = llm.RouteAsync req }

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
