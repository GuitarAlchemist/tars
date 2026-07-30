/// <summary>
/// LLM request routing module.
/// Routes requests to appropriate backends based on model hints and configuration.
/// </summary>
module Tars.Llm.Routing

open System
open Tars.Llm

/// <summary>
/// Configuration for LLM routing.
/// Defines endpoints and default models for each backend.
/// </summary>
type RoutingConfig =
    { OllamaBaseUri: Uri
      VllmBaseUri: Uri
      OpenAIBaseUri: Uri
      GoogleGeminiBaseUri: Uri
      AnthropicBaseUri: Uri
      DockerModelRunnerBaseUri: Uri option
      LlamaCppBaseUri: Uri option
      DefaultOllamaModel: string
      DefaultVllmModel: string
      DefaultOpenAIModel: string
      DefaultGoogleGeminiModel: string
      DefaultAnthropicModel: string
      DefaultDockerModelRunnerModel: string option
      DefaultLlamaCppModel: string option
      DefaultEmbeddingModel: string
      ReasoningModel: string option
      CodingModel: string option
      FastModel: string option
      OllamaKey: string option
      VllmKey: string option
      OpenAIKey: string option
      GoogleGeminiKey: string option
      AnthropicKey: string option
      DockerModelRunnerKey: string option
      LlamaCppKey: string option
      LlamaSharpModelPath: string option
      DefaultContextWindow: int option
      DefaultTemperature: float option
      PreferredProvider: string }

    static member Default =
        { OllamaBaseUri = Uri("http://localhost:11434")
          VllmBaseUri = Uri("http://localhost:8000")
          OpenAIBaseUri = Uri("https://api.openai.com")
          GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com")
          AnthropicBaseUri = Uri("https://api.anthropic.com")
          DockerModelRunnerBaseUri = None
          LlamaCppBaseUri = None
          DefaultOllamaModel = "llama3"
          DefaultVllmModel = "llama3"
          DefaultOpenAIModel = "gpt-4"
          DefaultGoogleGeminiModel = "gemini-1.5-flash"
          DefaultAnthropicModel = "claude-3-5-sonnet-latest"
          DefaultDockerModelRunnerModel = None
          DefaultLlamaCppModel = None
          DefaultEmbeddingModel = "nomic-embed-text"
          ReasoningModel = None
          CodingModel = None
          FastModel = None
          OllamaKey = None
          VllmKey = None
          OpenAIKey = None
          GoogleGeminiKey = None
          AnthropicKey = None
          DockerModelRunnerKey = None
          LlamaCppKey = None
          LlamaSharpModelPath = None
          DefaultContextWindow = None
          DefaultTemperature = None
          PreferredProvider = "Ollama" }

/// <summary>
/// Result of a routing decision.
/// </summary>
type RoutedBackend =
    { Backend: LlmBackend
      Endpoint: Uri
      ApiKey: string option }

/// Provider family inferred from an explicit model name. Isolates the fragile
/// substring matching so routing can decide over a closed, compiler-checked type.
type ModelFamily =
    | OpenAIFamily
    | AnthropicFamily
    | GeminiFamily
    | LocalFamily

module ModelFamily =
    /// Classify an explicit model name into its provider family.
    let classify (model: string) : ModelFamily =
        let has (s: string) = model.Contains(s, StringComparison.OrdinalIgnoreCase)

        if has "gpt" then OpenAIFamily
        elif has "claude" then AnthropicFamily
        elif has "gemini" then GeminiFamily
        else LocalFamily

/// Routing intent inferred from a model hint. The order of classification is
/// significant and mirrors the original sequential matching.
type RoutingHint =
    | CodeHint
    | CheapHint
    | ReasoningHint
    | DockerHint
    | LlamaCppHint
    | FastHint
    | DefaultHint

module RoutingHint =
    /// Classify a (possibly empty) model hint into a routing intent.
    let classify (hint: string) : RoutingHint =
        let has (s: string) = hint.Contains(s, StringComparison.OrdinalIgnoreCase)

        if has "code" then CodeHint
        elif has "cheap" then CheapHint
        elif has "reason" || has "analysis" || has "think" || has "math" || has "complex" || has "step" || has "smart" then
            ReasoningHint
        elif has "docker" then DockerHint
        elif has "llamacpp" || has "perf" || has "gguf" then LlamaCppHint
        elif has "fast" || has "quick" then FastHint
        else DefaultHint

/// <summary>
/// Routes an LLM request to the appropriate backend based on model hints.
/// </summary>
/// <param name="cfg">Routing configuration.</param>
/// <param name="req">The LLM request to route.</param>
/// <returns>The routed backend with endpoint.</returns>
let chooseBackend (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
    let llamaSharpFallback () =
        match cfg.LlamaSharpModelPath with
        | Some modelPath ->
            Some
                { Backend = LlamaSharp modelPath
                  Endpoint = Uri("local://llamasharp")
                  ApiKey = None }
        | None -> None

    let orLlamaSharp fallback =
        match llamaSharpFallback () with
        | Some routed -> routed
        | None -> fallback

    match req.Model with
    | Some model ->
        // If model is explicitly set, classify its provider family then route.
        match ModelFamily.classify model with
        | OpenAIFamily ->
            { Backend = OpenAI model
              Endpoint = cfg.OpenAIBaseUri
              ApiKey = cfg.OpenAIKey }
        | AnthropicFamily ->
            { Backend = Anthropic model
              Endpoint = cfg.AnthropicBaseUri
              ApiKey = cfg.AnthropicKey }
        | GeminiFamily ->
            { Backend = GoogleGemini model
              Endpoint = cfg.GoogleGeminiBaseUri
              ApiKey = cfg.GoogleGeminiKey }
        | LocalFamily ->
            // Check if it matches configured llama.cpp model
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel, cfg.LlamaSharpModelPath with
            | Some llamaUri, Some llamaModel, _ when
                String.Equals(model, llamaModel, StringComparison.OrdinalIgnoreCase)
                || model.Contains("magistral", StringComparison.OrdinalIgnoreCase)
                ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = llamaUri
                  ApiKey = cfg.LlamaCppKey }
            | _ ->
                // Default to Ollama for local models
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }
    | None ->
        let hint = req.ModelHint |> Option.defaultValue ""

        // Helper to route to preferred local backend
        let localRoute model =
            if cfg.PreferredProvider = "Ollama" then
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }
            else if
                // If VLLM is configured and not default, use it, otherwise fallback to Ollama
                cfg.VllmBaseUri.Host <> "localhost" || cfg.VllmBaseUri.Port <> 8000
            then
                { Backend = Vllm model
                  Endpoint = cfg.VllmBaseUri
                  ApiKey = cfg.VllmKey }
            else
                orLlamaSharp
                    { Backend = Ollama model
                      Endpoint = cfg.OllamaBaseUri
                      ApiKey = cfg.OllamaKey }

        match RoutingHint.classify hint with
        | CodeHint -> localRoute (cfg.CodingModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | CheapHint -> localRoute cfg.DefaultOllamaModel

        | ReasoningHint -> localRoute (cfg.ReasoningModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | DockerHint ->
            match cfg.DockerModelRunnerBaseUri, cfg.DefaultDockerModelRunnerModel with
            | Some uri, Some model ->
                { Backend = DockerModelRunner model
                  Endpoint = uri
                  ApiKey = cfg.DockerModelRunnerKey }
            | _ -> localRoute cfg.DefaultOllamaModel

        | LlamaCppHint ->
            match cfg.LlamaCppBaseUri, cfg.DefaultLlamaCppModel with
            | Some uri, Some model ->
                { Backend = LlamaCpp(model, Some LlamaCppConfig.Default)
                  Endpoint = uri
                  ApiKey = cfg.LlamaCppKey }
            | _ -> localRoute cfg.DefaultOllamaModel

        | FastHint -> localRoute (cfg.FastModel |> Option.defaultValue cfg.DefaultOllamaModel)

        | DefaultHint -> localRoute cfg.DefaultOllamaModel

/// What decode-time constraint a request actually needs. Closed and
/// compiler-checked so `supports` cannot silently acquire a missing case when a
/// backend is added.
type ConstraintNeed =
    /// An EBNF/CFG grammar must be enforced by the decoder.
    | NeedsGrammar
    /// A regex must be enforced. Deliberately NOT folded in with NeedsGrammar:
    /// vLLM enforces regex, llama.cpp does not, so one bucket would have claimed
    /// support llama.cpp lacks and suppressed the downgrade warning.
    | NeedsRegex
    /// A JSON schema is required; most OpenAI-wire backends can enforce this.
    | NeedsJsonSchema
    | NoNeed

/// A constraint the chosen backend cannot enforce. This travels in the routing
/// *result* rather than being logged here: `chooseBackend` is pure, and the
/// `ILogger` lives at the service boundary.
type ConstraintDowngrade = { RequestedGrammar: string; Backend: string }

/// `chooseBackend`'s result plus whatever constraint had to be given up to use it.
/// A separate wrapper on purpose — `RoutedBackend` has ~34 construction sites
/// across src and tests, and is the return type of `ILlmService.RouteAsync`, so
/// adding a field there would be a breaking change for no benefit.
type ChosenBackend =
    { Routed: RoutedBackend
      Downgrade: ConstraintDowngrade option }

module ConstraintNeed =

    /// Classify what a request needs the decoder to enforce.
    let ofRequest (req: LlmRequest) : ConstraintNeed =
        match req.ResponseFormat with
        | Some(ResponseFormat.Constrained(Grammar.JsonSchema _)) -> NeedsJsonSchema
        | Some(ResponseFormat.Constrained(Grammar.Ebnf _)) -> NeedsGrammar
        | Some(ResponseFormat.Constrained(Grammar.Regex _)) -> NeedsRegex
        | _ -> NoNeed

    /// Whether a backend can actually enforce the need — by capability, not by
    /// provider name. Every case is spelled out so adding an `LlmBackend` case
    /// fails the build here rather than silently defaulting to "unsupported".
    let supports (backend: LlmBackend) (need: ConstraintNeed) : bool =
        match need with
        | NoNeed -> true
        | NeedsJsonSchema ->
            match backend with
            // Ollama compiles a JSON schema to GBNF server-side; the OpenAI wire
            // protocol carries it as response_format.json_schema.
            | Vllm _
            | LlamaCpp _
            | Ollama _
            | OpenAI _
            | DockerModelRunner _ -> true
            // AnthropicClient only appends a prompt hint — it enforces nothing.
            | Anthropic _
            // GoogleGeminiClient DOES send the schema, as generationConfig.responseSchema.
            // But that field is an OpenAPI-subset Schema with no `additionalProperties`
            // and a scalar `type`, and every schema we author carries
            // additionalProperties:false (strict mode requires it). Gemini rejects
            // unknown fields, so our schemas 400 rather than degrade. Reported as a
            // downgrade because we cannot enforce them there — not because Gemini
            // lacks the capability in general.
            | GoogleGemini _
            // LlamaSharpService never reads ResponseFormat at all.
            | LlamaSharp _ -> false
        | NeedsGrammar ->
            match backend with
            // vLLM via structured_outputs.grammar; llama.cpp via top-level GBNF.
            | Vllm _
            | LlamaCpp _ -> true
            // Ollama has no raw GBNF API — schema only.
            | Ollama _
            | OpenAI _
            | DockerModelRunner _
            | Anthropic _
            | GoogleGemini _
            | LlamaSharp _ -> false
        | NeedsRegex ->
            match backend with
            // vLLM only. LlamaCppClient maps Regex to nothing, so claiming support
            // here would drop the pattern *and* silence the warning.
            | Vllm _ -> true
            | LlamaCpp _
            | Ollama _
            | OpenAI _
            | DockerModelRunner _
            | Anthropic _
            | GoogleGemini _
            | LlamaSharp _ -> false

/// The grammar kind a request asked for, named as it appears in logs and tests.
let private requestedGrammarName (req: LlmRequest) =
    match req.ResponseFormat with
    | Some(ResponseFormat.Constrained(Grammar.Ebnf _)) -> "ebnf"
    | Some(ResponseFormat.Constrained(Grammar.Regex _)) -> "regex"
    | Some(ResponseFormat.Constrained(Grammar.JsonSchema _)) -> "json_schema"
    | _ -> "none"

let private backendName (backend: LlmBackend) =
    match backend with
    | Ollama _ -> "Ollama"
    | Vllm _ -> "Vllm"
    | OpenAI _ -> "OpenAI"
    | GoogleGemini _ -> "GoogleGemini"
    | Anthropic _ -> "Anthropic"
    | DockerModelRunner _ -> "DockerModelRunner"
    | LlamaCpp _ -> "LlamaCpp"
    | LlamaSharp _ -> "LlamaSharp"

/// Route, and report whether the chosen backend can enforce the requested
/// constraint. Still pure — callers that do not care about constraints keep using
/// `chooseBackend` unchanged.
let chooseBackendWithConstraints (cfg: RoutingConfig) (req: LlmRequest) : ChosenBackend =
    let routed = chooseBackend cfg req
    let need = ConstraintNeed.ofRequest req

    let downgrade =
        if ConstraintNeed.supports routed.Backend need then
            None
        else
            Some
                { RequestedGrammar = requestedGrammarName req
                  Backend = backendName routed.Backend }

    { Routed = routed; Downgrade = downgrade }

/// Where constraint downgrades get reported. Routing stays pure, so the warning
/// is emitted here at the service boundary instead.
///
/// A swappable sink rather than an ILogger because Tars.Llm has no logging
/// dependency at all and LlmServiceConfig carries only Routing — introducing DI
/// for one warning would be a larger change than the feature. Defaults to stderr.
///
/// Deliberately logged on *every* downgraded request, not sampled or rate-limited:
/// silent degradation is the defect this whole slice exists to remove, and a
/// warning that fires once at startup is indistinguishable from silence.
module ConstraintDowngradeLog =

    let private defaultSink (msg: string) = eprintfn "%s" msg

    /// AsyncLocal, not a plain mutable: a process-global sink is a test-isolation
    /// hazard. xUnit runs distinct test collections in parallel, so one test
    /// redirecting the sink to its own buffer would capture another's warnings
    /// and drop its own. AsyncLocal scopes the override to the execution context
    /// that set it and flows into that context's continuations, so a redirect in
    /// one test is invisible to every other.
    let private scopedSink = new System.Threading.AsyncLocal<(string -> unit) option>()

    /// Redirect warnings (tests capture; hosts can forward to their logger).
    /// Scoped to the calling execution context, not the process.
    let setSink (f: string -> unit) = scopedSink.Value <- Some f

    let resetSink () = scopedSink.Value <- None

    let private sink (msg: string) =
        (scopedSink.Value |> Option.defaultValue defaultSink) msg

    let format (d: ConstraintDowngrade) =
        $"CONSTRAINT DOWNGRADE: {d.RequestedGrammar} grammar discarded — backend {d.Backend} cannot enforce it; falling back to JSON mode"

    let warn (d: ConstraintDowngrade) = sink (format d)

    /// Route, reporting any downgrade. The one call the service paths use.
    let routeAndWarn (cfg: RoutingConfig) (req: LlmRequest) : RoutedBackend =
        let chosen = chooseBackendWithConstraints cfg req
        chosen.Downgrade |> Option.iter warn
        chosen.Routed

module RoutingConfig =
    /// <summary>
    /// Creates a RoutingConfig from a TarsConfig.
    /// maps standard TARS configuration to LLM routing settings.
    /// </summary>
    let fromTarsConfig (tarsCfg: Tars.Core.TarsConfig) : RoutingConfig =
        let baseUri =
            tarsCfg.Llm.BaseUrl |> Option.defaultValue "http://localhost:11434" |> Uri

        let llamaCppUri = tarsCfg.Llm.LlamaCppUrl |> Option.map Uri

        { OllamaBaseUri = baseUri
          VllmBaseUri = Uri("http://localhost:8000/")
          OpenAIBaseUri = Uri("https://api.openai.com/")
          GoogleGeminiBaseUri = Uri("https://generativelanguage.googleapis.com/")
          AnthropicBaseUri = Uri("https://api.anthropic.com/")
          DockerModelRunnerBaseUri = None
          LlamaCppBaseUri = llamaCppUri

          DefaultOllamaModel = tarsCfg.Llm.Model
          DefaultVllmModel = "llama3"
          DefaultOpenAIModel = "gpt-4o"
          DefaultGoogleGeminiModel = "gemini-1.5-flash"
          DefaultAnthropicModel = "claude-3-5-sonnet-latest"
          DefaultDockerModelRunnerModel = None
          DefaultLlamaCppModel = if llamaCppUri.IsSome then Some tarsCfg.Llm.Model else None
          DefaultEmbeddingModel = tarsCfg.Llm.EmbeddingModel

          ReasoningModel = tarsCfg.Llm.ReasoningModel
          CodingModel = tarsCfg.Llm.CodingModel
          FastModel = tarsCfg.Llm.FastModel

          OllamaKey = tarsCfg.Llm.ApiKey
          VllmKey = tarsCfg.Llm.ApiKey
          OpenAIKey = tarsCfg.Llm.ApiKey
          GoogleGeminiKey = tarsCfg.Llm.ApiKey
          AnthropicKey = tarsCfg.Llm.ApiKey
          DockerModelRunnerKey = tarsCfg.Llm.ApiKey
          LlamaCppKey = tarsCfg.Llm.ApiKey
          LlamaSharpModelPath = tarsCfg.Llm.LlamaSharpModelPath
          DefaultContextWindow = Some tarsCfg.Llm.ContextWindow
          DefaultTemperature = Some tarsCfg.Llm.Temperature
          PreferredProvider = tarsCfg.Llm.Provider }
