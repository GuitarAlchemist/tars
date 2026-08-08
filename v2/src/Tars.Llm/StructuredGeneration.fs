namespace Tars.Llm

open System
open System.Threading.Tasks
open Tars.Core

/// <summary>
/// Profiles representing different backend technologies for structured generation.
/// Maps to the core capabilities of major inference engines.
/// </summary>
type StructuredOutputProfile =
    /// <summary>
    /// Strict JSON schema enforcement supported by OpenAI and Azure OpenAI.
    /// Uses response_format: { type: "json_schema", json_schema: { strict: true, ... } }
    /// </summary>
    | OpenAiStructuredOutputs of schema: string

    /// <summary>
    /// Standard JSON mode fallback supported by Ollama, Anthropic, Gemini, etc.
    /// Ensures output is syntactically valid JSON, but keys are guided via prompt instructions.
    /// </summary>
    | OllamaJsonFallback of schemaHint: string

    /// <summary>
    /// GBNF (GGML Backus-Naur Form) context-free grammar constraints used by llama.cpp.
    /// Enforces character-level decoding restrictions in-process.
    /// </summary>
    | LlamaCppGrammar of gbnf: string

    /// <summary>
    /// Guided decoding using Outlines, XGrammar, or SGLang on vLLM backends.
    /// Transmitted via guided_decoding parameters in the request body.
    /// </summary>
    | VllmGuidedDecoding of schemaOrGrammar: string * backend: string

    /// <summary>
    /// Graceful fallback for models without structured generation capabilities.
    /// Renders the target schema into system instructions and performs post-generation validation.
    /// </summary>
    | PromptOnlyFallback of schemaHint: string

/// <summary>
/// Configuration for the structured generation engine.
/// </summary>
type StructuredGenerationConfig =
    {
        Profile: StructuredOutputProfile
        MaxRetriesOnFailure: int
        EnableAutoRepair: bool
    }

    static member Default =
        { Profile = PromptOnlyFallback ""
          MaxRetriesOnFailure = 2
          EnableAutoRepair = true }

/// <summary>
/// Output wrapper containing the validated object or fallback text.
/// Enforces qualified access to prevent union case shadowing with ExecutionOutcome.Success.
/// </summary>
[<RequireQualifiedAccess>]
type StructuredResult<'T> =
    | Success of data: 'T * rawText: string
    | ValidationFailed of errors: string list * rawText: string
    | ParsingFailed of ex: Exception * rawText: string

/// <summary>
/// Defines the architectural boundary for providers of structured generation/constrained decoding.
/// </summary>
type IStructuredGenerationProvider =
    /// <summary>
    /// Determines whether the provider supports a given structured generation profile.
    /// </summary>
    abstract member SupportsProfile: profile: StructuredOutputProfile -> bool

    /// <summary>
    /// Modifies a generic LLM request to attach the specific parameters required by the profile.
    /// </summary>
    abstract member EnforceConstraints: req: LlmRequest * profile: StructuredOutputProfile -> LlmRequest

    /// <summary>
    /// Validates and parses the LLM response text into a typed contract or structured object.
    /// </summary>
    abstract member ParseResponse<'T> : response: LlmResponse * schema: string -> Result<'T, string>
