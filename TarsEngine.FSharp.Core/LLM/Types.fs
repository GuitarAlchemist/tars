namespace TarsEngine.FSharp.Core.LLM

open System
open System.Collections.Generic
open System.Threading.Tasks

/// LLM message role
type LLMRole =
    | System
    | User  
    | Assistant

/// LLM message
type LLMMessage = {
    Role: LLMRole
    Content: string
    Timestamp: DateTime
}

/// LLM conversation context
type LLMContext = {
    Messages: LLMMessage list
    SystemPrompt: string option
    Temperature: float
    MaxTokens: int
    Model: string
}

/// LLM response
type LLMResponse = {
    Content: string
    TokensUsed: int
    Model: string
    FinishReason: string
    ResponseTime: TimeSpan
}

/// Codestral-specific request
type CodestralRequest = {
    Context: LLMContext
    CodeContext: string option
    Language: string option
    Task: string
}

/// Codestral-specific response
type CodestralResponse = {
    GeneratedCode: string
    Explanation: string
    Suggestions: string list
    Confidence: float
    Response: LLMResponse
}

/// LLM client interface
type ILLMClient =
    abstract member SendMessageAsync: context: LLMContext * message: string -> Task<LLMResponse>
    abstract member GenerateCodeAsync: request: CodestralRequest -> Task<CodestralResponse>
    abstract member AnalyzeCodeAsync: code: string * language: string -> Task<LLMResponse>
    abstract member CreateContextAsync: systemPrompt: string option -> LLMContext

/// Autonomous reasoning service interface
type IAutonomousReasoningService =
    abstract member ReasonAboutTaskAsync: task: string * context: Map<string, obj> -> Task<string>
    abstract member GenerateMetascriptAsync: objective: string * context: Map<string, obj> -> Task<string>
    abstract member AnalyzeAndImproveAsync: code: string * language: string -> Task<string>
    abstract member MakeDecisionAsync: options: string list * criteria: string -> Task<string>

