namespace Tars.Cortex

open System.Threading.Tasks
open Tars.Core
open Tars.Llm

/// <summary>
/// Represents the context flowing through the Pre-LLM pipeline.
/// </summary>
type PreLlmContext =
    {
        /// The original user input or system prompt
        RawInput: string

        /// The current processed prompt (rewritten/summarized)
        CurrentPrompt: string

        /// Detected intent of the prompt
        Intent: AgentDomain option

        /// Safety status (Allowed or Blocked)
        IsSafe: bool

        /// Reason for blocking if unsafe
        BlockReason: string option

        /// Metadata/Tags added by stages
        Metadata: Map<string, string>
    }

    static member Create(input: string) =
        { RawInput = input
          CurrentPrompt = input
          Intent = None
          IsSafe = true
          BlockReason = None
          Metadata = Map.empty }

/// <summary>
/// Interface for a stage in the Pre-LLM pipeline.
/// </summary>
type IPreLlmStage =
    abstract member Name: string
    abstract member ExecuteAsync: PreLlmContext -> Task<PreLlmContext>

/// <summary>
/// Gate a prompt against explicit policies.
/// </summary>
type PolicyGateStage(requiredPolicies: string list) =
    let policies = requiredPolicies |> List.distinct

    interface IPreLlmStage with
        member _.Name = "PolicyGate"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else if policies.IsEmpty then
                    return ctx
                else
                    let input: PolicyEngine.PolicyInput =
                        { Text = ctx.CurrentPrompt
                          Metadata = ctx.Metadata }

                    let outcomes = PolicyEngine.evaluateDefault policies input

                    if PolicyEngine.anyFailed outcomes then
                        let reasons =
                            outcomes
                            |> List.filter (fun o -> not o.Passed)
                            |> List.collect (fun o -> o.Messages)
                            |> List.distinct

                        let reason =
                            match reasons with
                            | [] -> "Policy gate failed."
                            | xs -> String.concat "; " xs

                        return
                            { ctx with
                                IsSafe = false
                                BlockReason = Some reason }
                    else
                        return ctx
            }

/// <summary>
/// Safety filter mapped to explicit destructive-command policy.
/// </summary>
type SafetyFilterStage() =
    let inner = PolicyGateStage([ "no_destructive_commands" ]) :> IPreLlmStage

    interface IPreLlmStage with
        member _.Name = "SafetyFilter"
        member _.ExecuteAsync(ctx) = inner.ExecuteAsync(ctx)

type IIntentClassifier =
    abstract member ClassifyAsync: string -> Task<AgentDomain option>

type NoopIntentClassifier() =
    interface IIntentClassifier with
        member _.ClassifyAsync(_) = Task.FromResult None

type LlmIntentClassifier(llm: ILlmService) =
    let semanticClassifier: SemanticClassifier<AgentDomain> =
        SemanticClassifierFactory.createDomainClassifier llm

    interface IIntentClassifier with
        member _.ClassifyAsync(input) =
            task {
                // 1. Try fast semantic classification first (Vector Similarity)
                let! semanticResult =
                    task {
                        try
                            return! semanticClassifier.ClassifyAsync(input, 0.8) |> Async.StartAsTask
                        with _ ->
                            return None
                    }

                match semanticResult with
                | Some(domain, score) -> return Some domain
                | None ->
                    // 2. Fallback to LLM-based classification if ambiguous
                    let request: LlmRequest =
                        { ModelHint = Some "fast"
                          Model = None
                          SystemPrompt =
                            Some
                                "Classify the intent as one of: coding, planning, reasoning, chat. Respond with JSON: {\"intent\":\"...\"}."
                          MaxTokens = Some 80
                          Temperature = Some 0.0
                          Stop = []
                          Messages = [ { Role = Role.User; Content = input } ]
                          Tools = []
                          ToolChoice = None
                          ResponseFormat = Some ResponseFormat.Json
                          Stream = false
                          JsonMode = true
                          Seed = None

                          ContextWindow = None }

                    let! response =
                        task {
                            try
                                return! llm.CompleteAsync(request)
                            with _ ->
                                return
                                    { Text = ""
                                      FinishReason = Some "error"
                                      Usage = None
                                      Raw = None }
                        }

                    match JsonParsing.tryParseElement response.Text with
                    | Result.Ok elem ->
                        let mutable intentElem = Unchecked.defaultof<System.Text.Json.JsonElement>

                        if
                            elem.ValueKind = System.Text.Json.JsonValueKind.Object
                            && elem.TryGetProperty("intent", &intentElem)
                            && intentElem.ValueKind = System.Text.Json.JsonValueKind.String
                        then
                            // Also use semantic classifier here to map the "string" from LLM to AgentDomain enum
                            let! result =
                                task {
                                    try
                                        return! semanticClassifier.ClassifyAsync(intentElem.GetString(), 0.5)
                                        |> Async.StartAsTask
                                    with _ ->
                                        return None
                                }

                            return result |> Option.map fst
                        else
                            return None
                    | Result.Error _ -> return None
            }

/// <summary>
/// Classifies the intent of the user prompt using a pluggable classifier.
/// </summary>
type IntentClassifierStage(classifier: IIntentClassifier) =
    interface IPreLlmStage with
        member _.Name = "IntentClassifier"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let! intent = classifier.ClassifyAsync(ctx.CurrentPrompt)
                    return { ctx with Intent = intent }
            }

/// <summary>
/// Compresses the context using the compressor's adaptive policy.
/// </summary>
type ContextSummarizerStage(compressor: ContextCompressor) =
    interface IPreLlmStage with
        member _.Name = "ContextSummarizer"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let! compressed = compressor.AutoCompress(ctx.CurrentPrompt)

                    if compressed = ctx.CurrentPrompt then
                        return ctx
                    else
                        return { ctx with CurrentPrompt = compressed }
            }

/// <summary>
/// Runs the Pre-LLM pipeline.
/// </summary>
type PreLlmPipeline(stages: IPreLlmStage list) =

    member _.ExecuteAsync(input: string) =
        task {
            let mutable ctx = PreLlmContext.Create(input)

            for stage in stages do
                if ctx.IsSafe then
                    let! nextCtx = stage.ExecuteAsync(ctx)
                    ctx <- nextCtx

            return ctx
        }
