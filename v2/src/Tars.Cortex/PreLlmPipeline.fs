namespace Tars.Cortex

open System
open System.Threading
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
/// A simple safety filter that blocks known dangerous keywords.
/// </summary>
type SafetyFilterStage() =
    let dangerousKeywords =
        [ "rm -rf"; "format c:"; "drop database"; "sudo rm"; ":(){ :|:& };:" ]

    interface IPreLlmStage with
        member _.Name = "SafetyFilter"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let inputLower = ctx.CurrentPrompt.ToLowerInvariant()
                    let found = dangerousKeywords |> List.tryFind (fun k -> inputLower.Contains(k))

                    match found with
                    | Some keyword ->
                        return
                            { ctx with
                                IsSafe = false
                                BlockReason = Some $"Contains dangerous keyword: {keyword}" }
                    | None -> return ctx
            }

/// <summary>
/// Classifies the intent of the user prompt using heuristics (or LLM in future).
/// </summary>
type IntentClassifierStage() =
    interface IPreLlmStage with
        member _.Name = "IntentClassifier"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else
                    let input = ctx.CurrentPrompt.ToLowerInvariant()

                    // Simple heuristics for v2.0 Alpha
                    let intent =
                        if
                            input.Contains("create")
                            || input.Contains("write")
                            || input.Contains("code")
                            || input.Contains("function")
                        then
                            Some AgentDomain.Coding
                        elif input.Contains("plan") || input.Contains("step") || input.Contains("roadmap") then
                            Some AgentDomain.Planning
                        elif input.Contains("why") || input.Contains("what") || input.Contains("how") then
                            Some AgentDomain.Reasoning
                        else
                            Some AgentDomain.Chat

                    return { ctx with Intent = intent }
            }

/// <summary>
/// Compresses the context if it exceeds a certain length or entropy threshold.
/// </summary>
type ContextSummarizerStage(compressor: ContextCompressor) =
    interface IPreLlmStage with
        member _.Name = "ContextSummarizer"

        member _.ExecuteAsync(ctx) =
            task {
                if not ctx.IsSafe then
                    return ctx
                else if
                    // Only compress if length > 1000 chars (arbitrary threshold for now)
                    ctx.CurrentPrompt.Length > 1000
                then
                    let! compressed = compressor.AutoCompress(ctx.CurrentPrompt)
                    return { ctx with CurrentPrompt = compressed }
                else
                    return ctx
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
