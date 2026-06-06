namespace Tars.Evolution

open System
open System.Text.Json
open System.Threading.Tasks
open Tars.Llm
open Tars.Core.MetaCognition
open Tars.Cortex

/// LLM-enhanced failure analysis.
/// Collects failures from pattern outcomes and cycle results, then optionally
/// uses LLM to refine root causes beyond heuristic classification.
module FailureAnalyzer =

    /// Convert a PatternOutcome (from PatternSelector) to a FailureRecord.
    let fromPatternOutcome (outcome: PatternOutcomeStore.PatternOutcome) : FailureRecord option =
        if outcome.Success then None
        else
            Some
                { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
                  Goal = outcome.Goal
                  PatternUsed = sprintf "%A" outcome.PatternKind
                  ErrorMessage = "Pattern execution failed"
                  TraceStepCount = 0
                  FailedAtStep = None
                  Timestamp = outcome.Timestamp
                  Tags = GapDetection.extractDomainTags outcome.Goal
                  Score = 0.0 }

    /// Convert a CycleResult (from RetroactionLoop) to a FailureRecord.
    let fromCycleResult (cr: RetroactionLoop.CycleResult) : FailureRecord option =
        match cr.Score with
        | Some score when score.Success -> None  // Not a failure
        | _ ->
            let errorMsg =
                cr.Result
                |> Option.defaultValue "No result produced"
                |> fun s -> s.Substring(0, min 200 s.Length)
            Some
                { RunId = Guid.NewGuid().ToString("N").Substring(0, 12)
                  Goal = cr.Problem.Description
                  PatternUsed =
                      cr.NewPattern
                      |> Option.map (fun p -> p.Name)
                      |> Option.defaultValue "unknown"
                  ErrorMessage = errorMsg
                  TraceStepCount = 0
                  FailedAtStep = None
                  Timestamp = DateTime.UtcNow
                  Tags = cr.Problem.Tags
                  Score = cr.Score |> Option.map (fun s -> s.Overall) |> Option.defaultValue 0.0 }

    /// Collect all failure records from available sources.
    let collectFailures
        (outcomes: PatternOutcomeStore.PatternOutcome list)
        (cycleResults: RetroactionLoop.CycleResult list)
        : FailureRecord list =
        let fromOutcomes = outcomes |> List.choose fromPatternOutcome
        let fromCycles = cycleResults |> List.choose fromCycleResult
        fromOutcomes @ fromCycles
        |> List.sortByDescending (fun f -> f.Timestamp)

    /// Use LLM to refine the root cause classification for a cluster.
    let refineRootCause
        (llm: ILlmService)
        (cluster: FailureCluster)
        : Task<FailureRootCause> =
        task {
            let errorSamples =
                cluster.Members
                |> List.truncate 5
                |> List.map (fun f -> sprintf "- Goal: %s\n  Error: %s\n  Pattern: %s" f.Goal f.ErrorMessage f.PatternUsed)
                |> String.concat "\n"

            let prompt =
                sprintf """Analyze these failure records from a reasoning system and classify the root cause.

FAILURES:
%s

Classify as exactly ONE of:
- MISSING_TOOL: <tool_name> (a needed tool/capability is absent)
- WRONG_PATTERN: <used> -> <suggested> (wrong reasoning pattern for this task)
- KNOWLEDGE_GAP: <domain> (lacking domain knowledge)
- INSUFFICIENT_CONTEXT: <what's missing> (not enough input information)
- MODEL_LIMITATION: <detail> (model capacity/token limit issue)
- BAD_PROMPT: <issue> (prompt formatting or instruction problem)
- EXTERNAL_FAILURE: <service> (external service/API issue)
- UNKNOWN: <detail>

Output ONLY: CATEGORY: detail""" errorSamples

            let req =
                { ModelHint = Some "cheap"
                  Model = None
                  SystemPrompt = Some "You are a failure analysis engine. Be precise and concise."
                  MaxTokens = Some 100
                  Temperature = Some 0.0
                  Stop = []
                  Messages = [ { Role = Role.User; Content = prompt } ]
                  Tools = []
                  ToolChoice = None
                  ResponseFormat = None
                  Stream = false
                  JsonMode = false
                  Seed = None
                  ContextWindow = None }

            try
                let! resp = llm.CompleteAsync req
                let text = resp.Text.Trim().ToUpperInvariant()

                if text.StartsWith("MISSING_TOOL") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.MissingTool detail
                elif text.StartsWith("WRONG_PATTERN") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    let parts = detail.Split("->") |> Array.map (fun s -> s.Trim())
                    return FailureRootCause.WrongPattern(
                        (if parts.Length > 0 then parts.[0] else "unknown"),
                        (if parts.Length > 1 then parts.[1] else "try alternative"))
                elif text.StartsWith("KNOWLEDGE_GAP") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.KnowledgeGap detail
                elif text.StartsWith("INSUFFICIENT_CONTEXT") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.InsufficientContext detail
                elif text.StartsWith("MODEL_LIMITATION") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.ModelLimitation detail
                elif text.StartsWith("BAD_PROMPT") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.BadPrompt detail
                elif text.StartsWith("EXTERNAL_FAILURE") then
                    let detail = text.Substring(text.IndexOf(':') + 1).Trim()
                    return FailureRootCause.ExternalFailure detail
                else
                    return cluster.RootCause  // Keep heuristic classification
            with _ ->
                return cluster.RootCause  // Fallback to heuristic on LLM error
        }

    /// Full pipeline: collect -> cluster -> classify -> optionally refine with LLM.
    let analyzeFailures
        (llm: ILlmService option)
        (threshold: float)
        (outcomes: PatternOutcomeStore.PatternOutcome list)
        (cycleResults: RetroactionLoop.CycleResult list)
        : Task<FailureCluster list> =
        task {
            let failures = collectFailures outcomes cycleResults
            let clusters = FailureClustering.buildClusters threshold failures

            match llm with
            | Some llmService ->
                let mutable refined = []
                for c in clusters do
                    let! rootCause = refineRootCause llmService c
                    refined <- { c with RootCause = rootCause } :: refined
                return refined |> List.rev
            | None ->
                return clusters
        }
