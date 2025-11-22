namespace TarsEngine.FSharp.SelfImprovement

open System
open System.Collections.Generic
open System.IO
open System.Text.Json
open System.Text.RegularExpressions
open TarsEngine.FSharp.Core.Services.ReasoningTrace

/// Learns to critique reasoning traces and surfaces adaptive prompt guidance.
module MetaReasoningCritic =

    open PersistentAdaptiveMemory

    type CriticModel =
        { ScoreThreshold: float
          NegativeIndicators: (string * float) list
          AcceptanceMean: float option
          RejectionMean: float option
          SampleSize: int
          PromptAdjustments: Map<string, string> }

    [<CLIMutable>]
    type PromptAdjustmentEnvelope =
        { updatedAt: DateTime
          scoreThreshold: float
          sampleSize: int
          adjustments: Map<string, string> }

    let private serializerOptions =
        JsonSerializerOptions(PropertyNamingPolicy = JsonNamingPolicy.CamelCase, WriteIndented = true)

    let defaultPromptPath =
        Path.Combine(Environment.CurrentDirectory, "output", "adaptive_prompts.json")

    let private tokenize (text: string) =
        if String.IsNullOrWhiteSpace(text) then
            []
        else
            Regex.Split(text.ToLowerInvariant(), "[^a-z0-9]+")
            |> Array.filter (fun token ->
                token.Length >= 3
                && token <> "the"
                && token <> "and"
                && token <> "for"
                && token <> "that"
                && token <> "with"
                && token <> "from")
            |> Array.toList

    let private gatherScores (entries: MemoryEntry list) (statuses: Set<string>) =
        entries
        |> List.collect (fun entry ->
            match entry.critic with
            | Some critic when statuses.Contains critic.status ->
                entry.reasoning
                |> List.collect (fun trace -> trace.events)
                |> List.choose (fun evt -> evt.score)
            | _ -> [])

    let private average (values: float list) =
        if values.IsEmpty then None else Some(values |> List.average)

    let private clamp01 value =
        if value < 0.0 then 0.0
        elif value > 1.0 then 1.0
        else value

    let private computeIndicatorWeights (entries: MemoryEntry list) (statuses: Set<string>) =
        let counts = Dictionary<string, int>()
        let mutable totalTokens = 0

        for entry in entries do
            match entry.critic with
            | Some critic when statuses.Contains critic.status ->
                for trace in entry.reasoning do
                    for evt in trace.events do
                        let tokens = tokenize evt.message
                        totalTokens <- totalTokens + tokens.Length
                        for token in tokens do
                            match counts.TryGetValue(token) with
                            | true, current -> counts.[token] <- current + 1
                            | _ -> counts.[token] <- 1
            | _ -> ()

        if totalTokens = 0 then
            []
        else
            counts
            |> Seq.map (fun kvp -> kvp.Key, float kvp.Value / float totalTokens)
            |> Seq.filter (fun (_, weight) -> weight >= 0.01)
            |> Seq.sortByDescending snd
            |> Seq.truncate 12
            |> Seq.toList

    let private basePromptLibrary =
        dict [
            ("hallucination", "Cross-check every generated fact against repository context before finalizing.")
            ("fabricated", "Reject speculative claims; require verifiable evidence from the knowledge base.")
            ("contradiction", "Resolve conflicting statements by tracing each claim to concrete artifacts.")
            ("missing", "Fill in missing validation steps before delivering conclusions.")
            ("test", "Run or reference relevant tests to substantiate reasoning.")
            ("spec", "Align reasoning with the spec acceptance criteria before moving forward.")
            ("uncertain", "Flag uncertainty explicitly and gather additional evidence prior to approval.")
            ("guess", "Eliminate guesses; request precise data from tooling or repository state.")
            ("retry", "Reattempt the failing step with revised parameters informed by previous output.")
            ("timeout", "Instrument long-running steps and ensure watchdog timers guard execution.")
            ("drift", "Re-anchor reasoning to the spec summary when the chain of thought drifts.")
            ("error", "Surface the root cause error log and describe a remediation plan.")
        ]

    let private buildPromptAdjustments (keywords: (string * float) list) (threshold: float) (consensusFailureRate: float) =
        let adjustments =
            keywords
            |> List.fold
                (fun acc (keyword, _) ->
                    let instruction =
                        match basePromptLibrary.TryGetValue(keyword) with
                        | true, directive -> directive
                        | _ ->
                            if keyword.Contains("trace") then
                                "Produce a structured reasoning trace and verify each step with source references."
                            elif keyword.Contains("plan") then
                                "Outline a step-by-step plan and confirm each action against live repository data."
                            else
                                $"Strengthen verification around '%s{keyword}' before concluding."

                    Map.add keyword instruction acc)
                Map.empty
            |> fun mapped ->
                if mapped.ContainsKey("score_guardrail") then mapped
                else mapped |> Map.add "score_guardrail" $"Reject traces when average confidence falls below %.2f{threshold}."

        let adjustments =
            if consensusFailureRate >= 0.2 then
                adjustments
                |> Map.add "consensus_recovery" "Add an explicit cross-agent consensus check before applying changes."
            else
                adjustments

        if consensusFailureRate >= 0.1 then
            adjustments
            |> Map.add "evidence_checkpoint" "Pause after each major reasoning hop and gather new evidence to avoid cascading errors."
        else
            adjustments

    let train (entries: MemoryEntry list) =
        if entries.IsEmpty then
            None
        else
            let negativeStatuses = set [ "reject"; "needs_review" ]
            let acceptScores = gatherScores entries (set [ "accept" ])
            let rejectScores = gatherScores entries negativeStatuses

            let acceptanceMean = average acceptScores
            let rejectionMean = average rejectScores

            let baseThreshold =
                match acceptanceMean, rejectionMean with
                | Some accept, Some reject when reject < accept ->
                    clamp01 ((accept + reject) / 2.0)
                | Some accept, _ -> clamp01 (accept * 0.85)
                | _, Some reject -> clamp01 (reject * 1.05)
                | _ -> 0.55

            let keywords = computeIndicatorWeights entries negativeStatuses

            let consensusFailureRate =
                let failures =
                    entries
                    |> List.filter (fun entry -> entry.consensus |> Option.exists (fun c -> c.status = "failed"))
                    |> List.length
                float failures / float entries.Length

            let adjustments = buildPromptAdjustments keywords baseThreshold consensusFailureRate

            Some {
                ScoreThreshold = baseThreshold
                NegativeIndicators = keywords
                AcceptanceMean = acceptanceMean
                RejectionMean = rejectionMean
                SampleSize = entries.Length
                PromptAdjustments = adjustments
            }

    let private isKeywordPresent (keyword: string) (text: string) =
        let pattern = "\\b" + Regex.Escape(keyword) + "\\b"
        Regex.IsMatch(text.ToLowerInvariant(), pattern)

    let private collectTriggeredKeywords (model: CriticModel) (traces: ReasoningTrace list) =
        let messageTokens =
            traces
            |> List.collect (fun trace ->
                trace.Events
                |> List.collect (fun evt -> tokenize evt.Message))

        model.NegativeIndicators
        |> List.choose (fun (keyword, _) ->
            if messageTokens |> List.contains keyword then Some keyword else None)
        |> List.distinct

    let evaluate (model: CriticModel) (traces: ReasoningTrace list) =
        let events = traces |> List.collect (fun trace -> trace.Events)
        let scores =
            events
            |> List.choose (fun evt -> evt.Score)

        let averageScore =
            if scores.IsEmpty then None else Some(List.average scores)

        let keywordScore =
            events
            |> List.fold (fun acc evt ->
                model.NegativeIndicators
                |> List.fold (fun state (keyword, weight) ->
                    if isKeywordPresent keyword evt.Message then state + weight else state) acc) 0.0

        let triggered = collectTriggeredKeywords model traces

        match averageScore with
        | Some score when score >= model.ScoreThreshold && keywordScore < 0.7 ->
            CriticVerdict.Accept
        | _ when keywordScore >= 1.4 ->
            let detail =
                if triggered.IsEmpty then
                    "High-risk reasoning indicators detected."
                else
                    sprintf "Indicators: %s" (String.Join(", ", triggered))
            CriticVerdict.Reject detail
        | Some score when score < model.ScoreThreshold ->
            let detail =
                if triggered.IsEmpty then
                    $"Average confidence %.2f{score} below threshold %.2f{model.ScoreThreshold}."
                else
                    sprintf "Low confidence (%.2f) with indicators: %s." score (String.Join(", ", triggered))
            CriticVerdict.NeedsReview detail
        | _ ->
            if triggered.IsEmpty then
                CriticVerdict.NeedsReview "Insufficient scoring data; manual review required."
            else
                let reason = sprintf "Indicators triggered: %s." (String.Join(", ", triggered))
                CriticVerdict.NeedsReview reason

    let persistPromptAdjustments (path: string) (model: CriticModel) =
        if String.IsNullOrWhiteSpace(path) then
            ()
        else
            let fullPath =
                let resolved = Path.GetFullPath(path)
                let directory = Path.GetDirectoryName(resolved)
                if not (String.IsNullOrWhiteSpace(directory)) then
                    Directory.CreateDirectory(directory) |> ignore
                resolved

            let dto =
                { updatedAt = DateTime.UtcNow
                  scoreThreshold = model.ScoreThreshold
                  sampleSize = model.SampleSize
                  adjustments = model.PromptAdjustments }

            let json = JsonSerializer.Serialize(dto, serializerOptions)
            File.WriteAllText(fullPath, json)

    let loadPromptAdjustments (path: string) =
        if String.IsNullOrWhiteSpace(path) || not (File.Exists(path)) then
            Map.empty
        else
            try
                let json = File.ReadAllText(path)
                let dto = JsonSerializer.Deserialize<PromptAdjustmentEnvelope>(json, serializerOptions)
                if obj.ReferenceEquals(dto, null) then Map.empty else dto.adjustments
            with
            | _ -> Map.empty

    let buildCritic (model: CriticModel) =
        fun traces -> evaluate model traces
