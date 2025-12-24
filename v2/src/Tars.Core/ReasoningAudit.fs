namespace Tars.Core

open System
open System.Collections.Generic
open System.Globalization

/// Tracks decisions made during cognitive workflows (GoT/WoT) for transparency.
type BranchDecision =
    { NodeId: Guid
      Content: string
      NodeType: string
      Action: string
      Status: string
      Score: float option
      Confidence: float option
      Reasons: string list
      Risks: string list
      Timestamp: DateTime }

/// Audit accumulator for reasoning decisions emitted by GoT/WoT.
type ReasoningAudit =
    { Records: ResizeArray<BranchDecision>
      Lock: obj }

type ReasoningAuditStats =
    { Total: int
      Kept: int
      Pruned: int
      Scored: int
      Thresholded: int
      HeuristicFallbacks: int
      ParseFailures: int
      ScoreMin: float option
      ScoreMax: float option
      ConfidenceMin: float option
      ConfidenceMax: float option }

module ReasoningAudit =
    /// Creates a fresh audit collector.
    let create () =
        { Records = ResizeArray()
          Lock = obj() }

    /// Records a decision in the audit log.
    let record (audit: ReasoningAudit) (decision: BranchDecision) =
        lock audit.Lock (fun () -> audit.Records.Add(decision))

    /// Returns an immutable snapshot of all tracked decisions.
    let snapshot (audit: ReasoningAudit) =
        lock audit.Lock (fun () -> audit.Records |> Seq.toList)

    let private hasReasonPrefix (prefix: string) (decision: BranchDecision) =
        decision.Reasons
        |> List.exists (fun reason -> reason.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))

    let private hasRisk (risk: string) (decision: BranchDecision) =
        decision.Risks
        |> List.exists (fun r -> r.Equals(risk, StringComparison.OrdinalIgnoreCase))

    let private formatFloat (value: float) =
        value.ToString("0.00", CultureInfo.InvariantCulture)

    let private formatRange label minValue maxValue =
        match minValue, maxValue with
        | Some minScore, Some maxScore -> $"{label}={formatFloat minScore}-{formatFloat maxScore}"
        | _ -> $"{label}=n/a"

    let private tryMin values =
        if List.isEmpty values then
            None
        else
            Some(List.min values)

    let private tryMax values =
        if List.isEmpty values then
            None
        else
            Some(List.max values)

    /// Aggregates statistics for downstream analysis and diagnostics.
    let stats (audit: ReasoningAudit) =
        let records = snapshot audit
        let total = records.Length

        let kept =
            records
            |> List.filter (fun d -> d.Status.Equals("kept", StringComparison.OrdinalIgnoreCase))
            |> List.length

        let pruned =
            records
            |> List.filter (fun d -> d.Status.Equals("pruned", StringComparison.OrdinalIgnoreCase))
            |> List.length

        let scored =
            records
            |> List.filter (fun d -> d.Action.Equals("score", StringComparison.OrdinalIgnoreCase))
            |> List.length

        let thresholded =
            records
            |> List.filter (fun d -> d.Action.Equals("threshold", StringComparison.OrdinalIgnoreCase))
            |> List.length

        let scoreDecisions =
            records
            |> List.filter (fun d -> d.Action.Equals("score", StringComparison.OrdinalIgnoreCase))

        let heuristicFallbacks =
            scoreDecisions
            |> List.filter (hasReasonPrefix "heuristic_")
            |> List.length

        let parseFailures =
            scoreDecisions
            |> List.filter (hasRisk "score_parse_failed")
            |> List.length

        let scores = records |> List.choose (fun d -> d.Score)
        let confidences = records |> List.choose (fun d -> d.Confidence)

        { Total = total
          Kept = kept
          Pruned = pruned
          Scored = scored
          Thresholded = thresholded
          HeuristicFallbacks = heuristicFallbacks
          ParseFailures = parseFailures
          ScoreMin = tryMin scores
          ScoreMax = tryMax scores
          ConfidenceMin = tryMin confidences
          ConfidenceMax = tryMax confidences }

    /// True when heuristic scoring was used due to parse issues.
    let usesHeuristic (audit: ReasoningAudit) =
        let stats = stats audit
        stats.HeuristicFallbacks > 0 || stats.ParseFailures > 0

    /// Summarizes key metrics about the recorded decisions.
    let summary (audit: ReasoningAudit) =
        let stats = stats audit
        let scoreRange = formatRange "score" stats.ScoreMin stats.ScoreMax
        let confRange = formatRange "conf" stats.ConfidenceMin stats.ConfidenceMax

        sprintf
            "decisions=%d;kept=%d;pruned=%d;fallbacks=%d;parseFailures=%d;%s;%s"
            stats.Total
            stats.Kept
            stats.Pruned
            stats.HeuristicFallbacks
            stats.ParseFailures
            scoreRange
            confRange
