namespace Tars.Core

open System

// =============================================================================
// PHASE 15.3: EVIDENCE CHAINS
// =============================================================================
//
// Every belief links back to its evidence sources through traceable chains.
// Enables verification, weakest-link detection, and source tracing.
// Reference: docs/3_Roadmap/2_Phases/phase_15_symbolic_reflection.md

// Helper module for string truncation
[<AutoOpen>]
module private EvidenceChainHelpers =
    let truncateStr count (s: string) =
        if String.IsNullOrEmpty(s) then ""
        elif s.Length <= count then s
        else s.Substring(0, count) + "..."

/// An inference step showing how one belief led to another
type InferenceStep =
    { From: ReflectionBelief list
      Rule: ReflectionInferenceRule
      To: ReflectionBelief
      Confidence: float
      Timestamp: DateTimeOffset }

    static member Create(from, rule, toBelief, confidence) =
        { From = from
          Rule = rule
          To = toBelief
          Confidence = confidence
          Timestamp = DateTimeOffset.UtcNow }

/// Source trace showing the origin of evidence
type SourceTrace =
    | OriginatesFrom of url: string * timestamp: DateTimeOffset
    | ExtractedBy of agentId: AgentId * method: string
    | InferredFromBeliefs of beliefs: Guid list
    | ValidatedBy of agentId: AgentId * testName: string * passed: bool
    | UserProvided of userId: string option * providedAt: DateTimeOffset
    | SystemGenerated of componentName: string

/// A step in the confidence flow calculation
type ConfidenceStep =
    { EvidenceId: Guid
      EvidenceDescription: string
      Confidence: float
      ContributionWeight: float }

/// A broken link in an evidence chain
type BrokenLink =
    { FromBeliefId: Guid
      ToEvidenceId: Guid option
      Reason: string }

/// Complete evidence chain for a belief
type EvidenceChain =
    { Belief: ReflectionBelief
      DirectEvidence: ReflectionEvidence list
      InferredFrom: InferenceStep list
      TransitiveEvidence: ReflectionEvidence list
      SourceTraces: SourceTrace list
      ConfidenceFlow: ConfidenceStep list
      OverallConfidence: float
      WeakestLink: (ReflectionEvidence * float) option
      CreatedAt: DateTimeOffset }

    static member CalculateConfidence(directEvidence: ReflectionEvidence list, inferenceSteps: InferenceStep list) =
        let directConf =
            if directEvidence.IsEmpty then
                1.0
            else
                directEvidence |> List.averageBy (fun e -> e.Confidence)

        let inferenceConf =
            if inferenceSteps.IsEmpty then
                1.0
            else
                inferenceSteps |> List.map (fun s -> s.Confidence) |> List.min

        min directConf inferenceConf

module EvidenceChains =

    /// Build an evidence chain for a belief
    let buildChain (belief: ReflectionBelief) (allBeliefs: Map<Guid, ReflectionBelief>) : EvidenceChain =
        let directEvidence = belief.Evidence

        let weakestLink =
            if directEvidence.IsEmpty then
                None
            else
                directEvidence
                |> List.minBy (fun e -> e.Confidence)
                |> fun e -> Some(e, e.Confidence)

        let sourceTraces =
            directEvidence
            |> List.map (fun e ->
                match e.Source with
                | DirectObservation(_, ts) -> UserProvided(None, ts)
                | ExternalSource(url, ts) -> OriginatesFrom(url, ts)
                | InferredFrom _ -> SystemGenerated("inference_engine")
                | TestResult(name, passed) -> ValidatedBy(AgentId(Guid.Empty), name, passed)
                | UserFeedback(_, userId) -> UserProvided(userId, e.Timestamp)
                | AgentExperience(agentId, method) -> ExtractedBy(agentId, method))

        let confidenceFlow =
            directEvidence
            |> List.mapi (fun _ e ->
                { EvidenceId = e.Id
                  EvidenceDescription = truncateStr 50 e.Content
                  Confidence = e.Confidence
                  ContributionWeight = 1.0 / float (max 1 directEvidence.Length) })

        let overallConfidence =
            if directEvidence.IsEmpty then
                belief.Confidence
            else
                EvidenceChain.CalculateConfidence(directEvidence, [])

        { Belief = belief
          DirectEvidence = directEvidence
          InferredFrom = []
          TransitiveEvidence = []
          SourceTraces = sourceTraces
          ConfidenceFlow = confidenceFlow
          OverallConfidence = overallConfidence
          WeakestLink = weakestLink
          CreatedAt = DateTimeOffset.UtcNow }

    /// Verify that a chain has no broken links
    let verifyChain (chain: EvidenceChain) : Result<unit, BrokenLink list> =
        let brokenLinks = ResizeArray<BrokenLink>()

        for evidence in chain.DirectEvidence do
            match evidence.Source with
            | ExternalSource(url, _) when String.IsNullOrWhiteSpace(url) ->
                brokenLinks.Add(
                    { FromBeliefId = chain.Belief.Id
                      ToEvidenceId = Some evidence.Id
                      Reason = "External source URL is empty" }
                )
            | InferredFrom [] ->
                brokenLinks.Add(
                    { FromBeliefId = chain.Belief.Id
                      ToEvidenceId = Some evidence.Id
                      Reason = "Inferred evidence has no premises" }
                )
            | _ -> ()

        for evidence in chain.DirectEvidence do
            if evidence.Confidence < 0.0 || evidence.Confidence > 1.0 then
                brokenLinks.Add(
                    { FromBeliefId = chain.Belief.Id
                      ToEvidenceId = Some evidence.Id
                      Reason = sprintf "Invalid confidence: %f" evidence.Confidence }
                )

        if brokenLinks.Count = 0 then
            FSharp.Core.Ok()
        else
            FSharp.Core.Error(brokenLinks |> Seq.toList)

    /// Find the weakest link in a chain
    let findWeakestLink (chain: EvidenceChain) : (ReflectionEvidence * float) option = chain.WeakestLink

    /// Trace a belief back to its original sources
    let traceToSources (belief: ReflectionBelief) (allBeliefs: Map<Guid, ReflectionBelief>) : SourceTrace list =
        let chain = buildChain belief allBeliefs
        chain.SourceTraces

    /// Get chain completeness score (0.0 - 1.0)
    let getCompletenessScore (chain: EvidenceChain) : float =
        let hasDirectEvidence = if chain.DirectEvidence.IsEmpty then 0.0 else 0.3
        let hasSourceTraces = if chain.SourceTraces.IsEmpty then 0.0 else 0.3
        let hasConfidenceFlow = if chain.ConfidenceFlow.IsEmpty then 0.0 else 0.2

        let noWeakLink =
            match chain.WeakestLink with
            | Some(_, conf) when conf > 0.7 -> 0.2
            | Some _ -> 0.1
            | None -> 0.0

        hasDirectEvidence + hasSourceTraces + hasConfidenceFlow + noWeakLink

    /// Render chain as ASCII visualization
    let visualize (chain: EvidenceChain) : string =
        let sb = System.Text.StringBuilder()

        sb.AppendLine(sprintf "Belief: \"%s\"" (truncateStr 60 chain.Belief.Statement))
        |> ignore

        sb.AppendLine(sprintf "├─ Confidence: %.2f" chain.OverallConfidence) |> ignore
        sb.AppendLine("├─ Direct Evidence:") |> ignore

        for i, evidence in chain.DirectEvidence |> List.indexed do
            let prefix =
                if i = chain.DirectEvidence.Length - 1 then
                    "│  └─"
                else
                    "│  ├─"

            sb.AppendLine(sprintf "%s [%.2f] %s" prefix evidence.Confidence (truncateStr 40 evidence.Content))
            |> ignore

        sb.AppendLine("├─ Source Traces:") |> ignore

        for i, trace in chain.SourceTraces |> List.indexed do
            let prefix =
                if i = chain.SourceTraces.Length - 1 then
                    "│  └─"
                else
                    "│  ├─"

            let traceStr =
                match trace with
                | OriginatesFrom(url, _) -> sprintf "URL: %s" url
                | ExtractedBy(_, method) -> sprintf "Extracted via: %s" method
                | InferredFromBeliefs ids -> sprintf "Inferred from %d beliefs" ids.Length
                | ValidatedBy(_, test, passed) -> sprintf "Test: %s (%s)" test (if passed then "✓" else "✗")
                | UserProvided(userId, _) -> sprintf "User: %s" (userId |> Option.defaultValue "anonymous")
                | SystemGenerated comp -> sprintf "System: %s" comp

            sb.AppendLine(sprintf "%s %s" prefix traceStr) |> ignore

        match chain.WeakestLink with
        | Some(evidence, conf) ->
            sb.AppendLine(sprintf "└─ Weakest Link: [%.2f] %s" conf (truncateStr 40 evidence.Content))
            |> ignore
        | None -> sb.AppendLine("└─ No weak links detected") |> ignore

        sb.ToString()
