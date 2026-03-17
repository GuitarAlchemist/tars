namespace TarsEngine

open System
open System.Collections.Generic

/// Governance Evolution engine for meta-compounding governance artifacts
module GovernanceEvolution =

    /// Tetravalent logic states (True, False, Unknown, Contradictory)
    type TetravalentState =
        | True
        | False
        | Unknown
        | Contradictory

    /// Governance artifact types
    type ArtifactType =
        | Constitution
        | Policy
        | Persona
        | Schema
        | Contract

    /// Governance event types
    type EventType =
        | Created
        | Cited
        | Violated
        | Amended
        | Promoted
        | Demoted

    /// Individual governance event
    type GovernanceEvent = {
        Type: EventType
        Context: string
        Timestamp: DateTime
    }

    /// Metrics for artifact effectiveness
    type Metrics = {
        CitationCount: int
        ViolationCount: int
        ComplianceRate: float
        LastCited: DateTime option
        LastViolated: DateTime option
        PromotionCandidate: bool
        DeprecationCandidate: bool
    }

    /// Assessment of artifact effectiveness
    type Assessment = {
        Effectiveness: TetravalentState
        Recommendation: string // "maintain" | "promote" | "demote" | "deprecate" | "investigate"
    }

    /// Evolution log entry for a governance artifact
    type EvolutionEntry = {
        Id: string
        Artifact: string
        ArtifactType: ArtifactType
        Metrics: Metrics
        Events: GovernanceEvent list
        Assessment: Assessment
        CreatedAt: DateTime
        LastUpdated: DateTime
    }

    /// Compounding report summarizing governance evolution
    type CompoundingReport = {
        GeneratedAt: DateTime
        TotalArtifacts: int
        PromotionCandidates: EvolutionEntry list
        DeprecationCandidates: EvolutionEntry list
        EffectiveArtifacts: EvolutionEntry list
        ProblematicArtifacts: EvolutionEntry list
    }

    /// Promotion staircase path
    type PromotionStaircase =
        | Pattern
        | Policy
        | Constitutional

    /// Scan evolution log and collect all entries
    let scanEvolutionLog (entries: EvolutionEntry list) : EvolutionEntry list =
        entries
        |> List.sortByDescending (fun e -> e.LastUpdated)

    /// Detect promotion candidates: citation_count >= 3, compliance 100%
    let detectPromotionCandidates (entries: EvolutionEntry list) : EvolutionEntry list =
        entries
        |> List.filter (fun e ->
            e.Metrics.CitationCount >= 3 &&
            e.Metrics.ComplianceRate >= 1.0
        )
        |> List.sortByDescending (fun e -> e.Metrics.CitationCount)

    /// Detect deprecation candidates: zero citations for 90+ days
    let detectDeprecationCandidates (entries: EvolutionEntry list) (now: DateTime) : EvolutionEntry list =
        let ninetyDaysAgo = now.AddDays(-90.0)
        entries
        |> List.filter (fun e ->
            e.Metrics.CitationCount = 0 &&
            match e.Metrics.LastCited with
            | Some lastCited -> lastCited < ninetyDaysAgo
            | None -> e.CreatedAt < ninetyDaysAgo
        )
        |> List.sortBy (fun e -> e.LastUpdated)

    /// Assess effectiveness of an artifact based on metrics
    let assessEffectiveness (entry: EvolutionEntry) : Assessment =
        let effectiveness =
            if entry.Metrics.ComplianceRate >= 0.95 then
                True
            elif entry.Metrics.ComplianceRate >= 0.70 then
                Unknown
            elif entry.Metrics.ComplianceRate >= 0.50 then
                Contradictory
            else
                False

        let recommendation =
            match effectiveness, entry.Metrics.PromotionCandidate, entry.Metrics.DeprecationCandidate with
            | True, true, _ -> "promote"
            | True, false, _ -> "maintain"
            | Unknown, _, _ -> "investigate"
            | Contradictory, _, false -> "maintain"
            | Contradictory, _, true -> "demote"
            | False, _, _ -> "deprecate"

        { Effectiveness = effectiveness; Recommendation = recommendation }

    /// Generate compounding report from evolution log
    let generateCompoundingReport (entries: EvolutionEntry list) (now: DateTime) : CompoundingReport =
        let promotionCandidates = detectPromotionCandidates entries
        let deprecationCandidates = detectDeprecationCandidates entries now

        let effective =
            entries
            |> List.filter (fun e -> e.Metrics.ComplianceRate >= 0.95)

        let problematic =
            entries
            |> List.filter (fun e ->
                e.Metrics.ComplianceRate < 0.70 &&
                e.Metrics.CitationCount > 0
            )

        {
            GeneratedAt = now
            TotalArtifacts = entries.Length
            PromotionCandidates = promotionCandidates
            DeprecationCandidates = deprecationCandidates
            EffectiveArtifacts = effective
            ProblematicArtifacts = problematic
        }

    /// Map artifact type to promotion staircase level
    let artifactTypeToStaircase (artifactType: ArtifactType) : PromotionStaircase =
        match artifactType with
        | Constitution -> Constitutional
        | Policy -> Policy
        | Persona -> Pattern
        | Schema -> Pattern
        | Contract -> Pattern

    /// Determine next promotion level
    let nextPromotionLevel (current: PromotionStaircase) : PromotionStaircase option =
        match current with
        | Pattern -> Some Policy
        | Policy -> Some Constitutional
        | Constitutional -> None

    /// Create a new evolution entry
    let createEvolutionEntry (id: string) (artifact: string) (artifactType: ArtifactType) : EvolutionEntry =
        let now = DateTime.UtcNow
        {
            Id = id
            Artifact = artifact
            ArtifactType = artifactType
            Metrics = {
                CitationCount = 0
                ViolationCount = 0
                ComplianceRate = 1.0
                LastCited = None
                LastViolated = None
                PromotionCandidate = false
                DeprecationCandidate = false
            }
            Events = [{ Type = Created; Context = sprintf "Initial creation of %s" id; Timestamp = now }]
            Assessment = { Effectiveness = Unknown; Recommendation = "maintain" }
            CreatedAt = now
            LastUpdated = now
        }

    /// Record a citation event
    let recordCitation (entry: EvolutionEntry) (context: string) : EvolutionEntry =
        let now = DateTime.UtcNow
        let updatedMetrics = {
            entry.Metrics with
                CitationCount = entry.Metrics.CitationCount + 1
                LastCited = Some now
                DeprecationCandidate = false
        }
        let newEvent = { Type = Cited; Context = context; Timestamp = now }
        {
            entry with
                Metrics = updatedMetrics
                Events = entry.Events @ [newEvent]
                LastUpdated = now
        }

    /// Record a violation event
    let recordViolation (entry: EvolutionEntry) (context: string) : EvolutionEntry =
        let now = DateTime.UtcNow
        let totalAssessments = entry.Metrics.CitationCount + entry.Metrics.ViolationCount + 1
        let updatedMetrics = {
            entry.Metrics with
                ViolationCount = entry.Metrics.ViolationCount + 1
                LastViolated = Some now
                ComplianceRate = float entry.Metrics.CitationCount / float totalAssessments
        }
        let newEvent = { Type = Violated; Context = context; Timestamp = now }
        {
            entry with
                Metrics = updatedMetrics
                Events = entry.Events @ [newEvent]
                LastUpdated = now
        }

    /// Update metrics for promotion/deprecation evaluation
    let updatePromotionMetrics (entry: EvolutionEntry) : EvolutionEntry =
        let promotionCandidate =
            entry.Metrics.CitationCount >= 3 &&
            entry.Metrics.ComplianceRate >= 1.0

        let ninetyDaysAgo = DateTime.UtcNow.AddDays(-90.0)
        let deprecationCandidate =
            entry.Metrics.CitationCount = 0 &&
            match entry.Metrics.LastCited with
            | Some lastCited -> lastCited < ninetyDaysAgo
            | None -> entry.CreatedAt < ninetyDaysAgo

        let updatedMetrics = {
            entry.Metrics with
                PromotionCandidate = promotionCandidate
                DeprecationCandidate = deprecationCandidate
        }

        let assessment = assessEffectiveness { entry with Metrics = updatedMetrics }

        {
            entry with
                Metrics = updatedMetrics
                Assessment = assessment
                LastUpdated = DateTime.UtcNow
        }
