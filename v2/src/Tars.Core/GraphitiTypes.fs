namespace Tars.Core

open System

/// Bi-temporal validity model for tracking when facts are true
type TemporalValidity =
    { ValidFrom: DateTime
      InvalidAt: DateTime option
      CreatedAt: DateTime
      ExpiredAt: DateTime option }

module TemporalValidityOps =
    let now () =
        { ValidFrom = DateTime.UtcNow
          InvalidAt = None
          CreatedAt = DateTime.UtcNow
          ExpiredAt = None }

    let invalidate (v: TemporalValidity) =
        { v with
            InvalidAt = Some DateTime.UtcNow
            ExpiredAt = Some DateTime.UtcNow }

    let isValidAt (at: DateTime) (v: TemporalValidity) =
        v.ValidFrom <= at
        && (v.InvalidAt |> Option.map (fun inv -> at < inv) |> Option.defaultValue true)

/// Episode types for raw data ingestion
type Episode =
    | AgentInteraction of agentId: string * input: string * output: string * timestamp: DateTime
    | CodeChange of file: string * diff: string * author: string * timestamp: DateTime
    | Reflection of agentId: string * content: string * timestamp: DateTime
    | UserMessage of content: string * metadata: Map<string, string> * timestamp: DateTime
    | ToolCall of name: string * args: Map<string, string> * result: string * timestamp: DateTime
    | BeliefUpdate of agentId: string * belief: string * confidence: float * timestamp: DateTime
    | PatternDetected of patternType: string * location: string * details: string * timestamp: DateTime

module Episode =
    let timestamp =
        function
        | AgentInteraction(_, _, _, ts) -> ts
        | CodeChange(_, _, _, ts) -> ts
        | Reflection(_, _, ts) -> ts
        | UserMessage(_, _, ts) -> ts
        | ToolCall(_, _, _, ts) -> ts
        | BeliefUpdate(_, _, _, ts) -> ts
        | PatternDetected(_, _, _, ts) -> ts

    let typeTag =
        function
        | AgentInteraction _ -> "agent_interaction"
        | CodeChange _ -> "code_change"
        | Reflection _ -> "reflection"
        | UserMessage _ -> "user_message"
        | ToolCall _ -> "tool_call"
        | BeliefUpdate _ -> "belief_update"
        | PatternDetected _ -> "pattern_detected"

/// Pattern categories
type PatternCategory =
    | Structural
    | Behavioral
    | Creational
    | Agentic
    | Architectural
    | CustomCategory of string

/// Severity levels for anomalies
type AnomalySeverity =
    | Info
    | Low
    | Medium
    | High
    | Critical

/// Anomaly types
type AnomalyType =
    | Inconsistency of description: string
    | Duplication of locations: string list
    | DeadCode of path: string
    | StyleViolation of rule: string
    | PerformanceIssue of metric: string
    | SecurityConcern of cwe: string option
    | BeliefConflict of beliefs: string list

/// Code pattern entity
type CodePatternEntity =
    { Name: string
      Category: PatternCategory
      Signature: string
      Occurrences: int
      FirstSeen: DateTime
      LastSeen: DateTime }

/// Agent belief entity
type AgentBeliefEntity =
    { Statement: string
      Confidence: float
      DerivedFrom: string list
      AgentId: string
      ValidFrom: DateTime
      InvalidAt: DateTime option }

/// Grammar rule entity
type GrammarRuleEntity =
    { Name: string
      Production: string
      Examples: string list
      DistilledFrom: string list
      Version: int }

/// Code module entity
type CodeModuleEntity =
    { Path: string
      Namespace: string
      Dependencies: string list
      Complexity: float
      LineCount: int }

/// Anomaly entity
type AnomalyEntity =
    { Type: AnomalyType
      Location: string
      Severity: AnomalySeverity
      DetectedAt: DateTime
      ResolvedAt: DateTime option }

/// Concept entity
type ConceptEntity =
    { Name: string
      Description: string
      RelatedConcepts: string list }

/// Graphiti-style entity types for TARS
type TarsEntity =
    | CodePatternE of CodePatternEntity
    | AgentBeliefE of AgentBeliefEntity
    | GrammarRuleE of GrammarRuleEntity
    | CodeModuleE of CodeModuleEntity
    | AnomalyE of AnomalyEntity
    | ConceptE of ConceptEntity
    | EpisodeE of Episode
    | FileE of path: string
    | FunctionE of name: string

module TarsEntity =
    let getId =
        function
        | CodePatternE p -> $"pattern:{p.Name.ToLowerInvariant()}"
        | AgentBeliefE b -> $"belief:{b.Statement.GetHashCode():x8}"
        | GrammarRuleE g -> $"grammar:{g.Name}:v{g.Version}"
        | CodeModuleE m -> $"module:{m.Path.GetHashCode():x8}"
        | AnomalyE a -> $"anomaly:{a.Location.GetHashCode():x8}:{a.DetectedAt.Ticks}"
        | ConceptE c -> $"concept:{c.Name.ToLowerInvariant()}"
        | EpisodeE e -> $"episode:{e.GetHashCode():x8}:{Episode.timestamp e |> fun t -> t.Ticks}"
        | FileE p -> $"file:{p.GetHashCode():x8}"
        | FunctionE n -> $"func:{n.ToLowerInvariant()}"

/// Facts represent relationships between entities
type TarsFact =
    | Implements of source: TarsEntity * target: TarsEntity * confidence: float
    | DependsOn of source: TarsEntity * target: TarsEntity * strength: float
    | Contradicts of source: TarsEntity * target: TarsEntity * resolution: string option
    | EvolvedFrom of source: TarsEntity * target: TarsEntity * delta: string
    | BelongsTo of entity: TarsEntity * communityId: string
    | SimilarTo of source: TarsEntity * target: TarsEntity * similarity: float
    | DerivedFrom of source: TarsEntity * target: TarsEntity
    | Contains of source: TarsEntity * target: TarsEntity

module TarsFact =
    let source =
        function
        | Implements(s, _, _) -> s
        | DependsOn(s, _, _) -> s
        | Contradicts(s, _, _) -> s
        | EvolvedFrom(s, _, _) -> s
        | BelongsTo(e, _) -> e
        | SimilarTo(s, _, _) -> s
        | DerivedFrom(s, _) -> s
        | Contains(s, _) -> s

    let target =
        function
        | Implements(_, t, _) -> Some t
        | DependsOn(_, t, _) -> Some t
        | Contradicts(_, t, _) -> Some t
        | EvolvedFrom(_, t, _) -> Some t
        | BelongsTo(_, _) -> None
        | SimilarTo(_, t, _) -> Some t
        | DerivedFrom(_, t) -> Some t
        | Contains(_, t) -> Some t

/// Pattern tags for classification
type PatternTag =
    | StructuralTag of level: int
    | BehavioralTag of frequency: float
    | EvolutionaryTag of generation: int
    | AnomalousTag of severity: float
    | EmergentTag of confidence: float
    | CommunityTag of communityId: string

/// Result of tagging an entity
type TaggingResult =
    { Entity: TarsEntity
      Tags: PatternTag list
      AutoGenerated: bool
      Confidence: float
      Timestamp: DateTime }

/// Research opportunity
type ResearchOpportunity =
    | PatternInvestigation of pattern: TarsEntity * question: string * priority: float
    | AnomalyResolution of anomaly: TarsEntity * causes: string list * fixes: string list
    | GrammarEvolution of grammar: TarsEntity * extension: string * justification: string
    | ArchitectureReview of community: string * concern: string * recommendation: string
