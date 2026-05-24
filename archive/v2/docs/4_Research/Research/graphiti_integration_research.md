# TARS + Graphiti Integration Research

**Date:** November 29, 2025
**Status:** Deep Research Complete
**Author:** TARS Research Agent

## Executive Summary

This document presents a comprehensive analysis of how TARS can leverage **Graphiti** (Zep's temporal knowledge graph engine) to identify patterns, perform clustering, tag discoveries, find improvement opportunities, detect anomalies, and distill new grammars for F# reinjection.

---

## 1. Graphiti Architecture Overview

### 1.1 Core Concepts (from Zep Paper arXiv:2501.13956v1)

Graphiti implements a **temporally-aware dynamic knowledge graph** `G = (N, E, φ)` with three hierarchical tiers:

| Tier | Description | TARS Mapping |
|------|-------------|--------------|
| **Episode Subgraph (Gₑ)** | Raw input data (messages, text, JSON) - non-lossy store | Agent interactions, code changes, evolution history |
| **Semantic Entity Subgraph (Gₛ)** | Extracted entities + relationships (facts) | Code patterns, beliefs, agents, grammars |
| **Community Subgraph (Gₖ)** | Clusters of strongly connected entities | Pattern families, code modules, concept clusters |

### 1.2 Key Differentiators

1. **Bi-temporal Model**: Tracks both `T` (event timeline) and `T'` (ingestion timeline)
   - `t_valid`, `t_invalid` ∈ T: When facts were true
   - `t'_created`, `t'_expired` ∈ T': When facts entered the system

2. **Temporal Edge Invalidation**: Handles contradictions without lossy summarization
   - New facts can invalidate old facts by setting `t_invalid`
   - Maintains full history for point-in-time queries

3. **Hybrid Search**: `φ_cos` (semantic) + `φ_bm25` (full-text) + `φ_bfs` (graph traversal)

4. **Community Detection**: Label propagation with dynamic extension for real-time updates

---

## 2. TARS Current State Analysis

### 2.1 Existing Graph Infrastructure

**v2/src/Tars.Graph/Domain.fs** - Current node/edge types:
```fsharp
type GraphNode =
    | Concept of name: string
    | Agent of id: string
    | File of path: string
    | Task of id: string
    | Belief of id: string * content: string

type GraphEdge =
    | RelatesTo of weight: float
    | CreatedBy | DependsOn | Solves | HasBelief | IsA
```

**v2/src/Tars.Cortex/KnowledgeGraph.fs** - Current capabilities:
- `AddNode`, `AddEdge`, `GetNeighbors`
- `FindPath` (BFS), `MultiHopTraverse`
- `GetHubNodes` (highly-connected nodes)
- `IndexDocuments` (co-occurrence-based linking)

**Gap Analysis**: Missing temporal tracking, entity resolution, community detection, contradiction handling.

### 2.2 Existing Pattern Infrastructure

**v2/src/Tars.Core/Patterns.fs** - Agentic patterns:
- Chain of Thought (CoT)
- ReAct (Reason + Act)
- Plan and Execute

**v2/src/Tars.Cortex/GraphAnalyzer.fs** - K-Theory analysis:
- Cyclomatic complexity computation
- Independent cycle detection
- Gaussian elimination for kernel computation

### 2.3 Existing Grammar Infrastructure

**v2/src/Tars.Core/GrammarDistill.fs**:
- `GrammarSpec` type with Fields, Required, Example, PromptHint, Validator
- `fromJsonExamples`: Distills grammar from JSON examples
- `metascriptHint`: Metascript-oriented prompt generation

---

## 3. Integration Architecture

### 3.1 Proposed Graphiti-Enhanced Knowledge Graph

```
┌─────────────────────────────────────────────────────────────────┐
│                    TARS.Memory.GraphitiKG                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Episode Layer   │  │ Semantic Layer  │  │ Community Layer │  │
│  │ ─────────────── │  │ ─────────────── │  │ ─────────────── │  │
│  │ • Code Changes  │  │ • Patterns      │  │ • Pattern       │  │
│  │ • Agent Runs    │  │ • Beliefs       │  │   Clusters      │  │
│  │ • Reflections   │  │ • Grammars      │  │ • Module        │  │
│  │ • User Inputs   │  │ • Agents        │  │   Families      │  │
│  │ • Tool Calls    │  │ • Concepts      │  │ • Anomaly       │  │
│  │                 │  │ • Files         │  │   Groups        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
│           └────────────────────┼────────────────────┘           │
│                                │                                │
│  ┌─────────────────────────────┴─────────────────────────────┐  │
│  │              Temporal Tracking Engine                      │  │
│  │  • Bi-temporal model (event time + ingestion time)        │  │
│  │  • Edge invalidation for contradictions                   │  │
│  │  • Point-in-time queries                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Custom Entity Types for TARS

Following Graphiti's Pydantic-style entity definitions, we propose F# discriminated unions:

```fsharp
/// Graphiti-style entity types for TARS
type TarsEntity =
    | CodePattern of {|
        Name: string
        Category: PatternCategory  // Structural, Behavioral, Creational, Agentic
        Signature: string
        Occurrences: int
        FirstSeen: DateTime
        LastSeen: DateTime |}
    | AgentBelief of {|
        Statement: string
        Confidence: float
        DerivedFrom: string list
        ValidFrom: DateTime
        InvalidAt: DateTime option |}
    | GrammarRule of {|
        Name: string
        Production: string
        Examples: string list
        DistilledFrom: string list |}
    | CodeModule of {|
        Path: string
        Namespace: string
        Dependencies: string list
        Complexity: float |}
    | Anomaly of {|
        Type: AnomalyType  // Inconsistency, Duplication, DeadCode, StyleViolation
        Location: string
        Severity: float
        DetectedAt: DateTime |}

type TarsFact =
    | Implements of source: TarsEntity * target: TarsEntity * confidence: float
    | DependsOn of source: TarsEntity * target: TarsEntity * strength: float
    | Contradicts of source: TarsEntity * target: TarsEntity * resolution: string option
    | EvolvedFrom of source: TarsEntity * target: TarsEntity * delta: string
    | BelongsTo of entity: TarsEntity * community: string
```

---

## 4. Pattern Recognition & Clustering

### 4.1 Clustering Strategy (Drill-Down Approach)

```
Level 0: Global View
    └── All entities in knowledge graph

Level 1: Community Detection (Label Propagation)
    ├── Pattern Cluster: "Agentic Patterns"
    │   └── CoT, ReAct, Plan&Execute, Reflection
    ├── Pattern Cluster: "Budget Governance"
    │   └── TokenBudget, SemanticFanOut, AdaptiveReflection
    ├── Module Cluster: "Core Infrastructure"
    │   └── Kernel, Cortex, Memory, Graph
    └── Anomaly Cluster: "Inconsistencies"
        └── Duplicate patterns, conflicting beliefs

Level 2: Sub-Community Analysis
    └── Drill into specific cluster for detailed analysis

Level 3: Entity-Level Analysis
    └── Individual pattern/belief/grammar inspection
```

### 4.2 Pattern Detection Algorithms

| Algorithm | Purpose | TARS Application |
|-----------|---------|------------------|
| **Label Propagation** | Community detection | Group related patterns, modules |
| **Louvain/Leiden** | Hierarchical clustering | Multi-level pattern taxonomy |
| **PageRank** | Hub identification | Find central patterns/concepts |
| **Temporal Motifs** | Recurring sequences | Detect evolution patterns |
| **Contradiction Detection** | Inconsistency finding | Belief conflicts, API mismatches |

### 4.3 Tagging System

```fsharp
type PatternTag =
    | Structural of level: int        // Code structure patterns
    | Behavioral of frequency: float  // Runtime behavior patterns
    | Evolutionary of generation: int // Self-improvement patterns
    | Anomalous of severity: float    // Detected issues
    | Emergent of confidence: float   // Newly discovered patterns

type TaggingResult = {
    Entity: TarsEntity
    Tags: PatternTag list
    AutoGenerated: bool
    Confidence: float
    Timestamp: DateTime
}
```



---

## 5. Opportunity Discovery

### 5.1 Code Improvement Opportunities

| Opportunity Type | Detection Method | Example |
|------------------|------------------|---------|
| **Duplication** | Semantic similarity on code patterns | Two modules implementing same logic |
| **Dead Code** | Unreachable nodes in call graph | Unused functions/types |
| **Complexity Hotspots** | High cyclomatic complexity + many edges | Refactoring candidates |
| **Missing Abstractions** | Repeated pattern clusters | Extract common interface |
| **API Inconsistencies** | Contradicting facts about same entity | Signature mismatches |

### 5.2 Research Spawning

When Graphiti detects novel patterns or anomalies, TARS can spawn research tasks:

```fsharp
type ResearchOpportunity =
    | PatternInvestigation of {|
        Pattern: TarsEntity
        Question: string
        Priority: float
        SuggestedApproach: string |}
    | AnomalyResolution of {|
        Anomaly: TarsEntity
        PossibleCauses: string list
        SuggestedFixes: string list |}
    | GrammarEvolution of {|
        CurrentGrammar: GrammarRule
        ProposedExtension: string
        Justification: string |}
    | ArchitectureReview of {|
        Community: string
        Concern: string
        Recommendation: string |}
```

### 5.3 Anomaly Detection Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Ingest     │───▶│  Extract    │───▶│  Compare    │───▶│  Flag       │
│  Episode    │    │  Entities   │    │  with       │    │  Anomalies  │
│             │    │  & Facts    │    │  Existing   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                            │
                                            ▼
                                   ┌─────────────────┐
                                   │ Contradiction   │
                                   │ Detection       │
                                   │ • Temporal      │
                                   │ • Semantic      │
                                   │ • Structural    │
                                   └─────────────────┘
```

---

## 6. Grammar Distillation Pipeline

### 6.1 From Patterns to Grammars

The key innovation is using Graphiti's community detection to identify recurring patterns, then distilling them into F# grammars:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Grammar Distillation Pipeline                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. COLLECT                    2. CLUSTER                           │
│  ┌─────────────────┐          ┌─────────────────┐                   │
│  │ Code Examples   │          │ Community       │                   │
│  │ Agent Outputs   │─────────▶│ Detection       │                   │
│  │ User Patterns   │          │ (Label Prop)    │                   │
│  └─────────────────┘          └────────┬────────┘                   │
│                                        │                            │
│  3. EXTRACT                    4. DISTILL                           │
│  ┌─────────────────┐          ┌─────────────────┐                   │
│  │ Common          │          │ F# Grammar      │                   │
│  │ Structures      │◀─────────│ Generation      │                   │
│  │ (AST patterns)  │          │ (DU + CE)       │                   │
│  └─────────────────┘          └────────┬────────┘                   │
│                                        │                            │
│  5. VALIDATE                   6. REINJECT                          │
│  ┌─────────────────┐          ┌─────────────────┐                   │
│  │ Type Check      │          │ Hot-reload      │                   │
│  │ Test Against    │─────────▶│ into TARS       │                   │
│  │ Examples        │          │ Runtime         │                   │
│  └─────────────────┘          └─────────────────┘                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Grammar Distillation Types

```fsharp
/// Enhanced GrammarDistill module with Graphiti integration
module GraphitiGrammarDistill =

    type DistillationSource =
        | CommunityPatterns of communityId: string * patterns: TarsEntity list
        | TemporalSequence of episodes: Episode list
        | BeliefCluster of beliefs: AgentBelief list
        | CodeExamples of files: string list

    type DistilledGrammar = {
        Name: string
        Version: int
        Productions: Production list
        ComputationExpression: string option  // F# CE code
        DiscriminatedUnion: string option     // F# DU code
        Validator: string -> Result<obj, string>
        SourceCommunity: string
        Confidence: float
        DistilledAt: DateTime
    }

    type Production = {
        Name: string
        Pattern: string      // EBNF-like pattern
        FSharpType: string   // Corresponding F# type
        Examples: string list
    }

    /// Distill grammar from a Graphiti community
    let fromCommunity (kg: GraphitiKG) (communityId: string) : DistilledGrammar =
        // 1. Get all entities in community
        let entities = kg.GetCommunityMembers(communityId)

        // 2. Extract common structural patterns
        let patterns = extractCommonPatterns entities

        // 3. Generate F# types
        let duCode = generateDiscriminatedUnion patterns
        let ceCode = generateComputationExpression patterns

        // 4. Create validator
        let validator = createValidator patterns

        { Name = $"Grammar_{communityId}"
          Version = 1
          Productions = patterns
          ComputationExpression = Some ceCode
          DiscriminatedUnion = Some duCode
          Validator = validator
          SourceCommunity = communityId
          Confidence = calculateConfidence entities
          DistilledAt = DateTime.UtcNow }
```

### 6.3 F# Reinjection Mechanism

```fsharp
/// Hot-reload distilled grammars into TARS runtime
module GrammarReinjection =

    type ReinjectionResult =
        | Success of grammarName: string * loadedAt: DateTime
        | ValidationFailed of errors: string list
        | CompilationFailed of errors: string list
        | RuntimeConflict of existing: string * proposed: string

    /// Reinject a distilled grammar into the running TARS instance
    let reinject (grammar: DistilledGrammar) (runtime: TarsRuntime) : ReinjectionResult =
        // 1. Validate grammar doesn't conflict with existing
        match checkConflicts grammar runtime.LoadedGrammars with
        | Some conflict -> RuntimeConflict(conflict.Existing, conflict.Proposed)
        | None ->
            // 2. Compile F# code dynamically
            match compileGrammar grammar with
            | Error errors -> CompilationFailed errors
            | Ok assembly ->
                // 3. Load into runtime
                runtime.LoadGrammar(grammar.Name, assembly)
                Success(grammar.Name, DateTime.UtcNow)
```


---

## 7. Implementation Roadmap

### 7.1 Phase 1: Foundation (Weeks 1-2)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.1.1** | Extend `GraphNode` with temporal fields | `ValidFrom`, `InvalidAt`, `CreatedAt` |
| **7.1.2** | Extend `GraphEdge` with bi-temporal model | `T` and `T'` timeline support |
| **7.1.3** | Implement Episode ingestion | `IngestEpisode` function |
| **7.1.4** | Add entity extraction prompts | LLM-based entity extraction |

### 7.2 Phase 2: Semantic Layer (Weeks 3-4)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.2.1** | Implement entity resolution | Deduplication via embedding similarity |
| **7.2.2** | Implement fact extraction | Relationship extraction from episodes |
| **7.2.3** | Add temporal edge invalidation | Contradiction handling |
| **7.2.4** | Integrate with existing `KnowledgeGraph.fs` | Unified API |

### 7.3 Phase 3: Community Detection (Weeks 5-6)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.3.1** | Implement Label Propagation | Community detection algorithm |
| **7.3.2** | Add dynamic community extension | Real-time updates |
| **7.3.3** | Generate community summaries | LLM-based summarization |
| **7.3.4** | Build drill-down navigation | Hierarchical exploration |

### 7.4 Phase 4: Pattern Recognition (Weeks 7-8)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.4.1** | Implement pattern tagging | Automatic classification |
| **7.4.2** | Add anomaly detection | Contradiction + inconsistency detection |
| **7.4.3** | Build opportunity discovery | Research spawning |
| **7.4.4** | Create pattern visualization | Graph-based UI |

### 7.5 Phase 5: Grammar Distillation (Weeks 9-10)

| Task | Description | Deliverable |
|------|-------------|-------------|
| **7.5.1** | Implement grammar extraction | From community patterns |
| **7.5.2** | Generate F# DUs and CEs | Code generation |
| **7.5.3** | Add validation pipeline | Type checking + testing |
| **7.5.4** | Implement hot-reload | Runtime reinjection |

---

## 8. Concrete Use Cases

### 8.1 Use Case: Code Pattern Discovery

**Scenario**: TARS analyzes its own codebase to discover recurring patterns.

```
Input: All .fs files in v2/src/
Process:
  1. Ingest each file as Episode
  2. Extract entities: Functions, Types, Modules
  3. Extract facts: Calls, Implements, DependsOn
  4. Detect communities: "Error Handling", "Async Patterns", "DU Patterns"
  5. Tag patterns: Structural, Behavioral
  6. Identify anomalies: Inconsistent error handling across modules
Output:
  - Pattern catalog with 15 identified patterns
  - 3 anomalies flagged for review
  - 2 grammar candidates for distillation
```

### 8.2 Use Case: Belief Evolution Tracking

**Scenario**: Track how TARS's beliefs evolve over multiple evolution cycles.

```
Input: Evolution history from Tars.Evolution.Engine
Process:
  1. Ingest each reflection as Episode
  2. Extract beliefs with temporal validity
  3. Detect contradictions (belief A invalidates belief B)
  4. Build belief evolution graph
  5. Identify stable vs volatile beliefs
Output:
  - Belief timeline visualization
  - Contradiction report
  - Confidence decay analysis
```

### 8.3 Use Case: Grammar Evolution

**Scenario**: Distill a new grammar from successful agent interactions.

```
Input: 100 successful agent outputs in JSON format
Process:
  1. Cluster outputs by structural similarity
  2. Extract common fields and patterns
  3. Generate F# discriminated union
  4. Generate computation expression builder
  5. Validate against original examples
  6. Hot-reload into TARS runtime
Output:
  - New grammar: "AgentResponseGrammar_v2"
  - 98% validation success rate
  - Runtime loaded successfully
```

---

## 9. Key Findings & Recommendations

### 9.1 Findings

1. **TARS already has graph infrastructure** (`KnowledgeGraph.fs`, `GraphAnalyzer.fs`) but lacks temporal tracking and community detection.

2. **Graphiti's bi-temporal model** is essential for tracking belief evolution and handling contradictions - a key requirement for self-improving agents.

3. **Community detection enables pattern clustering** which is the foundation for grammar distillation.

4. **The existing `GrammarDistill.fs`** provides a starting point but needs enhancement for community-based distillation.

5. **Graphiti's hybrid search** (semantic + full-text + graph traversal) would significantly improve TARS's retrieval capabilities.

### 9.2 Recommendations

| Priority | Recommendation | Rationale |
|----------|----------------|-----------|
| **P0** | Add temporal fields to `GraphNode` and `GraphEdge` | Foundation for all temporal features |
| **P0** | Implement Episode ingestion | Entry point for all data |
| **P1** | Add Label Propagation for community detection | Enables clustering and drill-down |
| **P1** | Implement contradiction detection | Critical for belief management |
| **P2** | Build grammar distillation pipeline | Enables self-improvement |
| **P2** | Add pattern tagging system | Improves discoverability |
| **P3** | Implement hot-reload for grammars | Enables runtime evolution |

### 9.3 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM costs for entity extraction | High operational cost | Use smaller models, caching |
| Graph size explosion | Performance degradation | Implement pruning, archival |
| Grammar conflicts | Runtime errors | Strict validation, sandboxing |
| Temporal complexity | Implementation difficulty | Start with simple T timeline |

---

## 10. Conclusion

Integrating Graphiti's temporal knowledge graph architecture into TARS would provide:

1. **Pattern Recognition**: Community detection enables automatic clustering of code patterns, beliefs, and grammars.

2. **Temporal Awareness**: Bi-temporal tracking allows TARS to understand how its knowledge evolves and handle contradictions gracefully.

3. **Grammar Distillation**: Community-based pattern extraction provides a principled approach to generating new F# grammars.

4. **Anomaly Detection**: Contradiction detection and temporal analysis can identify inconsistencies and improvement opportunities.

5. **Self-Improvement**: The combination of pattern recognition, grammar distillation, and hot-reload enables true self-evolution.

The research synthesis document already mentions "Internal Knowledge Graph (Graphiti-style, backed by SQLite)" as a target for `Tars.Memory`. This research provides the detailed architecture and implementation roadmap to realize that vision.

---

## Appendix A: Graphiti Prompts (from Zep Paper)

### Entity Extraction Prompt
```
Given the above conversation, extract entity nodes from the CURRENT MESSAGE:
1. ALWAYS extract the speaker/actor as the first node
2. Extract other significant entities, concepts, or actors
3. DO NOT create nodes for relationships or actions
4. DO NOT create nodes for temporal information
5. Be as explicit as possible in your node names
```

### Fact Extraction Prompt
```
Given the above MESSAGES and ENTITIES, extract all facts:
1. Extract facts only between the provided entities
2. Each fact should represent a clear relationship between two DISTINCT nodes
3. The relation_type should be concise, all-caps (e.g., LOVES, WORKS_FOR)
4. Provide a more detailed fact containing all relevant information
5. Consider temporal aspects of relationships when relevant
```

### Temporal Extraction Prompt
```
Analyze the conversation and determine if there are dates that are part of the edge fact:
1. Use ISO 8601 format for datetimes
2. Use the reference timestamp as the current time
3. If the fact is written in present tense, use Reference Timestamp for valid_at
4. If no temporal information is found, leave fields as null
5. For relative time mentions, calculate actual datetime from reference
```

---

## Appendix B: References

1. Zep Paper: "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv:2501.13956v1)
2. Graphiti GitHub: https://github.com/getzep/graphiti
3. GraphRAG Paper: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"
4. TARS Architecture: v2/docs/2_Analysis/Architecture/00_Overview/tars_v2_architecture.md
5. TARS Research Synthesis: v2/docs/2_Analysis/research_synthesis.md
