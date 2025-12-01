# Teaching TARS How to Learn to Learn

**Last Updated:** November 29, 2025
**Status:** Living Document
**Related:** [Epistemic Governor](epistemic_governor.md) | [Graphiti Research](../Research/graphiti_integration_research.md) | [Grammar Integration](grammar-knowledge-integration.md)

---

## 1. Purpose

This document explains how to operationalize *meta-learning* in TARS v2 so the system improves its own reasoning, tooling, and prompts over time. It ties together existing components (Curriculum/Executor/Evaluator, Epistemic Governor, Grammar Distillation, Metascripts, Trace Recorder, Toolsmithing) into a governed loop with measurable progress.

---

## 2. Definition

**"Learn to learn"** = TARS runs a closed loop that:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        META-LEARNING CYCLE                              │
│                                                                         │
│    OBSERVE ──▶ ABSTRACT ──▶ EDIT ──▶ VALIDATE ──▶ DEPLOY               │
│        │                                              │                 │
│        └──────────────────────────────────────────────┘                 │
│                         (feedback loop)                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

The goal is to increase **transfer** (generalization to new tasks) while keeping **safety constraints** (budgets, K-theory invariants, epistemic gates) intact.

### 2.1 Key Principles

| Principle | Description | Mechanism |
|-----------|-------------|-----------|
| **Temporal Awareness** | Track when beliefs become valid/invalid | Graphiti-style bi-temporal model |
| **Contradiction Handling** | Detect and resolve conflicting beliefs | Temporal edge invalidation |
| **Pattern Clustering** | Group similar learnings for abstraction | Community detection algorithms |
| **Grammar Evolution** | Distill patterns into reusable F# code | Grammar distillation pipeline |
| **Safe Deployment** | Validate changes before promotion | A/B testing + auto-revert |

---

## 3. Prerequisites (Minimum Viable Stack)

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Tracing** | Reproduce runs deterministically | Trace Recorder + Mock Tool framework ([BRIDGING_THE_GAPS](03_Operational/BRIDGING_THE_GAPS.md)) |
| **Knowledge** | Store extracted principles | BeliefGraph (Graphiti-style) + VectorStore |
| **Constraints** | Enforce output formats | GrammarDistillation (GBNF) + style rules |
| **Policy Knobs** | Control execution | BudgetGovernor, Safety gates, Circuit Breakers |
| **Execution** | Run learning tasks | Metascript engine + Tars.Graph + Sandbox |
| **Temporal Store** | Track belief evolution | Bi-temporal episode/entity/community graph |

---

## 4. The Meta-Learning Loop (Ouroboros)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUROBOROS LOOP                                   │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ 1.PROPOSE│───▶│ 2.EXECUTE│───▶│3.EVALUATE│───▶│ 4.DISTILL│              │
│  │Curriculum│    │ Executor │    │ Governor │    │ Grammar  │              │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘              │
│       ▲                                               │                    │
│       │         ┌──────────┐    ┌──────────┐    ┌─────▼────┐              │
│       └─────────│7.MONITOR │◀───│ 6.POLICY │◀───│5.TOOLSMITH│              │
│                 │  Deploy  │    │  Update  │    │ Synthesize│              │
│                 └──────────┘    └──────────┘    └──────────┘              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.1 Step 1: Propose (Curriculum Agent)

The Curriculum Agent samples tasks using a **multi-armed bandit** strategy:

```fsharp
type TaskBucket =
    | Regression of knownGap: string      // Retry past failures
    | Exploration of domain: string       // Try new areas
    | Drill of strength: string           // Reinforce known skills
    | Contradiction of beliefA: Guid * beliefB: Guid  // Resolve conflicts

type CurriculumSample = {
    Task: TaskDefinition
    Bucket: TaskBucket
    Budget: TokenBudget
    AllowedTools: ToolId list
    KnowledgeBoundary: string list  // Which beliefs can be used
}
```

**Sampling Policy**: UCB (Upper Confidence Bound) or Thompson Sampling over buckets, with reward = `f(success, cost_delta, novelty)`.

### 4.2 Step 2: Execute (Executor Agent)

```fsharp
type ExecutionTrace = {
    TaskId: Guid
    Messages: Message list          // Full dialogue
    ToolCalls: ToolCall list        // Tools invoked
    BudgetDeltas: BudgetDelta list  // Token/time/cost changes
    Errors: Error list              // Failures encountered
    Outcome: ExecutionOutcome       // Success/Failure/Partial
    Timestamp: DateTime             // For temporal tracking
}
```

**Key**: Emit **dense traces** as Episodes for the Graphiti-style knowledge graph.

### 4.3 Step 3: Evaluate (Epistemic Governor)

The [Epistemic Governor](epistemic_governor.md) performs:

1. **Falsification**: Generate variants, test generalization
2. **Extraction**: Distill solution into abstract `Belief`
3. **Contradiction Detection**: Compare with existing beliefs

```fsharp
type EvaluationResult = {
    TaskId: Guid
    GeneralizationScore: float      // 0.0-1.0: how well it transfers
    ExtractedBeliefs: Belief list   // New principles learned
    Contradictions: Contradiction list  // Conflicts with existing beliefs
    BrittlePaths: string list       // Flagged as non-generalizable
}

type Contradiction = {
    ExistingBelief: Belief
    NewBelief: Belief
    Resolution: ContradictionResolution
}

type ContradictionResolution =
    | InvalidateOld of reason: string   // New belief supersedes
    | RejectNew of reason: string       // Existing belief wins
    | Merge of mergedBelief: Belief     // Combine both
    | Scope of oldScope: string * newScope: string  // Both valid in different contexts
```

### 4.4 Step 4: Distill & Constrain (Grammar Evolution Agent)

Convert patterns into grammars using community detection:

```fsharp
type GrammarDistillationInput = {
    SuccessfulTraces: ExecutionTrace list
    ExtractedBeliefs: Belief list
    PatternClusters: Community list  // From Graphiti-style clustering
}

type DistilledArtifact =
    | GBNFRule of name: string * production: string
    | StyleRule of name: string * constraint: string
    | MetascriptTemplate of name: string * dag: MetascriptNode list
    | FSharpType of name: string * definition: string  // New DU or CE
```

### 4.5 Step 5: Toolsmithing (Tool Synthesis Agent)

Detect patterns that warrant new tools:

```fsharp
type ToolCandidate = {
    Name: string
    TriggerConditions: TriggerCondition list
    ProposedSchema: ToolSchema
    EstimatedSavings: CostEstimate
    TestCases: TestCase list
}

type TriggerCondition =
    | RepetitionAboveThreshold of count: int
    | LatencyAboveBaseline of ratio: float
    | BudgetBurnAboveBaseline of ratio: float
    | FlakeRateAboveThreshold of rate: float
```

**Rule**: Require **two signals** before proposing a tool (e.g., repetition + latency).

### 4.6 Step 6: Policy Update (Governance)

```fsharp
type PolicyUpdate = {
    RoutingChanges: RoutingRule list      // Model selection heuristics
    RetrievalKnobs: RetrievalConfig       // Fan-out limits, top-k
    CurriculumWeights: Map<TaskBucket, float>  // Updated sampling
    CircuitBreakerUpdates: BreakerUpdate list
}
```

### 4.7 Step 7: Redeploy & Monitor

```fsharp
type DeploymentStrategy =
    | Shadow of baseline: Version * candidate: Version
    | ABTest of allocation: float  // % traffic to candidate
    | Canary of rolloutPct: float * duration: TimeSpan

type PromotionCriteria = {
    MinStabilityRuns: int
    MaxFlakeRateIncrease: float
    MaxCostIncrease: float
    MinSuccessRateDelta: float
}
```

**Auto-revert**: If metrics regress beyond thresholds, automatically roll back.

---

## 5. Data Products (Graphiti-Style Three-Tier Model)

Following the [Graphiti research](../Research/graphiti_integration_research.md), organize data into three tiers:

### 5.1 Episode Layer (Gₑ) - Raw Data
| Product | Format | Purpose |
|---------|--------|---------|
| **Traces** | JSON Lines | Full dialogue + tool calls + budgets + outcomes (source of truth) |
| **Code Changes** | Git diffs | What was modified |
| **Tool Invocations** | Structured logs | Who called what, when |

### 5.2 Semantic Entity Layer (Gₛ) - Extracted Knowledge
| Product | Format | Purpose |
|---------|--------|---------|
| **Beliefs** | Vector-embedded | Abstract principles with confidence, lineage, temporal validity |
| **Grammars** | GBNF + F# | Constrained output formats |
| **Tools** | Registry entries | Metadata: schema, health, cost, win rate |

### 5.3 Community Layer (Gₖ) - Clustered Patterns
| Product | Format | Purpose |
|---------|--------|---------|
| **Pattern Families** | Cluster IDs | Groups of related beliefs/grammars |
| **Module Communities** | Graph clusters | Strongly connected code areas |
| **Skill Trees** | DAG | Prerequisite relationships between capabilities |

---

## 6. Metrics to Track

### 6.1 Core Metrics Dashboard

| Category | Metric | Target | Alert Threshold |
|----------|--------|--------|-----------------|
| **Task Success** | Pass rate | > 80% | < 60% |
| | Time-to-first-solution | < 30s | > 120s |
| | Retry count | < 3 | > 5 |
| **Generalization** | Variant pass rate | > 70% | < 50% |
| | Transfer score | > 0.6 | < 0.4 |
| **Stability** | Flake rate | < 5% | > 15% |
| | Trace replay divergence | < 2% | > 10% |
| **Cost** | Tokens per task | < 10K | > 50K |
| | $ per solved task | < $0.10 | > $0.50 |
| **Tool Health** | Error rate | < 2% | > 10% |
| | Circuit breaker trips | 0/day | > 3/day |
| **Drift** | Grammar violations | < 5/100 | > 20/100 |
| | Belief entropy | < 0.3 | > 0.5 |

### 6.2 Temporal Metrics (Graphiti-Inspired)

```fsharp
type TemporalMetrics = {
    BeliefChurnRate: float        // Invalidations per day
    ContradictionRate: float      // New contradictions per 100 beliefs
    CommunityStability: float     // Cluster membership changes
    KnowledgeHalfLife: TimeSpan   // Time until belief confidence decays 50%
}
```

---

## 7. Implementation Phases

### Phase 0: Manual Loop (Current)
| Deliverable | Status | Notes |
|-------------|--------|-------|
| Record traces | 🟡 Partial | Basic tracing in Evolution.Engine |
| Hand-run Governor prompts | 🔴 Not started | Need prompt templates |
| Hand-edit grammar/style rules | 🟡 Partial | GrammarDistill.fs exists |
| Manual tool creation | 🟢 Working | Ad-hoc tool additions |

### Phase 1: Deterministic Replay
| Deliverable | Status | Owner |
|-------------|--------|-------|
| Trace Recorder (JSON Lines) | 🔴 | TBD |
| Mock Tool Framework | 🔴 | TBD |
| Trace Diff Tool | 🔴 | TBD |
| Golden trace baseline | 🔴 | TBD |

### Phase 2: Automated Extraction
| Deliverable | Status | Owner |
|-------------|--------|-------|
| Auto-run Governor post-execution | 🔴 | TBD |
| Belief → GrammarDistill pipeline | 🔴 | TBD |
| GBNF fragment appender | 🔴 | TBD |
| Temporal validity tracking | 🔴 | TBD |

### Phase 3: Curriculum Self-Play
| Deliverable | Status | Owner |
|-------------|--------|-------|
| Nightly batch runner | 🔴 | TBD |
| Multi-armed bandit sampler | 🔴 | TBD |
| Metrics dashboard (CLI) | 🔴 | TBD |
| Contradiction detection | 🔴 | TBD |

### Phase 4: Toolsmithing
| Deliverable | Status | Owner |
|-------------|--------|-------|
| Tool Synthesis Agent | 🔴 | TBD |
| ToolHealthMonitor | 🔴 | TBD |
| Auto-test generation | 🔴 | TBD |
| Registry promotion workflow | 🔴 | TBD |

### Phase 5: Safe Promotion
| Deliverable | Status | Owner |
|-------------|--------|-------|
| Shadow deployment | 🔴 | TBD |
| A/B testing framework | 🔴 | TBD |
| Auto-revert system | 🔴 | TBD |
| Signed changelog | 🔴 | TBD |

---

## 8. Safety & Governance Hooks

### 8.1 Defense-in-Depth Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                      SAFETY LAYERS                              │
│                                                                 │
│  Layer 1: Budget Governor                                       │
│  ├── Per-task token/time/money caps                            │
│  └── Graceful degradation: summarize → truncate → exit          │
│                                                                 │
│  Layer 2: Loop/Entropy Watchdogs                                │
│  ├── Detect: no state change for N iterations                  │
│  ├── Detect: context entropy rising (semantic drift)           │
│  └── Action: abort or "cooling cycle" (re-ground)              │
│                                                                 │
│  Layer 3: Circuit Breakers                                      │
│  ├── Quarantine tools with error_rate > threshold              │
│  ├── Exponential backoff on failures                           │
│  └── Fallback routing to alternative tools                     │
│                                                                 │
│  Layer 4: Knowledge Boundaries                                  │
│  ├── Tag tasks with allowed belief sources                     │
│  ├── Refuse to use beliefs outside boundary                    │
│  └── Audit trail for boundary crossings                        │
│                                                                 │
│  Layer 5: Human Checkpoints                                     │
│  ├── Manual approval for tool promotion (early phases)         │
│  ├── Manual approval for grammar tightening                    │
│  └── Escalation path for unresolvable contradictions           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Thermodynamic Safety Model

From the [Epistemic Governor](epistemic_governor.md):

| Concept | Role | Mechanism |
|---------|------|-----------|
| **Maxwell's Demon** | Entropy reduction | Governor sorts "hot" (true) from "cold" (noise) tokens |
| **Simulated Annealing** | Search strategy | High temp for exploration → quench to freeze verified truths |
| **Cooling Cycles** | Context restoration | Summarize, prune, re-ground when entropy threshold exceeded |

---

## 9. Operational Playbook

### 9.1 Daily Routine

```
08:00  ┌─ Self-Play Window Opens ─────────────────────────────┐
       │ 1. Curriculum samples N tasks (bandit strategy)      │
       │ 2. Executor runs each task in sandbox                │
       │ 3. Governor evaluates, extracts beliefs              │
       │ 4. Grammar distiller updates constraints             │
       └──────────────────────────────────────────────────────┘
20:00  ┌─ Nightly Batch ──────────────────────────────────────┐
       │ 5. Trace replay validation (determinism check)       │
       │ 6. A/B cohort preparation for next day               │
       │ 7. Metrics aggregation → dashboard update            │
       └──────────────────────────────────────────────────────┘
```

### 9.2 Weekly Review

- [ ] Review new beliefs: prune low-confidence (< 0.5)
- [ ] Review new tools: check health metrics
- [ ] Review grammar changes: verify no flake increase
- [ ] Refresh curriculum weights based on win/loss
- [ ] Check for unresolved contradictions

### 9.3 Monthly Audit

- [ ] Export self-modification changelog
- [ ] Measure impact of changes (before/after metrics)
- [ ] Archive old episodes (> 30 days)
- [ ] Recalculate community clusters
- [ ] Report on knowledge growth rate

---

## 10. Graphiti-Inspired Enhancements

Based on the [Graphiti Integration Research](../Research/graphiti_integration_research.md), enhance the meta-learning loop with:

### 10.1 Temporal Belief Management

```fsharp
type TemporalBelief = {
    Id: Guid
    Statement: string
    Status: EpistemicStatus
    Confidence: float
    // Bi-temporal fields
    ValidFrom: DateTime option   // When the belief became true
    ValidUntil: DateTime option  // When invalidated (None = still valid)
    CreatedAt: DateTime          // When ingested into system
    ExpiredAt: DateTime option   // When superseded by new belief
    // Lineage
    DerivedFrom: Guid list
    InvalidatedBy: Guid option   // What caused invalidation
}
```

### 10.2 Community-Based Pattern Detection

1. **Cluster beliefs** using label propagation
2. **Generate community summaries** for each cluster
3. **Distill grammar** from high-confidence communities
4. **Track community evolution** over time

### 10.3 Contradiction Resolution Pipeline

```
New Belief ──▶ Entity Resolution ──▶ Conflict Detection ──▶ Resolution
                    │                       │                   │
                    │                       │                   ├─▶ InvalidateOld
                    │                       │                   ├─▶ RejectNew
                    │                       │                   ├─▶ Merge
                    │                       │                   └─▶ Scope
                    ▼                       ▼
              Deduplicate           Compare with existing
              similar entities      beliefs via embedding
```

---

## 11. Open Questions & Decisions

### Resolved ✅

| Question | Decision | Rationale |
|----------|----------|-----------|
| Curriculum sampling | Multi-armed bandit (UCB/Thompson) | Heuristic buckets alone are brittle |
| Tool creation triggers | Two signals required | Prevents over-tooling |
| Grammar tightening | Gate behind stability checks | No flake rate increase over N runs |

### Open 🔴

| Question | Options | Blocking? |
|----------|---------|-----------|
| Belief storage backend | VectorStore vs. Neo4j vs. Graphiti | No (start with VS) |
| Community detection algorithm | Label propagation vs. Louvain | No (start with LP) |
| Grammar hot-reload | File watch vs. explicit API | No (start with explicit) |
| Human checkpoint granularity | Per-tool vs. per-batch | Yes (need policy) |

---

## 12. References

- [Epistemic Governor](epistemic_governor.md) - The "Scientist" component
- [Graphiti Integration Research](../Research/graphiti_integration_research.md) - Temporal knowledge graph patterns
- [Grammar-Knowledge Integration](grammar-knowledge-integration.md) - How grammars and knowledge interact
- [BRIDGING_THE_GAPS](03_Operational/BRIDGING_THE_GAPS.md) - Gap analysis and action items
- [K-Theory Integration](k_theory_integration.md) - Mathematical foundations
- [Zep Paper (arXiv:2501.13956v1)](https://arxiv.org/abs/2501.13956) - Graphiti academic reference
