# Phase 15: Symbolic Reflection

**Timeline**: Q3 2025 (July - September)  
**Status**: 🔜 Planned  
**Priority**: HIGH - Learning mechanism  
**Dependencies**: Phase 13 (Neuro-Symbolic Foundations), Phase 14 (Agent Constitutions)

---

## Vision

Transform agent reflection from text generation ("I should try harder") into **structured belief updates** with formal justification chains.

**Core Shift**: Reflection is not natural language - it's a formal knowledge update operation.

---

## Objectives

1. **Replace text reflection** with structured `SymbolicReflection` type
2. **Implement belief revision engine** for formal knowledge updates
3. **Create evidence chains** linking beliefs to their sources
4. **Build symbolic proof system** for validation

---

## Components

### 15.1 Structured Reflection Types

**Goal**: Reflection generates formal knowledge, not text

**Deliverables**:
- `src/Tars.Core/SymbolicReflection.fs` - Reflection types
- `src/Tars.Core/Observation.fs` - Observation types
- Migration from text-based reflection

**Types**:
```fsharp
type SymbolicReflection =
    { ReflectionId: Guid
      Timestamp: DateTime
      AgentId: AgentId
      Trigger: ReflectionTrigger
      Observations: Observation list
      BeliefUpdates: BeliefUpdate list
      Justifications: Justification list
      ImpactScore: float
      Confidence: float }

and ReflectionTrigger =
    | TaskCompleted of taskId: Guid * result: TaskResult
    | TaskFailed of taskId: Guid * error: string
    | ContradictionDetected of belief1: EvidenceBelief * belief2: EvidenceBelief
    | InvariantViolated of invariant: SymbolicInvariant * action: AgentAction
    | ResourceExhausted of resource: string
    | PatternRecognized of pattern: CodePatternEntity
    | AnomalyFound of anomaly: AnomalyEntity
    | EpistemicConflict of beliefs: EvidenceBelief list

and Observation =
    | ContradictionFound of source: EvidenceBelief * target: EvidenceBelief * reason: string
    | PatternDetected of pattern: CodePatternEntity * confidence: float
    | AnomalyDetected of anomaly: AnomalyEntity * severity: AnomalySeverity
    | PerformanceMetric of metric: string * value: float * trend: Trend
    | BehaviorChange of agent: AgentId * before: Behavior * after: Behavior
    | ResourceTrend of resource: string * usage: float list
    | SuccessPattern of action: AgentAction * outcome: Outcome * frequency: int

and BeliefUpdate =
    | AddBelief of belief: EvidenceBelief * evidence: Evidence list
    | RevokeBelief of beliefId: Guid * reason: string * replacement: EvidenceBelief option
    | AdjustConfidence of beliefId: Guid * newConfidence: float * reason: string
    | ResolveContradiction of resolution: ConflictResolution
    | MergeBeliefs of beliefs: EvidenceBelief list * merged: EvidenceBelief
    | SplitBelief of original: EvidenceBelief * refined: EvidenceBelief list

and Justification =
    { UpdateId: Guid
      Update: BeliefUpdate
      Evidence: Evidence list
      ReasoningChain: ReasoningStep list
      SymbolicProof: SymbolicProof option
      Confidence: float }

and ReasoningStep =
    | AssumptionMade of assumption: string * basis: Evidence
    | InferenceMade of from: EvidenceBelief list * to: EvidenceBelief
    | CounterevidenceConsidered of evidence: Evidence * weight: float
    | AlternativeRejected of alternative: EvidenceBelief * reason: string
```

**Implementation Tasks**:
- [ ] Define complete reflection type hierarchy
- [ ] Create observation extractors (from task results, contradictions, etc.)
- [ ] Build belief update generators
- [ ] Implement justification builders
- [ ] Add reflection schema to PostgreSQL
- [ ] Migrate existing text reflections

**Example Reflection**:
```fsharp
// Agent reflects on failed sorting task
let reflection = {
    ReflectionId = Guid.NewGuid()
    Timestamp = DateTime.UtcNow
    AgentId = AgentId("task-executor")
    Trigger = TaskFailed(taskId, "Assertion failed: output not sorted")
    Observations = [
        ContradictionFound(
            source = "Binary search requires sorted input",
            target = "Input validation accepts unsorted arrays",
            reason = "Precondition mismatch"
        )
    ]
    BeliefUpdates = [
        AdjustConfidence(
            beliefId = priorBeliefId,
            newConfidence = 0.95,  // Still mostly true
            reason = "Edge case in sorting assumption - validation too permissive"
        )
        AddBelief(
            belief = "Input validation should enforce sorting precondition",
            evidence = [
                CodeTrace executionTrace
                TestFailure failedTest
            ]
        )
    ]
    Justifications = [
        { UpdateId = ...
          Update = AddBelief(...)
          Evidence = [ CodeTrace; TestFailure ]
          ReasoningChain = [
            AssumptionMade("Binary search correctness", BinarySearchProof)
            InferenceMade(
                from = [ "Binary search requires sorted input" ],
                to = "Validation must check sorting"
            )
          ]
          SymbolicProof = Some(ValidationFailure "input_not_sorted")
          Confidence = 0.92 }
    ]
    ImpactScore = 0.85  // High impact - prevents future failures
    Confidence = 0.90   // High confidence in the reflection
}
```

**Success Criteria**:
- All reflections are structured (zero text-only)
- Observations are machine-parseable
- Belief updates are formally defined
- Justifications form logical chains

---

### 15.2 Belief Revision Engine

**Goal**: Apply symbolic reflections to update knowledge graph

**Deliverables**:
- `src/Tars.Knowledge/BeliefRevision.fs` - Revision engine
- Conflict resolution strategies
- Belief merging/splitting logic

**Core Functions**:
```fsharp
module BeliefRevision =
    /// Apply reflection to update knowledge graph
    val applyReflection : reflection:SymbolicReflection -> graph:BeliefGraph -> Result<BeliefGraph, RevisionError>
    
    /// Validate reflection doesn't create new contradictions
    val validateReflection : reflection:SymbolicReflection -> graph:BeliefGraph -> Result<unit, Conflict list>
    
    /// Resolve contradiction using resolution strategy
    val resolveContradiction : strategy:ResolutionStrategy -> beliefs:EvidenceBelief list -> EvidenceBelief
    
    /// Merge similar beliefs into one
    val mergeBeliefs : beliefs:EvidenceBelief list -> similarity:float -> EvidenceBelief option
    
    /// Split overly general belief into specific ones
    val splitBelief : belief:EvidenceBelief -> refinements:EvidenceBelief list -> EvidenceBelief list
```

**Revision Strategies**:

1. **AddBelief Revision**:
   - Check for contradictions with existing beliefs
   - If contradictory → trigger resolution strategy
   - If compatible → add with appropriate confidence

2. **RevokeBelief Revision**:
   - Remove belief from graph
   - Update dependent beliefs (cascade)
   - Log revocation reason for audit

3. **AdjustConfidence Revision**:
   - Update belief confidence
   - If confidence < threshold → auto-revoke
   - Track confidence trends

4. **ResolveContradiction Revision**:
   - Apply resolution strategy (highest confidence wins, merge, split)
   - Update both beliefs or create new merged belief
   - Log resolution decision

**Implementation Tasks**:
- [ ] Implement each belief update type
- [ ] Add conflict detection during revision
- [ ] Build resolution strategy system
- [ ] Create belief merging algorithm
- [ ] Add belief splitting algorithm
- [ ] Implement cascade updates
- [ ] Track revision history

**Success Criteria**:
- All reflection types can be applied
- Contradictions are detected before commit
- Resolution strategies are selectable
- Revision history is auditable

---

### 15.3 Evidence Chains

**Goal**: Every belief links back to its evidence sources

**Deliverables**:
- `src/Tars.Knowledge/EvidenceChain.fs` - Chain building
- Evidence graph visualization
- Chain verification

**Types**:
```fsharp
type EvidenceChain =
    { Belief: EvidenceBelief
      DirectEvidence: Evidence list
      InferredFrom: InferenceStep list
      TransitiveEvidence: Evidence list  // Indirect via inferences
      SourceTraces: SourceTrace list
      ConfidenceFlow: ConfidenceStep list }

and InferenceStep =
    { From: EvidenceBelief list
      Rule: InferenceRule
      To: EvidenceBelief
      Confidence: float }

and SourceTrace =
    | OriginatesFrom of url: Uri * timestamp: DateTime
    | ExtractedBy of agent: AgentId * method: ExtractionMethod
    | InferredFrom of beliefs: EvidenceBelief list
    | ValidatedBy of validator: AgentId * test: ValidationTest

module EvidenceChains =
    /// Build complete evidence chain for belief
    val buildChain : belief:EvidenceBelief -> graph:BeliefGraph -> EvidenceChain
    
    /// Verify chain has no broken links
    val verifyChain : chain:EvidenceChain -> Result<unit, BrokenLink list>
    
    /// Find weakest link in chain
    val findWeakestLink : chain:EvidenceChain -> (Evidence * float)  // (evidence, confidence)
    
    /// Trace belief back to original sources
    val traceToSources : belief:EvidenceBelief -> graph:BeliefGraph -> SourceTrace list
```

**Chain Visualization**:
```
Belief: "Binary search requires sorted input"
├─ Direct Evidence:
│  ├─ Wikipedia: "Algorithm correctness relies on sorted property"
│  └─ Test: binary_search_unsorted_fails.fs
├─ Inferred From:
│  └─ "Search algorithms have preconditions" + "Binary search is a search algorithm"
└─ Confidence Flow:
   ├─ Wikipedia source: 0.95
   ├─ Test evidence: 1.0
   └─ Inference: 0.90
   → Overall: 0.95 (min of critical path)
```

**Implementation Tasks**:
- [ ] Implement chain building (BFS/DFS through graph)
- [ ] Add chain verification
- [ ] Build weakest link detection
- [ ] Create source tracing
- [ ] Add confidence flow calculation
- [ ] Implement chain visualization (ASCII/DOT)
- [ ] Add CLI: `tars belief trace <belief-id>`

**Success Criteria**:
- Every belief has traceable evidence chain
- Chains are verifiable (no broken links)
- Easy to find weakest evidence
- Visualization aids understanding

---

### 15.4 Symbolic Proof System

**Goal**: Formal validation of belief updates

**Deliverables**:
- `src/Tars.Symbolic/ProofSystem.fs` - Proof checker
- Proof templates for common patterns
- Integration with reflection

**Types**:
```fsharp
type SymbolicProof =
    | Tautology of statement: string
    | Contradiction of statement: string
    | ValidationSuccess of test: ValidationTest * result: TestResult
    | ValidationFailure of test: ValidationTest * error: string
    | LogicalInference of premises: EvidenceBelief list * conclusion: EvidenceBelief * rule: InferenceRule
    | StatisticalEvidence of samples: int * successRate: float * confidence: float
    | ExpertAssertion of source: string * credibility: float

and InferenceRule =
    | ModusPonens          // If P and P→Q, then Q
    | ModusTollens         // If P→Q and ¬Q, then ¬P
    | Syllogism            // If P→Q and Q→R, then P→R
    | Contraposition       // P→Q ≡ ¬Q→¬P
    | Generalization       // P(x) for all observed x → ∀x P(x)
    | Specialization       // ∀x P(x) → P(c) for specific c

module ProofSystem =
    /// Verify proof is valid
    val verifyProof : proof:SymbolicProof -> Result<unit, ProofError>
    
    /// Check if proof supports belief
    val supports : proof:SymbolicProof -> belief:EvidenceBelief -> bool
    
    /// Calculate proof strength (0.0-1.0)
    val strengthOf : proof:SymbolicProof -> float
    
    /// Find all proofs supporting belief
    val findProofs : belief:EvidenceBelief -> graph:BeliefGraph -> SymbolicProof list
```

**Implementation Tasks**:
- [ ] Define proof types
- [ ] Implement proof verification for each type
- [ ] Build inference rule checker
- [ ] Create proof templates
- [ ] Add proof search (find supporting proofs)
- [ ] Calculate proof strength
- [ ] Integrate with belief revision

**Success Criteria**:
- Common inferences have proof templates
- Proofs are verifiable
- Proof strength correlates with belief confidence
- Invalid proofs are rejected

---

## Implementation Order

### Week 1-2: Types
1. Define SymbolicReflection types
2. Create observation extractors
3. Build belief update generators

### Week 3-4: Revision
4. Implement belief revision engine
5. Add conflict resolution strategies
6. Build cascade updates

### Week 5-6: Evidence
7. Implement evidence chain building
8. Add chain verification
9. Create visualization

### Week 7-8: Proofs
10. Define proof system types
11. Implement proof verification
12. Build inference rule checker

### Week 9-10: Integration
13. Wire into agent workflow
14. Replace text reflections
15. Add reflection metrics

### Week 11-12: Validation
16. Test with real agent reflections
17. Measure belief stability improvement
18. Document case studies

---

## Success Metrics

### Quantitative

| Metric | Baseline | Target |
|--------|----------|--------|
| **Structured Reflections** | 0% | 100% |
| **Belief Update Success** | N/A | >95% |
| **Evidence Chain Completeness** | Unknown | >90% |
| **Proof Verification Rate** | N/A | 100% |

### Qualitative

- [ ] Reflections produce actionable insights
- [ ] Belief updates improve system performance
- [ ] Evidence chains aid debugging
- [ ] Proofs increase confidence in beliefs

---

## Dependencies

**Required**:
- ✅ Phase 13 (Belief Stability) - Stability tracking
- ✅ Phase 14 (Constitutions) - Contract violations trigger reflection
- ✅ Phase 9 (Knowledge Ledger) - Belief storage

**Nice-to-have**:
- Phase 10 (3D Visualization) - Visualize evidence chains

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Too complex for agents | Low adoption | Provide high-level reflection API |
| Proof system incomplete | Limited validation | Start with common patterns, expand |
| Performance overhead | Slow reflection | Profile and optimize critical paths |
| Hard to migrate | Old reflections lost | Keep text fallback during transition |

---

## Deliverables

### Code
- [ ] `src/Tars.Core/SymbolicReflection.fs`
- [ ] `src/Tars.Knowledge/BeliefRevision.fs`
- [ ] `src/Tars.Knowledge/EvidenceChain.fs`
- [ ] `src/Tars.Symbolic/ProofSystem.fs`
- [ ] 80+ unit tests
- [ ] 20+ integration tests

### Documentation
- [x] This phase document
- [ ] Reflection authoring guide
- [ ] Evidence chain guide
- [ ] Proof system reference

### Tools
- [ ] CLI: `tars reflect <trigger>`
- [ ] CLI: `tars belief trace <id>`
- [ ] CLI: `tars proof verify <proof-id>`

---

## References

- Belief Revision Theory (Alchourrón, Gärdenfors, & Makinson, 1985)
- Truth Maintenance Systems (Doyle, 1979)
- Justification-Based Truth Maintenance (McAllester, 1990)
- TARS Neuro-Symbolic Plan: `docs/3_Roadmap/1_Plans/neuro_symbolic_architecture.md`

---

**Previous Phase**: [Phase 14: Agent Constitutions](./phase_14_agent_constitutions.md)  
**Future Phases**: TBD (Foundation complete for production neuro-symbolic AI)
