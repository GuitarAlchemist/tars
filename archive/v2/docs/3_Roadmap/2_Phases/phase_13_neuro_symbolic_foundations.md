# Phase 13: Neuro-Symbolic Foundations

**Timeline**: Q1 2025 (January - March)  
**Status**: 🔜 Planned  
**Priority**: HIGH - Core differentiator  
**Dependencies**: Phase 9 (Knowledge Ledger)

---

## Vision

Transform TARS from an LLM-powered system into a true neuro-symbolic AI by formalizing the feedback loop between neural generation and symbolic validation.

**Core Innovation**: Symbolic constraints shape neural behavior in real-time, not just validate after the fact.

---

## Objectives

1. **Formalize symbolic invariants** for grammar, beliefs, and alignment
2. **Implement constraint scoring** (Logic Tensor Network-style, no tensors needed)
3. **Calculate belief stability metrics** (probabilistic logic approach)
4. **Close the feedback loop** - symbolic scores affect agent behavior

---

## Components

### 13.1 Symbolic Invariants System

**Goal**: Define formal invariants that TARS must preserve

**Deliverables**:
- `src/Tars.Symbolic/Invariants.fs` - Core invariant types and checking
- `src/Tars.Symbolic/Tars.Symbolic.fsproj` - New project

**Types**:
```fsharp
type SymbolicInvariant =
    | GrammarValidity of rule: string * production: string
    | BeliefConsistency of beliefSet: EvidenceBelief list
    | AlignmentThreshold of metric: string * min: float
    | CodeComplexityBound of maxComplexity: float
    | ResourceQuota of resource: string * limit: int
    | TemporalConstraint of before: TarsEntity * after: TarsEntity

type InvariantCheck =
    { Invariant: SymbolicInvariant
      Satisfied: bool
      Score: float  // 0.0 (violated) to 1.0 (satisfied)
      Evidence: string list
      Timestamp: DateTime }
```

**Implementation Tasks**:
- [ ] Create `Tars.Symbolic` project
- [ ] Define core invariant types
- [ ] Implement checking functions for each invariant type
- [ ] Add continuous scoring (not just boolean pass/fail)
- [ ] Write unit tests for invariant checks

**Success Criteria**:
- All invariants testable with examples
- Scores are continuous (0.0-1.0), not binary
- Fast evaluation (<10ms per invariant)

---

### 13.2 Constraint Scoring Engine

**Goal**: Replicate Logic Tensor Networks without tensors

**Deliverables**:
- `src/Tars.Symbolic/ConstraintScoring.fs` - Scoring algorithms
- `src/Tars.Symbolic/Aggregation.fs` - Weighted aggregation

**Core Functions**:
```fsharp
module ConstraintScoring =
    /// Score belief consistency (no contradictions = 1.0)
    val scoreBeliefConsistency : belief:EvidenceBelief -> existing:EvidenceBelief list -> float
    
    /// Score grammar validity (parses = 1.0)
    val scoreGrammarValidity : rule:string -> production:string -> float
    
    /// Score alignment with goals (perfect alignment = 1.0)
    val scoreAlignment : action:AgentAction -> goals:Goal list -> float
    
    /// Aggregate multiple constraint scores with weights
    val aggregateConstraints : checks:InvariantCheck list -> weights:Map<SymbolicInvariant, float> -> float
```

**Implementation Tasks**:
- [ ] Implement scoring for each invariant type
- [ ] Create weighted aggregation function
- [ ] Add configurable weights per invariant
- [ ] Profile performance (target: <5ms per score)
- [ ] Write property-based tests

**Success Criteria**:
- Scores correlate with human judgment (>80% agreement)
- Fast enough for real-time use during generation
- Deterministic (same input → same score)

---

### 13.3 Belief Stability Metric

**Goal**: Quantify logical consistency over time (probabilistic logic style)

**Deliverables**:
- `src/Tars.Knowledge/BeliefStability.fs` - Stability calculation
- Extended `BeliefGraph` with stability tracking

**Type**:
```fsharp
type BeliefStability =
    { Belief: EvidenceBelief
      ConsistencyScore: float   // How consistent with other beliefs
      EvidenceStrength: float   // Quality of supporting evidence
      TemporalStability: float  // How long it's been stable
      ConflictCount: int        // Number of contradicting beliefs
      OverallStability: float } // Weighted aggregate

module BeliefStabilityOps =
    val calculate : belief:EvidenceBelief -> graph:BeliefGraph -> BeliefStability
    val getStablest : graph:BeliefGraph -> count:int -> EvidenceBelief list
    val getUnstable : graph:BeliefGraph -> threshold:float -> EvidenceBelief list
```

**Implementation Tasks**:
- [ ] Implement stability calculation
- [ ] Add temporal tracking to BeliefGraph
- [ ] Create stability trending (getting better/worse)
- [ ] Add PostgreSQL schema for stability history
- [ ] Visualize stability in diagnostics

**Success Criteria**:
- Stability scores identify problematic beliefs
- Temporal trends show improvement over time
- Integration with knowledge ledger for persistence

---

### 13.4 Neural-Symbolic Feedback Loop

**Goal**: Symbolic scores affect neural behavior (the key innovation!)

**Deliverables**:
- `src/Tars.Evolution/NeuralSymbolicFeedback.fs` - Feedback mechanisms
- Updated `AgentSelection` with symbolic biasing
- Updated `PromptShaping` with constraint warnings

**Mechanisms**:
```fsharp
module NeuralSymbolicFeedback =
    /// Bias agent selection based on historical stability
    val biasAgentSelection : agents:Agent list -> history:BeliefHistory -> Agent
    
    /// Shape prompts to avoid known contradictions
    val shapePrompt : basePrompt:string -> context:BeliefContext -> string
    
    /// Filter mutations that violate invariants
    val filterMutations : mutations:Mutation list -> invariants:SymbolicInvariant list -> Mutation list
    
    /// Select agents based on belief-stability impact
    val selectForStability : agents:Agent list -> graph:BeliefGraph -> Agent list
```

**Implementation Tasks**:
- [ ] Implement weighted agent selection (not random)
- [ ] Add contradiction warnings to prompts
- [ ] Filter mutations by invariant satisfaction
- [ ] Track agent → stability correlation
- [ ] Add feedback metrics to evolution reports

**Success Criteria**:
- Agents with high stability get selected more often
- Prompts include relevant warnings (reduces contradictions by >30%)
- Mutations that violate invariants are rejected
- Evolution runs show improving stability over time

---

## Implementation Order

### Week 1-2: Foundation
1. Create `Tars.Symbolic` project
2. Implement invariant types and basic checks
3. Write initial tests

### Week 3-4: Scoring
4. Implement constraint scoring engine
5. Add weighted aggregation
6. Performance tuning

### Week 5-6: Stability
7. Implement belief stability calculation
8. Integrate with BeliefGraph
9. Add PostgreSQL persistence

### Week 7-8: Feedback Loop
10. Implement agent selection biasing
11. Add prompt shaping
12. Mutation filtering

### Week 9-10: Integration
13. Wire into evolution engine
14. Add metrics and reporting
15. End-to-end testing

### Week 11-12: Validation
16. Run evolution experiments
17. Measure hallucination reduction
18. Document findings

---

## Success Metrics

### Quantitative

| Metric | Baseline | Target |
|--------|----------|--------|
| **Hallucination Rate** | ~20% of beliefs contradicted | <5% |
| **Belief Stability** | No measure | >0.8 avg |
| **Constraint Violations** | No tracking | <2% of actions |
| **Scoring Performance** | N/A | <5ms per check |

### Qualitative

- [ ] Agents avoid known contradiction patterns
- [ ] Evolution runs show increasing stability trend
- [ ] Prompts include relevant symbolic warnings
- [ ] Mutations respect formal invariants

---

## Dependencies

**Required**:
- ✅ Phase 9 (Knowledge Ledger) - Belief tracking
- ✅ Phase 6 (Evolution) - Agent selection

**Nice-to-have**:
- Phase 10 (3D Visualization) - Visualize stability
- Phase 11 (Cognitive Grounding) - External validation

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scoring too slow | Agent selection blocked | Profile early, optimize critical paths |
| False positives | Good ideas rejected | Tune weights via experimentation |
| Complexity explosion | Hard to maintain | Keep invariants simple, well-documented |
| Unclear benefit | Low adoption | Measure and report metrics prominently |

---

## Deliverables

### Code
- [ ] `src/Tars.Symbolic/` - New project (5-7 files)
- [ ] `src/Tars.Knowledge/BeliefStability.fs`
- [ ] `src/Tars.Evolution/NeuralSymbolicFeedback.fs`
- [ ] 50+ unit tests
- [ ] 10+ integration tests

### Documentation
- [x] This phase document
- [ ] API documentation (XML comments)
- [ ] User guide on invariants
- [ ] Evolution metrics dashboard

### Research
- [ ] Hallucination reduction report
- [ ] Belief stability case studies
- [ ] Comparison to pure LLM baseline

---

## References

- Logic Tensor Networks (Serafini & Garcez, 2016)
- Neurosymbolic AI: The 3rd Wave (Garcez et al., 2020)
- Probabilistic Logic Programming (De Raedt et al., 2007)
- TARS Architectural Vision: `docs/1_Vision/architectural_vision.md`
- Neuro-Symbolic Architecture Plan: `docs/3_Roadmap/1_Plans/neuro_symbolic_architecture.md`

---

**Next Phase**: [Phase 14: Agent Constitutions](./phase_14_agent_constitutions.md)
