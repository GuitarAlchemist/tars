# Neuro-Symbolic AI Architecture for TARS

## Executive Summary

TARS is already a **neuro-symbolic AI system** - it just needs to formalize and strengthen the connections. This document outlines how to evolve TARS from an LLM-powered agent system into a true neuro-symbolic cognitive architecture.

> **Core Thesis**: LLMs as stochastic generators + Symbolic systems as memory, law, and self-control = Durable Intelligence

---

## The Neuro-Symbolic Paradigm (2024-2025 State)

### Two Systems (Kahneman's Framework)

| System | Role | TARS Component | Status |
|--------|------|----------------|--------|
| **System 1 (Neural)** | Fast, intuitive pattern recognition | LLM Agents | ✅ Implemented |
| **System 2 (Symbolic)** | Slow, deliberate logical reasoning | Knowledge Ledger + Belief Graph | 🚧 Partial |

### Why This Matters Now

**Problem**: LLMs hallucinate because:
1. Optimized for plausibility, not truth
2. Lack memory of consequences
3. No accountability for logical consistency

**Solution**: Neuro-symbolic architecture where:
1. Neural proposes, symbolic validates
2. Consequences are recorded and learned from
3. Logical consistency is enforced at runtime

---

## Current TARS as Neuro-Symbolic (Mapping)

### ✅ Already Implemented

| Neuro-Symbolic Concept | TARS Implementation | Location |
|------------------------|---------------------|----------|
| **Neural Perception** | LLM Agents (propose, generate, explore) | `Tars.Graph`, `Tars.Evolution` |
| **Episodic Memory** | Episode Store + Graphiti | `Tars.Core/EpisodeStore.fs` |
| **Symbolic Facts** | TemporalKnowledgeGraph | `Tars.Core/TemporalKnowledgeGraph.fs` |
| **Belief Tracking** | Knowledge Ledger with confidence | `Tars.Knowledge/Ledger.fs` |
| **Contradiction Detection** | BeliefGraph consistency checks | `Tars.Knowledge/BeliefGraph.fs` |
| **Provenance** | Evidence sourcing + tracing | `Tars.Knowledge/Types.fs` |
| **Version Control of Cognition** | Agent evolution with versioning | `Tars.Evolution/Engine.fs` |

### 🚧 Partially Implemented

| Neuro-Symbolic Concept | Current State | Gap |
|------------------------|---------------|-----|
| **Logic Tensor Networks** | Constraint scoring exists | Not differentiable/continuous |
| **Probabilistic Logic** | Beliefs have confidence scores | No uncertainty propagation |
| **Symbolic Validation** | EpistemicGovernor checks beliefs | Not enforced at generation time |
| **Reward Signals** | Success/failure tracked | Not fed back into agent behavior |
| **Formal Invariants** | Grammar validity checks | Not systematically enforced |

### ❌ Missing (2025 State-of-Art)

| Technique | Purpose | Implementation Needed |
|-----------|---------|----------------------|
| **Differentiable Constraints** | Shape neural generation in real-time | Constraint-aware prompt shaping |
| **Belief Stability Metrics** | Quantify logical consistency | Aggregate rule satisfaction scoring |
| **Agent Constitutions** | Formal contracts for agent behavior | Per-agent symbolic guardrails |
| **Symbolic Reflection** | Structured belief updates | Replace text reflection with formal updates |
| **Penalty-Based Learning** | Avoid historical contradictions | Contradiction memory + biasing |

---

## Concrete Neuro-Symbolic Roadmap

### Phase 13: Neuro-Symbolic Foundations (Q1 2025)

**Goal**: Formalize symbolic invariants and close the neural-symbolic feedback loop

#### 13.1 Symbolic Invariants System
```fsharp
// Define formal invariants for TARS
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
      Evidence: string list }
```

**Implementation**:
- Create `src/Tars.Symbolic/Invariants.fs`
- Define core invariants for grammar, beliefs, alignment
- Implement checking functions that return continuous scores

#### 13.2 Constraint Scoring Engine
```fsharp
// Logic Tensor Network-style constraint scoring
module ConstraintScoring =
    /// Score how well a belief satisfies symbolic constraints
    let scoreBeliefConsistency (belief: EvidenceBelief) (existing: EvidenceBelief list) : float =
        let contradictions = 
            existing 
            |> List.filter (fun b -> contradicts belief b)
            |> List.length
        
        // Continuous score: 1.0 (no contradictions) to 0.0 (many contradictions)
        1.0 / (1.0 + float contradictions * 0.5)
    
    /// Aggregate multiple constraint scores
    let aggregateConstraints (checks: InvariantCheck list) : float =
        let weights = Map.ofList [...]  // Per-invariant weights
        checks 
        |> List.map (fun c -> weights.[c.Invariant] * c.Score)
        |> List.average
```

**Purpose**: Replicate Logic Tensor Networks without tensors - pure functional constraint scoring

#### 13.3 Belief Stability Metric
```fsharp
type BeliefStability =
    { Belief: EvidenceBelief
      ConsistencyScore: float  // Internal consistency
      EvidenceStrength: float  // Quality of supporting evidence
      TemporalStability: float // How long it's been stable
      ConflictCount: int       // Number of contradicting beliefs
      OverallStability: float } // Weighted aggregate

module BeliefStabilityOps =
    let calculate (belief: EvidenceBelief) (graph: BeliefGraph) : BeliefStability =
        let consistency = ConstraintScoring.scoreBeliefConsistency belief graph.Beliefs
        let evidence = belief.Evidence.Length |> float |> fun x -> min 1.0 (x / 5.0)
        let temporal = ...  // Based on belief age and edit history
        let conflicts = graph.GetContradictions belief |> List.length
        
        { Belief = belief
          ConsistencyScore = consistency
          EvidenceStrength = evidence
          TemporalStability = temporal
          ConflictCount = conflicts
          OverallStability = (consistency * 0.4 + evidence * 0.3 + temporal * 0.3) }
```

**Purpose**: Probabilistic logic-style uncertainty tracking

#### 13.4 Close the Feedback Loop
```fsharp
// Scores affect agent behavior
type AgentSelectionBias =
    { AgentId: AgentId
      HistoricalStability: float  // Past belief stability from this agent
      ConflictRate: float          // How often it proposes contradictions
      SelectionProbability: float } // Adjusted spawn probability

module NeuralSymbolicFeedback =
    /// Bias agent selection based on symbolic performance
    let biasAgentSelection (agents: Agent list) (history: BeliefHistory) : Agent =
        let biases = 
            agents 
            |> List.map (fun a ->
                let stability = history.GetStabilityForAgent a.Id
                let conflicts = history.GetConflictRate a.Id
                { AgentId = a.Id
                  HistoricalStability = stability
                  ConflictRate = conflicts
                  SelectionProbability = stability * (1.0 - conflicts) })
        
        // Weighted random selection
        selectWeighted biases agents
    
    /// Shape prompts based on contradiction history
    let shapeprompt (basePrompt: string) (context: BeliefContext) : string =
        let knownPitfalls = context.GetHistoricalContradictions()
        let warnings = 
            knownPitfalls 
            |> List.map (fun c -> $"AVOID: {c.Pattern} (caused {c.Count} contradictions)")
        
        $"{basePrompt}\n\nSymbolic Constraints:\n{String.concat "\n" warnings}"
```

**Purpose**: Differentiable glue - symbolic scores shape neural behavior


### Phase 14: Agent Constitutions (Q2 2025)

**Goal**: Every agent has a neural role + symbolic contract

#### 14.1 Formal Agent Contracts
```fsharp
type AgentConstitution =
    { AgentId: AgentId
      NeuralRole: NeuralRole
      SymbolicContract: SymbolicContract
      Invariants: SymbolicInvariant list
      Permissions: Permission list
      Prohibitions: Prohibition list }

and NeuralRole =
    | Generate of domain: AgentDomain
    | Explore of searchSpace: string
    | Summarize of contentType: string
    | Mutate of target: MutationTarget
    | Review of aspect: ReviewAspect

and SymbolicContract =
    { MustPreserve: Invariant list      // "Must not break these"
      MustAchieve: Goal list              // "Must accomplish these"
      ResourceBounds: ResourceLimit list  // "Cannot exceed these"
      DependsOn: AgentId list            // "Requires these agents first"
      ConflictsWith: AgentId list }       // "Cannot run alongside these"

// Example: Grammar Evolution Agent
let grammarEvolutionContract =
    { AgentId = AgentId("grammar-evolver")
      NeuralRole = Mutate(GrammarRules)
      SymbolicContract =
        { MustPreserve = 
            [ ParseCompleteness  // All existing parses still work
              BackwardCompatibility ]  // Old rules still valid
          MustAchieve =
            [ ReduceComplexity 10  // Simplify by >10%
              MaintainCoverage ]     // Don't lose edge cases
          ResourceBounds =
            [ MaxIterations 100
              MaxTokens 50000 ]
          DependsOn = [ AgentId("grammar-analyzer") ]
          ConflictsWith = [] }
      Invariants = [ GrammarValidity(...) ]
      Permissions = [ ModifyGrammar; ReadKnowledgeGraph ]
      Prohibitions = [ DeleteProductions; ModifyCore ] }
```

#### 14.2 Runtime Contract Enforcement
```fsharp
module ContractEnforcement =
    /// Check if an agent action violates its contract
    let validateAction (agent: Agent) (action: AgentAction) : Result<unit, Violation> =
        let contract = agent.Constitution.SymbolicContract
        
        // Check prohibitions
        match action with
        | ModifyCode path when contract.Prohibitions |> List.contains (CannotModifyPath path) ->
            Error (Violation.ProhibitionViolated("Cannot modify " + path))
        | _ -> ()
        
        // Check invariants
        let invariantChecks = 
            agent.Constitution.Invariants
            |> List.map (fun inv -> checkInvariant inv action)
        
        match invariantChecks |> List.tryFind (fun c -> not c.Satisfied) with
        | Some failed -> Error (Violation.InvariantBroken(failed))
        | None -> Ok ()
    
    /// Enforce contract before agent spawns
    let guardedSpawn (agent: Agent) (task: Task) : Result<Agent, string> =
        // Check dependencies
        if not (dependenciesMet agent.Constitution.DependsOn) then
            Error "Dependencies not satisfied"
        
        // Check resource bounds
        if exceedsResourceBounds task agent.Constitution then
            Error "Task exceeds agent resource limits"
        
        Ok agent
```

**Purpose**: Agents become governed cognitive modules, not just "LLM personalities"

### Phase 15: Symbolic Reflection (Q3 2025)

**Goal**: Reflection becomes structured belief update, not text

#### 15.1 Structured Reflection
```fsharp
// Replace text-based reflection
type SymbolicReflection =
    { ReflectionId: Guid
      Timestamp: DateTime
      Trigger: ReflectionTrigger
      Observations: Observation list
      BeliefUpdates: BeliefUpdate list
      Justifications: Justification list
      ImpactScore: float }

and Observation =
    | ContradictionDetected of belief1: EvidenceBelief * belief2: EvidenceBelief
    | PatternRecognized of pattern: CodePatternEntity
    | AnomalyFound of anomaly: AnomalyEntity
    | PerformanceMetric of metric: string * value: float

and BeliefUpdate =
    | AddBelief of EvidenceBelief
    | RevokeBelief of beliefId: Guid * reason: string
    | AdjustConfidence of beliefId: Guid * newConfidence: float * reason: string
    | ResolveContradiction of resolution: ConflictResolution

and Justification =
    { UpdateId: Guid
      Evidence: Evidence list
      ReasoningChain: ReasoningStep list
      SymbolicProof: SymbolicProof option }

// Example: Agent reflects on failed code generation
let reflectOnFailure (task: TaskResult) (agent: Agent) : SymbolicReflection =
    { ReflectionId = Guid.NewGuid()
      Timestamp = DateTime.UtcNow
      Trigger = TaskFailure(task.TaskId, task.Output)
      Observations =
        [ ContradictionDetected(
            priorBelief = "Binary search requires sorted input",
            newBelief = "Unsorted input was accepted") ]
      BeliefUpdates =
        [ AdjustConfidence(
            beliefId = priorBeliefId,
            newConfidence = 0.95,  // Still mostly true
            reason = "Edge case in sorting assumption") ]
      Justifications =
        [ { UpdateId = ...
            Evidence = [ CodeTrace task.ExecutionTrace ]
            ReasoningChain = [ ... ]
            SymbolicProof = Some (ValidationFailure "input_not_sorted") } ]
      ImpactScore = 0.75 }
```

**Purpose**: Reflection generates formal knowledge, not natural language

#### 15.2 Belief Revision Engine
```fsharp
module BeliefRevision =
    /// Apply symbolic reflection to update knowledge graph
    let applyReflection (reflection: SymbolicReflection) (graph: BeliefGraph) : BeliefGraph =
        reflection.BeliefUpdates
        |> List.fold (fun g update ->
            match update with
            | AddBelief b -> 
                g.AddBelief b reflection.Justifications
            | RevokeBelief(id, reason) ->
                g.RevokeBelief id reason reflection.Timestamp
            | AdjustConfidence(id, conf, reason) ->
                g.UpdateConfidence id conf reason
            | ResolveContradiction res ->
                g.ResolveConflict res) graph
    
    /// Check if reflection is consistent with existing beliefs
    let validateReflection (reflection: SymbolicReflection) (graph: BeliefGraph) : Result<unit, Conflict> =
        let proposedUpdates = reflection.BeliefUpdates
        let currentBeliefs = graph.GetAllBeliefs()
        
        // Check for new contradictions
        proposedUpdates
        |> List.tryFind (fun update ->
            match update with
            | AddBelief b -> currentBeliefs |> List.exists (contradicts b)
            | _ -> false)
        |> Option.map (fun conflict -> Error (NewContradiction conflict))
        |> Option.defaultValue (Ok ())
```

---

## Implementation Priority

### Immediate (Phase 13 - Q1 2025)
1. **Constraint Scoring** - Core neuro-symbolic glue
2. **Belief Stability Metrics** - Measure what matters
3. **Feedback Loop** - Symbolic scores → Agent selection

### Short-term (Phase 14 - Q2 2025)
4. **Agent Constitutions** - Formalize contracts
5. **Contract Enforcement** - Runtime guardrails

### Medium-term (Phase 15 - Q3 2025)
6. **Symbolic Reflection** - Structured belief updates
7. **Belief Revision Engine** - Formal knowledge updates

---

## Success Metrics

### Hallucination Reduction
- **Before**: LLM generates plausible but false statements
- **After**: Symbolic validation rejects ~90%+ of hallucinations before output

### Belief Stability
- **Before**: No measure of consistency
- **After**: Track stability score (0.0-1.0) and trending

### Agent Governance
- **Before**: Agents can do anything LLM allows
- **After**: Agents constrained by formal contracts, violations logged

### Reflection Quality
- **Before**: Text reflection ("I should try harder")
- **After**: Structured belief updates with evidence chains

---

## Why TARS is Uniquely Positioned

1. **Already Has the Architecture**: Episodic memory + belief graph + versioning
2. **Functional Foundation**: Pure F# makes constraint logic natural
3. **Evolution Engine**: Can self-improve under symbolic constraints
4. **Provenance Tracking**: Every belief traces back to source
5. **Phase 9 Ledger**: PostgreSQL+JSONB ready for complex symbolic queries

---

## Real-World Precedent

| Company/Project | Approach | TARS Equivalent |
|----------------|----------|-----------------|
| **Amazon** | Neural perception + symbolic safety rules | Phase 13-14 |
| **DeepMind AlphaGeometry** | Neural intuition + symbolic proof | Symbolic reflection (Phase 15) |
| **MIT-IBM Watson** | Neuro-symbolic program synthesis | Evolution under constraints |
| **Stanford LAION** | LLMs + formal verification | EpistemicGovernor + contracts |

**TARS Difference**: Applies neuro-symbolic to *thinking itself*, not just actions.

---

## Philosophical Foundation

> Intelligence without constraint is imagination.  
> Constraint without imagination is bureaucracy.  
> **Intelligence is the tension between them.**

TARS embodies this tension:
- **Neural (System 1)**: Proposes, generates, explores (imagination)
- **Symbolic (System 2)**: Validates, remembers, enforces (constraint)
- **Evolution**: Learns from the interplay

---

## Next Actions

1. **Create `src/Tars.Symbolic`** project for invariants and constraint scoring
2. **Extend `BeliefGraph`** with stability metric calculation
3. **Add agent constitution** to `Agent` type in `Domain.fs`
4. **Implement feedback loop** in evolution engine
5. **Replace text reflection** with `SymbolicReflection` type

---

## References

- Logic Tensor Networks (LTNs): Semantic-based regularization for learning and inference
- DeepProbLog: Neural probabilistic logic programming
- Neurosymbolic AI: The 3rd Wave (Artur d'Avila Garcez, 2020)
- Thinking, Fast and Slow (Daniel Kahneman, 2011)
- Amazon's Kiva: Neural perception with symbolic safety layer
- TARS Architectural Vision: `docs/1_Vision/architectural_vision.md`

---

**Status**: 🚧 Roadmap defined, implementation begins Phase 13  
**Priority**: HIGH - Core differentiator for production AI systems  
**Risk**: LOW - Builds on existing foundation, no new dependencies
