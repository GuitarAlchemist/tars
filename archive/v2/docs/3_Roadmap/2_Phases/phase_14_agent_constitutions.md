# Phase 14: Agent Constitutions

**Timeline**: Q2 2025 (April - June)  
**Status**: 🔜 Planned  
**Priority**: HIGH - Governance layer  
**Dependencies**: Phase 13 (Neuro-Symbolic Foundations)

---

## Vision

Transform agents from "LLM personalities" into **governed cognitive modules** with formal contracts defining what they can and cannot do.

**Core Idea**: Every agent has a neural role (what it does) + symbolic contract (what it's allowed to do).

---

## Objectives

1. **Define formal agent contracts** (permissions, prohibitions, invariants)
2. **Implement runtime contract enforcement** (validate before execution)
3. **Create constitution templates** for common agent patterns
4. **Build contract violation detection** and logging

---

## Components

### 14.1 Formal Agent Contracts

**Goal**: Specify what each agent can/cannot do in machine-readable format

**Deliverables**:
- `src/Tars.Core/AgentConstitution.fs` - Contract types
- Constitution schema (JSON/YAML)
- Constitution validator

**Types**:
```fsharp
type AgentConstitution =
    { AgentId: AgentId
      NeuralRole: NeuralRole
      SymbolicContract: SymbolicContract
      Invariants: SymbolicInvariant list
      Permissions: Permission list
      Prohibitions: Prohibition list
      ResourceBounds: ResourceLimit list }

and NeuralRole =
    | Generate of domain: AgentDomain
    | Explore of searchSpace: string
    | Summarize of contentType: string
    | Mutate of target: MutationTarget
    | Review of aspect: ReviewAspect
    | Coordinate of agents: AgentId list

and SymbolicContract =
    { MustPreserve: Invariant list      // Cannot break these
      MustAchieve: Goal list              // Must accomplish these
      ResourceBounds: ResourceLimit list  // Cannot exceed these
      Dependencies: AgentId list          // Requires these first
      ConflictsWith: AgentId list         // Cannot run alongside
      TimeConstraints: TimeConstraint list }

and Permission =
    | ReadKnowledgeGraph
    | ModifyKnowledgeGraph
    | ReadCode of pattern: string
    | ModifyCode of pattern: string
    | SpawnAgent of agentType: string
    | CallTool of toolName: string
    | AccessSecret of secretName: string

and Prohibition =
    | CannotModifyCore
    | CannotDeleteData
    | CannotAccessNetwork
    | CannotSpawnUnlimited
    | CannotExceedBudget
    | CannotViolateInvariant of SymbolicInvariant
```

**Implementation Tasks**:
- [ ] Define contract type system
- [ ] Create JSON schema for constitutions
- [ ] Implement constitution parser (JSON → F# types)
- [ ] Build constitution validator
- [ ] Add constitution editor CLI (`tars constitution edit <agent-id>`)
- [ ] Create standard templates (10+ common patterns)

**Example Constitution** (Grammar Evolution Agent):
```fsharp
let grammarEvolutionConstitution =
    { AgentId = AgentId("grammar-evolver")
      NeuralRole = Mutate(GrammarRules)
      SymbolicContract = {
        MustPreserve = [
          ParseCompleteness       // All existing parses still work
          BackwardCompatibility   // Old rules still valid
        ]
        MustAchieve = [
          ReduceComplexity 10%    // Simplify by >10%
          MaintainCoverage        // Don't lose edge cases
        ]
        ResourceBounds = [
          MaxIterations 100
          MaxTokens 50000
          MaxTimeMinutes 30
        ]
        Dependencies = [ AgentId("grammar-analyzer") ]
        ConflictsWith = [ AgentId("grammar-loader") ]  // Cannot run together
        TimeConstraints = [ MustCompleteWithin 30<minute> ]
      }
      Invariants = [ GrammarValidity ]
      Permissions = [
        ReadKnowledgeGraph
        ModifyCode "grammars/*.ebnf"
      ]
      Prohibitions = [
        CannotModifyCore
        CannotDeleteData
        CannotSpawnUnlimited
      ]
      ResourceBounds = [
        MaxMemoryMB 512
        MaxCpuPercent 50
        MaxDiskWritesMB 10
      ] }
```

**Success Criteria**:
- All agent types have constitution templates
- Constitutions are human-readable (JSON/YAML)
- Validation catches 100% of structural errors
- Easy to create new agent constitutions

---

### 14.2 Runtime Contract Enforcement

**Goal**: Validate agent actions before they execute

**Deliverables**:
- `src/Tars.Core/ContractEnforcement.fs` - Enforcement engine
- Contract violation detection
- Agent action guard

**Core Functions**:
```fsharp
module ContractEnforcement =
    /// Check if action violates agent's contract
    val validateAction : agent:Agent -> action:AgentAction -> Result<unit, Violation>
    
    /// Check if agent can spawn given current state
    val canSpawn : agent:Agent -> state:SystemState -> Result<unit, string>
    
    /// Check dependencies before spawning
    val checkDependencies : agent:Agent -> activeAgents:Agent list -> Result<unit, string>
    
    /// Track resource usage
    val trackResources : agent:Agent -> usage:ResourceUsage -> Result<unit, QuotaExceeded>
    
    /// Log violation for audit
    val logViolation : agent:Agent -> violation:Violation -> unit
```

**Enforcement Points**:
1. **Pre-spawn**: Check dependencies, conflicts, resource availability
2. **Pre-action**: Validate permissions for each tool call/mutation
3. **During execution**: Track resource usage against bounds
4. **Post-completion**: Verify invariants still hold

**Implementation Tasks**:
- [ ] Implement validateAction for all permission types
- [ ] Add dependency graph resolution
- [ ] Track resource usage (memory, CPU, tokens)
- [ ] Create violation logging (to knowledge ledger)
- [ ] Build enforcement middleware for agent execution
- [ ] Add enforcement metrics to telemetry

**Example Enforcement**:
```fsharp
// Before agent acts
let enforceContract (agent: Agent) (action: AgentAction) =
    match validateAction agent action with
    | Error (Violation.ProhibitionViolated msg) ->
        logViolation agent (Violation.ProhibitionViolated msg)
        Error $"Action blocked: {msg}"
    | Error (Violation.InvariantBroken inv) ->
        logViolation agent (Violation.InvariantBroken inv)
        Error $"Would violate invariant: {inv}"
    | Error (Violation.ResourceQuotaExceeded res) ->
        logViolation agent (Violation.ResourceQuotaExceeded res)
        Error $"Resource limit exceeded: {res}"
    | Ok () ->
        Ok ()  // Proceed with action
```

**Success Criteria**:
- 100% of prohibited actions are blocked
- Resource bounds enforced with <5% margin
- Violations logged with full context
- Zero false positives in standard workflows

---

### 14.3 Constitution Templates

**Goal**: Provide ready-made constitutions for common agent patterns

**Deliverables**:
- `constitutions/templates/` directory (15+ JSON templates)
- Template documentation
- CLI for instantiating templates

**Standard Templates**:

1. **Code Generator Agent**
   - Can: Read code, write new files, call LLM
   - Cannot: Modify existing core files, delete files
   - Must: Pass all tests, maintain code coverage

2. **Code Reviewer Agent**
   - Can: Read code, comment, call static analysis
   - Cannot: Modify code directly
   - Must: Find all P0/P1 issues

3. **Grammar Evolver Agent**
   - Can: Modify grammar files, test parsing
   - Cannot: Break existing parses
   - Must: Reduce complexity by X%

4. **Belief Extractor Agent**
   - Can: Call LLM, read Wikipedia, write to ledger
   - Cannot: Revoke existing beliefs
   - Must: Achieve >0.7 confidence

5. **Mutation Agent**
   - Can: Propose code changes, spawn testers
   - Cannot: Apply changes directly
   - Must: Preserve all test passes

6. **Coordinator Agent**
   - Can: Spawn other agents, allocate resources
   - Cannot: Exceed total budget
   - Must: Complete workflow in time limit

**Implementation Tasks**:
- [ ] Create 15+ JSON templates
- [ ] Document each template (when to use, parameters)
- [ ] Add CLI: `tars constitution create --template=<name> --agent-id=<id>`
- [ ] Build template validation
- [ ] Add template customization guide

**Success Criteria**:
- 80% of common agents use templates (not custom)
- Templates cover all major use cases
- Easy to customize (< 5 min per template)

---

### 14.4 Violation Detection & Audit

**Goal**: Comprehensive logging and analysis of contract violations

**Deliverables**:
- Violation schema in PostgreSQL
- Violation query API
- Violation dashboard

**Schema**:
```sql
CREATE TABLE agent_violations (
    id UUID PRIMARY KEY,
    agent_id TEXT NOT NULL,
    violation_type TEXT NOT NULL,  -- 'prohibition' | 'invariant' | 'resource'
    description TEXT NOT NULL,
    action_attempted JSONB,
    timestamp TIMESTAMPTZ NOT NULL,
    consequence TEXT,  -- 'blocked' | 'warned' | 'logged'
    stack_trace TEXT
);
```

**Queries**:
```fsharp
module ViolationQueries =
    /// Get all violations for agent
    val getByAgent : agentId:AgentId -> Violation list
    
    /// Get violations by type
    val getByType : violationType:ViolationType -> Violation list
    
    /// Get frequent violations (top 10)
    val getFrequent : count:int -> (Violation * int) list
    
    /// Get recent violations (last N)
    val getRecent : count:int -> Violation list
```

**Implementation Tasks**:
- [ ] Create PostgreSQL schema
- [ ] Implement violation DAO
- [ ] Add violation queries
- [ ] Build CLI: `tars violations ls --agent=<id>`
- [ ] Create violation rate metrics
- [ ] Add to diagnostics dashboard

**Success Criteria**:
- All violations logged with context
- Easy to query violations by agent/type/time
- Violation rates tracked over time
- Actionable insights (which agents violate most)

---

## Implementation Order

### Week 1-2: Contracts
1. Define contract type system
2. Create JSON schema
3. Implement parser and validator

### Week 3-4: Templates
4. Create 15+ standard templates
5. Build template CLI
6. Document usage

### Week 5-6: Enforcement
7. Implement validateAction
8. Add dependency checking
9. Build resource tracking

### Week 7-8: Integration
10. Wire enforcement into agent execution
11. Add pre/post checks
12. Test with evolution engine

### Week 9-10: Violations
13. Create PostgreSQL schema
14. Implement violation logging
15. Build query API

### Week 11-12: Validation
16. Run agents with constitutions
17. Measure violation rates
18. Tune enforcement rules

---

## Success Metrics

### Quantitative

| Metric | Baseline | Target |
|--------|----------|--------|
| **Agents with Constitutions** | 0% | >80% |
| **Prohibited Actions Blocked** | N/A | 100% |
| **False Positives** | N/A | <2% |
| **Violation Rate** | Unknown | <5 per 100 actions |

### Qualitative

- [ ] Agents respect formal boundaries
- [ ] Easy to create new constitutions
- [ ] Violations provide clear guidance
- [ ] Audit trail for compliance

---

## Dependencies

**Required**:
- ✅ Phase 13 (Symbolic Invariants) - Invariant checking
- ✅ Phase 6 (Evolution) - Agent spawning

**Nice-to-have**:
- Phase 15 (Symbolic Reflection) - Learn from violations

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Too restrictive | Agents can't do useful work | Start permissive, tighten gradually |
| Too complex | Hard to write constitutions | Good templates + documentation |
| Performance overhead | Slows agent execution | Profile and optimize hot paths |
| False positives | Blocks valid actions | Careful tuning with real scenarios |

---

## Deliverables

### Code
- [ ] `src/Tars.Core/AgentConstitution.fs`
- [ ] `src/Tars.Core/ContractEnforcement.fs`
- [ ] `constitutions/templates/` (15+ JSON files)
- [ ] 60+ unit tests
- [ ] 15+ integration tests

### Documentation
- [x] This phase document
- [ ] Constitution authoring guide
- [ ] Template catalog
- [ ] Enforcement rules reference

### Infrastructure
- [ ] PostgreSQL violation schema
- [ ] CLI commands for constitution management
- [ ] Metrics dashboard

---

## References

- Agent Constitutions (Anthropic Constitutional AI, 2022)
- Formal Methods for AI Safety (Russell & Norvig, 2021)
- Capability-based Security (Miller et al., 2003)
- TARS Neuro-Symbolic Plan: `docs/3_Roadmap/1_Plans/neuro_symbolic_architecture.md`

---

**Previous Phase**: [Phase 13: Neuro-Symbolic Foundations](./phase_13_neuro_symbolic_foundations.md)  
**Next Phase**: [Phase 15: Symbolic Reflection](./phase_15_symbolic_reflection.md)
