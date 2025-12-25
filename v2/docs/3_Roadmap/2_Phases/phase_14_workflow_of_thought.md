# 🔮 Phase 14: Workflow-of-Thought + Knowledge Graph Integration

**Based on**: ChatGPT recommendations from WoT Integration conversation  
**Status**: 📋 **PLANNED** (Q2-Q3 2025)  
**Priority**: 🔥 **CRITICAL** (foundational for AGI-scale reasoning)

---

## 🎯 Core Vision

**"GoT thinks. WoT acts. The KG remembers. The triple store never forgets."**

TARS should adopt **Workflow-of-Thought as the execution spine**, backed by a **hybrid Knowledge Graph + Triple Store** as its long-term semantic memory.

---

## 📊 The Problem TARS Currently Has

### Current State (Phase 13)
- ✅ Graph-of-Thought (GoT) - Exploration, hypothesis branching
- ✅ Symbolic invariants - Constraint scoring
- ✅ Belief graph - In-memory semantic relationships
- ⚠️ **Missing**: Persistent, auditable, replayable execution workflows
- ⚠️ **Missing**: Durable semantic memory with multi-hop reasoning
- ⚠️ **Missing**: Explainable, traceable decision chains

### The Gap
**GoT is ephemeral. It explores but doesn't persist.**  
**WoT is persistent. It executes, validates, audits, and can be replayed.**

---

## 🏗️ Proposed Architecture

### The Stack (4 Layers)

```
┌─────────────────────────────────────────────────────────┐
│  TARS Neuro-Symbolic Cognitive Stack                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Layer 1: WORKFLOW-OF-THOUGHT (Execution Spine)        │
│  ┌──────────────────────────────────────────┐          │
│  │ Runs .trsx / .flux workflows             │          │
│  │ Produces node graphs + metrics           │          │
│  │ Serializes execution traces              │          │
│  └──────────────────────────────────────────┘          │
│         ↓ executes ↓ reads from ↓ writes to            │
│                                                          │
│  Layer 2: KNOWLEDGE GRAPH (Semantic Cortex)             │
│  ┌──────────────────────────────────────────┐          │
│  │ Property Graph (Neo4j) - Fast traversal   │          │
│  │ RDF Triple Store (Fuseki) - Inference     │          │
│  │ Self-authored knowledge (not Wikipedia)   │          │
│  └──────────────────────────────────────────┘          │
│         ↓ grounds ↓ constrains                          │
│                                                          │
│  Layer 3: REASONING LAYER (Neural + Symbolic)          │
│  ┌──────────────────────────────────────────┐          │
│  │ LLMs (Ollama / llama.cpp) - Propose      │          │
│  │ Symbolic validators (F# DSLs) - Decide   │          │
│  │ Constraint scorers - Judge               │          │
│  └──────────────────────────────────────────┘          │
│         ↓ generates ↓ validates                         │
│                                                          │
│  Layer 4: MEMORY LAYER (Persistent State)              │
│  ┌──────────────────────────────────────────┐          │
│  │ Triple store (facts) - Never forgets     │          │
│  │ Vector store (embeddings) - Pattern match │          │
│  │ PostgreSQL (events) - Event sourcing     │          │
│  │ Versioned filesystem (artifacts)          │          │
│  └──────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Concepts

### 1. Workflow-of-Thought (WoT)

**What it is**: An operational pattern that models reasoning as a directed graph of typed nodes.

**Node Types**:
```fsharp
type WoTNode =
    | ThoughtNode of reasoning: string * confidence: float
    | KnowledgeNode of query: SparqlQuery
    | PolicyNode of constraint: Constraint * justification: string
    | ToolNode of tool: ITool * args: Map<string, obj>
```

**Each node produces**:
- Inputs
- Outputs  
- Confidence / Entropy
- Justification edges  
- Triple deltas (facts learned)

**Maps perfectly to** `.trsx` and `.flux` vision!

### 2. Knowledge Graph

**Not** Wikipedia-style world knowledge.  
**IS** self-authored knowledge.

**What belongs**:
```
Concepts:
  - Agent, Belief, Goal, Grammar, Module, Constraint
  
Relations:
  - depends_on, contradicts, refines, generated_by, validated_by
  
State:
  - belief_confidence, alignment_score, stability_metric
```

**Enables**:
- Multi-hop semantic checks: "Has a similar belief failed before?"
- Agent selection: "Which agent historically resolves this class of contradiction?"  
- Alignment validation: "Is this grammar mutation aligned with prior successful evolutions?"

### 3. Triple Store (RDF)

**Format**: Subject-predicate-object triples
```turtle
(agent_v7) —[proposed]→ (belief_X)
(belief_X) —[contradicts]→ (belief_Y)
(belief_Y) —[introduced_in]→ (run_042)
(run_042) —[
failed_due_to]→ (policy_alignment)
```

**Capabilities**:
- Provenance tracking
- Versioned reasoning
- Explainability
- Inference engines (derive new facts from existing)

---

## 🛠️ Implementation Plan

### Phase 14.1: MVP (Minimal Viable Product)

**Goal**: One complete loop that reads/writes facts

**Deliverables**:
1. ✅ WoT DSL (`.wot.trsx` files)
2. ✅ WoT Engine (execute workflows)
3. ✅ Apache Jena Fuseki (Docker triple store)
4. ✅ SPARQL client (F# wrapper)
5. ✅ Memory deltas (triples per run)
6. ✅ Audit trails (JSON traces)

**Commands**:
```bash
# Initialize triple store
docker-compose up fuseki

# Run workflow
tars run sample.wot.trsx --trace

# Query knowledge
tars kg query "SELECT ?belief WHERE { ?belief tars:confidence ?c . FILTER (?c > 0.9) }"

# Show audit
tars workflow show <run_id>
```

**Test Scenario**:
```
Run #1:
  - Thought proposes Belief("Use grammar contraction") confidence=0.72
  - Policy allows (no prior failures)
  - Upsert: records belief + run_id to triple store

Run #2:
  - Thought proposes same belief
  - Knowledge query finds "previously attempted and failed"
  - Policy blocks (writes failedDueTo tars:PreviouslyFailed)
  
Result: Run #2 changes behavior via memory, not prompts!
```

### Phase 14.2: Think-on-Graph (ToG)

**Goal**: Use KG paths as reasoning constraints

**Approach**:
```fsharp
// Instead of free-form reasoning:
let answer = llm.Complete("Reason about this...")

// Use graph-constrained reasoning:
let paths = kg.FindPaths(subject, predicate, maxDepth=3)
let answer = llm.CompleteWithConstraints("Reason only within these semantic paths:", paths)
```

**Impact**: Massively reduces hallucinations + improves alignment

### Phase 14.3: Policy as Graph

**Goal**: Queryable graph constraints instead of if/else code

**Example**:
```turtle
# Policy: Don't contradict high-confidence beliefs
tars:PolicyNoContradictHighConfidence
  a tars:Policy ;
  rdfs:label "No contradictions with confidence > 0.9" ;
  tars:constraint """
    ASK {
      ?newBelief tars:contradicts ?existing .
      ?existing tars:confidence ?c .
      FILTER (?c > 0.9)
    }
  """ .
```

**Result**: Explainable refusals without moralizing text

### Phase 14.4: Production Hardening

**Deliverables**:
- Graph-validated WoT compilation (invalid workflows won't execute)
- Memory Nodes (every run emits memory delta as triples)
- Rollback capability (revert to previous KG state)
- Multi-agent coordination (shared semantic memory)

---

## 📁 Folder Structure

```
v2/
├── wot/                      # Workflow definitions
│   ├── sample.wot.trsx
│   ├── belief_validation.wot.trsx
│   └── grammar_evolution.wot.trsx
├── memory/                   # Knowledge base
│   ├── schemas/
│   │   └── tars.ttl         # RDF vocabulary
│   └── output/
│       └── versions/<run_id>/
│           ├── wot_trace.json
│           ├── memory_delta.ttl
│           └── metrics.json
├── infra/
│   └── docker-compose.yml   # Fuseki + Neo4j
└── src/
    └── Tars.KnowledgeGraph/
        ├── TripleStoreClient.fs
        ├── WoTEngine.fs
        ├── KnowledgeQueries.fs
        └── PolicyValidator.fs
```

---

## 🎓 Why This Matters

### Current AGI Problem
Most "AI agents" are just LLMs with tool-calling. They have no:
- Persistent memory
- Auditable reasoning
- Explainable decisions
- Self-consistency enforcement

### TARS Solution
```
WoT + KG + Triple Store = Self-Auditing Epistemic Machine

Where:
  - Language proposes
  - Graphs constrain
  - Workflows decide
  - Memory judges
```

**This architecture scales toward superintelligence not by getting "smarter", but by getting less allowed to lie to itself.**

---

## 🔗 Integration with Existing Phases

| Phase | How WoT/KG Improves It |
|-------|------------------------|
| **Phase 8: GoT** | GoT explores, WoT persists successful paths |
| **Phase 9: Multi-Backend Storage** | Triple store becomes 5th backend for semantic facts |
| **Phase 10: 3D Viz** | Visualize KG + WoT execution traces |
| **Phase 11: Grounding** | KG provides grounding facts for ToG |
| **Phase 12: Web of Things** | WoT coordinates IoT agent workflows |
| **Phase 13: Neuro-Symbolic** | Symbolic invariants → KG constraints |

---

## 🚀 Quick Start (When Implemented)

```bash
# 1. Start infrastructure
cd infra
docker-compose up -d fuseki neo4j

# 2. Initialize knowledge base
tars kg init --schema memory/schemas/tars.ttl

# 3. Run a workflow
tars run wot/belief_validation.wot.trsx --verbose

# 4. Query knowledge
tars kg query "SELECT * WHERE { ?s ?p ?o } LIMIT 10"

# 5. Show audit trail
tars workflow list
tars workflow show <run_id> --trace
```

---

## 📚 References

- **ChatGPT Conversation**: `docs/conversations/ChatGPT-TARS WoT Integration.md`
- **Think-on-Graph Paper**: arXiv (Dec 2024)
- **Apache Jena Fuseki**: https://jena.apache.org/documentation/fuseki2/
- **Neo4j**: https://neo4j.com/
- **RDF/SPARQL Primer**: https://www.w3.org/TR/rdf-sparql-query/

---

## ✅ Success Criteria

| Metric | Target |
|--------|--------|
| **WoT Execution** | Can run .wot.trsx files end-to-end |
| **Memory Persistence** | Facts survive restarts |
| **Multi-hop Queries** | Can traverse 3+ semantic hops |
| **Policy Enforcement** | Invalid workflows blocked at compile-time |
| **Audit Trail** | 100% reproducible runs |
| **Hallucination Reduction** | ToG reduces false facts by 80%+ |

---

## 🔮 The Big Insight

**What TARS is becoming**:

Not an LLM system.  
Not a rule engine.  
Not a graph database.

**A self-auditing epistemic machine where**:
- Language proposes
- Graphs constrain  
- Workflows decide
- Memory judges

**The next natural step**: Formalizing WoT as a typed DSL with graph-validated compilation, so **invalid thoughts literally cannot execute**.

That's where the real fun begins. 🧠⚡

---

*Recommendation source: ChatGPT WoT Integration (Dec 2024)*  
*Priority: Critical for AGI-scale reasoning*  
*Timeline: Q2-Q3 2025*
