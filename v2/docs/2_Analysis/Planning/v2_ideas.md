# TARS v2 Architecture Ideas

> **Source**: Chat history from ChatGPT discussions on TARS v2 design
>
> **Date**: November 21, 2025

## Table of Contents

- [Overview](#overview)
- [Synthesized Feedback](#synthesized-feedback)
- [TARS v2.1 Practical Architecture Plan](#tars-v21-practical-architecture-plan)
- [Language Choice: Staying in F#](#language-choice-staying-in-f)
- [FLUX Engine: Universal Metaprogramming](#flux-engine-universal-metaprogramming)
- [Graphiti Integration](#graphiti-integration)
- [Erlang/OTP Concepts for TARS](#erlangotp-concepts-for-tars)
- [Detailed Examples](#detailed-examples)

---

## Overview

This document captures key architectural ideas and design decisions for TARS v2, synthesizing feedback from multiple AI assistants (ChatGPT and Gemini) to create a practical, evolvable, and safe agentic platform.

### Core Principles

1. **F#-First Core**: Leverage F#'s type system as a "compiler guardian" for safety
2. **Polyglot Periphery**: Support skills in any language via clean protocols
3. **FLUX Metaprogramming**: Language-agnostic code generation and transformation
4. **Erlang-Inspired Resilience**: Supervision trees, message passing, and "let it crash" philosophy
5. **Incremental Evolution**: Build in phases, defer advanced features to later versions

---

## Delta From `pre_v2` Review

Concrete updates folded in so the branch visibly diverges from `pre_v2` while keeping the original intent intact.

### Immutable Surfaces and Change Control

- Immutable core: `Tars.Kernel` contracts, `Tars.Cortex` cognitive interfaces, shared discriminated unions for `Message`/`AgentState`/`SkillResult`.
- Enforcement: code owners + ADRs + signed releases for immutable packages; all changes flow through the evolution gate (below) with rollback playbooks.

### Evolution Gate (Pass/Fail Rubric)

1. **Proposal** with owner + rationale.
2. **Static checks**: formatting, analyzer, exhaustive pattern-matching.
3. **Tests**: unit/integration + replay harness where available.
4. **Sandbox run**: containerized/process-isolated execution with budget caps.
5. **Metrics review**: latency/cost/constraint adherence vs baseline; auto rollback trigger defined up front.
6. **Canary** (if long-lived service) before promotion; lineage + metrics artifacts persisted.

### Skill Manifest Defaults (per skill)

- Fields: `name`, `version`, `isolation` (`process`/`docker`/`wasm`), `cpu_limit`, `memory_limit_mb`, `wall_clock_timeout_ms`, `network_policy` (`none`/`egress-allowlist`), `audit_level` (inputs/outputs/metrics), `log_path`.
- Default budget: 1 vCPU share, 256MB RAM, 30s wall clock, no external network unless allowlisted.

### Memory Plan and Migration

- Belief graph: SQLite schema with explicit version column on nodes/edges; migrations must ship with backfill scripts + compatibility matrix.
- Vector store: simple folder-based shim now; promote to vector DB once size/latency thresholds are exceeded, with checksum-backed backfill plan.

### Cortex Defaults and Fallbacks

- Deterministic-first defaults: `seed = 0`, `temperature = 0.1`, grammar-constrained requests where possible.
- Fallback chain: local runner → remote provider; all prompts/responses logged with request id, seed, grammar id, and timing.

### Minimal Graphiti/FLUX Slice for v2

- One ingestion/query path wired end-to-end; one FLUX metascript transforming a Graphiti subgraph into grammar rules stored via `IGrammarStore`.
- Grammars versioned alongside metascripts; audit log records the source graph snapshot and grammar id.

### Supervision Mapping Example

- Use `MailboxProcessor` or `System.Threading.Channels` per agent; parent/child links propagate cancellation/timeouts; restarts follow Erlang-style strategy (one-for-one) with exponential backoff.

### Ops/CI Baseline

- CI jobs: build, formatting, unit tests, and replay harness where present.
- Packaging: devcontainer or Dockerfile with pinned SDK/CLI versions to make the gate reproducible.

---

## V1 Component Reusability

### Vector Store: InMemoryVectorStore (High Reuse)

**Status:** ✅ **Recommended for v2 with simplification**

**V1 Location:** `v1/src/TarsEngine.FSharp.Core/VectorStore/Core/`

**Reusability Assessment:** ~70% direct code reuse, 2-3 hours adaptation effort

**What makes it reusable:**

- Already implements file-based JSON persistence (exactly what v2 Phase 3 needs)
- Clean `IVectorStore` interface with 8 well-defined methods
- Thread-safe with `ConcurrentDictionary` for in-memory caching
- Async API patterns throughout
- Solid CRUD operations and batch utilities

**Simplification required:**

- Remove `MultiSpaceEmbedding` complexity (9 mathematical spaces: Raw, FFT, Dual, Projective, Hyperbolic, Wavelet, Minkowski, Pauli, Belief)
- Use simple `float[]` vectors with metadata only
- Remove tetravalent logic and quantum-like operations
- Simplify similarity computation to cosine similarity only
- Defer advanced features (hyperbolic embeddings, etc.) to v3+

**V2 additions needed:**

- Checksum generation/validation per document
- Version tracking for schema evolution
- Migration metadata (status, target, progress)
- Migration trigger thresholds (10K docs or 500ms latency)

**Migration path:**

- Interface-based design allows swapping to Qdrant/Pinecone/Chroma without changing consumers
- File-based storage provides easy backfill for external vector DB migration

**Alignment with v2 principles:**

- ✅ Small, local, pragmatic
- ✅ Easy to evolve later
- ✅ Defers complexity to future phases
- ✅ Battle-tested code from v1

---

## Synthesized Feedback

### What Gemini Got Right

ChatGPT provided an ambitious, elegant architecture, while Gemini injected production realism. Here's what to take seriously:

#### 1. Avoiding Over-Engineering

**Complexity traps to defer to v3+:**

- Hyperbolic vector stores
- Full isolation sandboxing
- Deep triple-store integration
- Advanced math layers too early

These belong in the research branch, not the mainline.

#### 2. F# as the Safety Mechanism

> **Key Insight**: Your greatest safety mechanism isn't Docker or sandboxes—it's the F# compiler itself.

**Weaponize F# features:**

- **Discriminated Unions** → irrefutable states
- **Units of Measure** → no unit confusion for budgets/resources
- **Pattern matching exhaustiveness** → no unhandled transitions

The compiler becomes:

- Your safety net
- Your constraint engine
- Your guardian

#### 3. Containerized Skills for Isolation

ChatGPT's "sandbox runner" idea is elegant but not realistic in .NET. You need:

- **Docker**
- **WASM**
- **Separate OS processes** (strict IPC)

Otherwise one bad mutation freezes TARS's heart.

### What Gemini Got Wrong

#### 1. Hyperbolic Embeddings Aren't Useless

**What's true:**

- They will slow you down right now
- They require specialized libs
- You need V1 agents working first

**What's false:**

- That they shouldn't exist in your long-term trajectory
- They belong in TARS v3 or v4, not V2

**Why they matter:**
Hyperbolic embeddings are perfect for hierarchical agent memory.

#### 2. Triple-Store Removal Too Aggressive

Cut from early versions, but not from the architecture.

**Triple stores are uniquely suited for:**

- Belief graphs
- Agent lineage
- Provenance
- Temporal logic

They belong in **Phase 5 or later**, right when agents learn to critique each other.

---

## TARS v2.1 Practical Architecture Plan

This integrates the best of:

- Your architecture vision
- ChatGPT's refinement
- Gemini's cautionary realism

**Result**: Shippable, lean, future-proof, F#-native, evolvable

### Phase 0: Hard Skeleton (No Intelligence Yet)

**Deliverables:**

Full solution layout:

- `Tars.Kernel`
- `Tars.Cortex`
- `Tars.Memory`
- `Tars.Agents`
- `Tars.Skills`
- `Tars.Observability`
- `Tars.Interface`

**Define core types (signatures only):**

- `Message` (DU)
- `AgentState` (DU)
- `SkillResult` (DU or `Result<,>`)
- `Budget` (with F# Units of Measure)

**Why F# Matters Here:**
If the AI later tries to mutate one of these types incorrectly, you get a compiler error. This is your firewall.

### Phase 1: Minimal Kernel + Single Agent

Start with:

- EventBus using Channels
- One agent: `EchoAgent`
- No LLM
- No vector store
- No belief graph
- Just message routing and lifecycle

This is your "heartbeat" test.

### Phase 2: Add Cortex with Local LLM + Grammar Distillation

**Crucial simplification:**

- Use Ollama or LM Studio
- Only text-in / text-out at first
- Grammar distillation via JSON schema or GBNF
- No fancy reasoning loops yet
- No hyperbolic embeddings
- No triple stores

**You only need:**

- `ICognitiveProvider`
- `CognitivePlan`
- `GrammarConstraint`

This gets TARS thinking.

### Phase 3: Memory Grid (Simplified Version)

**Keep it simple:**

- SQLite for belief graph (node, edge tables)
- A simple folder-based vector store (`embedding.json` per document)
- Add fancy spatial geometry later

**Memory in V2 is:**

- Small
- Local
- Pragmatic
- Easy to evolve later

**Vector Store Implementation Strategy:**

Reuse v1's `InMemoryVectorStore` with simplification:

**✅ What to keep from v1:**

- `IVectorStore` interface (8 methods - perfect as-is)
- File-based JSON persistence (one file per document)
- `ConcurrentDictionary` for in-memory caching
- Async API patterns
- Search, filter, and CRUD operations
- Batch operations and utilities

**❌ What to simplify/remove:**

- `MultiSpaceEmbedding` complexity (9 different mathematical spaces)
- Hyperbolic embeddings (defer to v3+)
- FFT/Wavelet/Minkowski/Pauli matrices (defer to v3+)
- Tetravalent logic and quantum-like operations
- Complex similarity computers

**✅ What to add for v2:**

- Checksum generation/validation per document
- Version tracking for schema evolution
- Migration metadata (status, target, progress)
- Simple cosine similarity only

**Estimated effort:** 2-3 hours to adapt v1 code (~70% direct reuse)

**Migration path:** Interface-based design allows swapping to Qdrant/Pinecone/Chroma later without changing consumers

### Phase 4: Skills (But Safe)

> **Critical Rule**: Do NOT load mutated .dlls into the main process.

**Adopt this rule:**
All skills run in isolated workers. One of:

- WASM
- Docker
- Separate OS processes

**Prefer WASM** for deterministic behavior.

**Define:**

- `ISkillHost`
- `SkillManifest`
- `SkillInvocation`

Very strict type boundaries.

### Phase 5: Observability Tower

**You need to see:**

- Agentic traces
- Reasoning chains
- Memory diffs
- Skill invocations

Format everything in JSON + line-delimited logs. No UI yet.

**Simplest shape possible:**

```
runs/
  <timestamp>/
    agentic_trace.json
    memory_before.json
    memory_after.json
    metrics.json
    skills.json
```

This is where future TARS learns from its past.

### Phase 6: Controlled Self-Modification

**Gemini critique is crucial:**

- Use the F# type system as a guardian
- Use containerized builds for mutated code
- Never touch Kernel, Cortex, or basic types in early evolution

**Only allow self-modification of:**

- Skills
- DSL transformations
- Agents' internal strategies

**Implement:**

- `ModificationRequest`
- `CandidateChange`
- Test harness project
- Sandbox runner
- Metrics comparison

### Phase 7: Multi-Agent Protocols

**Add:**

- Negotiation protocol
- Proposals
- Counter-proposals
- Consensus graphs
- Contradictions

This is where the project starts becoming emergent.

### Phase 8+: What Gets Delayed to TARS v3+

Everything Gemini flagged is correctly postponed:

- Hyperbolic embeddings
- Triple stores
- Sedenions & exotic math DSLs
- Self-mutating grammars
- Closure factories generating hot F# functions
- Full TARS "cultural memory" and exchange across machines

These come after V2 is stable.

---

## Language Choice: Staying in F #

### Short Answer

**Yes**, stay in F# for TARS's core—but design it so you can lean on other ecosystems (Python, C#, etc.) at the edges.

### What TARS Actually Needs From Its Core Language

TARS core (Kernel + Cortex + Memory model + Evolution pipeline) needs:

1. A strong, expressive type system (for "compiler as guardian")
2. Great pattern matching & DUs (for agent states, messages, grammars)
3. Deterministic behavior (for evolution, replay, and debugging)
4. Good concurrency (for agents & event bus)
5. Long-term maintainability by you (solo/very small team)

### F# is Extremely Good At

- Discriminated unions for `Message`, `AgentState`, `SkillResult`, etc.
- Making illegal states unrepresentable (huge for self-modifying systems)
- Exhaustive pattern matching → evolution can't silently introduce gaps
- Sticking "budget" into units of measure so you literally can't mix tokens/ms/etc by accident

All of that directly reinforces the "self-evolving but safe" vision.

### Where F# Hurts You

**Ecosystem gap vs Python:**

- Vector DB clients, RAG frameworks, eval tools, model-serving utilities—all show up in Python first

**Less community content to steal from:**

- Most examples/recipes for LLM agents are in Python/TypeScript

**Interop friction:**

- Some things will be "just easier" as a Python script or a small Rust/C tool

### The Sensible Strategy: F# Core, Polyglot Periphery

Instead of "F# vs not F#", think:

- **F# = OS kernel**
- **Other languages = userland programs**

#### Keep in F #

- `Tars.Kernel` (agent runtime, event bus, budgets, safety)
- `Tars.Cortex` core abstractions (planning, grammar constraints, model routing)
- `Tars.Memory` high-level API + belief graph representation
- Evolution pipeline logic (what's allowed to change, lineage, scoring)

These are the parts where correctness and invariants matter more than library convenience.

#### Push Out of F #

**Individual skills:**

- Python skill that calls a vector DB / runs HF pipelines
- Rust/C++ skill for heavy compute
- Shell / WASM skills for simple utilities

**Tooling around TARS:**

- Log visualizers
- Dev CLIs/UX
- Experiment notebooks

As long as the protocol between core and skills is clean (e.g., JSON over stdio / gRPC / HTTP), you can happily mix ecosystems.

### Concrete Next Steps If We Commit to F #

1. **Explicitly codify "compiler as guardian" in the spec:**
   - Which modules are immutable
   - Which types are the "safety shell"

2. **Define a language-agnostic skill protocol:**
   - e.g., external skills = stdin/stdout JSON with a manifest
   - So writing a Python or Rust skill is trivial and safe

3. **Pick one Python-based "power skill" early:**
   - e.g., `python-vector-skill` that:
     - speaks to Qdrant / Chroma / FAISS
     - exposes "search", "upsert", "schema" to TARS core

4. **Keep TARS's public APIs neutral:**
   - HTTP/gRPC interfaces that any language can call
   - That way you're never locked into "F# everywhere"

---

## FLUX Engine: Universal Metaprogramming

### FLUX = Language-Agnostic Metaprogramming Layer

If the Cortex is the "mind," FLUX is the meta-mind—a way for TARS to:

- Generate code
- Transform code
- Refactor code
- Optimize code
- Reason about code

**In any language**, without hard-coding support for each language inside TARS itself.

### Why FLUX Is the Universal Layer

Because FLUX isn't tied to:

- Syntax rules
- Compiler APIs
- Language semantics

Instead, it works at the level of:

- ASTs
- Transformation patterns
- Rewrite rules
- Structured code grammars

**Meaning:**

- A FLUX metascript for Python → generates Python
- A FLUX metascript for Rust → generates Rust
- A FLUX metascript for C#, F#, Java → same pipeline
- A FLUX metascript for DSLs → also works

Your engine only needs:

- A grammar
- A parser
- A pretty-printer
- A transformation model

The FLUX runtime stays the same.

### This Gives TARS Polyglot Superpowers

#### 1. The Kernel stays in F #

Strong typing, compiler safety, deterministic behavior—perfect for the "OS layer."

#### 2. Cortex + FLUX can generate code in any language

Want a Python skill? FLUX writes it.
Want a Rust vector-search skill? FLUX writes it.
Want a Go networking skill? FLUX writes it.
Want to evolve a grammar? FLUX writes that too.

#### 3. Skills become language-agnostic

Because a skill is just:

- A manifest
- A protocol
- An executable script/module/container

TARS doesn't care whether it's:

- Python
- Rust
- F#
- WASM
- Bash
- Node.js

All it cares about is: **"Does this skill follow the protocol and return structured output?"**

### This Lets You Stay in F# and Be Polyglot

You don't have to choose.

**Your architecture becomes:**

- **F#** = kernel + cognitive control + safety + invariants
- **FLUX** = cross-language metaprogramming + code synthesis
- **Skills** = sandboxed polyglot modules

This is the cleanest possible design for an evolving agent platform.

---

## Graphiti Integration

### Why Graphiti Helps Grammar Distillation

[Graphiti](https://github.com/getzep/graphiti) is essentially a code understanding + structural extraction engine with:

- AST parsing
- Graph representations of code
- Cross-file and cross-module linking
- Structured queries
- Semantic relationships between code elements

That makes it a perfect "front end" for grammar extraction.

### Grammar Distillation Needs 3 Things

1. **Readable structure**
   - Graphiti gives parsed ASTs and semantic graphs

2. **Pattern extraction**
   - Graphiti can find repeated constructs or idioms

3. **Rule formation / summarization**
   - FLUX + Cortex can turn those patterns into grammar rules

Graphiti essentially becomes a **code analyzer → grammar generator**, while FLUX becomes a **grammar refiner → code synthesizer**.

### How Graphiti Integrates Into TARS

**The clean pipeline:**

1. **Graphiti reads some code** (F#, Python, Rust, whatever)
   - Produces: syntax tree, semantic graph, relationships (calls, imports, definitions, scopes)

2. **TARS Cortex → asks FLUX to interpret this structure**
   - FLUX converts the graph into: candidate grammar rules, idiom templates, pattern signatures, transformation rules

3. **Cortex evaluates and merges rules across many sources**
   - This becomes: generalized grammar distillation, cross-language grammar inference, style extraction, "meta-patterns" for code evolution

4. **Skills can then use these grammars to:**
   - Rewrite code
   - Optimize scripts
   - Generate idiomatic new code
   - Analyze foreign codebases
   - Build safe mutations
   - Generate multi-language equivalents

This is exactly the kind of recursive, evolvable meta-pipeline TARS is designed for.

### Where Graphiti Fits in the Architecture

**Module mapping:**

- **`Tars.Skills.GraphitiSkill`**
  - A Python or Rust-based skill that:
    - runs Graphiti on code
    - returns AST + semantic graph JSON
    - provides list of constructs, dependencies, scopes

- **`Tars.Cortex.Grammar.Distillation`**
  - Consumes Graphiti's output:
    - identifies patterns
    - converts structure into grammar fragments
    - feeds FLUX metascripts for rule creation

- **`Tars.Cortex.FLUXRuntime`**
  - Turns extracted patterns into:
    - rewrite rules
    - DSL definitions
    - language-agnostic transformations

- **`Tars.Kernel.Evolution`**
  - Uses these grammars to:
    - safely mutate code
    - generate new skills
    - refactor agent strategies
    - enforce consistency across languages

**This forms a loop:**

```
Code → Graph → Patterns → Grammar → Transformations → Improved Code
```

A literal "self-improving compiler ecosystem."

### What This Enables (Major Capabilities)

✅ **Automatic grammar distillation from ANY code**

- Even unfamiliar languages or messy repos

✅ **Cross-language grammar unification**

- F# idioms → Python equivalents → Rust equivalents

✅ **Metaprogramming grounded in real data**

- TARS evolves grammars based on actual code, not hallucination

✅ **Safety via structural constraints**

- Because transformations happen through AST rules, not raw text

✅ **Autonomous refactoring**

- Agents can propose refactors justified by real structural insight

✅ **Data-driven evolution of FLUX metascripts**

- Better grammars → better transformations → better grammars

---

## Erlang/OTP Concepts for TARS

TARS is begging for some OTP brain juice. Erlang gave us the best production story for "lots of tiny, semi-intelligent things talking over messages and failing all the time," which is exactly what you're building.

### 1. Supervision Trees → Agent & Skill Supervision

**Erlang idea:** Every worker process has a supervisor; supervisors form a tree. They don't prevent failure, they structure it.

**TARS translation:**

`Kernel.Supervision` module that manages:

- Agent supervisors (per agent type / per group)
- Skill supervisors (per class of skill)
- Each agent run (conversation / job) is a supervised "process"
- Evolution engine runs under its own strict supervisor with tighter policies

**Why it matters for AI:**

If a self-modified skill or agent "goes weird" (loop, nonsense output, violation), you:

- Kill that instance
- Log the trace
- Optionally roll back that version
- Keep the rest of the system alive

Lets you be bold with self-modification but contained in blast radius.

### 2. "Let It Crash" → "Let It Hallucinate (in a Box)"

**Erlang idea:** Don't defensively code every edge case; allow local failure and recover via supervisors.

**TARS translation:**

Don't over-micro-manage every LLM call or skill mutation.

Instead:

- Give each agent/skill strict budgets (tokens, time, retries)
- If they exceed or produce invalid outputs → treat as a crash
- Supervisor decides:
  - Restart with fresh context
  - Downgrade model (cheaper / safer)
  - Escalate to human or stricter agent

For evolution:

- A failed mutated skill is just a bad "offspring"; mark as unfit and move on

**Why it matters:**
Makes self-evolving behavior safe and cheap to try, instead of paralyzing you with "we must prove it's perfect."

### 3. Mailboxes & Message Passing → EventBus + Per-Agent Queues

**Erlang idea:** Each process has a mailbox; inter-process communication is pure message passing.

**TARS translation:**

`Kernel.EventBus` is global, but each agent has an inbox:

- Queue of messages (requests, tool results, negotiation proposals, etc.)

Allow:

- Selective receive semantics (agent can prioritize certain message types)
- Priorities / deadlines (user messages > background chatter)

Backpressure:

- Mailbox size limits
- Drop policies (oldest, lowest-priority, or "reject incoming" with error)
- Metrics for overloaded agents

**Why it matters:**

- Keeps agents decoupled
- Lets you evolve / replace one agent implementation without changing everyone else
- Makes concurrency intuitive: agents = Erlang processes

### 4. Behaviours (gen_server, gen_statem) → Agent & Skill "Templates"

**Erlang idea:** Behaviours define standard callback patterns (`init`, `handle_call`, `handle_cast`, `terminate`…).

**TARS translation:**

Define base behaviour contracts in F#:

- `IAgentBehaviour` (init, handleMessage, terminate, snapshotState)
- `ISkillBehaviour` (init, invoke, cleanup)
- Maybe a `IStateMachineAgent` analogue to `gen_statem`

Every agent implementation (even AI-generated ones) adheres to that fixed shape.

**Why it matters:**

- Evolvable agents are less scary if they live in a strict lifecycle box
- FLUX-generated code can be constrained to "fill in callbacks," not reinvent runtime

### 5. Links & Monitors → Dependency Graph Between Agents

**Erlang idea:** Processes can be linked/monitored so failures propagate in a controlled way.

**TARS translation:**

When one agent depends on another (e.g., `PlannerAgent` → `ToolAgent`):

- Kernel sets up a monitor
- If callee fails:
  - Caller is informed (equivalent of 'DOWN' message)
  - Can switch strategy / fallback agent / downgrade

For multi-agent workflows:

- Compose them as a small supervision tree:
  - Parent = orchestration agent
  - Children = specialist agents

**Why it matters:**
Gives you principled handling of cascading failures instead of ad-hoc "if tool fails, try again 3 times."

### 6. Hot Code Upgrade → Versioned Agents/Skills with Rolling Activation

**Erlang idea:** Swap process code without stopping the node (old + new versions coexist while calls drain).

**TARS translation:**

TARS already wants:

- Versioned skills
- Versioned agents
- Evolution pipeline

Add OTP-like semantics:

- Kernel can run old and new versions side by side:
  - Some sessions pinned to vN
  - New sessions canary on vN+1
- Migration rules:
  - How to transform state from old to new
  - When to retire old version

**Why it matters:**

- Evolution becomes continuous deployment with safety nets, not "overwrite and pray"
- Lets you do A/B testing between agent versions in production

### 7. OTP Applications & Releases → TARS "Subsystems"

**Erlang idea:** OTP application = unit of deployment; release = coherent bundle of apps.

**TARS translation:**

Treat TARS subsystems as applications:

- `Tars.Core`
- `Tars.DevAssistant`
- `Tars.ResearchAssistant`
- `Tars.CodeEvolutionLab`

Each has:

- Its own agents
- Skills
- Config
- Supervision tree

You can then cut releases of TARS as composable bundles rather than one monolith.

### 8. Observability & Tracing à la Erlang

Erlang's runtime + tools (observer, dbg, tracing) excel at introspecting live systems.

**TARS analogue:**

- `Observability.Traces` = per-agent, per-message traces (like `erlang:trace/3`)
- `Observability.Replay` = re-feeding sequences of messages to re-run a scenario
- Logging at message boundaries, not random printf in the middle of logic

This is critical for debugging weird emergent agent behavior.

### What to Bake Into the Spec Now

If we pick just 3 Erlang concepts to canonize into the TARS v2 doc:

1. **Supervision trees** for agents and skills
2. **Per-agent mailboxes** with selective receive + backpressure
3. **Versioned hot-upgrade semantics** for agents/skills (old+new in parallel + canarying)

Those three alone will make TARS feel much more like a robust AI "node" than a fancy LLM wrapper.

---

## Detailed Examples

### TARS.Graphiti Skill Manifest + Protocol

#### Skill Manifest (TARS side)

Assume your skills are external processes speaking JSON over stdin/stdout.

```json
{
  "id": "tars.graphiti",
  "version": "0.1.0",
  "runtime": "python",
  "entrypoint": "graphiti_skill.py",
  "description": "Bridge between TARS and Graphiti temporal knowledge graph.",
  "capabilities": [
    "ingest_code",
    "ingest_text",
    "query_patterns",
    "export_subgraph"
  ],
  "protocol": {
    "transport": "stdio-json",
    "request_schema": "GraphitiSkillRequest",
    "response_schema": "GraphitiSkillResponse"
  }
}
```

#### Protocol: Requests & Responses

Define a single envelope with an `op` field so you can add operations over time.

**Request schema:**

```json
{
  "type": "object",
  "title": "GraphitiSkillRequest",
  "required": ["op", "request_id"],
  "properties": {
    "request_id": { "type": "string" },
    "op": {
      "type": "string",
      "enum": [
        "ingest_episodes",
        "query_subgraph",
        "extract_code_patterns"
      ]
    },
    "payload": { "type": "object" }
  }
}
```

**Response schema:**

```json
{
  "type": "object",
  "title": "GraphitiSkillResponse",
  "required": ["request_id", "status"],
  "properties": {
    "request_id": { "type": "string" },
    "status": {
      "type": "string",
      "enum": ["ok", "error"]
    },
    "error": {
      "type": "object",
      "nullable": true,
      "properties": {
        "code": { "type": "string" },
        "message": { "type": "string" }
      }
    },
    "result": { "type": "object" }
  }
}
```

#### Operation: ingest_episodes

This aligns with Graphiti's core abstraction of "episodes" (text or JSON) that get turned into nodes/edges in a temporal KG.

**Request:**

```json
{
  "request_id": "123",
  "op": "ingest_episodes",
  "payload": {
    "group_id": "tars-codebase",
    "episodes": [
      {
        "episode_id": "fsharp-file-1",
        "type": "code",
        "language": "fsharp",
        "path": "src/Core/Tars/Kernel.fs",
        "content": "let add a b = a + b",
        "metadata": {
          "module": "Tars.Kernel",
          "commit": "abc123",
          "timestamp": "2025-11-21T05:20:00Z"
        }
      }
    ]
  }
}
```

**Response:**

```json
{
  "request_id": "123",
  "status": "ok",
  "error": null,
  "result": {
    "ingested_episodes": 1,
    "graph_node_ids": ["episode:fsharp-file-1"]
  }
}
```

#### Operation: query_subgraph

Pull out a relevant slice of the Graphiti graph (episodes + entities + relationships) for a file, function, or concept.

**Request:**

```json
{
  "request_id": "124",
  "op": "query_subgraph",
  "payload": {
    "group_id": "tars-codebase",
    "filters": {
      "episode_ids": ["fsharp-file-1"],
      "entity_types": ["Function", "Parameter", "Call"],
      "max_hops": 2
    }
  }
}
```

**Response (simplified):**

```json
{
  "request_id": "124",
  "status": "ok",
  "error": null,
  "result": {
    "nodes": [
      {
        "id": "func:add",
        "type": "Function",
        "language": "fsharp",
        "name": "add",
        "metadata": {
          "path": "src/Core/Tars/Kernel.fs",
          "arity": 2
        }
      },
      {
        "id": "param:a",
        "type": "Parameter",
        "name": "a"
      },
      {
        "id": "param:b",
        "type": "Parameter",
        "name": "b"
      }
    ],
    "edges": [
      {
        "id": "e1",
        "type": "HAS_PARAM",
        "from": "func:add",
        "to": "param:a"
      },
      {
        "id": "e2",
        "type": "HAS_PARAM",
        "from": "func:add",
        "to": "param:b"
      }
    ]
  }
}
```

This graph-shaped JSON is what your FLUX metascripts will consume.

#### Operation: extract_code_patterns

Optional higher-level helper: Graphiti + LLM summarize patterns across many episodes.

**Request:**

```json
{
  "request_id": "125",
  "op": "extract_code_patterns",
  "payload": {
    "group_id": "tars-codebase",
    "scope": {
      "language": "fsharp",
      "modules": ["Tars.Kernel"]
    },
    "limit": 20
  }
}
```

**Response (high-level patterns):**

```json
{
  "request_id": "125",
  "status": "ok",
  "error": null,
  "result": {
    "patterns": [
      {
        "id": "pattern:fsharp-function-basic",
        "description": "Simple curried F# functions with two parameters and a single expression body.",
        "example": "let add a b = a + b",
        "graph_signature": {
          "node_types": ["Function", "Parameter"],
          "edge_types": ["HAS_PARAM"]
        }
      }
    ]
  }
}
```

This is almost directly usable as grammar rule seeds.

### Example FLUX Metascript Consuming Graphiti Output

Assume FLUX is a functional-ish DSL that transforms JSON-shaped structures into other structured data (e.g., grammar rules).

**Input type (conceptual):**

```
GraphitiSubgraph:
  - nodes: Node[]
  - edges: Edge[]
```

**Output type:**

```
GrammarRules:
  - rules: Rule[]
```

**Pseudo-FLUX:**

```flux
metascript GraphitiToGrammar
  input GraphitiSubgraph
  output GrammarRules

  // Helper: find all functions with parameters
  def findFunctionPatterns(nodes, edges):
    let functions = filter(nodes, n -> n.type == "Function")
    in map(functions, f -> {
      let params =
        edges
        |> filter(e -> e.type == "HAS_PARAM" && e.from == f.id)
        |> map(e -> find(nodes, n -> n.id == e.to))

      {
        "functionNode": f,
        "parameters": params
      }
    })

  // Turn a function+params structure into a grammar rule
  def functionPatternToRule(pattern):
    let f = pattern.functionNode
    let params = pattern.parameters

    // Simple: only grammar for name + arity
    {
      "id": "rule:fsharp:function:" + f.name,
      "language": "fsharp",
      "kind": "function_declaration",
      "match": {
        "name": f.name,
        "arity": length(params)
      },
      "template": "let {{name}} {{#each params}}{{this.name}} {{/each}}= {{body}}"
    }

  def main(graph):
    let patterns = findFunctionPatterns(graph.nodes, graph.edges)
    let rules = map(patterns, p -> functionPatternToRule(p))

    { "rules": rules }
```

This script:

1. Walks Graphiti's graph
2. Finds Function nodes and their HAS_PARAM edges
3. Produces grammar rules describing "what a function declaration looks like" for each one

Later you generalize across many functions to produce more abstract rules (e.g., "any function with 2 parameters").

### Grammar Distillation Pipeline Using Graphiti

**End-to-end pipeline inside TARS:**

#### Step 0: Decide corpus & scope

- **Source**: TARS's own F# codebase (plus maybe external repos later)
- **Scope**:
  - Language: `fsharp`
  - Modules: `Tars.Kernel`, `Tars.Cortex`, etc.

#### Step 1: Ingest code into Graphiti

- TARS walks repo and chunks code files
- For each file, create a code episode (`type=code`, `language=fsharp`)
- Call `tars.graphiti` with `ingest_episodes`
- **Result**: Graphiti builds a temporal KG of episodes + entities + relations

#### Step 2: Extract structural information

Two options:

1. **Low-level**: use `query_subgraph` to pull subgraphs (episodes + function/entity nodes)
2. **High-level**: use `extract_code_patterns` to have Graphiti+LLM pre-summarize patterns

Either way, you end up with JSON graph data.

#### Step 3: FLUX transforms graph → grammar rules

- Feed the Graphiti subgraph JSON into a FLUX metascript (like `GraphitiToGrammar`)
- FLUX walks the graph, identifies patterns, outputs grammar rules

#### Step 4: Cortex consolidates & stores rules

- `Tars.Cortex.Grammar.Distillation` receives the rules
- Merges/deduplicates across multiple files
- Stores in `IGrammarStore`

#### Step 5: Use grammars for evolution

- When TARS wants to generate/mutate F# code:
  - Queries the grammar store
  - Uses rules to constrain LLM output (via JSON schema / GBNF)
  - Ensures generated code matches learned patterns

**This forms a loop:**

```
Code → Graphiti → Graph → FLUX → Grammar → Constrained Generation → Better Code
```

---

## Four Areas for Deeper Elaboration

There are four areas where deeper elaboration will have the highest leverage for TARS:

### 1. The F# "Compiler as Guardian" Model

This is the superpower of your stack, and it deserves a full design spec.

**Why elaborate?**
Because this defines:

- What the AI is allowed to mutate
- What it can break
- What it can't possibly break
- What guarantees you have before running mutated code

**What needs detailing:**

- Exactly which DUs form the "immutable kernel types"
- Which modules are never AI-modifiable
- Which modules are only modifiable via strongly typed interfaces
- A strategy for compiler-level test contracts
- How to encode budgets, capabilities, and constraints using:
  - Units of Measure
  - Exhaustive matches
  - Sealed modules

This deserves its own chapter in the TARS spec.

### 2. The Skill Isolation Architecture

We touched this, but it needs a real decision.

**Why elaborate?**
This is the security boundary for all self-evolving code.

**What needs detailing:**

- Execution model (process-per-skill, thread-per-skill, container-per-skill?)
- Communication protocol (JSON-RPC? Named pipes? gRPC?)
- Resource limits (memory, CPU, timeouts)
- Sandboxing:
  - File access
  - Network access
  - Environment isolation
- How the AI submits skill mutations
- How mutated skills are built, loaded, tested, and versioned

This architecture determines whether TARS is safe.

### 3. Agent Negotiation & Protocols

This is the core of multi-agent intelligence—and the WEIRDEST part of your system.

**Why elaborate?**
Because the protocol determines:

- Whether agents coordinate
- Whether they conflict
- Whether they evolve consistent world models
- Whether emergent behaviors appear

**What needs detailing:**

- The negotiation DSL:
  - Proposals
  - Evaluations
  - Objections
  - Counter-proposals
  - Consensus
- Shared goals vs. private goals
- Conflict resolution mechanisms
- How agents exchange beliefs
- How to avoid "belief drift" or knowledge fragmentation

We should design this like a miniature diplomatic treaty system.

### 4. The Evolution Pipeline (The Most Important Part of TARS)

ChatGPT, Gemini, and you all agree: you need a strict, formal pipeline for self-modification.

This is the heart of TARS.

**Why elaborate?**
Because without a fully specified pipeline, "AI rewriting its own code" is a minefield.

**What needs detailing:**

- The human-overridable evolution toggle
- Exact stages of mutation:
  - Proposal
  - Generation
  - Static analysis
  - Test-suite execution
  - Sandbox run
  - Metrics evaluation
  - Canary deployment
  - Lineage logging
- Fitness scoring:
  - Correctness
  - Performance
  - Resource usage
  - Consistency with constraints
- When mutations are allowed or forbidden
- How rollback works
- How lineage is stored
- How "agent species" diverge or converge

This is a system that allows safe Darwinism, and it must be crisp.

### Prioritization Suggestion

1. **Elaborate the F# Compiler-as-Guardian model first**
   - It's your strongest advantage and shapes everything else

2. **Then define the sandbox/isolation model for skills**
   - That's your safety boundary

3. **Then do the evolution pipeline**
   - That's your correctness boundary

4. **Then do the multi-agent protocol**
   - That's your emergent intelligence boundary

---

## Final Synthesis

ChatGPT gave you an ambitious, elegant architecture.
Gemini injected a healthy amount of production realism.

**The new combined plan:**

- Keeps the beauty of the vision
- Cuts the early complexity
- Leans into F#'s strengths
- Avoids .NET sandboxing traps
- Makes V2 feasible on your home PC
- Leaves doors open for the "interstellar" evolution you want

---

## Next Steps

Possible next actions:

1. A fully structured `TARS.sln` skeleton
2. The initial F# files for Kernel/Cortex/Memory
3. The canonical DUs and core types
4. A bootstrap script (`dotnet new`)
5. A roadmap visualization
6. Plug FLUX into `GrammarModule.ProposeUpdate` (real L0→L1 flow)
7. Add an `IL0Consolidator` stub that consumes session traces and outputs `L0ConsolidatedSignal`
8. Extend the FLUX metascript to do actual Graphiti-based pattern extraction

---

*Document compiled from ChatGPT conversation history, November 21, 2025*
