# TARS v2 Ideas Review and Development Roadmap

## Review of `v2_ideas.md`

### Strengths and Alignment
- **Phased delivery mindset** keeps the scope survivable by deferring risky research (hyperbolic embeddings, deep triple stores, full sandboxing) to later phases while landing a usable core.
- **F# compiler as safety boundary** is a strong, pragmatic interpretation of the “guardian” concept; emphasizing discriminated unions and units-of-measure should materially constrain unsafe mutations.
- **Polyglot skill model** with strict process isolation aligns with the need to mix ecosystems without compromising the F# core.
- **Observability-first logging plan** (structured, replayable traces) recognizes debugging and evolution safety as first-class requirements.

### Risks, Gaps, and Clarifications Needed
- **Immutable boundary definition** is still implicit. The doc should explicitly list which projects/files are immutable and how change-control is enforced (e.g., Kernel/Cortex core types vs. skill/agent code).
- **Evolution pipeline** needs a concrete, automatable gate sequence (proposal → static checks → tests → sandbox run → metrics → canary) with rollback and lineage policies.
- **Skill isolation** picks Docker/WASM/process isolation but lacks resource governance (CPU/memory/time/network budgets) and an auditable manifest format.
- **Memory plan** mixes SQLite belief graph and ad-hoc vector folder store; migration plan to a proper vector DB and schema/versioning for graph storage is not spelled out.
- **Cortex/LLM integration** calls for grammar-constrained local models but omits fallbacks, prompt/response audit logs, and deterministic sampling defaults for replayability.
- **Graphiti/FLUX coupling** describes benefits but not the minimal slice required for v2 (e.g., a single ingestion/query path and one FLUX transformation script).
- **Supervision tree semantics** need concrete mapping to F# constructs (mailboxes, cancellation, backpressure) and how failures propagate across agents/skills.
- **Operational story** (build, CI, packaging) is missing; without it, evolution/testing gates will be brittle.

#### Targeted Clarifications to Fold Back Into `v2_ideas.md`
- List the immutable surfaces (e.g., `Tars.Kernel` contracts, `Tars.Cortex` cognitive interfaces, shared discriminated unions) and declare how they are modified (code owners, ADRs, signed releases).
- Encode the evolution gate as a checklist with owners and exit criteria, including the rollback trigger and how to record lineage/metrics artifacts.
- Define `SkillManifest` fields that capture budgets (wall-clock, CPU, memory, network policy), isolation mode, and auditability flags; state the default limits.
- Specify the belief-graph schema versioning approach (migration scripts, compatibility matrix) and the cutoff for swapping the vector shim to a DB with a backfill plan.
- Document the default Cortex operating mode (seed/temperature/grammar) plus fallback hierarchy (local → remote) and required audit logs for prompts/responses.
- Pin the minimal Graphiti/FLUX path (one ingestion/query path, one metascript, one `IGrammarStore` implementation) needed for v2 demoability.
- Map supervision semantics to concrete primitives (e.g., MailboxProcessor, Channels) and show failure propagation/restart strategies with an example tree.
- Add CI/build/package story to the high-level plan so gates and replay harnesses have reproducible environments.

#### Actionable Edits for `v2_ideas.md`
- **Mark immutable surfaces and change control**: add a short table naming each immutable package, the enforcement mechanism (code owners + signed release), and rollback policy.
- **Document the gate rubric**: insert a numbered “evolution gate” section with pass/fail criteria, default rollback trigger, and artifacts to persist (trace, metrics, lineage manifest).
- **Skill manifest defaults**: add a `SkillManifest` appendix with default CPU/memory/wall-clock limits, network policy options, and audit flags; include an example YAML/JSON.
- **Memory migration note**: include a migration stub that defines the version numbering scheme, a required backfill script, and the decision point for promoting to a vector DB.
- **Cortex defaults and fallbacks**: add a table for seed/temperature/grammar defaults, plus the fallback chain (local → remote) and required prompt/response logging keys.
- **Minimal Graphiti/FLUX slice**: explicitly limit v2 scope to one ingestion/query path and one FLUX metascript, with a note on where grammars are stored and versioned.
- **Supervision mapping example**: add a one-page example showing MailboxProcessor/Channel usage, cancel/timeout propagation, and restart strategy for parent/child agents.
- **Ops/CI stanza**: add a subsection that names the baseline CI jobs (build, formatting, unit tests) and packaging targets (devcontainer/Dockerfile) that other epics depend on.

## Development Roadmap (Epics → Spikes/Stories)

### Epic 0: Repository Skeleton and Safety Rails
- **Spike:** Define immutable vs. mutable boundaries and document enforcement rules (e.g., code owners, protected directories, signed artifacts).
- **Story:** Scaffold `Tars.sln` with projects (`Tars.Kernel`, `Tars.Cortex`, `Tars.Memory`, `Tars.Agents`, `Tars.Skills`, `Tars.Observability`, `Tars.Interface`).
- **Story:** Add core type shells (`Message`, `AgentState`, `SkillResult`, `Budget` with units-of-measure) plus exhaustive pattern-matching guard rails.
- **Story:** Establish CI for build + formatting + unit tests; add minimum test harness to prevent regressions in core DUs.

### Epic 1: Kernel Messaging and Supervision Backbone
- **Spike:** Validate mailbox/event-bus design (Channels vs. MailboxProcessor) with backpressure and cancellation semantics.
- **Story:** Implement single-node EventBus with monitoring hooks and pluggable persistence for traces.
- **Story:** Add `EchoAgent` with lifecycle states and health probes to prove the heartbeat loop.

### Epic 2: Cortex Minimal LLM Integration
- **Spike:** Evaluate local model runner (Ollama/LM Studio) latency and JSON/GBNF constraint support.
- **Story:** Implement `ICognitiveProvider`, `CognitivePlan`, and `GrammarConstraint` interfaces with deterministic defaults (temperature, seed).
- **Story:** Add request/response audit logging and error taxonomy for LLM calls.
- **Story:** Wire Cortex to EchoAgent for a constrained text-in/text-out flow (no tool use yet).

### Epic 3: Pragmatic Memory Grid
- **Spike:** Choose schema for SQLite belief graph (nodes/edges with versioning and provenance fields).
- **Story:** Implement minimal belief graph CRUD with migration scripts and a test fixture.
- **Story:** Provide file-based vector store shim with checksum/version metadata; define migration path to a real vector DB.
- **Story:** Integrate memory lookups into Cortex planning (read-only in this phase).

### Epic 4: Skill Protocol and Isolation
- **Spike:** Define `SkillManifest`, `SkillInvocation`, and transport (stdio-JSON baseline) with resource limits (timeouts, CPU/memory caps if available).
- **Story:** Build `ISkillHost` that launches skills as external processes with structured stdout/stderr capture and budget enforcement.
- **Story:** Provide reference skills: a no-op skill and a simple Python utility skill to validate the protocol.
- **Story:** Add audit logging for every skill invocation (inputs, outputs, resource usage metrics).

### Epic 5: Observability Tower
- **Story:** Emit per-run structured artifacts (`agentic_trace.json`, `memory_before/after.json`, `metrics.json`, `skills.json`).
- **Story:** Provide a replay harness that re-sends recorded message sequences to agents for deterministic debugging.
- **Story:** Add health endpoints/CLIs for liveness/readiness and recent error summaries.

### Epic 6: Controlled Evolution Pipeline
- **Spike:** Formalize the gate sequence and scoring rubric (correctness, latency, resource cost, constraint adherence).
- **Story:** Implement `ModificationRequest` and `CandidateChange` models plus lineage logging.
- **Story:** Create sandbox build/test runner for mutated skills/agents (containerized or process-isolated), with automatic rollback on failure.
- **Story:** Add canary deployment toggle and metrics comparison against baseline before promotion.

### Epic 7: Multi-Agent Protocols and Supervision Trees
- **Spike:** Design negotiation DSL (proposal, objection, consensus) and map to `Message` variants.
- **Story:** Implement per-agent mailboxes with selective receive/backpressure; add monitoring links between parent/child agents.
- **Story:** Add hot-upgrade semantics for agents/skills (dual-running old/new versions with migration rules).
- **Story:** Deliver a minimal two-agent scenario (Planner → ToolAgent) exercising negotiation and failure handling.

### Epic 8: FLUX and Graphiti Integration (Minimum Viable Slice)
- **Spike:** Stand up Graphiti ingestion/query for a small F# module; document schemas used.
- **Story:** Add FLUX metascript to transform Graphiti subgraphs into grammar rules and store via `IGrammarStore`.
- **Story:** Demonstrate constrained generation using distilled grammar in a single skill evolution flow.

### Epic 9: Operational Hardening
- **Story:** Package reproducible dev environment (Dockerfile or Nix/Devcontainer) with pinned tool versions.
- **Story:** Add performance/latency benchmarks for EventBus, LLM calls, and skill invocations.
- **Story:** Document runbooks for log inspection, replay, rollback, and upgrading agents/skills.

## Suggested Sequencing (High Level)
1. **Epic 0 → 1** to lock safety rails and heartbeat.
2. **Epic 2 → 3 → 4** to deliver cognition, memory, and safe skills.
3. **Epic 5 → 6** to make the system observable and safely evolvable.
4. **Epic 7 → 8** to unlock multi-agent behaviors and grammar-driven evolution.
5. **Epic 9** continuously as operational hygiene.

### Milestones (Indicative)
- **M0 (Week 0–2):** Epic 0 baseline (scaffolds, DUs, CI) + mailbox spike from Epic 1.
- **M1 (Week 3–5):** EchoAgent loop working with Cortex minimal integration (Epics 1–2) and belief-graph CRUD (Epic 3).
- **M2 (Week 6–8):** Skill host/protocol and reference skills (Epic 4) plus initial observability artifacts and replay harness (Epic 5).
- **M3 (Week 9–11):** Evolution pipeline gates automated with sandbox/canary (Epic 6) and a two-agent negotiation demo (Epic 7).
- **M4 (Week 12–14):** Minimal Graphiti/FLUX slice in production path (Epic 8) with operational hardening and benchmarks (Epic 9 ongoing).

