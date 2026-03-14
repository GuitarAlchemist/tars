# Compound Evolution: TARS + Guitar Alchemist

**Date**: 2026-03-14
**Status**: Active compound engineering across two repos

---

## The Ecosystem at a Glance

| Metric | TARS (F#) | Guitar Alchemist (C#) | Combined |
|--------|-----------|----------------------|----------|
| Source files | 476 .fs | 1,422 .cs | 1,898 |
| Lines of code | 93,481 | 117,367 | 210,848 |
| Test lines | 20,938 | 156,192 | 177,130 |
| Tests passing | 790/790 | 1,267/1,283 | 2,057 |
| Modules/Projects | 20 | 14 | 34 |
| Agents | 5 reasoning patterns | 6 specialized agents | 11 |
| Skills/Tools | 124+ registered tools | 6 skills + 16 MCP tools | 146+ |
| Roadmap phases | 15/17 complete | 4/4 complete | 19/21 |

---

## TARS: Evolution Timeline

### Phase 1 — Foundation (commits a6487c13..be91ec18)
- Autonomous evolution loop (`tars evolve`) with curriculum generation
- Broke the "factorial trap" where the LLM kept generating the same task
- Neuro-symbolic reasoning benchmark suite
- Knowledge Ledger integration (WoT results → Postgres)

### Phase 2 — MAF + Multi-Agent (commits 65a84aa0..fd35deee)
- Microsoft AI Framework (MAF) integration: `IChatClient`, `FunctionInvokingChatClient`
- Agent orchestrator with multi-agent routing, pipeline, and fan-out modes
- Streaming per-step WoT via `ConcurrentQueue`
- Golden trace feedback loop and persistent learning
- `tars evolve --loop N` for back-to-back evolution cycles

### Phase 3 — Probabilistic Grammars (commits 043ce409..a415976d)
- Three-force pipeline: EBNF constrained decoding + PCFG Bayesian weights + replicator dynamics
- MctsBridge to Rust (MachinDeOuf) with F# fallback
- MCP grammar tools: `weights`, `update`, `evolve`, `search`
- Constrained decoding for structured IR: IntentPlan, BeliefUpdate, RepairProposal
- End-to-end integration tests (25 constrained decoding + 7 probabilistic grammar)

### Phase 4 — Self-Improvement Loop (commits 41589f87..d2b931f6) *current*
- **PromotionIndex**: Bridges promotion pipeline output to agent pattern selector
- **Cross-repo discovery**: 5 GA pattern families ingested via static code analysis
- **Closed feedback loop**: evolve → outcomes → promotion → index → selector → evolve
- **Robust evolve**: Curriculum agent uses reasoning model, direct LLM fallback
- **Meta-cognitive gap detection**: Found 60% search failure rate, improved heuristics
- 3 new MCP tools: `ingest_ga_traces`, `ga_trace_stats`, `promotion_index`

### What TARS Learned From Guitar Alchemist

Five structural patterns discovered via cross-repo code analysis, all promoted to `grammar_rule` (highest level):

| Pattern | Occurrences | Score | What TARS Learned |
|---------|-------------|-------|-------------------|
| `ga.confidence_evidence_response` | 20 | 0.91 | Structured JSON with confidence + evidence fields |
| `ga.domain_skill_fastpath` | 24 | 0.90 | Regex-matched domain computation bypassing LLM |
| `ga.routing_fallback_cascade` | 16 | 0.85 | Semantic router → keyword → default agent cascade |
| `ga.orchestrator_pipeline` | 12 | 0.87 | 6-stage hook lifecycle (pre→skill→classify→dispatch→post→return) |
| `ga.hook_lifecycle_fsm` | 12 | 0.80 | FSM-driven hook execution with Mutate/Cancel/Continue |

These patterns now influence TARS's agent pattern selection through the PromotionIndex → PatternSelector bridge, with context-gated boost (max 0.08, tiebreaker only).

---

## Guitar Alchemist: Evolution Timeline

### Phase 1 — Domain Foundation (commits dbc4548c..89c2a198)
- AI/embedding core with OPTIC-K v1.2.1 schema (96 dims)
- Fretboard analysis, chord theory, interval computation
- Rust WASM guitar demo, Dockerized Ollama

### Phase 2 — Chatbot Orchestration (commits a0bc7f88..e0e95b33)
- Production orchestrator: skills → semantic router → specialized agents
- GA Language DSL for music theory queries
- Compound engineering flywheel: pattern → skill → agent → orchestrator
- Plugin architecture with auto-discovery (`[ChatPlugin]`)

### Phase 3 — AG-UI + Streaming (commits ca2fa0aa..695c4b8f)
- AG-UI protocol: SSE with domain custom events (`ga:diatonic`, `ga:scale`, `ga:candidates`)
- Token-level streaming via `AnswerStreamingAsync`
- RAG pipeline: YAML content → vector index → retrieval-augmented agents
- Ian Ring scale IDs + MCP lookup tools

### Phase 4 — Full Interactive Stack (commits da85cc08..5864226d) *current*
- **Streaming CLI**: single-shot, JSON, interactive modes
- **Multi-turn sessions**: `ConversationHistoryStore` with bounded history
- **Cross-agent delegation**: `IAgentCoordinator` with depth-limited recursion
- **Persistent memory**: `MemoryStore` (JSON-backed), `MemoryHook` (auto-recall), `MemoryMcpTools`
- **TraceBridgeHook**: Writes orchestrator events to `~/.ga/traces/` for TARS consumption
- **Probabilistic grammars**: Bayesian weighted rules + replicator dynamics for harmonic fitness

### What Guitar Alchemist Gained From TARS

| Capability | Origin | GA Integration |
|-----------|--------|---------------|
| Probabilistic grammars | TARS Evolution | Bayesian rule weights for harmonic progression scoring |
| Promotion staircase concept | TARS compound engineering | Skills earn promotion through demonstrated value |
| TraceBridge pattern | TARS cross-repo bridge | Hooks emit traces for external consumption |
| Meta-cognitive analysis | TARS meta-cognition | Gap detection informs skill development |

---

## The Compound Engineering Flywheel

```
     TARS (F#)                          Guitar Alchemist (C#)
     ─────────                          ─────────────────────
     evolve tasks                       orchestrator events
         │                                      │
         ▼                                      ▼
     pattern outcomes              TraceBridgeHook writes
         │                          ~/.ga/traces/
         ▼                                      │
     promotion pipeline ◄───────────────────────┘
         │                          GaTraceBridge reads
         ▼
     PromotionIndex
     (persisted JSON)
         │
         ▼
     PatternSelector
     (context-gated boost)
         │
         ▼
     agent execution ──────────────► better patterns
         │                          inform GA skills
         ▼
     meta-cognitive
     gap analysis
         │
         ▼
     curriculum generation
     (targets weak areas)
         │
         └──────► loop repeats
```

### Key Properties

1. **Asymmetric learning**: TARS learns structural patterns from GA's domain-specific orchestration. GA gains meta-cognitive infrastructure from TARS.

2. **No circular dependency**: TARS reads GA traces via flat JSON files (`~/.ga/traces/`). GA writes traces via a hook. Neither repo imports the other.

3. **Promotion as proof**: Patterns aren't just copied — they climb TARS's 5-level staircase (Implementation → Helper → Builder → DslClause → GrammarRule) through demonstrated value. All 5 GA patterns reached GrammarRule, proving genuine structural merit.

4. **Self-correcting**: Meta-cognitive analysis detects capability gaps (e.g., 60% search failure rate) and generates targeted curriculum. The evolve loop now uses the reasoning model for curriculum generation, addressing the root cause.

---

## Benefits Realized

### For TARS
- **5 production-grade patterns** discovered from a real-world application, not synthetic traces
- **Closed self-improvement loop**: the system can now improve itself without human intervention
- **Better model routing**: curriculum generation uses reasoning model instead of 3B, reducing fallback rate
- **Robust error handling**: `diag reasoning` no longer crashes without Postgres

### For Guitar Alchemist
- **Full interactive stack**: streaming CLI, multi-turn sessions, cross-agent delegation, persistent memory
- **6 specialized agents** (Theory, Tab, Technique, Composer, Critic + base orchestration) with cross-agent delegation
- **6 orchestrator skills** (ScaleInfo, FretSpan, ChordSubstitution, KeyIdentification, ProgressionCompletion, SkillMd-driven)
- **4 lifecycle hooks** (sanitization, memory, observability, trace bridge)
- **16 MCP tool types** (chat, scales, chords, keys, instruments, DSL, web search, feed reader, etc.)
- **Persistent memory** with MemoryStore, MemoryHook, and MemoryMcpTools

### For the Ecosystem
- **210K+ lines of code** across 1,574 source files
- **2,057 tests** passing across both repos
- **Three-repo bridge** (TARS ↔ GA ↔ MachinDeOuf) via MCP and CLI
- **Probabilistic grammars** shared between TARS (pattern evolution) and GA (harmonic fitness)
- **Compound engineering staircase** validated: patterns genuinely climb from Implementation to GrammarRule through demonstrated, scored value

---

## What's Next

| Priority | Item | Repo | Impact |
|----------|------|------|--------|
| High | Install `deepseek-r1:8b` for reasoning tasks | TARS | Evolve loop generates real curriculum instead of fallbacks |
| High | Bidirectional trace bridge | Both | GA reads TARS meta-cognitive insights for skill prioritization |
| Medium | Live trace ingestion (watch `~/.ga/traces/`) | TARS | Real-time pattern discovery instead of batch |
| Medium | Embed TARS reasoning in GA skills | GA | Theory/Composer agents use WoT for complex queries |
| Low | Unified MCP federation | All 3 | Single MCP server exposes all repos' tools |
