# Beyond-LLM Relevance Matrix

- **Status:** Draft
- **Area:** Research / Architecture Watch
- **Owner:** TARS
- **Version:** 1.0.0

## 1. Relevance Matrix

This matrix maps candidate AI architectures to the GuitarAlchemist ecosystem components and defines their current adoption status.

| Architecture Family | TARS (Harness) | IX (Skill Engine) | GA (Domain) | Demerzel (Gov) | Recommended Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLMs (Baseline)** | Orchestration, extraction | Semantic tools | Content generation | Policy drafting | **Maintain** |
| **State Space Models (SSM)** | Long trace analysis | Log/sequence mining | N/A | Audit log analytics | **Experiment** (IX) |
| **JEPA / World Models** | Simulation-before-action | Skill trajectory eval | Env simulation | Risk simulation | **Watch** |
| **Neuro-symbolic** | DSL/Grammar validation | Logic benchmarks | Music theory rules | Policy enforcement | **Adopt** (Tracer) |
| **Graph Reasoning** | Memory contradictions | Knowledge retrieval | Musical relationships | Repo health maps | **Experiment** (TARS) |
| **Diffusion / Flow** | N/A | N/A | Media generation | N/A | **Defer** (GA-only) |
| **Continual Learning** | Memory-to-action | N/A | N/A | N/A | **Reject** (Model-level) |
| **Small Specialists** | Local workers | Task-specific skills | N/A | Cheap audit checks | **Adopt** (Local) |

## 2. Adoption Rationale & Gates

### Adopt (Small Specialists & Neuro-symbolic)
- **Rationale:** High local efficiency and deterministic reliability.
- **Gate:** Must run on standard local hardware (e.g., 8GB RAM) and pass grammar validation checks.
- **Tracer Bullet:** Use a 1B-3B model for local unit test generation and symbolic logic for `.trsx` validation.

### Experiment (SSM & Graph Reasoning)
- **Rationale:** High potential for solving long-context and corpus-wide retrieval issues.
- **Gate:** Requires a measurable benchmark improvement in trace processing or multi-hop RAG.
- **Tracer Bullet:** Prototype Mamba for log analysis in IX or a small GraphRAG index for the Streeling `research/` directory.

### Watch (JEPA / World Models)
- **Rationale:** Promising for future planning and simulation but currently high complexity and low maturity for general repository tasks.
- **Gate:** Wait for open-source implementations that demonstrate stable world modeling for software/system states.

### Defer / Reject
- **Rationale:** Diffusion is too niche for the core agent harness (though valuable for GA creative output). Continual Learning at the model level is rejected in favor of harness-level memory learning to prevent uncurated model drift.

## 3. Repo-Relevance Ranking (P1-P3)

1. **P1: Small Specialists (Local Efficiency)** - Immediate cost and latency reduction.
2. **P1: Neuro-symbolic (Reliability)** - Crucial for "anti-hallucination" in agent-written code and DSLs.
3. **P2: Graph Reasoning (Memory)** - Essential for scaling the "second brain" beyond simple RAG.
4. **P2: State Space Models (Tracing)** - Needed for deep analysis of long autonomous agent runs.
5. **P3: World Models (Planning)** - Future frontier for simulation-before-action.

## 4. Cost & Complexity Notes

| Architecture | Setup Cost | Runtime Cost | Complexity |
| :--- | :--- | :--- | :--- |
| **Small Specialists** | Low | Low (Local) | Low |
| **Neuro-symbolic** | Medium | Low | High |
| **Graph Reasoning** | High (Indexing) | Medium (Traversals) | Medium |
| **SSM / Mamba** | Medium | Medium | Medium |
| **World Models** | Very High | High | Very High |

### Cost Guidelines
- **Tier 1 (Free-Local):** Small Specialists, Neuro-symbolic, basic Graph retrieval.
- **Tier 2 (Cloud/High-Runner):** Extensive GraphRAG, SSM training/fine-tuning, World Model simulation.
- **Rule:** Prioritize Tier 1 for initial tracer bullets.
