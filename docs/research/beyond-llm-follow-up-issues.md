# Follow-up Tracer Bullet Issues: Beyond-LLM Architectures

This document contains draft issues for measurable tracer bullet experiments identified in the technology-watch research.

## 1. [Tracer][TARS-V2] Small Specialist Model for Local Unit Test Generation

```yaml
issue_meta:
  level: task
  parent: "GuitarAlchemist/tars#96"
  area: research
  priority: P1
  complexity: S
  risk: low
  afk:
    readiness: ready
    max_autonomy: pr
  budget:
    tier: free-local
    max_cost_usd: 0
  evidence_required:
    - local_inference_benchmark
    - generated_test_quality_score
```

**Goal:**
Evaluate the feasibility and quality of using a small specialist model (e.g., Phi-3-mini, Qwen2.5-Coder-3B) running locally via Ollama or similar for generating F# unit tests for `Tars.Core`.

**Tasks:**
- Set up a local inference endpoint for a 1B-3B coding model.
- Implement a small script to pipe a TARS source file to the model with a "generate unit test" prompt.
- Compare the output quality and latency against a frontier model (baseline).

---

## 2. [Tracer][TARS-V2] Neuro-symbolic Grammar Validation for `.trsx` DSL

```yaml
issue_meta:
  level: task
  parent: "GuitarAlchemist/tars#96"
  area: research
  priority: P1
  complexity: M
  risk: low
  afk:
    readiness: ready
    max_autonomy: pr
```

**Goal:**
Integrate a symbolic grammar validator into the TARS WoT execution pipeline to ensure LLM-generated `.trsx` files adhere to strict schema constraints before execution.

**Tasks:**
- Define a formal EBNF or JSON schema for the `.trsx` DSL.
- Implement a validation step in `Tars.DSL.Core.Workflow` that uses a symbolic parser to reject malformed plans.
- Verify that this catch rate is 100% for syntax errors, reducing runtime failures.

---

## 3. [Tracer][IX] SSM/Mamba for Long-Sequence Trace Analysis

```yaml
issue_meta:
  level: task
  parent: "GuitarAlchemist/ix#211"
  area: research
  priority: P2
  complexity: M
  risk: low
```

**Goal:**
Experiment with a State Space Model (SSM) like Mamba for analyzing long autonomous agent traces (e.g., >10k events) to identify patterns of failure or success that exceed standard Transformer context windows.

**Tasks:**
- Collect a dataset of long agent execution traces from IX logs.
- Use a pre-trained Mamba-based model to perform sequence classification or anomaly detection on these traces.
- Compare throughput and memory usage against a long-context Transformer (e.g., GPT-4o-mini or Claude-3-Haiku).

---

## 4. [Tracer][TARS-V2] Small GraphRAG Index for Streeling Research

```yaml
issue_meta:
  level: task
  parent: "GuitarAlchemist/tars#103"
  area: research
  priority: P2
  complexity: M
  risk: low
```

**Goal:**
Implement a lightweight GraphRAG index for a single "department" in the Streeling catalog (e.g., `research/`) to evaluate if multi-hop retrieval improves second-brain synthesis.

**Tasks:**
- Use a simple entity-extraction prompt to build a JSON-based knowledge graph of a few Streeling markdown files.
- Implement a "global query" handler that traverses the graph to answer cross-document questions.
- Benchmark retrieval accuracy against standard vector-only RAG.
