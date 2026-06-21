# CONTEXT — Domain glossary

Living glossary of TARS domain terms, built incrementally during design sessions.
Decisions live in `docs/adr/`; this file defines the *ubiquitous language*.

## IXQL pipeline mesh (feature: TARS ⇄ ix-pipeline)

> Seeded 2026-06-21 during a `grill-with-docs` session on whether TARS should
> construct and run 100+ node IXQL pipeline meshes. See `docs/adr/0001-tars-ixql-pipeline-mesh.md`.

- **ix** — sibling Rust workspace at `~/source/repos/ix`. ML/governance CLI for
  the GuitarAlchemist ecosystem. Builds a binary named `ix` from the `ix-skill` crate.
- **ix-registry** — ix's catalog of ~64 registered **skills** across 33 domains
  (math/ML/governance: `stats`, `fft`, `kmeans`, `pca`, `optimize`, `grammar.search`,
  `grammar.weights`, `governance.check`, `markov`, `bandit`, `nn.forward`, …).
  A skill is the atomic unit of compute a pipeline stage references **by name**.
- **ix-pipeline** — ix's DAG executor crate. Lowers a pipeline spec to a DAG,
  runs independent branches in parallel in-process (`parking_lot`, memoized),
  retains every node's output (`PipelineResult.node_results`).
- **ix.yaml** — the **executable** pipeline format (`ix-pipeline/src/spec.rs`,
  `version: "1"`). A `PipelineSpec` is `params` + `stages`. Each stage names a
  registry skill, supplies an `args` JSON blob, and declares `deps`. Inter-stage
  data flows via `{"from": "stage_id[.key]"}` refs; run-time values bind via
  `params` + `--param NAME=@file`. **This is what `ix pipeline run` executes.**
- **pipeline mesh** — informal term for a wide `ix.yaml` DAG (100+ stages).
  Parallelism is **implicit**: independent stages (no `deps` between them) run
  concurrently. `ix.yaml` has **no loop/map primitive** — width is achieved by
  the emitter *unrolling* N stage entries.
- **fan_out** — explicit parallel-branch construct in the **IXQL** surface syntax
  (`fan_out(p1, p2, p3)`). In `ix.yaml` the equivalent is N dep-free stages.
- **IXQL (`.ixql`)** — a *surface authoring language* for pipelines, living in the
  **demerzel governance repo** (`governance/demerzel/tools/ixql-parser`, F# parser →
  AST mapping to `grammars/sci-ml-pipelines.ebnf`; `pipelines/*.ixql`). Richer than
  `ix.yaml`: `<- / →` chains, `fan_out`, tetravalent `TFUC` gates, `tars.*`/`ix.*`
  tool calls. **Has a parser but NO executable backend** — nothing emits `ix.yaml`
  from the AST, and `ix` has zero knowledge of `.ixql`. The `.ixql` files are
  effectively design specs that parse but do not execute end-to-end (yet).
- **ix-duck / ix-duck-ext** — ix's DuckDB integration crates. Not on the critical
  path for the first mesh (the grammar sweep uses in-memory skill compute).
- **ConstitutionGate** — governance gate baked into *every* `ix pipeline run`:
  a template-time `governance_gate(spec)` pre-flight, then `lower_with_gate` vets
  each stage's **resolved** args against the constitution just before its skill runs.
  TARS inherits governance for free; emitted args must pass or the run fails fast.
- **ix.lock / provenance trail** — every run writes `ix.lock` (ix-lock/v2 provenance
  record) beside the spec and appends a durable run record. TARS inherits provenance.
- **tars_bridge** — the single ix skill that touches TARS (`ix-agent/src/handlers.rs`).
  **Not** a generic callback: fixed actions (`prepare_traces`, `prepare_patterns`,
  `export_grammar`) that prepare a payload and return a *handoff instruction*
  ("Call TARS tool `ingest_ga_traces` with this payload"). Cannot run TARS ops mid-mesh.
- **IxSkill** — TARS's single seam to ix (`v2/src/Tars.Core/IxSkill.fs`). Invokes
  `ix run <skill>`; prefers a prebuilt `target/{release,debug}/ix` binary, else
  `cargo run -p ix-skill`. **Caveat:** the current prebuilt debug binary reports
  `pipeline` as a "stub — full impl in Week 4"; source has the full impl. The seam
  can silently lack `pipeline` until ix is rebuilt.
- **MctsBridge** — `v2/src/Tars.Evolution/MctsBridge.fs`. Runs `grammar.search`
  single-shot via `IxSkill`, parses `best_derivation` → template indices, falls
  back to F# `MctsSolver` when ix is unavailable. The **grammar mesh is the parallel
  (N-config) generalization of what MctsBridge does serially.**
- **grammar sweep / first mesh** — the chosen first workload: fan `grammar.search`
  across N `(grammar variant, exploration, max_depth, seed)` configs over TARS's
  EBNF + pattern-outcome corpus; rank results TARS-side; feed winners to promotion.
