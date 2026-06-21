# ADR 0001 — TARS construction and execution of ix pipeline meshes

- **Status:** Accepted (design) — 2026-06-21
- **Context source:** `grill-with-docs` session. Glossary in `/CONTEXT.md`.
- **Question:** Should TARS be able to construct and run 100+ node IXQL pipeline
  meshes using ix's `ix-pipeline` (fan_out/DAG executor) + `ix-duck` (DuckDB)?

## Context

TARS reaches ix only through `IxSkill.fs` → `ix run <skill>`, scoped to
MCTS/grammar skills. It has no bridge to `ix-pipeline`, `ix-duck`, or IXQL
execution. Investigation during the session established:

1. `ix-skill` **already exposes** a headless pipeline surface:
   `ix pipeline run --file ix.yaml --json --param …` (+ `validate`, `dag`, `schema`,
   NDJSON streaming, and `ix pipeline compile "<NL>"`). Reachable through the
   existing `IxSkill` subprocess seam.
2. The executor's native format is **`ix.yaml`** (registry skills + `deps`,
   parallelism implicit). **IXQL (`.ixql`) has a parser but no executable backend** —
   no `.ixql`→`ix.yaml` lowering exists anywhere, and `ix` has no IXQL knowledge.
3. The node vocabulary is **ix-registry skills** (~64, all ix-namespaced). There is
   **no `tars.*` skill namespace** (the `.ixql` examples calling `tars.validate_schema`
   were aspirational). `tars_bridge` is a fixed prepare→handoff skill, not a callback.
4. `optimize` only minimizes synthetic benchmark functions; `grammar.search`
   (consumes `grammar_ebnf`) and `grammar.weights` (consumes `{rule_id, success}`)
   fit TARS's data directly.
5. `PipelineResult.node_results` retains **every** stage output. No reduce/argmax
   skill exists in the registry.
6. **Governance + provenance are baked into every run** (`ConstitutionGate` over
   resolved args + `ix.lock` provenance trail). Inherited, not opt-in.
7. The prebuilt `ix` binary currently reports `pipeline` as a stub; the seam can
   silently lack the capability until ix is rebuilt. There is no F# fallback for a
   100-node mesh — but there **is** a serial F# path (`MctsBridge`).

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **TARS owns the full lifecycle**: construct + execute + consume. | User intent; the value is a closed loop, not a passive consumer. |
| D2 | **TARS emits `ix.yaml` directly** (not `.ixql`, not NL-compile). | `ix.yaml` runs today via the existing seam; choosing `.ixql` means first building a missing compiler (demerzel's to own). 100+ nodes = unrolled stages. |
| D3 | **Nodes are pure ix-registry compute skills** over TARS data. | Keeps it cheap (one subprocess, in-process parallel, memoized) and a **TARS-only build** — TARS-native nodes would require new ix-registry skills (cross-repo) and break the cost model. |
| D4 | **First mesh = grammar-config param-sweep over the evolution corpus**, feeding promotion. | Fan-out is the natural path to 100+ nodes; closes TARS's existing self-improvement loop. (Sweep axis is `grammar.search`/`grammar.weights`, **not** `optimize`.) |
| D5 | **Hybrid data contract**: shared corpus (EBNF + observations) param-bound via `--param NAME=@file`; per-stage swept knobs (`exploration`, `max_depth`, `seed`) inline. | Small yaml, one canonical corpus, clean provenance — the split `spec.rs` was designed for. |
| D6 | **TARS-side rank, reusing `MctsBridge`→promotion plumbing**. | Mesh stays pure fan-out (no reduce skill exists); TARS reads all `node_results`, ranks by reward (`MctsBridge.parseMctsOutput`), routes winners through the validated template→pattern-outcome→promotion path. No new ix skill. |
| D7 | **Capability-probe, then degrade to sequential `MctsBridge`** when `ix pipeline` is absent/stale. | Never hard-fails; the mesh is a *parallel accelerator* of a path that already works serially. Mirrors the existing "try ix, fall back to F#" ethos. No auto-build of ix. |

## Consequences

- **Scope: TARS-only.** No ix-repo (Rust) or demerzel changes required for the
  first mesh. New surface lives in TARS (`Tars.Evolution`), reusing `IxSkill`.
- `IxSkill` gains a `pipeline run` path (file + `--param`, JSON parse of
  `node_results`); a new bridge (e.g. `GrammarMeshBridge`) sits alongside
  `MctsBridge`/`MachinBridge` and emits the `ix.yaml`, runs it, ranks, promotes.
- Governance/provenance come for free, but **emitted args must pass the
  ConstitutionGate** or the run fails fast — a real constraint on the emitter.
- **Deferred / Phase 2 (explicitly out of scope):** TARS-native operations as
  mesh nodes (needs new ix-registry skills, cross-repo); `.ixql` authoring +
  the `.ixql`→`ix.yaml` compiler (demerzel's to own); `ix-duck`/DuckDB-backed
  nodes; `ix pipeline compile` NL construction.

## Implementation status (2026-06-21)

Built and verified end-to-end (spike → implement, same session):

- `Tars.Core/IxSkill.fs` — added `runPipelineJson` (`ix pipeline run --file … --param
  NAME=@file --format json`) and `pipelineAvailable` (probes `pipeline schema`).
- `Tars.Evolution/GrammarMeshBridge.fs` — new module: `buildSweep`, `buildMeshYaml`
  (emits the param-bound mesh), `parseMeshOutput` (ranks `stages.<id>.output` by
  reward), `runSweep` (mesh path + serial `MctsBridge` fallback). Reuses
  `MctsBridge.templatesToEbnf` / `parseMctsOutput` / `indicesToActions` (made `internal`).
- `tests/Tars.Tests/GrammarMeshBridgeTests.fs` — 6 unit tests (sweep grid, emitter
  shape, invariant-culture floats, output parsing). All green; 48/48 in the
  MCTS/Machin/Mesh/Wot surface pass.
- **Verified against a rebuilt `ix`:** the emitter's yaml validates (`status: valid`)
  and a 6-node param-bound mesh runs, each node returning a reward.

### Resolved open items

- **Capability probe:** `ix pipeline schema` exit 0 + JSON output (the "stub" string
  is stale doc text on the parent command, so probe the subcommand, not `--help`).
- **`--param NAME=@file` contract:** the file must contain a **JSON value**; raw EBNF
  is written as `JsonSerializer.Serialize(ebnf)` (a JSON-encoded string).
- **Run stdout shape:** `{ "stages": { "<id>": { "output": {…}, "cache_hit", "duration_ms" } } }`
  — every node retained. The CLI key is `stages`, not the Rust-internal `node_results`.
- **`grammar.search` EBNF dialect:** `::=` + newline-separated (TARS's `templatesToEbnf`
  already emits this — no change needed).

### Consumer wiring (done)

- `tars evolve --grammar-mesh` runs `GrammarMeshBridge.runDefaultSweep ()` near the end
  of each cycle (before the promotion-index refresh), recording the winning derivation as
  a `WorkflowOfThought` `PatternOutcome` so the existing refresh picks it up. Off by
  default; degrades to serial F# when ix's pipeline surface is absent.
- Wiring: `EvolveOptions.GrammarMesh` (Evolve.fs) + `--grammar-mesh` arg (Program.fs).
- Verified: the default template pool's EBNF runs through `grammar.search` (reward 1.0).

### Still open (follow-up)

- Whether the shared corpus is one EBNF or a set of grammar variants (richer D4 sweep axis).
- Feed the *full* ranked sweep (not just the winner) once promotion can absorb the volume.
- Pre-existing build blocker: `NU1903` (SQLitePCLRaw audit) fails restore under
  `TreatWarningsAsErrors`; current builds need `-p:NuGetAudit=false`. Unrelated to this feature.
