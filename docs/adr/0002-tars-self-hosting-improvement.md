# ADR 0002 — Self-hosting recursive improvement: TARS improves TARS

- **Status:** Accepted (design) — 2026-06-21
- **Context source:** `grill-with-docs` session on boosting recursive
  self-improvement. Glossary in `/CONTEXT.md`. Follows ADR 0001 (grammar-mesh).
- **Question:** How should TARS apply its mutation→verify→promote machinery to its
  own F# codebase, gated by its own test suite — the canonical recursive target?

## Context

TARS already self-improves at five levels (prompt repair, pattern-weight bandit,
grammar staircase, model-weights SFT, parallel mesh exploration), but never on its
own source code. Investigation established the existing machinery and its gaps:

- `Mutation.Apply` (`Tars.Core/WorkflowOfThought/Mutation.fs`) writes a mutated
  **variant file** to `.wot/variants/` via string `Replace`. `IsApplied = true`
  means "variant written," **not "verified."** `Target` DU = `Prompt | RoutingConfig
  | Pattern` — **no source-code target case.**
- `Selection.evaluate` (`Selection.fs`) already encodes acceptance: compares a
  mutant `Performance {PassRate; TotalCost; DiffCount}` to a baseline →
  `Promote | Rollback | InsufficientData`, with a zero-regression check. But nothing
  computes that `Performance` from a real build+test of TARS source.
- `SelectionService.ApplyDecision` promotes via raw `File.Copy(variant, original)`
  — no git, no safety net. Unsafe for source mutation.
- `Tars.Sandbox` is a Docker client (heavyweight; Docker is flaky here — the 4
  skipped tests are Docker). `GitIntegration` wraps git via a generic
  `executeGit workingDir args` (branch/checkout/commit/status), but no worktree use.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **Fitness signal = a failing/skipped/`xfail` test made to pass**, tests as guardrail. | The test suite is the spec → unambiguous, hard-to-game accept signal; reuses `Selection.evaluate`'s zero-regression logic. Perf/quality signals are Phase 2 (noisy/gameable). |
| D2 | **Isolation = git worktree per variant.** `git worktree add` from HEAD → apply → build+test → `worktree remove` on rollback / commit on promote. | Full isolation (a broken build never reaches the live tree); reuses `executeGit`; keeps the working tree clean. Docker is fragile here and overkill for first-party code. |
| D3 | **Scope = explicit `(test, file)` pairs** (curated seed list). | Sidesteps fault localization (hard, error-prone) for v1; deterministic and surgical; `Mutation` already takes a target file. Heuristic localization is the Phase-2 unlock. |
| D4 | **Autonomy = commit verified variants to an isolated `self-improve/*` branch; human reviews + merges to main.** | Loop closes autonomously (branch + diff + rationale is its artifact) while main stays human-gated. Respects the standing "don't commit to main unless asked" preference. |
| D5 | **Loop = best-of-N parallel + one repair round.** N diverse proposals (~4) in concurrent worktrees, accept first green; if none pass, one error-fed repair via `analyzeAndPropose`. | The parallel recursion *boost*; reuses the worktree-isolation + parallel patterns from ADR 0001 and the existing failure→propose code. N bounded (concurrent `dotnet test` is heavy). |
| D6 | **Hermetic gate.** Test files immutable from HEAD; variant edits only the one designated non-test source file; gate asserts test-count unchanged + zero regressions + target passes. | Closes the Goodhart cheats (edit the spec, drop tests, break the untested). Without it, parallel best-of-N just finds the exploit faster. |

## Consequences

- **New surface:** a `Target.SourceFile` case (or equivalent) in `Mutation`; a
  worktree-based gate runner that produces `Selection.Performance` from
  `dotnet build` + `dotnet test -p:NuGetAudit=false`; a best-of-N parallel
  orchestrator; a `self-improve/*` branch-commit promotion path replacing the raw
  `File.Copy` in `ApplyDecision` for source targets.
- **Reuses:** `Selection.evaluate` (acceptance logic), `analyzeAndPropose` (repair),
  `GitIntegration.executeGit` (worktree commands), the worktree/parallel patterns
  from the grammar-mesh.
- **First usage (verifiable):** a short curated `(test, file)` seed list of real
  skipped/`xfail` tests TARS earns to green — the concrete "good usage."
- **Cost reality:** N concurrent full `dotnet test` runs are heavy; a tiered gate
  (build + targeted test to filter, full suite only before promote) is a likely
  optimization but deferred — v1 keeps the gate simple and correct.

## Deferred / Phase 2

- Heuristic fault localization (derive the file from a test, dropping the curated list).
- Perf/quality fitness signals (BenchmarkRunner already has an execution-time axis).
- Tiered gate for throughput; open-scope (mutate any file) once containment is proven.
- Couple to `SelfTrain`: feed verified self-hosting diffs into the SFT dataset so
  source wins become model weights (closes levels 5→4).

## Gate mechanic — spiked & verified (2026-06-21)

The core unproven mechanic was proven end-to-end before committing to implementation:

1. `git worktree add --detach <path> HEAD` — succeeds (repo already runs ~9 worktrees).
2. Fresh restore + `dotnet build` + `dotnet test -p:NuGetAudit=false` in the worktree —
   green baseline (6/6 target tests). Windows path length + fresh restore are fine.
3. Applied a deliberate source mutation in the worktree (`buildSweep` → `[]`).
4. Re-ran with `--logger trx` → the **exact** target test flipped to `Failed`, the
   other 5 stayed `Passed`. **Per-test outcomes parse cleanly from the TRX** — this is
   the gate's signal for "target now passes" + "zero regressions" + "test-count
   unchanged" (D1/D6).
5. `git worktree remove --force` — clean; main tree untouched throughout (D2).

**Conclusion:** the worktree gate is feasible as designed. TRX (not console scraping)
is the result source. Build-failure (uncompilable variant) is a distinct gate outcome
from test-failure — both map to `Rollback`.

## Open items to resolve in implementation

- Exact `(test, file)` seed list (which skipped/`xfail` tests first).
- `Target.SourceFile` representation + how the orchestrator selects the target test.
- TRX result-diffing helper (baseline vs variant) feeding `Selection.Performance`.
