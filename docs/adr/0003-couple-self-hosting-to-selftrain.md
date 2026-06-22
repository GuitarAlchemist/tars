# ADR 0003 — Couple the self-hosting gate to SelfTrain (the compounding loop)

- **Status:** Accepted (design) — 2026-06-22
- **Context source:** session on "how far can we improve TARS." Follows ADR 0002
  (self-hosting gate) and ADR 0001 (grammar-mesh). Glossary in `/CONTEXT.md`.
- **Question:** How do verified self-hosting diffs become model weights, so search
  → verify → consolidate becomes one compounding pipeline instead of three
  separate loops?

## Context

TARS self-improves at several loosely-coupled levels. The two that matter here:

- **`SelfHostingGate` (ADR 0002)** — on Accept, produces a *test-verified* source
  edit (zero regressions, target test flips to pass) and commits it to a
  `self-improve/*` branch. The Accept gate is a trustworthy, hard-to-game filter.
- **`SelfTrain.exportDataset`** — builds an SFT dataset (JSONL of
  `{messages:[system,user,assistant]}`) from **verified** benchmark attempts
  (`Validated && PropertiesValidated <> Some false`), prefers the fastest verified
  variant per problem, drops a Modelfile, and `self-train cycle` A/B-measures the
  fine-tuned model vs baseline.

These never meet: the gate's verified wins are thrown away after the branch commit;
the SFT dataset only ever sees benchmark solutions. The model never learns from its
own verified source fixes, so the loop doesn't compound.

**The key property to preserve:** SFT examples must be *verified*, never merely
plausible — pure self-distillation on unverified output collapses (model eats its
tail). The gate's Accept is precisely the verification that makes a self-hosting
example safe to train on. This is the anti-collapse anchor.

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | **On Accept, the gate emits a self-hosting SFT example** to a durable store (`~/.tars/self_host_wins.jsonl`), one record per verified diff. | The verified diff is the scarce, high-value signal; it must outlive the branch commit. Best-effort append, same discipline as `pattern_outcomes.json`. |
| D2 | **The training example mirrors the *generation* task**, not the patch format. `system` = the self-improvement system prompt; `user` = the failure prompt `analyzeAndPropose` already builds (file + error + trace); `assistant` = the `{rationale, old_text, new_text}` JSON it is asked to emit. | Trains the model on exactly the task it performs at generation time, so a better model proposes better edits next round. Same `{messages}` shape as benchmark SFT → merges into one dataset. |
| D3 | **`SelfTrain.exportDataset` gains self-hosting wins as a second source**, merged into the same JSONL (a new `exportWithSelfHosting` or an extra collect step). | One dataset, one fine-tune, one A/B. The model learns both "solve this problem" and "fix this failure" from verified data. |
| D4 | **Only Accept'd (verified-green, zero-regression) diffs are eligible** — Rejects never enter the dataset. | The gate *is* the quality filter, exactly analogous to `Validated`/`PropertiesValidated`. Preserves the verified-only invariant that prevents collapse. |
| D5 | **The A/B cycle (`self-train cycle`) is the success metric**: does the self-hosting-coupled model beat baseline on the benchmark + on closing seed-list tests? | Closes the meta-loop with the same trustworthy measurement SelfTrain already uses — no new unverified signal introduced. |

## The compounding pipeline (what this unlocks)

```
grammar-mesh / LLM  →  propose edits        (search, ADR 0001)
SelfHostingGate     →  verify (tests)       (trustworthy filter, ADR 0002)
self_host_wins.jsonl→  collect verified     (D1/D4, this ADR)
SelfTrain           →  fine-tune on wins    (D3)
better model        →  proposes better edits→ (loop)
```

Each turn, only test-verified diffs feed the model; the better model proposes
better edits; the gate keeps the bar. Externally grounded (real tests), so it
sharpens without collapsing — the realistic version of recursive self-improvement.

## Consequences & honest limits

- **New surface:** a `SelfHostingExample` record + emit-on-Accept in `SelfHostingGate`;
  a loader + merge in `SelfTrain`. Small, additive.
- **Reuses:** the SFT `{messages}` format, `BenchmarkRunner` prompt builders, the
  `self-train cycle` A/B harness, the gate's Accept signal.
- **Volume reality:** self-hosting wins are scarce (each needs a real failing test
  fixed), so early on the dataset stays benchmark-dominated. The coupling's value
  grows with the `(test,file)` seed list and the best-of-N hit rate. `log()` the
  self-hosting example count so the dilution is visible, never silent.
- **Ceiling unchanged:** this sharpens the model toward producing better *verified*
  edits; it does not let TARS exceed what its tests can verify. Raising that ceiling
  is spec-generation (proposing new tests), explicitly out of scope here.

## Prototype — implemented & verified (2026-06-22)

The emit + merge halves are built end-to-end:

- **Emit (`SelfHostingGate`):** `buildSftExample` (pure) distills a `GateTask` into one
  `{messages:[system,user,assistant]}` JSONL line with the mutation JSON as the
  assistant target (D2); `recordWin` appends it to `~/.tars/self_host_wins.jsonl`
  (best-effort); `runGate` calls `recordWin` on Accept (D1/D4 — verified-only).
- **Merge (`SelfTrain.exportDataset`):** reads `self_host_wins.jsonl` and appends those
  lines to the SFT dataset; `ExportStats.SelfHostingExamples` counts them; the CLI
  prints the count and the zero-guard now treats wins as real data (no silent dilution).
- **Tests:** `buildSftExample` shape unit-tested (15 gate tests green). Verified live
  via fsi: `recordWin` → `exportDataset` merged 1 example into the dataset.

This is the verifier-quality upgrade made concrete: the SFT dataset now has a
**test-verified** signal source (source fixes the gate proved green), not just
plausible benchmark output.

## Open items to resolve in implementation

- Exact `SelfHostingExample` schema + the failure-prompt capture (the gate currently
  takes a supplied edit; capturing the *generation* prompt needs the LLM-driven path).
- De-dup / cap policy when the same `(test,file)` is fixed repeatedly.
- Whether self-hosting examples get a higher sampling weight than benchmark examples.
