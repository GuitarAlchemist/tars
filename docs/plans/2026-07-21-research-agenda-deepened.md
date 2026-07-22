---
title: 'TARS Research Agenda — Deepened Plan'
date: 2026-07-21
source: docs/research/README.md §4
status: deepened
---

# TARS Research Agenda — Deepened Plan

This is the implementation-ready companion to the committed research agenda (`docs/research/README.md` §4). Each agenda item was deepened by a dedicated research agent and then adversarially reviewed; this document folds both passes. The committed research README stays untouched — this file is the working plan. All code anchors are as-audited on 2026-07-21; several reviews found line references already drifted by a few lines, so **re-anchor every `file:line` at implementation time** rather than trusting the snapshot.

## 1. Enhancement Summary

Deepened 2026-07-21 across **8 sections** (agenda items 1+2 merged, 3, 4, 5, 6, 7, 8, 9). Key improvements over the raw agenda:

- **Exact defect mechanics named at the line level.** The `sprintf "%A"` codec corruption (`PatternSelector.fs:33`), the O(n²) read-modify-write append (`:80-86`), the circular governor-decision signal (`PromotionPipeline.fs:353-359`), the dead `extra_body` vLLM wire format (`OpenAiCompatibleClient.fs:122-129`), and the always-false `RemovesComplexity` after the round-3 fix (`PromotionPipeline.fs:202` vs `run:332`) are each pinned to a one-line root cause with a surgical fix.
- **Literature-grounded designs, not novel inventions.** Beta-Bernoulli Thompson sampling (Agrawal & Goyal 2012), Stitch-style MDL utility (Bowers et al. POPL 2023), POET/GoalGAN admission bands, CRAG escalation ladders, replicator-mutator dynamics (Page & Nowak 2002), STaR/ReST-EM SFT hygiene — every mechanism maps to a published recipe.
- **Aggressive scope cuts from review.** Reviews consistently found ~30-50% of each plan speculative: shipped migrations for one-shot data fixes, CLI subcommands for throwaway scripts, MAP-Elites archives with no consumers, DPO schemas with no trainer, FsCheck as a new dependency for a 7-branch codec. The v1 scope per item is now explicitly marked.
- **Layering and F# compile-order violations caught before implementation.** Five concrete blockers found: Cortex↔Evolution reference direction (items 3, 5, 7, 9), `PromotionPipeline` → `PromotionIndex` compile-order impossibility (item 5), `SelfTrain.fs` ↔ `SelfHostingGate.fs` mutual-order unsatisfiability (item 8), `RetroactionLoop` → `BenchmarkRunner` ordering (item 6), and invalid `?config` syntax on module functions (item 9).
- **F# record-shape ripple fully enumerated.** Adding fields to `PatternOutcome`, `WeightedRule`, `RoutedBackend`, and `AgentConfig` breaks every record-literal construction site; reviews enumerated the missed sites (`ClaudeCodeBridge.fs:423`, `Evolve.fs:880`, `GrammarDistillation.fs:284-295`, `GrammarMlBridge.fs:225-236`, 8 test stubs constructing `RoutedBackend`, 8 `builtinDefaults`).
- **Telemetry-contamination channels identified across items.** Gate-B curriculum probes, search probe suites, and gate accept-rate telemetry all threaten to pollute `PatternOutcomeStore` — the single store feeding pattern selection and promotion ranking. Each item now carries an explicit isolation requirement.
- **Cross-item DTO coordination collapsed.** Items 1 and 3 both touch `PatternOutcome`/`PatternOutcomeDto`; the plan lands **one** schema change (provenance fields; `PatternId` deferred until a producer exists), avoiding two format bumps.

New considerations discovered during review (cross-section conflicts):

- `Weight` has **four writers** (Bayesian `updateWeight`, `ReplicatorDynamics`, `fromRecurrenceRecords`, `GrammarMlBridge`) — items 3, 5, and 9 each touch it; an ownership contract (posterior mean as base → replicator on top → normalize last, write-back gated) must be decided once, in item 3's PR, and respected by 5 and 9.
- Item 3's grounded fold (`PatternId` on outcomes) has **zero producers** today; severing the circular signal without a producer freezes grammar-rule posteriors entirely. The severing lands with at least one named producing call site or the fold is deferred.
- Stochastic `Recommend` (Thompson) breaks five existing `Assert.Equal` tests (`PatternSelectorTests.fs:76-100`) and the store reads the live `~/.tars` path with no test seam — test isolation is in-scope work, not a surprise.

## 2. Current State — Round 3 and the Human-Work Frontier

Round 3 of the self-improve loop (2026-07-21, commit `a3f730df`) already closed two agenda-adjacent seeds autonomously: the **template-not-name** criterion fix (`PromotionPipeline.fs:202` now requires template ≠ name) and the **replicator floor** clamp (partial — the post-renormalization residual remains). The gate's behavioral-fix accept rate was 2/5 at N=3 (vs 8/8 on union-case edits in rounds 1-2, N=4) — a difficulty escalation, not a regression.

Three seeds were **rejected by the gate** (best-of-3, none passed) and are the human-work frontier:

| Rejected seed | Agenda item | Residual |
|---|---|---|
| `outcome-kind-codec` | Item 1 | `%A`/`parseKind` corruption + O(n²) append still live; 25 Custom records nested up to 89 quoting layers |
| `beta-prior-degenerate` | Item 3 | `bayesianUpdate` still yields certainty from one observation (repair regressed 2 tests) |
| `posterior-to-weight` | Item 3 | `updateWeight` still never recomputes `Weight` from the posterior (0 applicable proposals) |

Item 9's floor residual (clamp-after-renormalize with no redistribution, `ReplicatorDynamics.fs:127-129`) is a fourth open residual, never seeded. These four are written below for **human-driven surgical implementation**; they are also the frozen re-measurement set for item 8's round-4 protocol. Items 4, 5, 6, 7, 8 are new construction the gate has never attempted.

---

## 3. Item 1 (+2) — PatternOutcomeStore Exact-Inverse Codec, Append-Only JSONL, and Provenance Covariates

### Intent

The audit's #1-ranked opportunity (`docs/research/empirical-learning-curves.md`, Findings 5/8/9): until this lands, `tars evolve --loop N` produces no per-cycle series and the self-improvement claim is unmeasurable from its own telemetry. Two blocking defects: (1) the store corrupts its own data — `toDto` serializes `PatternKind` via `sprintf "%A"` (`PatternSelector.fs:33`), `parseKind` lowercases and rewraps unknowns as `Custom` (`:39-47`), and `record()` round-trips the whole file per append (`:80-86`), adding one quoting layer per append to every Custom record (25 records now 88-89 layers deep, appends O(n²)); (2) the covariates required for learning-curve inference are absent — no cycle/model/strategy fields on `PatternOutcome` (`:17-22`), and `BenchmarkRunner` hardcodes `ModelUsed = "default"` (`BenchmarkRunner.fs:370`).

### Best practices

Exact-inverse codec discipline: any persisted identifier needs a total, injective print function and a parser proven to be its left inverse (`parse ∘ print = id`), never a pretty-printer. JSON Lines for append-only telemetry (jsonlines.org): one value per `\n`-terminated line, `WriteIndented=false`, O(1) `File.AppendAllText`, per-line tolerant reads — the repo already uses this for `~/.tars/self_train/dataset.jsonl` (`Evolve.fs:793-797`). Keep the DTO boundary STJ-primitive (strings, no `FSharpOption`, no DUs); domain types stay rich, mapping lives in `toDto`/`fromDto`. Provenance (CycleId/ModelId/SelectorStrategy) is the minimum covariate schema; session-scoped cycle ids (`{guid}/{index}`) make cycles orderable within a run and joinable across stores without a central counter.

### Implementation steps

1. **Codec** (`v2/src/Tars.Cortex/PatternSelector.fs`). Replace `sprintf "%A" o.PatternKind` (`:33`) with a total, injective `kindToString`: the six fieldless cases (`WoTTypes.fs:134-141`) map to exact case names; `Custom name` → `"Custom:" + name`, payload verbatim (no lowercasing, no quoting). Replace lossy `parseKind` (`:39-47`) with `parseKindCanonical`: exact case-sensitive match on the six tags; `StartsWith "Custom:"` → `Custom (s.Substring 7)` (further colons preserved); anything else → `Custom s` (idempotent parse). Keep the old lenient matcher only for a one-off recovery pass (see step 5, as revised).
2. **JSONL options.** Second `jsonlOptions = JsonSerializerOptions(WriteIndented = false, PropertyNamingPolicy = CamelCase)` beside the indented options (`:56-59`). `WriteIndented` MUST be false — one record per line.
3. **Append-only record.** Add `outcomeJsonlPath()` → `~/.tars/pattern_outcomes.jsonl` at the path seam (`:61-65`). Rewrite `record()` to serialize ONE DTO and `File.AppendAllText(path, json + "\n")` — no `loadAll`, no re-serialize, O(1). Keep the best-effort try/with but route the swallowed exception to a debug log — total silence is how 89 layers went unnoticed.
4. **Tolerant reader.** `loadAll()`: `File.ReadLines` + per-line `try Deserialize |> Some with _ -> None` via `List.choose` — a torn final line from a crash mid-append must not poison the store (the current `with _ -> []` would lose all 231+ records).
5. **Legacy handling** *(revised per review)*: do **not** ship a permanent `migrateLegacyStore`. On first store access, rename `pattern_outcomes.json` → `.json.bak` and start the JSONL fresh; recover the 25 `benchmark:<category>` records with a **one-off scratchpad script** (regex `benchmark:[a-z]+`, canonical casing restored from an inlined lowercase→canonical list — do NOT reference `ProblemCategory` from Cortex, that creates the Cortex→Evolution cycle). The 12 seeded March records migrate as-is (purging is the audit's separate opportunity 5). Optionally delete the 0-byte orphan `pattern_outcomes_new.json`.
6. **Sibling `%A`.** `kindKey` (`PatternSelector.fs:212`) also uses `sprintf "%A"` and would send multi-line rule ids to the ix `grammar.weights` skill for Custom kinds; point it at `kindToString`. Per review, its inverse `keyToKind` (`:214-222`) stays a lossy Contains-matcher returning None for all Custom kinds — either pair it with `parseKindCanonical` or explicitly document that the ix path excludes Custom arms. Same bug class exists in `parsePatternKind` (`:224-232`, GoldenTraceStore) — out of scope, tracked as follow-up.
7. **Provenance schema.** Extend `PatternOutcome` (`:17-22`) with `CycleId/ModelId/SelectorStrategy : string option`; DTO gets plain strings (`""` = unknown, null-normalized on load — STJ constructor deserialization yields null for members missing from old lines). No `FSharpOption` on the wire.
8. **Record seam** *(revised per review)*: use F# optional member parameters — `member _.RecordOutcome(kind, goal, success, durationMs, ?cycleId, ?modelId, ?strategy)` — instead of a new `OutcomeProvenance` type + `Empty` + overload; the two existing call sites (`TarsWoTAgent.fs:184, 245` — plan's 193/254 were stale) compile unchanged and gain `SelectorStrategy = "HistoryBased"` stamping. Pin the strategy-string mapping in one `strategyName : SelectorStrategy -> string` beside the DU (`:97-100`), never scatter literals. **Also enumerate the direct constructors the original plan missed**: `ClaudeCodeBridge.fs:423` and `Evolve.fs:880` construct `PatternOutcome` literally and break on the field additions — add a smart constructor (`PatternOutcome.Create` with empty provenance) so future sites don't repeat the omission.
9. **Benchmark provenance** *(revised per review)*: keep `ModelUsed = modelId` replacing the hardcoded `"default"` (`BenchmarkRunner.fs:370`) — that is a value change, no schema change. Do **not** add `CycleId` to `BenchmarkRunSummary` (persisted field, 7-param function growth, null-normalization in `loadHistory` for 7 old files, one consumer). Instead change only `recordOutcomes (summary) (cycleId: string option)` and stamp outcomes there; `RunId+Timestamp` already join runs to cycles if ever needed.
10. **Model-id resolution** *(revised per review)*: do not write a parallel `resolveModelId` that "mirrors" `createWithModel` (`LlmFactory.fs:53-57`) — drift by construction. Extract the resolution `createWithModel` already does and have both call it, or have `create`/`createWithModel` return the resolved id alongside the `ILlmService`. Recorded id = configured intent at creation; `createWithFallback` (`:66-88`) can silently substitute Claude Code — documented approximation, the deeper `ILlmService.Describe` seam is a later PR.
11. **Wire call sites.** `CodeBenchmark.fs:112-124` (pass resolved model, cycleId None); `Evolve.fs:258-261 + 769-780`: one `sessionId = Guid.NewGuid()` per evolve start, `cycleId = Some $"{sessionId:N}/{cycle}"` inside the loop (`:615`); also `CrossRepoDemo.fs:174-179`, `SelfTrainCommand.fs:99-100`.
12. **Tests** (`tests/Tars.Tests`, xUnit) *(revised per review)*: no FsCheck — it is a new dependency for one property over a nearly-finite type. An xUnit `[Theory]` with the ~10 adversarial payloads (newline, quote, extra colons, literal `Custom:` prefix, payload equal to a case name, empty string) exercises every branch of a 7-branch codec: assert `parseKindCanonical (kindToString k) = k`. Plus: append-only behavior (record 3, assert 3 lines; record a 4th, assert first 3 byte-identical — proves no rewrite); provenance (old-format line without the three fields loads all-None; `recordOutcomes` stamps ModelId/CycleId on every attempt).

### Signature sketches (not yet applied)

```fsharp
// PatternOutcomeStore (v2/src/Tars.Cortex/PatternSelector.fs)
val internal kindToString      : PatternKind -> string  // total, injective; Custom n -> "Custom:" + n verbatim
val internal parseKindCanonical: string -> PatternKind  // exact inverse: parseKindCanonical (kindToString k) = k

type PatternOutcome =
    { PatternKind: PatternKind; Goal: string; Success: bool
      DurationMs: int64; Timestamp: DateTime
      CycleId: string option; ModelId: string option; SelectorStrategy: string option }
    static member Create : PatternKind * string * bool * int64 -> PatternOutcome  // empty provenance

type PatternOutcomeDto =   // STJ-primitive: no DUs, no FSharpOption; "" = unknown, null normalized on load
    { PatternKind: string; Goal: string; Success: bool
      DurationMs: int64; Timestamp: DateTime
      CycleId: string; ModelId: string; SelectorStrategy: string }

val record  : PatternOutcome -> unit        // O(1): one DTO, File.AppendAllText(jsonl, line + "\n")
val loadAll : unit -> PatternOutcome list   // per-line tolerant JSONL read

// HistoryAwareSelector — optional params, no new public type
member RecordOutcome :
    PatternKind * goal:string * success:bool * durationMs:int64 *
    ?cycleId:string * ?modelId:string * ?strategy:string -> unit

// BenchmarkRunner.fs — cycleId at the recording seam only
val recordOutcomes : BenchmarkRunSummary -> cycleId:string option -> unit
```

### Edge cases

- Adversarial Custom payloads: newlines, quotes, `benchmark:musictheory` colons, literal `Custom:` prefix, payload equal to a case name. `Custom "ChainOfThought"` serializes to `Custom:ChainOfThought`, distinct from the bare tag.
- Parse idempotence vs strictness: non-canonical legacy blobs must not re-enter via the canonical parser — `parseKindCanonical` maps unknowns to `Custom s` (fixed point after one application).
- Torn final JSONL line: skip, never fail the whole load.
- Concurrent appenders (evolve loop + manual benchmark): `File.AppendAllText` opens exclusively; the loser's outcome drops through the existing best-effort try/with — acceptable for telemetry, log it, no speculative cross-process locking.
- Old lines missing the three provenance members deserialize those constructor params to null — normalize to None in `fromDto`.
- Timestamps mix 7-digit-subsecond and whole-second values; STJ ISO-8601 round-trips both — never reformat (the timestamp signature is itself audit evidence).
- CycleId collision: bare cycle index repeats per invocation; the session-guid prefix keeps series globally unique.

### Pitfalls

- `%A` is a debug pretty-printer with no format-stability contract — it renders `Custom "x"` multi-line and can change between compiler versions. This single call is the root cause of the 89-layer nesting.
- Lowercasing in a parser destroys payload case irreversibly; substring `Contains` matching misclassifies (`Custom "supply-chain-audit"` parses as ChainOfThought via "chain").
- Read-modify-write append compounds any codec asymmetry into progressive corruption — one lossy round-trip per append, forever. Append-only is the fix, not a faster rewrite.
- `WriteIndented=true` silently breaks JSONL; reusing the existing indented options is the obvious trap.
- `with _ -> ()` swallowing everything is how a month of corruption went unnoticed — log the failure path.
- Purging seeded March records or renaming problem ids "while in there" violates surgical scope (separate audit opportunities).

### Review notes

Reviews trimmed the plan substantially: (1) the shipped `migrateLegacyStore` + `parseKindLegacy`-kept-forever + byte-level-no-op tests are oversized for a payload whose recovered records lack the very covariates the item exists to add — rename-to-`.bak` + one-off scratch script instead; (2) no `CycleId` on `BenchmarkRunSummary` — stamp at `recordOutcomes`; (3) no parallel `resolveModelId` mirror — one authoritative resolution path; (4) no FsCheck dependency — enumerated `[Theory]`; (5) no `OutcomeProvenance` ceremony — optional member params; (6) migration casing-restore must not reference `ProblemCategory` from Cortex (layering); (7) scope undercount: `ClaudeCodeBridge.fs:423` and `Evolve.fs:880` also construct `PatternOutcome`; (8) the ix seam fix is one-directional — pair `keyToKind` with the canonical parser or document the Custom-arm exclusion; (9) placement verified correct (codec in Cortex, provenance threading in Evolution, model resolution in CLI-layer LlmFactory; Evolution→Cortex→Llm direction preserved, no new edges); (10) stale line refs (`TarsWoTAgent.fs:184/245`, not 193/254) — re-anchor at implementation.

### Effort

Medium — one focused day. Codec + JSONL localized to PatternOutcomeStore (~120 LOC); provenance threading mechanical but touches 6 files and must keep existing call sites and old persisted files loadable.

### Success criterion

```
/goal All hold: (a) dotnet test passes in v2/ including an enumerated-payload roundtrip
theory asserting parseKindCanonical (kindToString k) = k for all PatternKind values incl.
adversarial Custom payloads; (b) an append test shows recording a 4th outcome leaves the
first 3 JSONL lines byte-identical, each line parsing standalone; (c) the legacy file
survives as pattern_outcomes.json.bak and the one-off recovery yields Custom kinds
matching ^Custom:benchmark:[A-Za-z]+$; (d) `tars benchmark code run --model
qwen2.5-coder:7b --max 1` produces a run file with "modelUsed": "qwen2.5-coder:7b"
(no "default" in new runs) and a JSONL outcome whose modelId equals it; (e) `tars evolve
--loop 2 --benchmark` appends outcomes carrying exactly two distinct non-empty cycleIds
sharing one session prefix.
```

### References

Research: `docs/research/empirical-learning-curves.md` Findings 5, 8, 9, 10; Opportunities 1-2. Code: `PatternSelector.fs:17-22, 33, 39-47, 56-59, 61-65, 68-77, 80-86, 97-100, 212, 250-254, 373`; `WoTTypes.fs:134-141`; `BenchmarkRunner.fs:334, 370, 381, 392-399, 418`; `BenchmarkTypes.fs:6-16, 68-78`; `Evolve.fs:258-261, 615, 769-780, 793-797`; `CodeBenchmark.fs:104-128`; `LlmFactory.fs:34-57`; `Routing.fs:14-43`; `TarsWoTAgent.fs:184, 245`. External: jsonlines.org; ndjson.com; Tarmil/FSharp.SystemTextJson (library alternative, deliberately not taken); `File.AppendAllText` .NET 10 docs.

---

## 4. Item 3 — Statistical Soundness: Grounded Signal, Beta(1,1) Priors, Posterior-to-Ranking, Thompson Sampling

### Intent

Restore soundness to the Force-2 Bayesian layer (`docs/research/theory-bayesian-grammar-induction.md`, Findings 1-5) in one coherent change: (1) reparametrize `WeightedRule` around explicit Beta pseudo-counts with an always-present Beta(1,1) prior so a fresh rule's first observation yields 2/3, not certainty; (2) stop feeding the Grammar Governor's own Approve decision into `updateWeight` (the circular signal); (3) recompute `Weight` from the posterior mean on every update so `classifyWeighted`'s ordering responds to evidence (fixing the live r=-0.85 weight/fitness inversion); (4) replace mean-softmax-argmax in `Recommend` with Thompson sampling over Beta posteriors whose priors encode the keyword heuristics as fixed pseudo-count mass. Round 3 seeded red tests for pieces 1 and 3 (`SelfImproveRound3Tests.fs` Seeds 2 and 3) and the autonomous repair closed neither — make the seeds green first, then layer sampling behind them. Standard Beta-Bernoulli bandit recipe throughout (Agrawal & Goyal 2012; Chapelle & Li 2011; Russo et al. 2018; discounted-TS per Qi & Wang 2023).

### Best practices

Beta(1,1) prior with θ_k ~ Beta(S+1, F+1) is canonical Bernoulli-TS with O(log T) regret; keep the +1 terms in the read-time math, **never in the stored counts**, so decay can never erode the prior. Decay only empirical counts (discounted TS bounds effective sample size at 1/(1-γ) = 20 for γ=0.95); non-integer decayed counts are fine. Encode side information (keyword heuristics) as prior pseudo-counts α₀ = 1 + κ·h, β₀ = 1 + κ·(1−h) — never blended point estimates — so data overturns the prior at a rate governed by κ. Separate observation likelihood from decision policy: outcomes update the posterior; governor decisions only gate. Keep the stochastic path (`Recommend`) and deterministic path (`Score`) distinct. Inject the RNG (constructor param, default `Random.Shared`); sample Beta via two Marsaglia–Tsang Gamma draws — ~30 lines of F#, no MathNet. TS tolerates the batched per-cycle update regime gracefully.

### Implementation steps

**v1 scope (per review): steps 1-3 + the step-7 severing decision + tests — that alone makes the seeds green and satisfies criteria (a)/(b)/(d). PatternId threading and the grounded fold (original steps 5-7) are deferred until a producer exists — see Review notes.**

1. **Posterior state** (`v2/src/Tars.Evolution/WeightedGrammar.fs:30-48`). Add `Alpha: float`, `Beta: float` (empirical, decayed pseudo-counts; the Beta(1,1) prior is NOT stored — added at read time). Same fields on the DTO (`:196-207`). Migration in `fromDto` (`:223-242`): STJ defaults missing floats to 0.0, so when `Alpha + Beta <= 0.0` reconstruct `Alpha = SuccessRate * float SelectionCount`, `Beta = (1.0 - SuccessRate) * float SelectionCount`. Accessors: `posterior r = (r.Alpha + 1.0, r.Beta + 1.0)`, `posteriorMean`, `posteriorStd = sqrt(ab / ((a+b)² (a+b+1)))`. **Review addition:** `GrammarDistillation.fs:284-295` and `GrammarMlBridge.fs:225-236` construct `WeightedRule` literals — initialize the new fields with the same reconstruction, not 0.0, or those rules start as false fresh rules. Persist (Alpha, Beta) as source of truth; derive SuccessRate/Confidence/Weight at read time rather than writing four representations that drift.
2. **De-degenerate `bayesianUpdate`** (`:94-106`), signature preserved (Seed 2 calls `bayesianUpdate 0.5 0 true 0.95` directly). New body: `eff = float priorCount * decayFactor`; `alpha = 1.0 + priorRate*eff + 1{success}`; `beta = 1.0 + (1.0-priorRate)*eff + 1{failure}`; `newRate = alpha/(alpha+beta)`; `newConfidence = 1.0 - 2.0*posteriorStd` clamped [0,1]. Hand-verify: fresh rule + one success = 2/3 (Seed 2 at `SelfImproveRound3Tests.fs:52-60`); (0.5, 10, true) = 0.54 > 0.5, (0.5, 10, false) = 0.46 < 0.5; confidence monotone in count — the three pinned tests at `ProbabilisticGrammarTests.fs:113-127` stay green. Per review: do **not** invent a `PriorPseudoCount` config knob to make the dead `PriorSuccessRate` "representable" — delete the dead field or leave it untouched.
3. **`updateWeight` recomputes Weight** (`:136-147`, Seed 3). Decay empirical counts only (`alpha' = Alpha*γ + 1{s}`; `beta' = Beta*γ + 1{f}`), then `SuccessRate = (alpha'+1)/(alpha'+beta'+2)`, `Confidence = 1 - 2*posteriorStd`, `Weight = max config.MinWeight posteriorMean`, `SelectionCount+1`. Hand-check Seed 3 (`:70-88`): migrated (5,5); 10 failure folds at γ=0.95 drive posteriorMean to ~0.23 < 0.5 → Weight falls → green. **Weight-ownership contract (review):** posteriorMean is the base; `ReplicatorDynamics.fs:90/190` applies on top; `normalizeByLevel` last; `fromRecurrenceRecords` softmax seeding and `GrammarMlBridge` blending must be reconciled to this order or the replicator pass clobbers the posterior one cycle later.
4. **Thompson primitives** — per review these live in **Tars.Cortex next to their only consumer** (`PatternSelector.fs`), NOT in `WeightedGrammar.fs`: Evolution references Cortex, not vice versa, so Cortex cannot see samplers defined in Evolution. `sampleGamma` (Marsaglia–Tsang, with the shape<1 boost trick), `sampleBeta a b = ga/(ga+gb)` (clamp draws to [1e-12, 1-1e-12]), `thompsonSelect rng arms = argmax over draws`. rng is a parameter, never constructed per call. Moment test: mean within 0.02 of a/(a+b) over 10k seeded draws.
5. **Sever the circular signal** (`PromotionPipeline.fs:353-359`). Delete the `Approve → updateWeight success=true` block — governor decisions gate, they are not observations. **Binding review condition:** because no current `record()` call site knows a grammar-rule PatternId (all record PatternKind only), deleting the update without landing a producer means WeightedRule posteriors receive no observations ever — worse than the circular signal. Land at least one concrete producing call site in the same change (BenchmarkRunner or Evolve.fs are natural — both sit above PromotionIndex and can resolve which promoted pattern was applied), or keep the severing and the fold in one deferred change. The plan's fallback (local DTO re-reading the outcomes file) is dead weight — `Tars.Evolution.fsproj:77` already references Cortex; commit to the direct call.
6. **Thompson arms in PatternSelector** (`:247-289`). `armPosteriors goal`: per kind, S/F from `PatternOutcomeStore.loadAll()`; prior α₀ = 1 + κ·h_k, β₀ = 1 + κ·(1−h_k), h_k from `heuristicScore` (`:291-298`), κ = 3.0 (heuristic worth ~3 observations). **Per review: start heuristic-only** — do not fold golden traces or promotionBoost into the prior in v1 (three hand-tuned constants with no test asserting any matters); add later if a trace shows regression. Drop the ix delegation from this path (its softmax probabilities are not (α,β)); leave a comment.
7. **`Recommend` samples; `Score` stays deterministic** (`:381-386`). `HistoryAwareSelector(?rng: Random)` (default `Random.Shared`); `Recommend = thompsonSelect rng (armPosteriors goal) |> Option.defaultValue ChainOfThought`; `Score` returns posterior means — never samples.
8. **Tests** (`tests/Tars.Tests/`): (a) Seeds 2 and 3 green with zero edits to seed bodies; (b) regression: 10-failure rule sorts below 10-success rule in `classifyWeighted`; Pearson r(SuccessRate, Weight) > 0 on a synthetic 10-rule store; (c) exploration: seeded rng, arms A=20/25, B=0/5, C=5/10, keyword-neutral goal, 200 Recommend calls, B selected ≥ 10 times; (d) DTO migration round-trip (old-shape JSON without Alpha/Beta loads with reconstruction); (e) sampleBeta moments. **Review addition (in-scope, not a build-time surprise):** `PatternSelectorTests.fs:76-100` has five `Assert.Equal` tests on unseeded `Recommend`, and `HistoryAwareSelector` is constructed bare in `ClaudeCodeBridgeTests.fs:50/79`; the store reads the live `~/.tars` path with no injection point — seed the rng + isolate the store path, or rewrite the five assertions to set-membership.
9. **Live-store reprocess** *(revised per review)*: no new `tars promote` subcommand for a one-shot — the lazy `fromDto` migration converts on load and Weight recomputes on next update; check criterion (b)'s live correlation with a throwaway `dotnet fsi` script in scratch.
10. **Sibling degenerate math** (review): `ResearchWeights.fs:102` documents itself as "similar to WeightedGrammar.bayesianUpdate" — point it at the new posterior helpers or annotate the divergence, or the next self-improve round rediscovers Finding 1 in a different file.

### Signature sketches (not yet applied)

```fsharp
// WeightedGrammar.fs (Tars.Evolution) — prior NOT stored; added at read time
type WeightedRule = {
    (* existing fields *)
    Alpha: float   // decayed empirical success pseudo-count
    Beta: float }  // decayed empirical failure pseudo-count

val posterior     : WeightedRule -> float * float   // (Alpha+1, Beta+1)
val posteriorMean : WeightedRule -> float
val posteriorStd  : WeightedRule -> float

// signature preserved for Seed 2 and pinned callers
val bayesianUpdate : priorRate:float -> priorCount:int -> success:bool -> decayFactor:float -> float * float

val updateWeight : WeightConfig -> WeightedRule -> success:bool -> WeightedRule
// Weight = max config.MinWeight (posteriorMean rule')

// Tars.Cortex (next to PatternSelector — its only consumer; Evolution cannot be seen from Cortex)
val sampleGamma    : Random -> shape:float -> float
val sampleBeta     : Random -> a:float -> b:float -> float
val thompsonSelect : Random -> arms:('k * float * float) list -> 'k option

type HistoryAwareSelector(?rng: Random) =
    member private _.ArmPosteriors : goal:string -> (PatternKind * float * float) list
    // Recommend = thompsonSelect (stochastic); Score = posterior means (deterministic)
```

### Edge cases

- weights.json migration: STJ defaults missing Alpha/Beta to 0.0 — detect `Alpha+Beta <= 0` and reconstruct; a live rule with SelectionCount 4 must not read as "no observations".
- 231 existing outcome records feed the PatternKind-level bandit unchanged.
- Key-space mismatch: outcome arms are PatternKinds; WeightedRule ids name grammar patterns — two arm spaces; never bridge by fuzzy string match through `parseKind`.
- 0/5 vs 195/195: TS correctly almost never picks the 0/5 arm against a dominant one — criterion (c)'s simulation must use moderate competitors or it flakes.
- Confidence consumers: `GrammarMlBridge`'s max-blend consumes Confidence — `1 - 2*posteriorStd` changes scale; posteriorStd shrinks in n so monotonicity holds; verify the pinned test and the blend.
- Empty store / fresh install: all arms sit at heuristic priors; `thompsonSelect` still returns Some; `Option.defaultValue ChainOfThought` covers zero arms.
- `normalizeByLevel` (`:168-174`) rescales after updates — compute criterion (b)'s correlation before normalization or per level.

### Pitfalls

- Do NOT keep the governor update "as well as" the fold — feeding both re-introduces the circularity; the Approve branch is deleted, not augmented.
- Do NOT softmax posterior means: means in [0,1] bound the probability ratio at e ≈ 2.72, compressing 195/195 vs 0/5 into ~0.03 of score — count information must survive to the sampler.
- Do NOT store prior mass inside Alpha/Beta and decay it — reproduces the degenerate collapse asymptotically.
- Do NOT sample inside `Score()` — it feeds logs and tests.
- Do NOT construct `Random()` per call — same-tick constructions correlate.
- Do NOT re-apply the full outcome history every cycle (exponential double-counting under decay) — watermark first if/when the fold lands.
- Do NOT hand-tune an epsilon floor before trying plain TS — 0/5-arm exploration falls out of posterior width for free.
- Round-3 post-mortem guard: the failed repair regressed by touching `bayesianUpdate`'s signature — this plan deliberately keeps it and hand-verifies all five pinned assertions before build.

### Review notes

Decisive corrections: (1) **layering** — samplers move to Cortex (Evolution→Cortex is the only edge; verified in both .fsproj); (2) **signal vacuum** — the severing must ship with a named PatternId producer or freeze grammar posteriors; steps 5-7 of the original plan (PatternId + `applyExecutionOutcomes` + watermarked fold) are deferred as a unit until then, which also dissolves the DTO-migration coordination with item 1; (3) **four Weight writers** need one ownership contract, decided here; (4) prior-encoding starts heuristic-only (single κ), ix delegation dropped from the posterior path; (5) no shipped reprocess subcommand; (6) record-shape ripple enumerated (GrammarDistillation, GrammarMlBridge constructors); (7) existing deterministic Recommend tests and the live-path store are in-scope test work; (8) `ResearchWeights.fs:102` sibling annotated.

### Effort

3-5 days: ~1 day reparametrization + migration + seeds green; ~1 day severing + producer call site; ~1 day Thompson sampling + sampler; ~1 day tests + live-store check + `verify.ps1`.

### Success criterion

```
/goal From v2/, dotnet build && dotnet test pass with: (a) SelfImproveRound3Tests Seeds 2
and 3 green with zero edits to seed test bodies; (b) a 10-failure rule sorts below a
10-success rule in classifyWeighted and a synthetic 10-rule store yields Pearson
r(SuccessRate, Weight) > 0 (scratch fsi check confirms r > 0 on the live
~/.tars/promotion/weights.json); (c) seeded rng, 200 Recommend calls over arms A=20/25,
B=0/5, C=5/10 with a keyword-neutral goal select arm B >= 10 times; (d) grep -n "Approve"
v2/src/Tars.Evolution/PromotionPipeline.fs shows no line feeding a governance decision
into WeightedGrammar.updateWeight, AND at least one call site records a grammar-rule-level
outcome; pwsh Scripts/verify.ps1 passes.
```

### References

Research: `docs/research/theory-bayesian-grammar-induction.md` Findings 1-5, Opportunities 1-3; agenda README item 3. Code: `WeightedGrammar.fs:62-67, 94-106, 113-133, 136-147, 150-165, 196-242`; `PromotionPipeline.fs:172-183, 310-320, 353-359`; `PatternSelector.fs:14-86, 247-298, 354-386`; `SelfImproveRound3Tests.fs:44-88`; `ProbabilisticGrammarTests.fs:113-127, 157-170`. Papers: Russo et al. 2018 (arXiv:1707.02038); Agrawal & Goyal 2012 (arXiv:1111.1797); Chapelle & Li 2011 (NeurIPS); Qi & Wang 2023 (arXiv:2305.10718); Garivier & Moulines 2008 (arXiv:0805.3415); Marsaglia & Tsang 2000 (ACM TOMS 26(3)); Hongsup Shin 2025 practice notes.

---

## 5. Item 4 — Activate Constrained Decoding + Constraint-Aware Router (Type-3 → Type-5 Hop)

### Intent

The decode-time constraint path is fully plumbed but dead at three independent layers (`docs/research/theory-neuro-symbolic.md` Findings 1, 2, 8): no production code issues a `Constrained` request (every live evolution call is bare `JsonMode=true` — `Engine.fs:215/455`, `Evaluation.fs:110`, `Reflection.fs:74`, `Optimizer.fs:63`, `SymbolicReflector.fs:81`); the router never inspects `ResponseFormat` and Ollama silently discards grammars (`Routing.fs:133-228`, `OllamaClient.fs:186/355`); and `grammars/wot.ebnf` describes a language WotParser does not even parse. Additionally the existing vLLM wire format is a latent no-op: the nested `{"extra_body": {"guided_decoding": ...}}` property (`OpenAiCompatibleClient.fs:122-129`) was never the server-side API (extra_body is a Python-SDK client-side kwarg), and vLLM v0.12 removed all `guided_*` fields in favor of top-level `structured_outputs`. Syntactic failures become unrepresentable; retry budget shifts to semantic failures only.

**Per review, this splits into two real PRs:** slices A+B (schemas + call-site swap + router + wire formats) are complete and verifiable with no EBNF grammar existing; slice C (wot.ebnf rewrite + conformance harness + planner wiring) is a follow-up — and with C deferred, the `NeedsCfg` routing branch can also wait, shrinking the router change to "JsonSchema-capable or downgrade".

### Best practices

Schema-per-call-site, colocated with the prompt template it mirrors (schema drift from the prose "Return ONLY this JSON" instruction is the #1 structured-output bug class). Route by capability, not provider name — closed `ConstraintNeed` type + `supports` predicate, mirroring the repo's `ModelFamily`/`RoutingHint` pattern (`Routing.fs:86-125`). Keep routing pure; carry the downgrade in the return value and log at the service boundary where `ILogger` lives. Let the server pick the constraint backend (vLLM `--structured-outputs-config.backend`, default auto) — hardcoding "xgrammar"/"outlines" client-side coupled TARS to server internals that changed twice in two years. Constrain IR and judgments; never open-ended code generation (Grammar-Aligned Decoding: naive masking distorts distributions — syntax to the decoder mask, semantics to the external verifier). Fail-loud-degrade-gracefully: warning log always; the evolve loop's availability on default Ollama is a product requirement.

### Implementation steps

1. **Schemas.** New `v2/src/Tars.Evolution/EvolutionSchemas.fs` (before `Engine.fs` in compile order): one JSON-schema string per live call site, colocated with the existing `jsonTemplate` — `contradictionSchema` `{contradicts:bool, reason:string}` (Engine.fs:190-249); `evaluationSchema` mirroring `Evaluation.fs:81-98`; reflection/optimizer/symbolic-reflection shapes; the Engine.fs:455 direct-response shape. Reuse `ConstrainedDecoding.beliefUpdateSchema`/`intentPlanSchema` (`ConstrainedDecoding.fs:97-150`) where a site matches an existing IR. **Per review, only schema strings live here** — the request-shaping helper belongs in Tars.Llm: use/extend the existing `ConstrainedDecoding.withJsonSchema` (`ConstrainedDecoding.fs:59`), do not add a near-identical `constrainJson` in Evolution (Cortex has four parallel JsonMode sites — `Patterns.fs:1161/2044`, `PreLlmPipeline.fs:137`, `EntityExtractor.fs:446` — plus `Kernel/SemanticMemory.fs:112` that will want the same helper and cannot reference Evolution).
2. **Call-site swap.** At the six sites, `ResponseFormat = Some (Constrained (JsonSchema EvolutionSchemas.xxx))` while **keeping `JsonMode = true`** as the degradation belt (`OllamaClient.fs:184-185` maps Constrained(JsonSchema) to Ollama's schema-aware `format` field, compiled to GBNF server-side since v0.5 — honored even on default deployments). Do NOT constrain `BenchmarkRunner.fs:92` or other free-text/code-gen sites. Note (review): `SymbolicReflector.fs` sets `JsonMode=true` at `:83` with no `ResponseFormat = Some Json` line — the grep criterion is trivially true there while the site stays unconstrained; the capture-stub test must positively assert `Constrained` per site.
3. **Constraint-aware routing** (`v2/src/Tars.Llm/Routing.fs`). Per review, collapse to what is actually emitted: `type ConstraintNeed = NeedsCfg | NeedsJsonSchema | NoNeed` (`NeedsRegex` is dead on arrival; Regex falls into the CFG bucket if it ever appears). `ConstraintNeed.ofRequest : LlmRequest -> ConstraintNeed`; `supports : LlmBackend -> ConstraintNeed -> bool` (Vllm: all; LlamaCpp: CFG+JsonSchema via GBNF; Ollama/OpenAI: JsonSchema only; Anthropic/Gemini: none — they degrade to prompt hints per `AnthropicClient.fs:73-79`, `GoogleGeminiClient.fs:99-106`). With slice C deferred the NeedsCfg preference branch waits; ship "JsonSchema-capable or downgrade".
4. **Loud downgrade.** `chooseBackend` is pure — extend the routing **result**, not the function: `Downgrade: ConstraintDowngrade option` where `ConstraintDowngrade = { RequestedGrammar: string; Backend: string }` (Reason is derivable — review). In `DefaultLlmService`, `LogWarning("CONSTRAINT DOWNGRADE: {grammar} grammar discarded — backend {backend} cannot enforce it; falling back to JSON mode...")` on every downgraded request. **Per review, drop the `TARS_STRICT_CONSTRAINTS` strict mode** (no caller asks for fail-hard; the product requirement is the loop keeps running on Ollama) and drop rate-limiting — log every time. **Breaking-change scope (review):** `Routing.RoutedBackend` is the return type of `ILlmService.RouteAsync` (`LlmServiceTypes.fs:26/36/47`), constructed in `ChatClientAdapter.fs:334`, `ClaudeCodeService.fs:198`, `LlmService.fs`, and ≥8 test stubs — F# records have no default fields, so either budget the fan-out or return a new wrapper type from `chooseBackend` and leave `RouteAsync`'s contract alone (preferred). A second stale `RoutedBackend = { Backend; Endpoint }` exists at `Domain.fs:149` — delete or note it, or the wrong type gets extended.
5. **vLLM wire format** (`OpenAiCompatibleClient.fs:122-157` AND the streaming duplicate `:288-320`). Delete the nested `extra_body` DTO (silent no-op even against vLLM). Emit top-level `structured_outputs = {| grammar = ebnf |} / {| regex = ... |} / {| json = schemaElement |}`; drop client-side backend selection entirely. Thread `vllmExtensions: bool` from `Backends.resolve` (`Backends.fs:73` — Vllm, OpenAI, and DockerModelRunner share this adapter; OpenAI proper rejects unknown top-level params).
6. **llama.cpp grammars** (`LlamaCppClient.fs:181-184, :280-283` — currently Constrained silently drops to the wildcard): map `Constrained(Ebnf g)` → llama-server top-level `grammar` (GBNF), `Constrained(JsonSchema s)` → `json_schema`. This gives a grammar-capable **local** backend without a vLLM GPU deployment.
7. **MAF path lockstep.** `ChatClientAdapter.fs:64-80` encodes Ebnf/Regex as `guided_decoding_*` AdditionalProperties — update to the `structured_outputs` naming in the same PR or the two pipelines emit different wire formats for the same `Grammar` value.
8. **Tests.** Routing: Constrained(JsonSchema) on Ollama → Downgrade=None; Constrained(Ebnf) on Ollama → Downgrade=Some with RequestedGrammar="ebnf" + LogWarning asserted via test logger; wire-format serialization asserts top-level `structured_outputs`, no `extra_body`/`guided_decoding` for vLLM, and neither for OpenAI; capture-stub `ILlmService` test pattern-matching each of the six call sites' captured `ResponseFormat`.

**Slice C (follow-up PR):** rewrite `grammars/wot.ebnf` to the language `WotParser` actually accepts (`meta { } inputs { } policy { } workflow "id" { node "id" kind="reason" ... }` — `WotParser.fs:292-446` vs `wot.ebnf:4-29`), in xgrammar/GBNF dialect: `root ::=` rule required, `#` comments not `(* *)`, explicit `ws`/`nl` productions incl. CRLF (GBNF has no implicit token separation — without them a constrained model literally cannot emit a space). Conformance test **Direction A only** (per review — the `EbnfMatcher` reverse direction is a second grammar engine; check corpus acceptance once manually at rewrite time): seeded sampler (`Random(42)`, 200 samples, bounded depth) → `WotParser.parseLines` returns Ok for every sample; load the grammar from disk via `loadEbnfGrammar` so the test proves the file is loadable by name (today nothing loads wot.ebnf at all). Note the third-EBNF-parser caveat: `Tars.Cortex/Grammar.fs` already has `module internal Ebnf`; the test-local tool proves parser-vs-grammar-as-the-test-reads-it, not as xgrammar reads it — consider a Docker-guarded integration test against a real llama-server to close that gap. Wire one end-to-end consumer at the **Cortex planner call site** (`PlannerPrompts.fs`, not `PlanCmd.fs` — constraint policy in Interface.Cli would fragment across entry points and be invisible to agent-driven planning) via `ConstrainedDecoding.withNamedGrammar grammarsDir "wot"`.

### Signature sketches (not yet applied)

```fsharp
// Routing.fs — pure; downgrade travels in the result
type ConstraintNeed = NeedsCfg | NeedsJsonSchema | NoNeed
module ConstraintNeed =
    val ofRequest : LlmRequest -> ConstraintNeed
    val supports  : LlmBackend -> ConstraintNeed -> bool
type ConstraintDowngrade = { RequestedGrammar: string; Backend: string }
// preferred shape: new wrapper from chooseBackend, RouteAsync contract untouched
type ChosenBackend = { Routed: RoutedBackend; Downgrade: ConstraintDowngrade option }
val chooseBackend : RoutingConfig -> LlmRequest -> ChosenBackend

// OpenAiCompatibleClient.fs — vLLM >= 0.12 unified field, gated
type StructuredOutputsDto = { json: JsonElement option; regex: string option; grammar: string option }
// sendChatAsync gains: vllmExtensions:bool  (true only when Backends.resolve matched Vllm)

// Tars.Evolution/EvolutionSchemas.fs — schema strings ONLY; request shaping stays in
// ConstrainedDecoding.withJsonSchema (Tars.Llm)
module EvolutionSchemas =
    val contradictionSchema : string
    val evaluationSchema : string
    val reflectionSchema : string
    val optimizerSchema : string
    val symbolicReflectionSchema : string
```

### Edge cases

- wot.ebnf currently specifies a DIFFERENT LANGUAGE than WotParser accepts — any conformance test against today's grammar fails on 100% of fixtures; rewrite precedes test, and the test loads the grammar from disk or drift returns.
- GBNF gotchas: `root ::=` required; `(* *)` illegal; no implicit whitespace; must allow `\r\n` (the self-hosting-gate CRLF lesson).
- `extra_body` is a Python-SDK client-side kwarg merging into the top-level body — a literal nested JSON field is ignored by vLLM; the inner `guided_decoding` shape was never the server API.
- OpenAI cloud rejects unknown top-level params — gate `structured_outputs` on the resolved backend.
- OpenAI `json_schema` strict:true (`OpenAiCompatibleClient.fs:141-149`) requires `additionalProperties:false` + all properties required; existing schemas would 400 — author strict-compatible or set strict only where tolerated.
- Ollama honors Constrained(JsonSchema) but has NO raw GBNF/regex API — downgrade unavoidable there; loud logging, never throw-by-default.
- Streaming duplicates: fix both `OpenAiCompatibleClient` constructions and `OllamaClient.fs:351-357`, or paths silently diverge.
- The new `VllmConfigured` explicit flag must replace, not coexist with, the localhost:8000 heuristic inside `localRoute` (`Routing.fs:190-196`) — otherwise constraint-aware and hint routing disagree about whether vLLM "is configured".
- Keeping `JsonMode=true` alongside Constrained is deliberate belt-and-suspenders; the `JsonParsing.tryParseElement` fallbacks at each call site remain the last line of defense.

### Pitfalls

- Wiring Constrained requests without the router (or vice versa): Findings 1+2 are independent breaks — landing only one still silently does nothing.
- Trusting the existing extra_body plumbing because "the vLLM path is already built" — it routes correctly and enforces nothing; hence the serialized-JSON assertion in tests.
- Converting wot.ebnf to GBNF but forgetting whitespace/CRLF productions: grammar compiles, model deadlocks or emits single-line soup the line-based WotParser rejects — grammar-valid but parser-invalid.
- Logging the downgrade at Debug/Info or once at startup — silent degradation is the audit's core complaint.
- Scope creep into weighted-GBNF from Beta posteriors (Opportunity 5) — later agenda territory; land the unweighted activation first.

### Review notes

Reviews: (1) split slices — A+B now, C follow-up; NeedsCfg branch waits with it; (2) Direction B of the conformance test (EbnfGrammar/EbnfMatcher mini-CFG-toolkit) cut — halves slice C; (3) strict mode + rate-limiting cut (YAGNI); (4) `NeedsRegex` collapsed; `Reason` field dropped; (5) `constrainJson` duplicate of `withJsonSchema` cut — helper belongs in Tars.Llm; (6) `RoutedBackend` is public-interface surface with ≥12 construction sites (incl. 8 test stubs) — prefer a wrapper type; stale duplicate at `Domain.fs:149` flagged; (7) step-9 placement pinned to the Cortex planner seam; (8) layering verdict clean — Evolution already refs Llm (`Tars.Evolution.fsproj:75`), no new edges, `ChatClientAdapter` updated in lockstep.

### Effort

Slice A ~0.5 day (schemas + swap + capture tests); slice B ~1 day (router + downgrade logging + vLLM/llama.cpp wire fixes + serialization tests); slice C (deferred) ~1-1.5 days.

### Success criterion

```
/goal From v2/: dotnet build && dotnet test green (>= 821 existing + new), with:
(1) a capture-stub ILlmService test positively asserts all six Tars.Evolution call sites
send Constrained(JsonSchema ...) (not just grep absence of ResponseFormat.Json);
(2) routing tests: Constrained(Ebnf) on Ollama-only config yields Downgrade =
Some { RequestedGrammar = "ebnf" } and DefaultLlmService emits a LogWarning containing
"CONSTRAINT DOWNGRADE" (test logger); Constrained(JsonSchema) on Ollama yields
Downgrade = None; (3) serialization tests: vLLM request JSON contains top-level
"structured_outputs" and no "extra_body"/"guided_decoding"; the OpenAI-targeted request
contains neither. [Slice C, follow-up: 200/200 seeded samples from grammars/wot.ebnf
(loaded via loadEbnfGrammar) accepted by WotParser.parseLines.]
```

### References

Research: `docs/research/theory-neuro-symbolic.md` Findings 1, 2, 8; Opportunities 1-3. Code: `ConstrainedDecoding.fs:25-32, 54-90, 97-150`; `Routing.fs:74, 86-125, 133-228`; `OpenAiCompatibleClient.fs:122-157, 288-320`; `OllamaClient.fs:181-188, 351-357`; `LlamaCppClient.fs:181-184, 280-283`; `Backends.fs:70-79`; `Domain.fs:21-31, 149`; `ChatClientAdapter.fs:53-83, 334`; the six evolution call sites; `WotParser.fs:218-446`; `grammars/wot.ebnf:1-29`. Papers/docs: Willard & Louf (arXiv:2307.09702); Dong et al. XGrammar (arXiv:2411.15100); Geng et al. EMNLP 2023 (arXiv:2305.13971); Park et al. Grammar-Aligned Decoding (NeurIPS 2024); Kambhampati et al. LLM-Modulo (arXiv:2402.01817); vLLM structured-outputs docs (guided_* removed v0.12); Ollama PR #7900; llama.cpp GBNF README.

---

## 6. Item 5 — Compression-Gated + Reversible Promotion (MDL Gain over the Rollback Corpus)

### Intent

Replace the promotion staircase's recurrence-only gate with a DreamCoder/Stitch-style compression test: a candidate abstraction earns promotion only if ΔDL > 0 (tokens saved minus definition cost), replacing today's `RemovesComplexity` boolean — which after the round-3 fix (`PromotionPipeline.fs:202` requires template ≠ name) is **trivially FALSE on every live run** because `run:332` still passes `candidate.Record.PatternName` as the template. Thread the real WoT template (already on `TraceArtifact` via `RetroactionLoop.fs:410-417` but discarded by `run`) into validation; parse-check rollbacks so `AutoValidatable` means "parses" not "IsSome"; revive `GrammarDistillation`'s dead `CompressionRatio` (`GrammarDistillation.fs:64, 206-217`); sync `WeightedRule.Level` on Approve (fixing the live recurrence/weights desync). The 22 `RollbackExpansion` strings in `~/.tars/promotion/lineage.json` are the ready-made corpus.

**Per review, the demotion path (PromotionLevel.prev, Demote DU case, sweepDemotions, hysteresis) — roughly a third of the diff, explicitly blocked on item 3's regularized posteriors, and unable to fire meaningfully on live data today — splits into a follow-up item.** V1 ships the MDL gate + plumbing fix + level sync.

### Best practices

Stitch's utility function is the template: utility = per-match compression × match count − abstraction definition size, in integer tokens — don't invent a novel score. Two-part MDL: always charge the library for its own definition or the gate admits every abstraction (Grünwald 2007). Validate by the LILO pair — corpus compression AND downstream solve-rate at fixed budget — never compression alone. Keep scoring deterministic and LLM-free (token counts, parse checks are pure; reproducible in CI, immune to model drift). Version the metric (`DlVersion`) and persist per-cycle DL history — monotonicity claims need a fixed measuring stick. F# discipline: new module immutable + `Result<>`-typed, placed correctly in compile order; exhaustive matches enforced by `TreatWarningsAsErrors`.

### Implementation steps

1. **Prerequisite plumbing** (`PromotionPipeline.fs:291-333`). In `run`, build `templateByPattern : Map<string,string>` from inspected artifacts exactly like `rollbackByPattern` (`:303-308`); change `:332` from `propose candidate.Record.PatternName rollback` to the map lookup with `""` default — never the name, so `RemovesComplexity` fails honestly when no template exists. Nothing upstream changes (`RetroactionLoop.fs:410-417` already sets `TraceArtifact.PatternTemplate`). **This and the MDL gate land together** — criteria-only makes the criterion always false (current live state); plumbing-only reverts it to trivially true.
2. **Honest subjective defaults.** `MoreReadable`/`ComposesCleanly` default `false` in `validateDeterministic` (`:203, 207`); LLM assessment in `validate` (`:218-228`) still overrides. This plus the MDL gate drives governor approval below 80% on mixed batches (live approval is 95.7%).
3. **MDL scoring, v1-simplified per review.** New `v2/src/Tars.Evolution/MdlScoring.fs` (compile order: after `PromotionTypes.fs`, before `GrammarGovernor.fs` — load-bearing). With lineage holding ~1 expansion per pattern, the corpus-matching machinery degenerates to `OccurrenceCount × (DL(own expansion) − DL(invocation)) − DL(template)` — so **cut `corpusFromLineage`/`stepSignature`/`corpusDescriptionLength` for v1** and compute ΔDL from the candidate's own rollback + `OccurrenceCount` (documented: lineage stores one expansion per promotion event; OccurrenceCount ≥ 3 attests ≥ 3 real uses). Add corpus matching only when the store persists one expansion per contributing task. Contents: `tokenCount` (deterministic tokenizer reusing the delimiter set from `RoundtripValidation.fs:44`, versioned by `DlVersion`; CRLF→LF normalize first) and `scoreCandidate` producing a minimal `DlReport`.
4. **Rollback parser in its own file** (review): `RollbackFormat.fs` (parser + format constants), NOT inside `MdlScoring` (parsing is not scoring), placed so `TraceCompiler.fs` (order 38, the emitter at `:107-123`) can reference it — add a round-trip test (`TraceCompiler emit >> parse = Ok`) so emitter/parser drift is caught at build time. `RollbackParser.parse : string -> Result<RollbackStep list, string>` — strict parse of the `step: <id> (<kind>)` header + indented `tool:`/`goal:`/`output:` attrs format all 22 live expansions use. `PromoteCommand.fs:162`'s synthetic rollback is a required fixture.
5. **Rewire the gate.** `validateDeterministic` takes the score: `RemovesComplexity = (scoreCandidate candidate).DeltaDl > 0`; `AutoValidatable = rollback |> Option.map (RollbackParser.parse >> Result.isOk) |> Option.defaultValue false`. Strengthen the DslClause+ hard gate — **per review, one uniform check** (JsonDocument.Parse + required `nodes` array with id/kind per `TraceCompiler.fs:60-66`), no tiered WotCompiler stretch check until a malformed-but-JSON-valid template actually slips through. **Per review, the pipeline computes parse results once and passes them into the governor** (as criteria or a small evidence record) — `GrammarGovernor` stays a pure decision function, DSL-free, parse single-sourced.
6. **Revive CompressionRatio** (`GrammarDistillation.fs`): replace the len/4 estimate (`:206-209`) with `MdlScoring.tokenCount`; gate `SuggestedLevel` (`:217-221, 245-248`) on `CompressionRatio > 1.0`; add a compression term to `toWeightedRule.RawScore` (`:284-296`).
7. **Level sync** (fixes live desync, Finding 7). **Per review, make recurrence `CurrentLevel` authoritative**: in PromotionPipeline's single save path (before `WeightedGrammar.save` at `:375`), unconditionally rewrite every weights entry's Level from the recurrence store — one mechanism, guarded by one invariant test, instead of three sync points. Heal the 5 stale ga.* rules once with a scratch script, not permanent load-time code.
8. **Instrumentation, minimal per review**: append `{cycle; corpusDl; dlVersion; timestamp}` to `~/.tars/promotion/mdl_history.json` from the evolve cycle (this alone serves criterion (c)); the `tars promote mdl` CLI arm prints that file or is deferred.
9. **Tests**: tokenizer determinism + CRLF invariance; synthetic ΔDL (template matching 3×60-token expansions with 40-token definition gains; 1 expansion fails — falsifies boolean replacement); RollbackParser accepts all 22 live expansions (fixture) + rejects garbage; post-run invariant: recurrence.CurrentLevel = weights.Level for every shared PatternId; emit/parse round-trip; governor approval < 80% on a mixed synthetic batch.

**Follow-up item (deferred): demotion path.** `PromotionLevel.prev`; `Demote of reason` on `GovernanceDecision` (JSON-additive; confine construction to the sweep and test that `GrammarGovernor.evaluate` can never produce it — governor semantics stay a pure promotion verdict); `sweepDemotions` returning records only, with **RetroactionLoop** doing the `PromotionIndex.refresh` (compile-order: `PromotionIndex.fs` is order 23, AFTER `PromotionPipeline.fs` at 22 — the pipeline cannot call it; `RetroactionLoop.fs:507` already refreshes). Demote when regularized posterior mean < 0.4 with ≥ 5 post-promotion selections AND re-scored ΔDL ≤ 0; cooldown 2 cycles (drop the double-speculative `RepromotionMarginTokens`); decide seeded-ga.* exemption policy explicitly before enabling. Blocked on item 3.

### Signature sketches (not yet applied)

```fsharp
// RollbackFormat.fs — parser + format constants, referenced by both MdlScoring and TraceCompiler
type RollbackStep = { StepId: string; Kind: string; Tool: string option; Attrs: (string * string) list }
module RollbackParser =
    val parse : text: string -> Result<RollbackStep list, string>   // CRLF-normalized, strict

// MdlScoring.fs — after PromotionTypes.fs, before GrammarGovernor.fs (compile order load-bearing)
module MdlScoring =
    [<Literal>] val DlVersion : int = 1
    type DlReport = { PatternId: string; ExpansionTokens: int; InvocationTokens: int
                      DefinitionTokens: int; EffectiveMatches: int; DeltaDl: int; DlVersion: int }
    val tokenCount    : text: string -> int
    val scoreCandidate: candidate: PromotionCandidate -> DlReport
    // v1: DeltaDl = EffectiveMatches * (ExpansionTokens - InvocationTokens) - DefinitionTokens
    //     EffectiveMatches = max 1 candidate.Record.OccurrenceCount

// PromotionPipeline.fs — validation evidence computed once, governor stays pure
val validateDeterministic :
    evidence: {| RollbackParses: bool; TemplateIsWotJson: bool; Dl: MdlScoring.DlReport |} ->
    existing: RecurrenceRecord list -> candidate: PromotionCandidate -> PromotionCriteria
```

### Edge cases

- `run:332` currently makes RemovesComplexity ALWAYS FALSE (caps every candidate at 7/8) — the two halves of the fix are one change or approvals silently shift twice.
- Tiny corpus / self-match degeneracy handled by the OccurrenceCount formula (documented assumption).
- One live lineage record has null RollbackExpansion — skip, never throw.
- CRLF vs LF (self-hosting-gate memory): tokenCount and any hashes normalize line endings or DL drifts between machines and the monotonicity check false-fails.
- Tokenizer changes invalidate history — bump `DlVersion`, reset the comparison window.
- Saturated posteriors make demotion blind (live rules at SuccessRate exactly 1.0, SelectionCount 4) — demotion stays deferred until item 3's smoothing lands.
- JSON templates must NOT route through `WotParser.parseLines` (it parses .wot.trsx surface syntax; TraceCompiler templates are JSON node graphs) — guaranteed false rejection.
- Schema-breaking edits to lineage.json are forbidden — 23 live records are the corpus and audit trail; additive only.

### Pitfalls

- Do not compute DL over pattern names — measuring name length is exactly Finding 2's failure mode.
- Do not treat `RoundtripValidation.quickValidate`'s single-pair length heuristic as compression — per-instance sanity check, not a corpus objective.
- Do not add ΔDL as a ninth boolean while keeping the ≥6 threshold — it replaces `RemovesComplexity` (same slot, same 8-count); the continuous value flows to RawScore/SuggestedLevel separately.
- Do not let the (deferred) sweep run against unregularized posteriors — one failure yields instant demotion, all-success never demotes.
- Do not forget PromotionIndex refresh on any future demotion — PatternSelector keeps boosting from the stale index otherwise.

### Review notes

Reviews cut: corpus-matching machinery (self-defeating given the 1-expansion corpus), the entire demotion path (blocked + can't fire), `RepromotionMarginTokens`, load-time one-shot reconciliation (scratch script instead), the CLI arm + `corpusDescriptionLength` (history append suffices), and tiered template validation. Structural fixes: `sweepDemotions`→`PromotionIndex.refresh` is compile-order-impossible from the pipeline (RetroactionLoop is the caller); `RollbackParser` extracted to `RollbackFormat.fs` with an emit/parse round-trip test; validation evidence computed once in the pipeline, governor kept pure and DSL-free; recurrence store made the single level authority. Layering verified: no Cortex↔Evolution cycle (PatternSelector consumes promotions only via the index.json file seam — preserve this; no in-process notification from Evolution to Cortex).

### Effort

V1 (gate + plumbing + level sync + instrumentation + tests): ~4-6 days. Demotion follow-up: ~3 days, after item 3.

### Success criterion

```
/goal All of: (a) at least one organically-discovered pattern (Source <> seeded,
DerivedFrom a real task id) reaches Builder or above with a RollbackExpansion that
RollbackParser.parse accepts and, at DslClause+, a template that parses as a WoT node
graph; (b) governor approval on a mixed synthetic candidate batch (compressive and
non-compressive, in tests/Tars.Tests) is below 80%; (c) ~/.tars/promotion/mdl_history.json
shows DL at fixed DlVersion strictly decreasing over 5 consecutive tars evolve cycles
while ProblemBank pass rate at fixed LLM-call budget does not decrease; (d) post-run
invariant test green: every PatternId present in both recurrence.json and weights.json
has identical promotion level.
```

### References

Code: `PromotionPipeline.fs:199-208, 291-377`; `GrammarDistillation.fs:64, 206-221, 284-296`; `GrammarGovernor.fs:27-52`; `PromotionTypes.fs:21-26, 50-59, 77-80, 93-105`; `WeightedGrammar.fs:100-106, 136-147, 168-174`; `TraceCompiler.fs:60-66, 107-123, 129`; `RetroactionLoop.fs:405-418, 507`; `RoundtripValidation.fs:44, 132-177`; `PromoteCommand.fs:162, 274-283`; `WotCompiler.fs:298`; `~/.tars/promotion/lineage.json`. Research: `docs/research/frontier-program-synthesis.md` Findings 1-4, 7, 9; Opportunities 1-4, 9. Papers: Ellis et al. DreamCoder (PLDI 2021, arXiv:2006.08381); Bowers et al. Top-Down Synthesis/Stitch (POPL 2023, arXiv:2211.16605; github.com/mlb2251/stitch); Grand et al. LILO (ICLR 2024, arXiv:2310.19791); Grünwald, The MDL Principle (MIT Press 2007).

---

## 7. Item 6 — Open-Ended Self-Generating Curriculum (Generator–Filter–Archive)

### Intent

The benchmark curriculum is a closed set of exactly 24 problems (19 `ProblemBank` + 5 `GaProblemBank`) under a monotone mastery ratchet; when exhausted, `RetroactionLoop.runCycle` returns `Error "No more problems available in curriculum"` (`RetroactionLoop.fs:456`) — the ANNECS-equivalent progress measure is bounded above by 24. The existing LLM task generator emits free-text validation with `ReferenceSolution=None` and no solvability check. Fix: ACCEL-style LLM mutation of high-regret seeds via constrained JSON (emitting `ExpectedSignature`, executable `ValidationCode`, `ReferenceSolution`); a POET/GoalGAN admission filter (reference solution must compile and PASS; a wrong-probe must FAIL; agent pass rate over k=5 in the GOID band [0.1, 0.9]); a persisted generated bank; and a per-cycle health metric. Nearly all substrate exists — `runSuiteFromProblems` takes arbitrary problem lists, `ConstrainedDecoding` builds schema-constrained requests, `PatternOutcomeStore` records per-problem outcomes.

**Per review, v1 scopes to the benchmark path only (steps 1-7 below): the RetroactionLoop fallback (original step 8) is a lossy bridge** — converting an admitted `BenchmarkProblem` to curriculum `Problem` (free-text `ValidationCriteria`) discards the executable-validation contract the whole triad exists to enforce, and `RetroactionLoop.fs` (fsproj order 41) cannot call modules compiled after `BenchmarkRunner.fs` (order 61) anyway. The ~10-line CurriculumManager livelock fix (original step 9) is an unrelated seam change that ships as its own small commit with its own test.

### Best practices

Mutate, don't generate from scratch (ACCEL): small edits of high-regret seeds — recent `Compiled=true && Validated=false` attempts — stay near the capability frontier. Admission is a minimal criterion, not a quality score (POET): provably solvable AND neither trivial nor hopeless (GoalGAN band). Keep validation deterministic — the repo's "no LLM-as-judge" rule (`BenchmarkTypes.fs:19`) extends to generated problems: the LLM writes the validator once, the harness runs it forever. Probe the validator, not just the solution (AutoCode: ~14% of LLM reference solutions are wrong and vacuous validators pass everything; the wrong-probe filters ~27% of error-prone problems). Condition the generator on performance history (Voyager: list recently-failed problem ids). Retire, never delete — retired problems remain regression guards and verified SFT rows. All LLM access via `LlmFactory.create(logger)`; all state under `~/.tars/curriculum/`; `Result<>` everywhere so the evolve loop degrades gracefully (cold start / no LLM endpoint = curated banks only, never a crash).

### Implementation steps

1. **Extract the validator.** `BenchmarkRunner.validateCandidate (code: string) (problem: BenchmarkProblem) : Task<bool * string>` factored from `runProblem`'s compile + run-ValidationCode-under-dotnet-fsi path (`BenchmarkRunner.fs:314-328`) — test a ReferenceSolution without the LLM. Hard process timeout (~60s) on fsi: `TimeLimitSeconds` bounds LLM generation, not fsi execution.
2. **Types + bank, consolidated per review (one module, not four).** `GeneratedProblemBank.fs` in Tars.Evolution — **placed after `BenchmarkRunner.fs:61` and before `Engine.fs`** (compile-order requirement the original plan omitted). Contains the types (`GeneratedProblem` wrapping `BenchmarkProblem` + `ReferenceSolution`, `ParentId`, provenance, `Solved`/`Retired` flags — **no `PassHistory`**, see step 6), `AdmitDecision`, and a **flat NDJSON bank** at `~/.tars/curriculum/generated.ndjson` (full `BenchmarkProblem` shape incl. `ValidationCode` — the `ProblemIngestor` DTO lacks it). Per review: no MAP-Elites grid in v1 (96 slots, ~30 cycles to half-fill, nothing consumes the cell structure) — flat bank + Jaccard dedup; add cell-keyed eviction if a duplicate flood is observed. `BenchmarkProblem` itself needs no change — ReferenceSolution lives on the wrapper, never in the prompt.
3. **Generator.** `ProblemGenerator.fs` (same placement): `sampleSeed` prefers the ACCEL regret proxy (recent Compiled-but-not-Validated attempts), else uniform among in-band problems; `mutateProblem` builds a constrained request via `ConstrainedDecoding.withJsonSchema` against a new `grammars/generated_problem.schema.json` (id, title, description, category, difficulty, expected_signature, validation_code, reference_solution, hints). **Per review: one mutation prompt** ("produce a variant of this seed with one concrete edit, targeting these recently-failed ids"), not a five-case `MutationOp` DU with five templates — store the edit description as a string field. Ids namespaced `gen-{parentId}-{shortHash}`; reject id collisions.
4. **Filter.** `SolvabilityFilter.fs`: Gate A (deterministic, no LLM): `validateCandidate referenceSolution` must PASS AND a degenerate wrong-probe (stub of ExpectedSignature returning a default value) must FAIL → else `RejectedVacuousValidator`. Gate B (GOID): run the agent k=5 via `runProblem` with a quiet logger; pass rate in [0.1, 0.9] (k=5 granularity 0.2 — operationally "not 0/5 and not 5/5"; k configurable). Gate C: token-set Jaccard of normalized Description vs incumbents < 0.7. Short-circuit: Gate A failure skips B (cost control). Archive placement keys on **measured** difficulty, never the LLM's self-declared label.
5. **Contamination guard (binding, from review).** Gate B's k=5 calibration attempts **bypass `recordOutcomes`/`saveResults`** — probe noise must not enter the pattern-selection evidence stream, and the regret proxy must not count a candidate's own admission probes as regret signal for itself. Admitted generated problems record under a distinct PatternKind tag (e.g. `benchmark-gen:{Category}`) — the GOID filter engineers outcomes toward the 10-90% band, and injecting band-shaped outcomes untagged into the store that drives promotion ranking is a Goodhart channel with no off-switch.
6. **Single source of truth for pass rates (review).** Rolling pass rate and retirement decisions computed **solely from `PatternOutcomeStore`** (filtered on the `benchmark-gen:` prefix — the Goal field also carries non-benchmark outcomes); the bank holds only provenance + Solved/Retired flags. No `recordAttempt`, no duplicate PassHistory in NDJSON.
7. **Evolve wiring** (`Evolve.fs:763-767, 791-799`), no flag-surface change: **one** new benchSource arm (`"generated"` — not two aliases) returning curated banks @ `GeneratedProblemBank.all()`; post-benchmark, print health beside PassRate; **one** refill trigger (`ActiveCount < floor`) and **one** retirement rule (rolling rate > band max over ≥ k attempts); admit up to **3 candidates per cycle** (each Gate B costs 5 LLM attempts + 5 fsi runs). Health record minimal: derive InBandFraction/ANNECS later from data already on disk if needed.
8. **Tests** (stub `ILlmService`, canned schema-valid JSON): admission matrix (unsolvable ref rejected; vacuous validator rejected via wrong-probe; 0/5 and 5/5 rejected; in-band admitted); NDJSON round-trip; guard test that `ProblemBank.all()` still returns 19 and `GaProblemBank.all()` still 5.

**Separate small commit:** the CurriculumManager livelock fix (`getNextProblem`, `CurriculumManager.fs:26-46`): weight available problems by exponential backoff on `FailedProblems` count (written at `:62-68`, never read) with a seeded Random **threaded through `runCycle`** so existing RetroactionLoop tests stay deterministic; its own backoff-weighting unit test. **Separate follow-up PR:** the RetroactionLoop generation fallback, only if/when the fallback can deliver problems whose validation stays executable end-to-end — inject `generateFallback: (unit -> Async<Result<Problem,string>>) option` into `runCycle`'s signature wired at the Evolve/Engine layer (also solves the compile-order problem; RetroactionLoop stays harness-agnostic).

### Signature sketches (not yet applied)

```fsharp
type GeneratedProblem =
    { Problem: BenchmarkProblem; ReferenceSolution: string
      ParentId: string; EditDescription: string; CreatedAt: DateTime
      Solved: bool; Retired: bool }

type AdmitDecision =
    | Admitted
    | RejectedUnsolvable of reason: string
    | RejectedVacuousValidator
    | RejectedOutOfBand of measuredRate: float
    | RejectedDuplicate of incumbentId: string

module GeneratedProblemBank =
    val all    : unit -> BenchmarkProblem list          // active (non-retired) only
    val admit  : GeneratedProblem -> AdmitDecision -> unit
    val retire : problemId: string -> unit

module SolvabilityFilter =
    val checkMinimalCriterion : GeneratedProblem -> Task<Result<unit, string>>  // ref PASS + wrong-probe FAIL
    val measurePassRate : llm: ILlmService -> problem: BenchmarkProblem -> k: int -> Task<float>
    // NOTE: probe attempts MUST NOT flow into recordOutcomes/PatternOutcomeStore
    val admit : llm: ILlmService -> band: float * float -> incumbents: BenchmarkProblem list
                -> candidate: GeneratedProblem -> Task<AdmitDecision>

// BenchmarkRunner addition (extracted from runProblem)
val validateCandidate : code: string -> problem: BenchmarkProblem -> Task<bool * string>
```

### Edge cases

- Vacuous validator (always-PASS) — the wrong-probe gate exists for exactly this.
- ValidationCode output contract: require the literal line `PASS` via the same parsing `runProblem` uses; anything else is a Gate A rejection, not a crash.
- Infinite loop/hang in generated code — hard fsi process timeout, timeout = `RejectedUnsolvable`.
- Band drift: problems drifting above 0.9 retire (kept on disk, excluded from `all()`).
- JSON-embedded F# escaping: multi-line ValidationCode arrives JSON-escaped — round-trip through System.Text.Json before compiling; Gate A's compile catches mangling (and the CRLF concern).
- Cost blowout: 3-candidate cap + Gate-A short-circuit bounds a bad LLM day at 3 generations.
- `Unascertained` difficulty excluded; a Gate-B measured rate always maps to a real tier.
- Concurrent evolve runs appending: write-temp-then-rename.
- ReferenceSolution leakage: the wrapper keeps it off `BenchmarkProblem`, so `runProblem`'s prompt path can never show the agent the answer; SelfTrain may use it only for verified rows.

### Pitfalls

- The existing `mergeCurriculum` anti-pattern (`CurriculumPlanner.fs:158`) prepends unverified free-text tasks first in line — never route generator output past the filter.
- LLM-as-judge for pass/fail is banned by the repo's benchmark design; judge role only for optional interestingness, never solvability.
- Reading saturation from sparse telemetry (the research doc's refuted claim: 7 runs at n=1-2 misread as saturation) — gate refill on rolling windows.
- Extending `Problem`/`ProblemDto` instead of `BenchmarkProblem` — the ingestor DTO lacks ValidationCode and cannot feed the deterministic harness.
- Requiring FsCheck Properties on generated problems in v1 — LLMs emit property harnesses unreliably; `Properties = None` stays optional.

### Review notes

Reviews scoped v1 to steps 1-7 (benchmark path), splitting the RetroactionLoop fallback (lossy bridge voiding the filter's guarantee + compile-order violation, fixed by injection) and the livelock fix (unrelated seam change, own commit + test). Cut for v1: the MAP-Elites grid, the 5-case MutationOp DU, `recordAttempt`/PassHistory (dual source of truth), AnnecsCount/InBandFraction as gating metrics, the `"auto"` alias, and two of four modules (types fold into the bank, health into the filter/bank). Added as binding: Gate-B probe bypass of `recordOutcomes` and the `benchmark-gen:` tag (Goodhart channel), PatternOutcomeStore as sole rate authority (filtered), and module placement after `BenchmarkRunner.fs:61`. Layering otherwise sound: Evolution→Cortex direction already exists; `~/.tars/curriculum/` mirrors the promotion-index convention.

### Effort

M — single PR for steps 1-8, ~3-4 focused days including live smoke run; livelock fix and RetroactionLoop fallback separate.

### Success criterion

```
/goal With an LLM endpoint configured, `tars evolve --loop 3 --benchmark` with benchmark
domain `generated`: (a) admits >= 1 generated problem to ~/.tars/curriculum/generated.ndjson
whose ReferenceSolution compiles and prints PASS under its own ValidationCode via dotnet
fsi, whose wrong-probe prints FAIL, and whose measured agent pass rate over 5 attempts
lies in [0.1, 0.9]; (b) prints curriculum health (ActiveCount/RetiredCount) in each
cycle's output beside PassRate; (c) no Gate-B probe outcome appears in
~/.tars/pattern_outcomes.jsonl (probe bypass verified by test); (d) full suite green —
all ~820 existing tests plus new deterministic stub-LLM tests covering all AdmitDecision
branches, with ProblemBank.all() still 19 and GaProblemBank.all() still 5.
```

### References

Research: `docs/research/frontier-open-ended-curriculum.md` (Findings 1-11, refuted-claim record, Opportunities 1-8). Code: `CurriculumManager.fs:13-23, 26-46, 62-68`; `CurriculumTypes.fs:26-45`; `CurriculumPlanner.fs:56-66, 89-93, 137-158`; `BenchmarkTypes.fs:6-16, 20-43, 46-65, 68-78`; `BenchmarkRunner.fs:334-399`; `GaProblemBank.fs:28-109`; `ProblemBank.fs:183-196`; `ProblemIngestor.fs:61-115`; `RetroactionLoop.fs:446-456`; `Evolve.fs:763-767, 791-799`; `ConstrainedDecoding.fs:58, 76`; `PatternSelector.fs:68-84`. Papers: Parker-Holder et al. ACCEL (ICML 2022, arXiv:2203.01302); Wang et al. POET (arXiv:1901.01753) + Enhanced POET/ANNECS (arXiv:2003.08536); Florensa et al. GoalGAN (arXiv:1705.06366); Mouret & Clune MAP-Elites (arXiv:1504.04909); Zhang et al. OMNI (arXiv:2306.01711); OMNI-EPIC (arXiv:2405.15568); Wang et al. Voyager (arXiv:2305.16291); Clune AI-GAs (arXiv:1905.10985); AutoCode (arXiv:2510.12803); TTCS (arXiv:2601.22628); Lehman & Stanley Novelty Search (2011).

---

## 8. Item 7 — Close the Search Capability Gap Against a Measured Baseline

### Intent

The famous "60% search failure rate" rests on five outcome records (three seeded ReAct failures, March 2026) admitted by GapDetection's `total>=2` floor, and is stale — every current insight snapshot reports `gaps:[]`. The gap is nonetheless structurally real: `search_codebase` is gated on a mutable global index no code path ever initializes (`setCodebaseIndex` has zero call sites in v2) and degrades to a prose warning string; `CodebaseRAG.SearchAsync` is single-shot cosine top-K with silent `[]` and exception-swallowing fallbacks; GapDetection's prescribed remedies name tools (`glob-search`, `grep-search`) that exist nowhere; `CapabilityStore.TrackUsageAsync` is `task { return () }`.

**Per review, sequencing is the plan:** ship the cheap fixes (steps 1-4) first, re-run the probe suite, and **gate the symbol graph, corrective-wrapper machinery, and reputation work on a still-red measurement**. The plan's own citations say grep-in-a-loop reaches ~94.5% of RAG faithfulness with zero vector store — the 2-week symbol graph is not committed work.

### Best practices

Measure before fixing (the 60% baseline is seeded/stale — without re-measurement any "improvement" is unfalsifiable against a phantom number). Machine-readable failure: never prose strings type-indistinguishable from success — the repo's `Result<>` convention applies to tool envelopes. The retrieval ladder is regex/glob → symbols → semantic, not semantic-first. Canonical tool naming: update Core's emitted remedy names to the real registered names and validate against the live registry in Evolution (`CurriculumPlanner`) — aliases re-create the drift that produced unexecutable remedies (Core sees no tool registry and can never self-validate). Record which strategy answered and why escalation happened, or credit assignment can't attribute outcomes.

### Implementation steps

**Phase 1 (committed):**

1. **Harden gap detection, simplified per review.** In `GapDetection.fs:129` change `total >= 2` to a configurable `minSamples` (default 5), plus a **recency cutoff** (ignore outcomes older than N days) — a ~5-line change with the same effect as the originally-proposed exponential age decay, without threading timestamps through `(goal, tags)` tuples across `MetaCognitionOrchestrator` and without the decay's own "silently erases all history" edge case. `detectGaps` must distinguish "no gap" from "insufficient fresh data" — emit nothing rather than a false 0% signal.
2. **Baseline probe suite.** ~20 search-goal episodes (goals matching `extractDomainTags`' keywords, `GapDetection.fs:18-28`) with known file:line answers from v2/src, runnable via the benchmark harness. **Run once BEFORE any fix** (a fresh recomputation already gives 50%, not 60%). **Contamination guard (review):** probe outcomes must carry a distinct origin tag that PromotionPipeline and PatternSelector filter out, or go to a separate file only GapDetection reads — 20 synthetic episodes in `pattern_outcomes` can promote/demote unrelated patterns through the staircase. **This one suite serves all success criteria** — no separate 30-query and 50-item benchmark sets (review: benchmark inflation).
3. **Upgrade `search_code`** (`WorkflowTools.fs:110-132`): the guard `if results.Count >= 30 then ()` at `:129-130` is a no-op. Replace Contains-matching (`:116`) with compiled `Regex(pattern, IgnoreCase ||| Compiled)` in try/with falling back to literal substring on `ArgumentException`; add `maxResults` (default 50) + `contextLines`; make the cap actually break the loop (+ a unit test asserting it bounds output — the no-op shape compiles fine). Register a sibling `glob_search` (honoring `ExcludePatterns` from `CodebaseRAG.fs:71`). Update `GapDetection.fs:100`'s `ComposePatterns` to the real registered names; validate remedy names against the live registry in `CurriculumPlanner` before emitting tasks.
4. **Lazy index + machine-readable status for `search_codebase`.** Replace the prose warning (`SemanticCodeTools.fs:39-41`) with a JSON envelope `{"status": "ok"|"index_unavailable"|"no_results"|"error", "results":[...], "fallback_used":bool}` — per review, **fix `search_codebase`'s envelope first** (only failure paths are the defect; the other three tools follow if needed). Lazy init: on first search with `codebaseIndex=None`, call `CodebaseIndex.IngestQuickAsync` (`CodebaseRAG.fs:280-312`, chunk-only, fast); per review put the guarded `Lazy<Task<CodebaseIndex>>` **in Cortex next to CodebaseRAG** so the CLI full-embedding path and the Tools quick-ingest path share one initialization seam instead of racing on the Tools-side mutable global (`SemanticCodeTools.fs:11`). Critical edge: `IngestQuickAsync` builds chunks but NO embeddings and sets `ingested=true` — a subsequent `SearchAsync` embeds the query against an empty vector collection and returns `[]`; treat empty-vector-store as gate failure and drop to `SearchKeyword`, not `no_results`. Inline escalation (~30 straight-line lines inside `search_codebase`: semantic → score check → `SearchKeyword` → regex): per review, **no `SearchStrategy`/`RelevanceGate`/`runCorrective` combinator module, no separate `corrective_search` tool** (two tools doing the same thing confuses agent routing), **no LLM query-rewrite rung** until telemetry shows rungs a-c miss.

**Phase 2 (gated on a still-red post-fix measurement):**

5. **Symbol graph + `locate_entity`** — only if criterion (a) still fails. `SymbolGraph.fs` in Tars.Cortex with **in-memory mtime-keyed rebuild only** (the GraphPersistence option is a layering violation: `GraphPersistence` lives in Tars.Tools and Cortex cannot reference Tools). The pure symbol-extraction function factored from `CodeAnalysisTools.fs:245-282` must **land in Cortex/Core** with Tools' `extract_symbols` delegating to it. LocAgent hierarchy: exact/prefix symbol match → neighborhood → semantic fallback.
6. **`TrackUsageAsync`** (`CapabilityStore.fs:132-136`) — per review, a plain success/failure counter pair updating the existing `reputation` payload key unblocks `FindAgentsAsync` (frozen 0.5 blend at `:119-122`) with no `IVectorStore` interface change; Beta-Binomial priors add nothing at these sample sizes. If the Beta form is used anyway, note explicitly that it duplicates `WeightedGrammar.fs:102-105` (forced by layering — Cortex cannot reference Evolution) or lift the pure update into Tars.Core so both share one implementation. Adding `CapabilityKind.CodeSearch` to `Domain.fs:80-88` under `TreatWarningsAsErrors` errors every non-exhaustive match solution-wide — budget the audit, the compiler will find them.
7. **Closure.** Re-run the probe suite for ≥10 fresh episodes post-fix; verify RalphBridge's gate ("all gaps below 30%", `RalphBridge.fs:212`) computes green via `failureRateByDomain`.

### Signature sketches (not yet applied)

```fsharp
// GapDetection.fs (Tars.Core) — recency cutoff, not decay
val detectGaps : threshold:float -> minSamples:int -> maxAgeDays:int -> ... -> CapabilityGap list

// Tools envelope — never bare prose
type SearchEnvelope =
    { Status: string          // "ok" | "index_unavailable" | "no_results" | "error"
      Strategy: string        // which rung answered: "semantic" | "keyword" | "regex"
      FallbackUsed: bool
      Results: {| File: string; StartLine: int; EndLine: int; Score: float; Preview: string |} list }

// Cortex — shared lazy index seam (replaces the Tools-side mutable global race)
val codebaseIndexLazy : rootPath:string -> Lazy<Task<CodebaseIndex>>

// Phase 2, if gated in:
// SymbolGraph.fs (Tars.Cortex) — in-memory, mtime-keyed; extraction fn hoisted from Tools
val build        : rootPath:string -> excludes:string list -> SymbolGraph
val locateEntity : graph:SymbolGraph -> query:string -> topK:int -> (Symbol * float) list
// CapabilityStore — plain counters on the existing payload
member TrackUsageAsync : agentId:AgentId * kind:CapabilityKind * success:bool -> Task<Result<unit,string>>
```

### Edge cases

- Invalid user regex: catch ArgumentException, fall back to literal, `fallback_used=true` — never a raw exception string.
- Concurrent first-search on the index: the Cortex-side `Lazy<Task<_>>` is the guard.
- `extractDomainTags` greedy substring matching: "inter-binary-search" benchmark goals pollute the search domain — probe goals phrased to tag deterministically; consider excluding benchmark-origin outcomes from gap detection.
- Result-cap fix changes the output contract — keep the human-readable summary line, add `truncated: true`.
- Windows path separators: normalize to forward slashes in ids and gold answers or comparisons fail on cross-platform CI.

### Pitfalls

- Fixing tools first and measuring second — unfalsifiable improvement against a phantom baseline.
- Prose-string failure encoding — the exact anti-pattern behind Finding 4.
- Remedy names diverging from registered tool names — the unexecutable-remedy loop spin.
- Chunking/embedding tuning first — retriever strategy, not chunk granularity, is the binding constraint (CodeRAG-Bench); the never-applied `ChunkOverlap=200` is real but second-order.
- Swallowing exceptions into silent fallbacks in new code (the `CodebaseRAG.fs:339-341` pattern).
- In-memory-only reputation updates — reproduces the stub's failure across restarts.

### Review notes

Reviews restructured the item around its own premise: phase 2 (symbol graph, combinator wrapper, LLM rewrite, Beta reputation, extra benchmark sets) is speculative until re-measurement shows the cheap fixes insufficient. Layering fixes: GraphPersistence unusable from Cortex; extraction function lands in Cortex/Core with Tools delegating; escalation rungs constructed at the Tools layer if a strategy-injected form is ever needed. Naming: Core's emitted names updated (not Tools aliases), validated in Evolution. New binding requirement: probe-suite outcome isolation from the promotion/selection stores. `CapabilityKind` extension ripple budgeted.

### Effort

Phase 1: ~1 week (re-measure 2-3 days incl. probe suite; lexical tools 2-3 days; lazy index + envelope 2 days). Phase 2 (conditional): symbol graph ~2 weeks; TrackUsageAsync ~2-3 days.

### Success criterion

```
/goal Phase 1: (a) GapDetection.failureRateByDomain over the outcome store reports
search-domain failure rate < 30% computed over >= 10 search-tagged episodes timestamped
after the fix (RalphBridge's gate at RalphBridge.fs:212 green for the search domain),
with probe outcomes isolated from promotion/selection stores (asserted by test);
(b) search_code's result cap provably bounds output (unit test) and glob_search resolves
under the exact name GapDetection emits; (c) search_codebase with an uninitialized index
returns a parseable JSON envelope with status=index_unavailable or fallback results —
never the prose warning (unit test) — and the lazy-init path is exercised by >= 1 test;
(d) dotnet build && dotnet test pass with zero regressions. Phase 2 (only if (a) fails
after phase 1): locate_entity file-level Acc@5 >= 0.7 on the probe suite;
TrackUsageAsync round-trip shows FindAgentsAsync reputation != 0.5 after one tracked
success.
```

### References

Research: `docs/research/frontier-search-capability.md` (findings 1-9, refuted-claim record). Code: `WorkflowTools.fs:78-141`; `SemanticCodeTools.fs:10-41`; `CodebaseRAG.fs:68-74, 280-342, 346+`; `GapDetection.fs:10-163`; `MetaCognitionOrchestrator.fs:36-45`; `CapabilityStore.fs:29-136`; `Domain.fs:80-99`; `RalphBridge.fs:188-213`; `GraphTools.fs:9-61`; `CodeAnalysisTools.fs:245-282`. Papers: Yan et al. CRAG (arXiv:2401.15884); Asai et al. Self-RAG (arXiv:2310.11511); Chen et al. LocAgent (arXiv:2503.09089); Xia et al. Agentless (arXiv:2407.01489); Wang et al. CodeRAG-Bench (arXiv:2406.14497); Yang et al. SWE-agent (arXiv:2405.15793); "Is Grep All You Need?" (arXiv:2605.15184); OrcaLoca (arXiv:2502.00350).

---

## 9. Item 8 — Second-Order Metric + SFT Channel

### Intent

The self-hosting loop has verified first-order wins but no measurable second-order improvement (`docs/research/empirical-self-host-evidence.md` Findings 2, 4, 6, 7): rejected proposals are discarded as console strings (`SelfImprove.fs:107`); `recordWin` is a bare append with duplicates and contradictory targets (`SelfHostingGate.fs:312-316`); `exportDataset` merges the ledger unfiltered (`SelfTrain.fs:105-121`); the 8 promoted `AgentSkill` cases have zero consumers (`AgentRegistry.fs:25-30` drops Capabilities). Round 3 supplies the first two calibration points — union-case edits 8/8 (rounds 1-2, N=4) vs behavioral fixes 2/5 (round 3, N=3) — and three named rejects as the frozen re-measurement set.

**Per review, this is two PRs:** (A) telemetry + export hygiene (STaR/ReST-EM/dedup discipline); (B) the ~20-line AgentSkill routing wiring, which shares zero code with the SFT work and must not ship hostage to it.

### Best practices

Verified-only positives (STaR/ReST-EM): every SFT row traces to an external verifier Accept; hygiene work keeps contradicted/duplicated rows out. Treat the verifier as a free preference oracle — capture rejected proposals at the moment they exist (inside the gate run) because they are unreconstructable afterward. Status/filtering, never deletion. Hold the evaluation set frozen between self-training rounds; change one variable per round. Stratify every metric by difficulty class and N — averaging union-case 8/8 with behavioral 2/5 (10/13) manufactures a fake trend. Report the funnel (generated → parsed → viable → gate-passed), not just the endpoint. Keep the module's pure/IO separation (`SelfHostingGate.fs:20-21`): hashing, dedup decisions, metric folds pure; file IO in thin record functions at the CLI. Align train-time targets with inference-time prompts. Gate accept-rate telemetry is a **separate observability channel** and must never feed WeightedGrammar/PatternSelector — the judge stays independent of the thing being judged.

### Implementation steps

**PR A — telemetry + hygiene:**

1. **Gate-run telemetry.** `GateRunRecord` in `SelfHostingGate.fs` (after `GateVerdict`), **lean per review**: RoundId, TargetTest, TargetFile, Class, Model, N, PromptChars (load-bearing — `a3f730df` names Ollama `num_ctx=4096` truncation as the round-3b suspect), per-proposal outcomes (`{EditsHash; Decision; Rank}` — the rejected mutation JSON + reject reason stored here doubles as the DPO-pair raw material), Verdict, TimestampUtc. Drop MaxConcurrency/Repair*/DurationMs/TargetFileBytes until a metric needs them (~8 rows of data ever expected).
2. **Instrument `runGateBestOfN`** (`:560-646`): accumulate alongside the existing `rejects` list (`:589`); new `runGateBestOfNTelemetry` returns `GateVerdict * GateRunRecord`; `runGateBestOfN` stays a thin fst-wrapper so `runGateGenerated` (`:650`) and existing tests are untouched. **IO placement (review, resolving the plan's self-contradiction):** the telemetry value carries everything; persistence happens at the CLI (`SelfImprove.fs` runSingle `:38` / runBacklog `:101`) via `appendGateRun` to `~/.tars/self_host_gate_runs.jsonl` — no writes inside the parallel evaluation loop.
3. **Difficulty class.** `Class` field on the backlog schema (`SelfImproveBacklog.fs`): `union-case | behavioral | cross-file`, defaulting behavioral. **Seed history by hand** (review): hand-author the three seed JSONL rows (rounds 1-2: 8/8 union-case N=4; round 3: 2/5 behavioral N=3) or check a seed file into docs/ — no `--seed-history` subcommand.
4. **Metrics.** `SelfHostMetrics` module (new file, pure fold): per-(round, class, N) rows — attempts, accepts, accept-rate, viable-proposal rate. `tars self-improve metrics` (extend `SelfImprove.fs:142` dispatch) reads the JSONL. Never group across N values.
5. **Win-ledger hygiene, read-time per review.** No status machine, no v1→v2 envelope migration, no atomic-rewrite lock for a 12-line JSONL. Keep `recordWin` as a bare append **plus `editsHash` on the line** (SHA-256 over LF-normalized decoded OldText+NewText — normalize BEFORE hashing per `applyEditPure`'s CRLF logic `:143-153`; hash decoded strings, not serialized JSON, since STJ escapes `>` as `>`). **Both** recordWin call sites get the field — `:430` (promoteTask) and `:467` (repair-path promote), the second of which the original plan missed. Supersession is **computed at export**: `loadActive = parse → distinctBy (test,file,editsHash) → latest-per-(test,file) → drop records failing a HEAD contains-check` (mechanically catches the spec-gamed search→Reasoning and debugging→Coding records, absent from shipped `AgentDefinition.fs:70-77`). The generic `reconcile` CLI verb waits for a second occurrence; the 3 known-gamed records are handled by the export filter.
6. **Preference pairs, derived not stored** (review): no `PrefRecord` schema, no `self_host_prefs.jsonl`, no ChosenProvenance rules — there is no DPO trainer in the repo. The per-proposal outcomes in `GateRunRecord` (rejected mutation JSON + reason) let a future export derive chosen/rejected pairs when a DPO pipeline exists. Exclusions still apply at that future derivation: parse failures out; same-editsHash chosen/rejected pairs out; infra-noise rejects out.
7. **Prompt-schema alignment.** Fix `selfHostSystemPrompt` (`:274-277`) — it still promises a single `{rationale, old_text, new_text}` mutation while `buildSftExample` (`:283-300`) emits `{rationale, edits:[...]}`. **Review addition:** the 12 existing ledger lines embed the stale system prompt; normalize the system message at export envelope-strip, or the re-exported dataset trains on two contradictory system prompts — the exact inconsistency Finding 7 dings.
8. **Export rewrite.** `SelfTrain.exportDataset` (`SelfTrain.fs:105-121`) uses the filtered load; extend `ExportStats` (`:28-38`) with SelfHostActive/SelfHostSuppressed. Expected: 12 lines → ~8 active (−1 duplicate, −3 gamed). **Compile-order resolution (review — the plan as originally written was unsatisfiable):** `SelfTrain.fs` compiles at fsproj line 62, BEFORE `SelfHostingGate.fs` at 66, yet must call the ledger loader, while the gate must call recordWin. Hoist `Edit`/`GateTask`/`buildSftExample` into a `SelfHostTypes.fs` placed before `SelfTrain.fs`, put the ledger-filter module between it and `SelfTrain.fs`, and have the gate delegate; or move `SelfTrain.fs` later (check `InsightExporter.fs:63` consumers first).

**PR B — AgentSkill routing (small, independent):**

9. **Minimal wiring per review** (no `AgentConfig` field — that breaks 8 `builtinDefaults` records + tests via F# record-shape ripple): add `AgentSkill.toKeyword` (lowercase case name; `Custom s -> s.ToLowerInvariant()`) next to the union (`AgentDefinition.fs:10-26`); at the four `AgentOrchestrator.Register` sites in `Agent.fs` (`:623-627, 729-732, 825-828, 967-971`), look up the loaded `AgentDefinition` by role and **append** `def.Capabilities |> List.map toKeyword` to the existing literal lists. Keep `Register`/`Route` string-typed (the narrowest correct seam — do not couple the orchestrator to the union). Promote to an AgentConfig field only when a second consumer appears.
10. **Routing test:** two stub agents ([Refactoring; Testing] vs [Debugging]) route correctly; all 8 promoted cases round-trip `parseCapability >> toKeyword >> parseCapability`; assert `parseCapability "search" = AgentSkill.Search` exactly (closing the `<> Custom` underspecification that enabled the gaming). Document (don't fix) `Route`'s substring semantics: `goalLower.Contains(cap)` means "testing" doesn't match "test" and "search" matches inside "research".

11. **Round-4 protocol** (docs note in ADR 0002, no code): freeze the three rejected round-3 seeds as held-out; raise `num_ctx` above 4096 FIRST (de-confound truncation) and record inference params in the run record; re-run at identical (N=3, model, seeds); after any fine-tune from the re-exported dataset, same seeds, same settings — the accept-rate delta is the loop's first true second-order data point (DGM benchmark-as-judge).

### Signature sketches (not yet applied)

```fsharp
// SelfHostingGate — lean telemetry; per-proposal outcomes double as future DPO raw material
type ProposalOutcome = { EditsHash: string; Decision: string; Rank: int; MutationJson: string }
type GateRunRecord =
    { RoundId: string; TargetTest: string; TargetFile: string; Class: string
      Model: string; N: int; PromptChars: int
      Proposals: ProposalOutcome list; Verdict: string; TimestampUtc: DateTime }
val runGateBestOfNTelemetry : (* existing args *) -> roundId:string -> cls:string
                              -> Async<GateVerdict * GateRunRecord>
val appendGateRun : GateRunRecord -> unit   // called from CLI only

// Ledger hygiene — read-time, no status machine
val editsHash  : Edit list -> string        // SHA-256 over LF-normalized decoded texts
val loadActive : repoRoot:string -> string list
// parse -> distinctBy (test,file,editsHash) -> latest per (test,file) -> HEAD contains-check

// SelfHostMetrics (pure)
type MetricRow = { RoundId: string; Class: string; N: int
                   Attempts: int; Accepts: int; AcceptRate: float; ViableRate: float }
val acceptRateByRound : GateRunRecord list -> MetricRow list

// Routing (PR B)
module AgentSkill = val toKeyword : AgentSkill -> string
// Register sites: existing literals @ (defByRole role |> Option.map (fun d ->
//     d.Capabilities |> List.map AgentSkill.toKeyword) |> Option.defaultValue [])
```

### Edge cases

- CRLF/JSON-escaping in editsHash — hash decoded, LF-normalized strings or identical edits hash differently across records.
- Two of the 12 v1 lines are byte-identical (records 0-1) — collapse at export.
- `recordWin` stays best-effort: never throw into the gate path (current try/with preserves the Accept even if the write fails); dedup failure degrades to plain append.
- HEAD contains-check false positives/negatives — list per-record evidence in export stats; tombstone-by-filter is reversible by construction (nothing is rewritten).
- Fixed-N comparability: rounds 1-2 ran N=4, round 3 N=3 — group by N, never average across.
- `num_ctx` confound — record inference params or the round-4 delta conflates context-window fix with model improvement.

### Pitfalls

- Computing the metric from PROMOTED/total console summaries — unauditable narrative (Finding 12's exact complaint).
- Training on the ledger as-is — contradictory targets teach inconsistency; duplicates over-weight one template (Lee et al. 2021).
- Letting an LLM judge supersession — mechanical (same key, later Accept) or HEAD-anchored only (Self-Rewarding judge-inflation failure mode).
- Fabricating preference pairs by rationalizing a chosen answer for unfixed seeds (STaR rationalization hindsight bias) — rejected-only stays chosen-less until a verified fix exists.
- Wiring routing via a parallel frontmatter string list — the point is consuming the existing `AgentSkill` union so the 8 promoted cases gain a consumer.
- Running round 4 with a different model or N "because it is available".

### Review notes

Reviews cut the item roughly in half: DPO schema/file (no trainer exists — derive from gate-run records later), the ledger status machine + migration + lock (read-time filtering on a 12-line file), the `--seed-history` and `reconcile` CLI verbs (hand-authored rows; export filter), and five telemetry fields. Structural fixes: the SelfTrain↔SelfHostingGate compile-order contradiction (SelfHostTypes hoist), the missed second `recordWin` call site (`:467`), IO moved out of the parallel loop to the CLI (honoring the module's own pure/IO invariant), the stale-system-prompt mix in existing ledger lines normalized at export, and the routing wiring shrunk to a definition-lookup at the four Register sites (no `AgentConfig` ripple). Layering verified clean: string-typed `Register` is the narrowest seam; telemetry is read-only observability, never feeding the weights it measures.

### Effort

PR A ~1 day (telemetry + metrics ~0.5d; export hygiene + prompt alignment ~0.5d). PR B ~0.25 day. Fine-tune stays the external runbook step.

### Success criterion

```
/goal From v2/: (1) dotnet test green (no regressions); (2) `tars self-improve metrics`
prints per-(round, class, N) accept-rate rows covering rounds 1-3 including
{round-3, behavioral, N=3, 2/5}, sourced from persisted
~/.tars/self_host_gate_runs.jsonl — not hardcoded, not console-derived; (3) after
re-running `tars self-train`, the exported dataset contains zero assistant targets
mapping "search"->Reasoning or "debugging"->Coding, at most one example per
(test,file,editsHash), one consistent system prompt, and ExportStats reports
SelfHostSuppressed >= 4; (4) gate-run records for any best-of-N run carry every rejected
proposal's mutation JSON + reject reason (spot-checked by test on a stubbed run);
(5) [PR B] an agent whose loaded .md definition declares refactoring routes via
AgentOrchestrator.Route for a goal containing "refactoring", and grep shows >= 1
AgentSkill consumer file outside AgentDefinition.fs.
```

### References

Code: `SelfHostingGate.fs:20-21, 81-134, 143-153, 274-277, 283-300, 312-316, 418-431, 467, 560-650`; `SelfTrain.fs:28-43, 105-121`; `SelfImprove.fs:38-47, 96-115, 142-145`; `AgentDefinition.fs:9-26, 43, 61-78`; `AgentRegistry.fs:16-30, 36-133`; `AgentOrchestrator.fs:47-76`; `Agent.fs:623-627, 729-732, 825-828, 967-971`; commit `a3f730df`; ADRs 0002/0003. Research: `docs/research/empirical-self-host-evidence.md` Findings 2, 4, 6, 7; Opportunities 1, 3, 4. Papers: Zelikman et al. STaR (arXiv:2203.14465); AdaSTaR (arXiv:2505.16322); Lee et al. dedup (arXiv:2107.06499); Rafailov et al. DPO (arXiv:2305.18290); Yuan et al. Self-Rewarding LMs (arXiv:2401.10020); Singh et al. ReST-EM (arXiv:2312.06585); Zhang et al. Darwin Gödel Machine (arXiv:2505.22954); Krakovna et al., Specification gaming (DeepMind 2020).

---

## 10. Item 9 — Diversity Mechanism for the Evolutionary Layer

### Intent

`ReplicatorDynamics` is constant-fitness Fisherian selection whose textbook endpoint is monoculture (`docs/research/theory-replicator-dynamics.md` F1-F2), with no mutation term (F3), a floor mis-applied (F4 — round 3 fixed placement but left the residual: the clamp at `ReplicatorDynamics.fs:127-129` runs AFTER renormalization, so floored species push the sum above 1 with no redistribution), an MCP path fed `Map.empty` — a provable no-op (F5, `McpGrammarTools.fs:118`), and a dead `--steps` flag (F6, `GrammarCommand.fs:94-95` builds a config it never passes). Selection without variation cannot discover anything. **V1 (this item): the small high-leverage fixes — mutation term, sum-to-1 + floor closed properly, MCP no-op killed, `--steps` honored, F7 write-back gated.** Islands + temperature>0 variant generation + stepping-stone archive are **phase 2, filed as a separate gated issue** (blocked on items 3 and 5 so the population and its fitness are trustworthy first).

### Best practices

Uniform mutation matrix first (x ← (1−μ)·x + μ/n); escalate to a lineage-coupled Q only once the simple form is proven. Keep mutation rare — behavioral diversity (internal equilibria) is largest when mutation is rare (Dynamic Games & Applications 2019); start μ≈0.02, expose as config. FunSearch reset policy for phase 2: reset the worst half of islands from a top island's best member; archive elites BEFORE any reset. Cheap, low-dimensional behavior descriptors for any archive. Variants via `LlmFactory.create` only, deduped by template hash, parse failures dropped.

### Implementation steps

1. **Mutation term** (`ReplicatorDynamics.fs:37-53, 112`). Add `MutationRate: float` (default 0.02) to `ReplicatorConfig`. Per review, **pass the config to `step`** (`step (config: ReplicatorConfig) species`), not three loose positional floats. After the selection update x_i += dt·x_i·(f_i−f̄) and renormalize: x_i ← (1−μ)·x_i + μ/n. This is the simplest replicator-mutator (Q = (1−μ)I + (μ/n)J; Page & Nowak 2002) and admits interior equilibria.
2. **Floor: subsumed by mutation — delete, don't project** (review). After the mutation step the vector is exactly on the simplex with every entry ≥ μ/n; extinction is impossible by construction. **Do not add `projectToCappedSimplex`** (two mechanisms for one job, plus an infeasibility guard and termination proof for nothing). Remove the trailing `max floor` clamp; if a hard 1% bound is required, set μ so μ/n meets it. This closes the F4 residual with negative net LOC.
3. **Prune semantics — decide, don't defer** (review). With every entry pinned ≥ μ/n, the prune partition (`:179-180`, threshold 0.001) and `GrammarCommand.fs:141`'s `Weight > 0.001` filter are structurally unreachable — F6's shape again (a flag that parses and does nothing). Pick one: prune pre-mutation inside `simulate`; define pruning as "pinned at μ/n for K consecutive steps"; or drop `--prune`. State the choice in the PR.
4. **Kill the MCP no-op, honestly scoped** (review). The originally-proposed read-time fuzzy join (`WeightedRule.PatternName` vs `sprintf "%A" PatternKind`) is a **category error**: `PatternOutcomeStore` records reasoning-pattern outcomes (ChainOfThought/ReAct, keyed PatternKind+Goal) while grammar species are promoted grammar rules keyed by PatternId — two different pattern universes; the join's failure mode is a silent empty map, F5 reborn. V1: move the CLI's existing synthesized-outcomes builder (`GrammarCommand.fs:84-92`) into **Tars.Evolution at/before `McpGrammarTools.fs` in compile order** (`ReplicatorDynamics.fs` is fsproj item 56, `McpGrammarTools.fs` 58 — defining it in ReplicatorDynamics works; the CLI cannot host it or the MCP path can't reuse it) and call it from `McpGrammarTools.fs:118` (~10-line diff). Real per-rule outcomes require write-time PatternId-keyed records — item 3's grounded-signal work; co-design or scope explicitly as "best-effort, mostly fallback" with a coverage metric.
5. **Honor `--steps`/config** (F6). Per review, `let evolveEcosystem (?config) ...` is **invalid F#** — optional `?` params are only legal on type members. Use the two-function form to keep the three existing call sites source-compatible: `evolveEcosystemWith (config: ReplicatorConfig) rules outcomes` + existing `evolveEcosystem` delegating with `defaultConfig`. `GrammarCommand.fs:95` passes its already-built config; MCP tool gains optional MaxSteps.
6. **Gate the F7 write-back — in this PR, not deferred** (review). Today the MCP no-op and near-static dynamics accidentally protect Bayesian weights from `GrammarCommand.fs:139`'s proportions write-back; once mutation makes proportions genuinely move (equal-fitness cases drift toward 1/n), every `grammar evolve` run substantively clobbers `WeightedGrammar`'s Bayesian weights. Gate or dual-field the write-back in the same PR as fixes 1-5, coordinated with item 3's Weight-ownership contract.
7. **Tests — pure level only** (review): the proposed integration test writing through `~/.tars` live paths would clobber the developer's store. (a) post-step vector sums to 1 ± 1e-9 with every entry ≥ μ/n over random vectors; (b) 50-step run from a 90/10 two-species start with distinct fitness retains both species > 5% with mutation ON, collapses one below 5% with mutation OFF (proves the mutation term is the diversity mechanism); (c) a unit test that the parsed `--steps` value reaches `simulate` (covers F6 without touching disk).

**Phase 2 (separate gated issue, after items 3 & 5):** `IslandModel.fs` + `SteppingStoneArchive.fs` in Tars.Evolution (correct layer); `compileVariantsFromTrace` raising `TraceCompiler.fs:82`'s `Temperature = Some 0.0` to configurable >0 with k-sampling, `ILlmService` as a parameter (LlmFactory stays at CLI call sites). Constraints from review: if PatternSelector (Cortex) ever consumes archive elites it reads `~/.tars/promotion/archive.json` from disk (a Cortex→Evolution project reference is a cycle); archive Score sourced from the replicator/gate path, never from `EvolutionaryPatternBreeder`'s 8-float genome (F9 — name types distinctly: `ArchivedVariant`, not Individual/Genome); archive before any island reset.

### Signature sketches (not yet applied)

```fsharp
type ReplicatorConfig =
    { TimeStep: float; Steps: int; PruneThreshold: float; MutationRate: float }
    // SmoothingFloor deleted — mutation term guarantees every entry >= MutationRate/n

// config-passing step; no loose positional floats
let step (config: ReplicatorConfig) (species: GrammarSpecies list) : GrammarSpecies list
//   sel_i = p_i + dt*p_i*(f_i - fbar)   (renormalize)
//   out_i = (1-mu)*sel_i + mu/n         (already on the simplex, entries >= mu/n)

// two-function form: existing call sites stay source-compatible; ?config on a module
// let-binding is invalid F#
let evolveEcosystemWith (config: ReplicatorConfig) (rules: WeightedRule list)
                        (outcomesById: Map<string,(bool*int64) list>) : ReplicatorResult
let evolveEcosystem rules outcomesById = evolveEcosystemWith defaultConfig rules outcomesById

// v1 outcome supply — synthesized builder relocated to Tars.Evolution
// (at/before McpGrammarTools.fs in compile order) so MCP and CLI share it
let synthesizeOutcomes (rules: WeightedRule list) : Map<string,(bool*int64) list>
```

### Edge cases

- Very small populations: n=1 → μ/n barycentre is the point itself (no-op, correct); ensure prune-then-mutate ordering never drops the sole survivor.
- `--steps 0` → clean no-op (StepsRun=0); negative clamps to 0.
- All-equal fitness (the old MCP empty-outcome case): the mutator now actively drifts proportions toward uniform rather than freezing them — desired, but existing test expectations assuming unchanged proportions must be updated (F5 described the old behavior).
- Phase 2: island reset can lose the globally-best variant living only in a reset island — archive first; k-sampled LLM variants dedup by template hash, parse failures dropped.

### Pitfalls

- Shipping islands before the outcome signal is grounded — evolving on circular/confounded fitness amplifies noise.
- Treating the mutation term as a diversity guarantee while the MCP path stays on `Map.empty` — equal-fitness uniform drift "looks like diversity" but carries zero information (F5 in a new costume; the fuzzy-join variant is the same trap).
- Reusing the single Weight field for replicator proportions (F7) — do not extend this pattern; gate the write-back now.
- Calling the popularity heuristic `detectESS`/`IsStable` "evolutionary stability" after adding mutation — still not an ESS test without a payoff matrix (F8).
- Confusing the 8-float GA breeder with the replicator population — different formalisms (F9); diversity belongs in the replicator/island layer.

### Review notes

Reviews restructured the item: phase 2 cut entirely from this unit (speculative over a handful of rules); the capped-simplex projection deleted (mutation subsumes the floor — the single largest simplification); the `?config` signature bug caught (invalid F# on module functions → two-function form); the fuzzy outcome-join demoted from "Fix 3" to an honest relocation of the synthesized builder (real grounding is item 3's schema work); prune semantics forced to an explicit decision; F7 write-back gating pulled INTO phase 1 (the fixes otherwise make the collision strictly worse); tests kept pure (no live `~/.tars` writes); placement pinned (builder in Tars.Evolution before `McpGrammarTools.fs` in compile order; phase-2 modules in Evolution with the file-seam rule for any Cortex consumption).

### Effort

V1 (mutation, floor deletion, prune decision, MCP builder relocation, config plumbing, write-back gate, tests): ~2 days. Phase 2: ~2+ weeks, separate issue gated on items 3 and 5.

### Success criterion

```
/goal (a) property test: post-step vector satisfies sum = 1 +/- 1e-9 AND every
proportion >= MutationRate/n over random inputs; (b) a 50-step simulation from a 90/10
two-species start with distinct fitness retains BOTH species above 5% with the mutation
term on, and drives one below 5% with it off; (c) a unit test proves the parsed --steps
value reaches simulate (StepsRun = N) and `grammar_evolve` (MCP) no longer passes
Map.empty; (d) the proportions write-back to weights.json is gated or dual-fielded, with
a test asserting Bayesian Alpha/Beta survive a grammar-evolve run unchanged; (e) the
--prune decision is implemented as stated in the PR (reachable, redefined, or removed) —
no flag that parses and does nothing.
```

### References

Code: `ReplicatorDynamics.fs:37-53, 112-129, 179-180, 189-194`; `McpGrammarTools.fs:112-128`; `GrammarCommand.fs:84-95, 139-141`; `PatternSelector.fs:14-86`; `TraceCompiler.fs:70-84`; `EvolutionaryPatternBreeder.fs:33, 113-147`. Research: `docs/research/theory-replicator-dynamics.md` F1-F10, Opportunities 1, 3. Papers: Page & Nowak 2002 (J. Theor. Biol. 219:93-98); Hofbauer & Sigmund 1998 (Ch. 7); "Equilibrium Properties of the Replicator-Mutator Equation" (Dyn. Games Appl. 2019); Romera-Paredes et al. FunSearch (Nature 2023); Mouret & Clune MAP-Elites (2015).

---

## 11. Sequencing

Dependency-ordered execution across items. Arrows read "unblocks".

**Wave 1 — foundations (parallelizable):**

1. **Item 1 (outcome store + provenance)** — first, unconditionally. The codec/JSONL/provenance work unblocks: item 3 (shared `PatternOutcome` schema — one migration, not two), item 6 (per-problem rates + the `benchmark-gen:` tag need a sane store), item 7 (probe-suite isolation), item 9 (any future grounded join). It is also the smallest of the open round-3 residuals.
2. **Item 4 slices A+B (constrained decoding activation + router)** — independent of everything; improves JSON reliability for every downstream LLM call including item 6's generator. Slice C (wot.ebnf) is a detached follow-up.
3. **Item 8 PR B (AgentSkill routing)** — ~20 lines, fully independent; ship any time.

**Wave 2 — statistical core:**

4. **Item 3 (Beta priors + posterior-to-ranking + Thompson)** — after item 1 (DTO coordination). Decides the **Weight-ownership contract** that items 5 and 9 must respect, and lands the regularized posteriors item 5's demotion follow-up requires. The severing + producer decision made here determines whether item 9's grounded join ever becomes real.
5. **Item 9 v1 (mutation term + fixes)** — after item 3's Weight contract (the F7 write-back gate references it); mechanically small, can overlap late item-3 work.
6. **Item 8 PR A (telemetry + SFT hygiene)** — independent of waves; best landed before any round-4 gate runs so telemetry exists from the first re-measurement.

**Wave 3 — gates and generation:**

7. **Item 5 v1 (MDL gate + level sync)** — after item 1 (uses the same tokenizer/CRLF discipline); the demotion follow-up waits for item 3.
8. **Item 6 (curriculum triad)** — after item 1 (store tagging) and ideally item 4 A+B (constrained generator output). The livelock commit and RetroactionLoop-fallback PR trail separately.
9. **Item 7 phase 1 (search re-measure + lexical fixes)** — after item 1 (probe isolation); phase 2 only on a still-red measurement.

**Cross-cutting invariant (all waves):** `PatternOutcomeStore` is the single evidence stream for pattern selection and promotion ranking. Item 6's Gate-B probes, item 7's probe suite, and item 8's gate telemetry must each stay out of it (bypass, tag-and-filter, or separate file respectively) — three independent reviews flagged the same Goodhart channel.

**Gate-closeable vs human-only:**

- *Human-only (proven — the gate already tried and failed, or the change is architectural):* item 1 codec (`outcome-kind-codec` rejected best-of-3), item 3 steps 1-3 (`beta-prior-degenerate` repair regressed; `posterior-to-weight` had 0 applicable proposals), item 4's router/wire-format work (cross-file, new types), item 5's plumbing+gate (two-file coordinated change), item 6 and item 8 PR A (new modules, compile-order surgery).
- *Gate-closeable candidates (single-file, behavioral, seed-able as red tests once the surrounding schema work lands):* item 9's mutation term and floor deletion, item 7's `search_code` regex + cap fix, item 8 PR B's `toKeyword` + Register-site append, item 5's subjective-criteria default flip. Seed these into the round-4+ backlog with `Class = behavioral` — they double as the difficulty-calibrated measurement set item 8's protocol needs.
- *Round-4 protocol (item 8):* raise `num_ctx` first, then re-run the three frozen round-3 rejects at identical settings. Those three seeds correspond to work items 1 and 3 above — meaning wave 1-2 human implementation **is** the consolidation that makes them pairable as preference data, closing the loop between the human-work frontier and the second-order metric.

---

## 12. Peer-research reconciliation

Parallel deep-research sessions on items 3, 6, and 7 returned findings after the synthesis was written; most corroborated the sections above, but four points were dropped by the review's scope-cutting and are restored here as binding amendments. Sources: session scratch notes `item3-thompson-peer-findings.md`, `item6-curriculum-peer-findings.md`, `item7-search-peer-findings.md`.

1. **Item 7 — symbol-graph cache key is `{HEAD-sha}:{is-dirty}`, not mtime (correctness fix).** §8 step 5 says "in-memory mtime-keyed rebuild." mtime is a known-bad invalidation key: it changes without content changes and can change spuriously (Docker's build cache explicitly ignores it). Use the git HEAD SHA + dirty-worktree flag as the primary cache key; on a dirty worktree, validate only the touched/untracked set by content hash. This also aligns the index with the pinned-SHA benchmark harness (§8 step 2), so a probe's repo state matches its gold labels. mtime demoted to a secondary cheap check on the dirty set only.

2. **Item 7 — the benchmark must separate file-level from symbol-level recall.** §8 collapsed to one probe suite with a file-level `Acc@5 ≥ 0.7` gate. Per the peer literature (LocAgent 94% file-Acc@5; SWE-Explore "strong at files, recall-limited at the line level"), file-level recall is largely solved and symbol-level is the actual gap — so a passing file score can mask a weak symbol score. Keep the single suite, but **label each probe with both its gold file(s) and gold symbol(s)** and report the two recalls separately; the phase-2 trigger should be a still-red *symbol-level* recall, not file-level. Gold-set hygiene the peer flagged (hand-curate the tight fix-implementing file set, not every PR-touched file; state per-file vs strict-all-in-top-k semantics; pin each query's commit SHA; include multi-file fixes) applies when the suite is built.

3. **Item 6 — structure `CurriculumHealth`/selection so an ALP learning-progress signal can replace the static `[0.1, 0.9]` band later (forward-compat hook).** The review cut the health metrics; keep them cut for v1, but this was the peer's highest-value point and costs nothing now if the seam is shaped right. The static GOID band is a stateless proxy for the Zone of Proximal Development; the stronger, adaptive form (ALP-GMM, Portelas & Oudeyer, CoRL 2019) targets problem families where the agent's pass-rate is *changing fastest*. The fixed band will bite as the agent improves (in-band problems drift above 0.9). Concretely: make the admission predicate a function value (`admit : rate -> bool`) rather than an inline `0.1 ≤ p̂ ≤ 0.9` literal, and record per-family pass-rate history (already implied by the `benchmark-gen:` tagged outcomes) so an ALP signal can be computed later without a schema change.

4. **Item 6 — dedup on the reference solution's AST/edit-distance, not description Jaccard alone.** §8 Gate C uses token-set Jaccard on the normalized *description* (< 0.7). Surface-text dedup misses renamed-variable clones — two problems with different prose but structurally identical `ReferenceSolution`s. Add a secondary edit-distance/normalized-AST check on `ReferenceSolution` (Self-Instruct's ROUGE-L≥0.7 norm plus a code-structure check); generator mode-collapse is the highest-frequency failure of LLM task generation, so this is cheap insurance, not gold-plating.

Corroborated-and-already-present (no change needed): item 3's Alpha/Beta-persisted-on-the-DTO (the peer's "you cannot recover (α,β) from SuccessRate + SelectionCount" concern — §4 step 1 already persists them as source of truth); item 3's pseudo-count prior injection (§4 step 6); item 3's MathNet-vs-hand-rolled shape<1 boost-trick (§4 step 4); item 6's ReferenceSolution-on-a-wrapper and the wrong-probe vacuous-validator gate (§7 steps 2, 4); item 7's `not_found`-as-first-class-status envelope (§8 step 4) and the item-7↔item-3 shared-bandit layering note (§8 step 6).

**Cross-cutting theme the peers surfaced independently:** items 3 (pattern-selection bandit), 6 (curriculum pass-rate estimator), and 7 (search-tool reputation) all reduce to **one shared Beta-Binomial posterior substrate**. Layering forbids a single implementation (Cortex cannot reference Evolution), so the pure update belongs in `Tars.Core` where all three layers can call it — building it once, correctly, in item 3's wave-2 work and having items 6 and 7 consume it is the highest-leverage consolidation across the whole plan.
