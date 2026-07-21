---
title: "Promotion Dynamics in the TARS Staircase: An Empirical Audit of the Live Promotion State"
date: 2026-07-21
track: empirical
unit: promotion-dynamics
status: verified
---

# Promotion Dynamics in the TARS Staircase: An Empirical Audit of the Live Promotion State

## Abstract

We audit the live promotion state of the TARS self-improvement loop as persisted in the four JSON stores under `~/.tars/promotion/` (`index.json`, `weights.json`, `recurrence.json`, `lineage.json`), cross-checked against the pipeline code in `v2/src/Tars.Evolution/`. The population holds 9 patterns whose staircase distribution is sharply polarized: 5 at the top rung (GrammarRule), 0 at Builder or DslClause, 2 at Helper, and 2 at Implementation. All 5 top-rung patterns are GA-seeded; every one of the 4 organically-promoted patterns is stuck on the bottom two rungs. The weight ecosystem is statistically diverse (Gini 0.14–0.19, normalized Shannon entropy 0.97–0.98) — i.e. *not* collapsed to a few winners — but that diversity is an artifact of softmax over near-identical inputs rather than genuine competition, and weight anti-correlates with fitness (Pearson r(SuccessRate, Weight) = −0.85): the highest-success patterns carry the lowest weights. The Grammar Governor approved 22 of 23 promotions (95.7%, 21 at confidence 1.0), showing near-zero selectivity. Two organic patterns are demonstrably stalled at Implementation because they lack a RollbackExpansion (round-trip validation fails), and by construction the staircase is a monotone ratchet with no demotion path — oscillation is impossible, but irreversibility is a live risk. The dataset is tiny (essentially two batch-seed events in March 2026) and the four stores are mutually inconsistent (stale counts, stale Level fields, a default-fallback weight), which limits inference. All eight findings below survived two independent adversarial verification passes; no claims were flagged as unverifiable and none were refuted.

## Background

TARS promotes recurring behavioral patterns up a five-rung abstraction staircase — Implementation → Helper → Builder → DslClause → GrammarRule — defined as the `PromotionLevel` union at `v2/src/Tars.Evolution/PromotionTypes.fs:6` with ranks 0–4 assigned at `PromotionTypes.fs:13`. The Compound Engineering loop extracts candidate patterns from task traces, the Grammar Governor (`PromotionPipeline.fs`) evaluates them against an eight-criterion checklist, round-trip validation (`RoundtripValidation.fs`) checks that a pattern can be expanded back to its constituent steps via a `RollbackExpansion`, and approved candidates are persisted across four stores: a recurrence store (occurrence tracking), a lineage store (governance decisions), a weight store (replicator-dynamics / Bayesian weights consumed by `Tars.Cortex/PatternSelector.fs`), and a derived index (`PromotionIndex.fs`, persisted to `~/.tars/promotion/index.json`) that bridges promotion output to agent-side pattern selection.

Two provenances feed the ecosystem. *Organic* patterns are extracted from TARS's own task history (tasks `task_001`–`task_010`, seeded 2026-03-11). *GA-seeded* patterns are the product of a static code analysis of the Guitar Alchemist repository, injected by `GaPatternSeeder.fs` on 2026-03-14 with hand-authored templates, rollback expansions, and scores in the 0.78–0.90 range. The research question for this unit is whether the observed promotion dynamics — occupancy of the staircase, weight distribution, governor selectivity, and stall/oscillation behavior — reflect a functioning competitive ecosystem or artifacts of how the data was seeded and how the pipeline is wired.

The empirical basis is deliberately narrow: 9 patterns, 23 lineage records, 8 learned weights, all produced within two sessions in March 2026 (with an index/weights regeneration on 2026-06-23). Every quantitative claim below was recomputed independently during adversarial review, and every code anchor was checked at the cited line.

## Findings

### 1. The staircase occupancy is bimodal and hollow in the middle

**Claim.** The staircase occupancy is polarized at its extremes: 5 patterns at GrammarRule (rank 4), 0 at DslClause (rank 3), 0 at Builder (rank 2), 2 at Helper (rank 1), and 2 at Implementation (rank 0). No pattern currently rests on either middle rung. *(Confidence: high; confirmed by both review passes.)*

**Evidence.** `index.json` (PatternCount = 9) shows level counts of exactly {GrammarRule: 5, Helper: 2, Implementation: 2}. The middle rungs are purely transient: `lineage.json` shows that every GA pattern passed through Builder (16:37:17) and DslClause (16:39:48) within the same 2026-03-14 session, and none rested there. The 5 GrammarRule patterns are all GA-seeded; the 2 Helper patterns (`hypothesis_test_loop`, `decompose_and_solve`) and 2 Implementation patterns (`extract_test`, `verify_then_commit`) are all organic.

**Code anchors.** `v2/src/Tars.Evolution/PromotionTypes.fs:6` (the five-level union), `v2/src/Tars.Evolution/PromotionTypes.fs:13` (rank 0–4), `v2/src/Tars.Evolution/PromotionIndex.fs:82` (LevelRank derived from CurrentLevel).

The hollow middle means Builder and DslClause currently function as pass-through states with no durable inhabitants — a structural observation taken up in the Opportunities section.

### 2. The top of the staircase is a provenance monoculture

**Claim.** Five of nine patterns are GA-seeded (name prefix `ga.`) and they occupy 100% of the Builder, DslClause, and GrammarRule tiers ever reached; no organically-promoted pattern has ever risen above Helper. *(Confidence: high; confirmed by both review passes.)*

**Evidence.** The GA-seeded set — `ga.confidence_evidence_response`, `ga.domain_skill_fastpath`, `ga.orchestrator_pipeline`, `ga.routing_fallback_cascade`, `ga.hook_lifecycle_fsm` — all sit at GrammarRule. The organic set — `hypothesis_test_loop` (Helper), `decompose_and_solve` (Helper), `extract_test` (Implementation), `verify_then_commit` (Implementation) — tops out at Helper. Every lineage transition into the upper three tiers (15 records) belongs to a `ga.*` pattern id. `GaPatternSeeder.fs` feeds the pipeline hand-authored artifacts with pre-written rollback expansions and scores of 0.78–0.90, while organic task-derived patterns score lower and — in the two Implementation-stalled cases — lack rollbacks entirely.

Review noted two minor imprecisions in the original evidence text, which we correct here for the record: (a) two organic patterns (`hypothesis_test_loop`, `decompose_and_solve`) *do* carry rollback expansions via their lineage records — only `extract_test` and `verify_then_commit` lack them; and (b) `extract_test` scores 0.82, outside the 0.53–0.67 range quoted for organics. Neither correction affects the claim itself, which both reviewers confirmed in full.

**Code anchors.** `v2/src/Tars.Evolution/GaPatternSeeder.fs:187` (seeder runs the promotion pipeline on hand-authored artifacts), `v2/src/Tars.Evolution/GaPatternSeeder.fs:163` (artifact definitions), `v2/src/Tars.Evolution/PromotionPipeline.fs:205` (AutoValidatable gated on `RollbackExpansion.IsSome`).

The implication is uncomfortable for the self-hosting narrative: the only grammar-level abstractions in the system came from a static external seed, not from the loop's own discovery process. This mirrors the central difficulty in library-learning systems such as DreamCoder (Ellis et al., 2021), where bootstrapping a reusable abstraction library from the system's own traces is the hard part, and hand-seeded primitives dominate early libraries.

### 3. The weight ecosystem is statistically diverse, not collapsed

**Claim.** Over the 8 learned weights in `weights.json`, Gini = 0.139 and normalized Shannon entropy = 0.984 (2.95/3.0 bits); over the 9 index weights, Gini = 0.188 and normalized entropy = 0.972 (3.08/3.17 bits). The distribution has not collapsed to a few winners. *(Confidence: high; both reviewers independently recomputed all four statistics and matched them to three decimal places.)*

**Evidence.** The `weights.json` weight vector is [0.372, 0.306, 0.322, 0.206, 0.208, 0.199, 0.191, 0.196], summing to 2.0. The 5 GA weights sum to exactly 1.0 and are near-uniform (~0.20 each) because they were produced by a single softmax over near-identical logits (RawScore = 0 blended with AverageScore × 0.5 — near-identical rather than strictly identical, per review); the 3 evolved weights (0.31–0.37) are the replicator-dynamics survivors. The high entropy therefore reflects *near-uniformity within a batch*, not vigorous competitive spread: it is a mechanical consequence of softmaxing one homogeneous cohort, layered beside a separately-normalized cohort.

**Code anchors.** `v2/src/Tars.Evolution/WeightedGrammar.fs:79` (softmax), `v2/src/Tars.Evolution/WeightedGrammar.fs:113` (`fromRecurrenceRecords`), `v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:139` (replicator rewrite of evolved weights).

The interpretive lesson generalizes: entropy near maximum can mean "healthy diversity" or "the weight signal is uninformative," and on this data it means the latter. In replicator-dynamics terms (Accinelli & Carrera, 2010), the GA cohort has never actually played the game — its members share one fitness and one weight by construction.

### 4. Weight is inverted against fitness and against staircase level

**Claim.** Pearson r(SuccessRate, Weight) = −0.85 across the 8 weight-store entries. The 5 GA patterns have SuccessRate = 1.0 yet the lowest weights (~0.20); the organic patterns with SuccessRate 0.535–0.745 carry the highest weights (0.306–0.372). *(Confidence: high; both reviewers recomputed r = −0.852.)*

**Evidence.** From `weights.json`: `hypothesis_test_loop` SuccessRate 0.745 / Weight 0.372; `decompose_and_solve` 0.637 / 0.322; `verify_then_commit` 0.535 / 0.306; all five `ga.*` entries SuccessRate 1.0 / Weight 0.191–0.208. The mechanism is a two-path weight assignment that never reconciles: GA weights entered via the missing-record path (`fromRecurrenceRecords`, softmax over RawScore = 0, Source = Tars) and were never boosted by subsequent success — review confirmed that `updateWeight` (`WeightedGrammar.fs:136–147`) updates SuccessRate, Confidence, and SelectionCount but never Weight, so the GA patterns' four successful selections each left their weights untouched. The organic patterns, by contrast, were re-weighted by replicator dynamics from RawScores of 6–7. `PatternSelector` consumes this Weight additively as a boost signal, so the inversion is live in agent-side ranking, not merely cosmetic.

**Code anchors.** `v2/src/Tars.Evolution/PromotionPipeline.fs:316` (missing-record weight creation), `v2/src/Tars.Evolution/WeightedGrammar.fs:119` (softmax entry path), `v2/src/Tars.Cortex/PatternSelector.fs:151` (index Weight consumed in promotion boost), `v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:136` (replicator rewrite, Source = Evolved).

This is the single most actionable defect found in this unit: the selector's weight prior ranks the empirically best patterns (success = 1.0, grammar-level) *last*.

### 5. The Grammar Governor exhibits near-zero selectivity

**Claim.** 22 of 23 lineage decisions are Approve (95.7%), 21 of them at confidence 1.0 (8/8 criteria met). The single Reject is `verify_then_commit`, for a failed round-trip (semantic match 0.00, no RollbackExpansion). *(Confidence: high; both reviewers recomputed the decision and confidence distributions exactly.)*

**Evidence.** `lineage.json` decision counts: {Approve: 22, Reject: 1}; confidence distribution: {1.0: 21, 0.875: 1, 0.75: 1}. The sole Reject (id `7bf2cf99`) carries the message "Round-trip validation failed (semantic match: 0.00). No RollbackExpansion — automatic failure". Structurally, `validateDeterministic` defaults the subjective criteria `MoreReadable` and `ComposesCleanly` to `true`, and only hard-checks `MinOccurrences`, `StableSemantics`, `AutoValidatable`, `NoOverlap`, and `ImprovesPlanning`; review adds that `RemovesComplexity` is also deterministically checked but trivially satisfied (template length > 0). Because the live pipeline invokes `validate` with the LLM assessment argument set to `None` (`PromotionPipeline.fs:333`), the subjective criteria never bind — confidence 1.0 is the default outcome, not an earned one.

**Code anchors.** `v2/src/Tars.Evolution/PromotionPipeline.fs:199` (deterministic criteria with defaults), `v2/src/Tars.Evolution/PromotionPipeline.fs:211` (criteria aggregation), `v2/src/Tars.Evolution/PromotionPipeline.fs:335` (live call with no LLM assessment).

An approval gate that passes 96% of candidates supplies essentially no selection pressure; the staircase's occupancy is therefore governed by eligibility mechanics (chiefly the rollback gate, Finding 6), not by quality discrimination.

### 6. Two organic patterns are stalled at Implementation by the rollback gate

**Claim.** `extract_test` and `verify_then_commit` are stalled at Implementation for the same mechanical reason: a null RollbackExpansion forces `AutoValidatable = false` and round-trip validation to fail. `extract_test` (OccurrenceCount 6 ≥ threshold 3, score 0.82) has *zero* lineage records — it has never even been governed; `verify_then_commit` was actively Rejected. *(Confidence: high; confirmed by both review passes.)*

**Evidence.** `index.json`: `extract_test` has RollbackExpansion = null, Level = Implementation, OccurrenceCount = 6, Score = 0.8185; its PatternId (`f3eb7662`) is entirely absent from `lineage.json`. `verify_then_commit`'s lineage record `7bf2cf99` is the Reject quoted in Finding 5. In code, `validateDeterministic` sets `AutoValidatable = candidate.RollbackExpansion.IsSome`, and `RoundtripValidation.fs` states that a pattern without a RollbackExpansion automatically fails. One reviewer's caveat is worth preserving: for `extract_test` the rollback gate is the *predicted* blocker rather than the demonstrated one — the pattern never reached governance at all — but its null rollback guarantees failure if it ever is governed, and the fact that a threshold-eligible pattern was never submitted is itself a pipeline gap.

**Code anchors.** `v2/src/Tars.Evolution/PromotionPipeline.fs:205` (the gate), `v2/src/Tars.Evolution/PromotionPipeline.fs:340` (Approve converted to Reject on round-trip failure), `v2/src/Tars.Evolution/RoundtripValidation.fs:1` (automatic-failure rule).

Since `classify`/extract creates organic candidates with RollbackExpansion = None, the loop *structurally* caps organic patterns at Implementation unless a rollback is synthesized. Compression-based library learners face the same requirement in reverse — LILO (Grand et al., 2023) treats the ability to expand an abstraction back into its uses as the core of making a learned library trustworthy.

### 7. The staircase is a monotone ratchet: no oscillation is possible, and no demotion path exists

**Claim.** No stalled-then-oscillating promotions exist and none *can* exist: `PromotionLevel.next` only advances, every recurrence PromotionHistory is strictly increasing, and there is no code path that demotes. The promotion-depth distribution is {4 levels: 5 patterns, 1 level: 2, 0 levels: 2}. *(Confidence: high; confirmed by both review passes, including a repo-wide grep for any demotion/retirement path — none exists.)*

**Evidence.** `PromotionLevel.next` (`PromotionTypes.fs:21–26`) maps each level to `Some` higher level or `None` at the top, with no inverse. `persist` sets `CurrentLevel = ProposedLevel` only on Approve and never lowers it; review confirmed `CurrentLevel` is written nowhere else except at record creation. All PromotionHistory arrays in `recurrence.json` are strictly ascending with no repeats or reversals. Two corrections from review: (a) `recurrence.json` holds 9 records, not the 10 stated in the original evidence text; (b) "AverageScore is never re-evaluated downward" is an observed fact of this dataset, not a code invariant — extract's recomputation could in principle lower a score. Neither affects the claim.

**Code anchors.** `v2/src/Tars.Evolution/PromotionTypes.fs:21` (`next`), `v2/src/Tars.Evolution/PromotionPipeline.fs:254` (`persist`), `v2/src/Tars.Evolution/PromotionPipeline.fs:390` (promotion application).

For a self-improving system the absence of demotion is the real hazard: a GrammarRule whose post-promotion SuccessRate later collapses stays authoritative forever. In evolutionary-game terms the system implements selection without extinction — replicator dynamics with a floor (cf. Accinelli & Carrera, 2011) — which precludes the corrective pressure that makes such dynamics stabilizing.

### 8. The four persistence stores are mutually inconsistent

**Claim.** The four stores are not transactionally co-written and disagree on disk: (a) `index.json` OccurrenceCount is exactly 1.25× `recurrence.json` for all 5 GA patterns (25 vs 20, 30 vs 24, 20 vs 16, 15 vs 12, 15 vs 12) and 1.5× for `extract_test` (6 vs 4); (b) `weights.json` Level says all 5 GA patterns are "implementation" while index and recurrence say GrammarRule; (c) `extract_test` is absent from `weights.json`, so its index weight of 0.5 is the hard-coded `Option.defaultValue 0.5` fallback, not a learned value. *(Confidence: high; every ratio and field verified per-pattern by both reviewers.)*

**Evidence.** File modification times tell the story: `index.json` and `weights.json` were rewritten 2026-06-23 22:14, but `recurrence.json` and `lineage.json` still date from 2026-03-14 12:50 — the June regeneration never re-persisted the recurrence and lineage stores. `PromotionIndex.build` reads `r.OccurrenceCount` from the live in-memory records (hence the newer counts in the index), and applies the 0.5 default when a pattern has no weight entry. `WeightedRule.Level` is set once at creation (`WeightedGrammar.fs:126`) and `updateWeight` never touches it, so the GA rows still read "implementation" four promotions later. Organic patterns' counts match 1:1 across stores, isolating the discrepancy to records that continued accumulating occurrences in memory after the last recurrence-store write.

**Code anchors.** `v2/src/Tars.Evolution/PromotionIndex.fs:71` (0.5 default fallback), `v2/src/Tars.Evolution/PromotionIndex.fs:83` (index built from live records), `v2/src/Tars.Evolution/WeightedGrammar.fs:128` (Level frozen at creation), `v2/src/Tars.Evolution/PromotionPipeline.fs:39` (store wiring).

Anyone auditing lineage from disk sees a stale world: the index describes June while the ledger describes March.

### 9. The dataset supports description, not inference

**Claim.** The dataset is too small and too batchy for inferential statistics: 9 patterns and 23 lineage records produced by essentially two scripted seeding events — organic tasks on 2026-03-11 and the GA seed on 2026-03-14 — not a continuously-running live ecosystem. *(Confidence: high; confirmed by both review passes.)*

**Evidence.** `recurrence.json` TaskIds are synthetic (`task_001`–`task_010`, `ga-*` ids, `t1`/`t2`) with FirstSeen/LastSeen clustered on exactly two dates. Staircase velocity confirms the batch character: all 5 GA patterns climbed all 4 rungs in 769 seconds (16:37:11 → 16:50:00) within one 2026-03-14 session, and the 2 organic promotions completed in ~0.09 s — same-transaction, not observed-over-time. `GaPatternSeeder.fs` hard-codes its artifacts with fixed `DateTime(2026,3,14)` timestamps. The Gini and entropy figures of Finding 3 are therefore descriptive point estimates over 8–9 observations with no inferential power; a single added pattern moves them materially (the `extract_test` 0.5 default alone shifts Gini from 0.139 to 0.188 between the two weight vectors).

**Code anchors.** `v2/src/Tars.Evolution/GaPatternSeeder.fs:150` (hard-coded seed artifacts), `v2/src/Tars.Evolution/PromotionPipeline.fs:113` (extraction from task records).

Every conclusion in this document should accordingly be read as a smoke-test of the pipeline's mechanics, not as evidence about an emergent ecosystem.

## Flagged

No claims were flagged as unverifiable. All eight findings were confirmed by two independent adversarial review passes, with the minor evidence-text corrections noted inline in Findings 2, 6, and 7 (rollback coverage among organics, the predicted-vs-demonstrated status of `extract_test`'s blocker, and the recurrence record count of 9 rather than 10).

## Refuted during review

None. No claims were dropped; every claim submitted to adversarial review survived with a "confirmed" verdict from both reviewers.

## Opportunities for TARS

Ranked by expected impact on the self-improvement loop, most impactful first.

1. **Fix the weight–fitness inversion (Finding 4).** This is a concrete, live scoring bug: `PatternSelector` ranks the empirically best patterns (SuccessRate 1.0, grammar-level) last because their weights came from a one-shot softmax over zero logits and are never updated by success. Reconcile the two weight-assignment paths — either run GA-seeded patterns through the same replicator/Bayesian update the organic ones receive, or make `updateWeight` fold SuccessRate (and LevelRank) into Weight so that weight is monotone in observed fitness.

2. **Unblock organic ascent by synthesizing rollback expansions (Finding 6).** The rollback expansion is a hard promotion prerequisite, yet organic extraction creates candidates with RollbackExpansion = None, structurally capping organic patterns at Implementation. Auto-synthesize a rollback expansion at extract time from the pattern's own trace steps. Until at least one organically-discovered pattern climbs past Helper, the "self-hosting loop" claim rests entirely on the GA seed (Finding 2).

3. **Add a demotion/retirement path (Finding 7).** The staircase is a one-way ratchet; a GrammarRule whose post-promotion SuccessRate collapses stays authoritative forever. The Bayesian `updateWeight` already tracks post-promotion SuccessRate — drive a demotion or retirement decision from it so the staircase becomes a feedback controller rather than a ratchet.

4. **Give the Governor teeth (Finding 5).** A 95.7% approval rate with 21 of 23 decisions at confidence 1.0 provides almost no selection pressure, because the subjective criteria default to true and the LLM assessment is always None in the live path. Either wire the LLM assessment into the live pipeline or tighten the deterministic criteria (e.g. require ImprovesPlanning to reflect measured planning uplift rather than AverageScore > 0.6).

5. **Make the four stores describe one snapshot (Finding 8).** Have `refresh()` re-persist recurrence and lineage atomically with the index, and update `WeightedRule.Level` on each promotion, so on-disk audits are not reading a March ledger against a June index. Also replace the silent 0.5 weight fallback with an explicit "unweighted" marker so defaults are distinguishable from learned values.

6. **Instrument the middle rungs and re-measure over time (Findings 1, 9).** Builder and DslClause are pure pass-through states; consider dwell-time gating per rung so intermediate abstractions can be observed and reused before being superseded — or justify collapsing the staircase. Then run the evolve loop over many independent tasks across time and track occupancy, effective number of species (2^entropy), and the weight–fitness correlation as time series; that is where stall, oscillation, and collapse would actually become observable.

## References

- Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B. (2021). *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning.* PLDI 2021. DOI: 10.1145/3453483.3454080.
- Grand, G., et al. (2023). *LILO: Learning Interpretable Libraries by Compressing and Documenting Code.* arXiv:2310.19791.
- Accinelli, E., & Carrera, E. J. S. (2010). *Evolutionarily Stable Strategies and Replicator Dynamics in Asymmetric Two-Population Games.* In Momentum and Stochastic Momentum Methods (Springer). DOI: 10.1007/978-3-642-11456-4_3.
- Accinelli, E., & Carrera, E. J. S. (2011). *Replicator Dynamics and Evolutionary Stable Strategies in Heterogeneous Games.* University of Leicester, Department of Economics, Working Paper 11-54.

### Primary data sources

- `~/.tars/promotion/index.json` (9 patterns; rewritten 2026-06-23 22:14)
- `~/.tars/promotion/weights.json` (8 weighted rules; rewritten 2026-06-23 22:14)
- `~/.tars/promotion/recurrence.json` (9 recurrence records; last written 2026-03-14 12:50)
- `~/.tars/promotion/lineage.json` (23 governance decisions; last written 2026-03-14 12:50)
- Pipeline code: `v2/src/Tars.Evolution/` (`PromotionTypes.fs`, `PromotionPipeline.fs`, `PromotionIndex.fs`, `WeightedGrammar.fs`, `RoundtripValidation.fs`, `GaPatternSeeder.fs`), `v2/src/Tars.Cortex/PatternSelector.fs`, `v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs`
