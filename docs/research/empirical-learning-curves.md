---
title: "Outcome Learning Curves: An Empirical Audit of the TARS Learning Signal"
date: 2026-07-21
track: empirical
unit: outcome-learning-curves
status: verified
---

# Outcome Learning Curves: An Empirical Audit of the TARS Learning Signal

## Abstract

This unit audits the empirical learning signal available to the TARS self-improvement loop, using the complete telemetry under `~/.tars`: 231 pattern-outcome records (`pattern_outcomes.json`), 7 stored benchmark runs (`benchmark_results/`), and the current insight snapshot (`insights/latest.json`). The findings are largely negative, in the useful sense: most of what the system currently "knows" about its own performance is either confounded, synthetic, or statistically indistinguishable from noise. The headline success rate is inflated by trivial smoke-test records — the corrected figure is 217/231 (93.9%) overall but only 35/49 (71.4%) on non-trivial goals. Per-pattern success rates (ReAct 0/5 vs ChainOfThought 195/195) are causally worthless because pattern assignment is confounded with goal difficulty. The remembered "search capability at 60% failure" gap reproduces exactly on March-era data, but the underlying failure records bear the signature of seeded demo data, the rate dilutes to 50% under a benchmark-id substring leak, and the gap is not statistically distinguishable from the 25% general baseline (Fisher exact p = 0.149). No learning curve is detectable across months (p = 0.538), and none could be attributed to evolution cycles even in principle, because neither `PatternOutcome` nor `BenchmarkRunSummary` records a cycle identifier or a real model id. All benchmark runs come from a single day, the Expert tier was never attempted, and most benchmark failures are compile failures, so the binary reward mostly measures F# compilability. The one statistically solid signal is the within-run difficulty gradient (Advanced 1/4 vs Beginner+Intermediate 9/10, p = 0.041). Separately, a live data-corruption bug in `PatternOutcomeStore` wraps every Custom pattern kind in one additional quoting layer per append, progressively destroying the category labels the learning loop depends on. Every claim below survived independent adversarial review; two candidate claims were refuted on their headline counts and are recorded honestly in the final section.

## Background

TARS's central architectural claim is that it is a *self-improving* agent system: evolve tasks generate pattern outcomes, outcomes feed the promotion pipeline, the promotion index feeds `PatternSelector`, and improved selection closes the loop. For that claim to be testable, the telemetry must support at least three inferences: (1) which reasoning patterns work on which goals, (2) whether performance improves over evolution cycles, and (3) which capability gaps are real rather than sampling artifacts. Project memory carries one such inference already — a meta-cognitive analysis that found a "search capability gap (60% failure rate)" — and remediation work was directed by it.

The data available for this audit is modest and lopsided. The outcome store holds 231 records spanning 2026-03 to 2026-06, of which 182 are two endlessly repeated trivial goals ("Summarize a document" and "Temporary plan", 91 each, all successes). Benchmark telemetry consists of 7 runs, all recorded within a two-hour window on 2026-06-21, drawn from a 19-problem curated F# bank (`v2/src/Tars.Evolution/ProblemBank.fs`) plus 5 GA music-theory problems (`v2/src/Tars.Evolution/GaProblemBank.fs`). All statistics below use exact methods — Clopper–Pearson binomial intervals and Fisher exact tests — since every cell count is small; no normal approximations are involved. Each claim was checked by two independent adversarial reviewers who recomputed the statistics from the raw files and verified every code anchor; where a reviewer found a discrepancy in a supporting detail, the corrected figure is stated here.

## Findings

### 1. Per-pattern success rates exist but are causally worthless (confounded assignment)

The outcome store yields per-pattern success rates of ChainOfThought 195/195 (100%), GraphOfThoughts 2/2, WorkflowOfThought 1/1, ReAct 0/5 (0%), TreeOfThoughts 0/3 (0%), and benchmark-derived Custom kinds 19/25 (76%). The ReAct-vs-CoT contrast is enormous by naive test (Fisher exact p = 3.9 × 10⁻¹⁰), yet the comparison supports no causal conclusion about pattern quality: pattern assignment is completely confounded with goal difficulty. All 5 ReAct records are failures on hard search/debug goals ("search for security vulnerabilities in auth module", "debug the failing deployment pipeline", dated 2026-03-10/11), and all 3 TreeOfThoughts failures are explore/brainstorm goals, while 181 of ChainOfThought's 195 records are the two trivial goals (review correction: the count is 181, not 182 — the remaining trivial record is the lone WorkflowOfThought success, "Summarize a document", 2026-03-14, which means WoT's 1/1 is itself a trivial-goal success). Moreover, the sample sizes cannot support strong conclusions even setting confounding aside: the exact binomial 95% CI for ReAct's 0/5 is [0%, 52.2%], so the data cannot rule out ReAct succeeding half the time.

Code anchors: selection heuristic at `v2/src/Tars.Cortex/PatternSelector.fs:126`; `parseKind` at `v2/src/Tars.Cortex/PatternSelector.fs:39`.

Source: Clopper & Pearson (1934) for the exact interval.

### 2. The remembered "search 60% failure" gap reproduces exactly — and dissolves under scrutiny

Applying `GapDetection.extractDomainTags` (keyword pair at `v2/src/Tars.Core/MetaCognition/GapDetection.fs:28`, substring `Contains` matching at line 45) to records timestamped 2026-03 yields the "search" domain at exactly 3 failures out of 5 records — 60%, matching the figure preserved in project memory, whose formatting (`"%.0f%% failure rate"`) is produced at `GapDetection.fs:149`. The five March records are two successes ("search for and fix code smells in the codebase", twice) and three failures ("search for security vulnerabilities in auth module", "search codebase for deprecated API usage", "search for performance bottlenecks in the API").

Over all current data, however, the rate is 3/6 = 50%, because the June benchmark success `inter-binary-search` leaks the substring "search" into the domain tag: `BenchmarkRunner.fs` sets `Goal = attempt.ProblemId` (`v2/src/Tars.Evolution/BenchmarkRunner.fs:396`), and the substring matcher does the rest. The historical gap figure is thus an artifact of both a particular time window and a fragile tagging mechanism.

### 3. The search gap is not statistically distinguishable from the baseline

The 60% figure carries almost no evidential weight. The exact binomial 95% CI for 3/5 is [14.7%, 94.7%]. The "general"-domain failure baseline is 7/28 = 25% (CI [10.7%, 44.9%]), and a two-sided Fisher exact test of search (3 fail / 2 success) against general (7 fail / 21 success) gives p = 0.149. The gap detector's own confidence formula, `min(total/10, 1) × rate` (`v2/src/Tars.Core/MetaCognition/GapDetection.fs:155`), yields only 0.30 at n = 5. Structurally, `detectGaps` admits any domain with failure rate above threshold and total ≥ 2 (`GapDetection.fs:129`) — two failures in two records suffice to declare a capability gap — with no multiplicity control and no minimum-power requirement. Detecting a genuine 25% → 60% shift at α = 0.05 with power 0.8 would require roughly 25–30 records per domain; the store has 5.

Sources: Fisher (1935) for the exact 2×2 test; Clopper & Pearson (1934) for the intervals.

### 4. The failure records behind the gap appear to be seeded, not live

The records that generated the search gap look synthetic. All 10 ReAct/TreeOfThoughts/GraphOfThoughts records on 2026-03-10/11 carry exact-on-the-hour timestamps (10:00:00Z, 11:00:00Z, …, 16:00:00Z), whereas every live record in the file has 7-digit sub-second precision (e.g. 2026-06-21T16:05:40.6666823Z). Review strengthened this finding: exactly 12 of the 231 records lack fractional seconds, all in this same on-the-hour March-10/11 block (the extra two are ChainOfThought records with the same signature), and 3 of the 5 March search-tagged records are in the seeded set. The confidence on the *interpretation* (seeded demo data rather than an oddly quantized live path) is medium, but the timestamp pattern itself is unambiguous. The practical upshot is uncomfortable: the meta-cognitive "search gap" that drove remediation work was derived largely from synthetic records.

Code anchor: outcome timestamping at `v2/src/Tars.Cortex/PatternSelector.fs:22`.

### 5. No learning curve is detectable — and none could be attributed even in principle

Excluding the two trivial goals and bucketing by month, the non-trivial success rate moved from 16/24 (66.7%) in 2026-03 to 19/25 (76.0%) in 2026-06, with zero records in April and May. The Fisher exact two-sided p is 0.538: no detectable improvement. Worse, even if improvement existed it could not be attributed to evolution cycles from this data, because the `PatternOutcome` schema records only `PatternKind`/`Goal`/`Success`/`DurationMs`/`Timestamp` (`v2/src/Tars.Cortex/PatternSelector.fs:17`–22) — no cycle id, no model id, no selector-strategy id. The `tars evolve --loop N` machinery (`v2/src/Tars.Interface.Cli/Commands/Evolve.fs:33`; reviewers note the loop-count option itself sits at line 23) runs cycles whose central claim — self-improvement — is unmeasurable from its own telemetry.

### 6. No benchmark trajectory exists: one day of data, Expert tier untouched

All 7 saved benchmark runs date from a single day, 2026-06-21, between 16:06 and 17:59 (`run_20260621_{160603,171233,171858,173536,175131,175744,175912}.json` plus `latest.json`, a duplicate of the last run). The Expert tier — `exp-result-ce`, `exp-active-pattern`, `exp-graph-cycle`, `exp-memoize` (`v2/src/Tars.Evolution/ProblemBank.fs:117`) — was never attempted in any stored run, and the only full-suite run (`run_171233`, passRate 0.714) covered 14 of the 19 curated problems (`ProblemBank.fs:18` onward; the 5 GA problems at `v2/src/Tars.Evolution/GaProblemBank.fs:29`). Pooled across all runs, the pass rate is 17/23 attempts = 73.9% (exact 95% CI [51.6%, 89.8%]). The four same-day GA-suite re-runs go pass/pass → pass → fail/fail → pass/pass — pure run-to-run noise, not a curve. Results are persisted at `v2/src/Tars.Evolution/BenchmarkRunner.fs:402`.

### 7. The difficulty gradient is real — the one solid learning signal

Within the one full run (`run_20260621_171233.json`), pass rate falls with difficulty: Beginner 4/5 (80%), Intermediate 5/5 (100%), Advanced 1/4 (25%). The failures are `basic-palindrome`, `adv-balanced-parens`, `adv-eval-rpn`, and `adv-matrix-multiply`. Advanced vs Beginner+Intermediate is marginally significant (Fisher exact two-sided p = 0.041), though the Advanced CI is wide ([0.6%, 80.6%] on n = 4), and the gradient is not monotone (Beginner 80% < Intermediate 100%) — the tested contrast is precisely Advanced against the rest. Confidence is medium given the sample size, but this is the strongest genuine signal in the telemetry, and the natural target for `CurriculumPlanner` (`v2/src/Tars.Evolution/CurriculumTypes.fs:15`; attempt schema at `v2/src/Tars.Evolution/BenchmarkTypes.fs:46`).

Source: Fisher (1935).

### 8. A live data-corruption bug degrades Custom pattern-kind labels on every append

`PatternOutcomeStore` corrupts its own data. The mechanism has three parts: `toDto` serializes `PatternKind` via `sprintf "%A"` (`v2/src/Tars.Cortex/PatternSelector.fs:33`), which pretty-prints `Custom` cases as `Custom\n  "..."`; `parseKind` lowercases the entire string and rewraps anything unrecognized as `Custom` of the raw string (`PatternSelector.fs:47`); and `record()` round-trips the *entire file* through `fromDto`/`toDto` on every append (`PatternSelector.fs:82`). Each append therefore adds one layer of `custom "..."` wrapping to every existing Custom record, destroys case information via `ToLowerInvariant`, and makes appends O(n²) in file size. The data confirms the mechanism exactly: the oldest Custom record (index 142, goal `ga-pc-interval`, 2026-06-21T16:05:40Z) carries a 988-character patternKind with 88–89 nested layers (the original estimate of ~85 was slightly low; review found the nesting count equals exactly the number of subsequent appends), and the 25 Custom records decrease monotonically in length from 988 to 724 characters — newer records have been round-tripped fewer times. The innermost token, e.g. `benchmark:musictheory` (written as `benchmark:{Category}` at `v2/src/Tars.Evolution/BenchmarkRunner.fs:395`), is still recoverable from all 25 records, so migration is feasible.

### 9. Model provenance is lost: `ModelUsed` is hardcoded to "default"

Benchmark results cannot be conditioned on model: `runSuiteFromProblems` hardcodes `ModelUsed = "default"` (`v2/src/Tars.Evolution/BenchmarkRunner.fs:370`), and all 7 stored runs show `modelUsed: "default"`. Combined with the missing cycle id (Finding 5), the two covariates most needed for any learning-curve analysis — which model, which cycle — are both absent from the telemetry. A secondary provenance break: the perf problem id changed between runs (`perf-sum-squares` in `run_171858` vs `inter-perf-sum-squares` in `run_173536` and in the current `v2/src/Tars.Evolution/ProblemBank.fs:175`), so longitudinal per-problem joins are broken for that problem.

### 10. Orphaned state and a vacuous current insight snapshot

Two smaller integrity issues round out the audit. First, `~/.tars/pattern_outcomes_new.json` is a 0-byte orphan created 2026-03-12, referenced nowhere in the repository (the only live store path is `pattern_outcomes.json`, `v2/src/Tars.Cortex/PatternSelector.fs:65`). Second, the live insight snapshot (`~/.tars/insights/latest.json`, 2026-06-24) reports `gaps: []` — the historical search gap no longer reproduces in current meta-cognition output — while ranking patterns CoT 0.503 > WoT 0.417 > GoT 0.408 > ReAct 0.388 > ToT 0.304 with the *identical* 15-token `goalKeywords` list attached to every pattern. The shared keyword list makes pattern–goal affinity unlearnable by any downstream consumer; the exporter (`v2/src/Tars.Evolution/InsightExporter.fs:1`) computes no per-pattern attribution.

## Flagged (unverifiable)

No claims were flagged. Every claim submitted to adversarial review was either confirmed (reported above, with reviewer corrections noted inline) or refuted (reported below).

## Refuted during review

Two candidate claims failed verification on their headline numbers and are recorded here for the honesty of the record. Their *qualitative* substance survived in both cases; the corrected figures are used throughout this document.

1. **"Headline success rate 222/231 = 96.1% is inflated by 182 trivial records."** The inflation finding is real — 91× "Summarize a document" and 91× "Temporary plan" (182 records, all successes) do exist, the non-trivial rate is exactly 35/49 = 71.4% (CI [56.7%, 83.4%]), and the per-month CIs reproduce to the decimal — but the quoted headline statistic does not exist in the data. The file contains 217 successes out of 231 (93.9%), not 222/231, and the claim's own decomposition summed to 217, making it internally inconsistent. The corrected headline, 217/231 = 93.9% inflated down to 71.4% non-trivial, is what this document reports.

2. **"6 of the 7 failed benchmark attempts failed at compilation."** Both counts were wrong: there are 6 failed attempts across all stored runs (23 attempts, 17 validated), not 7, and 5 of them (not 6) have `compiled = false` — `basic-palindrome`, `adv-balanced-parens`, `adv-eval-rpn`, `adv-matrix-multiply` (run_171233) and `ga-transpose` (run_175744). The claim's own evidence enumerated exactly these 5, contradicting its headline. The corrected statement — 5 of 6 failures are compile failures, with `ga-pc-interval` in run_175744 the sole compiled-but-validation-failed attempt, and `compileRate == passRate` in every run except run_175744 — still supports the qualitative conclusion that the benchmark's binary reward measures F# compilability of LLM output far more than algorithmic correctness.

## Opportunities for TARS

Ranked by expected value to the learning loop, most valuable first.

1. **Add `CycleId`, `ModelId`, and `SelectorStrategy` to `PatternOutcome`, and thread the real provider/model id from `LlmFactory` into `BenchmarkRunSummary`** (Findings 5, 9). This is the prerequisite for everything else: until these covariates exist, `tars evolve --loop N` produces no per-cycle series and the system's core self-improvement claim is unmeasurable from its own telemetry.

2. **Fix the `PatternOutcomeStore` corruption bug** (Finding 8). Serialize `PatternKind` as a stable tag+payload (e.g. `Custom:benchmark:Algorithms`) with an exact inverse parser, switch to append-only JSONL to eliminate the O(n²) full-file round-trip, and write a one-off migration stripping the nested wrapping — the innermost `benchmark:<category>` token is recoverable from all 25 corrupted records.

3. **Point the curriculum at the difficulty gradient, not the search domain** (Findings 3, 7). The Advanced-tier compile failures are the strongest genuine signal available; `CurriculumPlanner` should target them. Schedule the full 19-problem suite (including the never-attempted Expert tier) per evolution cycle, persist the cycle number in `BenchmarkRunSummary`, and treat problem ids as frozen once results referencing them exist. Single-day GA re-runs already show variance that per-cycle aggregation must average over.

4. **Statistically gate `CapabilityGap` emission** (Finding 3). Gate on an exact binomial test against the pooled baseline (e.g. require the lower CI bound to exceed the baseline rate), or at minimum raise the sample floor from 2 to ~10; detecting a 25% → 60% shift at α = 0.05 and power 0.8 needs roughly 25–30 records per domain. Replace substring `Contains` tagging with word-boundary or structured tagging in `extractDomainTags`, and exclude benchmark problem-id goals from domain extraction (the `inter-binary-search` leak).

5. **Purge or flag the seeded March records, delete the orphaned store file, and fix per-pattern keyword attribution in `InsightExporter`** (Findings 4, 10). The 12 on-the-hour synthetic records should not be allowed to drive meta-cognitive conclusions again; the shared `goalKeywords` list currently makes pattern–goal affinity vacuous downstream.

6. **Randomize or interleave pattern assignment on comparable goals** (Finding 1). A bandit-style assignment with the counterfactual candidate set recorded at selection time is the minimum design under which per-pattern success rates become causally interpretable.

## References

- Clopper, C. J.; Pearson, E. S. (1934). "The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial." *Biometrika* 26(4): 404–413. DOI 10.1093/biomet/26.4.404.
- Fisher, R. A. (1935). *The Design of Experiments.* Edinburgh: Oliver & Boyd. (Exact test for 2×2 contingency tables.)

### Data and code examined

- `~/.tars/pattern_outcomes.json` (231 records, 2026-03 to 2026-06); `~/.tars/pattern_outcomes_new.json` (0-byte orphan); `~/.tars/benchmark_results/` (7 runs + `latest.json`, all 2026-06-21); `~/.tars/insights/latest.json` (2026-06-24).
- `v2/src/Tars.Cortex/PatternSelector.fs`; `v2/src/Tars.Core/MetaCognition/GapDetection.fs`; `v2/src/Tars.Evolution/BenchmarkRunner.fs`, `ProblemBank.fs`, `GaProblemBank.fs`, `BenchmarkTypes.fs`, `CurriculumTypes.fs`, `InsightExporter.fs`; `v2/src/Tars.Interface.Cli/Commands/Evolve.fs`.

All statistics are exact (Clopper–Pearson intervals; Fisher exact tests); each finding was independently reproduced by two adversarial reviewers against the raw telemetry and code anchors.
