---
title: "Open-Ended Curriculum: From a Closed 24-Problem Bank to a Self-Generating Task Frontier"
date: 2026-07-21
track: frontier
unit: open-ended-curriculum
status: verified
---

# Open-Ended Curriculum: From a Closed 24-Problem Bank to a Self-Generating Task Frontier

## Abstract

TARS's evolve/benchmark curriculum is a closed, hand-curated set of 24 problems (19 in `ProblemBank`, 5 in `GaProblemBank`) selected by a deterministic head-of-list policy under a monotone difficulty ratchet. This is precisely the configuration the open-endedness literature (POET/Enhanced-POET, GoalGAN, AI-GAs) predicts will saturate: once the bank is solved, the reward gradient vanishes and the system's ANNECS-equivalent progress measure — accumulated novel challenges created and solved — is bounded above by 24. An LLM task generator exists (`CurriculumPlanner.generateTasksWithLlm`) but emits tasks with free-text, non-executable validation and no solvability filter, so it cannot safely feed the deterministic dotnet-fsi benchmark pipeline. Adversarial review confirmed nine of ten claims about the current architecture; the tenth — that live telemetry already demonstrates the saturation regime — was refuted on statistical grounds and is recorded below. The central positive finding is that nearly all mechanical infrastructure for a minimal self-generating curriculum already exists in-repo: NDJSON problem ingestion, a compile-and-validate PASS/FAIL harness accepting arbitrary problem lists, FsCheck property checks against overfitting, and constrained JSON decoding. The proposed minimal design is ACCEL-style LLM mutation of existing `BenchmarkProblem`s (emitting description, signature, executable `ValidationCode`, and `ReferenceSolution`), a POET-style minimal-criterion filter (the reference solution must compile and PASS; the current agent's empirical pass rate must fall in a GoalGAN GOID band such as [0.1, 0.9]), a MAP-Elites archive keyed by Category × measured-difficulty band, and a curriculum-health metric combining the fraction of active problems in the success band with an ANNECS-style counter of novel-generated-then-solved problems per cycle.

## Background

The theoretical frame for this unit is the open-endedness and unsupervised-environment-design literature. POET (Wang, Lehman, Clune, Stanley 2019) established that coupled agent–environment co-evolution requires environments admitted under a *minimal criterion* — neither trivially easy nor hopelessly hard for the current agent population. Enhanced POET (Wang et al., ICML 2020) introduced ANNECS, the accumulated number of novel environments created *and* solved, as the standard progress measure for open-ended systems: a system whose task set is static has an ANNECS that saturates by construction. GoalGAN (Florensa, Held, Geng, Abbeel, ICML 2018) formalized *Goals of Intermediate Difficulty* as the two-sided band {g : Rmin ≤ R_g(π) ≤ Rmax} with Rmin = 0.1, Rmax = 0.9 — tasks whose current success probability carries maximal learning signal. ACCEL (Parker-Holder et al., ICML 2022) showed that *editing* previously high-regret tasks keeps the generated frontier near agent capability better than from-scratch generation. OMNI (Zhang, Lehman, Stanley, Clune 2023) added the complementary constraint that learnability filtering alone is insufficient — "countless learnable yet uninteresting tasks remain" — motivating an interestingness/novelty gate, extended to code-defined tasks by OMNI-EPIC (Faldor, Zhang, Cully, Clune, ICLR 2025). Voyager (Wang et al. 2023) is the closest published analogue of TARS's LLM task generator: its automatic curriculum conditions GPT-4 on the agent's state and its completed *and* failed task lists to keep proposals at the capability frontier. Finally, AI-GAs (Clune 2019) names environment generation as the third pillar of self-improving AI systems.

Against this frame, the empirical question is where TARS's evolve loop sits today. The answer, verified against source: a fully static task set, a selection policy with no exploration pressure, a promotion policy that cannot express a success band, and an LLM generator disconnected from both the execution harness and the agent's performance history. The gap between the current system and a minimal open-ended one is, however, unusually narrow — the missing pieces are glue, not infrastructure.

## Findings

All findings below survived two independent adversarial verification passes against the repository source and, where applicable, the cited papers.

### 1. The benchmark curriculum is a closed set of exactly 24 problems (`static-bank-24`)

**Claim.** The TARS benchmark curriculum is a closed set of exactly 24 problems — 19 from `ProblemBank` (5 basic, 5 intermediate, 4 advanced, 4 expert, 1 performance) and 5 from `GaProblemBank` — and no code path in the evolve CLI can add a new `BenchmarkProblem` at runtime.

**Evidence.** `ProblemBank.all()` concatenates five static lists (`basicProblems @ intermediateProblems @ advancedProblems @ expertProblems @ perfProblems`); `GaProblemBank` defines 5 hardcoded music-theory problems; the evolve command selects its bench source exclusively from these two static banks (`'ga'|'music' -> GaProblemBank.all()`, `'all' -> concat`, `_ -> ProblemBank.all()`). Neither bank exposes any insert or append API — only `all`/`byDifficulty`/`byCategory`/`tryFind`. Review confirmed the count by direct enumeration of `mkProblem` calls, and confirmed that `ProblemIngestor` loads a *different* type (`Problem`, without `ValidationCode`) and therefore cannot add `BenchmarkProblem`s.

**Code anchors.** `v2/src/Tars.Evolution/ProblemBank.fs:183-184`, `v2/src/Tars.Evolution/GaProblemBank.fs:28-87`, `v2/src/Tars.Evolution/GaProblemBank.fs:106-109`, `v2/src/Tars.Interface.Cli/Commands/Evolve.fs:763-767`.

### 2. The curriculum loop terminates instead of generating; ANNECS is bounded by bank size (`curriculum-terminates-not-open-ended`)

**Claim.** The retroaction curriculum loop terminates with an error when the static problem list is exhausted rather than generating new challenges. TARS's ANNECS-equivalent — accumulated novel challenges created and solved, the open-endedness progress measure from Enhanced POET — is therefore bounded above by the bank size instead of growing without bound.

**Evidence.** `CurriculumManager.getNextProblem` filters only `CompletedProblems` and returns `None` when nothing remains; `RetroactionLoop.runCycle` then returns `Error "No more problems available in curriculum"` — the exact string appears in source. No generation fallback exists anywhere in the loop. Enhanced POET introduced ANNECS precisely because open-ended systems must keep creating novel-and-solvable challenges; with a static set the counter saturates by construction. Both the code path and the ANNECS characterization were independently confirmed (the reviewers verified that ANNECS counts environments that pass the minimal criterion against all prior agents and are eventually solved).

**Code anchors.** `v2/src/Tars.Evolution/CurriculumManager.fs:26-46`, `v2/src/Tars.Evolution/RetroactionLoop.fs:454-456`.

**Sources.** Wang et al. 2020 (Enhanced POET, arXiv:2003.08536); Wang et al. 2019 (POET, arXiv:1901.01753).

### 3. Difficulty promotion is a monotone fixed-threshold ratchet, not band targeting (`monotone-ratchet-promotion`)

**Claim.** `CurriculumManager`'s difficulty policy is a monotone non-decreasing ratchet with hardcoded thresholds: promotion requires `CompletedProblems.Count > 5` and `MasteryScore > 0.8`; mastery moves +0.1 per success and −0.05 per failure regardless of problem difficulty; and difficulty never demotes. This is a fixed-threshold curriculum, not the success-band targeting the automatic-curriculum literature recommends.

**Evidence.** `checkPromotion` maps only upward (Beginner → Intermediate → Advanced → Expert → Expert; the sole other arm, Unascertained → Beginner, is initialization, not demotion). `recordSuccess` adds a flat 0.1 and `recordFailure` subtracts a flat 0.05. GoalGAN instead defines Goals of Intermediate Difficulty as the two-sided band {g : Rmin ≤ R_g(π) ≤ Rmax} with Rmin = 0.1, Rmax = 0.9 — sampling tasks whose current success probability lies in the band. A monotone ratchet with one-sided thresholds structurally cannot express this criterion. Both the code and the GoalGAN formalism were verified.

**Code anchors.** `v2/src/Tars.Evolution/CurriculumManager.fs:13-23`, `v2/src/Tars.Evolution/CurriculumManager.fs:49-59`, `v2/src/Tars.Evolution/CurriculumManager.fs:62-68`.

**Sources.** Florensa et al., ICML 2018 (GoalGAN, arXiv:1705.06366).

### 4. Deterministic head-of-list selection creates a livelock mode; failure counts are recorded but never read (`deterministic-head-selection-livelock`)

**Claim.** `getNextProblem` filters out only completed problems and then takes the deterministic head of the list, so a problem that repeatedly fails is re-selected every cycle with no exploration, novelty pressure, or interleaving. `FailedProblems` counts are recorded but never consulted by the selection policy.

**Evidence.** The available list excludes only `state.CompletedProblems` (lines 31-33); selection is `match relevant with head :: _ -> Some head` (line 39) with the same ordering on every call. A repo-wide search for `FailedProblems` finds exactly four occurrences — the type field, its initialization, and two writes inside `recordFailure` — and no reads in `getNextProblem`, `checkPromotion`, or any other selection code. The novelty-search literature (Lehman & Stanley 2011) shows that purely objective-driven, non-diverse selection is vulnerable to deceptive dead-ends that diversity pressure escapes. (One nuance from review: selection also filters by difficulty tier, which the claim's summary elides, but the quoted match expression is exact and the livelock mode is real — a failing problem is never added to `CompletedProblems` and is therefore re-selected indefinitely.)

**Code anchors.** `v2/src/Tars.Evolution/CurriculumManager.fs:26-46`, `v2/src/Tars.Evolution/CurriculumManager.fs:62-68`.

**Sources.** Lehman & Stanley, Evolutionary Computation 19(2):189-223, 2011.

### 5. LLM-generated tasks carry no executable validation and no solvability filter (`llm-tasks-lack-solvability-filter`)

**Claim.** `CurriculumPlanner`'s LLM-generated tasks carry only free-text validation (`ValidationCriteria: string option`), no executable `ValidationCode`, no `ReferenceSolution`, and no solvability check before entering the curriculum. Generated tasks may therefore be unsolvable, ambiguous, or vacuously easy — the exact failure modes POET's minimal criterion and Voyager's self-verification are designed to filter.

**Evidence.** `toProblem` sets `ReferenceSolution = None` and passes through the LLM's free-text `validation` string; template tasks receive validation of the form `"Task in domain '%s' completes without errors"` (line 36); `mergeCurriculum` prepends these unverified tasks *ahead of* curated problems (line 158). By contrast, the benchmark path requires executable F# `ValidationCode` printing PASS/FAIL, run under `dotnet fsi` (`BenchmarkTypes.fs:28-29`). No compile or solvability check exists anywhere in the module — confirmed by full reads of the file in both review passes.

**Code anchors.** `v2/src/Tars.Evolution/CurriculumPlanner.fs:31-37`, `v2/src/Tars.Evolution/CurriculumPlanner.fs:119-122`, `v2/src/Tars.Evolution/CurriculumPlanner.fs:137-158`, `v2/src/Tars.Evolution/BenchmarkTypes.fs:26-43`.

**Sources.** Wang et al. 2019 (POET, arXiv:1901.01753); Wang et al. 2023 (Voyager, arXiv:2305.16291).

### 6. The generator ignores the agent's performance history (`generator-ignores-history`)

**Claim.** `generateTasksWithLlm` conditions its curriculum prompt only on capability gaps (domain, failure rate, sample size, remedy) and not on the agent's completed/failed problem history or current mastery state. Voyager — the closest published analogue — explicitly conditions GPT-4 on agent state, inventory, and completed *and* failed task lists to keep proposals at the frontier of capability.

**Evidence.** The prompt (`CurriculumPlanner.fs:56-66`) interpolates only `gapDescriptions` built from `CapabilityGap` fields (lines 46-52). `CurriculumState` — which carries `CompletedProblems`, `FailedProblems`, and `MasteryScore` (`CurriculumTypes.fs:41-45`) — is never referenced anywhere in the module, confirmed by full reads in both review passes. Difficulty targeting in the current design is therefore grounded in gap statistics alone rather than observed per-problem performance.

**Code anchors.** `v2/src/Tars.Evolution/CurriculumPlanner.fs:40-66`, `v2/src/Tars.Evolution/CurriculumTypes.fs:41-45`.

**Sources.** Wang et al. 2023 (Voyager, arXiv:2305.16291).

### 7. Every mechanical component of a self-generating curriculum already exists in-repo (`infra-for-self-generation-exists`)

**Claim.** The mechanical substrate for a minimal self-generating curriculum is already present: NDJSON problem ingestion from a directory (`ProblemIngestor.loadFromDirectory`), a deterministic compile-and-validate harness accepting arbitrary problem lists (`BenchmarkRunner.runSuiteFromProblems`), FsCheck property hooks that catch overfit solutions (`BenchmarkProblem.Properties`), persistent curriculum state (`.tars/curriculum.json`), and constrained JSON decoding for schema-valid LLM output. Only the generator–filter–archive glue is missing.

**Evidence.** `loadFromDirectory` parses `*.ndjson` into `Problem` records (with the caveat, verified accurate, that this DTO lacks executable `ValidationCode` fields and needs a `BenchmarkProblem`-shaped extension); the doc comment at `BenchmarkRunner.fs:331-333` explicitly states callers can pass any problem list to `runSuiteFromProblems`; `GaProblemBank.fs:95-103` contains a working FsCheck property harness (`transposeProperties`); `WotCommand` persists `CurriculumState` to `.tars/curriculum.json`; and `ConstrainedDecoding.fs` builds `ResponseFormat.Constrained` requests from EBNF/JSON-schema/regex grammars. All five components were independently verified. This aligns with pillar three of AI-GAs: generating effective learning environments.

**Code anchors.** `v2/src/Tars.Evolution/ProblemIngestor.fs:61-115`, `v2/src/Tars.Evolution/BenchmarkRunner.fs:332`, `v2/src/Tars.Evolution/GaProblemBank.fs:95-103`, `v2/src/Tars.Interface.Cli/Commands/WotCommand.fs:479-499`, `v2/src/Tars.Llm/ConstrainedDecoding.fs`.

**Sources.** Clune 2019 (AI-GAs, arXiv:1905.10985).

### 8. Mutation of existing high-value tasks beats from-scratch generation for frontier calibration (`accel-mutation-beats-scratch-generation`)

**Claim** (medium confidence). The environment-design literature indicates that evolving small edits of existing high-value tasks (ACCEL) yields better-calibrated difficulty frontiers than generating tasks from scratch, because edits of problems the agent partially solves stay near the capability frontier while de-novo generation samples mostly degenerate or off-frontier tasks.

**Evidence.** ACCEL "evolves a curriculum by making small edits to previously high regret levels, thus constantly producing new levels at the frontier of the student agent's capabilities" — a quote review verified verbatim against the paper's abstract — and reports significant gains over from-scratch generative UED baselines such as PAIRED's learned generator. TARS already has the ingredients ACCEL needs: 24 curated seed problems with per-problem attempt history, since `BenchmarkAttempt` records `Compiled` and `Validated` separately (`BenchmarkTypes.fs:46-65`) and `BenchmarkRunner.recordOutcomes` feeds these outcomes to `PatternOutcomeStore` (called at `Evolve.fs:779`). A compile-success-but-validation-failure attempt is a usable regret proxy: the agent almost solves the problem. One caveat carried forward from review: transferring RL environment-editing results to LLM problem generation is an extrapolation; the claim is framed as what the literature indicates, not a guarantee, hence the medium confidence.

**Code anchors.** `v2/src/Tars.Evolution/BenchmarkTypes.fs:46-65`, `v2/src/Tars.Interface.Cli/Commands/Evolve.fs:779`.

**Sources.** Parker-Holder et al., ICML 2022 (ACCEL, arXiv:2203.01302).

### 9. A concrete curriculum-health metric: success-band fraction plus ANNECS counter (`band-targeting-health-metric`)

**Claim.** A literature-backed curriculum-health metric for the evolve loop is success-rate band targeting — the fraction of active problems whose rolling pass rate lies in an intermediate band (GoalGAN's GOID band is [0.1, 0.9]; a tighter operating band such as [0.3, 0.7] targets maximal learning signal) — plus an ANNECS-style cumulative count of generated-then-solved problems. TARS currently computes neither: only per-run aggregate `PassRate`/`CompileRate` exist.

**Evidence.** `BenchmarkRunSummary` exposes only run-level `PassRate` and `CompileRate` (`BenchmarkTypes.fs:75-76`). Review explicitly checked the obvious counterexample: `PatternOutcomeStore` (in `Tars.Cortex/PatternSelector.fs`) stores raw per-attempt outcomes keyed by goal (problem id) but aggregates only per `PatternKind` (`benchmark:{Category}`) — it never computes a per-problem rolling pass rate. `CurriculumState.FailedProblems` is a raw failure count, not a rate. The GoalGAN GOID formalism and Enhanced POET's ANNECS admission criteria (minimal criterion against all prior agents, eventually solved) were both verified. Together the two statistics give a frontier-quality gauge and a plateau alarm (ANNECS derivative near zero).

**Code anchors.** `v2/src/Tars.Evolution/BenchmarkTypes.fs:68-78`.

**Sources.** Florensa et al., ICML 2018 (arXiv:1705.06366); Wang et al., ICML 2020 (arXiv:2003.08536).

### 10. Solvability filtering alone is insufficient — an interestingness/novelty gate is needed (`interestingness-filter-needed`)

**Claim** (medium confidence). A solvability filter alone is insufficient for a self-generating curriculum: OMNI shows that after filtering for learnability, "countless learnable yet uninteresting tasks remain (e.g., minor variations of previously learned tasks)" — a quote verified verbatim from the paper's abstract. An LLM-mutation generator without a novelty check will converge to near-duplicate problems that inflate pass metrics without expanding capability; mutation being the proposed generation operator makes near-duplicates the *default* failure mode.

**Evidence.** OMNI demonstrates that foundation-model "models of interestingness" outperform uniform sampling and learning-progress-only task selection; OMNI-EPIC extends this to tasks expressed as code — directly analogous to TARS `BenchmarkProblem`s, which are code-defined tasks. MAP-Elites provides the standard structural remedy: an archive keyed by behavior descriptors that keeps only the elite per cell. TARS's existing `ProblemCategory` DU (8 cases, `BenchmarkTypes.fs:6-16`) and `ProblemBank.byCategory` (`ProblemBank.fs:190-192`) supply a ready-made first archive dimension.

**Code anchors.** `v2/src/Tars.Evolution/BenchmarkTypes.fs:6-16`, `v2/src/Tars.Evolution/ProblemBank.fs:190-192`.

**Sources.** Zhang et al. 2023 (OMNI, arXiv:2306.01711); Faldor et al., ICLR 2025 (OMNI-EPIC, arXiv:2405.15568); Mouret & Clune 2015 (MAP-Elites, arXiv:1504.04909).

### 11. Integration requires no change to the evolve CLI's command surface (`cli-compatible-integration-path`)

**Claim.** The self-generating curriculum can be integrated into the existing evolve CLI without changing its command surface. The only required touch points are the `benchSource` match in `Evolve.fs` (adding a `generated`/`auto` domain value that appends `GeneratedBank` problems) and a post-benchmark hook next to the existing SelfTrain dataset refresh, where generation and filtering run once per cycle using the cycle's fresh pass-rate data.

**Evidence.** `Evolve.fs` already has a per-cycle post-benchmark section that refreshes the self-train dataset (lines 791-799) — the natural insertion point for "refill frontier if in-band fraction low." The `benchSource` selection is a single match expression on the existing `options.BenchmarkDomain` string (lines 763-767), so a new arm accepts a new value without changing the flag surface. `BenchmarkRunner.runSuiteFromProblems` takes an arbitrary `BenchmarkProblem` list (doc comment at `BenchmarkRunner.fs:331-333`; `runSuite` delegates at line 389), so generated problems need no runner changes. `SelfTrain.fs:24` already concatenates `ProblemBank.all() @ GaProblemBank.all()`, demonstrating the composition pattern. All four touch points were verified.

**Code anchors.** `v2/src/Tars.Interface.Cli/Commands/Evolve.fs:763-767`, `v2/src/Tars.Interface.Cli/Commands/Evolve.fs:791-799`, `v2/src/Tars.Evolution/BenchmarkRunner.fs:389`, `v2/src/Tars.Evolution/SelfTrain.fs:24`.

## Flagged

No claims were flagged as unverifiable in this unit. All ten submitted claims received definitive verdicts (nine confirmed, one refuted).

## Refuted during review

One claim was dropped after adversarial review, and the record matters because its *data* were accurate while its *inference* was not.

**Dropped claim.** "Live telemetry already shows the saturation regime the open-endedness literature predicts for static benchmarks: of the 7 recorded benchmark runs (all 2026-06-21), five have PassRate 1.00, the only broad run (14 problems) scored 0.71, and one 2-problem run scored 0.00 — meaning the binary PASS/FAIL signal is near ceiling on repeated problems and provides a vanishing selection gradient for the promotion pipeline."

**Why refuted.** The telemetry recital was exactly accurate — review verified all 7 run files in `~/.tars/benchmark_results` (five at PassRate 1.00, the 14-problem run at 0.71, one 2-problem run at 0.00), and the Enhanced POET ~20k-iteration plateau detail checked out against the paper. But the conclusion does not follow from the data. The "repeated problems" (ga-pc-interval, ga-transpose) went 1.00 → 0.00 → 1.00 within twenty minutes on the same day — noise, not ceiling — and the only broad run failed 3 of 4 advanced problems (0.71 overall), meaning the PASS/FAIL signal still carries a substantial gradient on the harder tiers. Seven runs from a single day, five of them with n = 1–2 problems, cannot establish a "saturation regime," and the 0.00 repeated-problem run directly contradicts "near ceiling on repeated problems." The structural saturation argument in this document therefore rests on the *architecture* (Findings 1–3: bounded bank, terminating loop, monotone ratchet), which predicts saturation as the asymptotic regime, not on current telemetry, which is too sparse to demonstrate it.

## Opportunities for TARS

Ranked by leverage per unit of implementation effort. Items 1–3 together constitute the minimal open-ended loop and are shippable as a single PR (three new modules plus two small edits to `Evolve.fs`).

1. **Ship the generator–filter–bank triad** (from Findings 5, 7, 11). Three new modules: (a) `ProblemGenerator.fs` — the LLM mutates a sampled elite `BenchmarkProblem` via constrained JSON decoding, using ACCEL-style small edits (change data shape, add a constraint, compose two problems), emitting a full `BenchmarkProblem` record with signature, executable `ValidationCode`, and `ReferenceSolution`; (b) `SolvabilityFilter.fs` — admission requires the reference solution to compile and PASS its own `ValidationCode` under the existing dotnet-fsi harness, then the current agent's pass rate over k = 5 attempts to fall in a GOID band [0.1, 0.9]; (c) `GeneratedBank.fs` — persist accepted problems to `~/.tars/curriculum/generated.ndjson`, loaded via a new `generated`/`auto` arm in the `benchSource` match at `Evolve.fs:763-767`. No runner or command-surface changes needed.

2. **Replace the terminal error with a generation fallback** (from Finding 2). When `getNextProblem` finds the frontier below a threshold, invoke the generator + filter to refill it instead of returning `Error "No more problems available in curriculum"`, and log an ANNECS-style counter (novel problems generated and later solved) as the loop's open-endedness metric.

3. **Add `CurriculumHealth` to cycle output** (from Finding 9). `{ InBandFraction; AnnecsCount; RetiredCount }` computed from per-problem rolling pass rates (last k attempts each), printed beside `PassRate` in `tars evolve` cycle output; trigger generation when `InBandFraction < 0.5` or the active frontier shrinks below a floor.

4. **Fix the selection livelock (~10 lines)** (from Finding 4). Deprioritize problems with `FailedProblems` count ≥ k (exponential backoff by failure count) and sample among eligible problems instead of taking the deterministic head — cheap diversity pressure with no new modules.

5. **Replace the mastery ratchet with band targeting** (from Finding 3). Track per-difficulty empirical success rates and select the tier whose rolling success rate falls inside [0.3, 0.7]; allow demotion when the band is undershot. This subsumes the `MasteryScore` heuristics.

6. **Seed mutation from the regret proxy** (from Finding 8). Prefer as mutation seeds those problems whose recent attempts show compile-success-but-validation-failure (`Compiled = true`, `Validated = false` in `BenchmarkAttempt`) — the agent almost solves them, so edits stay near the frontier.

7. **Structure the generated bank as a MAP-Elites archive** (from Finding 10). Key cells by `ProblemCategory` × measured-difficulty band; admit a new problem only if its cell is empty or it displaces the incumbent (e.g., higher discrimination between model checkpoints), with an LLM judge rejecting mutations that are trivial paraphrases of an existing problem in the same cell.

8. **Thread performance history into the generation prompt** (from Finding 6). Pass `CurriculumState` plus the last N `BenchmarkAttempt` outcomes into `generateTasksWithLlm` so difficulty targeting is grounded in observed performance, following Voyager's completed/failed-task conditioning.

## References

- Clune, J. (2019). *AI-GAs: AI-generating algorithms, an alternate paradigm for producing general artificial intelligence.* arXiv:1905.10985.
- Faldor, M., Zhang, J., Cully, A., & Clune, J. (2025). *OMNI-EPIC: Open-endedness via Models of human Notions of Interestingness with Environments Programmed in Code.* ICLR 2025. arXiv:2405.15568.
- Florensa, C., Held, D., Geng, X., & Abbeel, P. (2018). *Automatic Goal Generation for Reinforcement Learning Agents.* ICML 2018. arXiv:1705.06366.
- Lehman, J., & Stanley, K. O. (2011). *Abandoning Objectives: Evolution Through the Search for Novelty Alone.* Evolutionary Computation, 19(2), 189–223.
- Mouret, J.-B., & Clune, J. (2015). *Illuminating search spaces by mapping elites.* arXiv:1504.04909.
- Parker-Holder, J., Jiang, M., Dennis, M., Samvelyan, M., Foerster, J., Grefenstette, E., & Rocktäschel, T. (2022). *Evolving Curricula with Regret-Based Environment Design (ACCEL).* ICML 2022. arXiv:2203.01302.
- Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., & Anandkumar, A. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models.* arXiv:2305.16291.
- Wang, R., Lehman, J., Clune, J., & Stanley, K. O. (2019). *Paired Open-Ended Trailblazer (POET): Endlessly Generating Increasingly Complex and Diverse Learning Environments and Their Solutions.* arXiv:1901.01753.
- Wang, R., Lehman, J., Rawal, A., Zhi, J., Li, Y., Clune, J., & Stanley, K. O. (2020). *Enhanced POET: Open-Ended Reinforcement Learning through Unbounded Invention of Learning Challenges and their Solutions.* ICML 2020. arXiv:2003.08536.
- Zhang, J., Lehman, J., Stanley, K., & Clune, J. (2023). *OMNI: Open-endedness via Models of human Notions of Interestingness.* arXiv:2306.01711.
