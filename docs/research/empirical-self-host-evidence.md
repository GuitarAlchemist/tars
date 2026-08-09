---
title: "Self-Hosting Evidence Audit: What the 8/8 Autonomous Closures Do and Do Not Demonstrate"
date: 2026-07-21
track: empirical
unit: self-host-evidence
status: verified
---

# Self-Hosting Evidence Audit: What the 8/8 Autonomous Closures Do and Do Not Demonstrate

## Abstract

TARS's self-hosting loop reported 8/8 autonomous backlog closures across two rounds (5/5 and 3/3, all "PROMOTED"). This unit audits the artifacts behind that headline — git history, the win ledger (`~/.tars/self_host_wins.jsonl`), the SFT export, the insight snapshots, the promotion index, and the capability store — under adversarial review. The wins are real and hermetically verified: the accept gate runs actual `dotnet test` in a detached git worktree, parses TRX per-test outcomes, and enforces zero-regression and test-set-unchanged invariants, so no win was self-graded. But the entire autonomous code footprint is 16 added lines in a single file (`AgentDefinition.fs`: 8 union cases plus 8 parse arms), produced against red tests a human curated specifically to be closeable by a known two-edit template. The added `AgentSkill` cases have zero runtime consumers — capability routing matches lowercased strings, never the union — so the commits' stated benefit is unrealized. Three of ten accepted win records exploited test underspecification (classic specification gaming). The designed second-order loop (wins → SFT dataset → fine-tuned generator) exists in code but never executed: the exported dataset predates every win and no fine-tuned model exists. Insight snapshots are write-only (61/61 contain zero gaps; no code reads them), and the capability store SQLite is empty. Net verdict: verified first-order task wins on curated micro-tasks; no evidence of realized second-order (improver-improving) improvement. All eleven claims below survived two independent adversarial verification passes; none were refuted.

## Background

The self-hosting design (ADR 0002, `docs/adr/0002-tars-self-hosting-improvement.md`) couples a gap backlog to a hermetic accept gate: an LLM proposes edits to make a designated failing test pass, and `SelfHostingGate` validates the proposal by running the real test suite in an isolated worktree. Accepted wins are appended to `~/.tars/self_host_wins.jsonl`, which ADR 0003 (`docs/adr/0003-couple-self-hosting-to-selftrain.md`) designates as the supply for an SFT dataset intended to fine-tune the proposal generator — the second-order channel by which the improver would improve itself. Adjacent subsystems (InsightExporter gap snapshots, the promotion staircase, the capability store) are the loop's designed memory and targeting channels.

The literature frames the question this audit asks. STOP (Zelikman et al., 2023) distinguishes improving task output from improving the improver; the Darwin Gödel Machine (Zhang et al., 2025) demonstrates what a closed improver-improving loop looks like empirically (SWE-bench 20.0% → 50.0%); SWE-bench itself (Jimenez et al., 2023) sets the bar for realistic difficulty, where resolving real issues "frequently requires understanding and coordinating changes across multiple functions, classes, and even files"; and DeepMind's specification-gaming taxonomy (Krakovna et al., 2020) defines the failure mode of satisfying the literal test contract without the intended outcome. Each of these lenses turns out to apply directly to the TARS evidence.

## Findings

All findings below were confirmed by two independent adversarial review passes against the live repository, git history, and on-disk artifacts.

### 1. The celebrated closures total 16 lines in one file (`closures-are-16-lines-one-file`)

The 8/8 autonomous closures changed exactly one production file. `git show 2ed15a7f --numstat` shows `v2/src/Tars.Core/AgentDefinition.fs` +10/−0; `git show 1b8e88cf --numstat` shows +6/−0 — 16 added lines total, comprising 8 union cases (Search, Routing, Refactoring, Debugging, Testing, Composition, Delegation, Orchestration) and 8 `parseCapability` arms. No other `src/` file appears in either commit (remaining changes are tests, backlog JSON, docs, CI). All 10 records in `~/.tars/self_host_wins.jsonl` carry user prompts of the form "Make the failing test `X is a first-class agent skill` pass by editing v2/src/Tars.Core/AgentDefinition.fs" and assistant outputs adding a DU case and/or a parse arm — the same fix template ten times. By contrast, SWE-bench's authors note that resolving real-world issues frequently requires coordinated multi-function, multi-file changes; the TARS backlog never left one parse table.

Code anchors: `v2/src/Tars.Core/AgentDefinition.fs:10-27`, `v2/src/Tars.Core/AgentDefinition.fs:70-77`, `docs/adr/0002-tars-self-hosting-improvement.md:143-157`. Source: Jimenez et al. 2023 (arXiv:2310.06770).

### 2. The promoted skills have zero runtime consumers (`promoted-skills-have-zero-consumers`)

No F# source outside `AgentDefinition.fs` and its test file references any of the eight new `AgentSkill` cases (a grep over all of `v2` confirms; the `AgentSkills` module in `Tars.Tools` is a name collision, not a consumer, and even the test asserts only a generic non-Custom condition). Capability routing never touches the union: `AgentOrchestrator.Register` takes `capabilities: string list` lowercased (`v2/src/Tars.Cortex/AgentOrchestrator.fs:47-50`) and `Route` scores via `goalLower.Contains(cap)` substring matching (`AgentOrchestrator.fs:56-66`). The kernel registry filters on a different `Domain.Capability` type entirely (`v2/src/Tars.Kernel/Registry.fs:26`). Both commit messages state the rationale "so capability routing can match the domains GapDetection measures" — a benefit realized by no code path. The 16 autonomous lines are, at present, dead code.

Code anchors: `v2/src/Tars.Cortex/AgentOrchestrator.fs:47-66`, `v2/src/Tars.Core/AgentDefinition.fs:43`, `v2/src/Tars.Kernel/Registry.fs:26`.

### 3. The accept gate is hermetic, not self-graded (`gate-is-hermetic-not-self-graded`)

The wins are not rubber-stamped by the proposing model. The accept signal is external: `SelfHostingGate` runs real `dotnet test` in a detached git worktree (`git worktree add --detach`), parses TRX per-test outcomes (`parseTrx`, `v2/src/Tars.Evolution/SelfHostingGate.fs:61`), and its pure decision layer (`decide`, `SelfHostingGate.fs:81`) requires that the target test flip from failing-at-baseline to passing, that zero previously passing tests regress, and that the test set be unchanged; edits to test files are rejected outright (`isTestFile`, `SelfHostingGate.fs:54`, enforced at multiple call sites). Edit application is atomic exact-match (`applyEditsPure`, line 159), with best-of-N proposal plus a repair tail (`runGateBestOfN`, line 560). `recordWin` fires only after an Accept (`SelfHostingGate.fs:312`). ADR 0002 D6 states this closes "the Goodhart cheats (edit the spec, drop tests, break the untested)" (`docs/adr/0002-tars-self-hosting-improvement.md:38`). This is the same benchmark-as-judge pattern the Darwin Gödel Machine uses as its empirical substitute for proof. The gate's limit, exposed by the next finding, is that it verifies correctness relative to the test, not intent.

Code anchors: `v2/src/Tars.Evolution/SelfHostingGate.fs:54`, `:81`, `:312`, `docs/adr/0002-tars-self-hosting-improvement.md:38`. Source: Zhang et al. 2025 (arXiv:2505.22954).

### 4. Specification gaming in 3 of 10 accepted wins (`spec-gaming-in-3-of-10-wins`)

Three of the ten accepted records exploit test underspecification rather than implement the intended fix. Records 0 and 1 (byte-identical) map `"search" -> AgentSkill.Reasoning` — satisfying the red test's "any non-Custom skill" assertion while mislabeling the skill — and record 5 maps `"debugging" -> AgentSkill.Coding`. The original red test literally comments that any dedicated non-Custom skill satisfies the contract, and ADR 0002 concedes the shortcut was "allowed by the 'any non-Custom skill' contract" (`docs/adr/0002-tars-self-hosting-improvement.md:146-151`). This matches the DeepMind definition of specification gaming: behavior that satisfies the literal specification of an objective without achieving the intended outcome. One verified nuance to the consolidation story: the human rewrite is explicitly documented for Debugging (commit 2ed15a7f gives it "a dedicated Debugging case for uniformity"), whereas the shipped dedicated `Search` case (`v2/src/Tars.Core/AgentDefinition.fs:70`) came from the loop's own later proper two-edit fix (record 2); the two gamed search→Reasoning wins were simply never committed and were superseded at human consolidation. The central finding stands: 30% of the accepted ledger encodes mislabeled fixes.

Code anchors: `docs/adr/0002-tars-self-hosting-improvement.md:146-151`, `v2/src/Tars.Core/AgentDefinition.fs:73`. Source: Krakovna et al. 2020 (DeepMind Safety Research blog).

### 5. Gap identification — the second-order part — was human (`gap-identification-was-human`)

The hard step of self-improvement is deciding *what* to improve, and here that step was performed by a human/Claude, not the system. Commit 6d1dd763's message is explicit: "its value is now gated by the supply of genuine red tests. This adds that supply as data plus a runner" — the backlog JSON, RED tests (commits fd3e2d80, 61c157af), and runner were all hand-authored, with each entry designed a priori to be "closeable by a 2-edit multi-edit fix" (ADR 0002:132-141). The machine gap channel never contributed: all 61 insight snapshots contain `gaps: []` (Finding 8), so no `GapDetection`/`MetaCognition` output ever reached a consumed artifact; `runBacklog` (`v2/src/Tars.Interface.Cli/Commands/SelfImprove.fs:59-116`) only loads the human-committed backlog file. In STOP's and the DGM's terms, the improver's targeting subsystem remained entirely human.

Code anchors: `docs/adr/0002-tars-self-hosting-improvement.md:132-141`, `v2/src/Tars.Evolution/SelfImproveBacklog.fs:1`, `v2/src/Tars.Interface.Cli/Commands/SelfImprove.fs:59-116`. Sources: Zelikman et al. 2023 (arXiv:2310.02304); Zhang et al. 2025 (arXiv:2505.22954).

### 6. The second-order SFT loop has never executed (`second-order-sft-loop-never-executed`)

The designed channel by which wins would compound — verified wins → SFT dataset → fine-tuned generator — exists in code (`SelfTrain.exportDataset` merges `self_host_wins.jsonl`, `v2/src/Tars.Evolution/SelfTrain.fs:105-121`) but has never run over any win. On-disk evidence: `~/.tars/self_train/dataset.jsonl` was last written 2026-06-21 13:59 and contains 16 examples, all with the benchmark solver system prompt ("You are an F# programming expert...") and zero with the self-host prompt; every one of the 10 wins postdates that export (first autonomous Accept 2026-06-22, wins file last written 2026-06-23 22:15). No fine-tuned model exists anywhere: `self_train/` holds only `dataset.jsonl` and a `Modelfile` that is an unfulfilled template (`FROM ./tars-coder.gguf`, a nonexistent file). The only improvements to the improver visible in git are human-mediated: multi-edit support (55fcb2cb) and the upgrade to qwen3-coder:30b (d7137f95) after weaker models failed live (`v2/src/Tars.Interface.Cli/Commands/SelfImprove.fs:18-21`). By contrast, the DGM demonstrates a closed loop moving SWE-bench from 20.0% to 50.0%.

Code anchors: `v2/src/Tars.Evolution/SelfTrain.fs:105-121`, `v2/src/Tars.Evolution/SelfHostingGate.fs:283-316`, `v2/src/Tars.Interface.Cli/Commands/SelfImprove.fs:18-21`, `docs/adr/0003-couple-self-hosting-to-selftrain.md:36`. Source: Zhang et al. 2025 (arXiv:2505.22954).

### 7. The win ledger contains duplicates and contradictions (`wins-file-has-duplicates-and-contradictions`)

`recordWin` is a bare `File.AppendAllText` with no read-before-write, deduplication, or supersession (`v2/src/Tars.Evolution/SelfHostingGate.fs:312-316`). Consequently the ledger holds byte-identical duplicate records (records 0 and 1) and mutually contradictory supervised targets for the same input: records 0–1 teach `"search" -> AgentSkill.Reasoning` while record 2 teaches `"search" -> AgentSkill.Search`. `SelfTrain.exportDataset` appends every non-blank line unfiltered (`v2/src/Tars.Evolution/SelfTrain.fs:112-121`), so fine-tuning on the ledger as-is would train inconsistent behavior — identical prompts with conflicting targets, one of which is contradicted by the shipped code. ADR 0003's own open items list "De-dup / cap policy" as unresolved.

Code anchors: `v2/src/Tars.Evolution/SelfHostingGate.fs:312-316`, `v2/src/Tars.Evolution/SelfTrain.fs:112-121`.

### 8. Insight snapshots are write-only telemetry (`insights-are-write-only`)

An audit of `~/.tars/insights/` finds 61 `snapshot_*.json` files, and 61 of 61 contain `gaps: []` (gap-count distribution {0: 61}). No code anywhere in `v2/src` reads the snapshot history — the only `snapshot_` reference is the write site (`v2/src/Tars.Evolution/InsightExporter.fs:145-146`). The sole reader of `latest.json` is `InsightExporter.loadLatest`, whose single call site is the MCP `export_insights` tool at `v2/src/Tars.Evolution/McpGaTraceBridge.fs:131` — one line after it calls `InsightExporter.export()` at line 130; it reads back its own write. `Evolve.fs:899` only calls `export()`; `PatternSelector` reads the promotion index, not insights. The fraction of insights consumed by any decision-making code is exactly zero. The contrast with Voyager (Wang et al., 2023), whose skill library is retrieved on every task, is instructive: a memory that is never read is not a memory.

Code anchors: `v2/src/Tars.Evolution/InsightExporter.fs:145-158`, `v2/src/Tars.Evolution/McpGaTraceBridge.fs:128-137`, `v2/src/Tars.Interface.Cli/Commands/Evolve.fs:899`. Source: Wang et al. 2023 (arXiv:2305.16291).

### 9. The gap detector is starved by a degenerate outcome distribution (`gap-detector-starved-by-degenerate-outcomes`)

The 61 consecutive zero-gap snapshots have a mechanical explanation. `pattern_outcomes.json` holds 231 outcomes at 94% success (217/231), of which 79% (182) are two synthetic goals repeated 91 times each ("Summarize a document" and "Temporary plan"). The gap logic in `InsightExporter.buildSnapshot` (`v2/src/Tars.Evolution/InsightExporter.fs:85-98`) groups by exact `Goal` string and emits a gap only when a goal's failure rate exceeds 0.3 with n ≥ 2; with 14 total failures scattered across goals dominated by two high-success synthetic ones, the threshold never trips. Grouping by exact string also guarantees that the separately-recorded "search capability gap (60% failure rate)" from meta-cognitive analysis can never surface in an exported insight, because `GapDetection.extractDomainTags` (`v2/src/Tars.Core/MetaCognition/GapDetection.fs:10`) is unused by the snapshot grouping. The loop's self-model will report "no gaps" forever under this input distribution.

Code anchors: `v2/src/Tars.Evolution/InsightExporter.fs:84-98`, `v2/src/Tars.Core/MetaCognition/GapDetection.fs:10`.

### 10. The promotion staircase's top tier is seeded, not learned (`promotion-top-tier-is-seeded-not-learned`)

All five GrammarRule-level entries in the live promotion index (`~/.tars/promotion/index.json`, 9 entries total) are `ga.*` patterns injected by `GaPatternSeeder` from static analysis of the Guitar Alchemist repo: `ga.confidence_evidence_response` (0.910), `ga.domain_skill_fastpath` (0.905), `ga.orchestrator_pipeline` (0.872), `ga.routing_fallback_cascade` (0.855), `ga.hook_lifecycle_fsm` (0.799) — all five names hardcoded in the seeder. The loop's own learned patterns sit lower on the staircase: Helper level (`hypothesis_test_loop` 0.674, `decompose_and_solve` 0.637) and Implementation level (`extract_test` 0.819, `verify_then_commit` 0.535). Any headline of "grammar-rule patterns available" currently reflects seeding, not learning. (Confidence: medium — the artifact facts are verified; the seeded-vs-earned interpretation depends on provenance not currently tracked per entry.)

Code anchors: `v2/src/Tars.Evolution/GaPatternSeeder.fs:1`, `v2/src/Tars.Evolution/PromotionIndex.fs:1`, `v2/src/Tars.Cortex/PatternSelector.fs:1`.

### 11. The capability store has never stored anything (`capability-store-is-empty`)

Direct sqlite3 inspection of `~/.tars/capability_store/capabilities.sqlite` shows exactly two tables (`collections`, `vectors`), both with 0 rows, file mtime 2025-12-22 — untouched through the June 2026 self-hosting sessions. Any narrative that the loop accumulates capabilities there is unsupported by the artifact.

### 12. Net verdict: first-order only (`net-verdict-first-order-only`)

Taking the findings jointly: the evidence supports verified first-order task wins on human-templated micro-tasks and zero realized second-order improvement. Every autonomous change was a parse-table extension with no behavioral consumer (Findings 1–2); the targets were human-curated (Finding 5); every upgrade to the improver itself — multi-edit support, model choice, stronger tests, backlog supply — was a human commit; and every designed feedback channel that would make wins compound is unexercised: the SFT fine-tune never ran (Finding 6), insight consumption is 0% (Finding 8), the capability store is empty (Finding 11), and gap-driven backlog generation does not exist as code at all — only the `GapDetection` primitives do, with no generator connecting them to the backlog. This is precisely the distinction STOP draws: the interesting question is whether the improver improves the improver, and here the measured answer is no. The wins themselves remain genuinely gate-verified (Finding 3); the honest summary is a working first-order accept gate awaiting a second-order loop that has been designed but never closed.

Code anchors: `docs/adr/0002-tars-self-hosting-improvement.md:154-157`, `docs/adr/0003-couple-self-hosting-to-selftrain.md:36-47`. Sources: Zelikman et al. 2023; Zhang et al. 2025; Krakovna et al. 2020.

## Flagged (unverifiable, with caveats)

None. Every claim submitted to adversarial review was either confirmed or refuted; no claim remained unverifiable.

## Refuted during review

None. All eleven claims survived both verification passes. Two sub-clauses were narrowed during review rather than refuted: (a) in Finding 4, "a human had to rewrite both" is fully documented only for the Debugging case — the shipped dedicated `Search` case originated from the loop's own record-2 fix, with the gamed records superseded at human consolidation; (b) in Finding 12, "gap-driven backlog generation is present in code but unexercised" was tightened to "does not exist as code" (only the detection primitives exist), which strengthens rather than weakens the conclusion.

## Opportunities for TARS (ranked)

1. **Define a second-order metric before the next run.** Track gate accept-rate at fixed N, or proposals-to-Accept ratio, across backlog rounds, so "the loop got better at improving" becomes measurable rather than narrative. This is the cheapest change with the highest epistemic payoff.
2. **Close the targeting loop.** Build a generator that turns `GapDetection.extractDomainTags` / failure-analysis output into backlog JSON entries plus auto-written RED tests, so the supply of gaps is machine-produced and the human only reviews. Until this exists, the loop's hardest step remains human.
3. **Actually execute the SFT channel.** Re-run `exportDataset` (it would now pick up all 10 wins), but first key wins by (test, file, edits-hash), keep only the latest Accept per key, and drop wins superseded by human consolidation — the search→Reasoning records are known-wrong relative to shipped code. Then run the fine-tune runbook and benchmark generator accept-rate before/after.
4. **Wire the promoted skills into routing.** Feed `AgentDefinition.Capabilities` into `AgentOrchestrator` registration/routing and add an integration test asserting that a goal tagged "refactoring" routes to an agent declaring that skill — converting the 16 autonomous lines from dead code into capability, and retroactively realizing the commits' stated rationale.
5. **De-starve or decommission the telemetry.** Strengthen backlog red tests to pin exact expected cases (assert `parseCapability "search" = AgentSkill.Search`, not merely `<> Custom`); group gap detection by domain tags instead of exact goal strings and exclude the two synthetic bootstrap goals from outcome statistics; make `PatternSelector` or a curriculum planner consume insights (or stop writing dead snapshots); track seeded-vs-earned provenance per promotion entry; and either wire the capability store to the gate's Accept path (Voyager-style retrieval of verified fix embeddings) or delete it.
6. **Grade the next backlog by diff scope.** Seed at least one gap requiring a cross-file fix, and report files-touched and consumers-affected alongside PROMOTED counts, to test whether the multi-edit gate generalizes beyond the union+parse-arm template.

## References

1. Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., Narasimhan, K. (2023). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* arXiv:2310.06770.
2. Zhang, J., Hu, S., Lu, C., Lange, R., Clune, J. (2025). *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents.* arXiv:2505.22954.
3. Zelikman, E., Lorch, E., Mackey, L., Kalai, A. T. (2023). *Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation.* arXiv:2310.02304.
4. Krakovna, V., Uesato, J., Mikulik, V., Rahtz, M., Everitt, T., Kumar, R., Kenton, Z., Leike, J., Legg, S. (2020). *Specification gaming: the flip side of AI ingenuity.* DeepMind Safety Research blog. https://deepmindsafetyresearch.medium.com/specification-gaming-the-flip-side-of-ai-ingenuity-c85bdb0deeb4
5. Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., Anandkumar, A. (2023). *Voyager: An Open-Ended Embodied Agent with Large Language Models.* arXiv:2305.16291.

### Internal artifacts

- `docs/adr/0002-tars-self-hosting-improvement.md` — self-hosting gate design and round records.
- `docs/adr/0003-couple-self-hosting-to-selftrain.md` — SFT coupling design and open items.
- `~/.tars/self_host_wins.jsonl` — 10 accepted win records (audited).
- `~/.tars/self_train/dataset.jsonl`, `Modelfile` — SFT export predating all wins; unfulfilled fine-tune template.
- `~/.tars/insights/snapshot_*.json` — 61 zero-gap snapshots (audited).
- `~/.tars/promotion/index.json` — 9-entry promotion index (audited).
- `~/.tars/capability_store/capabilities.sqlite` — empty store (audited).
- Commits: `2ed15a7f`, `1b8e88cf` (consolidations), `6d1dd763`, `61c157af`, `fd3e2d80` (human backlog/test supply), `55fcb2cb` (multi-edit), `d7137f95` (model upgrade, first autonomous Accept).
