---
title: "MCTS and Skill Search in TARS: UCT over Workflow Derivations, Contextual Bandits in Name Only"
date: 2026-07-21
track: theory
unit: mcts-skill-search
status: verified
---

# MCTS and Skill Search in TARS: UCT over Workflow Derivations, Contextual Bandits in Name Only

## Abstract

TARS implements textbook UCT — UCB1 selection with c = sqrt(2), random rollouts, full backpropagation — over a Workflow-of-Thought (WoT) derivation space, but omits every enhancement the MCTS literature prescribes for this class of domain: there is no tree reuse, no transposition handling, and no progressive widening, and the reward is a purely deterministic structural heuristic that never touches an LLM or an execution outcome. The stochastic-bandit machinery of UCT is therefore spent on what is really deterministic combinatorial optimization. The ix Rust bridge compounds the problem by searching a strictly smaller action space (nodes only, never edges or transformations), producing derivations that the F# reward function itself would penalize. On the skill-selection side, pattern selection is structurally a contextual bandit solved with ad-hoc machinery: the Beta-Binomial posterior discards the goal context it records, the final decision is a greedy argmax despite code comments claiming exploration, the softmax-of-means squashing mathematically caps the learned signal (~0.045–0.093) below keyword-heuristic gaps (0.4–0.6), and the promotion boost is an explicitly capped 0.08 word-overlap tiebreaker. Live telemetry (231 outcomes across 38 goals) exhibits the predicted degeneracy — ChainOfThought 195/195 successes, ReAct 0/5, TreeOfThoughts 0/3 — plus a serialization round-trip bug that nests `Custom` pattern kinds roughly 80 quoting levels deep, fragmenting all benchmark outcomes into unlearnable n = 1 arms. Theory prescribes Thompson sampling or LinUCB over goal features, evidence-dominant weighting, and coupling the MCTS reward to the outcome store. All ten findings below were confirmed by adversarial review against the code and live data; none were flagged as unverifiable and none were refuted.

## Background

TARS contains two search/learning subsystems that this unit examines against the bandit and tree-search literature.

The first is a Monte Carlo Tree Search engine (`v2/src/Tars.Evolution/MctsSolver.fs`, `MctsTypes.fs`) applied to WoT workflow derivation (`WotMctsState.fs`): states are partial workflow graphs, actions add nodes from a template pool, add edges between placed nodes, set transformations on Reason nodes, or complete the derivation. A bridge (`MctsBridge.fs`) optionally delegates the search to the ix Rust MCTS engine via an exported EBNF grammar, falling back to the F# solver. The relevant theory is the UCT algorithm of Kocsis & Szepesvári (2006), which applies Auer et al.'s (2002) UCB1 bandit rule at each tree node, and the catalogue of standard MCTS enhancements surveyed by Browne et al. (2012) — transposition tables, tree reuse, progressive widening — plus prior-guided PUCT selection (Silver et al. 2016). The recent LLM-search literature (Tree-of-Thoughts, RAP, PG-TD, LATS) supplies the comparison class for how tree search is coupled to task-grounded evaluation in agent systems.

The second subsystem is pattern selection (`v2/src/Tars.Cortex/PatternSelector.fs`): given a goal string, the `HistoryAwareSelector` recommends one of six reasoning `PatternKind`s (ChainOfThought, ReAct, TreeOfThoughts, GraphOfThoughts, PlanAndExecute, WorkflowOfThought) by combining a keyword heuristic, a Beta-Binomial "bandit" score over recorded outcomes, a golden-trace bonus, and a promotion-index boost. This is structurally a contextual bandit problem — arms are pattern kinds, the context is the goal text, the reward is binary success — for which LinUCB (Li et al. 2010) and linear contextual Thompson sampling (Agrawal & Goyal 2013) are the textbook prescriptions. Live behavior was checked against the on-disk telemetry at `~/.tars/pattern_outcomes.json` (231 outcomes, 38 distinct goals) and the promotion index at `~/.tars/promotion/index.json`.

The claims below survived a two-pass adversarial review in which each was independently re-verified against the source and the live data files.

## Findings

### 1. The F# MCTS is canonical UCT — with an untuned exploration constant

**Claim** (`uct-canonical-ucb1`, confidence: high). TARS's F# MCTS implements canonical UCT exactly: UCB1 selection at every internal node with score mean + c·sqrt(ln(parentVisits)/childVisits), default c = sqrt(2), which reproduces Auer et al.'s UCB1 bonus sqrt(2·ln n / n_j) for [0,1]-bounded rewards; unvisited children receive `Double.MaxValue`, forcing a first visit.

**Evidence.** The `ucb1` function computes `child.TotalReward / float child.Visits` plus `explorationConstant * sqrt (log (float parent.Visits) / float child.Visits)` (`v2/src/Tars.Evolution/MctsSolver.fs:16-22`), and `defaultMctsConfig` sets `ExplorationConstant = sqrt 2.0` with the comment "sqrt(2) is standard" (`v2/src/Tars.Evolution/MctsTypes.fs:43-59`). Algebraically, sqrt(2)·sqrt(ln n/n_j) = sqrt(2·ln n/n_j), Auer et al.'s bonus. The select/expand/rollout/backpropagate phases (`v2/src/Tars.Evolution/MctsSolver.fs:34-53` and onward) match the Kocsis–Szepesvári UCT template. The caveat is that the constant is never tuned to the actual reward spread: `computeReward` is a weighted sum whose realistic range is compressed (roughly 0.3–0.9, with a floor of 0.12 for any non-empty graph), so sqrt(2) over-explores relative to the effective reward scale. Kocsis & Szepesvári's consistency analysis assumes the exploration bias is scaled to the payoff range.

**Sources.** Kocsis & Szepesvári 2006; Auer, Cesa-Bianchi & Fischer 2002.

### 2. No tree reuse, no transposition handling

**Claim** (`no-tree-reuse-no-transpositions`, confidence: high). Every search call builds a fresh root, and states reached by permuted action orders (AddNode A then B versus B then A yield identical `WotDerivationState` node sets) are stored as distinct subtrees, duplicating statistics across the tree.

**Evidence.** `search` constructs `let root = createNode initialState None None` on every invocation (`v2/src/Tars.Evolution/MctsSolver.fs:91-112`), and `MctsResult` carries only best actions and statistics — no tree handle for warm-starting a subsequent call. `WotMctsState.applyAction` appends to lists (`v2/src/Tars.Evolution/WotMctsState.fs:150-164`), so AddNode/AddEdge actions commute up to list order, creating a heavily transposed DAG that the tree treats as a pure tree; with no transposition table, each permutation accrues independent Visit/TotalReward statistics. The bridge (`v2/src/Tars.Evolution/MctsBridge.fs:123-145`) likewise returns only the action list. Browne et al. (2012) list transposition tables and tree reuse/pondering as standard MCTS enhancements precisely for such domains.

**Sources.** Browne et al. 2012.

### 3. The reward is deterministic — UCT's bandit machinery is misapplied

**Claim** (`deterministic-proxy-reward`, confidence: high). The MCTS reward is a purely deterministic structural heuristic (six weighted terms: connectivity 0.25, check coverage 0.20, tool coverage 0.15, kind diversity 0.15, DAG-root 0.15, size bell 0.10) with no LLM call and no execution feedback. The only stochasticity UCT averages over is the random rollout policy; the underlying problem is deterministic combinatorial optimization, for which UCT's stochastic-bandit regret machinery is not the prescribed tool.

**Evidence.** `computeReward` is a pure function of the state with exactly those weights (`v2/src/Tars.Evolution/WotMctsState.fs:60-105`), and the module documentation states "This does NOT execute workflows against an LLM" (`v2/src/Tars.Evolution/WotMctsState.fs:3-9`). The only randomness enters through the rollout rng (`v2/src/Tars.Evolution/MctsSolver.fs:60-71`). UCB1/UCT guarantees target stochastic per-arm reward distributions; for a cheap deterministic terminal objective, best-first, beam, or exact search over the same heuristic dominates. The LLM-search literature derives its value from coupling tree search to task-grounded rewards — LLM self-evaluation in RAP, public-test execution in PG-TD — rather than static structural proxies.

**Sources.** Hao et al. 2023 (RAP); Zhang et al. 2023 (PG-TD); Kocsis & Szepesvári 2006.

### 4. Branching factor swamps the iteration budget; expansion order is biased

**Claim** (`branching-swamps-budget`, confidence: high). With no progressive widening and an expand-all-untried-first tree policy, the O(n²) edge-action branching factor swamps the default iteration budgets (200 for `quickSearch`, 1000 default), so UCB1 selection rarely activates below depth 1–2 and the search degenerates toward uniform shallow enumeration. Expansion order is additionally deterministic — the first untried action is simply template/edge enumeration order — biasing early-indexed templates.

**Evidence.** `treePolicy` expands whenever `UntriedActions` is non-empty (`v2/src/Tars.Evolution/MctsSolver.fs:34-53`, expansion at line 36-38) and `expand` takes the head of the list with no shuffle (line 48); `UntriedActions` is initialized verbatim from `LegalActions()`. `legalActions` enumerates AddEdge for every ordered node pair (`v2/src/Tars.Evolution/WotMctsState.fs:112-143`, edge enumeration at 124-131) — six placed nodes yield 30 edge actions, plus AddNode actions and up to four SetTransformation actions per untransformed Reason node, so a single node's children can consume more than 40 of `quickSearch`'s 200 iterations (`v2/src/Tars.Evolution/MctsBridge.fs:148-155`) before any UCB1-guided descent occurs. Progressive widening (Browne et al. 2012) and prior-guided PUCT selection — argmax over Q + c_puct·P·sqrt(N)/(1+n), as in AlphaGo (Silver et al. 2016) — are the standard remedies for wide action spaces.

**Sources.** Browne et al. 2012; Silver et al. 2016.

### 5. The ix Rust bridge searches a strictly smaller — and inconsistent — action space

**Claim** (`ix-bridge-action-space-loss`, confidence: high). `templatesToEbnf` encodes only node sequences (`root ::= (node_0 | … | node_k)+`), and `indicesToActions` maps derivations back to AddNode plus Complete only — never AddEdge or SetTransformation — so any ix-produced derivation is an edge-less workflow that the F# reward function itself would score 0 on its 0.25-weight connectivity term.

**Evidence.** `templatesToEbnf` emits only node productions (`v2/src/Tars.Evolution/MctsBridge.fs:52-64`), and `indicesToActions` constructs only `WotAction.AddNode` followed by `WotAction.Complete` (`v2/src/Tars.Evolution/MctsBridge.fs:69-84`). Meanwhile the F# space requires at least two nodes and at least one edge before Complete is legal (`v2/src/Tars.Evolution/WotMctsState.fs:140-141`), and `computeReward` gives connectivityScore = min 1.0 (edges/(nodes−1)) = 0 for an edge-less multi-node graph (`v2/src/Tars.Evolution/WotMctsState.fs:60-105`, connectivity at 67-69). The two backends therefore optimize inconsistent objectives, and which one ran is visible only through the returned `usedIx` flag. (Reviewers noted one pedantic edge case: a degenerate single-node ix derivation would score 0.5 on connectivity rather than 0; any multi-node ix output scores 0 as claimed.)

### 6. Pattern selection is a contextual bandit whose learned component is context-free

**Claim** (`pattern-selection-is-contextual-bandit`, confidence: high). Pattern selection is structurally a contextual bandit (arms = six PatternKinds, context = goal text, reward = binary Success), but the learned component discards the context: the Beta(successes+1, failures+1) posterior pools outcomes across all goals, even though the goal string is recorded with every outcome and 38 distinct goals exist in the live store. Context re-enters only through hand-written keyword heuristics.

**Evidence.** `PatternOutcome` stores `Goal` per record (`v2/src/Tars.Cortex/PatternSelector.fs:17-22`); `banditScores` groups solely by `PatternKind` — `List.groupBy (fun o -> o.PatternKind)` — and never reads `o.Goal` (`v2/src/Tars.Cortex/PatternSelector.fs:247-263`). Goal context influences scores only via hardcoded substring checks in `heuristicScore` (`v2/src/Tars.Cortex/PatternSelector.fs:291-298`) and word-overlap counts in `scoreEntry`/`promotionBoost`. The live telemetry at `~/.tars/pattern_outcomes.json` contains 231 outcomes over exactly 38 distinct goals. LinUCB (Li et al. 2010) and linear contextual Thompson sampling (Agrawal & Goyal 2013) are the textbook prescriptions: learn a per-arm reward model over context features with optimism or posterior-sampling exploration, replacing both the keyword heuristics and the pooled posterior.

**Sources.** Li et al. 2010; Agrawal & Goyal 2013.

### 7. Recommend is a greedy argmax — the claimed exploration does not exist

**Claim** (`greedy-argmax-no-exploration`, confidence: high). Despite the code comment claiming the softmax "never collaps[es] to zero so under-explored kinds keep a chance", `Recommend` performs a deterministic argmax over combined scores — the softmax probabilities are used as additive score components, never sampled — so the selector is a greedy bandit with zero directed exploration, which bandit theory shows suffers linear regret when the greedy arm is suboptimal.

**Evidence.** `member _.Recommend(goal, _state) = combineScores goal |> Map.toList |> List.maxBy snd |> fst` (`v2/src/Tars.Cortex/PatternSelector.fs:381-383`); the exploration claim sits at `v2/src/Tars.Cortex/PatternSelector.fs:241-246`, while the softmax output (`v2/src/Tars.Cortex/PatternSelector.fs:259-263`) is only ever added as a 0.2-weighted component. Live data confirms degenerate feedback: ChainOfThought has 195 of 231 pulls with a 100% success record, ReAct 5 pulls with 0 successes, TreeOfThoughts 3/0, GraphOfThoughts 2 pulls, WorkflowOfThought 1 pull — arm pulls track keyword routing, not learned value, and unexplored arms stay unexplored. Thompson sampling — draw θ_k ~ Beta(α_k, β_k), pick argmax θ — would restore exploration with essentially one line changed and is empirically competitive (Chapelle & Li 2011); UCB1 on the arms is the frequentist alternative (Auer et al. 2002).

**Sources.** Chapelle & Li 2011; Auer, Cesa-Bianchi & Fischer 2002.

### 8. The bandit signal is mathematically dominated by the keyword heuristic

**Claim** (`bandit-signal-mathematically-dominated`, confidence: high). The learned signal can never override the keyword heuristic no matter how much evidence accumulates: Beta means lie in (0,1), so softmax probability ratios are bounded by e, giving a maximum pairwise bandit-score gap of 0.2·(e−1)/(e+1) ≈ 0.093 with two arms (≈ 0.045 with six) versus keyword-heuristic gaps of 0.4–0.6. The combined score is therefore not asymptotically consistent: an arm with a perfect success record loses to any keyword-matched arm forever.

**Evidence.** `banditScores` softmaxes Beta means m_k = α/(α+β) ∈ (0,1) (`v2/src/Tars.Cortex/PatternSelector.fs:259-263`), so p_i/p_j = exp(m_i − m_j) ≤ e; `combineScores` weights this term by 0.2 (`v2/src/Tars.Cortex/PatternSelector.fs:354-370`), while `heuristicScore` assigns 0.8 versus 0.2/0.3 on keyword match (`v2/src/Tars.Cortex/PatternSelector.fs:291-298`), a 0.4–0.6 gap. The live data illustrates the ceiling: ReAct has 0 successes in 5 pulls yet still wins any goal containing "search" or "debug", because its 0.8 keyword score cannot be overcome by any achievable bandit + golden (0.3 max) + promotion (0.08 max) differential when competing arms lack keyword matches of their own. Review confirmed the arithmetic on independent re-derivation and added two live-configuration observations that strengthen the claim: the live softmax actually runs over ~30 arms (because of the fragmentation bug in finding 10), shrinking the bandit differential further, and no golden-trace store exists on disk, so the 0.3 golden term is zero in practice. In the abstract two-arm best case, the full stack 0.3 + 0.093 + 0.08 = 0.473 marginally exceeds a 0.4 gap — but golden and promotion terms are not driven by accumulated outcome evidence, so the headline claim, that outcome evidence alone can never overturn a keyword match, is exact.

**Sources.** Auer, Cesa-Bianchi & Fischer 2002 (logarithmic-regret algorithms require evidence to dominate any bounded prior asymptotically).

### 9. The promotion boost is an ad-hoc capped tiebreaker, not a bandit

**Claim** (`promotion-boost-adhoc-cap`, confidence: high). The promotion-index contribution is a capped tiebreaker: context match is a raw word-overlap count thresholded at ≥4 (boost 0.08) and ≥2 (boost 0.04); the boosted arm is chosen by substring-matching the promoted pattern's name ("routing" → WorkflowOfThought, "skill" → PlanAndExecute, and so on); and only the single top-scoring index entry contributes. The code comment itself states the 0.08 cap exists "so promotion never overrides heuristic margins".

**Evidence.** `scoreEntry` sums integer word-overlap hits × 2 plus LevelRank × 10 + Weight + Score (`v2/src/Tars.Cortex/PatternSelector.fs:304-311`); `promotionBoost` subtracts the base score to recover the context signal and maps it through the 4.0/2.0 thresholds to 0.08/0.04/0 (`v2/src/Tars.Cortex/PatternSelector.fs:313-352`, thresholds at 329-334, name-substring dispatch at 336-347, head-only use at 325). Because LevelRank × 10 dwarfs everything else, the subtraction trick is the only way any context signal survives. None of the Beta-Binomial weights stored in `~/.tars/promotion/weights.json` flow into this path — the selector reads only `index.json`. One evidence detail was corrected during review: of the live index's nine entries, five (not all nine) are at LevelRank 4 (Scores 0.80–0.91), with two at rank 1 and two at rank 0; the conclusion is unaffected because the top-scored entry is always a rank-4 entry (base ~41 versus ~11).

**Sources.** Li et al. 2010.

### 10. A serialization round-trip bug fragments Custom arms into unlearnable n = 1 statistics

**Claim** (`outcome-store-arm-fragmentation-bug`, confidence: high). `PatternOutcomeStore` has a serialization round-trip bug that destroys arm identity for `Custom` kinds: `toDto` prints `PatternKind` with `%A` (yielding e.g. `Custom "benchmark:algorithms"`), `parseKind` fails to match it and wraps the lowercased string in another `Custom`, and because `record()` rewrites the whole file on every append, each append adds one more nesting level of `custom "` to every stored Custom entry. The live file contains entries nested roughly 80 levels deep, and all ~25 benchmark outcomes are fragmented into unique n = 1 arms whose statistics can never accumulate.

**Evidence.** `toDto`: `PatternKind = sprintf "%A" o.PatternKind` (`v2/src/Tars.Cortex/PatternSelector.fs:33`); `parseKind`'s fallthrough `| other -> Custom other` on the lowercased string (`v2/src/Tars.Cortex/PatternSelector.fs:39-47`); `record()` = loadAll (parse) + re-serialize all (`v2/src/Tars.Cortex/PatternSelector.fs:80-86`), so every existing Custom entry passes through one parse→print cycle per append. Review verified the live file end-to-end: 25 Custom benchmark outcomes, each a distinct n = 1 arm, with per-entry "custom" nesting counts of 64–88 (matching the ~80-level estimate) and entry strings approaching 1,000 characters. The innermost tags — benchmark:algorithms (9), musictheory (9), stringmanipulation (3), datastructures (2), errorhandling (1), patternmatching (1) — are exactly the roughly six category arms that should have aggregated.

### 11. Positioning against the LLM-search literature: two learning loops that never share signal

**Claim** (`llm-search-literature-positioning`, confidence: medium). TARS's MCTS occupies an unusual position relative to the LLM reasoning-search literature. Tree-of-Thoughts (BFS/DFS with LLM self-evaluation), RAP (MCTS with LLM world-model rewards), and PG-TD (MCTS-guided decoding rewarded by public-test execution) all close the loop between search and task-grounded evaluation. TARS instead runs MCTS purely over workflow structure before execution and separately runs a bandit over pattern kinds after execution — and the two learning loops never share signal: the MCTS reward ignores `pattern_outcomes.json`, and the bandit ignores MCTS-derived structure.

**Evidence.** The `WotMctsState` module documentation states "This does NOT execute workflows against an LLM" (`v2/src/Tars.Evolution/WotMctsState.fs:3-9`), and `computeReward` (`v2/src/Tars.Evolution/WotMctsState.fs:60-105`) has no dependency on any outcome store; conversely, nothing in `banditScores`/`promotionBoost` (`v2/src/Tars.Cortex/PatternSelector.fs:247-263`) references MCTS results or workflow structure. Both halves of the no-shared-signal claim were verified in code; the loops are provably disjoint. ToT explicitly frames reasoning as tree search with heuristic self-evaluation over thoughts; RAP reformulates LLM reasoning as MCTS-based planning guided by an internal world model; PG-TD uses a planner that tests candidate programs on public test cases to reward tree nodes; LATS unifies reasoning, acting, and planning in one tree. The medium confidence attaches to the literature framing, not to the code facts, which are fully confirmed.

**Sources.** Yao et al. 2023 (ToT); Hao et al. 2023 (RAP); Zhang et al. 2023 (PG-TD); Zhou et al. 2023 (LATS).

### 12. Two minor implementation quirks: budget no-op loop and non-terminal Reward

**Claim** (`time-budget-loop-and-rollout-reward-quirks`, confidence: high). Two minor divergences from the spec: (a) when `TimeBudget` is exceeded, the search does not break — it idles through all remaining `MaxIterations` as no-op loop passes; (b) `rollout` returns `state.Reward()` for depth-capped non-terminal states even though `IMctsState.Reward` is documented as "Reward signal for this terminal state" — benign only because `computeReward` happens to be total over partial derivations.

**Evidence.** In the `for _ in 1 .. config.MaxIterations` loop, the budget check `| Some budget when stopwatch.Elapsed > budget -> ()` skips the body but cannot exit an F# for-loop (`v2/src/Tars.Evolution/MctsSolver.fs:101-112`); `rollout` exits at `maxDepth` and calls `current.Reward()` regardless of `IsTerminal` (`v2/src/Tars.Evolution/MctsSolver.fs:60-71`), against the documentation at `v2/src/Tars.Evolution/MctsTypes.fs:17-20`. Neither corrupts results — the iteration counter is not advanced during the no-op passes — but (a) wastes wall-clock under tight budgets and (b) silently couples correctness to an undocumented totality obligation on every `IMctsState` implementation.

## Flagged

No claims were flagged as unverifiable. Every claim in this unit was confirmed by two independent adversarial review passes against the source code and, where applicable, the live telemetry files.

## Refuted during review

None. No claims were dropped. Three evidence-level corrections were absorbed into the findings above without affecting any conclusion: (i) a single-node ix derivation would score 0.5 rather than 0 on connectivity (finding 5); (ii) the live promotion index has five (not nine) entries at LevelRank 4 (finding 9); (iii) in the abstract two-arm best case, the combined golden+promotion+bandit stack can marginally exceed a 0.4 keyword gap, though that configuration is unrealizable in the live system (finding 8).

## Opportunities for TARS

Ranked by expected impact per unit of implementation effort.

1. **Fix the outcome-store serialization bug (finding 10).** Serialize `PatternKind` as a stable tag + payload (e.g. `{kind: "custom", tag: "benchmark:algorithms"}`), make `parseKind` idempotent, append records rather than rewriting the whole file, and one-shot migrate the corrupted file by regex-extracting the innermost tag. This is a small, surgical fix that restores ~25 benchmark outcomes to six learnable arms; nothing downstream can learn until it lands.

2. **Replace the greedy argmax with Thompson sampling (finding 7).** Draw from each arm's existing Beta posterior and argmax the draws. Same stored statistics, principled exploration, essentially a one-line change to `Recommend`.

3. **Let evidence dominate the keyword heuristic (finding 8).** Drop the softmax-of-means squashing: use the posterior mean (or TS draw) directly with a weight that grows with evidence count, or convert the keyword heuristic into prior pseudo-counts on α so that data can eventually overturn it.

4. **Featurize goals and go contextual (findings 6, 9).** Bag-of-keywords or a small embedding over goal text, then LinUCB or linear-TS per arm. The existing keyword heuristic becomes the prior mean rather than an unlearnable fixed term, and promoted entries become arms in the same bandit (context features = overlap with entry Contexts, reward = downstream Success) instead of a capped 0.08 tiebreaker.

5. **Couple the MCTS reward to outcome telemetry (findings 3, 11).** Blend `computeReward` with per-template success statistics from `pattern_outcomes.json`/benchmark results, or add a cached LLM value head for terminal states — the LATS-style unification that makes tree search pay off in ToT/RAP-class systems. This bridges the two currently disjoint learning loops.

6. **Fix the search mechanics (findings 2, 4).** Shuffle `UntriedActions` on node creation (cheap, removes ordering bias); add progressive widening (expand at most ceil(C·N^α) children); add a transposition table keyed on canonical state (sorted node ids + edge set) or move to Monte Carlo Graph Search; retain the root subtree across evolve-loop cycles that reuse the same template pool; seed PUCT-style action priors from promotion-index weights.

7. **Reconcile the ix bridge's objective (finding 5).** Either extend the exported EBNF to encode edge/transformation choices, or post-process ix node sequences with a deterministic edge-completion pass (e.g. chain edges) before scoring, so both backends optimize the same reward.

8. **Tune or adapt the exploration constant (finding 1) and clean up the quirks (finding 12).** Normalize c by the observed reward standard deviation at the root (or grid-search on the benchmark suite); convert the iteration loop to a `while` with an explicit budget-and-iterations condition; document that `Reward` must be total over non-terminal states or add an explicit heuristic-value member for depth-capped rollouts.

## References

- Agrawal, S. & Goyal, N. (2013). Thompson Sampling for Contextual Bandits with Linear Payoffs. ICML 2013. arXiv:1209.3352.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. Machine Learning 47. DOI 10.1023/A:1013689704352.
- Browne, C., Powley, E., Whitehouse, D., Lucas, S., Cowling, P., Rohlfshagen, P., Tavener, S., Perez, D., Samothrakis, S. & Colton, S. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE Transactions on Computational Intelligence and AI in Games 4(1). DOI 10.1109/TCIAIG.2012.2186810.
- Chapelle, O. & Li, L. (2011). An Empirical Evaluation of Thompson Sampling. NeurIPS 2011.
- Hao, S., Gu, Y., Ma, H., Hong, J., Wang, Z., Wang, D. & Hu, Z. (2023). Reasoning with Language Model is Planning with World Model (RAP). EMNLP 2023. arXiv:2305.14992.
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo Planning. ECML 2006. DOI 10.1007/11871842_29.
- Li, L., Chu, W., Langford, J. & Schapire, R. (2010). A Contextual-Bandit Approach to Personalized News Article Recommendation. WWW 2010. arXiv:1003.0146.
- Silver, D., et al. (2016). Mastering the game of Go with deep neural networks and tree search. Nature 529. DOI 10.1038/nature16961.
- Yao, S., Yu, D., Zhao, J., Shafran, I., Griffiths, T., Cao, Y. & Narasimhan, K. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS 2023. arXiv:2305.10601.
- Zhang, S., Chen, Z., et al. (2023). Planning with Large Language Models for Code Generation (PG-TD). ICLR 2023. arXiv:2303.05510.
- Zhou, A., et al. (2023). Language Agent Tree Search Unifies Reasoning, Acting and Planning in Language Models (LATS). arXiv:2310.04406.
