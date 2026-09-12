---
title: "Replicator Dynamics in TARS: A Constant-Fitness Selection Operator, and Where It Falls Short of the Theory It Names"
date: 2026-07-21
track: theory
unit: replicator-dynamics
status: verified
---

## Abstract

TARS carries a module named `ReplicatorDynamics.fs` and frames its grammar-evolution
machinery in the vocabulary of evolutionary game theory — species, fitness, ESS,
ecosystems. This document examines what that code actually computes and finds a gap
between the name and the mathematics. The implementation is a discrete-time,
forward-Euler selection update over a probability simplex, but its fitness function is
**frequency-independent**: each rule's fitness is fixed once from execution outcomes and
never recomputed as population proportions change. That makes the system not an
evolutionary game at all but pure constant-fitness (Fisherian) selection, whose textbook
behaviour is monotone increase of mean fitness and **fixation on the single fittest
species** — i.e. the destruction of pattern diversity, not its maintenance. There is no
mutation term, so nothing endogenously introduces or preserves variants; the
`SmoothingFloor` that appears to protect diversity is applied before renormalization and
therefore does not even guarantee its own stated 1% lower bound. Beyond the mathematics,
the engine is largely decoupled from live state: the MCP tool feeds it empty outcomes
(every fitness collapses to the neutral 0.5, so the dynamics are a complete no-op), the
CLI constructs a step-count config it never passes, and replicator proportions collide
with Bayesian-softmax weights on the single `Weight` field of `weights.json`. Two
sibling subsystems labelled "evolutionary" — the replicator and a genetic-algorithm
pattern breeder — use different, unconnected formalisms. The net assessment: a
**demonstrably correct but diversity-collapsing selection operator, wired in a way that
mostly no-ops on real data**, and a naming/labelling layer (game, ESS) that the
underlying model does not earn. Each finding below is confirmed against the source and
paired with a concrete, ranked remediation.

## Background

The replicator equation is the central object of evolutionary game theory. In its
continuous form, the rate of change of a strategy's frequency is proportional to its
frequency times the gap between its fitness and the population mean fitness:
ẋ_i = x_i · (f_i(x) − f̄(x)). The defining feature that makes this a *game* is that
fitness f_i(x) depends on the population composition x — typically through a payoff
matrix A, so that f_i = (A x)_i. Frequency dependence is what produces the rich behaviour
evolutionary game theory is known for: polymorphic equilibria, cyclic dynamics, and
evolutionarily stable strategies (ESS) in the sense of Maynard Smith and Price.

When fitness is *not* frequency-dependent — when each f_i is a constant — the replicator
equation degenerates into classical selection dynamics. Here the theory is unambiguous
and old: the mean fitness f̄(x) is a strict Lyapunov function that increases monotonically
until the population concentrates entirely on the strategy of maximal fitness (Fisher's
Fundamental Theorem, in its replicator-dynamics form). Diversity is not preserved; it is
eliminated. The standard remedy is a **mutation term** — the replicator-mutator (or
quasispecies) equation — which couples strategies through a mutation matrix Q and admits
interior equilibria that keep sub-optimal variants alive against selection pressure.

TARS uses this machinery to manage a library of grammar rules that climb a promotion
staircase (Implementation → Helper → Builder → DslClause → GrammarRule). The stated goal
of the surrounding system is to maintain a *healthy, diverse* library of candidate
patterns from which promotions are drawn. That goal is in direct tension with the actual
dynamical content of the replicator module, as the findings show. The relevant code lives
in `v2/src/Tars.Evolution/ReplicatorDynamics.fs`, with call sites in
`v2/src/Tars.Evolution/McpGrammarTools.fs` and
`v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs`, and a sibling genetic-algorithm
subsystem in `v2/src/Tars.Evolution/EvolutionaryPatternBreeder.fs`. Persistent weight
state is written to `~/.tars/promotion/weights.json`.

## Findings

### 1. Fitness is frequency-independent — this is selection, not a game

The module computes each species' fitness once, from execution outcomes, and never
revises it as the population proportions evolve. `computeSpeciesFitness` takes only the
`(success, durationMs)` outcome tuples and returns `successRate − durationPenalty`
(`ReplicatorDynamics.fs:75`). `buildSpecies` sets each species' `Fitness` field a single
time from that function (`ReplicatorDynamics.fs:98`), and the iteration in `step` reads
`s.Fitness` unchanged on every step (`ReplicatorDynamics.fs:115`). No payoff matrix `A`
and no term of the form `(A x)_i` appears anywhere in the module.

Because fitness carries no dependence on the population state x, the model is not an
evolutionary game. There is no frequency-dependent selection, no interaction structure,
and therefore no basis for the game-theoretic framing that the module's vocabulary
implies. What remains is the constant-fitness special case: pure Fisherian selection
dynamics. This is the root cause from which findings 2, 3, and 8 follow.

**Code anchors:** `v2/src/Tars.Evolution/ReplicatorDynamics.fs:75`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:98`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:115`.

**Sources:** Hofbauer & Sigmund (2003); Sandholm (2010).

### 2. Constant fitness guarantees fixation and loss of diversity

Under constant fitness f_i, the mean fitness f̄(x) = Σ x_i f_i is a strict Lyapunov
function that increases until the population concentrates on argmax_i f_i. This is the
replicator-dynamics form of the Fundamental Theorem of Natural Selection: the single
highest-fitness species converges to proportion 1 and every other species to 0. Applied
to TARS, this means the grammar ecosystem structurally *destroys* pattern diversity over
successive iterations rather than maintaining the polymorphic library the promotion
staircase depends on.

The TARS code is exactly this constant-fitness case (finding 1). The selection term is
applied in `step` (`ReplicatorDynamics.fs:115`, `ReplicatorDynamics.fs:120`), and the
*only* thing standing between the dynamics and literal 0/1 fixation is the ad-hoc
`SmoothingFloor = 0.01` (`ReplicatorDynamics.fs:52`) — a numerical clamp, not a
dynamical mechanism. There is no niching, no fitness sharing, no frequency-dependent
penalty for over-representation. Absent the floor, the code does precisely what the
theory predicts: monoculture.

**Code anchors:** `v2/src/Tars.Evolution/ReplicatorDynamics.fs:115`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:120`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:52`.

**Sources:** Hofbauer & Sigmund (1998), Ch. 7; Cressman & Tao (2014).

### 3. No mutation term — it is the pure replicator, not a replicator-mutator

The `step` function contains only the selection term dt·x_i·(f_i − f̄)
(`ReplicatorDynamics.fs:112`, `ReplicatorDynamics.fs:118`). There is no mutation matrix
Q, no inflow term of the form Σ_j Q_ji f_j x_j, and no innovation or exploration term of
any kind. New species enter the system only externally, via `buildSpecies` reading
`WeightedGrammar.load` (`ReplicatorDynamics.fs:89`); they are never generated
endogenously.

This matters because the mutation term is precisely the mechanism that, in the
replicator-mutator (quasispecies) equation, sustains an interior equilibrium and keeps
sub-optimal-but-diverse variants alive against selection. Without it, the TARS dynamics
cannot introduce new pattern variants and cannot resist the fixation pressure of finding
2. The system can only prune what it is handed; it can never explore.

**Code anchors:** `v2/src/Tars.Evolution/ReplicatorDynamics.fs:112`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:118`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:89`.

**Sources:** Page & Nowak (2002); Nowak (2006).

### 4. The SmoothingFloor is not a mutation substitute and does not guarantee its own bound

The `SmoothingFloor` is documented as preventing "instant extinction" with a "1% floor"
(`ReplicatorDynamics.fs:45`), and one might read it as a stand-in for a diversity-
preserving mechanism. It is neither. In `step`, each species' provisional proportion is
computed as `newProp = max floor (Proportion + delta)` *per species*
(`ReplicatorDynamics.fs:121`), and only afterwards is the whole vector renormalized:
`total = Σ newProp` (`ReplicatorDynamics.fs:125`) followed by dividing each entry by
`total`.

The clamp is therefore applied *before* the normalizing divide. The deltas sum to zero, so
Σ(Proportion + delta) = 1 exactly; taking a max-with-floor can only raise values, so the
post-clamp total is ≥ 1, and strictly > 1 whenever any species was lifted up to the
floor. A species pinned at exactly `floor` then becomes `floor / total < floor` after
renormalization. The stated 1% minimum is thus violated in exactly the situation the
floor was meant to handle. No post-normalization floor is ever enforced. The floor is a
numerical guard against division artefacts, not a principled lower bound and certainly
not a mutation term.

**Code anchors:** `v2/src/Tars.Evolution/ReplicatorDynamics.fs:121`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:125`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:45`.

**Sources:** Bomze (1983).

### 5. The MCP grammar_evolve tool runs on empty outcomes and is a no-op

The MCP `grammar_evolve` tool calls `ReplicatorDynamics.evolveEcosystem rules Map.empty`
(`McpGrammarTools.fs:118`) — it passes an empty outcome map. `buildSpecies` then looks up
each rule in that empty map and hands the resulting empty list to
`computeSpeciesFitness []`, which returns the neutral fitness 0.5 for rules with no
observations (`ReplicatorDynamics.fs:78`, `ReplicatorDynamics.fs:97`).

With every f_i = 0.5, the mean fitness f̄ = Σ x_i · 0.5 = 0.5, so every delta =
dt·x_i·(0.5 − 0.5) = 0. `buildSpecies` renormalizes the initial proportions to sum to 1
(each well above the 0.01 floor), so the floor never triggers and the proportions are
returned unchanged across all 50 steps — while the tool reports that "evolution" took
place. The tool is a complete no-op on the data it actually receives, and its output is
therefore misleading.

**Code anchors:** `v2/src/Tars.Evolution/McpGrammarTools.fs:118`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:78`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:97`.

**Sources:** Sandholm (2010), stationarity at equal fitness.

### 6. The CLI --steps flag is silently ignored

The `grammar evolve --steps N` command constructs a config that honours the requested
step count — `let config = { ReplicatorDynamics.defaultConfig with Steps = steps }`
(`GrammarCommand.fs:94`) — but the very next line calls
`ReplicatorDynamics.evolveEcosystem rules outcomesById` (`GrammarCommand.fs:95`), which
internally invokes `simulate defaultConfig species` (`ReplicatorDynamics.fs:192`). The
constructed `config` is never passed anywhere; it is dead code. As a result, `--steps`
(and, by the same route, `--prune`) has no effect: every invocation runs the hardcoded
50-step default.

**Code anchors:** `v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:94`,
`v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:95`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:192`.

### 7. Two writers collide on the single Weight field of weights.json

`~/.tars/promotion/weights.json` has one `Weight` field per rule, but two mechanisms with
incompatible semantics both write it. After a replicator run, `GrammarCommand` overwrites
each rule's weight with the replicator's population proportion —
`{ r with Weight = s.Proportion }` (`GrammarCommand.fs:139`) — and persists it via
`WeightedGrammar.save` (`GrammarCommand.fs:142`). The same file is otherwise populated by
`WeightedGrammar.fromRecurrenceRecords`, which writes softmax logits
(`WeightedGrammar.fs:123`), and by `updateWeight`, which applies Beta-Binomial Bayesian
updates; `save` lives at `WeightedGrammar.fs:257`. There is no reconciliation between the
two: a run of `grammar evolve` silently replaces Bayesian weights with replicator
proportions, and subsequent Bayesian updates silently overwrite them back.

The live file bears this out. It mixes rows with `Source` "tars" and "evolved", and the
eight weights sum to roughly 2.0 (0.372 + 0.306 + 0.322 + 0.206 + 0.208 + 0.199 + 0.191 +
0.196 ≈ 2.0). That is neither a global simplex (which would sum to 1) nor a clean
per-level normalization (the two helper rules and six implementation rules do not form
groups that each sum to 1). The `Weight` field has, in practice, no single well-defined
meaning.

**Code anchors:** `v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:139`,
`v2/src/Tars.Interface.Cli/Commands/GrammarCommand.fs:142`,
`v2/src/Tars.Evolution/WeightedGrammar.fs:123`,
`v2/src/Tars.Evolution/WeightedGrammar.fs:257`.

### 8. detectESS is a popularity heuristic mislabelled as evolutionary stability

`detectESS` (`ReplicatorDynamics.fs:137`) marks a species `IsStable` when
`(s.Fitness >= maxFitness − 0.01) && (s.Proportion >= (1/n) * 0.5)`
(`ReplicatorDynamics.fs:144`) — that is, near-maximal fitness plus a proportion above half
the uniform level. This is a popularity threshold, not the concept it is named after.

The Maynard Smith & Price ESS condition requires either strict payoff dominance
E(i,i) > E(j,i), or E(i,i) = E(j,i) together with E(i,j) > E(j,j) — a test of
uninvadability against rare mutants, stated in terms of a payoff/interaction structure.
That structure does not exist anywhere in the module (finding 1), so the true ESS
condition is not merely unimplemented but *uncomputable* here. Labelling the heuristic's
output as "stable" overclaims: it reports which rule is currently dominant and reasonably
common, nothing about evolutionary stability.

**Code anchors:** `v2/src/Tars.Evolution/ReplicatorDynamics.fs:137`,
`v2/src/Tars.Evolution/ReplicatorDynamics.fs:144`.

**Sources:** Maynard Smith & Price (1973); Maynard Smith (1982).

### 9. The pattern breeder is a genetic algorithm, not replicator dynamics

TARS has a second subsystem in the "evolutionary" family that uses an entirely different
and unconnected formalism. `EvolutionaryPatternBreeder.fs` is a real-valued genetic
algorithm operating on an eight-gene genome (`genomeDimension = 8`). `breed` configures
`MutationRate`, `CrossoverRate`, `EliteCount`, and `PopulationSize`
(`EvolutionaryPatternBreeder.fs:133`) and delegates to `MachinBridge.FallbackGA.minimize`
(`EvolutionaryPatternBreeder.fs:137`); `computeFitness` returns a penalty where "lower is
better (GA minimizes)" (`EvolutionaryPatternBreeder.fs:65`). It returns a single
`BestGenome` (`EvolutionaryPatternBreeder.fs:149`), not a distribution over patterns, and
it never touches `ReplicatorDynamics` or `WeightedGrammar` proportions.

In other words, the breeder optimizes the hyperparameters of a *single* best strategy by
minimizing a penalty, whereas the replicator (nominally) allocates proportion *across*
strategies. These are the two classic and distinct regimes — optimization/adaptive
dynamics versus population/selection dynamics — and TARS runs them side by side with no
coupling between them. The shared "evolutionary" branding obscures that they are solving
different problems with different mathematics.

**Code anchors:** `v2/src/Tars.Evolution/EvolutionaryPatternBreeder.fs:65`,
`v2/src/Tars.Evolution/EvolutionaryPatternBreeder.fs:133`,
`v2/src/Tars.Evolution/EvolutionaryPatternBreeder.fs:149`.

**Sources:** Weibull (1995).

## Flagged (unverifiable)

No claims in this unit required flagging. Every kept finding was confirmed against the
source code (and, for finding 7, against the live `weights.json`) by adversarial review.

## Refuted during review

One claim was dropped because it did not survive scientific-accuracy checking, even
though its code observations were correct:

- **"Forward-Euler discretization vs. multiplicative normalization form."** The claim
  asserted that TARS implements the replicator step as an explicit forward-Euler update
  x_i(t+1) = x_i(t) + dt·x_i·(f_i − f̄) with dt = 0.1, rather than the simplex-invariant
  multiplicative form x_i' = x_i·f_i/f̄, and forces simplex membership by a max-with-floor
  clamp plus renormalization. The code description is entirely accurate (`step` at
  `ReplicatorDynamics.fs:112–127`, `TimeStep` at `ReplicatorDynamics.fs:49`), and the
  mathematical characterization (the multiplicative form is simplex-invariant for
  f_i > 0; forward-Euler can drive proportions negative) is textbook-correct. The claim
  was refuted solely on a **misattributed citation**: arXiv:2402.09824, "On the
  discrete-time origins of the replicator dynamics," is a real and correctly-characterized
  paper, but its authors are Fryderyk Falniowski and Panayotis Mertikopoulos, not the
  "Pangallo, Sanders, Galla, Farmer et al." named in the claim. Under a strict
  scientific-accuracy standard a fabricated attribution refutes the claim as stated, so it
  is recorded here rather than presented as a finding. The underlying code observation may
  be re-raised in a future revision with a correct citation.

## Opportunities for TARS (ranked)

1. **Wire the engine to live telemetry (highest impact, lowest risk).** The MCP
   `grammar_evolve` tool must load the real `PatternOutcomeStore` / `pattern_outcomes.json`
   into `outcomesById` instead of passing `Map.empty` (finding 5), and the CLI must
   actually use the config it builds — call
   `ReplicatorDynamics.simulate config (buildSpecies rules outcomesById)`, or add a config
   parameter to `evolveEcosystem`, so `--steps`/`--prune` take effect (finding 6). Until
   these are fixed, every downstream question about the dynamics is moot because the
   dynamics do not run on real data.

2. **Separate the two weight semantics.** Split `Weight` into distinct
   `BayesianWeight` and `ReplicatorProportion` fields, or define one canonical pipeline
   (Bayesian success-rate → fitness → replicator proportion → selection weight) with a
   documented normalization invariant, so a `grammar evolve` run and a Beta-Binomial update
   no longer silently overwrite each other (finding 7).

3. **Add a genuine diversity-preserving force.** Introduce an explicit mutation/innovation
   term — either a simple x_i += μ·(uniform − x_i) or a proper mutation matrix Q coupling
   patterns that share a promotion lineage — turning the engine into a replicator-mutator
   system whose interior equilibrium preserves exploration (findings 2, 3). This is the
   standard, principled fix for selection-driven collapse and directly serves the promotion
   staircase's need for many candidate patterns. As an interim numerical correction,
   re-apply the floor *after* renormalization (project onto the truncated simplex) and
   document the floor as a numerical guard, not a diversity mechanism (finding 4).

4. **Make the framing honest, or make the game real.** Either rename the module to
   "selection dynamics" / "Fisherian selection" and relabel `IsStable` as `Dominant` to
   match what the code does today (findings 1, 8), or introduce a real payoff/interaction
   matrix — e.g. patterns competing for the same promotion slot or co-occurring in
   lineages get frequency-dependent payoffs — so the "game" and "ESS" vocabulary become
   mathematically justified. If a payoff matrix is added, implement the real ESS test (or
   invoke the folk-theorem link ESS ⇒ asymptotically stable rest point) and reserve
   `IsStable` for species that pass it.

5. **Compose the two evolutionary loops.** Clarify the division of labour in code and docs
   — the GA breeder tunes within-strategy hyperparameters; the replicator selects across
   strategies — and consider feeding the GA's per-strategy fitness into the replicator's
   f_i, giving a coherent two-timescale system: breed the genome, then let selection
   allocate proportion (finding 9).

## References

- Bomze, I. M. (1983). "Lotka-Volterra equation and replicator dynamics: a two-dimensional
  classification." *Biological Cybernetics*, 48, 201–211.
- Cressman, R., & Tao, Y. (2014). "The replicator equation and other game dynamics."
  *Proceedings of the National Academy of Sciences*, 111(Suppl. 3), 10810–10817.
  doi:10.1073/pnas.1400823111.
- Hofbauer, J., & Sigmund, K. (1998). *Evolutionary Games and Population Dynamics.*
  Cambridge University Press. (Ch. 7, mean-fitness monotonicity.)
- Hofbauer, J., & Sigmund, K. (2003). "Evolutionary game dynamics." *Bulletin of the
  American Mathematical Society*, 40(4), 479–519.
- Maynard Smith, J. (1982). *Evolution and the Theory of Games.* Cambridge University
  Press.
- Maynard Smith, J., & Price, G. R. (1973). "The logic of animal conflict." *Nature*, 246,
  15–18.
- Nowak, M. A. (2006). *Evolutionary Dynamics: Exploring the Equations of Life.* Harvard
  University Press. (Quasispecies and replicator-mutator equations.)
- Page, K. M., & Nowak, M. A. (2002). "Unifying evolutionary dynamics." *Journal of
  Theoretical Biology*, 219(1), 93–98.
- Sandholm, W. H. (2010). *Population Games and Evolutionary Dynamics.* MIT Press.
- Weibull, J. W. (1995). *Evolutionary Game Theory.* MIT Press. (Replicator vs.
  adaptive/optimization dynamics distinction.)
