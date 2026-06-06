# State-Space Lens for Agent Meshes: Observability, Controllability, Stability

## Goal
Frame a mesh of TARS agents as a state-space system so we can reason about observability (can we infer internal state?), controllability/commandability (can we steer it?), and stability (will it converge or drift). Outcome: practical tests and instrumentation hooks to decide when to reroute, damp, or reconfigure agents.

## System Model
- **State (x):** Vector of per-agent latent variables and shared substrates: belief graph embeddings, budget levels, tool health, queue depths, recent error modes, epistemic confidence.
- **Input (u):** Human commands, curriculum tasks, tool invocations allowed/blocked, model choices, budget allocations.
- **Output (y):** Task results, trace summaries, belief updates, budget deltas, tool metrics, watchdog alerts.
- **Dynamics (A/B):** Approximated via learned linear operators over aggregated telemetry (e.g., VAR over state slices) or piecewise-linear regimes (high-load vs. nominal). Nonlinear policy lives in agents; we fit linear surrogates for analysis.
- **Observation (C/D):** What our tracing and metrics expose (C) plus direct feedthrough from inputs to outputs (D) such as immediate tool errors.

## Observability
- **State coverage:** Ensure tracing captures enough dimensions: budgets, queue depths, tool breaker states, belief confidence, grammar violations, loop detector flags.
- **Rank test (pragmatic):** Fit a local linear surrogate (LDS/VAR) over recent windows; compute observability matrix rank. If rank deficient, expand telemetry (add counters, probes).
- **Sparse probes:** Inject known perturbations (small budget reductions, synthetic tasks) and see if state estimates adjust; if not observed, add probes or sensors.
- **Gaps to watch:** Hidden contention (locks, GPU queue), silent tool degradation, retrieval drift. Add explicit metrics to surface them.

## Controllability / Commandability
- **Inputs that matter:** Budget knobs, allowed tool sets, routing policies (fan-out, planner depth), model temperature/top-p, retrieval parameters, circuit-breaker thresholds.
- **Rank test (pragmatic):** On the same surrogate model, compute controllability matrix rank; if deficient, add actuators (new control knobs) or coarser policies (pause submeshes, freeze agents).
- **Control surfaces:** Fast (per-run): budget clamps, planner depth caps, tool blacklists. Slow (per-shift): curriculum sampling weights, model selection, grammar tightening/loosening.
- **Actuation safety:** Changes go through governance: budget floor/ceil, loop watchdog, entropy guard; rollback on instability.

## Stability
- **Indicators:** Divergence between replay and live traces, rising entropy of context vs. belief anchors, oscillating tool breakers, growing queue depths, increased variance in outcomes.
- **Lyapunov-lite:** Track monotone quantities: error count, budget debt, grammar violations. Require non-increasing trends under control actions; revert if monotonicity breaks.
- **Damping actions:** Lower fan-out, lower temperature, tighten budgets, reduce planner depth, force summarization, blacklist flaking tools, slow curriculum.

## Instrumentation & Tests
- **Telemetry schema:** Standardize state vector slices in traces: `budgets`, `queues`, `tool_health`, `belief_conf`, `grammar_violations`, `loop_flags`.
- **Probing tests:** Scheduled micro-perturbations (budget nudges, synthetic traceable tasks) to measure response; use them to estimate A/B/C matrices.
- **Regression checks:** Observability rank must not drop release-to-release; controllability rank must stay above threshold on surrogate; stability indicators must stay within bounds.

## Integration Points in TARS
- **Trace Recorder:** Source of state/output sequences; include structured state slices.
- **Mock Tool framework:** Safe environment to probe without side effects.
- **BudgetGovernor / Circuit Breakers / Loop Watchdog:** Primary actuators for stabilization.
- **Curriculum Agent:** Acts as input scheduler; can inject probes and vary excitation for identifiability.
- **Epistemic Governor:** Flags entropy/drift; can request damping.

## Roadmap
1) Define telemetry schema for state slices and ensure Trace Recorder emits it.
2) Add scheduled probes in self-play to estimate surrogate A/B/C/D weekly.
3) Compute pragmatic observability/controllability ranks; alert on regression.
4) Wire damping/rollback policies to stability indicators (entropy, queues, breakers).
5) Report a simple dashboard: ranks, probe responses, stability signals, recent rollbacks.
