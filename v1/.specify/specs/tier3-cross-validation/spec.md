# Tier3 Initiative: Multi-Agent Cross-Validation Loop

**Feature Branch**: `tier3/cross-validation-bootstrap`  
**Created**: 2025-10-22  
**Status**: Draft

## Vision
Elevate the Tier2 loop into a Tier3-ready pipeline where independent agents cross-validate every autonomous change. Each iteration must collect reasoning traces from at least two validator cohorts, reconcile critic verdicts, and only succeed when consensus confirms all safety and performance gates.

## User Story 1 – Dual-Agent Consensus on Every Iteration

**Priority:** P0

> *As the safeguards council, I want each Tier2 loop to capture reasoning from both Safety and Performance agents so we can enforce multi-perspective validation before accepting autonomous changes.*

### Acceptance Criteria
1. When the Tier2 loop runs, Safety and Performance agents must each emit a reasoning trace and a pass/fail verdict; the ledger records both.
2. If either agent flags `needs_review` or `fail`, the iteration is marked unsuccessful and remediation is enqueued automatically.
3. Reasoning critic evaluates a combined trace and logs its verdict; adaptive policy applies consensus tightening rules when disagreement occurs.
4. Governance ledger entries include agent verdicts, critic status, and structured remediation suggestions.

### Guardrails
- No iteration may promote changes if consensus is missing required roles.
- Reasoning traces must be persisted for critic training (adaptive memory, ledger).
- Human override remains possible via metascript command hooks.

## User Story 2 – Critic Federation and Adaptive Thresholds

**Priority:** P0

> *As the Tier3 oversight agent, I want the metascript critic to evolve thresholds automatically so cross-validation responds quickly to regressions.*

### Acceptance Criteria
1. Reasoning traces feed MetaReasoningCritic training every successful iteration.
2. When the critic detects degraded confidence, Tier2 policy toggles `RequireCriticApproval` without manual intervention.
3. Adaptive memory entries record critic threshold changes and their triggers.
4. Governance ledger includes before/after critic params per iteration.

## Non-Goals
- No automatic production deployment.
- No external API promotion; changes stay in repository.

## Implementation Notes
- Metascript staging folder: `.specify/meta/tier3/`.
- Validator reasoning traces collected via enhanced `ReasoningTraceProvider`.
- New team mapping in `.specify/teams/tier3-agents.yaml`.

```metascript
# Tier3 cross-validation blueprint
SPAWN SAFETY 1 HIERARCHICAL
SPAWN PERFORMANCE 1 HIERARCHICAL
SPAWN FEDERATION 1 FRACTAL
CONNECT safety performance reconciliation
CONNECT federation safety oversight
CONNECT federation performance oversight
METRIC safety_signal 0.88
METRIC performance_signal 0.84
METRIC critic_confidence 0.90
REPEAT cross-validation 3
```

```expectations
spawn_count=3
connection_count=3
metric.critic_confidence=0.90
max_depth=3
```

## Pilot Execution (2025-10-27)

- **Iteration**: `tier3/cross-validation-bootstrap` via `tars auto-loop`, harness success with `dotnet test Tars.sln -c Release`.
- **Ledger evidence**: `.specify/ledger/iterations/20251027020218_11b0686b0c5b4ca8bc75567a39ab2d02.json` shows `agents.safety.outcome = pass` and `agents.performance.outcome = pass` with 0.90 confidence for both roles, `capability.pass_ratio = 1.0`.
- **Critic federation**: MetaReasoningCritic model `model:53` approved the run (`safety.critic_status = accept`) with threshold `0.8075`, `safety.critic_samples = 53`, and no indicators triggered. Adaptive memory entry (latest line in `output/adaptive_memory_spec-kit.jsonl`) records the same source/threshold for future training.
- **Remediation**: No cross-agent disagreement surfaced, so no Tier3 remediation ticket was opened and Tier2 policy remained in the tightened state (`RequireConsensus = true`, `RequireCritic = true`). Subsequent validation tasks (T105/T106) will monitor disagreement scenarios explicitly.

## Remediation Drill (2025-10-27 02:28 UTC)

- **Setup**: Executed `TARS_SPEC_OVERRIDE=tier3-cross-validation TARS_FORCE_PERFORMANCE_STATUS=fail dotnet run --project src/TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli -- auto-loop` to force the Performance Benchmarker agent into a fail state while keeping the harness validation (`dotnet test`) intact.
- **Outcome**: Harness command failed because the orchestrated performance override rippled into the validation tests, creating a consensus failure with `SafetyGovernor = pass` and `PerformanceBenchmarker = fail`. Ledger evidence: `.specify/ledger/iterations/20251027022821_5085f0c42c744fe9aa85edbd239f3321.json` (note `agents.performance.*` vs. `agents.safety.*`). Adaptive memory entry timestamp `2025-10-27T02:28:21Z` captures the same critic context (`model:56`, threshold `0.717`).
- **Remediation Tracking**: Tier2Runner logged `RemediationEnqueued`, and the Tier3 task board now references the failure for manual review. This satisfies **T105**, demonstrating that the first disagreement automatically triggered the remediation channel.

## Policy Auto-tightening Verification (2025-10-27 02:29 UTC)

- **Setup**: Reset `.specify/tier2_policy.json` to `RequireConsensus=false` / `RequireCritic=false`, then repeated the forced-failure command above.
- **Observation**: Tier2Runner log: `Tier2 policy updated. RequireConsensus=True RequireCritic=True` followed by `actions=PolicyTightened`. Ledger entry `.specify/ledger/iterations/20251027022939_58fb0f47b402452bab48c497dd2b2e37.json` captures the disagreement metrics and failure status, while the adaptive memory append shows the critic context preserved for the tuning loop.
- **Result**: Verified automatic policy tightening (consensus + critic switches flipped back to `true`) after the disagreement streak, satisfying **T106**.

## Phase 2 Enhancements

- **Remediation traces**: Every agent that returns `fail`/`needs_review` now emits a dedicated reasoning trace summarizing the remediation playbook (e.g., “Initiate rollback; Schedule manual review”). These traces are stored alongside the base metascript output and surface in adaptive memory + ledger entries for auditing (**T107**).
- **Critic parameter deltas**: Governance ledger metrics now include `safety.critic_*_previous` and `*_delta` fields showing how threshold, sample count, and source changed relative to the prior iteration. Adaptive memory already stores the per-run context; this makes the before/after visible directly in the ledger summary (**T108**).
