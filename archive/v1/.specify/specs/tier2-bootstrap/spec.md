# Tier2 Bootstrap: Autonomous Release Notes Maintenance

**Feature Branch**: `tier2/bootstrap-release-notes`
**Created**: 2025-09-07
**Status**: Draft

## Vision
Establish a minimal-but-complete Tier2 feedback loop that demonstrates end-to-end autonomous improvement on a safe target: keeping `docs/RELEASE_NOTES.md` fresh after each successful iteration. The loop must execute a metascript, apply AI-authored edits safely, and finish with an auditable harness run plus governance logging.

## User Story 1 – Tier2 Loop Validates Release Notes

**Priority:** P0

> *As the governance team, I want Tier2 to update the release notes automatically whenever a Spec Kit iteration succeeds, so that audit trails always include fresh human-readable summaries.*

### Acceptance Criteria
1. Given a pending Tier2 bootstrap task, when the Tier2 auto-loop runs, then the metascript `Tier2ReleaseNotes` is invoked before harness validation.
2. Given the metascript runs, when it calls Ollama-powered analysis, then the suggested changes are applied to `docs/RELEASE_NOTES.md` and persisted to disk.
3. Given the release notes were modified, when the harness `dotnet test Tars.sln -c Release` completes successfully, then the iteration status is recorded in the governance ledger with a link to the modified file.
4. Given the iteration completes (pass or fail), when the adaptive memory entry is written, then it includes the release-notes metascript outcome and the updated Tier2 policy snapshot.

```metascript
# Fractal bootstrap to satisfy dynamic closure requirements.
SPAWN QRE 2 HIERARCHICAL
SPAWN ML 1 FRACTAL
CONNECT leader agent-1 directive
CONNECT agent-1 agent-2 support
METRIC adaptation 0.80
METRIC documentation 0.92
REPEAT release-cycle 3
```

```expectations
rules=7
max_depth=3
spawn_count=2
connection_count=2
metric.documentation=0.92
```

### Non-Goals / Guardrails
- Do **not** promote changes if the harness fails; policy tightening must be triggered instead.
- Do **not** commit release notes automatically; humans will review the diff.
- Metascript must operate only on `docs/RELEASE_NOTES.md`, leaving other files untouched.

## Implementation Notes
- Metascript lives at `.specify/meta/tier2/release-notes.tars`.
- Team mapping is defined in `.specify/teams/core-team.yaml`; `release-scribe` and `qa-sentinel` share responsibility for release-note spikes/stories.
- Harness command is the default Tier2 validation: `dotnet test Tars.sln -c Release`.
