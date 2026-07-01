# AFK Live Agent Board

- **Status:** Active
- **Area:** Cloud Agent Observability
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Purpose

The AFK Live Agent Board is the repo-native, machine-readable source of truth for tracking parallel Away From Keyboard (AFK) work executed by cloud agents such as Jules or Codex.

This layer allows humans and agents to observe state transitions across multiple issues and PRs without opening each one manually. It defines standard JSON and JSONL artifacts for consumption by GitHub Actions, cloud agents, and local tools.

## Artifacts

### 1. `afk-runs.json`

This file serves as the index of all currently tracked AFK runs.

**Location:** `governance/agents/live/afk-runs.json`

**Shape:**

```json
{
  "version": "1.0",
  "last_updated": "2026-06-28T13:00:00Z",
  "active_runs": [
    "afk-20260628-issue-115-jules"
  ]
}
```

### 2. Per-Run State (`runs/<run-id>.json`)

Each active or recent AFK unit is represented as a distinct run.

**Location:** `governance/agents/live/runs/<run-id>.json`

**Shape:**

```json
{
  "run_id": "afk-20260628-issue-115-jules",
  "repo": "GuitarAlchemist/tars",
  "issue": 115,
  "agent": "jules",
  "pr": 129,
  "state": "needs-agent-revision",
  "risk": "medium",
  "last_signal_at": "2026-06-28T13:02:41Z",
  "summary": "PR exists but needs revision before merge.",
  "evidence": [
    {
      "type": "review_comment",
      "url": "https://github.com/GuitarAlchemist/tars/pull/129#pullrequestreview-4587561762",
      "finding": "Skill path should load <skill>.skill.md"
    }
  ],
  "next_action": "Agent should patch workflow path handling and avoid overwriting merged skill docs.",
  "stop_condition": "Do not merge until CI is green and human review passes."
}
```

### 3. Event Log (`events/<yyyy-mm-dd>.jsonl`)

An append-only event log tracking state transitions for observability and watchdog integration.

**Location:** `governance/agents/live/events/<yyyy-mm-dd>.jsonl`

**Shape:**

```jsonl
{"ts":"2026-06-28T06:17:57Z","repo":"GuitarAlchemist/tars","issue":115,"agent":"jules","event":"pr_opened","pr":129,"state":"pr-opened"}
{"ts":"2026-06-28T13:02:41Z","repo":"GuitarAlchemist/tars","issue":115,"agent":"human","event":"review_requested_changes","pr":129,"state":"needs-agent-revision"}
```

## State Vocabulary

The board supports the following standard states for AFK work:

- `queued`: Agent task is queued but has not started.
- `delegated`: Issue has been formally delegated to an agent.
- `agent-working`: Agent has acknowledged the task and is actively working.
- `pr-opened`: Agent has opened a pull request.
- `needs-agent-revision`: Human reviewer requested changes from the agent.
- `needs-human-review`: Agent PR is complete and awaiting human review.
- `ci-failing`: Agent PR failed CI checks.
- `ci-green`: Agent PR passed CI checks.
- `stale`: Agent has not provided a signal in the expected timeout window.
- `duplicate`: Task or PR was marked as a duplicate.
- `blocked`: Task is blocked by an external dependency or governance rule.
- `done`: PR was merged or issue was completed successfully.

## Governance & Safety Constraints

To ensure the safety of parallel AFK work, the following constraints strictly govern the Live Agent Board:

1. **No Secrets:** Never store API keys, raw private logs, or prompt contents that may contain secrets.
2. **No Auto-Merge:** The state layer tracks merge readiness but must **never** auto-merge PRs.
3. **No Bypass of Human/Demerzel Review:** The board reflects governance gates; it cannot override them.
4. **Halt Marker Enforcement:** Writing to or progressing states must respect `governance/state/afk-halt.json`. If a halt marker is present, auto-delegation and agent-driven transitions must pause.
5. **Deterministic Event Writes:** Keep event writes low-noise. Only record meaningful state transitions such as CI failures or reviews, not internal agent monologues.
6. **Reviewable Artifacts:** All generated state artifacts (`.json`, `.jsonl`) must be easily reviewable in a PR.
