# Jules PR Watchdog

## Purpose

The Jules PR watchdog makes AFK cloud-agent work easier to observe after delegation. The existing `jules-auto-delegate.yml` workflow starts work when an issue receives `ready-for-agent`; this watchdog tracks the PR that Jules opens and classifies useful state transitions.

The watchdog is conservative:

- no automatic merge;
- no automatic approval;
- no PR-branch checkout;
- comments only when state changes;
- labels are used as the primary state surface.

## Trigger model

The first implementation is event-first with a low-frequency scheduled fallback.

Candidate triggers:

- `pull_request_target` for PR metadata changes;
- `pull_request_review` for submitted reviews;
- `issue_comment` for PR conversation changes;
- `workflow_run` for completed CI workflows;
- `schedule` every 30 minutes;
- `workflow_dispatch` for manual refresh.

The workflow performs metadata inspection only.

## Jules PR detection

A PR is treated as a Jules PR when its body contains the standard Jules footer:

```text
PR created automatically by Jules
```

This keeps the watchdog scoped to AFK cloud-agent PRs.

## State labels

The watchdog owns these mutually exclusive labels:

| Label | Meaning |
| --- | --- |
| `jules-waiting` | The PR exists but still needs agent, CI, or review movement. |
| `jules-updated` | The PR changed and needs a fresh look. |
| `needs-human-review` | The PR is ready for maintainer review. |
| `needs-agent-revision` | A detectable issue remains and the agent should revise it. |
| `ready-to-merge` | Checks look green and no watchdog blocker is detected; a human still decides. |
| `stale-jules-pr` | The PR has not moved for the configured stale window. |
| `duplicate-agent-pr` | The PR appears superseded by another agent PR. |

The watchdog removes older state labels before applying a new state label.

## Transition policy

The watchdog comments only when the state label changes. This avoids noisy polling while keeping an audit trail.

Examples:

- `jules-waiting -> needs-agent-revision` when checks fail or a targeted issue is detected;
- `jules-waiting -> jules-updated` when a new commit arrives;
- `jules-updated -> ready-to-merge` when checks are green and no watchdog blocker remains;
- `jules-waiting -> stale-jules-pr` when no meaningful movement is detected within the stale window.

## Targeted check: skill file suffix

For the #115/#129 line of work, the watchdog can detect one specific known issue: workflow code that looks for `docs/agents/skills/<skill>.md` while the merged skill files use `docs/agents/skills/<skill>.skill.md`.

This check only sets `needs-agent-revision`; it does not modify the PR.

## Boundaries

The watchdog is an observability helper, not a maintainer.

It does not:

- merge PRs;
- approve PRs;
- change reviews;
- change agent branches;
- start new agent tasks;
- run as a high-frequency poller.

## Relationship to the AFK loop

The watchdog complements the dispatch queue and PR hygiene policies:

```text
ready-for-agent
  -> Jules delegation
  -> PR opened
  -> watchdog classification
  -> maintainer review
  -> merge or revise
```
