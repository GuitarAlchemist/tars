# Jules PR Watchdog

This document defines the state model and operational rules for the Jules PR watchdog, which provides fine-grained observability for cloud-agent PRs in the TARS repository.

## Goal
The watchdog classifies meaningful state transitions on Jules-created PRs to help maintainers monitor progress without manual polling or noisy notification loops.

> **Note**: For security reasons (Agent Blackbox policy), the watchdog workflow is provided as a draft in `examples/agents/jules-pr-watchdog.workflow.yml`. A maintainer must manually move it to `.github/workflows/` to activate it.

## State Model

The watchdog uses the following labels to track the state of a Jules PR:

| Label | Meaning | Transition Trigger |
|-------|---------|--------------------|
| `jules-waiting` | PR is open and waiting for an event (e.g., CI completion or human review). | Initial state on PR open. |
| `jules-updated` | Jules has pushed new commits to the PR. | `pull_request.synchronize` by Jules. |
| `needs-human-review` | CI is green and there are no pending requested changes. | `workflow_run.completed` (success) AND no "Changes Requested" review. |
| `needs-agent-revision` | A reviewer has requested changes or CI has failed. | `pull_request_review.submitted` (changes_requested) OR `workflow_run.completed` (failure). |
| `ready-to-merge` | CI is green and a human has approved the PR. | `pull_request_review.submitted` (approved) AND CI green. |
| `stale-jules-pr` | No activity from Jules or reviewers for > 3 days. | Scheduled check. |
| `duplicate-agent-pr` | A newer or more complete PR exists for the same issue. | Manual or automated detection of overlapping issue links. |

## Event Triggers

The watchdog reacts to the following GitHub events:

1.  **`pull_request.synchronize`**: Detects new commits. If the actor is Jules, transition to `jules-updated`.
2.  **`pull_request_review.submitted`**:
    *   If `approved`, and CI is green, transition to `ready-to-merge`.
    *   If `changes_requested`, transition to `needs-agent-revision`.
3.  **`workflow_run.completed`**:
    *   If the workflow is the primary CI (e.g., `build`), update state to `needs-human-review` (if success) or `needs-agent-revision` (if failure).
4.  **`issue_comment.created`**:
    *   If a maintainer uses a keyword like `/jules refresh`, force a watchdog re-evaluation.

## Governance Rules

1.  **Halt Respect**: The watchdog MUST check `governance/state/afk-halt.json`. If a halt is active, the watchdog will cease label/comment updates until the halt is lifted.
2.  **No Auto-Merge**: The watchdog is for observability only. It MUST NOT merge PRs.
3.  **Review Gating**: The watchdog MUST NOT bypass human or Demerzel review. `ready-to-merge` is an informative label, not a trigger for automated merging.
4.  **Minimal Noise**: The watchdog should only comment on meaningful transitions (e.g., CI failure or state changes that require immediate human attention). It should prefer labels over comments for routine updates.
5.  **Auditability**: All watchdog actions are performed by `github-actions[bot]`, providing a clear audit trail in the PR history.

## Hygiene Alignment

The watchdog enforces the [Cloud-Agent PR Hygiene Policy](./cloud-agent-pr-hygiene.md) by:
*   Identifying potential duplicates via issue link analysis.
*   Flagging PRs that modify out-of-scope files (e.g., documentation issues touching runtime code).
