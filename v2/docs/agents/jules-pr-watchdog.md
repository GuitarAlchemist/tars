# Jules PR Watchdog

- **Status:** Draft
- **Area:** Runtime / AFK Loop / Cloud-Agent Observability
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Purpose
The Jules PR watchdog is a lightweight observability mechanism for cloud-agent work. It provides finer granularity than manual polling or hourly external checks, classifying meaningful state transitions on Jules-created PRs without creating comment noise, bypassing human review, or auto-merging.

## Context
The AFK delegation workflow invokes Jules, which then opens a PR. After this, maintainers have to manually poll to check for:
- New commits pushed by Jules.
- Changes in CI/check status.
- Mergeability updates.
- Whether a requested fix has been addressed.
- Stale PRs waiting on Jules.
- Duplicate agent PRs resulting from fan-out.

This watchdog bridges the gap between Jules opening a PR and its eventual merge or closure, maintaining observability into the PR's lifecycle.

## State Model
The watchdog uses labels to track the state of a Jules PR. The following states are defined:

*   `jules-waiting`: The PR is open, and Jules is expected to act (e.g., pushing requested changes).
*   `jules-updated`: Jules has pushed new commits or updated the PR.
*   `needs-human-review`: The PR is ready for a maintainer or Demerzel tribunal review.
*   `needs-agent-revision`: A reviewer has requested changes from Jules.
*   `ready-to-merge`: The PR has passed human review and CI, and is ready for manual merge.
*   `stale-jules-pr`: The PR has been inactive for a defined period (e.g., 7 days) and may need re-delegation or closure.
*   `duplicate-agent-pr`: The PR is identified as a duplicate of another active PR for the same issue.

## Triggers
The watchdog reacts to the following GitHub event triggers:

*   `pull_request.synchronize`: Triggered when Jules pushes new commits.
*   `pull_request.edited`: Triggered when the PR base branch changes or mergeability status is computed.
*   `pull_request_review.submitted`: Triggered when a human or Demerzel tribunal submits a review.
*   `issue_comment.created`: Triggered when a comment is added to the PR.
*   `workflow_run.completed`: Triggered when CI/checks complete (e.g., `dotnet build && dotnet test`).
*   `schedule`: A low-frequency cron job (e.g., daily) to detect stale PRs.

## Transition Rules
*   **New Jules Commit (`pull_request.synchronize`):**
    *   Transition to: `jules-updated`.
    *   Action: Add `jules-updated` label, remove `jules-waiting`, `needs-human-review`, `needs-agent-revision`.
*   **Human Review Requests Changes (`pull_request_review.submitted`):**
    *   Transition to: `needs-agent-revision` (and subsequently `jules-waiting`).
    *   Action: Add `needs-agent-revision` label, invoke Jules for revision.
*   **Human Review Approves (`pull_request_review.submitted`):**
    *   Condition: CI must also be green.
    *   Transition to: `ready-to-merge`.
    *   Action: Add `ready-to-merge` label, remove `needs-human-review`.
*   **CI Completes (`workflow_run.completed`):**
    *   If failed: Stay in current state (or add an alert label).
    *   If passed and in `jules-updated`: Transition to `needs-human-review`.
*   **Mergeability Status Update (`pull_request.edited` or API polling):**
    *   If GitHub API flags the PR as having merge conflicts (`mergeable: false`), the watchdog labels it `needs-agent-revision` (or adds a specific `has-conflicts` label) and comments (once) requesting Jules to rebase/fix.
*   **Targeted File Fix Addressed (Event hook/Check run):**
    *   When a reviewer requests a specific file edit via review comment, the watchdog scans the next `pull_request.synchronize` commit diff. If the requested path is modified, it transitions the PR directly back to `needs-human-review` (pending CI), noting the fix was attempted.
*   **Scheduled Check (`schedule`):**
    *   If PR inactive for > 7 days: Transition to `stale-jules-pr`.

## Labels and Comments Policy
*   **Label-Driven:** State is primarily tracked via GitHub labels to avoid comment noise.
*   **Meaningful Transitions Only:** Comments are only posted on critical state transitions (e.g., entering `stale-jules-pr`, or when a PR is marked `duplicate-agent-pr` with a link to the superseding PR). No comments for every poll or minor update.

## Governance Rules
*   **No Autonomous Merge:** The watchdog **never** merges automatically. Merging remains a human action.
*   **No Bypassing Reviews:** Demerzel and human reviews remain mandatory. The watchdog only surfaces the state to make it observable.
*   **Respects Halt Marker:** The watchdog respects the `governance/state/afk-halt.json` marker. If the marker is active, the watchdog will not trigger Jules for revisions or update states that would imply active agent work.

## Integration
*   **PR Hygiene Policy:** The watchdog assists the PR hygiene policy by automatically classifying `duplicate-agent-pr` or `stale-jules-pr`.
*   **Backpressure and Locks:** The watchdog's state labels provide visibility into the active count of Jules PRs, aiding the admission controller in making backpressure decisions.
