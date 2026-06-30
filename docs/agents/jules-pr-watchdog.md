# Jules PR Watchdog

This document defines the state model and operational rules for the Jules PR watchdog, which provides fine-grained observability for cloud-agent PRs in the TARS repository.

## Goal
The watchdog classifies meaningful state transitions on Jules-created PRs to help maintainers monitor progress without manual polling or noisy notification loops.

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

## Reference Implementation

Copy the following into `.github/workflows/jules-pr-watchdog.yml` to enable the watchdog.

```yaml
name: Jules PR Watchdog

on:
  pull_request:
    types: [opened, synchronize, reopened]
  pull_request_review:
    types: [submitted]
  issue_comment:
    types: [created]
  workflow_run:
    # Replace ".NET" with the name of your primary CI workflow
    workflows: [".NET"]
    types: [completed]
  schedule:
    - cron: '0 */6 * * *' # Every 6 hours

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  watchdog:
    runs-on: ubuntu-latest
    if: |
      (github.event_name == 'pull_request' && (github.event.pull_request.user.login == 'google-labs-jules[bot]' || contains(github.event.pull_request.labels.*.name, 'jules'))) ||
      (github.event_name == 'pull_request_review' && (github.event.pull_request.user.login == 'google-labs-jules[bot]' || contains(github.event.pull_request.labels.*.name, 'jules'))) ||
      (github.event_name == 'issue_comment' && github.event.issue.pull_request && (github.event.issue.user.login == 'google-labs-jules[bot]' || contains(github.event.issue.labels.*.name, 'jules'))) ||
      (github.event_name == 'workflow_run' && (github.event.workflow_run.head_repository.full_name == github.repository)) ||
      (github.event_name == 'schedule')
    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Governance halt-gate
        id: gate
        run: |
          MARKER="governance/state/afk-halt.json"
          if [ -f "$MARKER" ]; then
            EXPIRES=$(jq -r '.expires_at // empty' "$MARKER" 2>/dev/null || true)
            HALTED=true
            if [ -n "$EXPIRES" ]; then
              NOW=$(date -u +%s)
              EXP=$(date -u -d "$EXPIRES" +%s 2>/dev/null || echo 0)
              if [ "$EXP" -gt 0 ] && [ "$NOW" -ge "$EXP" ]; then
                HALTED=false
                echo "::notice::afk-halt marker present but expired ($EXPIRES) — proceeding"
              fi
            fi
            if [ "$HALTED" = "true" ]; then
              echo "::notice::AFK watchdog halted by governance marker — skipping"
              echo "proceed=false" >> "$GITHUB_OUTPUT"
              exit 0
            fi
          fi
          echo "proceed=true" >> "$GITHUB_OUTPUT"

      - name: Evaluate and Label PR
        if: steps.gate.outputs.proceed == 'true'
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          if [ "${{ github.event_name }}" == "pull_request" ] || [ "${{ github.event_name }}" == "pull_request_review" ]; then
            PRS="${{ github.event.pull_request.number }}"
          elif [ "${{ github.event_name }}" == "issue_comment" ]; then
            PRS="${{ github.event.issue.number }}"
          elif [ "${{ github.event_name }}" == "workflow_run" ]; then
            PRS=$(gh pr list --head "${{ github.event.workflow_run.head_branch }}" --json number --jq '.[].number')
          elif [ "${{ github.event_name }}" == "schedule" ]; then
            PRS=$(gh pr list --label jules --json number --jq '.[].number')
          fi

          if [ -z "$PRS" ]; then
            echo "No PRs to evaluate."
            exit 0
          fi

          for pr in $PRS; do
            echo "--- Evaluating PR #$pr ---"

            DATA=$(gh pr view "$pr" --json author,labels,state,reviews,statusCheckRollup,updatedAt,body)
            if [ $? -ne 0 ]; then echo "Failed to fetch data for PR #$pr"; continue; fi

            AUTHOR=$(echo "$DATA" | jq -r '.author.login')
            IS_JULES=false
            if [ "$AUTHOR" == "google-labs-jules[bot]" ] || echo "$DATA" | jq -e '.labels[] | select(.name == "jules")' > /dev/null; then
              IS_JULES=true
            fi

            if [ "$IS_JULES" != "true" ]; then
              echo "Not a Jules PR, skipping."
              continue
            fi

            CURRENT_LABELS=$(echo "$DATA" | jq -r '.labels[].name')

            # 1. Check for Duplicates
            BODY=$(echo "$DATA" | jq -r '.body')
            PARENT_ISSUE=$(echo "$BODY" | grep -oEi "fixes #([0-9]+)" | head -1 | grep -oE "[0-9]+")
            IS_DUPLICATE=false
            if [ -n "$PARENT_ISSUE" ]; then
               OTHER_PRS=$(gh pr list --state open --json number,body --jq ".[] | select(.number != $pr) | select(.body | test(\"fixes #$PARENT_ISSUE\"; \"i\")) | .number")
               if [ -n "$OTHER_PRS" ]; then
                  for other in $OTHER_PRS; do
                    if [ "$other" -gt "$pr" ]; then
                      IS_DUPLICATE=true
                      break
                    fi
                  done
               fi
            fi

            # 2. Check CI status
            CI_RESULTS=$(echo "$DATA" | jq -r '.statusCheckRollup[] | .conclusion // .state' 2>/dev/null || echo "")
            CI_FAILING=false
            if echo "$CI_RESULTS" | grep -qE "FAILURE|ERROR|CANCELLED|TIMED_OUT"; then
              CI_FAILING=true
            fi
            CI_PASSING=false
            if [ -n "$CI_RESULTS" ] && ! echo "$CI_RESULTS" | grep -qvE "SUCCESS|NEUTRAL|SKIPPED|EXPECTED"; then
              if echo "$CI_RESULTS" | grep -q "SUCCESS"; then
                CI_PASSING=true
              fi
            fi

            # 3. Check Review status
            REVIEWS=$(echo "$DATA" | jq -r '.reviews[] | .state' 2>/dev/null || echo "")
            HAS_APPROVAL=false
            if echo "$REVIEWS" | grep -q "APPROVED"; then
              HAS_APPROVAL=true
            fi
            HAS_CHANGES_REQUESTED=false
            if echo "$REVIEWS" | grep -q "CHANGES_REQUESTED"; then
              HAS_CHANGES_REQUESTED=true
            fi

            # 4. Check for staleness
            UPDATED_AT=$(echo "$DATA" | jq -r '.updatedAt')
            UPDATED_TS=$(date -d "$UPDATED_AT" +%s)
            NOW_TS=$(date +%s)
            IS_STALE=false
            if [ $((NOW_TS - UPDATED_TS)) -gt $((3 * 24 * 3600)) ]; then
              IS_STALE=true
            fi

            # State Logic
            NEW_LABEL="jules-waiting"
            if [ "$IS_DUPLICATE" == "true" ]; then
              NEW_LABEL="duplicate-agent-pr"
            elif [ "$IS_STALE" == "true" ]; then
              NEW_LABEL="stale-jules-pr"
            elif [ "$HAS_APPROVAL" == "true" ] && [ "$CI_PASSING" == "true" ]; then
              NEW_LABEL="ready-to-merge"
            elif [ "$HAS_CHANGES_REQUESTED" == "true" ] || [ "$CI_FAILING" == "true" ]; then
              NEW_LABEL="needs-agent-revision"
            elif [ "$CI_PASSING" == "true" ]; then
              NEW_LABEL="needs-human-review"
            fi

            if [ "${{ github.event_name }}" == "pull_request" ] && [ "${{ github.event.action }}" == "synchronize" ] && [ "${{ github.actor }}" == "google-labs-jules[bot]" ]; then
               NEW_LABEL="jules-updated"
            fi

            # Apply label if changed
            if ! echo "$CURRENT_LABELS" | grep -q "^$NEW_LABEL$"; then
              echo "Transitioning PR #$pr to $NEW_LABEL"
              for l in jules-waiting jules-updated needs-human-review needs-agent-revision ready-to-merge stale-jules-pr duplicate-agent-pr; do
                if echo "$CURRENT_LABELS" | grep -q "^$l$"; then
                  gh pr edit "$pr" --remove-label "$l" || true
                fi
              done
              gh pr edit "$pr" --add-label "$NEW_LABEL"

              if [ "$NEW_LABEL" == "needs-agent-revision" ]; then
                 if [ "$CI_FAILING" == "true" ]; then
                   gh pr comment "$pr" --body "🤖 Watchdog: CI failed on this PR. Jules, please review the logs and push a fix."
                 elif [ "$HAS_CHANGES_REQUESTED" == "true" ]; then
                   gh pr comment "$pr" --body "🤖 Watchdog: Changes were requested by a reviewer. Jules, please address the feedback."
                 fi
              elif [ "$NEW_LABEL" == "duplicate-agent-pr" ]; then
                 gh pr comment "$pr" --body "🤖 Watchdog: This PR appears to be a duplicate of a newer PR for the same issue. Please verify and close if appropriate."
              fi
            else
              echo "PR #$pr already in state $NEW_LABEL"
            fi
          done
```
