# Jules PR Watchdog

- **Status:** Active (design + classifier script; workflow draft below)
- **Area:** Cloud Agent Observability
- **Owner:** TARS Cortex
- **Version:** 1.0.0
- **Issue:** [#132](https://github.com/GuitarAlchemist/tars/issues/132)

## Purpose

The AFK router ([`.github/workflows/afk-router.yml`](../../.github/workflows/afk-router.yml)) gets
cloud-agent work *started* — it routes a triaged `ready-for-agent` issue to Jules/Codex/Claude and
the agent opens a PR. But once Jules has a PR open, maintainers still have to poll it by hand for:

- new Jules commits,
- CI/check status changes,
- mergeability changes,
- whether a requested fix was actually addressed,
- PRs that have gone stale waiting on Jules,
- noisy duplicate PRs from fan-out.

The **Jules PR watchdog** is the finer-grained observer that fills that gap. It reacts to PR events
(and a low-frequency schedule), classifies a Jules-authored PR into **exactly one** state from the
vocabulary below, and reflects that state as a label — **without** creating comment noise and
**without** bypassing human/Demerzel review.

It is deliberately the *transition-detection/classification* layer only. The durable state store
(`afk-runs.json`, per-run JSON, JSONL events) is [#136](https://github.com/GuitarAlchemist/tars/issues/136)
and the human HTML dashboard is [#137](https://github.com/GuitarAlchemist/tars/issues/137) — both are
out of scope here. The classifier writes no durable state; it prints its classification to stdout in a
shape (`examples/agents/jules-watchdog-state.example.json`) that a future #136 consumer can persist
verbatim.

## State model

Each open, Jules-authored PR resolves to exactly one **watchdog state**. States are reflected as
mutually-exclusive labels namespaced `jules-state:<state>` (matching the repo's `worker:*` / `skill:*`
/ `agent:*` label conventions); applying one removes the others.

| State | Label | Meaning |
|-------|-------|---------|
| `jules-waiting` | `jules-state:jules-waiting` | Open, no decisive signal yet — CI running or awaiting a reviewer. |
| `jules-updated` | `jules-state:jules-updated` | Jules just pushed new commits (via `synchronize`); signals not yet resolved. |
| `needs-human-review` | `jules-state:needs-human-review` | CI green, no review yet (or a targeted fix now addressed) — ready for a human/Demerzel look. |
| `needs-agent-revision` | `jules-state:needs-agent-revision` | CI failing, or a reviewer requested changes not yet addressed — the agent must act. |
| `ready-to-merge` | `jules-state:ready-to-merge` | CI green **and** approved — awaiting a human/Demerzel merge. **The watchdog never merges.** |
| `stale-jules-pr` | `jules-state:stale-jules-pr` | Open with no activity for `STALE_DAYS` (default 3) and no terminal signal. |
| `duplicate-agent-pr` | `jules-state:duplicate-agent-pr` | A newer open Jules PR targets the same parent issue (this one is the older duplicate). |

## Transition rules

The classifier reads the currently-observable signals for the PR and applies the **first** matching
rule (priority is deliberate — a duplicate is triaged first; a hard failure or explicit
changes-requested outranks positive signals; positive signals resolve last):

1. **Duplicate fan-out** → `duplicate-agent-pr` — a *newer* open Jules PR links the same parent issue
   (`Fixes/Closes #N`). Aligns with the PR-hygiene policy "review the most recent" (see below).
2. **CI failure** → `needs-agent-revision` — checks are failing on the latest commit. **This is the
   only state that also posts a comment** (see policy below).
3. **Changes requested, not addressed** → `needs-agent-revision` — GitHub `reviewDecision` is
   `CHANGES_REQUESTED` and (for targeted PRs) the expected path is not yet touched.
4. **CI pending** → `jules-waiting` — checks still running on the latest commit.
5. **CI success + approved** → `ready-to-merge` — awaiting a human/Demerzel merge.
6. **CI success + no review** → `needs-human-review` — green and ready for a human look.
7. **Targeted fix addressed** → `needs-human-review` — a changes-requested review named a path
   (`EXPECT_PATH`) and the PR's current file set now includes it — re-review.
8. **New commits** (`synchronize`) → `jules-updated` — Jules pushed but signals aren't resolved yet.
9. **Stale** → `stale-jules-pr` — no activity for `≥ STALE_DAYS` and otherwise only `jules-waiting`.
10. **Default** → `jules-waiting`.

### How each acceptance signal is detected

- **New commits from Jules** — carried by the `pull_request.synchronize` trigger (no durable diff
  store needed; that would be #136). `EVENT_NAME=synchronize` ⇒ candidate `jules-updated`.
- **CI/check status changes** — `gh pr checks` classified into `failure | pending | success | none`.
- **Mergeability changes** — the PR's `mergeable` field is recorded in the classification's
  `signals.mergeable` (`MERGEABLE | CONFLICTING | UNKNOWN`) for downstream consumers.
- **Requested fix addressed (targeted PRs)** — optional `EXPECT_PATH`: on a `CHANGES_REQUESTED`
  review, the fix counts as addressed only if the PR's current files include that path.
- **Stale PRs** — the PR's own `updatedAt` vs. `STALE_DAYS` (no state store required).
- **Duplicates** — parent issue parsed from the PR body; the open Jules PR with the highest number
  wins, older ones become `duplicate-agent-pr`.

## Labels / comments policy

- **Labels change only on a genuine transition.** Re-running on an unchanged state is a no-op — the
  target label is already present, so nothing is added and nothing else is stripped. No poll-spam.
- **Comments are posted on exactly one state: `needs-agent-revision` caused by CI failure.** That is
  the single critical transition where a human/agent must act, per the issue's minimal-noise
  requirement. The comment is idempotent: a hidden `<!-- jules-watchdog:ci-fail:<sha> -->` marker
  means at most one comment per failing head SHA, so re-runs and other triggers never re-spam.
- **No comment for any other state** — `ready-to-merge`, `needs-human-review`, `stale-jules-pr`,
  `duplicate-agent-pr`, etc. are conveyed by labels only.
- **Only Jules-authored PRs are touched.** Recognition is by author login (default substring
  `jules`, case-insensitive) or the routing `jules` label. Any other PR is left completely alone.

## Governance notes

- **Halt gate.** The watchdog reads `governance/state/afk-halt.json` exactly as the afk-router does:
  a present, unexpired marker means the watchdog **stands down** and makes no labels or comments.
  (`expires_at` in the past ⇒ treated as expired ⇒ proceed.) See ADR 0004.
- **Never merges, never approves.** `ready-to-merge` is an *observation*. Merging stays a
  human/Demerzel action — branch protection and review-gating hold by default.
- **Never bypasses review.** The watchdog only labels and (once) comments; it changes no review
  state and requests no approvals.
- **Minimal permissions.** The workflow needs `contents: read`, `pull-requests: write` (labels +
  the single comment), and `checks: read` / `actions: read` (CI status). Nothing else.

### Interaction with the already-merged policies

- **PR hygiene** ([`cloud-agent-pr-hygiene.md`](./cloud-agent-pr-hygiene.md)) — the
  `duplicate-agent-pr` rule mechanizes hygiene rule #1 ("review the most recent; older duplicates
  are candidates to close"). The watchdog only *flags* the older duplicate; a human still decides to
  close it, using the documented duplicate-close template.
- **Backpressure / locks** ([`backpressure-decision.example.json`](../../examples/agents/backpressure-decision.example.json)) —
  the watchdog is read-mostly and event-driven; it acquires no locks and opens no PRs, so it never
  competes for the global WIP limit. Its labels are a cheap input a backpressure decision can read
  (e.g. treat `ready-to-merge` PRs as draining the queue, `stale-jules-pr` as recoverable slots).
- **AFK Live Agent Board** ([`afk-live-agent-board.md`](./afk-live-agent-board.md)) — the watchdog's
  state vocabulary is the same one the board's per-run `state` field uses, so a #136 consumer can map
  a classification straight onto a run record without translation.

## Non-goals

- No autonomous merge. No approvals.
- No replacement for CI — it reads CI, it doesn't gate on it.
- No high-frequency busy-loop polling — reactive events plus a low-frequency schedule.
- No comment for unchanged or non-critical state.
- No durable state artifacts (#136) and no HTML dashboard (#137).

## Files

- [`.github/scripts/jules-pr-watchdog.sh`](../../.github/scripts/jules-pr-watchdog.sh) — the classifier.
- [`examples/agents/jules-watchdog-state.example.json`](../../examples/agents/jules-watchdog-state.example.json) —
  the machine-readable classification the script prints.
- Workflow draft — below (see note on wiring).

## Workflow draft

> **Wiring note.** This PR was opened by the Claude GitHub App, whose permissions do not allow
> committing under `.github/workflows/`. A maintainer should add the file below as
> `.github/workflows/jules-pr-watchdog.yml`. It is otherwise ready to run and calls only the
> committed classifier script.

```yaml
name: Jules PR watchdog

# Finer-grained observability for Jules-authored PRs (issue #132). Reacts to PR events and a
# low-frequency schedule, classifies each Jules PR into one jules-state:* label, and comments only
# on the critical CI-failure transition. Never merges; never bypasses review; halt-gated on
# governance/state/afk-halt.json. See docs/agents/jules-pr-watchdog.md.

on:
  pull_request_target:
    types: [synchronize, opened, reopened, ready_for_review]
  pull_request_review:
    types: [submitted]
  workflow_run:
    workflows: [".NET"]
    types: [completed]
  schedule:
    - cron: '23 */6 * * *'   # low-frequency sweep for staleness/duplicates
  workflow_dispatch:
    inputs:
      pr:
        description: PR number to classify
        required: true

permissions:
  contents: read
  pull-requests: write
  checks: read
  actions: read

concurrency:
  # One classification per PR at a time; never cancel an in-flight run.
  group: jules-watchdog-${{ github.event.pull_request.number || github.event.workflow_run.id || github.event.inputs.pr || github.run_id }}
  cancel-in-progress: false

jobs:
  classify:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Checkout (read the governance halt marker + the classifier script)
        uses: actions/checkout@v4

      # Event-driven paths: a single PR number is available directly.
      - name: Classify the triggering PR
        if: ${{ github.event_name == 'pull_request_target' || github.event_name == 'pull_request_review' || github.event_name == 'workflow_dispatch' }}
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          PR: ${{ github.event.pull_request.number || github.event.inputs.pr }}
          EVENT_NAME: ${{ github.event.action || github.event_name }}
        run: bash .github/scripts/jules-pr-watchdog.sh

      # workflow_run gives a head SHA, not a PR — resolve the open PR(s) for that SHA.
      - name: Classify the PR(s) for a completed check run
        if: ${{ github.event_name == 'workflow_run' }}
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          HEAD_SHA: ${{ github.event.workflow_run.head_sha }}
          EVENT_NAME: workflow_run
        run: |
          set -euo pipefail
          for PR in $(gh pr list --repo "$REPO" --state open --search "$HEAD_SHA" \
                        --json number --jq '.[].number'); do
            PR="$PR" bash .github/scripts/jules-pr-watchdog.sh
          done

      # Scheduled sweep: re-classify every open PR (catches staleness + duplicates).
      - name: Scheduled sweep of open PRs
        if: ${{ github.event_name == 'schedule' }}
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          EVENT_NAME: schedule
        run: |
          set -euo pipefail
          for PR in $(gh pr list --repo "$REPO" --state open --json number --jq '.[].number'); do
            PR="$PR" bash .github/scripts/jules-pr-watchdog.sh
          done
```
