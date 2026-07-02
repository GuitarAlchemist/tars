#!/usr/bin/env bash
# jules-pr-watchdog.sh — classify a Jules-authored PR into a single watchdog state (issue #132).
#
# The AFK router (.github/workflows/afk-router.yml) gets cloud-agent work started, but once
# Jules opens a PR maintainers still have to poll it manually for new commits, CI changes,
# mergeability, requested-fix follow-through, staleness, and fan-out duplicates. This script
# is the finer-grained observer: given one PR, it reads the currently-observable signals,
# classifies the PR into exactly one state from the #132 vocabulary, prints that classification
# as JSON (machine-readable), and reflects it as a namespaced `jules-state:<state>` label.
#
# It is deliberately conservative:
#   - It NEVER merges and NEVER approves — it only labels, and comments on one transition.
#   - It comments ONLY on the critical CI-failure transition, once per failing head SHA
#     (idempotent via a hidden marker), per the issue's minimal-noise requirement.
#   - It NEVER bypasses Demerzel/human review — `ready-to-merge` is an observation, not an act.
#   - It respects the governance halt marker `governance/state/afk-halt.json` (present+unexpired
#     => the watchdog stands down and makes no changes), mirroring the afk-router gate.
#   - It only touches PRs it recognizes as Jules-authored (see JULES_AUTHOR_PATTERN / `jules`).
#
# State artifacts (#136) and the HTML dashboard (#137) are intentionally out of scope: this
# script is transition detection/classification only. It writes no durable state; "new commit"
# is carried by the `pull_request.synchronize` trigger and staleness by the PR's own updatedAt.
#
# Environment:
#   REPO         owner/name of the repo                                   (required)
#   PR           pull-request number to classify                          (required)
#   GH_TOKEN     token for `gh` (needs pull-requests + issues: write)      (required)
#   EVENT_NAME   triggering event ("synchronize"|"review"|"schedule"|...)  (optional, hints only)
#   STALE_DAYS   days of no PR activity before `stale-jules-pr`            (optional, default 3)
#   EXPECT_PATH  if a changes-requested review targeted this path, the     (optional)
#                latest commits must touch it to count the fix as addressed
#   JULES_AUTHOR_PATTERN  case-insensitive login substring marking Jules   (optional, default "jules")
#
# Exit codes: 0 on success or benign no-op (halt gate, not-a-jules-PR, PR closed). Non-zero
# only on a genuine failure (missing required env, `gh` failure on a required call).
set -euo pipefail

: "${REPO:?REPO is required}"
: "${PR:?PR is required}"
STALE_DAYS="${STALE_DAYS:-3}"
EXPECT_PATH="${EXPECT_PATH:-}"
EVENT_NAME="${EVENT_NAME:-}"
JULES_AUTHOR_PATTERN="${JULES_AUTHOR_PATTERN:-jules}"

# All watchdog state labels, namespaced under `jules-state:` (matches the repo's
# `worker:*` / `skill:*` / `agent:*` label conventions). This is also the mutually
# exclusive set the script rotates through — applying one removes the others.
STATE_LABELS=(
  "jules-state:jules-waiting"
  "jules-state:jules-updated"
  "jules-state:needs-human-review"
  "jules-state:needs-agent-revision"
  "jules-state:ready-to-merge"
  "jules-state:stale-jules-pr"
  "jules-state:duplicate-agent-pr"
)

log() { echo "::notice::[jules-watchdog] $*"; }

# --- Governance halt gate (mirrors .github/workflows/afk-router.yml) ----------------------
# Present + unexpired marker => the watchdog stands down and makes no labels/comments.
halted() {
  local marker="governance/state/afk-halt.json"
  [ -f "$marker" ] || return 1
  local expires now exp
  expires=$(jq -r '.expires_at // empty' "$marker" 2>/dev/null || true)
  if [ -n "$expires" ]; then
    now=$(date -u +%s)
    exp=$(date -u -d "$expires" +%s 2>/dev/null || echo 0)
    if [ "$exp" -gt 0 ] && [ "$now" -ge "$exp" ]; then
      log "afk-halt marker present but expired ($expires) — watchdog proceeding"
      return 1
    fi
  fi
  return 0
}

if halted; then
  log "AFK governance halt is active (governance/state/afk-halt.json) — watchdog standing down."
  exit 0
fi

# --- Load the PR --------------------------------------------------------------------------
PR_JSON=$(gh pr view "$PR" --repo "$REPO" \
  --json number,state,isDraft,author,labels,headRefOid,updatedAt,mergeable,reviewDecision,body,files \
  2>/dev/null || echo '')
if [ -z "$PR_JSON" ]; then
  log "PR #$PR not found or not readable — nothing to do."
  exit 0
fi

STATE=$(jq -r '.state' <<<"$PR_JSON")
if [ "$STATE" != "OPEN" ]; then
  log "PR #$PR is $STATE (not OPEN) — watchdog only classifies open PRs."
  exit 0
fi

AUTHOR=$(jq -r '.author.login // ""' <<<"$PR_JSON")
HEAD_SHA=$(jq -r '.headRefOid // ""' <<<"$PR_JSON")
UPDATED_AT=$(jq -r '.updatedAt // ""' <<<"$PR_JSON")
MERGEABLE=$(jq -r '.mergeable // "UNKNOWN"' <<<"$PR_JSON")
REVIEW_DECISION=$(jq -r '.reviewDecision // ""' <<<"$PR_JSON")
LABELS=$(jq -r '[.labels[].name] | join(" ")' <<<"$PR_JSON")

# --- Jules-authorship guard ----------------------------------------------------------------
# We recognize a Jules PR by author login (default substring "jules", case-insensitive) OR by
# the routing `jules` label the afk-router applies. Anything else is left completely untouched.
is_jules=false
shopt -s nocasematch
[[ "$AUTHOR" == *"$JULES_AUTHOR_PATTERN"* ]] && is_jules=true
shopt -u nocasematch
grep -qw "jules" <<<"$LABELS" && is_jules=true
if [ "$is_jules" != "true" ]; then
  log "PR #$PR (author '$AUTHOR') is not a Jules-authored PR — leaving it untouched."
  exit 0
fi

# --- Signal: CI / checks -------------------------------------------------------------------
# failure | pending | success | none. `gh pr checks` exits non-zero when checks fail/pending,
# so we capture output without letting `set -e` abort.
CHECKS=$(gh pr checks "$PR" --repo "$REPO" 2>/dev/null || true)
ci="none"
if [ -n "$CHECKS" ]; then
  if grep -qiE $'\tfail' <<<"$CHECKS" || grep -qiw "fail" <<<"$CHECKS"; then
    ci="failure"
  elif grep -qiE "pending|in_progress|queued|expected" <<<"$CHECKS"; then
    ci="pending"
  else
    ci="success"
  fi
fi

# --- Signal: review decision ---------------------------------------------------------------
# GitHub's reviewDecision: APPROVED | CHANGES_REQUESTED | REVIEW_REQUIRED | "" (none yet).
review="none"
case "$REVIEW_DECISION" in
  APPROVED)          review="approved" ;;
  CHANGES_REQUESTED) review="changes_requested" ;;
esac

# --- Signal: requested-fix addressed (optional, targeted PRs) ------------------------------
# When a change was requested AND a specific path was named (EXPECT_PATH), treat the fix as
# "addressed" only if the PR's current file set touches that path. Lets a targeted PR move from
# needs-agent-revision back to needs-human-review once the agent pushes the expected change.
fix_addressed="n/a"
if [ -n "$EXPECT_PATH" ] && [ "$review" = "changes_requested" ]; then
  if jq -e --arg p "$EXPECT_PATH" '.files[]?.path | select(. == $p)' <<<"$PR_JSON" >/dev/null 2>&1; then
    fix_addressed="yes"
  else
    fix_addressed="no"
  fi
fi

# --- Signal: duplicate fan-out -------------------------------------------------------------
# If this PR links a parent issue (Fixes/Closes #N) and a *newer* open Jules PR links the same
# issue, this one is the older duplicate (per the PR-hygiene policy: review the most recent).
duplicate=false
superseded_by=""
LINKED_ISSUE=$(jq -r '.body // ""' <<<"$PR_JSON" \
  | grep -ioE "(clos|fix|resolv)[a-z]* #[0-9]+" | grep -oE "[0-9]+" | head -n1 || true)
if [ -n "$LINKED_ISSUE" ]; then
  while read -r other; do
    [ -z "$other" ] && continue
    [ "$other" -le "$PR" ] && continue
    OA=$(gh pr view "$other" --repo "$REPO" --json author,body --jq \
      '{a:(.author.login//""),b:(.body//"")}' 2>/dev/null || echo '')
    [ -z "$OA" ] && continue
    OL=$(jq -r '.a' <<<"$OA")
    OB=$(jq -r '.b' <<<"$OA")
    shopt -s nocasematch; oj=false; [[ "$OL" == *"$JULES_AUTHOR_PATTERN"* ]] && oj=true; shopt -u nocasematch
    OTHER_ISSUE=$(grep -ioE "(clos|fix|resolv)[a-z]* #[0-9]+" <<<"$OB" | grep -oE "[0-9]+" | head -n1 || true)
    if [ "$oj" = "true" ] && [ "$OTHER_ISSUE" = "$LINKED_ISSUE" ]; then
      duplicate=true; superseded_by="$other"; break
    fi
  done < <(gh pr list --repo "$REPO" --state open --json number --jq '.[].number' 2>/dev/null || true)
fi

# --- Signal: staleness ---------------------------------------------------------------------
age_days=0
if [ -n "$UPDATED_AT" ]; then
  upd=$(date -u -d "$UPDATED_AT" +%s 2>/dev/null || echo 0)
  now=$(date -u +%s)
  [ "$upd" -gt 0 ] && age_days=$(( (now - upd) / 86400 ))
fi

# --- Signal: new commits -------------------------------------------------------------------
# No durable state store (that's #136), so "new commit" is carried by the synchronize event.
new_commits=false
[ "$EVENT_NAME" = "synchronize" ] && new_commits=true

# --- Classify (first match wins) -----------------------------------------------------------
# Priority is deliberate: a duplicate is triaged before anything else; a hard CI failure or an
# explicit changes-requested outranks positive signals; positive signals resolve last.
state=""
reason=""
if [ "$duplicate" = "true" ]; then
  state="duplicate-agent-pr"
  reason="A newer open Jules PR (#$superseded_by) targets the same issue (#$LINKED_ISSUE)."
elif [ "$ci" = "failure" ]; then
  state="needs-agent-revision"
  reason="CI is failing on the latest commit — the agent must push a fix."
elif [ "$review" = "changes_requested" ] && [ "$fix_addressed" != "yes" ]; then
  state="needs-agent-revision"
  reason="A reviewer requested changes that are not yet addressed."
elif [ "$ci" = "pending" ]; then
  state="jules-waiting"
  reason="CI is still running on the latest commit."
elif [ "$ci" = "success" ] && [ "$review" = "approved" ]; then
  state="ready-to-merge"
  reason="CI is green and the PR is approved — awaiting a human/Demerzel merge (watchdog never merges)."
elif [ "$ci" = "success" ] && [ "$review" = "none" ]; then
  state="needs-human-review"
  reason="CI is green with no review yet — ready for a human/Demerzel look."
elif [ "$review" = "changes_requested" ] && [ "$fix_addressed" = "yes" ]; then
  state="needs-human-review"
  reason="The requested change to '$EXPECT_PATH' now appears in the PR — re-review."
elif [ "$new_commits" = "true" ]; then
  state="jules-updated"
  reason="Jules pushed new commits (synchronize) — signals not yet resolved."
elif [ "$age_days" -ge "$STALE_DAYS" ]; then
  state="stale-jules-pr"
  reason="No PR activity for ${age_days}d (>= ${STALE_DAYS}d threshold)."
else
  state="jules-waiting"
  reason="No decisive signal yet — waiting on CI or a reviewer."
fi

# A stale open PR that isn't already terminal is surfaced as stale regardless of a soft state.
if [ "$age_days" -ge "$STALE_DAYS" ] && [ "$state" = "jules-waiting" ]; then
  state="stale-jules-pr"
  reason="No PR activity for ${age_days}d (>= ${STALE_DAYS}d threshold)."
fi

WANT_LABEL="jules-state:$state"

# --- Emit the machine-readable classification ----------------------------------------------
# Shape intentionally mirrors examples/agents/jules-watchdog-state.example.json so a later
# consumer (#136) could persist it verbatim. This script itself writes no durable state.
jq -n \
  --arg repo "$REPO" --argjson pr "$PR" --arg agent jules \
  --arg state "$state" --arg label "$WANT_LABEL" --arg sha "$HEAD_SHA" \
  --arg ci "$ci" --arg review "$review" --arg mergeable "$MERGEABLE" \
  --arg fix "$fix_addressed" --argjson age "$age_days" \
  --arg dup "$duplicate" --arg sup "$superseded_by" \
  --arg issue "${LINKED_ISSUE:-}" --arg event "$EVENT_NAME" \
  --arg updated "$UPDATED_AT" --arg reason "$reason" '
  {
    repo: $repo, pr: $pr, agent: $agent, state: $state, label: $label,
    linked_issue: (if $issue == "" then null else ($issue|tonumber) end),
    head_sha: $sha, event: $event, last_activity_at: $updated,
    signals: {
      ci: $ci, review: $review, mergeable: $mergeable,
      requested_fix_addressed: $fix, age_days: $age,
      duplicate: ($dup == "true"),
      superseded_by: (if $sup == "" then null else ($sup|tonumber) end)
    },
    reason: $reason
  }'

log "PR #$PR classified as '$state' (ci=$ci review=$review dup=$duplicate age=${age_days}d) — $reason"

# --- Reflect the state as a mutually-exclusive label ---------------------------------------
# Ensure the target label exists, add it, then strip any other watchdog state labels. Only a
# genuine transition changes labels; re-running on an unchanged state is a no-op (no noise).
DESC="TARS Jules PR watchdog state (#132)"
gh label create "$WANT_LABEL" --repo "$REPO" --color 1A73E8 --description "$DESC" --force >/dev/null 2>&1 || true
if ! grep -qw "$WANT_LABEL" <<<"$LABELS"; then
  gh pr edit "$PR" --repo "$REPO" --add-label "$WANT_LABEL" >/dev/null 2>&1 || true
  log "Applied $WANT_LABEL to PR #$PR."
fi
for l in "${STATE_LABELS[@]}"; do
  [ "$l" = "$WANT_LABEL" ] && continue
  if grep -qw "$l" <<<"$LABELS"; then
    gh pr edit "$PR" --repo "$REPO" --remove-label "$l" >/dev/null 2>&1 || true
    log "Removed stale $l from PR #$PR."
  fi
done

# --- Comment ONLY on the critical CI-failure transition ------------------------------------
# One comment per failing head SHA (idempotent via a hidden marker) so re-runs never re-spam.
if [ "$ci" = "failure" ]; then
  MARKER="<!-- jules-watchdog:ci-fail:${HEAD_SHA} -->"
  ALREADY=$(gh pr view "$PR" --repo "$REPO" --json comments \
    --jq "[.comments[].body | select(contains(\"$MARKER\"))] | length" 2>/dev/null || echo 0)
  if [ "${ALREADY:-0}" -eq 0 ]; then
    gh pr comment "$PR" --repo "$REPO" --body "$MARKER
🔴 **Jules PR watchdog — CI failure detected**

The latest commit (\`${HEAD_SHA:0:8}\`) has failing checks, so this PR is now \`${WANT_LABEL}\`. @google-labs-jules please push a fix. The watchdog will not merge and does not bypass human/Demerzel review.

<sub>Automated transition notice — see \`docs/agents/jules-pr-watchdog.md\`. This is the only state the watchdog comments on.</sub>"
    log "Posted the CI-failure transition comment on PR #$PR (SHA ${HEAD_SHA:0:8})."
  else
    log "CI-failure comment already present for SHA ${HEAD_SHA:0:8} — not re-commenting."
  fi
fi
