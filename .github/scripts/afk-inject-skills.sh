#!/usr/bin/env bash
# afk-inject-skills.sh — skill-injection hook for the Jules AFK prompt path (issue #115).
#
# The Jules lane of the AFK router (.github/workflows/afk-router.yml) delegates by
# applying the `jules` label; Jules then reads the *issue thread* as its prompt. This
# hook lets an issue *declare* which cloud-agent skills apply, resolves them against a
# small allowlist under docs/agents/skills/, and posts the matching skill docs as a
# single governed comment so they become part of what Jules reads.
#
# Declaration (either form, unioned):
#   1. YAML in the issue body:      agent_skills:
#                                     - docs-only-contract
#                                     - evidence-bundle
#   2. Labels:                      skill:docs-only-contract
#
# Safety invariants (see docs/agents/skills/README.md):
#   - Never reads arbitrary paths from issue content. Declared names are sanitized to
#     ^[a-z0-9-]+$ and can only ever resolve to docs/agents/skills/<name>.skill.md,
#     which must already exist. Anything else (path traversal, unknown skill) is dropped.
#   - The allowlist is derived from the *.skill.md files actually present in the repo,
#     so it can never point outside the curated skill library.
#   - If no valid skill is declared, the script exits 0 without commenting — current
#     behavior is preserved (backward compatible).
#   - This hook only injects context; it never applies/removes labels, never triggers,
#     and does not touch the halt gate or the PAT preflight (those stay in the workflow).
#
# Environment:
#   REPO         owner/name of the repo             (required unless AFK_INJECT_DRYRUN=1)
#   ISSUE        issue number                        (required unless AFK_INJECT_DRYRUN=1)
#   GH_TOKEN     token used for `gh issue comment`   (required unless AFK_INJECT_DRYRUN=1)
#   ISSUE_BODY   override issue body (skips gh read; used for tests/dry-run)
#   ISSUE_LABELS override labels, one per line       (used for tests/dry-run)
#   AFK_INJECT_DRYRUN=1  print the assembled prompt to stdout instead of posting it
#
# Exit codes: 0 = injected or nothing-to-do (both are success). Non-zero only on a
# genuine failure (e.g. `gh` unavailable when a post was required).
set -euo pipefail
# noglob: declared names are untrusted issue content that we word-split below; a token
# like `*` must never undergo pathname expansion. The allowlist check uses direct
# `-f` file tests (not globs), so disabling globbing here is safe.
set -f

SKILL_DIR="docs/agents/skills"

# --- gather inputs -----------------------------------------------------------------
body="${ISSUE_BODY-}"
labels="${ISSUE_LABELS-}"
if [ -z "${body}${labels}" ] && [ "${AFK_INJECT_DRYRUN:-0}" != "1" ]; then
  body=$(gh issue view "$ISSUE" --repo "$REPO" --json body --jq '.body // ""')
  labels=$(gh issue view "$ISSUE" --repo "$REPO" --json labels --jq '.labels[].name')
fi

# --- collect declared skill names --------------------------------------------------
# (a) the `agent_skills:` YAML list in the body: the list header followed by `- name`
#     lines, stopping at the first line that is neither blank nor a `-` list item.
declared=$(printf '%s\n' "$body" | awk '
  /^[[:space:]]*agent_skills:[[:space:]]*$/ { grab=1; next }
  grab==1 {
    if ($0 ~ /^[[:space:]]*-[[:space:]]*[^[:space:]]/) {
      line=$0; sub(/^[[:space:]]*-[[:space:]]*/, "", line); sub(/[[:space:]].*$/, "", line); print line
    } else if ($0 ~ /^[[:space:]]*$/) {
      next
    } else { grab=0 }
  }
')

# (b) `skill:<name>` labels.
label_skills=$(printf '%s\n' "$labels" | sed -n 's/^skill:\(.*\)$/\1/p')

# --- resolve against the allowlist -------------------------------------------------
# Allowlist = basenames of the *.skill.md files that actually exist. A declared name
# is accepted only if it is [a-z0-9-]+ AND docs/agents/skills/<name>.skill.md exists.
selected=""
seen=" "
for raw in $declared $label_skills; do
  name=$(printf '%s' "$raw" | tr -d '[:space:]')
  case "$name" in
    *[!a-z0-9-]*|"" ) continue ;;   # reject anything outside the safe charset
  esac
  case "$seen" in *" $name "*) continue ;; esac   # dedupe
  if [ -f "$SKILL_DIR/$name.skill.md" ]; then
    selected="$selected $name"
    seen="$seen$name "
  else
    echo "::notice::afk-inject-skills: dropping undeclared/unknown skill '$name' (no $SKILL_DIR/$name.skill.md)" >&2
  fi
done
selected=$(printf '%s' "$selected" | sed 's/^ *//')

if [ -z "$selected" ]; then
  echo "::notice::afk-inject-skills: no allowlisted skills declared — preserving default behavior" >&2
  # Emit an empty selection marker so the workflow can record it in evidence if it wants.
  if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "selected=" >> "$GITHUB_OUTPUT"; fi
  exit 0
fi

# --- assemble the prompt comment ---------------------------------------------------
comment=$(
  printf '### 🧩 Declared cloud-agent skills (auto-injected)\n\n'
  printf 'This issue declared the skills below (via `agent_skills:` and/or `skill:*` labels). '
  printf 'They are reproduced here from `%s/` so the cloud agent applies them while working this issue. '
  printf 'Follow every skill contract; report the requested evidence in the PR body.\n\n' "$SKILL_DIR"
  printf 'Selected skills:'
  for name in $selected; do printf ' `%s`' "$name"; done
  printf '\n'
  for name in $selected; do
    printf '\n<details>\n<summary><code>%s.skill.md</code></summary>\n\n' "$name"
    cat "$SKILL_DIR/$name.skill.md"
    printf '\n</details>\n'
  done
)

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "selected=$(printf '%s' "$selected" | tr ' ' ',')" >> "$GITHUB_OUTPUT"
fi

if [ "${AFK_INJECT_DRYRUN:-0}" = "1" ]; then
  printf '%s\n' "$comment"
  exit 0
fi

printf '%s' "$comment" | gh issue comment "$ISSUE" --repo "$REPO" --body-file -
echo "::notice::afk-inject-skills: injected skills [$(printf '%s' "$selected" | tr ' ' ',')] into issue #$ISSUE"
