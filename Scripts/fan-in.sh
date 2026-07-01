#!/usr/bin/env bash
# fan-in.sh — fast multi-repo reconcile sweep.
#
# The "fan-in" problem: work fans OUT across ~20 repos, dozens of branches,
# worktree-agents, cloud agents (Jules/Codex) and stashes — then reconciling it
# all back IN is slow manual archaeology. This turns that sweep into one command.
#
# Usage:
#   Scripts/fan-in.sh [ROOT ...]          # report only (default ROOT: ~/source/repos + ~)
#   Scripts/fan-in.sh --ff                # also fast-forward each repo's default branch
#   Scripts/fan-in.sh --push-clean        # also push branches that are a clean fast-forward ahead
#
# It NEVER force-pushes, rebases, commits, or discards. Non-FF / dirty / stashed
# state is only reported, never mutated — you decide what to do with it.
set -uo pipefail

DO_FF=0; DO_PUSH=0; ROOTS=()
for a in "$@"; do
  case "$a" in
    --ff) DO_FF=1 ;;
    --push-clean) DO_PUSH=1 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) ROOTS+=("$a") ;;
  esac
done
[ ${#ROOTS[@]} -eq 0 ] && ROOTS=("$HOME/source/repos" "$HOME/source" "$HOME")

# Discover git repos (depth 2) under each root, de-duplicated.
mapfile -t REPOS < <(
  for r in "${ROOTS[@]}"; do
    find "$r" -maxdepth 2 -name .git -type d 2>/dev/null | sed 's#/\.git$##'
  done | sort -u
)

printf '%-40s %-34s %-9s %-6s %-6s %-6s\n' REPO BRANCH AHD/BHD DIRTY UNPSH STASH
printf '%.0s-' {1..105}; printf '\n'

total_unpushed=0; total_dirty=0; total_stash=0
for p in "${REPOS[@]}"; do
  [ -d "$p/.git" ] || continue
  git -C "$p" fetch --quiet --all --prune 2>/dev/null

  br=$(git -C "$p" rev-parse --abbrev-ref HEAD 2>/dev/null)
  ab=$(git -C "$p" rev-list --left-right --count HEAD...@{u} 2>/dev/null | tr '\t' '/'); [ -z "$ab" ] && ab='no-ups'
  dirty=$(git -C "$p" status --porcelain --untracked-files=no 2>/dev/null | grep -c .)
  unpsh=$(git -C "$p" for-each-ref --format='%(upstream:track)' refs/heads/ 2>/dev/null | grep -c ahead)
  stash=$(git -C "$p" stash list 2>/dev/null | grep -c .)
  printf '%-40s %-34s %-9s %-6s %-6s %-6s\n' "$(basename "$p")" "${br:0:34}" "$ab" "$dirty" "$unpsh" "$stash"
  total_unpushed=$((total_unpushed+unpsh)); total_dirty=$((total_dirty+dirty)); total_stash=$((total_stash+stash))

  # Default branch (main|master) — fast-forward only if requested and strictly behind.
  db=''
  git -C "$p" show-ref -q --verify refs/heads/main   && db=main
  [ -z "$db" ] && git -C "$p" show-ref -q --verify refs/heads/master && db=master
  if [ "$DO_FF" = 1 ] && [ -n "$db" ]; then
    beh=$(git -C "$p" rev-list --count "$db..origin/$db" 2>/dev/null || echo 0)
    ahd=$(git -C "$p" rev-list --count "origin/$db..$db" 2>/dev/null || echo 0)
    if [ "${beh:-0}" -gt 0 ] && [ "${ahd:-0}" -eq 0 ]; then
      git -C "$p" fetch --quiet origin "$db:$db" 2>/dev/null && echo "    ↳ FF $db +$beh"
    fi
  fi

  # Push branches that are cleanly ahead (fast-forward) of an existing upstream.
  if [ "$DO_PUSH" = 1 ]; then
    while read -r b track; do
      case "$track" in
        *ahead*) case "$track" in *behind*) : ;; *) git -C "$p" push --quiet origin "$b" 2>/dev/null && echo "    ↳ pushed $b" ;; esac ;;
      esac
    done < <(git -C "$p" for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads/)
  fi
done

printf '%.0s-' {1..105}; printf '\n'
echo "TOTALS  unpushed-ahead-branches=$total_unpushed  dirty-trees=$total_dirty  stashes=$total_stash"
echo "Report only. Re-run with --ff to fast-forward defaults, --push-clean to push clean-FF branches."
