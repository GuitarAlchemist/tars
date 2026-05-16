---
name: digest
description: Capture meaningful session state (current cursor, in-flight work, live hypotheses, open questions, do-NOT-carry-forward, success criteria) to state/digests/latest.md so the next session — including one after auto-compaction — can re-enter without re-discovering context cold. Distinct from /learnings (which captures surprises). Validates against docs/contracts/digest-schema.json.
allowed-tools: Bash, Read, Write, Edit
last_verified: 2026-05-14
karpathy_rules: [R1-think-before-coding, R4-goal-driven-execution]
---

# /digest

Captures the **meaningful state of the current session** to
`state/digests/latest.md`. The `.claude/hooks/sessionstart-digest.ps1` hook
reads it on the next session start and emits it as `additionalContext` for
the model. Pairs with `.claude/hooks/precompact-digest.ps1` which provides
a metadata-only fallback when /digest isn't invoked before auto-compaction.

## When to run

- **Before compaction is imminent** (context feels >60% full).
- **At a natural breakpoint** — after finishing a feature, before a risky
  operation, before handing off to another agent.
- **Before launching long-running background work.**

**Do NOT** invoke on every message or tool call. The digest is for
*meaningful state changes*, not transcript capture.

## What it captures

```yaml
---
schema_version: 1
session_id: <session_id>
written_at: <RFC3339 UTC>
trigger: digest-skill
branch: <git branch>
head_sha: <short SHA>
head_subject: <commit subject>
open_pr: <"#N" or null>
last_model_update: <RFC3339 UTC>
success_criteria:
  - criterion: "<testable assertion for the Next action>"
    status: pending | in-progress | achieved | abandoned
    evidence: "<file:line | PR# | metric path | null>"
---

# Session digest — <branch> @ <sha>

## Next action

ONE imperative sentence. The Next action must map 1:1 to entries in the
`success_criteria` frontmatter array — each criterion testable, not vibes.

## In-flight

Bulleted list of work currently mid-execution. For each item: the
file/feature, current step number out of total, and immediate next sub-step.

## Live hypotheses

Bulleted list of working hypotheses the next session should inherit.
*Unconfirmed*; do not promote to MEMORY.md until validated.

## Open questions

Numbered. Questions you would ask the user if they walked in right now.

## Do NOT carry forward

**Highest-leverage field.** Things the next session must NOT re-propose:
rejected designs, abandoned approaches, paths the user explicitly closed.

## Prior success criteria status (Karpathy R4)

When a prior digest exists, this section reports the status delta:

- ✅ achieved: <prior criterion> — evidence: <file:line | commit | PR>
- ⏳ in-progress: <prior criterion> — last touched: <where>
- ⛔ abandoned: <prior criterion> — reason: <one sentence>
```

## How to run

1. **Read existing** `state/digests/latest.md` if present — preserve content
   sections still current; rewrite stale ones.
2. **Karpathy R4 — review prior success criteria** if the prior digest had
   them. Mark `achieved` with evidence, `in-progress` with where parked, or
   `abandoned` with a one-sentence reason.
3. **Capture git state** via Bash:
   ```bash
   git rev-parse --abbrev-ref HEAD
   git rev-parse --short HEAD
   git log -1 --format='%s'
   gh pr view --json number 2>$null
   ```
4. **Synthesize the content sections** from your current working context.
   Derive 1–3 testable `success_criteria` entries from the Next action.
5. **Write** the full digest to `state/digests/latest.md` (overwrite). Set
   `trigger: digest-skill` and `last_model_update` to current RFC3339 UTC.
6. **Reset the activity counter**:
   ```bash
   rm -f state/digests/.activity-counter
   ```
7. **Validate** against the schema (Karpathy R11):
   ```bash
   pwsh -NoProfile -File .claude/hooks/digest-validate.ps1
   ```
   Non-zero exit = malformed; fix and rewrite.
8. **Report**:
   `Digest updated: <branch>@<sha> · next: <one-line> · criteria: <N>`.

## Driving criteria autonomously with `/goal`

After writing the digest, **consider `/goal <condition>` (native Claude
Code v2.1.139+) for substantial autonomous work**. `/goal` mechanizes
Karpathy R4: a small fast model evaluates after every turn whether the
condition holds and either fires another turn or clears the goal.

Use `/goal` when the Next action has:

- A verifiable end state (build green, tests pass, file count under budget)
- 5+ minutes of expected autonomous work
- An evaluator-checkable result (checked against the transcript — no
  tool calls)

Skip `/goal` for short tasks, visual/UX judgement, or ambiguous specs.

When `/goal` lands a "yes," the next `/digest` should mark the
corresponding `success_criteria` entry as `achieved` with the `/goal`
evidence.

## Anti-patterns

- **Transcript capture.** Git log is the transcript. Write the cursor.
- **Empty digest.** "Continue" with no In-flight is noise. If nothing
  is in flight, don't write — the prior digest still applies.
- **Forgetting "Do NOT carry forward."** Always populate, or write "none".

## Related

- `/learnings` — captures surprises into `docs/solutions/` (different artifact).
- `/correct` — captures user corrections as permanent CLAUDE.md rules.
- `.claude/hooks/precompact-digest.ps1` — automatic fallback.
- `.claude/hooks/sessionstart-digest.ps1` — reads `latest.md` back.
- `state/digests/README.md` — directory layout.
