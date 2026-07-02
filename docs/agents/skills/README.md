# Cloud-Agent Skill Library

This directory contains explicit skills for cloud agents (like Jules or Codex) when working AFK (Away From Keyboard) from GitHub issues.

Cloud agents cannot access local TARS skills or local MCP state. However, they can reliably read repo files and issue bodies. This repo-readable skill library provides standardized behavior patterns that cloud agents must follow when executing delegated issues.

## Available Skills

- **Docs-Only Contract** (`docs-only-contract.skill.md`): Used for documentation-only updates, enforcing that no runtime behavior is changed.
- **Evidence Bundle** (`evidence-bundle.skill.md`): Used for grounding claims and design decisions in explicitly referenced source material.
- **Governed Delegation** (`governed-delegation.skill.md`): Used for navigating the Demerzel governance constraints and respecting AFK halt markers.
- **Anti-Ball-of-Mud** (`anti-ball-of-mud.skill.md`): Detect architecture entropy before/within feature or refactor work, name **one** seam, keep the PR reviewable, and escalate architectural decisions to human/Demerzel review. TARS-native port of `mattpocock/skills`. Has a user-invoked mode (`/anti-ball-of-mud`) and a model-invoked guard (`anti-ball-of-mud-guard`). Supporting docs: `codebase-design-vocabulary.md`, `entropy-signal-catalog.md`, `matt-pocock-skills-mapping.md`. Example invocation + expected output under `examples/agents/anti-ball-of-mud*`.

## Usage

Cloud agents should parse the `issue_meta` block in the GitHub issue body and use these definitions to guide their approach, ensuring that they respect project constraints, execute required checks, and provide the expected evidence in the pull request body.

## Declaring skills on an issue

An issue can declare which of the skills above apply. The AFK router resolves the
declaration against the allowlist (below) and injects the matching skill docs into the
Jules prompt path (see "Injection hook"). Two equivalent forms — they are unioned and
de-duplicated:

1. **YAML in the issue body** (preferred — travels with the issue):

   ```yaml
   agent_skills:
     - docs-only-contract
     - evidence-bundle
   ```

2. **Labels** (useful when triaging without editing the body):

   ```text
   skill:docs-only-contract
   skill:evidence-bundle
   ```

If nothing is declared, the router behaves exactly as before — no comment is posted and
no behavior changes (backward compatible).

## Allowlist (skill-selection rules)

The allowlist is **derived from the files that exist here**, never from issue content:

- A declared name is accepted only if it matches `^[a-z0-9-]+$` **and**
  `docs/agents/skills/<name>.skill.md` exists. Currently that means:
  `docs-only-contract`, `evidence-bundle`, `governed-delegation`, `anti-ball-of-mud`.
- Supporting docs without a `.skill.md` suffix (e.g. `codebase-design-vocabulary.md`)
  are intentionally **not** selectable — they are references, not skill contracts.
- Any name that fails the charset check (path traversal like `../../etc/passwd`,
  absolute paths, spaces) or does not resolve to an existing `*.skill.md` file is
  dropped with a `::notice::` and never read.

## Injection hook

`.github/scripts/afk-inject-skills.sh` implements the hook. Given an issue's body and
labels it: collects declared names → filters to the allowlist → posts a **single**
governed comment reproducing the selected skill docs so Jules reads them as part of the
issue thread (Jules's prompt is the issue thread — the Jules lane delegates by label,
not by an API prompt). Run it in dry-run to preview:

```bash
AFK_INJECT_DRYRUN=1 ISSUE_BODY="$(gh issue view 115 --json body --jq .body)" \
  ISSUE_LABELS="$(gh issue view 115 --json labels --jq '.labels[].name')" \
  bash .github/scripts/afk-inject-skills.sh
```

### Workflow wiring

The hook is called from the Jules lane of `.github/workflows/afk-router.yml`, **after**
the halt-gate + PAT preflight and only when routing to Jules, so neither guard is
touched:

```yaml
      - name: Inject declared skills into the Jules prompt (issue #115)
        if: steps.gate.outputs.proceed == 'true' && steps.gate.outputs.agent == 'jules'
        env:
          GH_TOKEN: ${{ github.token }}
          REPO: ${{ github.repository }}
          ISSUE: ${{ github.event.issue.number }}
        run: bash .github/scripts/afk-inject-skills.sh
```

> The `GH_TOKEN` here uses the default `github.token`: this step only *posts context*
> that Jules reads, it does not *trigger* Jules, so it does not need the user-attributed
> PAT (the label application in the next step remains PAT-attributed).

### Safety notes

- **No arbitrary path reads.** Declared names are sanitized and can only ever resolve to
  `docs/agents/skills/<name>.skill.md`. Public commenters cannot point the hook at any
  other file.
- **Allowlist-gated.** Only the curated `*.skill.md` files are selectable.
- **Halt gate & PAT preflight intact.** The hook runs downstream of both and changes
  neither; it never applies/removes labels and never merges.
- **Backward compatible.** No declaration ⇒ no comment ⇒ current behavior.
- **Evidence.** The step exports `selected=<comma-list>` on `$GITHUB_OUTPUT`, so the
  selected skills can be surfaced in run logs / PR evidence.

See `examples/agents/skill-selection.example.json` for a worked selection and
`examples/agents/skill-injection.example.json` for a worked injection (including how
hostile input is dropped).
