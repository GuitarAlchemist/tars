# Claude Code 3PM Handoff

Source distilled from: `C:\Users\spare\Downloads\ChatGPT-AI Repo IX Structure.md`
Prepared: 2026-03-14

## Goal

Give Claude Code a compact, implementation-ready brief after the token limit resets at 3:00 PM.

This handoff replaces the long exported chat with a small set of decisions, boundaries, and the exact first work package.

## Current Decisions

### 1. Do not create a separate `IX` repo right now

Treat `MachinDeOuf` as the practical `IX` machine-forge for the ecosystem.

Use this responsibility split:

- `MachinDeOuf`: reusable Rust machine primitives, algorithms, CLI contracts, Claude skills, hooks, subagents
- `tars`: cognition, reasoning loops, memory, belief graphs, DSLs, reflective architecture
- `ga`: Guitar Alchemist product and music-domain logic

Working rule:

> If it is reusable outside a single domain, it belongs in `MachinDeOuf`.

Repo rename note:

- include a plan for renaming `MachinDeOuf` to `ix`
- do the rename deliberately, with an inventory of impacted paths, badges, crate names, docs, MCP config, and downstream references
- prefer staged migration over ad hoc string replacement

### 2. Create a separate governance repo for personas / alignment / constitutions

This should be a separate repo, not mixed into `MachinDeOuf` or `tars`.

Reason:

- `MachinDeOuf` should stay focused on machine capability
- `tars` should stay focused on cognition and runtime architecture
- governance artifacts should be reusable across agents and repos

Suggested repo role:

- personas
- constitutions
- alignment policies
- rollback / self-modification policies
- tetravalent logic artifacts
- behavioral tests

Working codename from the source material: `Demerzel`

### 3. Use TARS v1 chats as source material, not as direct persona artifacts

Do not copy raw exploratory chats into the governance repo as personas.

Transform them into:

- canonical persona YAML files
- constitutions
- policies
- behavioral test cases
- extraction notes documenting provenance

Rule:

> TARS v1 chats are inspiration and extraction material, not normative governance artifacts.

## Recommended Repo Directions

### `MachinDeOuf`

Priority structure:

```text
MachinDeOuf/
├─ README.md
├─ Cargo.toml
├─ rust-toolchain.toml
├─ crates/
│  ├─ mdo-core/
│  ├─ mdo-search/
│  ├─ mdo-graph/
│  ├─ mdo-grammar/
│  ├─ mdo-eval/
│  └─ mdo-cli/
├─ skills/
├─ agents/
├─ hooks/
├─ schemas/
├─ examples/
│  ├─ ga/
│  └─ tars/
└─ docs/
```

First principle:

> Every major capability should exist as a library API, CLI command, and Claude skill/subagent entrypoint.

### `Demerzel` governance repo

Priority structure:

```text
Demerzel/
├─ README.md
├─ docs/
│  └─ architecture.md
├─ sources/
│  └─ tars-v1-chats/
│     ├─ README.md
│     └─ extracted-archetypes.md
├─ personas/
│  ├─ default.persona.yaml
│  ├─ reflective-architect.persona.yaml
│  ├─ skeptical-auditor.persona.yaml
│  ├─ kaizen-optimizer.persona.yaml
│  └─ system-integrator.persona.yaml
├─ constitutions/
│  └─ default.constitution.md
├─ policies/
│  ├─ alignment-policy.yaml
│  ├─ rollback-policy.yaml
│  └─ self-modification-policy.yaml
├─ logic/
│  ├─ tetravalent-logic.md
│  └─ tetravalent-state.schema.json
├─ schemas/
│  └─ persona.schema.json
├─ tests/
│  └─ behavioral/
│     └─ contradiction-cases.md
└─ examples/
   └─ claude-code/
      └─ minimal-agent-config.md
```

## What Claude Should Do First

Claude should work in stages, not try to redesign everything in one pass.

### Session objective

Produce a bounded architecture package that:

- locks the repo split
- defines the staged rename and restructuring plan from `MachinDeOuf` to `ix`
- bootstraps a minimal but coherent `Demerzel` repo skeleton and starter artifacts

### In scope for the first session

- restate the architecture decisions briefly
- produce the staged rename plan for `MachinDeOuf` to `ix`
- define the first-milestone target shape for `ix`
- create the repo tree
- create the starter governance files from the distilled source
- add `docs/architecture.md`
- add `tests/behavioral/contradiction-cases.md`
- make schemas and examples internally consistent
- keep the repo independent from `tars` and `MachinDeOuf` internals

### Out of scope for the first session

- building a runtime
- inventing new frameworks
- wiring deep repo integrations
- importing raw TARS v1 chats wholesale
- expanding beyond the minimum viable governance layer

## Exact Prompt For Claude Code

Paste this into Claude Code at 3:00 PM:

```md
Use this file as the handoff source:
C:\Users\spare\source\repos\tars\organization\claude-3pm-handoff-2026-03-14.md

Primary task:
Execute the staged architecture package in this order:
1. restate the architecture decisions
2. produce the rename plan from `MachinDeOuf` to `ix`
3. define the first-milestone target shape for `ix`
4. bootstrap a new governance repo, codename `Demerzel`

Instructions:
1. Read the handoff file first and follow its decisions exactly.
2. Do not start with freeform brainstorming.
3. Produce the rename and migration plan before making architecture changes.
4. Create only the minimum coherent repo skeleton for `Demerzel`.
5. Materialize the starter governance artifacts as real files.
6. Add missing support docs only when needed for consistency.
7. Keep personas, constitutions, policies, schemas, examples, and tests aligned.
8. Treat TARS v1 chats as extraction sources, not direct persona artifacts.
9. Do not build a runtime or invent extra frameworks.
10. Finish with:
   - created file list
   - unresolved questions
   - recommended next small iteration

Important constraints:
- `MachinDeOuf` remains the practical `IX` machine-forge
- include the repo rename to `ix` in the plan
- governance belongs in a separate repo
- keep everything reusable across agents and repos
- preserve contradiction handling, bounded autonomy, rollback, and explicit uncertainty
```

## Optional Second Prompt

If you want Claude to prepare `MachinDeOuf` next, use this after the governance bootstrap is done:

```md
Now prepare a concrete restructuring and rename plan for `MachinDeOuf` as the practical `IX` repo.

Requirements:
1. Define the crate map for the first six crates:
   - mdo-core
   - mdo-search
   - mdo-graph
   - mdo-grammar
   - mdo-eval
   - mdo-cli
2. Include a staged plan to rename the repo to `ix`, including:
   - repo folder / remote / README / badges
   - workspace package naming strategy
   - crate rename strategy from `machin-*` to either `ix-*` or a justified alternative
   - CLI / MCP / Claude skill naming updates
   - downstream impact on `ga` and `tars`
3. Define the first Claude skills / subagents / hooks that map onto those capabilities.
4. Keep domain-specific logic out of the repo.
5. Show how `ga` and `tars` consume it through library, CLI, and skill entrypoints.
6. Do not implement everything at once; produce a staged execution plan with migration risk notes.
```

## Compression Of The Original Chat

The exported chat contained four useful outcomes:

- `IX` should be a reusable machine-tooling layer, not another app repo
- `MachinDeOuf` already fills that role well enough to avoid creating `IX` now
- personas / alignment / constitutions deserve their own repo
- TARS v1 chat history should be distilled into governed artifacts rather than copied directly

Everything else was mostly elaboration around those same four points.
