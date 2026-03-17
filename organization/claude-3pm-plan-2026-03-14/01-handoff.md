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
