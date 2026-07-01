# Cloud-Agent Skill Library

This directory contains explicit skills for cloud agents (like Jules or Codex) when working AFK (Away From Keyboard) from GitHub issues.

Cloud agents cannot access local TARS skills or local MCP state. However, they can reliably read repo files and issue bodies. This repo-readable skill library provides standardized behavior patterns that cloud agents must follow when executing delegated issues.

## Available Skills

- **Docs-Only Contract** (`docs-only-contract.skill.md`): Used for documentation-only updates, enforcing that no runtime behavior is changed.
- **Evidence Bundle** (`evidence-bundle.skill.md`): Used for grounding claims and design decisions in explicitly referenced source material.
- **Governed Delegation** (`governed-delegation.skill.md`): Used for navigating the Demerzel governance constraints and respecting AFK halt markers.
- **Anti-Ball-of-Mud** (`anti-ball-of-mud.skill.md`): Detect architecture entropy before/within feature or refactor work, name **one** seam, keep the PR reviewable, and escalate architectural decisions to human/Demerzel review. TARS-native port of `mattpocock/skills`. Has a user-invoked mode (`/anti-ball-of-mud`) and a model-invoked guard (`anti-ball-of-mud-guard`). Supporting docs: `codebase-design-vocabulary.md`, `entropy-signal-catalog.md`, `matt-pocock-skills-mapping.md`. Example invocation + expected output under `examples/agents/anti-ball-of-mud*`.

## Usage

Cloud agents automatically receive the content of declared skills in their prompt. Skills can be declared in a GitHub issue in two ways:

1. **Via Labels:** Apply a label with the prefix `skill:`, for example `skill:docs-only-contract`.
2. **Via Issue Body YAML:** Include an `agent_skills` list in the `issue_meta` block (or anywhere in the body as a top-level YAML-like key):

```yaml
agent_skills:
  - docs-only-contract
  - evidence-bundle
```

The Jules AFK workflow validates these declarations against the allowlist of `.skill.md` files in this directory and injects the corresponding documentation into the agent's prompt.
