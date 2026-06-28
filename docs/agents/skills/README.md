# Cloud-Agent Skill Library

This directory contains repo-readable skills that cloud agents (e.g., Jules, Codex) can follow when working Away From Keyboard (AFK) from GitHub issues.

Since cloud agents cannot access local TARS skills or local MCP state, this explicit library serves as their primary operational guide.

## Available Skills

- `docs-only-contract`: Use when writing or modifying documentation, architecture design records (ADRs), or JSON schemas without changing runtime code.
- `evidence-bundle`: Use when gathering context or compiling evidence for a specific task or research finding.
- `governed-delegation`: Use when preparing tasks for execution by agents under strict governance constraints.

## Skill Structure

Each skill in this library defines:
- **When to use it:** The scenario where the skill applies.
- **Allowed files/surfaces:** The specific directories or file types the skill permits modifying.
- **Required checks:** Validations the agent must perform during execution.
- **Stop conditions:** Triggers that should immediately halt execution.
- **Evidence expected in PR body:** Requirements for documenting the PR.
- **Common failure modes:** Pitfalls to avoid.

For an example of skill selection, see `examples/agents/skill-selection.example.json`.
