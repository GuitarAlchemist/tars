# Agent Skills

This directory contains optional, modular instructions (skills) that can be dynamically injected into the AI agent's prompt during issue delegation.

## Usage

When creating an issue, you can declare which skills the agent should use. This helps keep the agent's context focused and relevant to the specific task.

### Declaration Formats

1. **YAML block in the issue body:**
   ```yaml
   agent_skills:
     - docs-only-contract
     - evidence-bundle
   ```

2. **Issue Labels:**
   ```text
   skill:docs-only-contract
   skill:evidence-bundle
   ```

When the `jules-auto-delegate` workflow runs, it will scan both the YAML block and the labels, aggregate the requested skills, and append the content of the corresponding Markdown files from this directory to the agent's prompt.

## Security & Constraints

- The workflow uses a strict allowlist constraint: only alphanumeric/hyphen/underscore file names are processed.
- Path traversal (e.g., `skill:../secrets`) is rejected by regex and limited strictly to this `docs/agents/skills/` directory.
- Missing skills emit a warning but do not halt delegation.
