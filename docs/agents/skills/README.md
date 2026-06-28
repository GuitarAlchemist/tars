# Agent Skills

This directory contains optional agent skills that can be declared in an issue body or labels to be injected into the agent prompt.

## How to use

In an issue body:

```yaml
agent_skills:
  - docs-only-contract
  - evidence-bundle
```

Or via labels:

```text
skill:docs-only-contract
```

When a matching issue is picked up by an agent, the skills will be added to the prompt automatically.
