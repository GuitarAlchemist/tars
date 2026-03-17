# Architecture Restatement

Produced: 2026-03-14, Phase 1 of execution plan.

## Repo Responsibilities

| Repo | Codename | Role | Owns |
|------|----------|------|------|
| **MachinDeOuf** (rename to **ix**) | IX | Machine forge | Reusable Rust primitives, algorithms, CLI, MCP server, Claude skills/agents/hooks |
| **tars** | TARS | Cognition | Reasoning loops, memory, belief graphs, DSLs, reflective architecture |
| **ga** | Guitar Alchemist | Product | Music-domain logic, UI, domain-specific features |

**Boundary rule:** If it is reusable outside a single domain, it belongs in IX.

## Why Governance is Separate

A new repo (**Demerzel**) holds personas, constitutions, alignment policies, and behavioral tests because:

1. **IX** should stay focused on machine capability — adding persona YAML and alignment policies would dilute its purpose
2. **TARS** should stay focused on cognition — governance is not runtime logic
3. Governance artifacts must be **reusable across agents and repos** — they apply equally to a TARS reasoning agent and a GA music agent
4. Separation enables independent versioning and review of alignment-sensitive content

## Explicitly Out of Scope

- Building a runtime
- Inventing new frameworks
- Wiring deep repo integrations between ix/tars/ga
- Importing raw TARS v1 chats as final persona artifacts
- Expanding beyond minimum viable governance layer
- Refactoring all 31 crates in one pass
