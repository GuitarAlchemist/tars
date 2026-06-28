# Skill: Governed Delegation

## When to use it
Use this skill when preparing or executing tasks that involve delegating work to agents under strict governance, policy constraints, and bounds defined in the TARS ecosystem.

## Allowed files/surfaces
- `docs/governance/**/*.md`
- `docs/plans/**/*.md`
- `.github/workflows/**/*.yml` (only if explicitly allowed by issue instructions)
- GitHub issues/PR descriptions and bodies.

## Required checks
- Ensure all delegations adhere to the `research-driven-agent-delegation-policy.md`.
- Verify that limits such as `max_cost_usd` and `max_runtime_minutes` are explicitly defined.
- Confirm that required approval gates (e.g., `requires_human_approval`, `requires_demerzel_gate`) are properly set and acknowledged.

## Stop conditions
- Halt if the task risk level is mismatched with the permitted delegation level.
- Halt if explicit constraints (like budgets or allowed files) are missing.
- Halt if the `governance/state/afk-halt.json` marker is present.

## Evidence expected in the PR body
- Confirmation that the delegation request conforms to governance policies.
- A summary of the enforced bounds (cost, time, risk level).
- Links to relevant governance documentation.

## Common failure modes
- Delegating a high-risk task without defining appropriate approval gates.
- Missing explicit stop conditions or rollback procedures in the delegation plan.
- Overriding or ignoring existing Demerzel governance rules.
