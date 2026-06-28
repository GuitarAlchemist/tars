# Skill: Governed Delegation

## When to use
Use this skill when processing an issue delegated to a cloud agent (e.g., Jules or Codex) to ensure the agent explicitly acknowledges and complies with the Demerzel governance constraints, including respecting AFK (Away From Keyboard) halt markers and checking limits on scope and risk.

## Allowed files/surfaces
- Checking `governance/state/afk-halt.json` (read-only)
- Reading `docs/governance/` (read-only)
- `issue_meta` in the GitHub issue description.
- PR body generation.

## Required checks
1. **Halt Marker Check:** Before beginning any code changes, verify that the `governance/state/afk-halt.json` file does NOT exist, or if it does, that its `expires_at` date has passed. If an active halt is detected, execution must stop immediately.
2. **Autonomy Limit Check:** Verify that the task does not exceed the `max_autonomy` specified in the `issue_meta` (e.g., if it states `max_autonomy: pr`, do not attempt to merge or push directly to main).
3. **Risk Profile Check:** Confirm that the work aligns with the risk constraints documented in the governance policies.

## Stop conditions
- Stop and gracefully report failure (via an issue comment or PR note) if `governance/state/afk-halt.json` is present and active.
- Stop if the requested task violates Demerzel governance rules (e.g., requesting unreviewed merges).
- Stop when the task is complete and a PR has been opened, adhering to the `max_autonomy` boundary.

## Evidence expected in the PR body
- A statement confirming that `governance/state/afk-halt.json` was checked and found to be inactive or absent.
- A statement confirming adherence to the `max_autonomy` limit specified in the issue.
- A checklist of any specific governance policies observed.

## Common failure modes
- Ignoring the `governance/state/afk-halt.json` file and executing the task during an ecosystem-wide halt.
- Proceeding to merge a pull request or push directly to the main branch when `max_autonomy` is limited to `pr`.
- Failing to document the governance check in the final PR body, leading to rejection by the Demerzel qa-tribunal.
