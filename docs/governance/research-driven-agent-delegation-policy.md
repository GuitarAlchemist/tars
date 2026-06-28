# Research-Driven Agent Delegation Policy

This document defines the policy and governance constraints for delegating research-driven GitHub issues to AI agents within the TARS ecosystem. It guarantees that agent execution is bounded by explicit risk evaluation, scope, cost limits, and rigorous approval gating.

## 1. Goal

Define the policy for delegating research-driven GitHub issues to AI agents. The technology-watch loop may eventually propose or create issues, but agent execution must be gated by risk, scope, cost, tests, and human/Demerzel approval rules.

### Non-Goals
- Do not implement automatic agent execution.
- Do not allow high-risk code changes without explicit human approval.
- Do not bypass repository-specific CI or test suites.
- Do not treat issue creation itself as proof of value without verification.

## 2. Delegation Levels

Agents are constrained to specific levels of autonomy based on the task:

- `none`: Only store the research finding. No action is taken.
- `suggest`: Propose a follow-up idea within the agent's memory.
- `issue_draft`: Create a local or PR-based issue draft artifact, but do not push.
- `issue_create`: Create a GitHub issue without assigning it to an agent for execution.
- `agent_candidate`: Mark an issue as AFK-ready, but do not initiate execution.
- `agent_execute`: Allow bounded agent execution after explicit approval has been granted.

## 3. Required Policy Dimensions

Any request for delegation must provide the following policy dimensions:

- `risk_level`: Evaluated against the Risk Matrix below (e.g., low, medium, high).
- `repo_target`: The specific repository or repositories where changes are permitted.
- `files_or_surfaces_allowed`: Explicit list or patterns of files/surfaces the agent may modify.
- `requires_tests`: Boolean indicating whether verifiable tests must accompany the changes.
- `requires_human_approval`: Boolean enforcing manual review before merge.
- `requires_demerzel_gate`: Boolean enforcing Demerzel's meta-governance gate.
- `max_runtime_minutes`: The maximum allowed execution time before the agent process is terminated.
- `max_cost_usd`: A hard ceiling on the financial cost for the agent's operations.
- `allowed_tools`: An explicitly allowed list of tools the agent can execute.
- `forbidden_tools`: Tools explicitly banned (e.g., shell commands for deployment).
- `stop_conditions`: Specific criteria that immediately halt the agent's execution.
- `rollback_or_recovery`: Defined procedures to revert changes if execution fails or stops unexpectedly.

## 4. Risk Matrix

Tasks are evaluated on their potential impact to the system, which governs the allowable delegation level.

| Risk Level | Scope & Impact | Maximum Allowed Delegation | Approval Rules |
|------------|----------------|----------------------------|----------------|
| **Low**    | Documentation, research synthesis, passive metrics. | `agent_execute` | Peer review PR; tests not strictly required. |
| **Medium** | New encapsulated features, refactoring non-critical paths. | `agent_candidate` | Requires tests, explicit human PR approval, bounded cost. |
| **High**   | Core architecture, cross-repo interfaces, Demerzel/governance files. | `issue_draft` | Requires explicit human + Demerzel gate approval, strict stop conditions. |

## 5. Stop Conditions & Approval Rules

### Stop Conditions
Agent execution MUST halt immediately if any of the following occur:
1. The budget (`max_cost_usd`) or time limit (`max_runtime_minutes`) is reached.
2. The agent attempts to modify files outside `files_or_surfaces_allowed`.
3. Banned tools (`forbidden_tools`) are invoked.
4. The system detects a recursive loop or excessive repeated tool failures.
5. The `governance/state/afk-halt.json` file is present (indicating a Demerzel or operator halt).

### Approval Rules
- **No Direct Merging**: Agents may only open Pull Requests or draft artifacts. Merging directly to `main` is prohibited.
- **Verification**: Changes flagged as `requires_tests` must include tests that pass locally and in CI.
- **Demerzel Check**: For sensitive areas (`requires_demerzel_gate=true`), the PR must pass Demerzel's automated governance checks.
- **Human Oversight**: For `agent_execute` at medium risk and above, explicit human review on the PR is mandatory.
