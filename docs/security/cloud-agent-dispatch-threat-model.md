# Cloud-Agent Dispatch Threat Model

## Overview
This document outlines the threat model for dispatching cloud-agents (specifically Codex) within the TARS ecosystem. It details potential attack vectors, inherent risks, and the corresponding mitigations enforced by our governance and dispatch architecture.

## Identified Threats & Attack Vectors

### 1. Unauthorized Agent Triggering (The "Public Commenter" Threat)
**Threat:** A malicious user or bot attempts to trigger Codex execution by posting comments on public issues, intending to consume quota, disrupt workflows, or attempt prompt injection.
**Mitigation:**
- Agent dispatch is completely decoupled from issue comments.
- Codex execution is **only** triggered by the explicit application of the `codex-ready` label.
- Only repository maintainers and authorized collaborators have the permissions to apply labels.

### 2. Malicious Prompt Injection in Issue Body
**Threat:** An attacker crafts an issue description with malicious instructions designed to trick the agent into deleting code, exfiltrating data, or executing banned tools.
**Mitigation:**
- **Bounded Execution:** Codex operates under strict tool allowlists and forbidden tool deny-lists (e.g., shell access is forbidden).
- **Mandatory Draft PRs:** Codex cannot merge code directly. It is restricted to opening Draft PRs.
- **Review Gating:** All agent-generated code must pass CI (`dotnet build`, `dotnet test`) and undergo mandatory human and Demerzel review before integration.

### 3. Runaway Agent Execution
**Threat:** The agent enters a recursive loop or performs excessively expensive operations, draining the financial budget or locking up CI resources.
**Mitigation:**
- Strict `max_runtime_minutes` and `max_cost_usd` boundaries are defined in the delegation policy.
- Execution halts immediately if these thresholds are breached.

### 4. Privilege Escalation or Modification of Governance Rules
**Threat:** The agent attempts to modify governance policies (e.g., Demerzel rules) or circumvent its own constraints.
**Mitigation:**
- The delegation policy explicitly restricts the `files_or_surfaces_allowed` that the agent can modify.
- High-risk areas (like `governance/` or `docs/contracts/`) require Demerzel meta-governance gate approval.

### 5. Failure to Halt (The "Kill Switch" Bypass)
**Threat:** System operators attempt to stop agent activity, but in-flight processes continue executing.
**Mitigation:**
- Absolute reliance on the `governance/state/afk-halt.json` marker.
- The dispatch workflow checks this marker *before* triggering the agent, and the agent execution environment continuously polls or respects this marker to abort mid-flight if the halt state is activated.

## Summary of Controls

| Threat Vector | Primary Control | Secondary Control |
|---------------|-----------------|-------------------|
| Unauthorized Trigger | Maintainer-only Label (`codex-ready`) | - |
| Prompt Injection | Draft PR output | Mandatory Human Review |
| Runaway Execution | `max_runtime_minutes` / `max_cost_usd` limits | - |
| Scope Violation | `files_or_surfaces_allowed` restrictions | Demerzel review gate |
| Emergency Stop | `governance/state/afk-halt.json` | - |
