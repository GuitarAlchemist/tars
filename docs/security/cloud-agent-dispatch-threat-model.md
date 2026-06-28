# Threat Model: Cloud-Agent Dispatch

- **Status:** Draft
- **Area:** Security / AFK Loop
- **Target:** Governed Codex Dispatch Seam

## Overview
This document outlines the threat model for integrating cloud-based autonomous agents (specifically Codex and Jules) into the TARS repository via automated dispatch seams. The transition from un-gated comment triggers to governed label triggers introduces new attack surfaces and mitigations.

## 1. Unauthorized Agent Triggering
- **Threat:** A malicious actor (e.g., an unauthorized issue commenter) attempts to trigger a cloud agent to run arbitrary tasks, potentially executing cryptojacking scripts, exhausting quotas, or spamming PRs.
- **Vulnerability:** The previous `@codex` trigger was vulnerable as any GitHub user could post a comment.
- **Mitigation:**
  - **Label Gating:** Transitioning to a `codex-ready` (and `ready-for-agent` for Jules) label mechanism. Only users with repository write/triage permissions can apply labels.
  - **Comment Trigger Deprecation:** The dispatch workflow will explicitly ignore `@codex` comments from unauthorized users or disable the comment trigger entirely.

## 2. Governance Bypass
- **Threat:** An agent loop runs out of control or executes during a planned halt (e.g., during an incident response or budget freeze).
- **Vulnerability:** Cloud agents run outside the local Demerzel environment, so they cannot read local halt markers (`~/.demerzel/HALT-ALL`).
- **Mitigation:**
  - **Cloud-Reachable Halt Marker:** The dispatch workflow explicitly checks `governance/state/afk-halt.json`. If present and unexpired, the workflow parks the task and halts execution, maintaining ecosystem-wide control.

## 3. Malicious Code Generation & Supply Chain Attacks
- **Threat:** An attacker manipulates the issue description to prompt inject the agent into generating malicious code, vulnerable dependencies, or backdoors.
- **Vulnerability:** AI agents can be susceptible to prompt injection, generating code that introduces security flaws.
- **Mitigation:**
  - **Strict Review Gates:** All agent output is pushed to a branch and opened as a Pull Request. Code is automatically evaluated by Demerzel (`qa-verdict-dispatch.yml`).
  - **Mandatory Human Review:** Branch protection rules require human approval and passing CI checks (`dotnet build && dotnet test`) before merging to `main`.
  - **Draft PRs for High Risk:** Medium or high-risk tasks mandate the agent to open a **Draft PR**, preventing accidental merges of unverified, complex logic.

## 4. Quota Exhaustion & Denial of Service
- **Threat:** Agents run out of API quota (e.g., OpenAI Codex limits) or experience rate limiting, causing legitimate tasks to fail or hang indefinitely.
- **Vulnerability:** External APIs have strict usage caps, creating a bottleneck if a surge of tasks is queued.
- **Mitigation:**
  - **Fallback Routing:** The workflow monitors for quota exhaustion or execution failure. If triggered, it degrades gracefully by removing the `codex-ready` label, applying `ready-for-agent` to dispatch Jules, and logging the fallback event.
  - **Bounded Execution:** Risk policies (`examples/agents/codex-dispatch-policy.example.json`) enforce max autonomy and scope limits to prevent runaway tasks from draining quotas.

## 5. Privilege Escalation via Skill Injection
- **Threat:** A compromised skill definition (e.g., in `.claude/skills/`) is injected into the agent's prompt, manipulating its behavior to bypass constraints or exfiltrate secrets.
- **Vulnerability:** Dynamically injected skills alter the agent's operational context.
- **Mitigation:**
  - **Verified Skill Sources:** Skills are sourced from the locked `skills-lock.json` and reviewed directories (`.claude/skills/`). Any modifications to these skill files require PR review.
  - **Isolated Execution:** The workflow injects skills in a structured manner, and the agent executes in a sandboxed environment without access to high-privileged repository secrets (other than the required `JULES_API_KEY` or Codex equivalent, which are scoped specifically to the agent APIs).
