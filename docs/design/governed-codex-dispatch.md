# Governed Codex Dispatch Design

## Overview
This document defines the architectural seam and process for dispatching Codex for cloud-agent work within the TARS repository. The goal is to bring Codex into the same governed, review-gated, and halt-gated workflow as other agent paths (such as the Jules AFK path), ensuring that execution is safe, predictable, and fully reviewed before changes are merged.

## Workflow

The governed workflow for triggering Codex is as follows:

1. **Explicit Maintainer Trigger:** A repository maintainer or collaborator applies the `codex-ready` label to an issue. Public comments or external triggers are ignored to prevent uncontrolled execution.
2. **Governance Gate Check:** A GitHub Actions workflow (or equivalent CI automation) intercepts the label event and strictly checks the `governance/state/afk-halt.json` marker. If the halt marker is present, the workflow aborts immediately.
3. **Policy and Allowlist Check:** The workflow reads the associated agent delegation policy (e.g., `codex-dispatch-policy.json`) and verifies that the requested task conforms to the allowed risk level, scope, and allowed surfaces.
4. **Dispatch:** If all gates pass, the workflow either posts a bounded Codex trigger (via a secure webhook/API) or generates a `codex-task` artifact detailing the scope of work.
5. **Execution:** Codex receives the task, executes the work within the bounded constraints (e.g., cost, time, surface limitations), and outputs a Draft Pull Request (or a detailed plan in the issue if `delegation_level` is lower). Direct merges to `main` are strictly prohibited.
6. **Review and Merge Gate:** The Draft PR must pass all CI tests (e.g., `dotnet build` and `dotnet test`). Furthermore, the PR is gated by mandatory human review and Demerzel's automated governance checks before it can be merged.

## Skills Provisioning

To successfully complete tasks, Codex may require specific skills (e.g., those defined in #114 and #115). These skills are supplied to Codex as follows:
- **Skill Injection:** Skills located in `.claude/skills/` (or a similar secure repository location) are dynamically injected into the system prompt or provided as bounded tool endpoints when the `codex-task` artifact is generated.
- **Allowed Tools Check:** Codex is explicitly restricted to use only the tools defined in the `allowed_tools` list of the delegation policy. Banned tools (such as deployment scripts) remain inaccessible.

## Fallback Behavior

When Codex quota is unavailable or the service is unreachable, the system must degrade gracefully:
- **Notification:** The workflow logs the failure and automatically posts a comment on the corresponding issue indicating that the Codex quota is exceeded or the service is temporarily unavailable.
- **Re-queuing/Fallback:** The issue is labeled with `codex-quota-exceeded` (or similar) and transitioned back to a manual backlog state. Maintainers can re-trigger the workflow later once quota is restored, or a fallback agent (like Jules) may be suggested if appropriate.

## Relationship to Jules AFK Path

The Codex dispatch mechanism is designed to sit alongside the existing Jules AFK path (`ready-for-agent`). Both paths share the same underlying governance principles:
- **Halt Gated:** Both respect `governance/state/afk-halt.json`.
- **Review Gated:** Both require Draft PRs and mandatory human/Demerzel review.
- **Trigger Distinctions:** Jules is triggered via `ready-for-agent` and auto-delegation workflows, while Codex is triggered specifically via the `codex-ready` label for targeted cloud-agent work. This clear separation allows maintainers to route work to the appropriate agent based on capability and cost considerations.
