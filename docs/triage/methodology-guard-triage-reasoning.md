# Methodology Guard Triage Reasoning

This document defines how TARS interprets findings from the `.github` Methodology Guard during triage. TARS acts as a **reasoning layer**, translating raw workflow output into actionable intent for human maintainers and Demerzel governance.

## Core Philosophy

- **TARS is not a policy engine.** It does not enforce gates, close issues, or mutate labels.
- **Reasoning over Compliance.** TARS distinguishes between "missing template fields" (low risk, structural) and "intent/evidence gaps" (high risk, semantic).
- **Advisory Output.** Findings are interpreted to suggest a **Review Mode** and **Readiness State**, but final authority rests with Humans or Demerzel.

## Readiness State Mapping

TARS maps Methodology Guard findings to the following internal readiness states:

| Finding Type | TARS Interpretation | Suggested Readiness |
| :--- | :--- | :--- |
| **Missing required section** | Structural gap; may hide intent. | `needs-revision` (Minor) |
| **Empty "Evidence" block** | Critical verification risk; "Summary Wall" detected. | `needs-evidence` (P1) |
| **Undefined "Scope"** | Architecture risk; "Anti-Ball-of-Mud" violation possible. | `blocked-on-scope` |
| **Low-confidence IX metric** | Usefulness risk; task may be gamed. | `needs-justification` |
| **Perfect structural match** | Ready for semantic reasoning. | `ready-for-review` |

## Review Mode Suggestions

TARS uses the [Review Mode Router](../v2/docs/teaching/lessons/review-mode-router.md) to suggest the appropriate depth for human intervention based on Guard findings:

1. **focused-review**: Suggested when the Guard reports a "clean" structural pass on a non-trivial change.
2. **decision-gate**: Suggested when Guard findings indicate high-impact architecture changes without clear "Before/After" evidence.
3. **escalate-review**: Suggested when the Guard detects a "Goodhart Risk" or a direct violation of repository-native contracts.
4. **fast-review**: Suggested for documentation-only changes that meet template standards.

## Interpreting Intent vs. Template

A Methodology Guard failure does not always mean a "bad" PR. TARS applies the following logic:

- **False Negative (Safe Fail):** If a template field is missing but the *Intent* is clear from the body and the *Evidence* is present in the diff, TARS notes the structural failure but recommends moving to `focused-review`.
- **False Positive (Dangerous Pass):** If a template is perfectly filled but contains "slop" or "hallucinated evidence," TARS flags a **Verification Horizon** violation regardless of the Guard's green status.

## Non-Goals

- **Do not** hide the raw output of the Methodology Guard.
- **Do not** automate state changes (labels/closing) based on reasoning.
- **Do not** override a Demerzel policy decision if one exists.
- **Do not** treat template compliance as a proxy for correctness.

## Reference Contracts

- [Evidence Bundle Contract](../v2/docs/contracts/evidence-bundle.contract.md)
- [Review Mode Router](../v2/docs/teaching/lessons/review-mode-router.md)
- [Verification Horizon](../v2/docs/teaching/lessons/verification-horizon.md)
