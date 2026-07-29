---
review_agents: [code-simplicity-reviewer, security-sentinel, performance-oracle, architecture-strategist]
plan_review_agents: [code-simplicity-reviewer]
---

# Review Context

Add project-specific review instructions here.
These notes are passed to all review agents during /workflows:review and /workflows:work.

- F# repo (net10.0), functional-first: immutable types, `Result<>` for errors — flag mutable state and exceptions used for control flow.
- Working directory is `v2/`, not repo root; build with `dotnet build -p:NuGetAudit=false` on this machine (NU1902/NU1903 restore-audit blocker).
- Warnings-as-errors is on (`TreatWarningsAsErrors=true`); NU1608 is the only exemption.
- LLM access must go through `LlmFactory.create(logger)` — direct `DefaultLlmService` instantiation is a defect.
- Self-modification surfaces (SelfHostingGate, PromotionPipeline, GrammarGovernor) deserve extra scrutiny: docs/research/ (2026-07) documents spec-gaming and rubber-stamp-approval failure modes — review gate/criteria changes adversarially.
- Deliberately-red tests tagged `Category=SelfImproveBacklog` are backlog seeds, not failures — do not flag them.
