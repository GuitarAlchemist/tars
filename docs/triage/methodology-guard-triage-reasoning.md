# Methodology Guard Triage Reasoning

Date: 2026-07-01

Status: draft / advisory

## Purpose

Define how TARS should interpret Methodology Guard findings during triage without turning `.github` workflow checks into policy authority.

## Core split

```text
.github emits findings.
TARS explains what the findings mean for intent and review readiness.
IX measures whether the findings were useful.
Demerzel decides policy gates.
Human decides high-impact transitions.
```

## Input contract

TARS may consume:

```text
methodology-guard.json
methodology-guard.md
issue body
PR body
evidence packet
review-mode context
Demerzel policy recommendation when available
IX usefulness/noise metrics when available
```

## Output contract

TARS should emit a triage reasoning artifact:

```json
{
  "artifact_type": "tars-methodology-triage",
  "version": "0.1-draft",
  "source_findings": [],
  "intent_risk": "low | medium | high",
  "readiness": "rough | shaped | ready | needs-evidence | needs-review | decision-gate",
  "review_mode_suggestion": "silent-classify | batch-digest | fast-review | focused-review | decision-gate | escalate-review",
  "reasoning_summary": "",
  "human_question": "",
  "safe_default": ""
}
```

## Reasoning rules

```text
Missing business value usually means keep shaping.
Missing hierarchy usually means clarify parent/child/related map.
Missing scope/boundary usually means risk of hidden coupling.
Missing evidence usually means not ready for merge/review.
Research notes should not be forced into execution-ready templates.
```

## Non-goals

```text
Do not mutate GitHub labels directly.
Do not close issues.
Do not enforce policy gates.
Do not hide raw findings.
Do not treat template compliance as proof of correctness.
```

## /teach candidate

This concept should become a `/teach` lesson:

```text
over-abstraction vs under-abstraction in agentic workflow systems
```

Lesson should include:

```text
.github as mechanism
Demerzel as policy
TARS as judgment
IX as measurement
human as final authority
```

## Related

- GuitarAlchemist/.github#28
- GuitarAlchemist/.github#30
- GuitarAlchemist/.github#31
- GuitarAlchemist/Demerzel#588
- GuitarAlchemist/ix#217
- GuitarAlchemist/ix#218
