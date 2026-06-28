# Research-to-Issue Factory Design

- **Status:** Draft
- **Area:** Runtime / Technology Watch
- **Owner:** TARS Cortex
- **Version:** 1.0.0

## Goal
Design the research-to-issue factory that turns validated research digests and Streeling findings into agent-ready GitHub issue drafts.

The factory supports delegation to AI agents by producing clear, bounded, evidence-backed issues with acceptance criteria, non-goals, cost/risk policy, and AFK readiness metadata.

## Candidate Input Artifacts
The factory processes the following input artifacts:
- `ResearchDigest`
- `ThesisDigest`
- `ResearchFinding`
- `HarnessImprovementProposal`
- `Streeling EvidenceBundle`
- `IX scorecard`
- `Seldon learning assessment`

## Candidate Output Artifact
The factory produces an `IssueDraft` artifact.

### Minimum Issue Fields
The `IssueDraft` must contain the following minimum fields:
- `title`
- `repository`
- `parent_issue`
- `related_issues`
- `issue_meta`
- `problem_statement`
- `goal`
- `context`
- `evidence_required`
- `work_breakdown`
- `acceptance_criteria`
- `non_goals`
- `risk_policy`
- `budget_policy`
- `afk_readiness`
- `routing`

## Candidate Routing Rules
Issues are routed to the appropriate repository based on the following rules:
- **TARS:** harness, closures, grammar contracts, runtime, second brain
- **IX:** algorithms, evals, DuckDB analytics, scoring, retrieval metrics
- **Demerzel:** governance, budget, approval, escalation, AIW gates
- **GA:** product/music/guitar domain scenarios and user-facing validation

## Governance Gates
- **Low-risk documentation issue:** May be created after validation.
- **Medium-risk design issue:** May be drafted or created with explicit trace.
- **High-risk implementation issue:** Draft only unless approved.
- **Agent Code Changes:** Any issue that could trigger agent code changes must include tests and stop conditions.

## Non-goals
- Do not create issues without evidence/provenance.
- Do not auto-assign agents without governance policy.
- Do not implement GitHub mutation in this story.
- Do not let novelty alone override risk/cost gates.
