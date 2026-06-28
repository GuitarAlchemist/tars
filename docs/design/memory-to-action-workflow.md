# Memory-to-Action Workflow

## Goal
Design the memory-to-action workflow that turns second-brain findings into GitHub issue drafts and safe agent delegation candidates.

This captures the useful part of agentic workspace systems: memory is valuable when it improves action. However, action must remain bounded, evidence-backed, reviewable, and governed.

## Workflow

The flow from a passive memory item to a safe agent execution follows these steps:

1. **Input Generation**: `MemoryItem`, `ResearchFinding`, or `LearningGap` is identified.
2. **Evidence Linking**: A source-grounded evidence bundle is attached to ensure findings are traceable and verified.
3. **Scoring**:
   - **IX scorecard**: Evaluates the actionability and impact on the IX skill engine.
   - **Seldon learning assessment**: Assesses the knowledge transfer value.
4. **Governance Check**: Validates the action against the agent delegation policy and Demerzel governance constraints. Evaluates risk, budget, and AFK-readiness.
5. **Draft Generation**: Formats the validated action into an `IssueDraft`.
6. **Execution Paths**:
   - Optional GitHub issue creation (pending human review for high risk).
   - Optional agent delegation candidate (ready for Jules/Codex).
7. **Replay / Outcome Tracking**: The outcome feeds back into the TARS compounding metrics loop.

## Action Levels

The workflow assigns an action level to determine the degree of autonomy permitted:

- `no_action`: Findings are ingested but require no further steps.
- `memory_link_only`: Updates internal graph/Metascripts but does not trigger external tasks.
- `followup_question`: Requests human or agent clarification on missing evidence.
- `issue_draft`: Prepares a GitHub issue draft but requires human submission.
- `issue_create`: Automatically creates a GitHub issue in the backlog (low-risk only).
- `agent_candidate`: Issue created and tagged as `ready-for-agent` (requires manual dispatch).
- `agent_execute_after_approval`: Auto-delegated to Jules/Codex post-human approval via governance checks.

## Issue Draft Properties

An issue draft must contain the following required properties before creation or delegation:

- `title`
- `repo`
- `parent_epic`
- `related_issues`
- `evidence_bundle`
- `problem_statement`
- `goal`
- `non_goals`
- `acceptance_criteria`
- `test_plan`
- `risk_policy`
- `budget_policy`
- `afk_readiness`
- `stop_conditions`

## Governance and Approval Boundaries

Action must remain governed. High-risk actions or those exceeding budget policies must halt and await approval.
- High-risk issues cannot be auto-created.
- Memory findings cannot bypass evidence/provenance checks.
- Agents cannot be assigned or run without a delegation policy.
- Output generation should prioritize issue quality over quantity.

## Compounding Metrics Integration

This workflow feeds directly into the second-brain compounding metrics framework defined in #100.
Specifically, outcomes of the `IssueDraft` -> Agent execution pipeline will influence:
- `agent_ready_issue_rate`
- `missing_criteria_rate`
- `issue_rework_rate`
- `research_to_issue_quality`

A successful completion increases the `cycle_score` by lowering `rework_penalty` and increasing `quality_gain`.
