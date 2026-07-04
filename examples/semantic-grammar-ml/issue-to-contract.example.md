# Example: Issue-to-Contract Conversion

This example demonstrates how a raw GitHub issue body is transformed into a structured `IssueContract` through the semantic-to-contract pipeline.

## 1. Natural Language Input (GitHub Issue Body)

```markdown
Title: [Story] Implement anti-hallucination guard for TARS DSL

Problem: TARS agents sometimes generate metascripts that use tools which are not currently registered in the ISkillRegistry, leading to runtime failures.

Goal: Add a validation step in the metascript compiler that checks all tool calls against the ISkillRegistry before execution.

Acceptance Criteria:
- Compiler fails with a clear error if a tool is not found.
- Unit test for invalid tool call.
- Registry lookup is cached for performance.
```

## 2. Semantic Interpretation (LLM Monologue)

> The user wants to prevent runtime errors caused by calling non-existent tools in TARS DSL.
> This is a story for the runtime area.
> I need to map this to an `IssueContract`.
> Goals: Tool validation in compiler.
> Acceptance Criteria: Error handling, unit tests, caching.
> Risks: Performance hit if registry lookup is slow (mitigated by caching).
> AFK: This is a bounded compiler change, ready for Jules.

## 3. Candidate IssueContract (JSON)

```json
{
  "title": "[Story] Implement anti-hallucination guard for TARS DSL",
  "issue_meta": {
    "level": "story",
    "area": "runtime",
    "priority": "P1",
    "complexity": "M",
    "risk": "medium",
    "afk": {
      "readiness": "ready",
      "max_autonomy": "pr"
    }
  },
  "problem_statement": "TARS agents sometimes generate metascripts that use unregistered tools, leading to runtime failures.",
  "goal": "Implement a validation step in the metascript compiler to verify tool calls against ISkillRegistry.",
  "acceptance_criteria": [
    "Compiler emits error for unregistered tools.",
    "Unit tests verify validation logic.",
    "ISkillRegistry lookups are cached."
  ],
  "budget_policy": {
    "tier": "free-local",
    "max_cost_usd": 0,
    "max_runner_minutes": 30
  }
}
```

## 4. Validation & Finalization

- **Schema Check**: JSON matches `IssueContract` schema.
- **Grammar Check**: N/A for this contract type.
- **IX Scoring**: IX assigns a confidence score of 0.92 based on clear acceptance criteria and bounded scope.

**Result**: Finalized `IssueContract` ready for dispatch.