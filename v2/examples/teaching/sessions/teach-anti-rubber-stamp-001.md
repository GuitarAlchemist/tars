# `/teach` Session Example — Anti-Rubber-Stamp Review

## Metadata

```yaml
lesson_id: teach-anti-rubber-stamp-001
concept_id: anti-rubber-stamp-review
date: 2026-07-01
mode:
  - quick-check
  - teach-back
student_role: student-maintainer
```

## Prompt

```text
/teach anti-rubber-stamp-review --quick-check --teach-back
```

## Questions

1. Why can asking the human to approve everything make decisions worse?
2. Which review mode fits a docs-only, reversible, known-safe PR?
3. Which review mode fits a decision that changes the architecture boundary between TARS and IX?
4. What is the difference between fast-review and focused-review?
5. Teach-back: explain anti-rubber-stamp review as if teaching another maintainer.

## Student answers summary

```text
1. Asking the human for everything causes blind approval.
2. Use a docs-only/safe mode.
3. Use an important architecture decision mode.
4. Fast review is surface-level for low-risk items.
5. Anti-rubber-stamp means the AI should verify that decisions are not applied blindly, and that a human or another agent weighed pros and cons using verifiable facts and logic.
```

## Evaluation

```yaml
mastery_level: 3
mastery_delta: 2
status: concept assimilated; routing vocabulary should be strengthened
```

## Strong points

- Correctly identified blind approval as the key risk.
- Correctly recognized low-risk docs work as lightweight review.
- Correctly recognized architecture boundary changes as important decisions.
- Good teach-back emphasis on evidence, logic, and weighing pros/cons.

## Weak areas

- Exact review-mode names should be reinforced:
  - `fast-review`
  - `focused-review`
  - `decision-gate`
- The concept should be framed as routing before the decision, not only checking after the decision.

## Misconception detected

```text
Anti-rubber-stamp review is only a post-decision verification step.
```

Correction:

```text
Anti-rubber-stamp review primarily routes work before the human is asked:
classify, batch, fast-review, focused-review, decision-gate, or escalate-review.
```

## Improved teach-back

```text
The anti-rubber-stamp principle prevents the human from mechanically approving every AI action. The system should first classify the decision type: low-risk, targeted judgment, or major architecture/governance decision. The human should only be asked when their judgment can materially change the result. Important decisions must expose options, risks, verifiable evidence, and the reasoning behind the recommendation.
```

## Next recommended concept

```text
review-mode-router
```

Goal:

```text
Memorize the six review modes and apply them to real project scenarios.
```

## Learning-state update candidate

```json
{
  "concept_id": "anti-rubber-stamp-review",
  "mastery_level_before": 1,
  "mastery_level_after": 3,
  "mastery_delta": 2,
  "misconceptions_detected": [
    "Anti-rubber-stamp review is only a post-decision verification step."
  ],
  "recommended_next_concepts": [
    "review-mode-router",
    "verification-horizon",
    "goodhart-risk"
  ]
}
```
