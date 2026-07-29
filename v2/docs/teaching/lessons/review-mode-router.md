# `/teach` Lesson: Review Mode Router

## Metadata

```yaml
lesson_id: "teach-review-mode-router-001"
concept_id: "review-mode-router"
concept_name: "Review Mode Router"
level: intermediate
source_artifacts: ["v2/examples/teaching/sessions/teach-anti-rubber-stamp-001.md"]
prerequisites: ["anti-rubber-stamp-review"]
next_concepts: ["verification-horizon", "goodhart-risk"]
```

## Why this matters
The Review Mode Router is the engine of "Anti-Rubber-Stamp Review." It prevents human burnout and "blind approval" by ensuring the human maintainer is only asked for judgment at the appropriate depth. Without it, the maintainer either becomes a bottleneck or a rubber stamp.

## Short explanation
The router classifies every incoming task or PR into one of six modes. This classification determines how the information is presented to the reviewer and what level of evidence is required.

1. **silent-classify**: The agent handles it autonomously; no human review needed (e.g., trivial formatting).
2. **batch-digest**: Multiple low-risk items grouped together for a single "OK".
3. **fast-review**: Surface-level check for reversible, low-impact changes (e.g., docs).
4. **focused-review**: Deep dive into a specific, bounded logic change.
5. **decision-gate**: High-impact architecture or governance choice; requires explicit pros/cons and evidence.
6. **escalate-review**: Risk is too high or outside agent capability; requires immediate human intervention.

## Project example
A PR that updates the `AGENTS.md` file with a new tip.
**Suggested Mode**: `fast-review` or `batch-digest`.
**Why?** It is documentation, easily reversible, and has zero impact on the system runtime.

## Counterexample
Routing a change that modifies the core F# `Tars.Engine.Grammar` as `batch-digest`.
**Risk**: A bug here could break the entire DSL. A batch digest would likely miss subtle logic errors. This should be a `focused-review` or `decision-gate`.

## Common misconception
*Misconception*: "Higher mode (like decision-gate) is always better because it's safer."
*Reality*: Over-routing to `decision-gate` causes reviewer fatigue and "approval blindness." The goal is *appropriate* routing, not maximum routing.

## Worked example: Routine Maintenance
**Task**: Update the version number in `Directory.Build.props` and update a README link.
1. **Analyze**: Are these breaking changes? No. Are they reversible? Yes.
2. **Route**: `batch-digest`.
3. **Execute**: The agent groups these into a single "Maintenance Batch" notification for the human.

## Active recall
**Question**: Name the six review modes and describe the difference between `focused-review` and `decision-gate`.

**Expected answer shape**: List: silent-classify, batch-digest, fast-review, focused-review, decision-gate, escalate-review. `focused-review` is for deep logic in a bounded area; `decision-gate` is for high-impact architecture/governance choices involving trade-offs and wide-reaching effects.

## Quick check
1. Which mode is best for a critical security patch in the authentication layer?
2. What mode avoids "rubber stamping" by grouping minor, related tasks?
3. True or False: `silent-classify` means the change is never recorded.

## Teach-back prompt
Explain the Review Mode Router concept as if you were teaching it to a new maintainer who is feeling overwhelmed by PR notifications.

## Scenario exam
**Scenario**: An agent proposes a new MCP tool that allows TARS to delete files in the repository to "clean up artifacts."

**Decision to make**: Which review mode should be assigned?

**What concept applies?** Review Mode Router / Anti-Rubber-Stamp.
**What evidence is needed?** Security policy, impact analysis, "blast radius" check.
**What would be a bad decision?** Routing this as `fast-review` or `silent-classify`.

## Mastery rubric
```text
0 = not introduced
1 = recognizes the 6 mode names
2 = can explain the difference between fast and focused review
3 = can correctly route a standard PR
4 = can identify when an agent has mis-classified a high-risk task
5 = can design routing rules for new task types
```
