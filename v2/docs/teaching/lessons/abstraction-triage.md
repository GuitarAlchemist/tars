# `/teach` Lesson: Abstraction Triage (Over vs. Under)

## Metadata

```yaml
lesson_id: "teach-abstraction-triage-001"
concept_id: "abstraction-triage"
concept_name: "Abstraction Triage"
level: intermediate
source_artifacts: ["docs/triage/methodology-guard-triage-reasoning.md", "docs/methodology/agentic-engineering.md"]
prerequisites: ["verification-horizon", "review-mode-router"]
next_concepts: ["goodhart-risk"]
```

## Why this matters
Triage is the act of deciding how much attention a task needs. If we **over-abstract**, we lose the specific evidence needed to verify a change (creating a "Summary Wall"). If we **under-abstract**, we drown the maintainer in trivial details that should have been handled by the agent. Getting this balance right is the difference between a productive AFK loop and a "rubber-stamp" or "bottleneck" failure.

## Short explanation
Abstraction Triage is the skill of finding the "Goldilocks Zone" of information density for a reviewer.

- **Over-abstraction (The Summary Wall):** Providing a high-level conclusion without the underlying evidence.
  - *Symptom*: "I have improved the system performance." (No metrics, no traces).
  - *Risk*: Hidden bugs, "beyond the horizon" reasoning.
- **Under-abstraction (The Detail Flood):** Providing raw data without a narrative or reasoning.
  - *Symptom*: A 5,000-line diff with no explanation of why it changed.
  - *Risk*: Reviewer fatigue, missing the "big picture" architecture shifts.

## Project example
**Balanced Triage**: An agent proposes a refactor to `Tars.Llm.Prompt`.
- **Finding**: Methodology Guard failed (missing "Performance Impact" field).
- **Reasoning**: TARS notes the failure but observes that the diff contains specific benchmark results and a clear "Before/After" logic comparison.
- **Decision**: Suggested `focused-review` with a pointer to the specific evidence in the diff.

## Counterexample
**Over-abstraction**: An agent closes an issue with: "Completed according to instructions. All tests passed."
- **Problem**: The maintainer has no idea *what* was done or *how* it was tested. This is a "Summary Wall" that forces a rubber-stamp.

## Common misconception
*Misconception*: "More detail is always better for safety."
*Reality*: Too much detail (under-abstraction) is its own form of noise. It hides the "Intent" behind the "Action." Triage should surface the *Intent* and link to the *Evidence*.

## Worked example: The Missing Field
1. **Scenario**: A PR is opened. The Methodology Guard flags that the "Risk Assessment" section is empty.
2. **Under-abstracted response**: "The guard failed. Here is the log: [100 lines of JSON]."
3. **Over-abstracted response**: "The guard failed, but it's probably fine."
4. **Abstraction Triage response**: "The Guard detected a missing Risk Assessment. While the template is incomplete, the PR only modifies documentation links (low risk). I suggest `fast-review` despite the structural failure."

## Active recall
**Question**: What is the "Summary Wall" and why is it a symptom of over-abstraction?

**Expected answer shape**: The Summary Wall is a high-level conclusion presented without supporting evidence. It is over-abstraction because it removes the necessary detail a human needs to verify the claim.

## Quick check
1. True or False: Triage should always favor the most detailed information available.
2. What is the main risk of "Under-abstraction"?
3. Which review mode is typically suggested for a PR that is "Over-abstracted"? (Hint: It usually needs to be pulled back inside the Verification Horizon).

## Teach-back prompt
Explain to another maintainer why a PR with a "100% green" Methodology Guard pass might still be dangerously "Over-abstracted."

## Scenario exam
**Scenario**: You are triaging a PR from an agent that completely replaces the `Tars.Cortex` state machine. The PR has a perfect template match, but the "Evidence" section says: "Manual verification confirmed correctness."

**Decision to make**: Is this over-abstracted or under-abstracted? What should you do?

**What concept applies?** Abstraction Triage / Verification Horizon.
**What evidence is needed?** The raw logs of the "manual verification," state transition traces, and a breakdown of why the rewrite was necessary.
**What would be a bad decision?** Approving based on the "100% green" Guard status (Rubber-stamping the Summary Wall).

## Mastery rubric
```text
0 = not introduced
1 = can define over and under abstraction
2 = can identify a "Summary Wall" in an agent's report
3 = can balance findings to suggest an appropriate Review Mode
4 = can critique a Methodology Guard's "Green" status if evidence is missing
5 = can coach an agent to provide the correct abstraction level for its task
```

## Feedback notes
- Misconceptions detected:
- Strong points:
- Weak areas:
- Suggested next review:
- Suggested next concept:
