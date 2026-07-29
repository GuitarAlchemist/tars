# `/teach` Lesson: Verification Horizon

## Metadata

```yaml
lesson_id: "teach-verification-horizon-001"
concept_id: "verification-horizon"
concept_name: "Verification Horizon"
level: advanced
source_artifacts: ["v2/docs/design/source-grounded-second-brain-synthesis.md", "v2/docs/contracts/evidence-bundle.contract.md"]
prerequisites: ["review-mode-router", "anti-rubber-stamp-review"]
next_concepts: ["causal-scorecards"]
```

## Why this matters
The "Verification Horizon" is the point where a human can no longer verify an AI's claim based on a summary alone. If we cross this horizon without providing raw evidence, the human is forced into a "rubber stamp" position because they cannot see the "how" or "why" behind the "what."

## Short explanation
As tasks become more complex, the distance between the **AI's Conclusion** and the **Raw Evidence** increases.

- **Near Horizon**: A human can verify the claim in seconds (e.g., "I fixed the typo in line 10").
- **Far Horizon**: A human needs deep context to verify (e.g., "I optimized the memory usage of the vector store by 15%").
- **Beyond the Horizon**: The human cannot verify the claim without external tools or massive effort (e.g., "This 500-line rewrite is mathematically identical to the original").

To manage the horizon, we use **Evidence Bundles** and **Source Grounding**.

## Project example: Second-Brain Synthesis
When TARS synthesizes a "Finding" from 50 different research papers, it must provide the **Verification Horizon** markers:
- Which specific quote supports this claim?
- What is the confidence score?
- Link to the raw PDF/Markdown source.

Without these, the Finding is "Beyond the Horizon."

## Counterexample
An agent provides a summary: "I have verified that all 200 tests pass and the architecture is sound."
**Problem**: This is a "Summary Wall." The human has to take the agent's word for it. The verification horizon has been hidden.

## Common misconception
*Misconception*: "If the AI is 99% accurate, we don't need to worry about the verification horizon."
*Reality*: Accuracy is not the same as verifiability. Even a 100% accurate AI needs to show its work so the human can maintain *governance* and *mental models* of the system.

## Worked example: High-Impact Bug Fix
1. **AI Claim**: "Fixed a race condition in the message bus."
2. **Horizon Check**: This is "Far Horizon." A human cannot see a race condition fix in a summary.
3. **Evidence Required**:
   - A failing test case that reproduces the race.
   - A trace of the message flow before/after the fix.
   - The specific diff of the lock/concurrency logic.

## Active recall
**Question**: What happens to a human reviewer when an AI's claims move "Beyond the Horizon" without raw evidence?

**Expected answer shape**: The human is forced to either "rubber stamp" (approve without understanding) or "reject everything" (bottleneck). They lose the ability to provide meaningful governance.

## Quick check
1. Define the "Summary Wall."
2. When should an agent provide an "Evidence Bundle"?
3. True or False: The Verification Horizon is fixed for every human. (Answer: False, it depends on the human's expertise and the task complexity).

## Teach-back prompt
Explain the Verification Horizon to an AI agent that is providing overly-brief summaries of complex code changes.

## Scenario exam
**Scenario**: An agent proposes a complex refactor of the `Tars.Engine.FSharp.Core` project. It provides a 1-paragraph summary and a 2,000-line diff.

**Decision to make**: Is this "Beyond the Horizon"? What evidence should you demand before reviewing the diff?

**What concept applies?** Verification Horizon / Evidence Ladder.
**What evidence is needed?** Decomposition of the diff into logical chunks, unit test results for each chunk, "Before/After" performance metrics, and a mapping of "Old Logic" to "New Logic."
**What would be a bad decision?** Attempting to read the 2,000-line diff without grounded evidence, or trusting the summary.

## Mastery rubric
```text
0 = not introduced
1 = can define Near and Far horizons
2 = can identify a "Summary Wall"
3 = can request specific evidence to pull a claim back "Inside the Horizon"
4 = can design evidence requirements for a new, high-risk agent skill
5 = can detect when a system's "Abstraction Level" has permanently outpaced human verification capacity
```
