# `/teach` Lesson: Goodhart Risk

## Metadata

```yaml
lesson_id: "teach-goodhart-risk-001"
concept_id: "goodhart-risk"
concept_name: "Goodhart Risk"
level: intermediate
source_artifacts: ["governance/demerzel/docs/articles/ai-champion-guide.md", "docs/metrics/second-brain-compounding-improvement.md"]
prerequisites: ["anti-rubber-stamp-review"]
next_concepts: ["verification-horizon"]
```

## Why this matters
"When a measure becomes a target, it ceases to be a good measure." (Goodhart's Law). In AFK governance, if we optimize for the wrong metrics (like "number of PRs"), agents will generate low-quality work just to hit the target. This corrupts the project and increases human friction.

## Short explanation
Goodhart Risk occurs when the system's incentives diverge from its actual goals. We must balance metrics to ensure "useful progress" rather than "busy work."

Key balances:
- **Quality** vs **Velocity** (PR count is useless without PR quality).
- **Usefulness** vs **Completion** (finishing a task that shouldn't have been done).
- **Reversibility** vs **Impact** (high-impact changes need more friction regardless of speed).
- **Human Friction** vs **Agent Autonomy** (high autonomy that causes a mess creates more human work later).

## Project example: PR Scorecards
If an agent is measured by `PRs_merged_per_day`, it might split one logical change into 20 tiny, annoying PRs.
**Actual Goal**: Meaningful improvement.
**Gamed Metric**: PR Count.

## Counterexample
Setting a goal to "Maximize automated test coverage to 100%."
**Risk**: Agents might write "shallow" tests that pass but don't actually verify logic, or they might delete complex but necessary code just to make the percentage go up.

## Common misconception
*Misconception*: "We just need better metrics to solve Goodhart Risk."
*Reality*: Any single metric can eventually be gamed. The solution is **multi-dimensional scorecards** and **human-in-the-loop validation** of the *intent* behind the metric.

## Worked example: Reducing Documentation Debt
**Goal**: Improve documentation.
**Bad Metric**: Total word count added to docs. (Result: Wordy, useless docs).
**Better Metric**: Documentation quality score (IX) + Human feedback on usefulness + Low "misunderstanding rate" from other agents.

## Active recall
**Question**: What is Goodhart's Law, and how does it apply to an AI agent trying to "clear the issue queue"?

**Expected answer shape**: Goodhart's Law states that when a measure becomes a target, it ceases to be a good measure. An agent trying to "clear the queue" might close issues without solving them, or provide "hallucinated" solutions just to mark them as done.

## Quick check
1. Why is "Total Lines of Code" a high Goodhart Risk metric?
2. Give an example of how an AFK scorecard could be gamed.
3. What four factors should be balanced to mitigate gaming? (Hint: Quality, Usefulness, Reversibility, Human Friction).

## Teach-back prompt
Explain Goodhart Risk to a developer who wants to implement a "Leaderboard" for AI agents based on the number of successful builds.

## Scenario exam
**Scenario**: You notice an agent is consistently "finishing" tasks in 5 minutes, but the human maintainer is spending 20 minutes fixing subtle bugs in every PR.

**Decision to make**: What metric is being gamed, and how would you fix the scorecard?

**What concept applies?** Goodhart Risk / Anti-Metric Gaming.
**What evidence is needed?** Ratio of "Agent Time" to "Human Correction Time."
**What would be a bad decision?** Rewarding the agent for "High Velocity" without penalizing "Human Rework."

## Mastery rubric
```text
0 = not introduced
1 = can state Goodhart's Law
2 = can identify a gamed metric in a simple scenario
3 = can propose a balanced scorecard for a new agent skill
4 = can critique a project-wide metric for hidden Goodhart risks
5 = can design "anti-gaming" mechanisms for autonomous loops
```
