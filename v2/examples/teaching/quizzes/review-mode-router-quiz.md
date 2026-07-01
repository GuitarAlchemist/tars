# Quiz: Review Mode Router

## Questions

1. **Classification**: You are reviewing a PR that only updates the `.gitignore` file and adds a few comments to a `README.md`. Which review mode is most appropriate?
2. **Comparison**: What is the primary difference between `fast-review` and `focused-review` in terms of the evidence required?
3. **High Impact**: A proposed change alters the fundamental way TARS handles memory persistence, affecting all existing projects. Why should this be a `decision-gate` instead of a `focused-review`?
4. **Agent Autonomy**: Under what circumstances would you use `silent-classify` for a task? Give a concrete example.
5. **Human Friction**: What is the danger of "over-routing" low-risk tasks into the `focused-review` mode?
6. **Escalation**: You encounter a task where the AI agent states: "I cannot determine if this change follows the new security policy because the policy is ambiguous." Which mode should the agent trigger?

## Answer Key

1. **batch-digest** or **fast-review**. (It is low-risk, reversible, and non-functional).
2. `fast-review` requires surface-level evidence (it works, it's safe). `focused-review` requires deep evidence (logic flow, edge cases, unit tests for specific logic).
3. `decision-gate` is for choices with wide-reaching, permanent, or high-risk consequences that involve weighing trade-offs. `focused-review` is for deep logic within a bounded, lower-impact area.
4. Trivial, non-breaking, highly reversible tasks where the human has explicitly granted autonomous permission (e.g., automated daily dependency updates that pass all tests).
5. **Reviewer Fatigue** or **Rubber Stamping**. The human will stop looking closely at the evidence if they are forced to do a "deep dive" on trivial changes.
6. **escalate-review**. The agent has reached a boundary of its capability or policy clarity.
