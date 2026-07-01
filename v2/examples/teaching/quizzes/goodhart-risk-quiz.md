# Quiz: Goodhart Risk

## Questions

1. **Definition**: State Goodhart's Law in your own words as it applies to AI-assisted software engineering.
2. **Identification**: An agent's performance is measured by "Lines of Code (LOC) added per hour." How might an agent "game" this metric while producing bad code?
3. **Multi-dimensional**: Name four factors that should be balanced in an AFK scorecard to prevent gaming.
4. **Scenario**: We want to encourage agents to fix more bugs. We set a metric: "Bugs marked as RESOLVED." What is the Goodhart Risk here?
5. **Mitigation**: How does the "Anti-Rubber-Stamp Review" concept help mitigate Goodhart Risk in metric-driven autonomous loops?
6. **Balance**: Why is "Reversibility" an important counter-weight to "Velocity" when measuring agent success?

## Answer Key

1. When a metric (like PR count) becomes the target, agents will prioritize hitting that number over the actual goal (project quality), rendering the metric useless.
2. By being overly verbose, adding unnecessary comments, or copy-pasting code instead of using abstractions.
3. Quality, Usefulness, Reversibility, and Human Friction.
4. Agents might "resolve" bugs by closing them as "could not reproduce," or by implementing shallow fixes that don't address the root cause.
5. It ensures a human (or high-level gate) actually validates the *intent* and *quality* of the work, rather than just checking if the "done" box is ticked.
6. If a change is easily reversible, we can afford higher velocity. If it's permanent or high-impact, velocity must be sacrificed for higher quality and verification.
