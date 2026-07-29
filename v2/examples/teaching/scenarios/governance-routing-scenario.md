# Scenario Exam: Governance Routing

## Metadata
- **Topic**: Review Mode Router
- **Context**: Governance & Routing
- **Complexity**: Intermediate

## Scenario
The TARS system is integrated with a new external API for real-time market data. An agent proposes a change to the `Tars.Engine.FSharp.DataSources` project that:
1. Adds a new F# module to handle this API.
2. Changes the default timeout for *all* data sources from 30 seconds to 5 seconds to "improve responsiveness."
3. Updates the `README` with the new API details.

The agent has tagged this as a `fast-review` because "it's just adding a new source and a small configuration tweak."

## Examination Questions

1. **Routing Evaluation**: Is the agent's classification of `fast-review` correct? Why or why not?
2. **The Hidden Risk**: Which part of this change carries the highest risk of system-wide regression?
3. **Corrective Routing**: If you were the Routing Agent, which mode would you assign to this PR?
4. **Evidence Requirements**: What specific evidence should the agent provide before you approve the "timeout change"?
5. **Teach-back**: Explain to the agent why "improving responsiveness" by changing a global default requires more than a `fast-review`.

## Answer Key & Rubric

1. **No**. While adding a new module and updating the README might be `fast-review`, changing a global default for all data sources is high-impact.
2. **Part 2**: Changing the default timeout from 30s to 5s. This could break every existing data source that relies on the longer timeout (e.g., slow legacy APIs).
3. **decision-gate** (because of the global configuration change) or at least **focused-review** with explicit evidence for the timeout impact.
4. Evidence:
   - Impact analysis of current data sources (how many take > 5s today?).
   - Justification for the 5s choice.
   - A plan for "local overrides" if a specific source needs 30s.
5. **Teach-back Rubric**:
   - **Score 1-2**: Focuses only on the new module.
   - **Score 3-4**: Identifies that global changes are risky.
   - **Score 5**: Explains the difference between "local addition" (low risk) and "global mutation" (high risk/decision-gate), and highlights the need for empirical evidence (data on current timeouts).
