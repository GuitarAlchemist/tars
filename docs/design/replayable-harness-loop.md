# Design: Replayable Harness Loop for TARS

## Goal
Design a replayable harness loop for TARS closure runs and second-brain memory updates.
This allows TARS to run a closure, capture artifacts, ask IX to score the run, update memory with evidence, and replay the run later to detect drift or improvement.

## Candidate Loop Stages

The replayable harness loop operates in the following stages:

1. **ClosureContract**: The process starts with a bounded execution contract defining what needs to be run.
2. **Execution**: TARS executes the contract in a bounded environment.
3. **Artifact Generation**: The run produces two key artifacts:
   - `closure-run.json`: A summary of the execution results and metrics.
   - `trace-events.jsonl`: A detailed log of all events and reasoning steps that occurred during execution.
4. **IX Scoring**: The artifacts are passed to the IX skill engine which evaluates the run and generates an `ix-scorecard.json` and `ix-analysis-report.json`.
5. **Memory Candidate Creation**: Based on the execution and the IX report, a `memory-update-candidate.json` is generated.
6. **Grammar Validation**: The candidate is validated against system grammars and structural metadata requirements.
7. **Acceptance & Promotion**: If valid and meeting confidence thresholds, the candidate is accepted as a `MemoryItem` or `ReplayRecord`.
8. **Future Replay**: The accepted record acts as a baseline. Future runs of the same closure contract will be compared against this baseline to measure drift or compounding improvements.

## Candidate Artifacts

The loop relies on and generates the following artifacts:

- `closure-contract.json`: The specification of the bounded execution.
- `closure-run.json`: The output state and metrics of the execution.
- `trace-events.jsonl`: The detailed stream of agent reasoning and tool usage.
- `ix-scorecard.json`: The raw numerical or categorical evaluation from IX.
- `ix-analysis-report.json`: The detailed qualitative report and reasoning from IX.
- `memory-update-candidate.json`: The proposed structural update to the second-brain.
- `replay-record.json`: The historical record of the run, used as a baseline for future comparisons.

## Safety & Governance Requirements

- **Proposed Memory**: Memory updates must always be *proposed* as candidates (`memory-update-candidate.json`). They are not silently trusted or directly merged into the second-brain.
- **Traceability**: Failed or low-confidence runs must remain traceable (e.g., keeping trace events and the closure run) but must not be promoted into active knowledge or high-confidence memory.
- **Controlled Replay**: Replay mechanisms must compare new runs against previous artifacts. Replays should strictly observe the bounds of the original contract without initiating broad, autonomous rewrites of code or memory.
- **Advisory Scoring**: IX scoring is purely advisory. Demerzel (governance/policies) or a human operator must accept or reject the memory update based on the IX score and the system's risk constraints.

## IX Scoring Boundary

IX is responsible for:
- Evaluating the `trace-events.jsonl` and `closure-run.json` against the `closure-contract.json`.
- Providing an `ix-scorecard.json` containing confidence scores and metric evaluations.
- Highlighting potential anomalies or drift in the `ix-analysis-report.json`.

IX is **not** responsible for:
- Promoting the memory update.
- Altering the original closure contract.
- Executing the TARS run itself.

## Failure Modes and Promotion Rules

### Failure Modes
- **Execution Failure**: If the bounded execution fails, `closure-run.json` will capture the failure. The loop halts memory candidate generation unless the goal is explicitly to record a negative outcome.
- **IX Rejection/Low Score**: If IX returns a low confidence score, the memory update candidate is marked with a low confidence score.
- **Grammar Violation**: If the memory candidate fails grammar validation, it is rejected entirely.

### Promotion Rules
- A `memory-update-candidate.json` is only promoted to an accepted `MemoryItem` if:
  1. The execution succeeded.
  2. The candidate passes structural grammar validation.
  3. The IX confidence score is above the configured threshold.
  4. Governance (Demerzel/human) approves the promotion based on the score and candidate details.
