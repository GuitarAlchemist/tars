# TARS Competence Metrics & Offline Eval (v2)

This documents the lightweight metrics and golden eval harness we added to measure cognitive performance over time.

## What’s instrumented (runtime)
- `agent.bind` events: every AgentWorkflow bind records status (`success|partial|failure`) and duration (ms) tagged by agent id.
- `budget.check` events: records whether a budget consume succeeded or was blocked.
- In-memory store; dump to CSV when desired.

## How to collect metrics
1) Auto-flush on exit:
```
set TARS_METRICS_PATH=C:\path\to\metrics.csv
```
2) Manual flush in code:
```
Metrics.dumpCsv @"C:\path\to\metrics.csv"
```
`Metrics.snapshot()` / `Metrics.clear()` are available for ad-hoc use.

CSV columns: `timestamp,agent_id,kind,status,duration_ms,details`.

## Offline golden eval
Signals currently checked:
- ResponseParser tool-call parse
- Chunking overlap sanity
- Budget overrun prevention
- Grammar distillation (JSON shape) sanity

Run:
```
dotnet test --filter OfflineEvalTests
```
Optional CSV:
```
set TARS_EVAL_OUT=C:\path\to\eval.csv
dotnet test --filter OfflineEvalTests
```
Eval emits `eval.case` metrics; failures appear in test output.

## Extending
- Add cases in `tests/Tars.Tests/OfflineEval.fs` (append to `cases` list).
- For richer runtime metrics, call `Metrics.record(kind, status, durationMs, agentId, details)` at call sites.
