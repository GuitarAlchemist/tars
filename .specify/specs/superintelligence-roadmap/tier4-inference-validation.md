# Tier4 Inference & Validation Workflow

This playbook explains how Tier4 iterations capture real inference telemetry, how those metrics flow through the Spec Kit harness, and where operators can inspect the results. Use it as the reference when verifying superintelligence progress or debugging failed autonomous runs.

## Runtime Flow
- The Tier4 metascripts call the telemetry-aware `TARS.AI.Inference` engine via the `ollamaCompatibleInference` pathway. Each invocation produces an `OllamaResponse.metrics` payload containing token counts, latency (ms), CUDA usage, and the deterministic context keys used by downstream agents.
- The Spec Kit harness converts that diagnostics payload into evolution metadata. During a run, `PersistentAdaptiveMemory.captureTelemetry` flattens the map structure and canonicalises arrays (for example, `top_terms` becomes a comma-separated string). The flattened keys are prefixed (`inference.metrics.token_count`, `inference.metrics.analysis_elapsed_ms`, etc.) so they can be merged with other harness metrics.
- When the harness finishes, the Spec Improvement service writes the iteration snapshot to both persistence layers:
  - `output/adaptive_memory_spec-kit.jsonl` receives a new entry whose `inferenceTelemetry` object mirrors the flattened metrics.
  - `.specify/ledger/iterations/<timestamp>_<runId>.json` now includes the same telemetry under `inferenceTelemetry` and in the general `metrics` dictionary so dashboards and alerts can consume it.

## Validation Checklist
1. Run a Tier4 spec iteration  
   `TARS_SPEC_OVERRIDE=superintelligence-roadmap dotnet run --project src/TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli -- auto-loop`
2. Confirm the ledger entry exists and contains telemetry  
   `jq '.inferenceTelemetry' .specify/ledger/iterations/latest.json`
3. Inspect adaptive memory for long-term learning signals  
   `tail -n1 output/adaptive_memory_spec-kit.jsonl | jq '.inferenceTelemetry'`
4. Track live metrics during the run via the CLI dashboard (`dotnet run --project src/TarsEngine.FSharp.Cli/TarsEngine.FSharp.Cli -- auto-dashboard`) to ensure `inference.metrics.*` fields update alongside consensus and critic scores.

## Key Metric Fields
- `inference.metrics.token_count` — number of prompt tokens processed for the request.
- `inference.metrics.distinct_terms` — distinct token count driving context salience.
- `inference.metrics.analysis_elapsed_ms` — end-to-end inference latency for the run.
- `inference.metrics.context_length` — size of the generated deterministic context window.
- `inference.metrics.top_terms` — comma-separated list of the highest salience terms detected during analysis.

## Planner Integration
- The Spec Kit next-step planner now interprets latency and token pressure when ranking candidates. Iterations with sustained slow inference (≥800 ms) or bloated context windows boost tasks tagged with inference/CUDA keywords and lightly penalise unrelated work until the telemetry stabilises.
- If the ledger shows `inference.metrics.used_cuda = false`, the planner gives additional weight to CUDA remediation features and flags the condition in recommendation rationales.
- Trend calculations use the last 10 ledger records; ensure `.specify/ledger/iterations` remains populated so the planner can compute pressure scores.

## Operational Notes
- Telemetry values are culture-invariant and safe for JSON ingestion; times are recorded in milliseconds, and dates use ISO-8601.
- Arrays are normalised to comma-separated strings. If full fidelity is required, consult the raw metascript artifacts in `output/versions/<runId>/`.
- When CUDA is unavailable, the engine still emits telemetry with `inference.metrics.used_cuda = false`, enabling automatic fallbacks without breaking the ledger schema.
