# Testing Tips (Tars v2)

## Kill stuck testhost locks (Windows)
- Symptom: `MSB3026`/`MSB3021` copy failures on `Tars.*.dll` because `testhost.exe` holds a lock.
- Fix: run `scripts\kill-testhost.ps1` before rerunning tests.
- CI: run tests after a clean or with `--no-build`; add blame hang timeout if needed: `dotnet test --filter Category!=Slow --blame-hang-timeout 60s`.

## Standard test invocations
- Fast suite (skips integration): `dotnet test --filter Category!=Slow`
- Offline golden eval: `dotnet test --filter OfflineEvalTests` (optional CSV via `TARS_EVAL_OUT`).
- Golden demo (SemanticKernel-dependent): currently skipped; see `tests/Tars.Tests/GoldenRun.fs`.
- Faster local build cycle: run `dotnet restore` once, then `dotnet build --no-restore` (or `dotnet test --no-restore`) to avoid repeated restores/timeouts in constrained environments.

## Metrics capture during tests
- Set `TARS_METRICS_PATH` to collect runtime metrics (auto-flush on process exit).
- Offline eval emits `eval.case` metrics; `TARS_EVAL_OUT` writes them to CSV.

## One-liner helper
`./test-evolution.ps1` (clean/build, kill testhost, run fast tests, run offline eval).
