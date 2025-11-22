# PLAN

## 1) Robust validation + metrics
**Rationale:** Keep iterations reliable.  
**Steps**
- Add validators: `dotnet build`, `dotnet test --no-build` (fail fast).
- Time each stage (planning, validation) and record to `metrics.jsonl`.
- Emit: `id, startedAt, completedAt, durationMs, success, failureReason, model, projectRoot`.
**Acceptance**
- `dotnet run -- validate` returns non-zero on any failing validator.
- `output/metrics/metrics.jsonl` contains a new line per run with the above fields.

## 2) Integrate local LLM via Ollama
**Rationale:** Local-first planning.  
**Steps**
- Use `qwen2.5-coder:latest` from `tars.supervisor.json`.
- On 404, print Ollama error body and suggest `ollama pull <model>`.
**Acceptance**
- `dotnet run -- plan` writes `PLAN.md` and `next_steps.trsx` under `output/versions/<ts>/`.

## 3) Refine feedback loop
**Rationale:** Continuous improvement.  
**Steps**
- After each successful validate, append metrics and keep `SUCCESS` flag.
- On failure, write `FAILED`, keep full `validate.log`.
- Add `report` to summarize totals (OK).
**Acceptance**
- `dotnet run -- report` shows accurate totals after multiple runs.

## 4) Automate rollback
**Rationale:** Fast recovery.  
**Steps**
- `rollback` sets `output/versions/CURRENT` to last `SUCCESS`.
- (Optional) Add hook to `git reset --hard` to that tag/commit.
**Acceptance**
- Running `rollback` updates `CURRENT`; optional hook performs project restore.

## 5) Config & hooks (recommended)
Add to `tars.supervisor.json`:
```json
{
  "validators": [
    "powershell: dotnet build",
    "powershell: dotnet test --no-build"
  ],
  "postValidateHooks": [
    "powershell: if (Test-Path ./output/versions/CURRENT) { Remove-Item ./output/versions/CURRENT -Force }",
    "powershell: (Get-ChildItem ./output/versions -Directory | Sort-Object Name -Descending | Select-Object -First 1).FullName | Out-File ./output/versions/CURRENT -Encoding ASCII"
  ]
}
