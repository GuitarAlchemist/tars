# TARS v2 - Canonical Task

> **"Every phase must demonstrably improve this one task."**

## The Constraint

TARS is at risk of becoming excellent at thinking about thinking before it has proven it can outperform a human on one narrow, boring task.

To prevent architectural drift without killing ambition, we define a **canonical task** that serves as the grounding metric for all development. The task must be one TARS can **run today** — a canonical task that cannot be executed measures nothing.

---

## Canonical Task: "Solve Curated F# Problems"

### Task Description

Given the curated F# problem bank (19 problems: 5 basic, 6 intermediate, 4 advanced, 4 expert, as of 2026-09-13):
1. **Generate** a solution with the configured LLM
2. **Compile** it with `dotnet fsi`
3. **Validate** it against the problem's checks
4. **Record** each outcome into the pattern-outcome store, so the self-improvement loop learns from it
5. **Persist** the run summary for comparison across runs

### Benchmark Command

```bash
# Run the suite (needs a configured LLM; see LlmFactory)
dotnet run --project src/Tars.Interface.Cli -- benchmark code run

# Narrow it while iterating
dotnet run --project src/Tars.Interface.Cli -- benchmark code run --difficulty basic --max 5

# Offline: problem bank + latest result, and run history
dotnet run --project src/Tars.Interface.Cli -- benchmark code status
dotnet run --project src/Tars.Interface.Cli -- benchmark code report
```

Results are written to `~/.tars/benchmark_results/run_<timestamp>.json` (plus `latest.json`).

### Metrics

| Metric | Source field | Direction |
|--------|--------------|-----------|
| Pass rate | `passRate` | Higher is better — the headline number |
| Compile rate | `compileRate` | Higher is better |
| Duration | `totalDurationMs` | Lower is better, secondary |

A phase improves the canonical task when the **full-suite** pass rate rises on the same model, measured before and after.

### Why This Task?

1. **Narrow**: One problem, one solution, clear success/failure
2. **Boring**: No novelty bias — just competent execution
3. **Measurable**: Compiles or doesn't, validates or doesn't
4. **Runs today**: Implemented in `Commands/CodeBenchmark.fs` and `Tars.Evolution/BenchmarkRunner.fs`
5. **Compound**: Outcomes feed `PatternSelector`, closing the loop the rest of the system is built around

---

## Anti-Patterns to Avoid

1. **"It works on paper"** - If it doesn't improve the canonical task, it waits
2. **"We need this for the future"** - Future features that don't help now are deferred
3. **"It's elegant"** - Elegance is a bonus, not a requirement
4. **"It enables X"** - Enablers without demonstrated value are speculative

---

## Current Status

| Question | Answer |
|----------|--------|
| Can the benchmark run today? | **Yes** |
| Is there a full-suite baseline? | **No.** The only saved run (2026-06-21) covered 2 problems. Record a full-suite baseline before claiming any phase improves the task. |

---

## History

The original canonical task (2025-12-26) was **"Refactor an F# File"**, benchmarked by `tars refactor <file.fs> --validate --measure` against `src/Tars.Core/Example.fs`. It was retired on 2026-09-13 (issue #224) because it never ran:

- `tars refactor` was never registered in the CLI, and `RefactorCommand.fs`, `TrsxParser.fs`, `IrCompiler.fs`, `Example.fs` and `CognitionCompilerTests.fs` were all absent from their projects' compile lists — none of it was ever built.
- By 2026-09 the chain no longer type-checked against `Tars.Core.WorkflowOfThought` (missing `Metadata` fields, a reshaped `WotCheck`, a removed `Budget.default`).
- Restoring it would have revived a second `.trsx` parser inside the frozen Metascript project, duplicating the WoT DSL.

The refactoring goal itself is still worthwhile; if it returns, it should be built on the WoT DSL and ship with a benchmark that actually executes.

---

## The Rule

> **If a new mechanism (agent constitution, reflection layer, grammar evolution…) doesn't move the needle on the canonical task, it waits.**

This prevents architectural drift without killing ambition.

---

*Last updated: 2026-09-13*
