# Cargo Cult Code (LLM-Generated) — Risks and Countermeasures

## What is “cargo cult” code?
- Copy/pasted or auto-generated code that mimics shape/patterns without real necessity.
- Redundant layers that duplicate existing capabilities (e.g., shadow kernels, duplicate registries).
- Over-engineered abstractions (multi-tier grammars, unused monads) that aren’t referenced anywhere.
- Placeholder validators or stubs that give a false sense of safety.

## Why it happens (LLM-specific)
- Pattern bias: LLMs tend to emit familiar scaffolding (registry, service layers) even when not requested.
- Context drift: Mixing v1/v2 references leads to “hybrid” artifacts that don’t fit the current architecture.
- Safety padding: Models insert validators/guards to seem robust without integration or tests.
- Speculation: Filling gaps with visionary/roadmap code that isn’t actionable yet.

## Detection patterns
- **Zero references:** Types/files with no usages (`rg` shows nothing).
- **Shadowing core modules:** Parallel implementations of kernel/registry/bus already owned by another project.
- **Untested layers:** Files with no tests and no integration points.
- **Over-wide enums/tiers:** Large tiered DUs/DSLs that are never consumed.
- **Stub validators:** Regex “validators” that don’t parse or enforce schema.

## Countermeasures (process)
- **Reference checks:** Run `rg`/`dotnet` unused detection; delete or quarantine unreferenced artifacts.
- **Single-source ownership:** Declare one project the owner of each concern (e.g., `Tars.Kernel` for kernel/bus; delete local copies).
- **PR checklist:** “Is this used?”, “Does this duplicate existing module?”, “Is there a test?”, “Is this MVP-scoped?”.
- **Timeboxing vision:** Move roadmap/visionary abstractions to docs instead of code until scheduled.
- **Schema-first validation:** Prefer JsonSchema/GBNF-backed validators over ad-hoc regex.
- **ASCII/encoding fallback:** Avoid exotic UI glyphs in CLI/log output to prevent garbling when piped/redirected.

## Countermeasures (tooling)
- **CI lint for unused:** Enable analyzers/`dotnet format /warnon` or F# analyzers to flag unused bindings/files.
- **Dead-code sweeps:** Scheduled `rg` zero-ref sweeps; maintain a “quarantine” folder for review, then delete.
- **Golden traces/tests:** Tie new modules to tests or trace fixtures; reject modules without coverage or wiring.
- **Architecture ownership map:** One owner per capability (kernel, bus, tool registry, grammar). New code must point to the owner.

## What we just did (example clean-up)
- Deleted `Kernel.fs` shadow registry in `Tars.Core` (owned by `Tars.Kernel`).
- Deleted `GrammarTypes.fs` (16-tier evolution DU unused).
- Trimmed `Functional.fs` to only AsyncResult helpers actually used by LLM code.

## When to keep speculative code
- If it’s roadmap-critical and immediately scheduled.
- If it’s documented, test-backed, and wired into a feature flag.
- Otherwise, move to docs (not src) until it’s funded/owned.
