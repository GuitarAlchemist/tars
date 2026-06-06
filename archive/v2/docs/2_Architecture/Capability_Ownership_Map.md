# Capability Ownership Map (TARS v2)

Single-source ownership prevents shadow implementations and cargo cult drift.

## Kernel / Event Bus / Routing
- **Owner:** `Tars.Kernel`
- **Notes:** Do not duplicate kernel/context/registry logic in `Tars.Core`; use the kernel interfaces from `Tars.Kernel`.

## Tool Registry & Execution
- **Owner:** `Tars.Tools`
- **Notes:** Tool registry/health/circuit breakers live here; avoid parallel registries in agents.

## Agent Workflow & Budget Governance
- **Owner:** `Tars.Core`
- **Notes:** AgentWorkflow CE, BudgetGovernor, Metrics, TraceRecorder.

## Grammar / Output Validation
- **Owner:** `Tars.Core` (interim `GrammarDistill.fs` placeholder)
- **Notes:** Plan to replace regex-based distill with schema/GBNF-backed validator behind an interface.

## Metascript DSL & Execution
- **Owner:** `Tars.Metascript`
- **Notes:** Parsing, validation, execution templates.

## LLM Adapters
- **Owner:** `Tars.Llm`
- **Notes:** AsyncResult-based LLM client interfaces and implementations.

## CLI / TUI
- **Owner:** `Tars.Interface.Cli`
- **Notes:** Encoding/ASCII fallbacks, diagnostics, chat/evolve/demo commands.

## Evolution / Self-Play
- **Owner:** `Tars.Evolution`
- **Notes:** Curriculum/Executor/Evaluator loop; do not introduce separate evolution loops elsewhere.
