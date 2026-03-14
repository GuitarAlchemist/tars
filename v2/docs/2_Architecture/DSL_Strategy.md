# DSL Strategy: Metascripts, WoT, and the Promotion Pipeline

**Status**: ACTIVE
**Date**: 2026-03-11
**Decision**: WoT DSL is the primary workflow language. Metascripts are retained as a lightweight scripting mode but receive no further investment.

---

## Context

TARS has two workflow definition systems:

1. **Metascripts** (`Tars.Metascript`) -- YAML-based workflow definitions (`.tars` files) and rich block-based scripts (`.trsx`) with polyglot execution (F#, Python, shell commands).

2. **Workflow-of-Thought DSL** (`Tars.DSL.Wot` + `Tars.Core.WorkflowOfThought`) -- Typed workflow definitions (`.wot.trsx` files) with compiler support, execution traces, and integration with the promotion pipeline.

Both solve the same problem: defining multi-step reasoning workflows for TARS agents. This document clarifies their roles going forward.

---

## The Decision

### WoT DSL is the primary workflow language

All new reasoning workflow features go into WoT. It has the full 5-layer semantic stack:

| Layer | What | Where |
|-------|------|-------|
| 1. Surface Grammar | `.wot.trsx` syntax, parser | `Tars.DSL.Wot.WotParser` |
| 2. Typed AST | `WotStep`, `WotWorkflow`, typed IR | `Tars.Core.WorkflowOfThought.WotTypes` |
| 3. Constraints | Variable resolution, schema validation | `Tars.Core.WorkflowOfThought.VariableResolution` |
| 4. Operational Semantics | Execution, tracing, golden runs | `Tars.Core.WorkflowOfThought.WotExecutor` |
| 5. Evolution Metadata | Promotion level, lineage, mutation | `Tars.DSL.Wot.WotAST.DslEvolutionMetadata` |

WoT integrates with the Compound Engineering promotion pipeline:

- Patterns observed in WoT traces are fed into `PromotionPipeline.run`
- The Grammar Governor gates promotions using 8 criteria
- Round-trip validation proves abstractions can expand without semantic loss
- Patterns climb the staircase: `Implementation -> Helper -> Builder -> DslClause -> GrammarRule`
- Promoted patterns become new WoT constructs with compiler support

This closed loop -- **execute, trace, score, compile, promote, govern** -- is what makes WoT a living, evolving language. Metascripts have no equivalent mechanism.

### Metascripts are retained as a lightweight scripting mode

Metascripts remain useful for:

- **Quick ad-hoc workflows** that don't need tracing or evolution
- **Polyglot execution** -- F# Interactive, Python, shell commands in one script
- **Backward compatibility** -- existing `.tars` files continue to work
- **`tars run <script.tars>`** -- the simplest way to define a sequence of steps

### No further investment in Metascript engine

The following are explicitly NOT planned for Metascripts:

- No new step types beyond what exists today
- No execution tracing or golden run support
- No integration with the promotion pipeline
- No compiler or type-checking for metascript syntax
- No LSP/IDE support for `.tars` files

If a metascript pattern proves valuable through repeated use, the correct path is to rewrite it as a WoT workflow where it can be traced, scored, and promoted.

---

## Rationale

### What WoT has that Metascripts don't

| Capability | Metascript | WoT DSL |
|-----------|-----------|---------|
| Typed AST | No | Yes (`WotStep`, `WotWorkflow`) |
| Compiler | No | Yes (`WotCompiler`) |
| Execution traces | No | Yes (canonical trace events) |
| Golden run comparison | No | Yes (`GoldenTraceStore`) |
| Schema validation | No | Yes (JSON Schema on outputs) |
| Variable resolution | Basic `${var}` | Typed with scope chains |
| Evolution metadata | No | Yes (Layer 5) |
| Promotion pipeline | No | Yes (7-step CompoundCore loop) |
| Round-trip validation | No | Yes (Jaccard + LLM semantic) |
| MCP tool exposure | No | Yes (`McpPatternResources`) |

### What Metascripts have that WoT doesn't (and why that's OK)

| Capability | Status |
|-----------|--------|
| Polyglot blocks (F#, Python, shell) | Useful but niche. WoT can delegate to tools. |
| YAML simplicity | Lower learning curve, but less powerful. |
| Macro system | `MacroRegistry` exists but is lightly used. |

The polyglot execution capability is the one area where Metascripts genuinely offer something WoT doesn't. However, WoT's tool-call mechanism (`kind: "tool"` steps) can invoke the same operations through the tool registry, making polyglot blocks a convenience rather than a necessity.

---

## Migration Path

There is no forced migration. Existing metascripts continue to work. The natural migration happens through the promotion pipeline:

1. A pattern is first used as a metascript (ad-hoc, untraced)
2. If the pattern recurs, a developer rewrites it as a WoT workflow
3. The WoT workflow gets traced, scored, and enters the promotion pipeline
4. If it meets 6/8 criteria and passes round-trip validation, it gets promoted
5. Eventually it becomes a first-class DSL construct or grammar rule

This mirrors the Compound Engineering staircase -- patterns earn their way up through demonstrated value, not top-down design.

---

## Architecture Map

```
User writes workflow
        |
        v
  +-----------+     +---------------+
  | Metascript |     | WoT DSL       |
  | (.tars)    |     | (.wot.trsx)   |
  | Simple,    |     | Typed, traced, |
  | untraced   |     | evolvable     |
  +-----+------+     +-------+-------+
        |                     |
        v                     v
  [Execute steps]       [WoT Executor]
        |                     |
        |                     v
        |             [Canonical Trace]
        |                     |
        |                     v
        |             [Pattern Compiler]
        |                     |
        |                     v
        |             [Promotion Pipeline]
        |              (7-step CompoundCore)
        |                     |
        |                     v
        |             [Grammar Governor]
        |              (8 criteria gate)
        |                     |
        |                     v
        |             [Round-trip Validation]
        |                     |
        |                     v
        |             [Promoted Pattern]
        |              (Helper -> Builder -> DSL -> Grammar)
        |                     |
        v                     v
  [Result]            [Evolved WoT DSL]
```

---

## Related Files

### Metascript (stable, no new development)
- `src/Tars.Metascript/Parser.fs` -- YAML/block parser
- `src/Tars.Metascript/Engine.fs` -- Step execution engine
- `src/Tars.Metascript/V1Parser.fs` -- Rich block parser (`.trsx`)
- `src/Tars.Metascript/V1Executor.fs` -- Polyglot executor
- `src/Tars.Metascript/MacroRegistry.fs` -- Reusable workflow macros

### WoT DSL (active development)
- `src/Tars.DSL/Wot/WotParser.fs` -- WoT syntax parser
- `src/Tars.DSL/Wot/WotCompiler.fs` -- WoT compiler
- `src/Tars.DSL/Wot/WotAST.fs` -- AST types including evolution metadata
- `src/Tars.Core/WorkflowOfThought/WotExecutor.fs` -- Execution engine
- `src/Tars.Core/WorkflowOfThought/WotTypes.fs` -- Core types
- `src/Tars.Core/WorkflowOfThought/VariableResolution.fs` -- Typed variables

### Promotion Pipeline (active development)
- `src/Tars.Evolution/PromotionTypes.fs` -- Staircase levels, criteria, governance
- `src/Tars.Evolution/GrammarGovernor.fs` -- 8-criteria gating
- `src/Tars.Evolution/RoundtripValidation.fs` -- Semantic round-trip proof
- `src/Tars.Evolution/PromotionPipeline.fs` -- 7-step CompoundCore loop
- `src/Tars.Evolution/StructuredOutput.fs` -- Strict JSON output schemas
- `src/Tars.Evolution/PatternCompiler.fs` -- Trace-to-pattern compilation
- `src/Tars.Evolution/RetroactionLoop.fs` -- Execute-and-learn cycle

### CLI
- `tars wot <file>` -- Execute a WoT workflow
- `tars run <file>` -- Execute a metascript
- `tars promote [status|lineage|run|report]` -- Promotion pipeline

### MCP Tools
- `promotion_status` -- Query promotion pipeline state
- `promotion_lineage` -- Query governance decisions
- `run_promotion_pipeline` -- Trigger pipeline programmatically
- `list_patterns` / `get_pattern` / `suggest_pattern` -- Pattern library
