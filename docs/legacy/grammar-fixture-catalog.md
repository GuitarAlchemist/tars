# TARS Grammar, FLUX, and Metascript Fixture Catalog

This document indexes the catalog of core testing and verification fixtures spanning TARS V1 and V2. These fixtures demonstrate grammar compliance, `.tars` compatibility, `.trsx` execution paths, and the integration of closure contracts.

---

## 1. `.tars` Metascript Fixtures

These fixtures showcase the legacy TARS block format structure, featuring metadata headers and block segment delimiters (`{ ... }`).

| Fixture Path | Description | Key Features Demonstrated | V2 Compatibility Status |
|---|---|---|---|
| `v1/Examples/agent_with_inline.tars` | Legacy agent with nested inline context-free grammars. | Grammar block, `meta` descriptor, and multi-line parsing rules. | **Fully Compatible** via `V1Parser.fs`. Inline grammar is extracted to a `.ebnf` temporary asset. |
| `v1/Examples/agent_meta_script.tars` | Standard multi-block file featuring text commands and query prompts. | `query` blocks, `command` execution sequences, and custom block names. | **Fully Compatible** via `V1Executor.fs` mapping to local task execution. |
| `v1/Examples/sample.tars` | A foundational base example of the legacy format. | Simplified block structures and parameter pairs (e.g., `key="value"`). | **Fully Compatible**. Used as a regression fixture in the parser tests. |

---

## 2. `.trsx` Workflow Fixtures

These XML-based or unified block formats describe workflow structures, agent-to-agent interactions, and parallel node scheduling.

| Fixture Path | Description | Key Features Demonstrated | V2 Compatibility Status |
|---|---|---|---|
| `v2/examples/got_test.wot.trsx` | Graph of Thoughts (GoT) workflow test fixture. | Dynamic node branching, parallel execution pathways, state transitions. | **Native V2 Support** via the Cortex `.wot.trsx` adapter pipeline. |
| `v2/examples/incident_analysis.wot.trsx` | Automated incident triage and analysis workbook. | Sequential task orchestration, schema verification, and tool-calling interfaces. | **Native V2 Support**. Acts as a critical tracer bullet for agent verification. |
| `v1/Examples/agent_with_inline_unified.trsx` | Combined grammar and step schema workflow from V1. | Synthesized XML/block grammar nodes mapping to specific action definitions. | **Translated** via `TrsxParser.fs` into a Cortex `WorkflowStep` IR. |
| `v1/Examples/unified-evolutionary-closure-demo.trsx` | Multi-agent closure validation script. | Inline execution conditions, dependency tracing, validation gates. | **Partially Supported**. Executed as compiled sequence steps. |

---

## 3. `.flux` Multi-Modal & Scheduling Fixtures

V1 FLUX files represent parallel scheduling blocks and dynamic self-modifying interfaces.

| Fixture Path | Description | Key Features Demonstrated | V2 Compatibility Status |
|---|---|---|---|
| `v1/Examples/self_modifying_ui.flux` | Standard FLUX scheduling flow containing reactive UI nodes. | Parallel task triggers, state dependencies, dynamically synthesized loops. | **Deprecated**. Executed as equivalent task workflows in the Cortex executor. |
| `v1/Examples/pure_flux_self_modifying_ui.flux` | A pure, no-compilation pipeline script representing direct UI state manipulation. | Reactive node rendering, stream subscriptions. | **Deprecated**. Replaced by native F# reactive patterns in the UI runtime. |

---

## 4. WebAPI & Closure Factory Demos

These fixtures prove integration between parsed grammar parameters and code execution backends (Closures).

| Fixture / Source Path | Description | Key Features Demonstrated | V2 Compatibility Status |
|---|---|---|---|
| `TarsEngine.FSharp.FLUX.Standalone/` | Standard V1 standalone closure orchestration. | Dynamic code loading, F# reflection, input/output validation. | **Replaced** by clean `Tars.Metascript` assemblies and V2 Closures. |
| `v2/src/Tars.Metascript/V1Executor.fs` | V2 host adapter for legacy metascript execution. | Intercepts block content and redirects it to valid .NET methods or terminal commands. | **Fully Supported**. Active bridging code in TARS V2. |

---

## 5. Grammar Evolution and Mutation Fixtures

Fixtures that exercise the automated synthesis and optimization of grammar schemas.

| Fixture Path | Description | Key Features Demonstrated | V2 Compatibility Status |
|---|---|---|---|
| `unified-grammar-evolution-demo.trsx` | Orchestrated sequence that tests mutating and evolving BNF grammar rules. | Generates new rule constraints, checks syntax, and ranks performance. | **Moved to IX** (GuitarAlchemist/ix#189). TARS coordinates the execution logs. |
| `v2/examples/tars-v1-agent-trace.example.json` | Captured trace from V1 grammar-driven execution. | Raw inputs, generated structured output blocks, and error/correction metrics. | **Supported** as a baseline evaluation test case in the test harness. |

---

## 6. How to Run Compatibility Tests

To verify that the compatibility layer is successfully reading and executing the legacy inventory, run:

```bash
# Verify v2 CLI can parse legacy tars files
dotnet run --project v2/src/Tars.Interface.Cli/Tars.Interface.Cli.fsproj -- parse v1/Examples/agent_with_inline.tars

# Verify the Cortex Workflow of Thought parses and compiles the TRSX fixtures
dotnet test v2/tests/Tars.Tests --filter "WorkflowTests"
```
