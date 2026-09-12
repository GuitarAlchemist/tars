# TARS V1-to-V2 Grammar, FLUX, and Metascript Reconciliation Assessment

This document outlines the architectural status, reconciliation strategy, and ownership boundaries for grammar, FLUX, and metascript systems as TARS transitions from V1 to V2.

---

## 1. Context and Strategic Shift

In **TARS V1**, the system grew to support extensive grammar features, dynamic language switching, fractal grammar generators, mathematical L-systems, and custom script runtimes. While highly capable, this coupling created excessive complexity, performance bottlenecks, and a large runtime footprint.

In **TARS V2**, the project adopts a **lean, local-first, contract-driven architecture**. High-complexity math transforms and heavy evolutionary algorithm optimization are routed to the **IX engine (GuitarAlchemist/ix#189)**. The V2 runtime focuses on robust, predictable execution interfaces, utilizing **constrained decoding (JSON Schema, GBNF/EBNF)** directly at the LLM provider boundary to achieve semantic validation.

---

## 2. V1 Grammar & FLUX Assets Inventory

The legacy corpus contains several major components under the `Tars.Engine.Grammar/` directory, legacy metascript executors, and standalone FLUX engines:

| V1 Module / System | Description | V1 Source Files | TARS V2 Status |
|---|---|---|---|
| **GrammarSource / Resolver** | Basic EBNF grammar representation and FileInfo-based lookup. | `GrammarSource.fs`, `GrammarResolver.fs` | **Replaced** by native `ConstrainedDecoding.fs` in `Tars.Llm` using standardized files in `v2/grammars/`. |
| **LanguageDispatcher** | Multi-language code execution wrapper supporting 9+ langages inside `LANG("...")` blocks. | `LanguageDispatcher.fs` | **Deferred**. Polyglot logic is replaced by sandboxed closures and metascript executors. |
| **RFCProcessor** | Download and extract EBNF grammar blocks directly from RFC text files. | `RFCProcessor.fs` | **Moved to IX** as an off-line pipeline asset or deferred to V3+. |
| **FractalGrammar** | Generates self-similar recursive grammars (Sierpinski, Koch, Dragon Curve). | `FractalGrammar.fs`, `FractalGrammarParser.fs` | **Moved to IX** (GuitarAlchemist/ix#189) as an advanced mathematical algorithm. |
| **FLUX Standalone Engine** | Orchestrates multi-modal logic, scheduling, and custom workflows. | `TarsEngine.FSharp.FLUX/`, `test-flux.flux` | **Deprecated** in favor of the clean Cortex Workflow of Thought (`.wot.trsx`) execution model. |
| **Metascript Runner / Console** | V1 execution of `.tars` scripts with text, query, and command blocks. | `TarsEngine.FSharp.Metascript.Runner/` | **Compatibility Layer** implemented via `V1Parser.fs` and `V1Executor.fs` in `v2/src/Tars.Metascript/`. |

---

## 3. V2 Compatibility Status

TARS V2 implements a clear compatibility map for existing files:

* **`.tars` Files:** Successfully parsed by `V1Parser.fs`. The V2 Metascript Engine exposes a V1 executor that maps text, query, command, and meta blocks to modern workflows.
* **`.trsx` Files:** Parsed by `TrsxParser.fs`. Trsx acts as the serialization format for TARS workflows, bridging legacy specification-driven agent plans with the Cortex Workflow of Thought executor.
* **`.flux` Files:** **Not supported natively**. While V1 FLUX had powerful parallel scheduling properties, TARS V2 replaces FLUX with the unified Cortex scheduler which supports parallel steps via task-based execution.

---

## 4. Minimal V2 Scope

To ensure a high-velocity, lightweight release, V2 defines the following strict scope boundaries:

### Included in V2
1. **Schema-Constrained Outputs:** Native support for OpenAi Compatible, Ollama, and llama.cpp backends using JSON Schema parameters.
2. **EBNF / GBNF Constraints:** Direct compilation of standard context-free grammar strings to LLM decoding parameters.
3. **Structured Contracts:** Compilation of F# record types and discriminated unions (DUs) to schema contracts.
4. **Metascript Command/Query Blocks:** Support for running basic command, query, and reflection blocks via the compatibility runner.
5. **V2 Closure Contracts:** Direct alignment where output schemas from the distillation pipeline validate closure inputs.

### Excluded from V2 (Deferred or Deallocated)
1. **Multi-Language Runtimes:** No native compile-and-run for C#, Python, or Rust from inline blocks inside the core. Sandboxing is delegated to separate worker tools.
2. **Live RFC Parsing:** The ability to auto-ingest RFC rules over HTTP is removed to protect the system's deterministic validation footprint.
3. **Always-On Evolution Loops:** Continuous background mutation of EBNF grammars via local genetic optimization is removed.

---

## 5. IX-Owned Grammar & Evaluation Candidates

Per the division of responsibilities (TARS for orchestration and semantic interfaces; IX for math and algorithms), the following components belong entirely in the **IX Engine**:

1. **Fractal and L-System Grammar Generation:** Calculating mathematical Hausdorff dimensions, recursive scaling, and SVG/DOT exports.
2. **Grammar Evolution and Mutations:** High-overhead probabilistic algorithm optimization used to discover new syntax structures.
3. **Advanced Semantic Scorecards:** Evaluating the logical validity, confidence weighting, and formal syntax correctness of LLM outputs.

---

## 6. Deferred V3+ Items

These capabilities are officially designated as out-of-scope for the V2 lifecycle:

* **Fractal Dimensional Scaling:** Recursive scaling of grammar constraints based on token budget constraints.
* **Dynamic Multi-Language Closure Synthesis:** Real-time generation of polyglot WebAPI boundaries directly from grammars.
* **Federated Grammar Registries:** Shared repositories for distributing structured grammar contracts across multiple distributed runtimes.
