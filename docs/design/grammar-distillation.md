# GrammarDistiller: Design Specification and Metascript Compatibility Map

This document details the architectural design, inputs/outputs, pipeline phases, and compatibility mappings for the **TARS V2 GrammarDistiller** system.

---

## 1. Architectural Overview

The **GrammarDistiller** acts as the core contract generation engine for the TARS V2 semantic-to-contract pipeline. Its purpose is to transform diverse, often unstructured or multi-paradigm input specifications (natural language examples, F# structures, GraphQL schemas, legacy script blocks) into deterministic, machine-enforceable output schemas (such as GBNF/EBNF grammars, JSON Schemas, and prompt injection guidelines).

```
+-------------------------------------------------------+
|                       INPUTS                          |
|  JSON Examples | .tars/.trsx/.flux Blocks | OpenAPI   |
|  GraphQL SDL   | F# Records & DUs         | Artifacts |
+------------------------------------------+------------+
                                           |
                                           v
+-------------------------------------------------------+
|               GRAMMARDISTILLER PIPELINE               |
|                                                       |
|  Phase 1: Raw Extraction & Adapter Routing           |
|  Phase 2: Common AST Construction                     |
|  Phase 3: Schema Normalization & Type Resolution       |
|  Phase 4: Target Emission (Schema / GBNF / Hint)       |
|  Phase 5: Validation, Packaging & Storage             |
+------------------------------------------+------------+
                                           |
                                           v
+-------------------------------------------------------+
|                       OUTPUTS                         |
|  JSON Schema  | EBNF/GBNF Grammars | Typed Validators |
|  Prompt Hints | Evaluation Cases   | Closure Contracts|
+-------------------------------------------------------+
```

---

## 2. Pipeline Architecture & Phases

The distillation pipeline executes in five sequential, isolated stages:

### Phase 1: Raw Extraction & Adapter Routing
Inputs are fed into a router that identifies the input archetype. An extraction adapter parses the raw structure (e.g., extracting YAML/JSON snippets, parsing `.tars` blocks via `V1Parser`, or running reflection over F# metadata assemblies).

### Phase 2: Common AST Construction
The extracted tokens are compiled into a unified Intermediate Representation (IR) called the **TarsContractAST**. This AST represents:
* Namespaces and contract identities
* Nested fields, structural keys, and descriptions
* Logical types (Primitive, Object, Array, Union, Enums)
* Numerical/regex constraints (bounds, string matching, formats)

### Phase 3: Schema Normalization & Type Resolution
The raw AST is cleaned and augmented. Missing fields in partial inputs are inferred from examples, standard defaults are injected, and F# Discriminated Unions (DUs) are normalized into clean standard JSON union representations (using `oneOf` or tagged-union styles).

### Phase 4: Target Emission
The normalized contract AST is passed to specialized emission drivers to generate the exact targets required:
* **JSON Schema Emitter:** Produces a draft-07/draft-2020-12 compatible JSON specification.
* **EBNF/GBNF Emitter:** Translates the constraints into context-free grammar production rules for constrained decoding (specifically targeting llama.cpp and outlines/vLLM format).
* **Prompt Hint Emitter:** Formulates a concise textual markdown specification containing type rules and format examples to inject directly into the LLM system prompt.

### Phase 5: Validation, Packaging & Storage
The generated assets are bundled together as a **ContractBundle**. This bundle is run against a syntactic validator (checking JSON Schema validity and GBNF syntax) and cached locally within the `.tars/contracts/` index.

---

## 3. Input-to-Output Transformation Mapping

The following matrix details how each input type is parsed, resolved, and translated into corresponding output contracts:

| Input Format | Distillation / Parse Strategy | Primary Output Target | Validation/Fallback Path |
|---|---|---|---|
| **JSON Examples** | Schema inference via key/value shape analysis; type generalization. | JSON Schema | Prompt hint fallback |
| **.tars / .trsx Blocks** | Parsed via `V1Parser` / `TrsxParser`. Extracts variables, inputs, and step types. | GBNF Grammar & Workflows | Inline JSON schema validator |
| **OpenAPI Spec** | Resolves endpoints to individual execution contracts; maps requests/responses to schema objects. | JSON Schema | API model routing |
| **GraphQL SDL** | Parses schema types, queries, and mutations into AST field definitions. | JSON Schema & EBNF | Structural key validator |
| **F# Records & DUs** | Reflection over F# assemblies (`Microsoft.FSharp.Reflection`) parsing field types and case unions. | JSON Schema & Type Validators | Standard F# deserializer |

---

## 4. Metascript Compatibility Map

Legacy V1 Metascript blocks are reconciled into TARS V2 contracts as follows:

```
+-------------------------------------------+
| V1 Legacy Block Type                      |
+-------------------------------------------+
       |
       |  (Compiled via GrammarDistiller)
       v
+-------------------------------------------+
| V2 Distilled Contract Archetype           |
+-------------------------------------------+
```

### Detailed Mapping Rules

1. **`meta` block:**
   * *V1 Syntax:* `meta { name: "AgentName", version: "v1.0" }`
   * *V2 Contract:* Maps directly to contract header metadata (`ContractId`, `Version`, `SemanticDomain`).
2. **`command` / `cmd` block:**
   * *V1 Syntax:* `command { LANG("BASH") { dotnet build } }`
   * *V2 Contract:* Translated into a standard `MetascriptContract` workflow step that triggers a local sandboxed executor tool (`standard.run_command`).
3. **`query` / `ask` block:**
   * *V1 Syntax:* `query { "What are prime numbers?" }`
   * *V2 Contract:* Becomes an execution trace contract requesting unstructured/structured generation, mapping input prompt constraints to the `LlmRequest` object.
4. **`grammar` block:**
   * *V1 Syntax:* `grammar "Name" { LANG("EBNF") { rule = ... } }`
   * *V2 Contract:* Compiles directly to an EBNF grammar asset in `v2/grammars/` and registers it under the `ResponseFormat.Constrained` type.
5. **`transform` block:**
   * *V1 Syntax:* `transform { input = Json; transform_to = Markdown; }`
   * *V2 Contract:* Becomes a dual-schema input/output contract mapped to a `ClosureContract` for execution.

---

## 5. Closure Factory V2 Integration Path

Grammar distillation is key to safe, automated closure execution (bridging natural language intent to typed F# execution).

* **Input Contract Validation:** Before any closure runs, its input arguments are validated against the distilled JSON Schema generated for that closure. If validation fails, the `RepairProposal` pipeline is invoked.
* **Output Contract Enforcement:** The output from the closure (e.g. JSON strings or objects) is validated against the distilled output contract before being returned to the parent workflow or subsequent steps.
* **Dynamic Generation of Closure Schemas:** When a user registers a static F# method as a skill using the `[<TarsSkill(name, domain)>]` attribute, the `GrammarDistiller` automatically generates the input/output schema contracts via reflection.

---

## 6. Cost Notes and Design Constraints

* **Execution Overhead:** Reflection-based distillation over complex assemblies is cached in-memory at startup to avoid high runtime overhead.
* **Token Efficiency:** GBNF grammars significantly save tokens by enforcing schema validity directly during decoding, eliminating the need for extensive "Please return valid JSON" system prompt padding and reducing retry rates to near zero.
* **Provider Limitations:** When utilizing proprietary cloud models (e.g., Anthropic Claude) that do not support raw GBNF/EBNF decoding natively, the distiller automatically falls back to translating the contract AST to detailed prompt schemas and validates outputs on receipt, raising `Result.Error` if repair is needed.
