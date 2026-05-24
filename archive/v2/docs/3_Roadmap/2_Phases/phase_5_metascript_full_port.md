# Phase 5: Metascript Full Port & Grammar Integration

**Status:** Planned  
**Priority:** High  
**Epic:** Epic 5 (The Mind)

## 🎯 Goal
Fully port the V1 Metascript/FLUX engine to TARS v2, upgrading it from a simple JSON-based orchestrator to a rich, human-readable DSL that integrates deeply with the V2 Grammar system and F# Interactive.

## 🏗️ Architecture
The new `Tars.Metascript` will follow a modular "Handler-based" architecture:
1. **Core Parser**: Parses `.tars` files into a list of `MetascriptBlock`s.
2. **Context Manager**: Manages variables, state, and execution environment.
3. **Block Handlers**: Specialized executors for each block type (F#, Python, LLM, etc.).
4. **Grammar Bridge**: Middleware that intercepts LLM-related blocks and applies grammar constraints.

## 📋 Task List

### 1. Core Parser Port (v1 -> v2)
- [ ] Port `MetascriptParser.fs` and related types.
- [ ] Support block markers, parameter parsing, and metadata.
- [ ] Implement robust error reporting with line/column numbers.

### 2. Basic Block Handlers
- [ ] **CommandBlockHandler**: Execute shell commands.
- [ ] **FSharpBlockHandler**: Integrate `FSharp.Compiler.Service` for FSI execution.
- [ ] **Text/Markdown**: Passive content blocks.

### 3. LLM Block & Grammar Integration
- [ ] **Action/Query Blocks**: Blocks that trigger LLM inference.
- [ ] **Grammar Parameter**: Add support for `grammar="Name"` parameter on blocks.
- [ ] **Integration with Tars.Cortex.Grammar**: 
    - Resolve grammar by name from `v2/resources/grammars/`.
    - Apply `GrammarConstraint` to `ICognitiveProvider.AskAsync`.
- [ ] **Inline Grammar Support**: Support `GRAMMAR { ... }` blocks that define local EBNF rules.

### 4. Advanced FLUX Features
- [ ] **Python Sandbox**: Integrate `PythonBlockHandler` with `Tars.Sandbox`.
- [ ] **Variable Interpolation**: Port the `${var}` syntax for all block content.
- [ ] **State Persistence**: Ensure variables persist across block boundaries and can be updated by deterministic code.

### 5. CLI & UX
- [ ] Update `tars run` to handle both JSON and `.tars` formats.
- [ ] Add `tars check` for validating metascript syntax and grammar references.
- [ ] Implement live execution trace (Spectre.Console) showing block progress.

## ✅ Acceptance Criteria
1. Can execute a `.tars` script that calculates a value in F#, uses it in an LLM prompt, and validates the LLM response against an EBNF grammar.
2. F# blocks can access and modify Metascript variables.
3. Python blocks run within the Docker sandbox.
4. Syntax errors in scripts point to the exact line/column.
