# F# Expert Tools Reference

The **F# Expert Tools** suite provides deterministic, structure-aware capabilities for the agent to read, understand, and modify F# codebases. Unlike generic text editing, these tools leverage the F# compiler and type system concepts.

## 🛠️ Compilation & Execution

### `fsharp_compile`
**Description**: Compiles an F# file or project and returns structured diagnostics.
**Usage**: Validate code changes immediately.
**Input**: `{ "path": "src/MyProject.fsproj" }`
**Output**: 
```text
❌ BUILD FAILED
Errors (1):
[FS0001] File.fs:10:5 - Type mismatch...
```

### `fsharp_eval`
**Description**: Evaluates F# code using F# Interactive (FSI).
**Usage**: Test small snippets or verify logic (e.g., regex, math).
**Input**: `{ "code": "let x = 1 + 1 in x" }`

### `fsharp_check_syntax`
**Description**: Quick syntax check without full compilation.
**Usage**: fast feedback loop before compiling.
**Checks**: Balanced parentheses, indentation, incomplete patterns.

## 🔍 Code Analysis

### `fsharp_analyze_structure`
**Description**: Analyzes the structure of an F# file (AST-lite).
**Usage**: Understand file layout without reading the entire raw text.
**Output**: Lists modules, types, and top-level functions with line numbers.

### `fsharp_check_file_order`
**Description**: Checks if files in an F# project are in correct dependency order.
**Usage**: Fix "Undefined Identifier" errors caused by file ordering.

## 🔧 Error Fixing

### `fsharp_explain_error`
**Description**: Explains an F# error code (e.g., `FS0001`) with common fixes.
**Input**: `"FS0001"`
**Output**: Detailed explanation and fix strategies.

### `fsharp_suggest_fix`
**Description**: Analyzes a compilation error context and suggests a specific fix.
**Input**: `{ "error": "...", "code": "..." }`

## 🏗️ Code Generation

### `fsharp_ce_template`
**Description**: Generates a computation expression builder template.
**Usage**: Quickly scaffold `tars {}` or other monads.
**Input**: `{ "name": "MyBuilder", "operations": ["bind", "return"] }`
