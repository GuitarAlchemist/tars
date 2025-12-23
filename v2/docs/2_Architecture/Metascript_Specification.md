# TARS v2 Metascript Specification

## Overview

Metascript is the domain-specific language (DSL) for defining autonomous workflows in TARS v2. It allows users to define complex, multi-step tasks that orchestrate agents, tools, and control flow.

## File Format

Metascripts use the `.tars` extension and are written in **YAML**.

## Structure

```yaml
name: "Refactor Component"
description: "Refactors a component and runs tests."
version: "1.0"

inputs:
  - name: component_path
    type: string
    description: "Path to the component file"

steps:
  - id: analyze
    type: agent
    agent: "Analyzer"
    instruction: "Analyze the code at {{component_path}} and identify refactoring opportunities."
    outputs:
      - name: analysis_report

  - id: plan
    type: agent
    agent: "Planner"
    instruction: "Create a refactoring plan based on the analysis."
    context:
      - step: analyze
        output: analysis_report
    outputs:
      - name: plan

  - id: execute
    type: agent
    agent: "Coder"
    instruction: "Apply the refactoring plan."
    context:
      - step: plan
        output: plan
    tools:
      - "read_file"
      - "write_file"

  - id: verify
    type: tool
    tool: "run_command"
    params:
      command: "dotnet test"
```

## Step Types

### 1. `agent`

Delegates a task to an AI Agent.

- **agent**: Name of the agent persona (e.g., "Coder", "Planner").
- **instruction**: The prompt/task for the agent.
- **context**: Data from previous steps to include in the prompt.
- **tools**: List of tools the agent is allowed to use.

### 2. `tool`

Executes a specific tool directly (deterministic).

- **tool**: Name of the tool (e.g., `run_command`, `read_file`).
- **params**: Arguments for the tool.

### 3. `loop` (Future)

Iterates over a list or until a condition is met.

### 4. `decision` (Future)

Branching logic based on previous outputs.

## Execution Model

The **Metascript Engine** reads the YAML, validates it, and executes steps sequentially (or in parallel where possible). It maintains a **Workflow Context** that stores variables and step outputs.

## V1 Rich Metascript (.tars / .trsx)

TARS v2 includes a full port of the capability-rich metascript system from v1. This system uses a block-based, bracketed or Markdown-style format that enables "Literate AI Programming."

### Features
- **Deterministic Logic**: Use `FSHARP { ... }` blocks powered by F# Interactive (FSI).
- **Polyglot Execution**: Support for `PYTHON`, `COMMAND`, and `QUERY` blocks.
- **State Persistence**: Variables and state persist across blocks in the same metascript run.
- **Data Flow**: Output from one block can be used in subsequent blocks using `${var}` interpolation.
- **Grammar Integration**: `QUERY` blocks support a `grammar="Name"` parameter to constrain and validate LLM output using the v2 grammar system.

### Example
```tars
meta {
    Name: "Verify TARS"
}

FSHARP(output="version") {
    "2.0-rich"
}

QUERY(output="goal", grammar="Goal") {
    Generate a simple goal for TARS v${version}.
}

COMMAND {
    echo "Goal generated: ${goal}"
}
```
