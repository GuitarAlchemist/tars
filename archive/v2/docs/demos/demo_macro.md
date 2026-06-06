# Macro Demo

**Command**: `tars macro-demo`

## Purpose

Demonstrates the Evolutive Grammar system—workflow macros that can be registered, composed, and recursively executed.

## What It Does

1. **Register Macro**: Creates a `demo_greeting` macro (a sub-workflow)
2. **Execute Main Workflow**: Runs a workflow that invokes the macro
3. **Review Step**: Shows output piped through agents

## Example Output

```
[INF] Starting Evolutive Grammar (Macro) Demo...
[INF] Registering macro 'demo_greeting'...
[INF] Registry contains: ["demo_greeting"]
[INF] Executing main workflow...
[INF] Macro Output: {greeting: "..."}
[INF] Review Output: {review: "..."}
```

## Key Components

- `IMacroRegistry`: Interface for macro storage
- `FileMacroRegistry`: File-based implementation (JSON)
- `Engine.executeStep`: Handles recursive macro invocation

## Status

✅ Core macro functionality working (registration, recursive execution)

> [!NOTE]
> The `register_macro`, `list_macros`, and `get_macro` tools are not yet available as standalone tools due to a circular dependency between `Tars.Metascript` and `Tars.Tools`. Macros work via `Engine.executeStep` which handles macro invocation directly when it encounters an unknown step type that matches a registered macro name.
