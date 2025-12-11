# TARS Cognitive QA Strategy

## Overview

This document defines the Quality Assurance methodology for TARS's "Cognitive Abilities" — the AI's ability to Reason, Remember, and Use Tools effectively.

Unlike traditional unit tests that verify code logic (e.g., "Does function X return Y?"), Cognitive QA verifies **behavioral outcomes** (e.g., "Does TARS remember the user's deadline?").

## Testing Levels

### Level 1: Unit Tests (Automated)

**Scope**: Verifies cognitive pattern structure and logic.

* **Location**: `tests/Tars.Tests/CognitivePatternTests.fs`
* **Coverage**: 13 tests for SemanticWatchdog, UncertaintyGatedPlanner, ConsensusCircuitBreaker

### Level 2: System Integration (Automated)

**Scope**: Verifies that the "brain" components are wired correctly.

* **Location**: `tests/integration/`
* **Tools**: `pytest`, `mcp_test_client.py`
* **Checks**:
  * Can TARS start as an MCP Server?
  * Are the expected tools (`search_memory`, `save_memory`, `list_files`) registered?
  * Do tools accept valid arguments and return valid JSON?

### Level 3: Cognitive Evaluations (Evals)

**Scope**: Verifies the *quality* and *correctness* of AI reasoning and memory.

* **Location**: `tests/evals/`

| Eval | Tests |
|------|-------|
| `eval_memory.py` | Memory recall (needle in haystack) |
| `eval_budget.py` | Token budget enforcement |
| `eval_compression.py` | Context compaction >50% |
| `eval_watchdog.py` | Loop/budget detection |

### Level 4: Manual Verification

**Scope**: Exploratory testing of complex flows.

* **Method**: `tars chat` interactive sessions.

## Running Tests

### Unit Tests

```bash
dotnet test Tars.sln --filter "CognitivePatternTests"
```

### Integration Tests

```bash
python tests/integration/test_mcp_capabilities.py
```

### Cognitive Evals

```bash
python tests/evals/eval_memory.py
python tests/evals/eval_budget.py
python tests/evals/eval_compression.py
python tests/evals/eval_watchdog.py
```

## Future Work

* Integrate into GitHub Actions
* Add **Reasoning Evals** (Tool Selection accuracy)
* Add **Benchmark Suite** (HotpotQA, LoCoMo)
