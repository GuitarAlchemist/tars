# TARS Cognitive QA Strategy

## Overview

This document defines the Quality Assurance methodology for TARS's "Cognitive Abilities" — the AI's ability to Reason, Remember, and Use Tools effectively.

Unlike traditional unit tests that verify code logic (e.g., "Does function X return Y?"), Cognitive QA verifies **behavioral outcomes** (e.g., "Does TARS remember the user's deadline?").

## Testing Levels

### Level 1: System Integration (Automated)

**Scope**: Verifies that the "brain" components are wired correctly.

* **Location**: `tests/integration/`
* **Tools**: `pytest`, `mcp_test_client.py`
* **Checks**:
  * Can TARS start as an MCP Server?
  * Are the expected tools (`search_memory`, `save_memory`, `list_files`) registered?
  * Do tools accept valid arguments and return valid JSON?

### Level 2: Cognitive Evaluations (Evals)

**Scope**: Verifies the *quality* and *correctness* of AI reasoning and memory.

* **Location**: `tests/evals/`
* **Tools**: Custom Python scripts, LLM-as-a-Judge (optional)
* **Methodology**: "Golden Set" / "Needle in a Haystack"
  * **Setup**: Inject a specific "Needle" (fact) into TARS's memory.
  * **Query**: Ask a natural language question requiring that fact.
  * **Assert**: Verify the answer matches the fact.

### Level 3: Manual Verification

**Scope**: Exploratory testing of complex flows.

* **Method**: `tars chat` interactive sessions.

## Running Tests

### Integration Tests

```bash
python tests/integration/test_mcp_capabilities.py
```

### Cognitive Evals

```bash
python tests/evals/eval_memory.py
```

## Future Work

* Integrate into GitHub Actions.
* Add **Reasoning Evals** (Tool Selection accuracy).
