# V1 AgenticTraceCapture to AIW Episode Mapping

This document provides a concrete mapping from legacy V1 `AgenticTraceCapture` concepts to the modern AIW/TARS episode artifacts.

## Mapping Targets

| V1 `AgenticTraceCapture` Concept | Modern AIW/TARS Artifact | Description |
| :--- | :--- | :--- |
| **Session / Metadata** | `aiw-episode.json` | The core episode metadata, including IDs, timestamps, and overall status. |
| **Costs and Tokens** | `budget-ledger.json` | Detailed tracking of token usage, compute time, and financial cost (e.g., `cost_usd`). |
| **LLM Interactions** | Provider trace | Raw inputs, outputs, and latencies from the underlying model provider. |
| **Verification** | Test log summary | Aggregated results from test suites and CI runs during the episode. |
| **Modifications** | Diff summary | The unified diff or patch representing the code changes made. |
| **Termination** | Stop/escalation reason | The exact reason the episode ended (e.g., success, error, max steps, budget exceeded). |
