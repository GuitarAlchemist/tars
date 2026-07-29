# Design: Semantic-to-Contract Pipeline

## Goal
To establish a robust, grammar-validated pipeline that transforms ambiguous human or LLM-generated natural language into machine-readable, schema-validated TARS contracts. This pipeline ensures that intent is captured accurately and constraints are enforced before any execution occurs.

## Pipeline Architecture

The pipeline consists of the following stages:

1.  **Input Ingestion**: Raw input from GitHub issues, markdown notes, metascript blocks, or LLM-generated intent.
2.  **Semantic Interpretation**: An LLM (acting as a "Semantic Interpreter") parses the input to extract intent, goals, constraints, and dependencies.
3.  **Candidate Contract Generation**: The interpreter generates a candidate structured contract (e.g., in JSON or a DSL format) based on the interpreted intent.
4.  **Grammar & Schema Validation**: The candidate contract is validated against its respective grammar (e.g., `cortex.ebnf`, `wot.ebnf`) and JSON schema.
5.  **Repair Loop (Optional)**: If validation fails, the error trace and the invalid candidate are fed back to the interpreter for correction.
6.  **Contract Finalization**: Once validated, the `IssueContract`, `ClosureContract`, or `MetascriptContract` is produced as a finalized artifact.

## Core Contract Types

- **IssueContract**: Defines the scope, goals, and metadata for a GitHub issue, including AFK-readiness and budget constraints.
- **ClosureContract**: Specifies the bounds for a discrete unit of execution (a "closure"), including resource limits, invariants, and required tools.
- **MetascriptContract**: A block of TARS DSL code (metascript) that defines a workflow or agentic logic.
- **EvidenceContract**: Captures the required evidence and provenance for a finding or action.
- **RiskContract**: Documents the risk assessment, mitigation strategies, and escalation paths for a proposed action.

## IX Scoring Boundary

IX (the evaluation engine) is involved **after** contract generation or **during** contract execution evaluation.
- The Semantic-to-Contract pipeline is responsible for **structural and grammatical validity**.
- IX is responsible for **semantic scoring and quality assessment**.
IX evaluates if a contract is "good," "actionable," or "high-confidence," but it does not replace the grammar validation stage.

## Failure Modes & Recovery

| Failure Mode | Description | Recovery Strategy |
| :--- | :--- | :--- |
| **Schema Mismatch** | Candidate JSON does not match the expected structure. | Automated repair loop with specific schema error messages. |
| **Grammar Violation** | DSL code in a metascript violates the EBNF grammar. | Parser-driven repair loop highlighting the exact token error. |
| **Hallucination** | LLM includes non-existent fields or invalid tool names. | Validation against the `ISkillRegistry` and schema; repair loop. |
| **Ambiguity** | Natural language input is too vague for contract generation. | Semantic interpreter requests clarification via a `FollowupQuestion`. |

## Success Criteria
- Input is successfully transformed into a valid, machine-readable contract.
- Validation errors are caught and corrected via the repair loop.
- All finalized contracts include mandatory metadata (provenance, risk, budget).