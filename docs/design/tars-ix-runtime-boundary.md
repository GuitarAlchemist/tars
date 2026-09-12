# TARS-IX Runtime Boundary and Integration Architecture

- **Status:** Approved
- **Area:** Runtime / Architecture
- **Ecosystem Domain:** TARS V2 & IX Engine
- **Related Issues:** #70, #71, #76, #82, GuitarAlchemist/ix#188, GuitarAlchemist/ix#189, GuitarAlchemist/ix#191, GuitarAlchemist/ix#197, GuitarAlchemist/Demerzel#473

---

## 1. Executive Summary

This document defines the strict architectural and operational boundary between the **TARS V2 Runtime** and the **IX Algorithm/ML Engine**, as well as the governance layer owned by **Demerzel**.

In the legacy TARS V1 architecture, boundaries were blurred: high-dimensional vector transforms, math utilities, and complex reasoning graphs were tightly coupled inside the runtime itself. TARS V2 introduces a clean, decoupled boundary:
*   **TARS owns execution, orchestration, and structural contracts.** It executes workflows and outputs raw telemetry/artifacts.
*   **IX owns algorithms, analytics, heavy math, and scoring.** It processes the telemetry emitted by TARS and returns structured scorecards, drift analysis, and cognitive assessments.
*   **Demerzel owns constitutional governance, risk gating, and budget policy.** It consumes the output of both layers to enforce gate decisions and halt execution when bounds are exceeded.

By decoupling execution (TARS) from evaluation (IX), the system achieves maximum runtime stability, prevents "Anti-Ball-of-Mud" entropy, and allows IX to evolve its ML/MCTS engines independently of the .NET TARS runtime.

---

## 2. Shared Architecture Vision & Directional Flow

To maintain a local-first, lightweight, and cost-safe execution, the interaction between TARS and IX is **asynchronous, file-centric, and artifact-driven**.

### Operational Flow

```
                      +---------------------------------------+
                      |             TARS RUNTIME              |
                      |   - Parses & executes WoT DSL         |
                      |   - Manages progressive disclosure   |
                      |   - Drives tool executions            |
                      +---------------------------------------+
                                          |
                                          | Emits structured JSON / JSONL
                                          v
                      +---------------------------------------+
                      |          LOCAL FILE ARTIFACTS         |
                      |        (Stored in .wot/runs/)         |
                      |   - closure-contract.json             |
                      |   - closure-run.json                  |
                      |   - trace-events.jsonl                |
                      +---------------------------------------+
                                          |
                                          | Read & queried (Offline / Async)
                                          v
                      +---------------------------------------+
                      |         IX DUCKDB / ANALYTICS         |
                      |   - Reads JSON/Parquet artifacts      |
                      |   - Computes complex UDF scoring      |
                      |   - Tracks drift & regressions        |
                      +---------------------------------------+
                                          |
                                          | Produces scores & reports
                                          v
                      +---------------------------------------+
                      |         IX SCORECARDS & REPORTS       |
                      |   - ix-scorecard.json                 |
                      |   - ix-analysis-report.json           |
                      +---------------------------------------+
                                          |
                                          | Evaluates compliance & budget
                                          v
                      +---------------------------------------+
                      |          DEMERZEL GOVERNANCE          |
                      |   - Verifies constitution & budget    |
                      |   - Enforces AIW (Integrity) Gates    |
                      |   - Writes halt markers if needed     |
                      +---------------------------------------+
```

---

## 3. Domain Responsibility Matrix

| Feature / Capability | TARS V2 Runtime | IX Algorithm Engine | Demerzel Governance |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Workflow execution, tool integration, and state orchestration. | Compute-heavy math, optimization, and analytical evaluations. | Policy compliance, budget control, and AIW gates. |
| **Memory & Storage** | Lean `IVectorStore` (Save/Search float32 vectors), PARA & Zettelkasten local structures. | High-dimensional transforms (FFT, Hyperbolic, Projective), DuckDB OLAP tables. | Audit log validation, budget ledger persistence. |
| **Validation** | **Structural / Grammatical:** validates EBNF syntax, JSON schemas, tool availability. | **Semantic / Quality:** scores reasoning patterns, confidence weights, semantic drift. | **Compliance:** ensures actions align with Agentic Constitution and budget tiers. |
| **Algorithms** | Simple, deterministic flow control, string templates, state machine transitions. | MCTS (Monte Carlo Tree Search), PSO (Particle Swarm), genetic algorithms, tensor logic. | None (Policy rules and risk calculation thresholds). |
| **Invocation** | Synchronous driver. Calls local CLI, MCP servers, or executes `.wot.trsx` files. | Asynchronous analysis. Triggered via MCP tools, A2A tasks, or offline CLI runs. | Passive/Active Gatekeeper. Inspects state files; writes halt markers to abort loops. |

### 3.1. TARS-Owned Responsibilities
1.  **Workflow-of-Thought (WoT) Execution**: Parsing and executing `.wot.trsx` workflows. Managing steps, branching, and state transitions.
2.  **Context Engineering**: Progressive context disclosure, providing each step with only its necessary and scoped inputs to avoid LLM context-window drowning.
3.  **Local Memory Persistence**: Providing the lightweight `IVectorStore` abstraction with local, flat-file JSON storage to minimize dependencies.
4.  **Telemetry Generation**: Recording all step executions, LLM interactions, tool invocations, and error states into a standardized, streaming trace event log.
5.  **Syntactic/Grammatical Validation**: Ensuring LLM outputs conform to EBNF grammars and structural contracts (e.g., verifying that a generated block matches `wot.ebnf` before trying to execute it).

### 3.2. IX-Owned Responsibilities
1.  **Heavy Mathematical Transforms**: Multi-space embedding similarity aggregations, FFT, Pauli matrices, Hyperbolic and Projective embeddings (legacy V1 vector capabilities harvested as specialized IX skills).
2.  **Quantitative & Semantic Scoring**: Analyzing trace logs to calculate confidence weights, logic consistency, reasoning depth, and step quality scores.
3.  **DuckDB-Powered OLAP Analytics**: Hosting and querying historical logs, detecting performance drift, code performance regression, and patterns across millions of trace events.
4.  **UDF-backed Queries**: Running high-speed, compiled analysis routines directly inside DuckDB over Parquet/JSON artifacts.
5.  **Research Paper Mining**: Parsing and digesting research papers (as defined in `research-digest.contract.md`) to synthesize actionable proposals for the ecosystem.

### 3.3. Demerzel-Owned Governance Responsibilities
1.  **AIW Gates (Agentic Integrity Workflow)**: Evaluating execution metadata before allowing code promotion, committing, or pushing to remote repositories.
2.  **Budget Control**: Monitoring financial (USD) and runner (minutes) usage, ensuring runs respect defined tiers (e.g., `free-local`, `cloud-low`, `cloud-high`).
3.  **Halt Policy Enforcer**: Reading active halt-markers (e.g., `afk-halt.json`) to terminate autonomous agent execution loops instantly if an anomaly is detected.
4.  **Escalation Path Router**: Routing complex, high-risk, or budget-exceeding tasks to humans or multi-agent tribunals (e.g., decision-gate review mode).

---

## 4. Forbidden Duplication Patterns

To maintain a clean architectural separation and avoid "anti-ball-of-mud" entropy, the following duplication patterns are **strictly forbidden**:

1.  **No Local Math in TARS**: TARS must never contain complex mathematical code (e.g., FFT, multidimensional geometric mapping, vector projection, or signal processing). These must be requested as tools from the IX MCP server or CLI.
2.  **No Local Scorecards in TARS**: TARS must not implement semantic scoring models, confidence-weighting calculations, or logic consistency checkers. TARS emits the execution traces; IX scores them.
3.  **No Always-On Database in TARS**: TARS must not manage or require an active PostgreSQL, DuckDB, or MongoDB server for its core, basic execution. TARS is filesystem-first and local-first.
4.  **No Direct DuckDB Queries in TARS Critical Path**: During a standard workflow execution loop, TARS must never execute SQL queries against a DuckDB database to make routing decisions. It must read static JSON/YAML configuration or use structured, cached memory lookups via its lean vector store. DuckDB analysis belongs strictly in IX.
5.  **No Governance Rules in TARS**: TARS must not evaluate constitutional compliance or perform budget ledger deductions directly. It simply reads the static policy rules or queries the Demerzel ACP/MCP server.

---

## 5. Artifact Exchange Flow & Directory Structure

To keep the ecosystem lean, all interaction is done via structured local files. TARS writes artifacts to the local directory; IX processes them; Demerzel inspects both.

### 5.1. Directory Structure

The standardized workspace layout is organized as follows:

```
workspace-root/
├── .wot/
│   └── runs/
│       ├── run-{run-id}/
│       │   ├── closure-contract.json      <-- TARS writes: Goal & constraints
│       │   ├── closure-run.json           <-- TARS writes: Summary of execution
│       │   ├── trace-events.jsonl         <-- TARS writes: Granular telemetry stream
│       │   ├── ix-scorecard.json          <-- IX writes: Score card evaluation
│       │   └── ix-analysis-report.json    <-- IX writes: Optimization advice
│       └── active-run.json                <-- TARS writes: Symlink or reference to active run
└── governance/
    └── agents/
        └── live/
            ├── afk-runs.json              <-- Demerzel/Board writes: Tracked state
            └── afk-halt.json              <-- Demerzel writes: Global kill-switch
```

### 5.2. Step-by-Step Scenario: End-to-End Execution & Analysis

1.  **Contract Definition**: The TARS Cortex decides to execute a discrete task. It generates a `closure-contract.json` detailing the target goal, constraints, and required skills.
2.  **Structural Validation**: TARS validates the contract structure against its schema.
3.  **Execution**: TARS executes the closure, invoking necessary tools (from its `ISkillRegistry`) and prompting LLMs. During execution, TARS streams trace events to `trace-events.jsonl` (one JSON object per line).
4.  **Execution Logging**: Upon completion, TARS writes a finalized `closure-run.json` containing the execution outcome, duration, and output metadata.
5.  **IX Trigger (Analysis)**: An offline process, MCP command, or A2A task triggers IX. IX loads the `trace-events.jsonl` and `closure-run.json` files from `.wot/runs/run-{run-id}/` using its built-in DuckDB pipelines.
6.  **Scoring and Reporting**: IX runs UDF-backed analysis to calculate:
    *   Semantic drift from the original contract goal.
    *   Confidence/quality scores for each step.
    *   Synthesized improvements for future runs.
    It writes out `ix-scorecard.json` and `ix-analysis-report.json` in the same directory.
7.  **Demerzel Evaluation**: The Demerzel governance agent inspects the scorecard and runs compliance gates. If the execution is deemed safe, cost-compliant, and within the budget, the results are promoted. If a budget violation or malicious pattern is detected, Demerzel writes an `afk-halt.json` marker to block further loops.

---

## 6. DuckDB + IX Integration Plan (IX #191 & #197)

While TARS remains database-agnostic and relies on local JSON files, IX manages high-performance querying and analytics through DuckDB. This integration is structured as follows:

1.  **No Runtime Blockers**: IX processes telemetry as a non-blocking background task. TARS does not pause standard workflow execution to wait for a full DuckDB analytics run unless a critical "decision-gate" requires it.
2.  **DuckDB Schema Definition**: DuckDB schemas, table mappings, and view definitions are owned and maintained by IX.
3.  **Parquet Exporters**: IX is responsible for compiling older JSONL trace files into Parquet format for long-term, high-performance OLAP queries (e.g., `vector-benchmark.parquet`).
4.  **UDF Registries**: Mathematical functions and evaluation heuristics are registered as DuckDB User-Defined Functions (UDFs) within the Rust-based IX codebase.

---

## 7. IX Pipeline Invocation Model

Ecosystem integration should always favor **local-first and free-of-cost** communication. TARS invokes the IX processing pipelines using the following preferred hierarchy:

1.  **Artifact File Trigger (Default/Primary)**: TARS writes files to disk. A separate, lightweight local listener or cron-like agent triggers IX when new trace files are completed.
2.  **Local CLI Command**: TARS calls the local IX command line (e.g., `ix analyze --path .wot/runs/run-123/`). This command is invoked as a standard shell process, avoiding network overhead.
3.  **Local MCP Tool Interface**: TARS invokes tools exposed by the local IX MCP server (e.g., `ix.evaluate_trace(run_id="123")`).
4.  **Agent-to-Agent (A2A) Task**: An asynchronous task contract is submitted to the local dispatch queue, where a specialized IX worker agent picks it up, analyzes it, and writes the scorecard back to the workspace.

---

## 8. Closure Factory V2 Integration

The **Closure Factory V2** is the designated subsystem in TARS responsible for instantiating and running closures (discrete execution environments).

To ensure continuous optimization:
1.  **Mandatory Emission**: Every execution initiated by the Closure Factory V2 **must** emit a `closure-contract.json`, a `closure-run.json`, and stream to `trace-events.jsonl` by default.
2.  **Isolated Workspaces**: The factory creates a separate subfolder under `.wot/runs/` for each closure to prevent race conditions during concurrent runs.
3.  **Halt Detection**: At the boundary of every step execution, the Closure Factory must check for active governance halt-markers (`afk-halt.json`) and abort instantly if one is detected.

---

## 9. Cost-Control & Privacy Constraints

To ensure safety and financial sustainability:

1.  **Local-First / Free Tier**: All trace logging, DuckDB parsing, and local evaluations must run completely free-of-cost on the local machine (using local CPUs and free open-source tooling like DuckDB and Rust compilers).
2.  **Zero Raw Secret Propagation**: Trace files must strictly filter out raw API keys, bearer tokens, or user passwords. The Closure Factory must scrub known credential formats from LLM inputs/outputs before writing them to `trace-events.jsonl`.
3.  **Intellectual Property Protection**: Code files or private repositories analyzed by IX during development must not be exfiltrated to cloud services. All scoring and analysis must remain strictly on the local developer sandbox unless cloud analysis is explicitly configured by the user.
