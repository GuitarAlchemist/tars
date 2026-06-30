# Agentic Experience (AX) for Compound Engineering

- **Status:** Draft
- **Area:** Runtime / Compound Engineering
- **Owner:** TARS
- **Version:** 1.0.0

## 1. Definition of AX for the GuitarAlchemist Ecosystem

Agentic Experience (AX) is the design of repository surfaces, interfaces, and workflows optimized for autonomous and semi-autonomous AI agents collaborating with human engineers. In the GuitarAlchemist ecosystem, AX ensures that agents can reliably understand, act on, and verify tasks through explicit, machine-readable contracts, evidence requirements, handoffs, and recovery paths.

For TARS compound engineering, AX means moving away from informal prose and ambiguous requests toward:
- Structured issues and parseable plans.
- Explicit scope bounds and strict non-goals.
- Typed task contracts and deterministic artifact schemas.
- Clear run/test commands and evidence requirements.
- Predictable handoff/stop conditions, recovery, and escalation paths.

AX is not a replacement for human-centric design but a complementary layer that guarantees predictability when delegating work to agents like Jules, Codex, or the TARS WoT engine.

## 2. Comparison with UX and DX

- **UX (User Experience):** Optimizes for human users interacting with a product (e.g., visual hierarchy, intuitive navigation, emotional resonance). It assumes human intuition and contextual adaptability.
- **DX (Developer Experience):** Optimizes for human software engineers building a product (e.g., clear APIs, readable documentation, fast feedback loops, helpful error messages). It assumes human reasoning and problem-solving skills.
- **AX (Agentic Experience):** Optimizes for AI agents operating within a codebase or system. It eliminates ambiguity, relies on deterministic parsing, demands explicit contracts, and prioritizes structured telemetry over conversational logs. AX assumes agents lack intuition and require strict boundaries, explicit verification steps, and unambiguous recovery paths.

## 3. TARS, IX, and Demerzel Role Split

To implement effective AX across the ecosystem, responsibilities are split among the core components:

### TARS (Validation & Runtime)
- Acts as the primary orchestrator and cross-model F# theory validator.
- Evaluates AX compliance at runtime using probabilistic grammars and validations.
- Manages the Workflow-of-Thought (WoT) execution and ensures state transitions follow structured contracts.

### IX (Skill Engine & Deterministic Tools)
- Provides the deterministic Rust-based tools and MCTS evaluation that agents rely on to execute tasks.
- Exposes "doctor checks" and deterministic health assessments to evaluate if an environment or artifact is agent-ready.
- Executes the structured run/test commands mandated by the AX contract.

### Demerzel (Governance & Policy)
- Enforces risk and cost gates before an agent is allowed to execute a task.
- Validates the AX scorecard dimensions (e.g., risk policy, cost limits, explicit scope).
- Manages escalation paths and human-in-the-loop approvals when an agent hits a stop condition.

## 4. AX Scorecard Draft

To evaluate if an issue or task is "agent-ready," it must be scored against the following AX dimensions:

| Dimension | Description | Requirement for Agent-Ready |
| :--- | :--- | :--- |
| **clarity** | Unambiguous description of the problem or task. | No implicit context required. Direct language. |
| **scope_bounds** | Explicit definition of what is included in the task. | Specific files, components, or boundaries listed. |
| **non_goals** | Explicit definition of what is *not* included. | Prevents agent hallucination or scope creep. |
| **evidence_requirements** | What proof is needed to verify completion. | Specific logs, screenshots, or test results. |
| **testability** | How the agent verifies its work. | Executable test commands provided. |
| **artifact_contracts** | Structured schemas for expected outputs. | Expected JSON/MD formats or API schemas. |
| **handoff_quality** | Clear transition points between human/agent. | Defined `status` or label transitions. |
| **risk_policy** | Assessment of potential blast radius. | Governance tier specified (e.g., low-risk, draft-only). |
| **cost_policy** | Budget for compute or token usage. | Maximum runner minutes or USD cost limit. |
| **recovery_path** | What the agent should do if it fails. | Clear escalation instructions or fallback steps. |
| **machine_parseability** | Format of the issue or task definition. | Use of YAML frontmatter, markdown checklists, and structured headers. |

## 5. Example: Poor Issue vs. Agent-Ready Contract

### Poor Issue (Low AX)
**Title:** Fix the memory leak in the orchestrator
**Body:**
> The orchestrator seems to be using too much memory when handling large traces. Can you optimize it? Also, maybe clean up the logging while you're at it. Thanks!

*Why it fails AX:* Unclear scope, open-ended "optimize," implicit context, no test commands, missing budget/risk controls.

### Agent-Ready Contract (High AX)
**Title:** [Bug][Orchestrator][P2] Fix memory leak in `TraceIngestor.fs` buffer

```yaml
issue_meta:
  level: task
  area: runtime
  priority: P2
  complexity: S
  risk: low
  budget:
    max_runner_minutes: 15
  routing:
    preferred_workers: [jules]
  evidence_required: [memory_profile_before_after, test_pass_log]
```

**Goal:**
Resolve a memory leak in `v2/src/Tars.Cortex/TraceIngestor.fs` caused by unbounded buffering of GA traces over 50MB.

**Scope Bounds:**
- Modify only `TraceIngestor.fs` and `TraceIngestorTests.fs`.
- Implement a bounded queue (max 1000 items) or streaming read.

**Non-goals:**
- Do not refactor the logging system.
- Do not change the trace JSON schema.

**Testability:**
- Command: `dotnet test tests/Tars.Tests/ --filter "FullyQualifiedName~TraceIngestor"`

**Recovery Path:**
- If the test suite fails to compile, undo changes and escalate to human review with compilation errors.

## 6. Candidate Follow-up Tasks

1. **Integrate AX Scorecard into GitHub Actions:** Create a CI check that uses Demerzel to evaluate issue YAML frontmatter against the AX scorecard before applying the `ready-for-agent` label.
2. **IX Doctor Checks for AX:** Implement a deterministic `ix-doctor` command to validate that the requested test commands in an issue contract are actually executable in the target environment.
3. **Closure Factory V2 Integration:** Update the Closure Factory V2 (#82) templates to emit AX-friendly artifacts by default, enforcing the inclusion of evidence requirements and non-goals.
4. **Markdown Cleanup Phase:** Apply the markdown cleanup process (#85/#86) to existing standard operating procedures (SOPs), converting them into structured AX contracts.
