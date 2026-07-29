# Second-Brain Compounding Improvement Metrics

## Goal
The objective is to measure whether the TARS second-brain harness is producing **compounding improvement** rather than merely linear accumulation of data.

In a compounding system, each successful cycle (N) improves the harness, which in turn improves the performance (quality, speed, context, routing, or evidence) of the next cycle (N+1).

## Core Hypothesis
A healthy second-brain harness improves its future performance by transforming every validated run, research finding, issue, trace, and replay into:
- Better memory (higher precision, lower noise).
- Better contracts (sharper requirements, clearer handoffs).
- Better retrieval (contextual relevance, token efficiency).
- Better scoring (accurate feedback loops).
- Better delegated work (higher agent success rate).

---

## Metrics Framework

### 1. Leading Indicators
These measure the immediate health of the cycle and provide early signals of potential compounding effects.

| Metric | Definition | Compounding Signal |
|--------|------------|--------------------|
| `retrieval_precision_at_k` | % of retrieved context chunks that are actually relevant. | High precision reduces noise for subsequent cycles. |
| `context_token_efficiency` | Ratio of relevant information to total tokens used in context. | Improves cost-effectiveness and speed. |
| `agent_ready_issue_rate` | % of generated issues that are accepted by agents without rework. | Indicates better automated issue drafting. |
| `missing_criteria_rate` | % of issues missing critical acceptance criteria. | Lower rate implies better requirement extraction. |
| `issue_rework_rate` | % of tasks requiring multiple attempts to pass validation. | Lower rate suggests better initial task definition. |
| `trace_completeness` | % of mandatory trace fields populated with valid data. | Better traces enable better future learning. |
| `artifact_reuse_rate` | % of current cycle that uses existing promoted patterns/assets. | Direct measure of reuse. |
| `research_to_issue_quality` | Scored quality of issues derived from research digests. | Indicates efficiency of the research-to-action pipeline. |
| `closure_replay_success` | Success rate of replaying a failed cycle with improved memory. | Validates that learning prevents recurrence. |
| `stale_memory_hit_rate` | % of retrieved memories that are outdated or contradictory. | Measures the cleanliness of the second brain. |

### 2. Compounding Indicators
These measure the delta across cycles, indicating the presence of a compounding curve.

| Metric | Definition | Linear vs. Compounding |
|--------|------------|------------------------|
| `cycle_quality_delta` | Improvement in output quality vs. previous baseline. | Compounding shows an accelerating or stable positive delta. |
| `cycle_time_delta` | Reduction in wall-clock time for a standard task type. | Compounding shows time reduction even as complexity grows. |
| `reuse_multiplier` | Ratio of value created to new information added. | Compounding > 1.0 (leveraging existing knowledge). |
| `issue_quality_delta` | Improvement in auto-generated issue clarity over N cycles. | Indicates the "issue factory" is learning. |
| `retrieval_quality_delta` | Improvement in retrieval relevance for recurring topics. | Indicates "memory tuning" is effective. |
| `agent_success_delta` | Increase in autonomous success rate on first attempt. | Direct result of better context and contracts. |
| `failure_recurrence_delta` | Change in the rate of previously solved failure types. | Should trend toward zero in a compounding system. |
| `knowledge_reuse_delta` | Increase in the volume of cross-referenced knowledge nodes. | Indicates a denser, more useful knowledge graph. |

### 3. Guardrails (The "Anti-Metrics")
These ensure that "improvement" isn't just gaming the system or adding fragility.

| Guardrail | Target | Risk if ignored |
|-----------|--------|-----------------|
| `false_confidence_rate` | < 5% | The system claims it knows more than it does. |
| `stale_context_rate` | < 10% | Agents are acting on outdated or overridden instructions. |
| `invalid_issue_rate` | < 2% | The "issue factory" produces busy-work or junk tasks. |
| `unvalidated_promotion` | 0% | Patterns are promoted without passing the 7-step pipeline. |
| `agent_rework_rate` | Decreasing | High rework indicates failing contracts despite "learning." |
| `cost_per_improvement` | Stable/Declining | Compounding should eventually reduce the cost of quality. |
| `human_review_burden` | Stable/Declining | The system should require less human intervention over time. |

---

## Minimal Scoring Model

Each cycle is evaluated using a simple scorecard to make the improvement curve visible and auditable.

### The Formula
`cycle_score = (quality_gain * reuse_multiplier * confidence) - cost_penalty - rework_penalty - stale_context_penalty`

*   **quality_gain**: [0.0 - 1.0] Improvement over baseline.
*   **reuse_multiplier**: [1.0 - 2.0] Degree of existing artifact/pattern leverage.
*   **confidence**: [0.0 - 1.0] Evidence-backed confidence in the result.
*   **penalties**: [0.0 - 0.5 each] Deductions for inefficiencies or errors.

### Success Thresholds
- **Linear Accumulation**: Score remains static or grows slowly as data grows.
- **Compounding Improvement**: Score increases as `reuse_multiplier` and `quality_gain` grow while `rework_penalty` shrinks.

---

## Downstream Integration
- **IX Analytics**: Metrics feed into the IX skill engine for MCTS path scoring.
- **Demerzel Governance**: Guardrail violations trigger constitutional tribunal reviews.
- **TARS Evolution**: `cycle_score` informs the curriculum generation for the next evolution loop.
