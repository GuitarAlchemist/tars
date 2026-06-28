# Contract: AFK Dispatch Queue

## Status

Draft.

## Purpose

This contract describes the queue, admission, backpressure, and locking state for the AFK cloud-agent loop.

The first implementation may derive this state from GitHub issues, labels, and pull requests. A later scheduler may materialize it as JSON.

## Entities

### AfkDispatchQueue

| Field | Type | Description |
|---|---|---|
| `queue_id` | string | Stable queue identifier. |
| `repo` | string | Repository owner/name. |
| `generated_at` | RFC3339 timestamp | When this queue snapshot was produced. |
| `global_wip_limit` | integer | Maximum open cloud-agent PRs. |
| `open_cloud_agent_prs` | integer | Current open cloud-agent PR count. |
| `lanes` | array of `AfkLaneState` | Lane-specific capacity and occupancy. |
| `locks` | array of `AfkLock` | Exact active locks. |
| `probabilistic_indexes` | array of `ProbabilisticConflictIndex` | Optional pre-filter indexes. |
| `pending_decisions` | array of `BackpressureDecision` | Recent admission decisions. |

### AfkLaneState

| Field | Type | Description |
|---|---|---|
| `lane` | enum | `contract`, `evidence`, `policy`, `workflow`, `runtime`, `experiment`. |
| `wip_limit` | integer | Maximum concurrent work for this lane. |
| `open_prs` | integer | Current open PRs in this lane. |
| `queued_issues` | integer | Issues waiting for admission. |
| `priority` | integer | Lower number means higher priority. |

### AfkLock

| Field | Type | Description |
|---|---|---|
| `lock_id` | string | Stable lock id. |
| `kind` | enum | `issue`, `surface`, `lane`, `repo`. |
| `resource` | string | Locked resource key, such as `issue:GuitarAlchemist/tars#118`. |
| `owner` | string | Agent, workflow, PR, or maintainer holding the lock. |
| `owner_pr` | string | Optional PR URL or number. |
| `acquired_at` | RFC3339 timestamp | When the lock was acquired. |
| `expires_at` | RFC3339 timestamp or null | Optional lease expiration. |
| `release_policy` | string | When the lock should be released. |

### ProbabilisticConflictIndex

| Field | Type | Description |
|---|---|---|
| `index_id` | string | Stable index id. |
| `kind` | enum | `bloom_filter`, `counting_bloom_filter`, `cuckoo_filter`. |
| `scope` | string | What the index summarizes. |
| `false_positive_rate` | number | Expected false-positive rate. |
| `authoritative` | boolean | Must be false. |
| `exact_check_required` | boolean | Must be true. |

Probabilistic structures may only say “there may be a conflict”. They must never be used to prove that a lock is free.

### BackpressureDecision

| Field | Type | Description |
|---|---|---|
| `decision_id` | string | Stable decision id. |
| `issue` | string | Issue being evaluated. |
| `decision` | enum | `admit`, `defer_backpressure`, `defer_lock_conflict`, `reject_policy`, `halted`. |
| `reason` | string | Human-readable reason. |
| `observed_open_prs` | integer | Open cloud-agent PRs at decision time. |
| `global_wip_limit` | integer | Limit used by the decision. |
| `conflicting_locks` | array of string | Lock ids or resource keys that caused a conflict. |
| `retry_after` | string or null | Optional retry hint. |

## Invariants

- An issue may have at most one open cloud-agent PR.
- A workflow lane may have at most one open workflow-change PR unless explicitly approved.
- Runtime lane dispatch is disabled by default unless an issue explicitly carries approval metadata.
- Bloom filters and other probabilistic structures are never authoritative.
- Governance halt overrides all admission decisions.
- Human/Demerzel review remains mandatory before merge.
