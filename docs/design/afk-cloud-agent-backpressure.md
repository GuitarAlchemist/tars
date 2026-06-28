# AFK Cloud-Agent Backpressure and Locks

## Purpose

The AFK cloud-agent loop needs two related controls:

1. **Backpressure**: limit how much work is admitted into the cloud-agent pipeline.
2. **Locks**: prevent two agents from modifying the same issue, lane, or surface at the same time.

Backpressure protects review capacity. Locks protect correctness and reduce merge churn.

## Current risk

The current `ready-for-agent` flow can fan out quickly:

```text
ready-for-agent label
  -> workflow
  -> Jules/Codex
  -> PR
  -> CI / QA / review
```

Without admission control, this can create too many open PRs, duplicate work for the same issue, stale branches, and agents touching the same files.

## Backpressure model

```text
issue_ready_signal
  -> admission controller
  -> queue / policy decision
  -> lock acquisition
  -> cloud-agent dispatch
  -> PR opened
  -> CI / blackbox / QA verdict
  -> review queue
  -> merge / revise / close duplicate
  -> lock release
  -> metrics feedback
```

## Initial hard limits

The first implementation should be intentionally conservative:

| Control | Initial value | Reason |
|---|---:|---|
| Global open cloud-agent PR limit | 5 | Keeps review queue human-sized. |
| Open PRs per issue | 1 | Prevents duplicate agent work on the same issue. |
| Workflow-change PRs | 1 | Prevents competing edits to automation. |
| Runtime-change PRs | 0 by default | Runtime work needs explicit approval. |

## Lock types

### Issue lock

Only one open PR may claim a given issue number.

Lock key:

```text
issue:GuitarAlchemist/tars#118
```

### Surface lock

A surface is a coarse-grained repo area such as:

```text
.github/workflows/
docs/contracts/
docs/design/
docs/agents/
examples/agents/
v2/src/
v2/tests/
```

Lock key examples:

```text
surface:.github/workflows
surface:docs/contracts
surface:v2/tests
```

### Lane lock

A lane groups work by policy intent:

```text
lane:contract
lane:evidence
lane:policy
lane:workflow
lane:runtime
```

Workflow and runtime lanes should have the strictest concurrency.

## Exact locks vs probabilistic filters

Locks must be exact and authoritative. A Bloom filter can be useful as a fast pre-filter, but it must never be the source of truth because false positives are possible.

Recommended pattern:

```text
candidate work
  -> probabilistic conflict pre-filter
  -> exact GitHub PR/issue/label check
  -> lock decision
```

Bloom filters are acceptable for:

- quickly detecting that a surface might already be touched;
- reducing API calls before exact checks;
- summarizing large historical conflict sets;
- local/offline schedulers where occasional false positives only delay work.

Bloom filters are not acceptable for:

- deciding that a lock is free;
- deciding that a PR is safe to merge;
- releasing locks;
- replacing GitHub issue/PR state.

## Admission outcomes

| Outcome | Meaning |
|---|---|
| `admit` | Dispatch an agent. |
| `defer_backpressure` | Too many open PRs; retry after review queue shrinks. |
| `defer_lock_conflict` | Another PR already claims the issue or surface. |
| `reject_policy` | Task violates governance, risk, or allowed surfaces. |
| `halted` | Governance halt marker is active. |

## Release conditions

Locks are released when the PR is:

- merged;
- closed as duplicate;
- closed as not planned;
- superseded by a higher-priority PR;
- explicitly abandoned by the maintainer.

## Next implementation steps

1. Add admission control to `.github/workflows/jules-auto-delegate.yml`.
2. Count open cloud-agent PRs before dispatch.
3. Check whether the current issue already has an open PR.
4. Add a PR hygiene policy for stale/duplicate agent work.
5. Later, add surface-level locks using labels or a queue-state artifact.
