# Feature Specification: Tiered Dynamic Evolution

**Feature Branch**: `421-dynamic-tier`
**Created**: 2025-03-15
**Status**: Draft

### User Story 1 - Deterministic spawn plan (Priority: P1)

**Acceptance Scenarios**:
1. Given the spec, When the dynamic closure runs, Then at least two agent spawns are recorded.

### Edge Cases
- What happens when the metascript inline grammar is malformed?

```metascript
SPAWN QRE 2 HIERARCHICAL
SPAWN ML 1 FRACTAL
CONNECT leader agent-1 directive
CONNECT agent-1 agent-2 support
METRIC innovation 0.85
METRIC stability 0.72
REPEAT adaptive 3
```

```expectations
rules=7
max_depth=3
spawn_count=2
connection_count=2
pattern=adaptive
metric.innovation=0.85
metric.stability=0.72
```
