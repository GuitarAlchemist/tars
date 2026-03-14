# Feature Specification: Demo Tier4 Bootstrap

**Feature Branch**: `demo-tier4-bootstrap`
**Created**: 2025-03-15
**Status**: Draft

### User Story 1 - Establish Baseline (Priority: P1)

**Acceptance Scenarios**:
1. When the harness runs, it executes a validation command.
2. When the iteration completes, a follow-up goal is considered.

### Edge Cases
- Validation command fails due to missing project references.

```metascript
SPAWN GOAL_MANAGER 1 DIRECTIVE
DEFINE goal "Demo Tier4 Bootstrap"
ATTRIBUTE priority P1
ATTRIBUTE source_spec demo-feature
ATTRIBUTE follow_up "false"
```

```expectations
goals=1
priority=P1
parent="demo-feature"
```