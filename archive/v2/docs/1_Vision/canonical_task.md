# TARS v2 - Canonical Task

> **"Every phase must demonstrably improve this one task."**

## The Constraint

TARS is at risk of becoming excellent at thinking about thinking before it has proven it can outperform a human on one narrow, boring task.

To prevent architectural drift without killing ambition, we define a **canonical task** that serves as the grounding metric for all development.

---

## Canonical Task: "Refactor an F# File"

### Task Description

Given an F# source file:
1. **Analyze** the file for code smells, complexity, and improvement opportunities
2. **Generate a plan** for refactoring (with steps, preconditions, postconditions)
3. **Validate the plan** against project constraints (build must pass, tests must pass)
4. **Execute the refactoring** safely (with rollback capability). *Note: TARS may refactor user code, but never its own `Tars.Core` unless explicitly authorized via strict Constitution Update protocols.*
5. **Verify the result** (build, test, comparison)
6. **Record evidence** of improvement (metrics before/after)

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| Build passes after refactoring | 100% | Non-negotiable |
| Tests pass after refactoring | 100% | Non-negotiable |
| Code complexity reduced | > 0% | Measurable improvement |
| No human intervention required | True | Fully autonomous |
| Time to complete | < 5 minutes | Practical for CI/CD |

### Why This Task?

1. **Narrow**: One file, one operation, clear success/failure
2. **Boring**: No novelty bias - just competent execution
3. **Measurable**: Build passes or doesn't, tests pass or don't
4. **Real**: Actual value delivered to the codebase
5. **Compound**: Exercises the full loop (Plan → IR → Validate → Execute → Reflect)

---

## Phase Validation Gate

Before any phase is marked "complete", it must demonstrate improvement on the canonical task:

| Phase | How It Improves the Canonical Task |
|-------|-----------------------------------|
| Phase 7 (Production Hardening) | Reliable metrics/health during execution |
| Phase 9 (Symbolic Knowledge) | Knowledge of F# patterns and anti-patterns |
| Phase 13 (Neuro-Symbolic) | Constraint-guided refactoring suggestions |
| **Phase 17 (Hybrid Brain)** | **Plan validation prevents breaking changes** |
| Phase 14 (Agent Constitutions) | Safety rails for code modification |
| Phase 15 (Symbolic Reflection) | Learn from failed refactoring attempts |

---

## Benchmark Command

```bash
# The canonical task benchmark
dotnet run --project src/Tars.Interface.Cli -- refactor <file.fs> --validate --measure
```

### Expected Output

```
═══════════════════════════════════════════════════════════════
                    CANONICAL TASK: REFACTOR
═══════════════════════════════════════════════════════════════

Target: src/Tars.Core/Example.fs

PHASE 1: ANALYSIS
  ✓ Parsed 142 lines
  ✓ Found 3 code smells
  ✓ Complexity score: 7.2 (high)

PHASE 2: PLAN GENERATION
  ✓ Generated 4-step refactoring plan
  ✓ Plan validated against IR
  ✓ No policy violations

PHASE 3: EXECUTION
  ✓ Step 1: Extract helper function (lines 45-62)
  ✓ Step 2: Simplify pattern match (lines 78-95)
  ✓ Step 3: Remove dead code (lines 112-118)
  ✓ Step 4: Add documentation (lines 1-10)

PHASE 4: VERIFICATION
  ✓ Build: PASSED
  ✓ Tests: 24/24 PASSED
  ✓ Complexity: 7.2 → 4.1 (-43%)

RESULT: SUCCESS
Duration: 47.3s
Evidence: .tars/refactors/20251226_012000/evidence.json
═══════════════════════════════════════════════════════════════
```

---

## Anti-Patterns to Avoid

1. **"It works on paper"** - If it doesn't improve the canonical task, it waits
2. **"We need this for the future"** - Future features that don't help now are deferred
3. **"It's elegant"** - Elegance is a bonus, not a requirement
4. **"It enables X"** - Enablers without demonstrated value are speculative

---

## Current Status

| Question | Answer |
|----------|--------|
| Can TARS refactor a file today? | **Partially** (analysis yes, execution fragile) |
| What blocks full capability? | Plan → Execution pipeline not integrated |
| What Phase 17 adds | **Typed IR prevents breaking refactors** |

---

## The Rule

> **If a new mechanism (agent constitution, reflection layer, grammar evolution…) doesn't move the needle on the canonical task, it waits.**

This prevents architectural drift without killing ambition.

---

*Last updated: 2025-12-26*
*Status: Canonical task defined, benchmark not yet implemented*
