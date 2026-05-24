# TARS Autonomous Instruction

**Task**: Analyze Guitar Alchemist codebase for performance optimization opportunities
**Priority**: High
**Estimated Duration**: 2-3 hours
**Complexity**: Moderate
**Dependencies**: Guitar Alchemist codebase, TARS analysis tools

---

## OBJECTIVE

**Primary Goal**: Identify and document performance optimization opportunities in Guitar Alchemist

**Success Criteria**:
- [ ] Complete codebase analysis performed
- [ ] Performance bottlenecks identified
- [ ] Optimization recommendations generated
- [ ] Implementation priority assigned

**Expected Outputs**:
- Analysis Report: Detailed performance analysis in Markdown format
- Metrics Dashboard: Performance metrics and benchmarks
- Optimization Plan: Prioritized list of improvements

## CONTEXT

**Background**: Guitar Alchemist requires performance optimization for real-time audio processing

## WORKFLOW

### Phase 1: Codebase Analysis
**Objective**: Analyze codebase structure and identify performance-critical areas
**Duration**: 1 hour

**Steps**:
1. **Scan Codebase**: Identify all source files and dependencies
   - Action: Recursive analysis of project structure
   - Validation: Verify all source files cataloged
   - Error Handling: Continue with available files if some are inaccessible

2. **Identify Hot Paths**: Find performance-critical code paths
   - Action: Analyze function call patterns and complexity
   - Validation: Verify hot path identification accuracy
   - Error Handling: Use heuristics if static analysis fails

### Phase 2: Performance Analysis
**Objective**: Measure and analyze current performance characteristics
**Duration**: 1-2 hours

**Steps**:
1. **Benchmark Current Performance**: Establish baseline metrics
   - Action: Run performance tests and collect metrics
   - Validation: Verify benchmark completeness and accuracy
   - Error Handling: Use alternative metrics if primary benchmarks fail

2. **Identify Bottlenecks**: Find specific performance issues
   - Action: Analyze metrics to identify bottlenecks
   - Validation: Verify bottleneck identification accuracy
   - Error Handling: Flag uncertain areas for manual review
