# Phase 6.3 Implementation Tracker

**Date**: 2025-12-21  
**Status**: IN PROGRESS  
**Components**: A (Speech Acts) + D (Budget Priority) + B (Memory) + C (Epistemic)

---

## A. Semantic Speech Acts Implementation

### Current State
- `SemanticMessage<'T>` type exists in `Domain.fs`
- `SpeechActs` module has validation logic
- Evolution Engine uses direct function calls (no semantic messages)
- No speech act logging or validation in Evolution loop

### Implementation Tasks

#### A.1: Create Speech Act Wrapper Functions ✅ NEXT
**File**: `src/Tars.Evolution/SpeechActBridge.fs` (NEW)
- [ ] `requestTask`: Curriculum → Request → Executor
- [ ] `informResult`: Executor → Inform → Curriculum  
- [ ] `queryStatus`: Monitor → Query → Any Agent
- [ ] `refuseTask`: Executor → Refuse → Curriculum (budget/safety)

#### A.2: Update Evolution Engine
**File**: `src/Tars.Evolution/Engine.fs`
- [ ] Wrap `generateTask` call in Request message
- [ ] Wrap `executeTask` return in Inform message
- [ ] Add speech act validation before execution
- [ ] Log all speech acts with performative

#### A.3: Add Speech Act Logging
**File**: `src/Tars.Evolution/Engine.fs`
- [ ] Create `logSpeechAct` helper
- [ ] Track Request → Inform pairs
- [ ] Detect orphaned messages (Request without Inform)

#### A.4: Update Tests
**File**: `tests/Tars.Tests/EvolutionSemanticTests.fs`
- [ ] Test Request → Inform flow
- [ ] Test Refuse for budget exceeded
- [ ] Test speech act validation failures

---

## D. Budget-Aware Task Prioritization

### Implementation Tasks

#### D.1: Add Cost Estimation
**File**: `src/Tars.Core/Domain.fs`
- [ ] Add `EstimatedCost` field to `TaskDefinition`
- [ ] Add `ExpectedValue` (learning value) field

#### D.2: Implement Scoring Function
**File**: `src/Tars.Evolution/TaskPrioritization.fs` (NEW)
- [ ] `scoreTask`: Cost/Value ratio
- [ ] `prioritizeQueue`: Sort by score
- [ ] Budget projection for queue

#### D.3: Integrate into Evolution
**File**: `src/Tars.Evolution/Engine.fs`
- [ ] Score tasks after generation
- [ ] Sort task queue by priority
- [ ] Skip tasks if budget insufficient

---

## B. Working Memory Capacitor

### Implementation Tasks

#### B.1: Create Working Memory Type
**File**: `src/Tars.Core/WorkingMemory.fs` (NEW)
- [ ] `WorkingMemory<'T>` type with capacity
- [ ] Importance scoring function
- [ ] Time-based decay
- [ ] Automatic pruning

#### B.2: Integrate into Evolution
**File**: `src/Tars.Evolution/Engine.fs`
- [ ] Add to `EvolutionContext`
- [ ] Store recent task results
- [ ] Prune before new iteration
- [ ] Query for relevant memories

#### B.3: Add Tests
**File**: `tests/Tars.Tests/WorkingMemoryTests.fs` (NEW)
- [ ] Test capacity limits
- [ ] Test decay calculation
- [ ] Test pruning logic

---

## C. Epistemic Verification Checkpoints

### Implementation Tasks

#### C.1: Add Verification Points
**File**: `src/Tars.Evolution/Engine.fs`
- [ ] Post-task execution checkpoint
- [ ] Belief consistency check
- [ ] Hallucination detection heuristics

#### C.2: Enhance Epistemic Governor
**File**: `src/Tars.Cortex/EpistemicGovernor.fs`
- [ ] `VerifyTaskResult` method
- [ ] Consistency checking
- [ ] Quality scoring

#### C.3: Integration
- [ ] Call verification after task completion
- [ ] Log verification results
- [ ] Adjust curriculum based on quality

---

## Progress Tracking

### Session 1 (Current)
- [ ] A.1: Speech Act wrapper functions
- [ ] A.2: Evolution Engine integration  
- [ ] A.3: Logging
- [ ] A.4: Tests

**Estimated**: 3 hours

### Session 2  
- [ ] D.1-D.3: Budget prioritization
- [ ] B.1: Working Memory type

**Estimated**: 3-4 hours

### Session 3
- [ ] B.2-B.3: Memory integration
- [ ] C.1-C.3: Epistemic verification

**Estimated**: 4-6 hours

---

## Files to Create
1. `src/Tars.Evolution/SpeechActBridge.fs`
2. `src/Tars.Evolution/TaskPrioritization.fs`
3. `src/Tars.Core/WorkingMemory.fs`
4. `tests/Tars.Tests/WorkingMemoryTests.fs`

## Files to Modify
1. `src/Tars.Evolution/Engine.fs`
2. `src/Tars.Evolution/Tars.Evolution.fsproj`
3. `src/Tars.Core/Domain.fs`
4. `src/Tars.Core/Tars.Core.fsproj`
5. `src/Tars.Cortex/EpistemicGovernor.fs`
6. `tests/Tars.Tests/EvolutionSemanticTests.fs`
