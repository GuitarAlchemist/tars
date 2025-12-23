# Phase 6.7: Circuit Flow Control (Resistors, Capacitors, Transistors)

**Status**: ✅ **COMPLETED**
**Date**: 2025-12-03
**Priority**: High (Architecture Hardening)

## Overview

This phase implements the "Circuit Flow Control" patterns inspired by electrical engineering to manage information flow within the TARS cognitive architecture. These patterns prevent system overload (Resistors), smooth out bursts (Capacitors), and control execution flow (Transistors).

*Note: Previous documentation referred to "Epistemic Governor" as Phase 6.7. To avoid confusion, this work is explicitly titled "Circuit Flow Control" and fulfills the requirements of Phase 6.7 in the original Implementation Plan.*

## Components

### 1. Resistors (Throttling & Backpressure)

**Goal**: Prevent any single component from overwhelming the system.

**Implementation**:

* **Bounded Channels**: Replace `Channel.CreateUnbounded` with `Channel.CreateBounded` in `EventBus`.
* **Backpressure Handling**: When the channel is full, producers must either wait (async) or drop the message (if low priority).
* **Configurable Capacity**: Allow capacity to be tuned via configuration.

```fsharp
type EventBusConfig = {
    ChannelCapacity: int
    FullMode: BoundedChannelFullMode
}
```

### 2. Capacitors (Buffering & Batching)

**Goal**: Smooth out bursts of activity and enable efficient batch processing.

**Implementation**:

* **BufferAgent**: A specialized agent that accumulates messages and releases them in batches or after a timeout.
* **Use Case**: Batching log entries, aggregating partial thoughts, or collecting voting results.

```fsharp
type BufferAgent<'T>(capacity: int, timeWindow: TimeSpan, onFlush: 'T list -> Async<unit>)
```

### 3. Transistors (Gating & Conditional Flow)

**Goal**: Control the flow of execution based on dynamic conditions (e.g., "Wait for 3 agents to agree").

**Implementation**:

* **TaskDependencyGate**: A mechanism to block execution until specific conditions are met.
* **Logic Gates**: AND, OR, NOT gates for signal combination.

```fsharp
type Gate(condition: unit -> Async<bool>)
member this.WaitForOpen() : Async<unit>
```

## Implementation Plan

1. **Refactor EventBus**:
    * Update `EventBus` constructor to accept configuration.
    * Switch to `Channel.CreateBounded`.
    * Handle `TryWrite` failures (backpressure).

2. **Implement BufferAgent**:
    * Create `src/Tars.Kernel/Capacitor.fs`.
    * Implement `BufferAgent` logic using `MailboxProcessor` or `Channel`.

3. **Implement Gates**:
    * Create `src/Tars.Kernel/Transistor.fs`.
    * Implement `Gate` and `DependencyGate`.

4. **Testing**:
    * Verify backpressure behavior (Resistor).
    * Verify batching behavior (Capacitor).
    * Verify gating behavior (Transistor).

## Acceptance Criteria

* [x] `EventBus` drops or blocks messages when capacity is exceeded (Resistor).
* [x] `BufferAgent` correctly flushes on size or time (Capacitor).
* [x] `Gate` blocks execution until condition is true (Transistor).
* [x] All existing tests pass.
