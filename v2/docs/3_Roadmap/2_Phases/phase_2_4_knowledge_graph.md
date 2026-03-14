# Phase 2.4: Internal Knowledge Graph (Graphiti)

## Overview

Implement a temporal knowledge graph inspired by Graphiti. This graph will represent the relationships between agents, tasks, code artifacts, and beliefs within the TARS system. It will support temporal queries to understand how the system's state and knowledge evolve over time.

## Goals

1. **Graph Structure**: Implement a directed property graph supporting `GraphNode` and `GraphEdge` types defined in `Domain.fs`.
2. **Temporal Versioning**: Track changes to the graph over time (nodes/edges added/removed/updated).
3. **Persistence**: Persist the graph to disk (JSON or SQLite).
4. **Querying**: Support basic graph traversal and temporal queries (e.g., "What did the graph look like at time T?").
5. **Integration**: Connect the graph to the `KnowledgeBase` (Markdown files) and `AgentRegistry`.

## Architecture

### Components

* **Tars.Core**:
  * `KnowledgeGraph.fs`: Core graph data structures and algorithms.
  * `TemporalGraph.fs`: Wrapper for managing graph history/snapshots.
* **Tars.Data** (New Project? Or `Tars.Core`?):
  * Likely keep in `Tars.Core` for now to avoid project sprawl, or `Tars.Kernel`. `Tars.Core` seems appropriate for domain logic.

### Data Model

* **Node**: `GraphNode` (from `Domain.fs`) + Metadata (Created, Updated).
* **Edge**: `GraphEdge` (from `Domain.fs`) + Source + Target + Metadata.
* **Snapshot**: A state of the graph at a specific timestamp.
* **Event**: A change to the graph (AddNode, RemoveNode, AddEdge, RemoveEdge).

## Implementation Steps

### Step 1: Core Graph Data Structure

Implement an in-memory graph structure using an adjacency list or similar.

* `Graph`: `Map<NodeId, Node>` and `Map<NodeId, Edge list>`.

### Step 2: Temporal Layer

Implement an event-sourced or snapshot-based approach.

* `TemporalGraph`: List of `GraphEvent`.
* `Rehydrate(timestamp)`: Replay events up to timestamp.

### Step 3: Persistence

Implement serialization for the graph/events.

* JSON serialization for simplicity initially.

### Step 4: Integration

* Update `KnowledgeBase` to emit graph events when entries are created/linked.
* Update `AgentRegistry` to emit graph events when agents are created.

## Acceptance Criteria

* [ ] Can add nodes and edges to the graph.
* [ ] Can retrieve the state of the graph at the current time.
* [ ] Can retrieve the state of the graph at a past time.
* [ ] Can traverse neighbors of a node.
* [ ] Graph state persists across restarts.
