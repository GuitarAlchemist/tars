# TARS v2 - Phase 10: 3D Knowledge Graph Visualization

**Date:** December 22, 2025  
**Status:** Planning  
**Priority:** Medium (After Phase 9)

---

## 🎯 Executive Summary

This phase implements a **3D Knowledge Graph Visualization** system for TARS, following a three-layer architecture:

```
Graph Export → Layout → 3D Viewer
```

**Core Principle:** Keep the viewer dumb; keep the semantics in TARS.

The visualization is a **read-only lens** over the symbolic ledger, not a primary interface for knowledge manipulation.

---

## 📊 Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                    TARS Knowledge Visualization Pipeline               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Ledger          Graph              API                 Viewer          │
│   (Postgres)  →   (Neo4j/Graphiti) → (GraphSlice)    →   (3D UI)        │
│                                                                          │
│   ┌──────────┐    ┌──────────┐      ┌──────────┐       ┌──────────┐     │
│   │ Beliefs  │    │ Nodes    │      │ JSON     │       │ Three.js │     │
│   │ Events   │ => │ Edges    │  =>  │ {nodes,  │  =>   │ 3d-force │     │
│   │ Plans    │    │ Indexes  │      │  edges}  │       │ -graph   │     │
│   └──────────┘    └──────────┘      └──────────┘       └──────────┘     │
│                                                                          │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Phase 0: Data Contract (1-2 hours)

### 10.0.1 GraphSliceDto

```fsharp
type GraphNode = {
    Id: string           // e.g., "b:102", "c:hazard", "a:executor"
    Type: NodeType       // Belief | Concept | Agent | Run | File
    Label: string        // Human-readable name
    Confidence: float option
    Timestamp: DateTime option
    SourceRef: string option  // URL/file/line
    Version: int option
}

type NodeType = 
    | Belief | Concept | Agent | Run | File

type GraphEdge = {
    Source: string
    Target: string
    Type: EdgeType       // supports | contradicts | derived_from | mentions | produced_by
    Weight: float
    EvidenceRefs: string list
}

type EdgeType =
    | Supports | Contradicts | DerivedFrom | Mentions | ProducedBy

type GraphSliceDto = {
    Nodes: GraphNode list
    Edges: GraphEdge list
    QueryParams: SliceQueryParams
    Timestamp: DateTime
}

type SliceQueryParams = {
    RootId: string option
    Depth: int
    Limit: int
    NodeTypes: NodeType list option
    EdgeTypes: EdgeType list option
    MinConfidence: float option
}
```

### 10.0.2 API Endpoints

```
GET /api/graph/slice?rootId={id}&depth={n}&limit={m}
GET /api/graph/neighborhood/{nodeId}?depth=2
GET /api/graph/by-run/{runId}
GET /api/graph/contradictions?threshold=0.5
GET /api/graph/top-concepts?by=centrality&limit=20
GET /api/graph/stats
```

### Tasks

- [ ] Define `GraphSliceDto` in `Tars.Core/GraphDto.fs`
- [ ] Define query parameter types
- [ ] Add JSON serialization support
- [ ] Document API contract

---

## 🔧 Phase 1: Backend Graph Slice (Half Day)

### 10.1.1 Neo4j Query Templates

```cypher
// Neighborhood query
MATCH (n {id: $nodeId})-[r*1..$depth]-(m)
RETURN n, r, m
LIMIT $limit

// By run query
MATCH (r:Run {id: $runId})-[:PRODUCED|USED*1..3]-(n)
RETURN r, n

// Contradictions query
MATCH (a:Belief)-[c:CONTRADICTS]-(b:Belief)
WHERE c.weight >= $threshold
RETURN a, c, b

// Top concepts by degree
MATCH (c:Concept)-[r]-()
WITH c, count(r) as degree
ORDER BY degree DESC
LIMIT $limit
RETURN c, degree
```

### 10.1.2 Caching Strategy

```fsharp
type GraphCache = {
    Slices: ConcurrentDictionary<string, GraphSliceDto * DateTime>
    TTL: TimeSpan  // Default 5 minutes
}

let getCachedOrCompute (cache: GraphCache) (key: string) (compute: unit -> GraphSliceDto) =
    // Check cache, refresh if stale
```

### Tasks

- [ ] **10.1.1** Create `Tars.Api/GraphController.fs`
- [ ] **10.1.2** Implement Neo4j query templates
- [ ] **10.1.3** Add in-memory caching
- [ ] **10.1.4** Add rate limiting
- [ ] **10.1.5** Add `/api/graph/stats` endpoint
- [ ] **10.1.6** Unit tests for slice queries

---

## 🎨 Phase 2: Frontend 3D Viewer (Half Day)

### Technology Choice: **Option A** (Fast Path)

| Library | Purpose |
|---------|---------|
| `three` | WebGL 3D rendering |
| `3d-force-graph` | Force-directed layout |
| `d3-force-3d` | Physics simulation |
| React/Vite | UI framework |

> Option B (WebGPU) deferred to Phase 10.5 when scale demands it.

### 10.2.1 Component Structure

```
src/
├── components/
│   ├── GraphViewer.tsx        # Main 3D canvas
│   ├── NodePanel.tsx          # Side panel for node details
│   ├── FilterControls.tsx     # Filter chips, sliders
│   ├── SearchBox.tsx          # Node search
│   └── TimeSlider.tsx         # Future: version navigation
├── hooks/
│   ├── useGraphData.ts        # Fetch + cache slices
│   └── useGraphInteraction.ts # Click, hover, expand
└── api/
    └── graphApi.ts            # API client
```

### 10.2.2 Core Interactions

| Interaction | Behavior |
|-------------|----------|
| **Click node** | Side panel shows full details |
| **Double-click** | Expand neighborhood (fetch 1-2 hops) |
| **Hover** | Tooltip with label + type |
| **Scroll** | Zoom in/out |
| **Drag** | Orbit camera |
| **Right-click** | Context menu (hide, focus, expand) |

### Tasks

- [ ] **10.2.1** Initialize Vite + React project in `src/Tars.Visualize`
- [ ] **10.2.2** Install three, 3d-force-graph, d3-force-3d
- [ ] **10.2.3** Create `GraphViewer` component
- [ ] **10.2.4** Implement node click → side panel
- [ ] **10.2.5** Implement expand on double-click
- [ ] **10.2.6** Add filter chips (node types, edge types)
- [ ] **10.2.7** Add confidence range slider
- [ ] **10.2.8** Add search box with highlighting

---

## 🎨 Phase 2.5: Visual Encoding

### Node Encoding

| Attribute | Visual |
|-----------|--------|
| **Type** | Color (see palette below) |
| **Confidence** | Size (larger = more confident) |
| **Degree** | Halo brightness |

### Node Color Palette

| Type | Color | Hex |
|------|-------|-----|
| Belief | Orange | #F59E0B |
| Concept | Blue | #3B82F6 |
| Agent | Green | #10B981 |
| Run | Purple | #8B5CF6 |
| File | Gray | #6B7280 |

### Edge Encoding

| Type | Style |
|------|-------|
| `supports` | Thin solid line (green) |
| `contradicts` | Thick dashed (red glow) |
| `derived_from` | Curved arrow (blue) |
| `mentions` | Thin dotted (gray) |
| `produced_by` | Medium solid (purple) |

### Tasks

- [ ] **10.2.5.1** Create color constants module
- [ ] **10.2.5.2** Implement node sizing by confidence
- [ ] **10.2.5.3** Implement edge styling by type
- [ ] **10.2.5.4** Add legend component

---

## 🔗 Phase 3: TARS Integration (Later)

Make the viewer useful for reasoning, not just exploration.

### 10.3.1 "Explain Cluster" Button

1. User selects cluster of nodes
2. TARS generates symbolic summary (LLM)
3. Display as overlay or side panel
4. **Does not modify the graph**

### 10.3.2 "Create Plan from Contradictions"

1. User selects contradicting beliefs
2. TARS generates `plan.trsx` draft
3. Opens in editor/CLI
4. User can approve or modify

### 10.3.3 "Time Travel" (Version Navigation)

1. Time slider at bottom
2. Scrub to see graph "as of" version N
3. Useful for debugging belief evolution

### Tasks

- [ ] **10.3.1** Add cluster selection UI
- [ ] **10.3.2** Implement "Explain" → LLM call
- [ ] **10.3.3** Implement "Create Plan" button
- [ ] **10.3.4** Add time slider (requires version tracking)

---

## 📐 Implementation Timeline

### MVP (2 Weekends)

| Weekend | Deliverable |
|---------|-------------|
| **1** | Backend API + Neo4j queries + basic viewer rendering |
| **2** | Interactions (click, expand, filter) + visual polish |

### Full Feature Set (4 Weekends)

| Weekend | Deliverable |
|---------|-------------|
| **3** | Search, confidence slider, legends |
| **4** | TARS integration (Explain, Create Plan) |

---

## 🔗 Integration with Existing Systems

### Where Data Comes From

| Source | What It Provides |
|--------|-----------------|
| **Postgres (Ledger)** | Ground truth beliefs and events |
| **Neo4j/Graphiti** | Materialized graph view (fast traversal) |
| **Chroma** | Semantic search (for search box) |
| **TARS Agents** | Explanations and plan generation |

### Current Tools Ready

The following tools (from Phase 9+ work) support this visualization:

| Tool | Purpose |
|------|---------|
| `graph_add_node` | Populate graph |
| `graph_add_edge` | Add relationships |
| `graph_get_neighborhood` | Neighborhood query |
| `graph_query` | Filtered queries |
| `graph_export_json` | Export for viewer |
| `graph_find_contradictions` | Find conflicts |
| `graph_stats` | Overview metrics |

---

## 📊 Success Criteria

At the end of Phase 10, TARS should:

1. ✅ Expose `/api/graph/slice` endpoint with configurable queries
2. ✅ Render belief graphs in 3D with interactive navigation
3. ✅ Support click → details, double-click → expand
4. ✅ Filter by node type, edge type, confidence
5. ✅ Search nodes by label
6. ✅ Display contradictions with visual emphasis
7. ✅ Provide "Explain cluster" integration with TARS LLM

---

## 🚀 Phase 10.5: WebGPU Migration (Future)

When scale demands exceed Three.js capabilities (100K+ nodes):

1. Port renderer to WebGPU compute shaders
2. Implement GPU-accelerated force layout
3. Add LOD (Level of Detail) rendering
4. Support streaming graph updates

---

## 📚 References

- [3d-force-graph](https://github.com/vasturiano/3d-force-graph) - Force-directed 3D graph
- [Three.js](https://threejs.org/) - WebGL framework
- [D3-force-3d](https://github.com/vasturiano/d3-force-3d) - 3D physics
- [Neo4j Cypher](https://neo4j.com/docs/cypher-manual/) - Graph query language
- [WebGPU](https://gpuweb.github.io/gpuweb/) - Future GPU API

---

## ✅ Dependencies

| Dependency | Status |
|------------|--------|
| Phase 9 (Symbolic Knowledge Ledger) | Required for beliefs |
| Neo4j/Graphiti integration | Already integrated |
| Postgres ledger | Phase 9.1 |
| TARS Tools (graph_*) | ✅ Implemented today |

---

*This phase transforms TARS from a text-based reasoning system into a visual exploration tool that maintains symbolic rigor.*
