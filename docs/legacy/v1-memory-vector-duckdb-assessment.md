# Boundary Note: V1 Memory, VectorStore, and DuckDB Assessment (TARS V2)

**Status:** Draft | **Area:** Runtime/Intelligence | **Priority:** P1
**Parent Issue:** #76
**Related:** GuitarAlchemist/ix#191, GuitarAlchemist/ix#189

## Overview
This document defines the architectural boundary for TARS V2 memory management, contrasting it with the legacy TARS V1 (Multi-Space) implementation and delineating the responsibilities between the **TARS Runtime** and the **IX Algorithm Engine**. It also frames the adoption of DuckDB/Parquet as a local-first, cost-effective persistence strategy.

## 1. Minimum TARS V2 Memory Interface
TARS V2 prioritizes a lean, "Agent-to-Agent" (A2A) compatible memory interface. Unlike V1, which embedded complex mathematical transforms into the store itself, V2 treats the VectorStore as a commodity persistence layer.

### Core Interface (`IVectorStore`)
Located in `v2/src/Tars.Core/Abstractions.fs`:
```fsharp
type IVectorStore =
    abstract member SaveAsync:
        collection: string * id: string * vector: float32[] * payload: Map<string, string> -> Task

    abstract member SearchAsync:
        collection: string * vector: float32[] * limit: int -> Task<(string * float32 * Map<string, string>) list>
```

### Key Principles
- **Simplicity:** The runtime only cares about `float32[]` and metadata.
- **Independence:** Embedding generation is decoupled from storage.
- **Local-First:** Implementation defaults to `InMemoryVectorStore` with optional file-based JSON persistence.

## 2. V1 Legacy Sources for Inspection
The legacy V1 implementation contains advanced logic for multi-dimensional reasoning that should be harvested as **IX Skills** rather than ported directly into the TARS V2 Cortex.

**Key Sources:**
- `Tars.Engine.VectorStore/Types.fs`: Definition of `MultiSpaceEmbedding` (Raw, FFT, Dual, Projective, Hyperbolic, Wavelet, Minkowski, Pauli).
- `Tars.Engine.VectorStore/VectorStore.fs`: Logic for multi-space similarity aggregation and `TruthValue` (tetravalent logic) integration.
- `TarsEngine.CUDA.VectorStore/`: (Deferred) GPU-accelerated vector operations.

## 3. DuckDB and Parquet: Local-First Persistence
DuckDB is framed as a **Local-First / Cost-Safe** alternative to always-on infrastructure (like Postgres+pgvector or Milvus).

### Candidate Uses
- **Memory Analytics:** Running OLAP queries over millions of historical reasoning traces.
- **Cold Storage:** Offloading older memory collections to Parquet files, searchable via DuckDB without a running server.
- **Evidence Bundling:** Packaging research findings into portable Parquet files for cross-repo sharing.
- **Consistency Gating:** Validating memory integrity using SQL constraints before promotion to "Golden Traces."

## 4. Boundary: TARS vs. IX
A clear separation is maintained to prevent "Anti-Ball-of-Mud" entropy.

| Feature | TARS (Runtime/Cortex) | IX (Engine/Algorithms) |
|---------|-----------------------|------------------------|
| **Primary Goal** | Orchestration & Governance | Heavy Computation & Search |
| **Memory Role** | Retrieval of Context/Skills | Execution of MCTS/Optimization |
| **Logic** | Symbolic, DSL-driven, Rules | Probabilistic, Tensor-driven |
| **Complex Spaces** | Metadata/Tags (PARA/Zettel) | FFT, Pauli, Minkowski Transforms |
| **State** | Epistemic Status (Hypothesis/Fact) | Weight Optimization / PSO |

## 5. Deferred Capabilities (Out of Scope for V1/V2 Alpha)
To maintain the "free-local" tier and low infrastructure overhead, the following are explicitly deferred:
- **CUDA/GPU Acceleration:** TARS-side vector operations remain CPU-bound (SIMD optimized). GPU work is delegated to specialized IX backends or Ollama.
- **Always-on Services:** Full deployments of Chroma (as a service), MongoDB, Redis, or heavyweight RDF stores (Virtuoso/Fuseki) are not required for core TARS operation.
- **Distributed Vector Indices:** Sharding and multi-node vector consistency are deferred until enterprise-tier requirements are met.

---
**Verification Marker:** This note satisfies the documentation-only boundary requirement for Issue #76. Follow-up prototypes for DuckDB implementation should be tracked in separate technical spikes.
