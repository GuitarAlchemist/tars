# ADR: Vector Store GPU Kernel Implementation Strategy

**Date:** November 26, 2025  
**Status:** Accepted  
**Decision:** Defer GPU kernel implementation to v3+; use CPU-based similarity with strategy pattern for future extensibility

---

## Context

The TARS v1 codebase contained CUDA-based GPU acceleration for vector similarity computations:

- `CudaKernels.cu` (12,990 bytes)
- `CudaInterop.fs` (17,405 bytes)
- `CudaNeuralNetwork.fs` (15,011 bytes)
- `CustomCudaInferenceEngine.fs` (15,790 bytes)
- Plus GPU directory

The question arose whether to port this GPU kernel implementation to the v2 `InMemoryVectorStore` and introduce multiple computation strategies.

---

## Decision

**We will NOT port the GPU kernel implementation to v2.**

The current CPU-based cosine similarity implementation in `InMemoryVectorStore.fs` is sufficient for v2 goals. GPU acceleration is explicitly deferred to v3+.

---

## Rationale

### 1. Alignment with v2 Philosophy

The v2 design principles emphasize:
> "Small, Local, Pragmatic, Easy to evolve later"

GPU kernels add complexity without clear benefit at the current scale.

### 2. Deployment Complexity

CUDA introduces significant deployment requirements:
- NVIDIA GPU hardware dependency
- CUDA toolkit installation
- Platform-specific builds
- Driver compatibility issues

### 3. Not Required for Current Scope

Phase 2.2 (Memory Grid) specifies:
> "One collection, basic cosine similarity"

The current implementation fulfills this requirement.

### 4. Low Reusability Assessment

Per `v1_component_reusability_analysis.md`:
- CUDA components: **~10% reusability** (concepts only)
- Category: **❌ DEFER - Keep as reference, not for v2**
- Timeline: **v3+**

### 5. Graceful Degradation Pattern

The `BRIDGING_THE_GAPS.md` document recommends:
> "For non-essential services like CUDA, the system should detect failures and fall back to a CPU-based implementation"

This implies GPU should be optional, not core.

---

## Future Strategy Pattern (v3+)

When performance optimization becomes necessary, introduce a strategy pattern:

```fsharp
/// Similarity computation strategy interface
type ISimilarityStrategy =
    abstract member ComputeSimilarity: float32[] -> float32[] -> float32
    abstract member ComputeBatch: float32[] -> float32[][] -> float32[]
    abstract member Name: string
```

### Planned Implementations

| Strategy | Complexity | Expected Benefit | Target Phase |
|----------|------------|------------------|--------------|
| **CPU Sequential** | Low ✓ | Baseline (current) | v2 |
| **CPU Parallel** | Low | 2-4x speedup | v2.x |
| **CPU SIMD** | Medium | 4-8x speedup | v2.x |
| **Batching** | Medium | Better throughput | v3 |
| **FP16 Storage** | Medium | 2x memory savings | v3 |
| **CUDA Kernels** | High | 10-100x for large datasets | v3+ |
| **Multi-GPU** | Very High | Distributed workloads | v3+ |

---

## Consequences

### Positive

- Simpler deployment (no CUDA dependency)
- Faster development velocity
- Lower maintenance burden
- Easier testing and debugging
- Cross-platform compatibility

### Negative

- Limited performance for very large vector stores (millions of vectors)
- May need refactoring when GPU support is added

### Mitigations

- Design `InMemoryVectorStore` to accept a strategy parameter (future)
- Keep v1 CUDA code as reference for v3+ implementation
- Monitor performance metrics to determine when optimization is needed

---

## References

- `docs/4_Research/V1_Insights/v1_component_reusability_analysis.md` - Component reuse assessment
- `docs/4_Research/Architecture/03_Operational/BRIDGING_THE_GAPS.md` - Performance optimization roadmap
- `docs/3_Roadmap/1_Plans/implementation_plan.md` - v2 phased roadmap
- `src/Tars.Cortex/InMemoryVectorStore.fs` - Current implementation
