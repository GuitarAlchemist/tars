# Memory System Evaluation

## Current State (v2.0 Alpha)

- **Implementation**: `InMemoryVectorStore` (ephemeral)
- **Status**: Functional for Evolution Loop Phase 4
- **Limitations**: No persistence, no advanced retrieval

## Candidates for Phase 2.2 (Memory Grid)

### Option 1: ChromaDB (Original Plan)

- **Pros**: F# friendly, simple, persistent
- **Cons**: Basic retrieval, no memory "intelligence"
- **Effort**: Low (already in roadmap)

### Option 2: General Agentic Memory (GAM)

- **GitHub**: <https://github.com/VectorSpaceLab/general-agentic-memory>
- **Paper**: <https://arxiv.org/abs/2511.18423>

#### Architecture

- **MemoryAgent**: Constructs structured memory from sessions
- **ResearchAgent**: Iterative retrieval + reflection + summarization
- **JIT Principle**: Deep research at runtime vs. pre-computed

#### Performance

- SOTA on LoCoMo, HotpotQA, RULER, NarrativeQA
- Surpasses Mem0, MemoryOS, LightMem

#### Integration Path

1. **Python Microservice** (Recommended for v2.1):
   - Run GAM as HTTP service
   - F# calls via `HttpClient`
   - Pros: Full GAM features, minimal port effort
   - Cons: Python dependency, inter-process overhead

2. **F# Port** (Future):
   - Translate core concepts to `Tars.Cortex`
   - Pros: Native performance, type safety
   - Cons: High effort, ongoing maintenance

#### Alignment with TARS

- ✅ Dual-agent model matches our Evolution Engine
- ✅ "Deep research" fits Executor task solving
- ✅ Proven benchmarks validate approach
- ⚠️ Python vs F# interop needed

### Option 3: Graphiti (via AutoGen)

- **Status**: Phase 2.4 (Graph Memory)
- **Pros**: Temporal knowledge graphs, AutoGen integration
- **Cons**: Python dependency, graph complexity

## Recommendation

### For v2.0 Alpha (Now)

- ✅ Keep `InMemoryVectorStore`
- ✅ Complete Phase 4 (Evolution Loop)

### For v2.1 (Architecture Hardening)

- **Simple Persistence**: Use SQLite or File-based storage for VectorStore & Beliefs.
- **Internal Knowledge Graph**: Adopt **Graphiti** (or similar internal engine) for the Belief Graph.
- **Drop**: External Triple Stores and complex Vector DBs (ChromaDB) for now.

### For v2.2+ (Future)

- Evaluate GAM microservice if "Deep Research" becomes a bottleneck.
- Consider ChromaDB only when scale demands it.

## Decision Criteria

1. **Simplicity & Control** vs. Feature Reach (Internal KG wins)
2. **F# Native Safety** vs. Python Interop (SQLite/File wins)
3. **Evolution Support** vs. Raw Performance (Versioned Graph wins)

## Action Items

- [ ] Complete Phase 4 with InMemoryVectorStore
- [ ] Evaluate GAM Python microservice integration (PoC)
- [ ] Compare ChromaDB + GAM vs. Graphiti
- [ ] Document interop patterns for F# ↔ Python services
