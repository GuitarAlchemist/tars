# TARS V1-to-V2 Decision Matrix

This document tracks the strategic decisions for porting, refactoring, or redesigning components from TARS v1 to v2. Decisions are based on reusability analysis and alignment with the v2 pragmatic architecture.

| Component | Owner | Decision | Priority | Complexity | Risk | Rationale |
|-----------|-------|----------|----------|------------|------|-----------|
| **VectorStore** | TARS | Port (High Reuse) | P0 | Low | Low | Core distillation logic is solid (70-90% reusable); already analyzed and ready for porting to `Tars.Memory.Vector`. |
| **AgenticTraceCapture** | TARS | Port (High Reuse) | P0 | Low | Low | Excellent observability foundation (80% reusable); perfectly aligns with v2's Observability Tower (Epic 5). |
| **Grammar components** | TARS | Port (High Reuse) | P0 | Low | Low | Core logic for grammar resolution is solid (75% reusable); essential for structured reasoning in `Tars.Cortex.Grammar`. |
| **TarsInferenceEngine** | TARS | Refactor | P1 | Medium | Low | Good orchestration logic (60% reusable) but needs to be decoupled and implementation-agnostic via `ICognitiveProvider`. |
| **AgentSystem** | TARS | Refactor | P1 | Medium | Low | Strong patterns (channels, capabilities) but requires genericization and transition to data-driven agent definitions. |
| **FLUX** | TARS | Refactor / Simplify | P2 | Medium | Low | Powerful multi-engine capability but needs simplification for v2; defer advanced fractal features to v3. |
| **TarsApiServer** | TARS | Redesign | P2 | High | Low | Extract route patterns and models but rebuild using modern stack (Giraffe/ASP.NET Core) for better kernel integration. |
| **Metascript executor** | TARS | Redesign | P2 | High | Low | Simplify execution model and focus on core block handlers; defer presentation-heavy features. |
| **DuckDB / Parquet storage** | [IX](https://github.com/GuitarAlchemist/ix/issues/189) | Candidate | P2 | Medium | Low | Recommended for analytical snapshots and high-performance memory queries. |
| **CUDA / GPU references** | [IX](https://github.com/GuitarAlchemist/ix/issues/189) | Defer to V3+ | P3 | Very High | Medium | Defer specialized hardware acceleration to maintain v2's pragmatic, local-first focus. |
| **Advanced Math** | [IX](https://github.com/GuitarAlchemist/ix/issues/189) | Defer to V3+ | P3 | Very High | Low | Exotic math DSLs (Sedenions, etc.) are out of scope for the v2 kernel's initial delivery. |
| **Tree / Workflow-of-Thought** | [IX](https://github.com/GuitarAlchemist/ix/issues/189) | Port / Unify | P1 | Medium | Low | Unify fragmented reasoning patterns (ToT, GoT, etc.) into a single "Workflow-of-Thought" (WoT) execution spine. |

## Sources
- `v2/docs/4_Research/V1_Insights/v1_reuse_strategy.md`
- `v2/docs/4_Research/V1_Insights/v1_component_reusability_analysis.md`
- `v2/docs/4_Research/V1_Insights/v1_chat_insights.md`
- `v2/docs/3_Roadmap/1_Plans/strategic_validation_dec2025.md`
