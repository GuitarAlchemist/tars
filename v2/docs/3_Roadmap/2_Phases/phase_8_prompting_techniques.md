# Phase 8: Advanced Prompting Techniques

**Start Date**: 2025-12-22  
**Status**: 🚧 Partial (GoT Implemented)  
**Goal**: Implement state-of-the-art prompting strategies for enhanced reasoning.

---

## Overview

Phase 8 focuses on advanced prompting techniques that enhance TARS's reasoning capabilities beyond simple chain-of-thought prompting.

**Research Basis**: 
- [Awesome-Graph-Prompting](https://github.com/AndrewZhou924/Awesome-Graph-Prompting)
- [LearnPrompting.org](https://learnprompting.org/)
- [PromptingGuide.ai](https://www.promptingguide.ai/)

---

## Already Implemented (Pre-Phase 8)

| Technique | Location | Status |
|-----------|----------|--------|
| ReAct (Reason-Act) | `Tars.Cortex/Patterns.fs` | ✅ |
| Chain of Thought | `Tars.Cortex/Patterns.fs` | ✅ |
| Few-Shot Prompting | `Tars.Core/Persona.fs` | ✅ |
| RAG | `Tars.Cortex/VectorStore.fs` | ✅ |
| Tool Use (MRKL) | `Tars.Tools/*` | ✅ |

---

## 8.1 Tree of Thoughts (ToT)

**Priority**: High  
**Effort**: 4-6 hours  
**Status**: ✅ Completed (Dec 2025)

Explore multiple reasoning paths with BFS/DFS and backtracking.

| Task | Status | Description |
|------|--------|-------------|
| ThoughtNode type | ✅ | Define thought tree structure |
| BFS/DFS exploration | ✅ | Multi-path reasoning |
| Self-evaluation scoring | ✅ | Quality assessment |
| CLI `tars agent tot <task>` | ✅ | User interface |

---

## 8.2 Self-Consistency

**Priority**: High  
**Effort**: 2-3 hours  
**Status**: 🔜 Planned

Generate multiple CoT paths and majority-vote the answer.

| Task | Status | Description |
|------|--------|-------------|
| selfConsistent wrapper | 🔜 | Multiple CoT generation |
| Sample count config | 🔜 | Configurable sampling |
| Majority voting logic | 🔜 | Answer aggregation |

---

## 8.3 Graph Prompting

**Priority**: High  
**Effort**: 6-8 hours  
**Status**: 🔜 Planned

Leverage knowledge graph context in prompts.

**Key Papers**:
- StructGPT (reasoning over structured data)
- GraphPrompt (GNN prompting)
- PRODIGY (in-context learning over graphs)

| Task | Status | Description |
|------|--------|-------------|
| GraphReasoning.fs | 🔜 | Graph reasoning module |
| Subgraph extraction | 🔜 | Extract relevant context |
| Graph context injection | 🔜 | Inject into prompts |
| Knowledge queries | 🔜 | "Reason over knowledge" |

---

## 8.4 Graph-of-Thoughts (GoT) & Workflow-of-Thought (WoT)

**Priority**: High  
**Effort**: 8-12 hours  
**Status**: ✅ Core Implemented (Dec 2025)

Graph-structured reasoning with search over thought nodes and evaluation edges.

**References**: 
- https://github.com/spcl/graph-of-thoughts
- Medium: "Graph-of-Thought & Workflow-of-Thought" by Raktim Singh

### Implementation Status

| Task | Status | Description |
|------|--------|-------------|
| Edge Types (GoTEdgeType) | ✅ | Supports, Contradicts, DependsOn, etc. |
| Node Types (WoTNodeType) | ✅ | ThoughtNode, ToolNode, PolicyNode, etc. |
| Router Decisions | ✅ | Expand, Merge, Rollback, Escalate, etc. |
| CLI `tars agent got <task>` | ✅ | User interface implemented |
| Scoring and pruning | ✅ | Score threshold, top-K selection |
| WoT Controller | 🔜 | Planner, Generator, Critic, Router, Distiller |
| Policy node integration | 🔜 | Tars.Security integration |
| Memory node integration | 🔜 | Knowledge ledger integration |
| Human-in-the-loop | 🔜 | CLI prompts for escalation |
| Verifier node integration | 🔜 | External validators |
| Edge tracking audit trail | 🔜 | Persistence |

---

## 8.5 Prompt Chaining DSL

**Priority**: Medium  
**Effort**: 3-4 hours  
**Status**: 🔜 Planned

Formalize complex task handoffs in Metascript.

| Task | Status | Description |
|------|--------|-------------|
| Extended DSL syntax | 🔜 | Explicit chains |
| Validation points | 🔜 | Intermediate checks |
| Branching chains | 🔜 | Conditional flows |

---

## 8.6 Zero-Shot CoT

**Priority**: Low  
**Effort**: 1 hour  
**Status**: 🔜 Planned

Simple "Let's think step by step" enhancement.

| Task | Status | Description |
|------|--------|-------------|
| Prompt templates | 🔜 | Add to templates |
| Auto-apply | 🔜 | For reasoning tasks |

---

## Success Criteria

- [ ] Tree of Thoughts works for complex reasoning tasks
- [ ] Self-consistency improves answer accuracy
- [ ] Graph prompting leverages knowledge graph context
- [x] GoT CLI command functional
- [ ] Prompt chaining DSL integrated with Metascript

---

*Phase 8 initiated: 2025-12-22*
*GoT implemented: 2025-12*
