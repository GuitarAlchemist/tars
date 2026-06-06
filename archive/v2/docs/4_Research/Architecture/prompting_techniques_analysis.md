# TARS Prompting Techniques Analysis

**Date**: 2025-12-21  
**Status**: Research Complete  
**Goal**: Identify prompting techniques to incorporate into TARS

---

## Executive Summary

After analyzing resources from:
- [Awesome-Graph-Prompting](https://github.com/AndrewZhou924/Awesome-Graph-Prompting)
- [LearnPrompting.org](https://learnprompting.org/docs/introduction)
- [PromptingGuide.ai](https://www.promptingguide.ai/)

This document outlines which techniques TARS already implements, which are gaps, and recommendations for the roadmap.

---

## Current TARS Capabilities

### ✅ Already Implemented

| Technique | Location | Status |
|-----------|----------|--------|
| **ReAct (Reason-Act-Observe)** | `Tars.Cortex/Patterns.fs` | ✅ Full implementation |
| **Chain of Thought (CoT)** | `Tars.Cortex/Patterns.fs` | ✅ Full implementation |
| **Few-Shot Prompting** | `Tars.Core/Persona.fs` | ✅ `FewShotExample` type |
| **RAG** | `Tars.Cortex/VectorStore.fs` | ✅ Semantic retrieval |
| **Prompt Templates** | `Tars.Metascript/Templates.fs` | ✅ 9 templates |
| **Tool Usage (MRKL)** | `Tars.Tools/*` | ✅ 75+ tools |
| **Reflection** | `Tars.Core/GraphitiTypes.fs` | ✅ Episode type |

---

## Gap Analysis: Missing Techniques

### 🔴 High Priority (Should Implement for Phase 8)

#### 1. **Tree of Thoughts (ToT)**
**What**: Maintains a tree of reasoning paths, evaluates progress, uses search algorithms (BFS/DFS) for exploration with backtracking.

**Benefits for TARS**:
- Better at complex multi-step tasks
- Can evaluate multiple reasoning paths
- Enables self-correction through backtracking

**Integration Point**: Extend `Patterns.fs` with `treeOfThoughts` function

**Estimated Effort**: 4-6 hours

---

#### 2. **Self-Consistency**
**What**: Generate multiple reasoning chains, then aggregate (majority vote) for final answer.

**Benefits for TARS**:
- Improves reliability of complex reasoning
- Reduces single-path errors
- Works well with existing CoT

**Integration Point**: Add `selfConsistent` wrapper to `Patterns.fs`

**Estimated Effort**: 2-3 hours

---

#### 3. **Graph Prompting**
**What**: Leverage graph structures in prompts - schema-aware prompts, graph-to-text, knowledge graph reasoning.

**Benefits for TARS**:
- Already has `TemporalKnowledgeGraph` and `BeliefGraph`
- Can enhance reasoning with graph context
- Enable "reason over the knowledge graph"

**Key Papers**:
- StructGPT: LLM reasoning over structured data
- GraphPrompt: Unifying pre-training and downstream tasks
- PRODIGY: In-context learning over graphs

**Integration Point**: New module `GraphReasoning.fs` in `Tars.Cortex`

**Estimated Effort**: 6-8 hours

---

### 🟡 Medium Priority (Consider for Phase 9+)

#### 4. **Prompt Chaining**
**What**: Break complex tasks into subtasks with explicit handoffs between prompts.

**Current State**: TARS has task decomposition but could formalize chaining.

**Integration Point**: Extend `Metascript` DSL

**Estimated Effort**: 3-4 hours

---

#### 5. **Automatic Prompt Optimization (APE)**
**What**: Use LLM to generate/optimize prompts automatically.

**Benefits for TARS**:
- Self-improving prompts
- Aligns with Ouroboros vision

**Integration Point**: New module in `Tars.Evolution`

**Estimated Effort**: 8-10 hours

---

#### 6. **Iterative Verification (PiVe)**
**What**: Iteratively verify and refine LLM outputs through multiple rounds.

**Benefits for TARS**:
- Complements existing Epistemic Verification
- Improves output quality

**Integration Point**: Enhance `EpistemicGovernor`

**Estimated Effort**: 4-5 hours

---

### 🟢 Low Priority (Nice to Have)

| Technique | Description | Effort |
|-----------|-------------|--------|
| **Zero-Shot CoT** | "Let's think step by step" | 1 hour |
| **Least-to-Most Prompting** | Simplest first | 2 hours |
| **Active Prompting** | Uncertainty sampling | 4 hours |
| **Directional Stimulus** | Hint-based prompts | 2 hours |

---

## Recommended Roadmap Addition

### Phase 8: Advanced Prompting Techniques

| Task | Priority | Status | Effort |
|------|----------|--------|--------|
| **8.1 Tree of Thoughts** | High | 🔜 | 4-6h |
| **8.2 Self-Consistency** | High | 🔜 | 2-3h |
| **8.3 Graph Prompting** | High | 🔜 | 6-8h |
| **8.4 Prompt Chaining DSL** | Medium | 🔜 | 3-4h |
| **8.5 Zero-Shot CoT** | Low | 🔜 | 1h |

**Total Estimated Effort**: ~18-24 hours

---

## Implementation Recommendations

### Tree of Thoughts Implementation

```fsharp
/// Tree of Thoughts: Explore multiple reasoning paths with backtracking
type ThoughtNode = {
    Thought: string
    Score: float  // "sure" = 1.0, "maybe" = 0.5, "impossible" = 0.0
    Children: ThoughtNode list
    IsTerminal: bool
}

/// ToT with BFS exploration
let treeOfThoughts 
    (llm: ILlmService) 
    (branching: int) 
    (maxDepth: int) 
    (task: string) 
    : AgentWorkflow<string> =
    workflow {
        // 1. Decompose task into steps
        // 2. For each step, generate `branching` thoughts
        // 3. Evaluate each thought (sure/maybe/impossible)
        // 4. Keep top-k candidates
        // 5. Continue until solution or max depth
        // 6. Return best path
    }
```

### Self-Consistency Implementation

```fsharp
/// Self-Consistency: Run CoT multiple times and vote
let selfConsistent 
    (llm: ILlmService) 
    (samples: int) 
    (task: string) 
    : AgentWorkflow<string> =
    workflow {
        // Run CoT `samples` times
        let! results = 
            [1..samples] 
            |> List.map (fun _ -> chainOfThought llm task)
            |> Async.Parallel
        
        // Extract answers and vote
        let answer = majorityVote results
        return answer
    }
```

### Graph Prompting Integration

```fsharp
/// Graph-enhanced prompt context
let withGraphContext 
    (graph: TemporalKnowledgeGraph) 
    (query: string) 
    (basePrompt: string) 
    : string =
    // 1. Find relevant entities in graph
    // 2. Extract subgraph around entities
    // 3. Format as structured context
    // 4. Prepend to base prompt
    sprintf """
You have access to the following knowledge graph context:

%s

Query: %s

%s
""" (formatSubgraph subgraph) query basePrompt
```

---

## References

1. **Tree of Thoughts** - Yao et al., 2023 - [Paper](https://arxiv.org/abs/2305.10601)
2. **Self-Consistency** - Wang et al., 2022 - [Paper](https://arxiv.org/abs/2203.11171)
3. **Graph Prompting Survey** - [GitHub](https://github.com/AndrewZhou924/Awesome-Graph-Prompting)
4. **StructGPT** - 2023 - [Paper](https://arxiv.org/abs/2305.09645)
5. **PRODIGY** - 2023 - [Paper](https://arxiv.org/pdf/2305.12600.pdf)
6. **Prompt Report** - Schulhoff et al. - 200+ techniques analyzed

---

## Conclusion

TARS already has solid implementations of the core agentic patterns (ReAct, CoT, Tool Use). The main gaps are in **advanced reasoning strategies** (ToT, Self-Consistency) and **graph-aware prompting** which would synergize with the existing knowledge graph infrastructure.

**Recommended Next Steps**:
1. Add Phase 8 to roadmap
2. Implement Tree of Thoughts first (highest ROI)
3. Add Self-Consistency wrapper
4. Explore Graph Prompting for knowledge graph integration

---

*Analysis complete - 2025-12-21*
