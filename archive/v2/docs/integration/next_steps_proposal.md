# TARS Improvement Proposal - Next Steps

**Date**: 2025-12-21  
**Current Status**: Phase 6.2 Complete (Budget Governance + Episode Ingestion)  
**Target**: Phase 6.3 and beyond

---

## 🎯 Completed (Phase 6.2)

✅ **Budget Governance**
- Budget Governor integrated into Evolution context
- CLI `--budget` parameter
- Real-time token/cost tracking
- Budget allocation and monitoring

✅ **Episode Ingestion**
- Graphiti integration via environment variable
- MetascriptContext updates across all files
- Graceful fallback when Graphiti unavailable

✅ **Infrastructure**
- All Evolution tests passing (4/4)
- Ollama connectivity verified
- Model configuration updated
- Documentation complete

---

## 🚀 Recommended Next Steps (Priority Order)

### **Option 1: Phase 6.3 - Semantic Speech Acts in Evolution** ⭐ RECOMMENDED
**Priority**: HIGH | **Effort**: Medium | **Impact**: HIGH

**What**: Integrate semantic speech acts into the Evolution loop for better agent communication.

**Why**: 
- Enables intent-driven communication between Curriculum and Executor agents
- Improves traceability and debugging
- Foundation for multi-agent collaboration

**Tasks**:
1. Update Evolution Engine to use `SemanticMessage` instead of plain messages
2. Implement speech act validation (Request → Inform flow)
3. Add performative logging for all agent interactions
4. Update tests to validate speech act patterns

**Files to Modify**:
- `src/Tars.Evolution/Engine.fs`
- `src/Tars.Core/SpeechActs.fs` (extend validation)
- `tests/Tars.Tests/EvolutionSemanticTests.fs`

**Estimated Time**: 2-3 hours

---

### **Option 2: Working Memory Capacitor** 
**Priority**: MEDIUM | **Effort**: Medium | **Impact**: MEDIUM

**What**: Implement working memory as a capacitor with importance-based pruning.

**Why**:
- Prevents memory bloat during long evolution sessions
- Maintains most relevant context
- Improves token efficiency

**Tasks**:
1. Create `WorkingMemory<'T>` type with decay function
2. Implement importance scoring for memory items
3. Add automatic pruning when capacity reached
4. Integrate with Evolution context

**Files to Create**:
- `src/Tars.Core/WorkingMemory.fs`
- `tests/Tars.Tests/WorkingMemoryTests.fs`

**Estimated Time**: 3-4 hours

---

### **Option 3: Enhanced Epistemic Verification**
**Priority**: HIGH | **Effort**: Large | **Impact**: VERY HIGH

**What**: Add epistemic verification checkpoints in Evolution loop to prevent hallucinations.

**Why**:
- Validates reasoning quality
- Detects and prevents hallucinations
- Ensures knowledge consistency

**Tasks**:
1. Add epistemic checkpoints after task completion
2. Implement belief consistency checking
3. Add hallucination detection heuristics
4. Integrate with curriculum generation

**Files to Modify**:
- `src/Tars.Evolution/Engine.fs`
- `src/Tars.Cortex/EpistemicGovernor.fs`
- Add hallucination detection module

**Estimated Time**: 4-6 hours

---

### **Option 4: Budget-Aware Task Prioritization**
**Priority**: MEDIUM | **Effort**: Small | **Impact**: MEDIUM

**What**: Prioritize tasks based on budget efficiency and expected value.

**Why**:
- Maximizes learning per dollar
- Prevents wasting budget on low-value tasks
- Enables smarter curriculum generation

**Tasks**:
1. Add cost estimation to `TaskDefinition`
2. Implement value/cost scoring function
3. Sort task queue by budget efficiency
4. Add budget projection to curriculum agent

**Files to Modify**:
- `src/Tars.Evolution/Engine.fs`
- `src/Tars.Core/Domain.fs` (add cost estimates)

**Estimated Time**: 2-3 hours

---

### **Option 5: Guitar Alchemist Integration** 🎸
**Priority**: LOW | **Effort**: Variable | **Impact**: HIGH (for GA)

**What**: Use TARS to help develop/improve Guitar Alchemist.

**Why**:
- Real-world validation of TARS capabilities
- Demonstrates practical value
- Synergy between projects (both use Semantic Kernel)

**Potential Applications**:
1. **Code Analysis**: TARS analyzes GA codebase for improvements
2. **Music Theory Algorithm Generation**: Evolution generates new chord/scale algorithms
3. **Test Generation**: TARS creates comprehensive tests for GA features
4. **API Documentation**: Auto-generate API docs from code
5. **Performance Optimization**: Identify bottlenecks

**Estimated Time**: Varies by task

---

## 📊 Impact vs Effort Matrix

```
High Impact
    │
    │  [Epistemic]    [Speech Acts]
    │       ●              ●
    │                          
    │  [GA Integration]  [Budget Priority]
    │       ●              ●
    │                  
    │       ●  [Working Memory]
    │
    └────────────────────────────── High Effort
   Low                              
```

---

## 🎯 My Recommendation: **Start with Phase 6.3 (Speech Acts)**

**Why this first?**
1. **Natural progression** from Phase 6.2
2. **Foundation** for advanced multi-agent features
3. **Clear deliverable** with existing test framework
4. **Medium effort** but high impact
5. **Unblocks** future epistemic and collaboration work

**Implementation Path**:
```
1. Update SemanticMessage usage in Evolution (30 min)
2. Add speech act validation (45 min)
3. Update logging and tracing (30 min)
4. Write tests for speech act flows (45 min)
5. Integration testing (30 min)
─────────────────────────────────────────────
Total: ~3 hours
```

---

## 🔮 Long-Term Vision (Phase 7+)

Based on the implementation plan:

### Phase 7: Multi-Agent Collaboration
- Agent swarms
- Hierarchical task delegation
- Consensus mechanisms

### Phase 8: Advanced Knowledge Graph
- Multi-hop reasoning
- Causal inference
- Knowledge graph traversal

### Phase 9: Production Hardening
- Performance optimization
- Scalability testing
- Production deployment

---

## 💡 Quick Wins (Can do in parallel)

While we work on the main features, these small improvements add value:

1. **Enhanced Logging**: Add structured logging with semantic context
2. **Better Error Messages**: User-friendly error descriptions
3. **Progress Indicators**: Real-time progress bars for long operations
4. **Cost Dashboards**: Visual budget consumption tracking
5. **Episode Replay**: Debug tool to replay evolution sessions

---

## ❓ Decision Point

**What would you like to focus on?**

A. **Phase 6.3 - Semantic Speech Acts** (Recommended - natural next step)
B. **Working Memory Capacitor** (Memory optimization)
C. **Epistemic Verification** (Quality/safety focus)
D. **Budget-Aware Prioritization** (Efficiency focus)
E. **Guitar Alchemist Integration** (Real-world application)
F. **Quick Wins** (Multiple small improvements)
G. **Something else** (Tell me what you have in mind!)

Let me know and I'll start implementing immediately! 🚀
