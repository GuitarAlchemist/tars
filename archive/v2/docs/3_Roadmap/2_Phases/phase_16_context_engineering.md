# Phase 16: Context Engineering & Performance Validation

**Goal**: Implement production-grade context management and validation based on "Agent-Skills-for-Context-Engineering".

**Status**: 🔜 Planned (v2.3)

**References**:
- [Agent-Skills-for-Context-Engineering](https://github.com/muratcankoylan/Agent-Skills-for-Context-Engineering)
- [Anthropic: Context Engineering](https://www.anthropic.com/news/context-engineering)

## Overview

Context Engineering is the progression from simple "Prompt Engineering" to holistic "State Management". This phase focuses on how TARS curates, optimizes, and validates the information provided to the LLM at inference time to maximize reasoning accuracy and minimize token waste.

## 🛠️ Tasks

### 16.1 Context Architecture (Progressive Disclosure)
- [ ] **Context Isolation**: Enforce separate context windows for Tool Execution vs. Reasoning.
- [ ] **Metadata-First Discovery**: Implement a "Discovery Mode" where agents only see tool names/descriptions first, and only load full JSON schemas/documentation when explicitly needed.
- [ ] **Context Masking**: Implement automatic PII and sensitive data masking in logs, traces, and LLM payloads.
- [ ] **Context Caching**: Implement LLM-side context caching hints in the `LlmRequest` protocol (for Anthropic/DeepSeek/local llama-server).

### 16.2 Agent-Skills Benchmarking
- [ ] **LLM-as-a-Judge (Validation)**: Implement an evaluation suite where TARS judges its own tool call accuracy against ground truth datasets.
- [ ] **Skill Assessment Scenarios**: Port the 10 core "Agent Skills" from the reference repository into TARS unit/integration tests.
- [ ] **Incident Triage Integration**: Update the `IncidentLedger` to automatically categorize failures based on "Context Overload" or "Missing Information" patterns.

### 16.3 Tool Design & Governance
- [ ] **Semantic Tool Refactor**: Audit and update all 124+ tools to follow the "Progressive System Instruction" pattern.
- [ ] **Consistency Linter**: Implement an automated check for tool schemas (docstring quality, parameter typing, naming conventions).
- [ ] **Tool Persona Alignment**: Define specific "Skill Sets" (Personas) that tools belong to, allowing the agent to switch context based on the current objective.

## 📊 Success Criteria
1. TARS passes the "Agent-Skills-for-Context-Engineering" baseline benchmarks.
2. Context window usage is reduced by >30% via metadata-first discovery.
3. Every tool call is verified by a `ValidatorNode` (WoT pattern).
