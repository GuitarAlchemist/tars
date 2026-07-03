# AI Architectures Beyond LLMs

- **Status:** Draft
- **Area:** Research / Architecture Watch
- **Owner:** TARS
- **Version:** 1.0.0

## Purpose
This document tracks AI architectures beyond standard Large Language Models (LLMs) to evaluate their potential for improving the TARS second-brain / compounding harness. LLMs remain the primary tool for semantic interpretation and synthesis, but other architectures offer specialized capabilities in memory, replay, scoring, simulation, and local efficiency.

## 1. State Space Models (SSM) / Mamba-like Models
**Core Mechanics:** SSMs use linear recurrence or convolutions to process sequences with linear scaling relative to sequence length, unlike the quadratic scaling of standard Transformers.
**Strengths:** Extremely long context windows, high throughput, and constant-time inference per token.
**Weaknesses:** May struggle with complex, multi-hop reasoning compared to state-of-the-art Transformers.
**Harness Relevance:**
- **Trace Analytics:** Efficiently processing long event streams, execution logs, and historical traces.
- **Sequence Modeling:** Analyzing temporal patterns in agent behavior over thousands of steps.
- **Local Efficiency:** High-performance inference on local hardware for log analysis.

## 2. JEPA (Joint-Embedding Predictive Architecture)
**Core Mechanics:** Proposed by Yann LeCun, JEPA learns by predicting missing parts of an abstract representation of data rather than predicting every pixel or token.
**Strengths:** Focuses on high-level semantic features; ignores unpredictable noise.
**Weaknesses:** Still largely in the research phase for general-purpose tasks; complex to train.
**Harness Relevance:**
- **World Representations:** Creating stable, predictive models of the repository and environment.
- **Outcome Modeling:** Predicting the semantic "result" of an action without simulating every line of code.

## 3. World Models / Simulation Models
**Core Mechanics:** Explicitly modeling the dynamics of an environment to allow an agent to "dream" or simulate trajectories before taking action.
**Strengths:** Enable simulation-before-action, safety checking, and planning in complex environments.
**Weaknesses:** Computationally expensive; prone to "model drift" if the simulation diverges from reality.
**Harness Relevance:**
- **Replay & Simulation:** Simulating a TARS workflow execution in a sandbox before committing changes.
- **Trajectory Prediction:** Evaluating multiple potential paths for a WoT (Workflow-of-Thought) plan.

## 4. Neuro-symbolic Systems
**Core Mechanics:** Combining neural networks (for perception/intuition) with symbolic logic (for reasoning/constraints).
**Strengths:** Provable correctness, transparency, and high performance on structured data.
**Weaknesses:** "Search space explosion" in symbolic logic; difficulty in mapping fuzzy neural outputs to discrete symbols.
**Harness Relevance:**
- **Grammar & Contracts:** Using symbolic engines to validate LLM-generated DSL (`.trsx`) or code.
- **Deterministic Evaluation:** IX scoring using symbolic rules combined with neural ranking.

## 5. Graph Reasoning / Knowledge Graphs / GraphRAG
**Core Mechanics:** Representing data as entities and relationships in a graph, often combined with vector retrieval.
**Strengths:** Multi-hop retrieval, understanding global corpus structure, and detecting contradictions.
**Weaknesses:** High cost of graph construction and maintenance.
**Harness Relevance:**
- **Corpus-level Analysis:** Answering global questions about the Streeling catalog or codebase.
- **Staleness Analysis:** Tracking how a change in one part of the "second brain" affects distant, linked nodes.

## 6. Diffusion / Flow Models
**Core Mechanics:** Generative models that learn to reverse a noise process (Diffusion) or map simple distributions to complex ones (Flow).
**Strengths:** State-of-the-art for visual, audio, and continuous data generation.
**Weaknesses:** Slow iterative sampling; not designed for discrete symbolic reasoning.
**Harness Relevance:**
- **GA (GuitarAlchemist):** Simulating audio, gestures, and music-game pose data.
- **Synthetic Data:** Generating synthetic "traces" for training and benchmarking music-related skills.

## 7. Continual Learning / Adaptive Architectures
**Core Mechanics:** Models that can update their knowledge incrementally without "catastrophic forgetting."
**Strengths:** Enables models to adapt to a specific codebase or user over time.
**Weaknesses:** High risk of drift and instability; currently more reliable at the "harness" level (memory) than the "model" level.
**Harness Relevance:**
- **Long-term Adaptation:** Researching how local specialist models can adapt to specific repository patterns.
- **Note:** TARS prefers harness-level learning (updating files/memory) over direct model weight modification.

## 8. Small Specialist Models
**Core Mechanics:** Highly optimized models (often <7B parameters) trained for specific tasks like coding, math, or summarization.
**Strengths:** Low cost, high local performance, and easily "swappable" for specific steps in a pipeline.
**Weaknesses:** Limited general-purpose reasoning outside their specialty.
**Harness Relevance:**
- **Local Task Execution:** Using a specialist 1B-3B model for local linting, extraction, or unit-test generation.
- **Cost Efficiency:** Reducing dependence on expensive frontier models for repetitive "worker" tasks.
