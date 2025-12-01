---
id: about-tars
title: About TARS
category: meta
confidence: high
source: told
tags: identity, core, overview
created: 2024-01-01T00:00:00Z
updated: 2024-01-01T00:00:00Z
---

# What is TARS?

TARS (Transformative Autonomous Reasoning System) is a multi-agent AI framework built in F#.

## Core Principles

1. **Functional-First** - Immutable data, pure functions, composition
2. **Type-Safe** - Leveraging F#'s type system for correctness
3. **Self-Improving** - Evolution module for autonomous learning
4. **Epistemically Aware** - Tracks confidence and verifies claims

## Architecture

- **Tars.Core** - Domain types, patterns, budget management
- **Tars.Llm** - LLM backends (Ollama, OpenAI, vLLM, Anthropic)
- **Tars.Graph** - Agent state machine and prompt building
- **Tars.Tools** - Tool registry and standard tools
- **Tars.Evolution** - Self-improvement engine

## Key Capabilities

- Multi-backend LLM routing
- Agent state machine with tool calling
- Vector-based RAG (Retrieval Augmented Generation)
- Budget governance for token limits
- Streaming responses
- Curriculum-based evolution

