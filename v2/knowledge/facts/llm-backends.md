---
id: llm-backends
title: Supported LLM Backends
category: facts
confidence: high
source: told
tags: llm, backends, configuration
created: 2024-01-01T00:00:00Z
updated: 2024-01-01T00:00:00Z
---

# Supported LLM Backends

## Ollama (Local)
- **URL**: `http://localhost:11434/`
- **API**: `/api/chat`, `/api/embed`
- **Streaming**: NDJSON format
- **Best For**: Local development, code models

## vLLM (Self-Hosted)
- **URL**: `http://localhost:8000/`
- **API**: OpenAI-compatible `/v1/chat/completions`
- **Streaming**: SSE format
- **Best For**: High-throughput serving, reasoning models

## OpenAI
- **URL**: `https://api.openai.com/`
- **API**: `/v1/chat/completions`
- **Streaming**: SSE with `data:` prefix
- **Best For**: GPT-4, general tasks

## Anthropic
- **URL**: `https://api.anthropic.com/`
- **API**: `/v1/messages`
- **Best For**: Claude models, long context

## Google Gemini
- **URL**: `https://generativelanguage.googleapis.com/`
- **API**: `/v1beta/models/{model}:generateContent`
- **Best For**: Multimodal tasks

## Routing Hints
- `code` → Ollama (local code model)
- `reasoning` → vLLM (reasoning model)
- `fast` → Ollama (small fast model)
- `best` → OpenAI (GPT-4)

