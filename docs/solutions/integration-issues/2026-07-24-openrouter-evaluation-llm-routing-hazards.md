---
title: OpenRouter evaluation — why it's deferred, and the routing hazards any aggregator integration must fix
date: 2026-07-24
category: integration-issues
component: Tars.Llm (Routing / OpenAiCompatibleClient / ConstrainedDecoding)
problem_type: integration
status: evaluated-deferred
verified: code-level analysis of Routing.fs + OpenAiCompatibleClient.fs; pricing verified against July 2026 sources
tags:
  - openrouter
  - llm-routing
  - model-family
  - constrained-decoding
  - guided-decoding
  - vllm
  - pattern-outcome-store
  - provider-aggregator
---

# OpenRouter evaluation — deferred, with routing hazards documented

## Problem

Should TARS route LLM traffic through OpenRouter (OpenAI-compatible
multi-provider aggregator)? Evaluated 2026-07-24; decision: **deferred**
("not for today"). This doc captures why, and the two latent hazards found
during the evaluation that apply to *any* future aggregator integration.

## Findings

### 1. `ModelFamily.classify` misroutes slash-style aggregator model IDs

`Routing.fs` classifies explicit model names by substring
(`"gpt"` → OpenAI, `"claude"` → Anthropic, `"gemini"` → Gemini, else local).
OpenRouter IDs are namespaced — `anthropic/claude-sonnet-4.5`,
`qwen/qwen3-coder` — so:

- `anthropic/...` matches "claude" → routed to **api.anthropic.com with the
  wrong key** (OpenRouter key, native endpoint) → auth failure at best.
- `qwen/...` matches nothing → LocalFamily → **silently sent to Ollama**,
  which either 404s the model or runs a different local model of the same name.

Any aggregator integration must handle slash-containing model IDs *before*
family classification (route them to the OpenAI-compatible endpoint), or add
an explicit `PreferredProvider = "OpenRouter"` branch.

### 2. Aggregators silently break EBNF constrained decoding

The three-force probabilistic grammar pipeline sends EBNF grammars via vLLM
`guided_decoding` in `extra_body` (`OpenAiCompatibleClient.fs`, `extra_body`
field). OpenRouter does not pass vLLM guided-decoding through; it supports
only `json_schema` response_format on some models. `ConstrainedDecoding.fs`
documents graceful degradation to prompt hints — meaning the failure is
**silent**: IntentPlan/BeliefUpdate/RepairProposal IR requests would return
unconstrained text with no error. Constrained-decoding traffic must stay
pinned to vLLM.

### 3. Provider-routing nondeterminism poisons the outcome store

OpenRouter load-balances a single model ID across providers with different
quantizations. PatternOutcomeStore / Bayesian pattern weights assume outcomes
for "model X" are comparable over time; silent provider switches add
unattributable variance. If ever adopted, pin the provider
(`provider.order` / `allow_fallbacks: false`) for any traffic that feeds
learning loops.

### 4. Economics and integration cost (July 2026)

- Token pricing is pass-through; platform fee is 5.5% on credit purchases
  ($0.80 minimum — small top-ups are proportionally expensive); BYOK is 5%
  past 1M requests/month. Some Anthropic models reportedly carry markup —
  for Claude, the native `AnthropicClient` or Claude Code fallback is
  strictly better.
- Integration is near-zero code: `OpenAiCompatibleClient` works as-is with
  `OpenAIBaseUri = https://openrouter.ai/api/v1` + key via CredentialVault
  (`OPENAI_API_KEY`). The routing fix above is the only required change.

## Decision

Deferred. If revisited, adopt narrowly: opt-in cloud tier for
`tars benchmark code` (large open models without local VRAM), keeping
Ollama/vLLM as the execution path and Claude Code as the strong-model
fallback. Not as primary backend, for reasons 2–3 above.

## Prevention

- Before wiring any OpenAI-compatible aggregator, test `ModelFamily.classify`
  against namespaced IDs (`vendor/model`) — add cases to the routing tests.
- Treat "graceful degradation" of constrained decoding as a red flag in
  provider selection: prefer hard failure over silent prompt-hint fallback
  for IR-schema requests.

## Related

- `v2/src/Tars.Llm/Routing.fs` — `ModelFamily.classify`, `chooseBackend`
- `v2/src/Tars.Llm/ConstrainedDecoding.fs` — grammar → `extra_body` bridge
- `v2/docs/TARS_ARCHITECTURE.md` §LLM backends (already lists OpenRouter as a
  supported OpenAI-compatible target)
- Memory: `openrouter-deferred.md` (project memory pointer to this decision)
