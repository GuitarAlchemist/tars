# CONTEXT — tars domain glossary

> The shared language of the TARS repo. `/grill-with-docs` grows this lazily as
> terms get resolved; `/improve-codebase-architecture`, `/diagnose`, and `/tdd`
> read it so their output uses **our** words. This is a **seed** — add terms when
> a real ambiguity is resolved, not speculatively.

## What tars is

A neuro-symbolic, self-improving **F#** agent system: an LLM stochastic generator
paired with symbolic systems for memory, law, and self-control. It ships a WoT DSL,
a closed-loop evolution pipeline, and an MCP tool surface (150+ tools). Sibling of
**ga** (music theory), **ix** (Rust ML), and **Demerzel** (governance). In the
ecosystem, tars is the **cross-model F# theory validator** for music-theory
hypotheses originating in ga.

## Architecture invariant

All active development lives in **`v2/`** (`v2/Tars.sln`, `net10.0`); `v1/` is
legacy C# kept for reference, and CI targets `v2/` only. **F# throughout** —
functional-first, immutable types, `Result<>` for errors. LLM access is **always**
via `LlmFactory.create(logger)`, never `DefaultLlmService` directly.

## Core terms (seed)

- **WoT DSL** — Workflow-of-Thought, the primary agent DSL (`.wot.trsx` files,
  compiled by `Tars.DSL`). **Metascript is frozen** — no new features go there.
- **Cortex** — the agent brain / orchestrator (`Tars.Cortex`) that drives
  multi-agent reasoning over the WoT engine.
- **Evolution / promotion pipeline** — the closed self-improvement loop
  (`Tars.Evolution`): 7 steps, **Inspect → Extract → Classify → Propose → Validate
  → Persist → Govern**. Patterns climb *Implementation → Helper → Builder →
  DslClause → GrammarRule*; the Bayesian-weighted index persists to
  `~/.tars/promotion/index.json`.
- **Probabilistic grammar** — EBNF grammars (`v2/grammars/`) with Bayesian-updated
  weights for constrained decoding; managed via `grammar_weights` / `grammar_evolve`
  / `grammar_search`.
- **GA Trace Bridge** — discovers patterns from ga orchestrator traces
  (`~/.ga/traces/`) and chatbot skill-routing claims, promoting them through the
  evolution pipeline for cross-model validation (`ingest_ga_traces`,
  `ChatbotClaimsBridge.fs`).
- **fsharp_* tools** — the F# tool family on the MCP surface (`fsharp_compile`,
  `fsharp_check_syntax`, `fsharp_eval`, …) used for live F# analysis.

## Conventions

See `CLAUDE.md` for authoritative build/convention rules and `v2/README.md` for the
full project map.
