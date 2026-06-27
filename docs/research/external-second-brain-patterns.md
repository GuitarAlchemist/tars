# External Second-Brain Patterns Benchmark

This document benchmarks external second-brain and PKM (Personal Knowledge Management) patterns against the TARS second-brain harness model. The goal is to extract reusable design principles and decide how they map to our ecosystem components: TARS (F# agent system & theory validator), IX (Rust MCTS/skill engine), Demerzel (governance), Streeling, and GitHub-driven agent workflows.

*We are not cloning a note-taking app. We optimize for actionable, safe memory over beautiful graphs.*

## 1. PARA / Building a Second Brain
**Core Idea:** Organizing information by actionability (Projects, Areas, Resources, Archives) rather than by subject.
**Candidate Synthesis:** Active work belongs in GitHub issues, durable knowledge stays in docs.
**Ecosystem Mapping:**
- **TARS:** Reads durable knowledge (e.g., `CONTEXT.md`) for theory validation and context.
- **GitHub Workflows:** Issues are the "Projects" (active work). PRs close issues and move findings to "Resources" (docs) or "Archives".

## 2. Zettelkasten
**Core Idea:** Atomic notes connected via links, creating a network of thought rather than a hierarchical tree.
**Candidate Synthesis:** Memory should be atomic, linked, and provenance-rich.
**Ecosystem Mapping:**
- **TARS:** Memory snippets in the WoT (Workflow-of-Thought) and evolution pipeline should carry rich provenance back to their origin (e.g., GA traces).
- **IX:** Skill schemas and learned routines should be linked atomic units.

## 3. Obsidian-style Local Markdown + Backlinks
**Core Idea:** Using plain, local-first text files (Markdown) linked bidirectionally to ensure longevity and data ownership.
**Candidate Synthesis:** Local-first Markdown/JSON files should remain inspectable.
**Ecosystem Mapping:**
- **TARS:** Patterns and grammars (`v2/grammars/`) are stored locally as text. Evolution index (`~/.tars/promotion/index.json`) is an inspectable JSON.

## 4. Logseq-style Outliner / Graph Workflows
**Core Idea:** Block-level referencing and outliner formats that treat every bullet point as an addressable node in a graph.
**Candidate Synthesis:** Local-first Markdown/JSON files should remain inspectable. (Similar to Obsidian, but focusing on block-level atomicity).
**Ecosystem Mapping:**
- **TARS & IX:** When extracting constraints or theory rules, addressability should ideally be at the block/rule level, allowing specific granular rules to be composed or modified without pulling in entire documents.

## 5. Readwise-style Capture / Highlight / Review / Export
**Core Idea:** Automated ingestion of highlights, coupled with spaced repetition (review) to surface past knowledge.
**Candidate Synthesis:** Capture is not enough; review/promotion is mandatory.
**Ecosystem Mapping:**
- **TARS Evolution Pipeline:** Mere capture of traces is insufficient. The closed-loop evolution pipeline (Inspect → Extract → Classify → Propose → Validate → Persist → Govern) is our rigorous "review" equivalent, ensuring patterns climb the ladder (Implementation → Helper → Builder → DslClause → GrammarRule) only when validated.

## 6. NotebookLM-style Source-Grounded Synthesis
**Core Idea:** LLMs generating answers strictly grounded in user-provided source documents (evidence bundles) with inline citations.
**Candidate Synthesis:** Answers must be source-grounded and cite evidence bundles.
**Ecosystem Mapping:**
- **TARS:** Agent reasoning via Cortex must be constrained by the probabilistic grammar and grounded in `CONTEXT.md` / `docs/adr/`.
- **Demerzel:** Governance decisions and QA verdicts must cite the specific rules or PR diffs they are evaluating.

## 7. GraphRAG / Knowledge-Graph RAG
**Core Idea:** Augmenting LLMs with structured knowledge graphs to answer global, corpus-wide questions through community summaries.
**Candidate Synthesis:** Global corpus questions need graph/community summaries, not only vector search.
**Ecosystem Mapping:**
- **TARS / GA:** When cross-validating music theory rules, searching across the entire GA trace history requires aggregated graph knowledge to identify systemic contradictions, beyond what simple vector similarity provides.

## 8. Tana-style Agentic Workspace / Memory-to-Action Workflows
**Core Idea:** Supertags and structured fields that turn text nodes into executable database records, driving automated workflows.
**Candidate Synthesis:** Memory becomes valuable when it drives safe action and delegation.
**Ecosystem Mapping:**
- **Streeling / IX:** Using parsed memory structures to immediately trigger specific actions, such as MCTS rollout scenarios.
- **Demerzel:** Memory (like the `afk-halt.json` marker) directly drives safe action (halting autonomous loops).
