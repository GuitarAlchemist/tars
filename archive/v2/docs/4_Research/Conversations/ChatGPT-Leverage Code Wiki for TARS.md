# Leverage Code Wiki for TARS

**User:** Stephane Pareilleux (spareilleux@gmail.com)  
**Created:** 11/22/2025 0:10:16  
**Updated:** 11/22/2025 0:11:10  
**Exported:** 11/27/2025 0:57:42  
**Link:** [https://chatgpt.com/c/692145a3-eb30-8326-8f70-1f9320cd3c9a](https://chatgpt.com/c/692145a3-eb30-8326-8f70-1f9320cd3c9a)  

## Prompt:
Teach me how to leverage Code Wiki https://share.google/KiOoDuaibWB5cwhpM for tars

## Response:
Pensé pendant16s

Short version: treat that Code Wiki as TARS’s **brain manual** – both for you and for future TARS-agents that will read it. Let’s make it structured enough that a human _and_ a future RAG pipeline can exploit it.

I can’t see the actual doc from here, so I’ll give you a concrete “TARS-shaped” way to use it.

* * *

1\. Decide what the Code Wiki _is_ for TARS
-------------------------------------------

For TARS, the wiki should be:

1.  **The Constitution + Map**
    *   Core principles, invariants, safety rules.
    *   High-level map of modules and their responsibilities.
2.  **The “Why” Layer above the code**
    *   Code shows _how_; wiki captures _why this shape_.
    *   Every weird design choice gets a 3–5 line explanation somewhere.
3.  **Future RAG Source**
    *   Written as short, tight sections that can be embedded and queried later by TARS itself.
    *   That means: clean headings, one concept per section, minimal rambling.

Think of it as: `README.md × ADRs × prompt library` but centralized.

* * *

2\. A TARS-specific structure for the wiki
------------------------------------------

Here’s a structure I’d use directly in that Code Wiki, with headings you can paste.

### 2.1. TARS Overview

*   **Goal & Non-Goals**
    *   5–10 bullets: what TARS _will_ do and what it explicitly won’t (yet).
*   **Core Concepts**
    *   TARS as kernel/cortex.
    *   Agents, beliefs, grammars, `.trsx` / `.flux` / `.smem`.
*   **High-Level Architecture Diagram (text + link)**
    *   Even ASCII is fine:
        *   `tarscli` → `Kernel` → `Cortex` → Agents → Vector store / Filesystem.

### 2.2. Constitution & Invariants

This is huge for a self-modifying system:

*   **Immutability rules**
    *   What is never mutated in place (versions, logs, previous `.trsx`).
*   **Versioning rules**
    *   How versions are named (`agentic_module_vNN.trsx`, `belief_graph_vN.trsx`, etc.).
*   **Safety Gates**
    *   Conditions under which TARS:
        *   refuses to apply a change,
        *   rolls back,
        *   requires human confirmation.

Short, crisp bullets. This page is the “law” TARS should obey.

### 2.3. Modules (one section per major component)

For each:

*   **Kernel**
    *   Responsibilities: scheduling, IO, versioning, rollback, metrics.
    *   “It _never_ does”: LLM calls, creative reasoning.
    *   Interfaces:
        *   `RunCycle()`
        *   `LoadState(versionId)`
        *   `ApplyMutation(diff)` (with constraints)
*   **Cortex**
    *   Responsibilities: orchestrating LLM calls, reasoning, choosing agents.
    *   Inputs/outputs:
        *   Inputs: `.trsx`, belief graph, logs, local vector store.
        *   Outputs: new `.trsx`, diffs, commentary.
*   **Agent Runtime**
    *   Agent lifecycle: load config → run → emit trace → score.
    *   What “fitness” means in TARS.
*   **RAG / Memory**
    *   Where embeddings live.
    *   What gets embedded: `.trsx`, wiki exports, logs, code snippets.
    *   Query patterns (e.g., “Given this diff, fetch 5 closest past diffs”).

Each module gets:

*   1-paragraph summary
*   Table of key types and functions
*   “Invariants” subsection

Example snippet for the wiki:

> **Kernel Invariants**
> 
> *   Never mutates an existing version directory.
> *   A run is valid only if `diagnostics.json` and `meta-summary.trsx` exist.
> *   Rollback is always a _copy_ into a new version, never a deletion.
>     

That kind of thing becomes gold later.

### 2.4. DSLs: `.trsx`, `.flux`, `.smem`

Have a dedicated page per DSL:

*   **Purpose** – what this DSL is for.
*   **Minimal example** of a real file (short!).
*   **Grammar sketch** – not full EBNF, just the important constructs:
    *   `version`, `agents`, `belief_graph`, `reflection`, `metrics`, `commentary`, etc.
*   **Compatibility rules**
    *   What must stay backwards compatible.
    *   How you plan schema evolution (e.g., “no field is removed; only deprecated”).

This is where you make the DSLs _legible to future TARS_. Later, you can mirror these pages into `.bnf`/.md files in the repo.

### 2.5. Agent Library

One section per _agent type_ (not per instance):

*   **Name**
*   **Role**
*   **Inputs**
*   **Outputs**
*   **Heuristics / fitness criteria**
*   **Known failure modes**

Example:

> **language\_contraction\_agent**
> 
> *   Role: Reduce grammar entropy while preserving expressive power.
> *   Inputs: current `.flux` grammar, historical `.flux` diffs, belief graph hints.
> *   Outputs: new grammar candidate, compatibility hints, risk level.
> *   Fitness: penalize deletion that breaks >X% of existing `.trsx` examples.
> *   Failure modes: over-contraction (too minimal), inconsistent enum changes.
>     

This becomes your “Pokédex of agents”.

### 2.6. Execution Flows & Scenarios

Describe **end-to-end flows** from the user’s perspective:

*   “Run a normal evolution cycle”
*   “Rollback a bad generation”
*   “Import an external belief graph”
*   “Sync memory from remote TARS”

Each flow:

1.  Trigger (CLI command: `tarscli evolve ...`)
2.  Steps (Kernel → Cortex → Agents → Logs).
3.  Expected artifacts: which files/folders appear.
4.  Failure points and what should happen.

These are both documentation and future _test scenarios_.

### 2.7. Decision Log (Lightweight ADRs)

You do _not_ need heavyweight ADRs; use the wiki with a simple pattern:

For each important decision:

*   **Title**: “Adopt `.trsx` as primary meta format”
*   **Date**
*   **Context**: 3–5 lines.
*   **Decision**: 3–5 lines.
*   **Consequences**: 3–5 lines.

Keep each decision short; one per section with a stable heading (so code & scripts can link to `#decision-adopt-trsx_meta` later).

* * *

3\. Integrate Code Wiki into your daily TARS workflow
-----------------------------------------------------

Here’s how to actually _use_ it instead of letting it fossilize.

### 3.1. “Wiki-first” for big changes

When you’re about to:

*   Add a new agent,
*   Change DSL syntax,
*   Change how rollback works,

do this:

1.  Add/update the relevant section in the wiki:
    *   Short description of the change.
    *   Expected impact.
2.  Only then touch the code / `.trsx`.

This gives you a built-in design review, even when working solo.

### 3.2. “Wiki patch” per PR

For each non-trivial PR:

*   Ask: _What page would future-me or future-TARS need updated to understand this?_
*   Update that page with at least:
    *   a short bullet,
    *   or a link to the code/commit.

Even 2 minutes of doc updates compound massively over time.

* * *

4\. Make the wiki machine-friendly for future TARS
--------------------------------------------------

Since your long-term goal is TARS reading its own docs, write with that in mind.

Some pragmatic rules:

*   **One concept per heading**
    *   No giant blobs. Chunk it.
*   **Avoid “it”, “this thing”, “that module” without names**
    *   Always name the module/agent explicitly: `Kernel`, `language_contraction_agent`, etc.
*   **Stable IDs**
    *   For major concepts, pick a canonical name and reuse it everywhere:
        *   `belief_meta_score`
        *   `language_state_snapshot`
        *   `agent_fitness_score`
*   **Examples over theory**
    *   Each DSL concept: at least one tiny, real snippet.

Later you can:

*   Export wiki → Markdown.
*   Store under `docs/wiki-import/`.
*   Run your local RAG (Ollama + embeddings) on those files.
*   Let TARS query its own documentation.

* * *

5\. Tiny, concrete next steps
-----------------------------

If you want a practical “tonight plan”:

1.  Open Code Wiki.
2.  Create these top-level sections:
    *   `TARS Overview`
    *   `Constitution & Invariants`
    *   `Modules`
    *   `DSLs (.trsx / .flux / .smem)`
    *   `Agent Library`
    *   `Execution Flows`
    *   `Decision Log`
3.  For each section, write **only bullets**, no essays:
    *   One paragraph max per subsection.
4.  Pick **one** real existing thing (e.g. `.trsx` current grammar or an existing agent) and fully document it in the wiki as a “gold standard” example.

From there, any time you touch TARS, you update the wiki in tiny increments. Over time, this becomes both your **operator manual** and TARS’s own **self-knowledge base**.

Later on, we can design the pipeline that exports that Code Wiki into markdown, embeds it, and wires it into `tarscli` as a first RAG source so TARS can literally “read its own Constitution.”



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)