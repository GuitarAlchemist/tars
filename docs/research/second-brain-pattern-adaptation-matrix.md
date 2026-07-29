# Second-Brain Pattern Adaptation Matrix

This matrix maps external Personal Knowledge Management (PKM) patterns to the TARS/IX/Demerzel/Streeling/Seldon ecosystem.

| pattern | cost | core_idea | what_to_adopt | what_to_reject | TARS_mapping | IX_mapping | Demerzel_mapping | Streeling_mapping | risk | first_tracer_bullet |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PARA** | Low | Organize by actionability | Active work belongs in Issues/PRs | Over-categorization of "Resource" folders | Context window prioritization | MCTS task pruning | Governance focus areas | Departmental organization | Stale "Archives" cluttering search | Auto-move closed Issues to `archive.md` |
| **Zettelkasten** | Med | Atomic, linked notes | Atomic memory units with provenance | Manual ID management (use UUID/Content Hash) | `Tars.Evolution` extraction | Skill atomicity | Policy cross-references | Course module granularity | Fragmented context | Extract 3 atomic facts from one GA trace |
| **Obsidian** | Low | Local Markdown + Backlinks | Inspectable, local-first files | Proprietary plugins/binary formats | DSL and `docs/` storage | Trace visualization | Readable policy audit | Departmental wiki pages | Format drift over time | `diag docs` check for broken links |
| **Logseq** | Med-High | Daily outliner / Graph | Daily trace logs as source of truth | Indentation-as-logic (prefer explicit DSL) | GA Trace Bridge | MCTS trial logs | Daily governance reports | Research cycle logs | Large graph traversal cost | Convert one `trace.yaml` to bulleted log |
| **Readwise** | Med | Capture -> Review -> Promote | Mandatory promotion pipeline | Passive "hoarding" of data | `Tars.Evolution` pipeline | Skill refinement | Policy evolution | Knowledge harvesting | "Learning" without validation | Implement "Review Required" flag on new facts |
| **NotebookLM** | Low | Source-grounded synthesis | Mandatory citations for every claim | Global model weights as truth | LLM grounding checks | Verifiable skill claims | Constitutional grounding | Evidence bundles | citation hallucinations | Seldon rejects claim if citation is missing |
| **GraphRAG** | Highest | Hierarchical graph summaries | Global corpus summaries | Pure vector search for global queries | `diag reasoning` overview | Skill-tree topology | Cross-repo health map | Departmental summaries | Graph construction overhead | Generate summary for one Streeling department |
| **Tana** | Med-High | Agentic workspace / Tags | Memory-to-action (Supertags) | Manual tag management | WoT DSL (`.trsx`) | Executable skills | Policy-as-code | Actionable research | Brittle schema definitions | Map one "Bug" tag to a TARS fix workflow |

## Synthesis Notes

### 1. The "Actionability" Filter
Following PARA, we distinguish between **active projects** (GitHub Issues, current WoT runs) and **durable knowledge** (Streeling departments, docs). If a piece of memory doesn't help an agent make a decision or explain a result, it is moved to "Archives" or "Resources" (lower priority for RAG).

### 2. Provenance is Non-Negotiable
Inspired by Zettelkasten and NotebookLM, every fact in the TARS harness MUST have a parent artifact ID. We reject "unbounded" learning where the LLM integrates knowledge without knowing where it came from.

### 3. Review Over Capture
Readwise taught us that capture is easy, but retention is hard. The `Tars.Evolution` pipeline is our implementation of the "Review/Promote" cycle. No knowledge enters the "Permanent" store without passing through the Extract -> Classify -> Propose -> Validate steps.

### 4. Graph vs. Vector
While vector search is used for local retrieval, GraphRAG principles drive our "Global" understanding. Demerzel uses graph-like summaries to report on the state of the entire ecosystem.

## Cost & Complexity Analysis

### 1. Free/Local Feasibility
Patterns like **PARA**, **Obsidian**, and **NotebookLM-style citation discipline** are low-cost because they rely on process and documentation standards rather than expensive compute. They are highly feasible for local-first development.

### 2. Runner & Runtime Impact
- **Low Impact:** Markdown-based retrieval and PARA-based context pruning.
- **Medium Impact:** Zettelkasten-style atomization and Readwise-style review pipelines (requires additional agent passes).
- **High Impact:** Logseq-style block referencing and GraphRAG. Graph construction and block-level indexing significantly increase runner latency and token usage.

### 3. Storage and Indexing Overhead
- **Text-only (Low):** PARA, Obsidian, Zettelkasten.
- **Structured (Medium):** Tana-style supertags (JSON schema maintenance).
- **Compute-Intensive (High):** GraphRAG requires persistent graph database or complex hierarchical index maintenance, which adds substantial overhead compared to simple vector search.

### 4. Tracer Bullets vs. Future Work
- **Cheap Tracer Bullets:** Implement "citation required" checks (NotebookLM) or "auto-archive" scripts (PARA). These prove the value with minimal infra.
- **Heavier Work:** Building a full GraphRAG index or a Tana-like agentic workspace should remain as tracer-only experiments until the "cheap" process-driven improvements are exhausted and the cost is justified by performance gains.
