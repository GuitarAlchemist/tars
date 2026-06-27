# Second-Brain Pattern Adaptation Matrix

This matrix maps external Personal Knowledge Management (PKM) patterns to the TARS/IX/Demerzel/Streeling/Seldon ecosystem.

| pattern | core_idea | what_to_adopt | what_to_reject | TARS_mapping | IX_mapping | Demerzel_mapping | Streeling_mapping | risk | first_tracer_bullet |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PARA** | Organize by actionability | Active work belongs in Issues/PRs | Over-categorization of "Resource" folders | Context window prioritization | MCTS task pruning | Governance focus areas | Departmental organization | Stale "Archives" cluttering search | Auto-move closed Issues to `archive.md` |
| **Zettelkasten** | Atomic, linked notes | Atomic memory units with provenance | Manual ID management (use UUID/Content Hash) | `Tars.Evolution` extraction | Skill atomicity | Policy cross-references | Course module granularity | Fragmented context | Extract 3 atomic facts from one GA trace |
| **Obsidian** | Local Markdown + Backlinks | Inspectable, local-first files | Proprietary plugins/binary formats | DSL and `docs/` storage | Trace visualization | Readable policy audit | Departmental wiki pages | Format drift over time | `diag docs` check for broken links |
| **Logseq** | Daily outliner / Graph | Daily trace logs as source of truth | Indentation-as-logic (prefer explicit DSL) | GA Trace Bridge | MCTS trial logs | Daily governance reports | Research cycle logs | Large graph traversal cost | Convert one `trace.yaml` to bulleted log |
| **Readwise** | Capture -> Review -> Promote | Mandatory promotion pipeline | Passive "hoarding" of data | `Tars.Evolution` pipeline | Skill refinement | Policy evolution | Knowledge harvesting | "Learning" without validation | Implement "Review Required" flag on new facts |
| **NotebookLM** | Source-grounded synthesis | Mandatory citations for every claim | Global model weights as truth | LLM grounding checks | Verifiable skill claims | Constitutional grounding | Evidence bundles | citation hallucinations | Seldon rejects claim if citation is missing |
| **GraphRAG** | Hierarchical graph summaries | Global corpus summaries | Pure vector search for global queries | `diag reasoning` overview | Skill-tree topology | Cross-repo health map | Departmental summaries | Graph construction overhead | Generate summary for one Streeling department |
| **Tana** | Agentic workspace / Tags | Memory-to-action (Supertags) | Manual tag management | WoT DSL (`.trsx`) | Executable skills | Policy-as-code | Actionable research | Brittle schema definitions | Map one "Bug" tag to a TARS fix workflow |

## Synthesis Notes

### 1. The "Actionability" Filter
Following PARA, we distinguish between **active projects** (GitHub Issues, current WoT runs) and **durable knowledge** (Streeling departments, docs). If a piece of memory doesn't help an agent make a decision or explain a result, it is moved to "Archives" or "Resources" (lower priority for RAG).

### 2. Provenance is Non-Negotiable
Inspired by Zettelkasten and NotebookLM, every fact in the TARS harness MUST have a parent artifact ID. We reject "unbounded" learning where the LLM integrates knowledge without knowing where it came from.

### 3. Review Over Capture
Readwise taught us that capture is easy, but retention is hard. The `Tars.Evolution` pipeline is our implementation of the "Review/Promote" cycle. No knowledge enters the "Permanent" store without passing through the Extract -> Classify -> Propose -> Validate steps.

### 4. Graph vs. Vector
While vector search is used for local retrieval, GraphRAG principles drive our "Global" understanding. Demerzel uses graph-like summaries to report on the state of the entire ecosystem.
