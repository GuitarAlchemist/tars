# TARS V1 Corpus Inventory

This document provides a high-level inventory and classification of the TARS V1 corpus to guide the transition to V2 and identify candidates for the IX engine.

## Inventory Sections

### V1 Documentation (Markdown)
path_group: "v1/**/*.md"
category: "documentation"
summary: "High-level design docs, implementation reports, and research summaries from the V1 era. Contains critical architectural decisions and 'lessons learned'."
possible_owner: tars
recommended_next_action: summarize
notes: Includes numerous 'IMPLEMENTATION_COMPLETE' and 'SUMMARY' files that document the evolution of the V1 engine.

### V1 Source Code
path_group: "v1/src/**"
category: "source-code"
summary: "The core F# logic of the V1 engine, including Grammar, FLUX, Cuda, and Metascript services."
possible_owner: tars
recommended_next_action: inspect
notes: **IX Candidate (GuitarAlchemist/ix#189)**: `v1/src/TarsEngine.FSharp.Core/Grammar/`, `v1/src/TarsEngine.FSharp.Core/Reasoning/`, and `v1/src/TarsEngine.FSharp.Core/VectorStore/CUDA/` contain algorithms highly relevant to IX.

### Parked Legacy Assets
path_group: "v1/parked_legacy/**"
category: "archive"
summary: "Older C# projects, experimental UI components (Elmish), and early VS Code extension prototypes."
possible_owner: archive
recommended_next_action: archive
notes: Includes `tars-analytics-dashboard` and various UI helpers that are not part of the core V2 path.

### V1 Output & Artifacts
path_group: "v1/output/**"
category: "artifacts"
summary: "Generated projects, 3D app prototypes, autonomous QA reports, and improved code snippets."
possible_owner: tars
recommended_next_action: defer
notes: Useful for verifying V2's generative capabilities against V1 benchmarks.

### V1 TODOs & Roadmap
path_group: "v1/TODOs/**"
category: "governance"
summary: "Unfinished tasks, implementation plans, and granular roadmaps from the final stages of V1."
possible_owner: demerzel
recommended_next_action: summarize
notes: Helps identify 'missed opportunities' and feature gaps to be addressed in V2.

### TARS Internal Metadata
path_group: ".tars/**"
category: "metadata"
summary: "Internal agent organization, metascripts, grammars, and session traces. The 'soul' of TARS V1."
possible_owner: tars
recommended_next_action: port
notes: **IX Candidate (GuitarAlchemist/ix#189)**: `.tars/system/metascripts/` and `.tars/system/knowledge/` are prime candidates for porting to the new IX-based registry.

### V2 Research Insights
path_group: "v2/docs/4_Research/V1_Insights/**"
category: "research"
summary: "V2-era analysis of V1 artifacts, reuse strategies, and component reusability reports."
possible_owner: hari
recommended_next_action: inspect
notes: Provides the grounding for why certain V1 components are being ported or archived.

## IX Candidates Summary (GuitarAlchemist/ix#189)

The following areas are flagged for evaluation by the IX engine team:
1. **Grammar Evolution**: `v1/src/TarsEngine.FSharp.Core/Grammar/` - EBNF parsers and evolutionary logic.
2. **Reasoning Engines**: `v1/src/TarsEngine.FSharp.Core/Reasoning/` - Advanced reasoning DSLs.
3. **CUDA Vector Store**: `v1/src/TarsEngine.FSharp.Core/VectorStore/CUDA/` - Low-level GPU acceleration for semantic search.
4. **Metascripts**: `.tars/system/metascripts/` - Declarative workflow definitions.
