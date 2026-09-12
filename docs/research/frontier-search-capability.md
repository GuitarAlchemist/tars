---
title: "The Search Capability Gap: Anatomy of a 60% Failure Rate and a Path to Closure"
date: 2026-07-21
track: frontier
unit: search-capability-gap
status: verified
---

# The Search Capability Gap: Anatomy of a 60% Failure Rate and a Path to Closure

## Abstract

TARS's meta-cognitive analysis has long reported a "60% failure rate" for the search capability domain. This document traces that number to its source and finds it rests on exactly five search-tagged outcome records — three seeded ReAct failures and two successes, all from a three-day window in March 2026 — admitted through `GapDetection`'s minimal `total >= 2` sample gate. Current insight snapshots report no active gaps at all, so the statistic is stale rather than actively reproduced. The gap is nonetheless structurally real: TARS's only registered semantic search tool (`search_codebase`) is gated on a mutable global index that silently degrades to a prose warning string; retrieval in `CodebaseRAG` is single-shot cosine top-K with no corrective loop; `GapDetection`'s own prescribed remedy names patterns (`glob-search`, `grep-search`) that exist nowhere in v2; and `CapabilityStore.TrackUsageAsync` is a no-op, so search outcomes can never update capability reputation. Drawing on state-of-the-art work in agent-computer interfaces (SWE-agent), hierarchical localization (Agentless), graph-guided localization (LocAgent), and corrective retrieval (CRAG, Self-RAG), we propose three minimal integrations — a lexical search tool module, a CRAG-style corrective search wrapper, and a symbol-graph `locate_entity` tool composed from tools TARS already ships — with a machine-checkable closure criterion that already exists in `RalphBridge`. All findings below survived two-pass adversarial review; one initially attractive claim ("TARS has no lexical search tool at all") was refuted and is recorded honestly.

## Background

TARS records per-episode pattern outcomes in `~/.tars/pattern_outcomes.json`, tags them with domain keywords via `extractDomainTags`, and periodically runs `GapDetection.detectGaps` (invoked from `MetaCognitionOrchestrator`) to compute per-domain failure rates. Domains whose failure rate exceeds a threshold with at least two samples are declared capability gaps; each gap carries a suggested remedy (a `ComposePatterns` list) intended to be executable by the self-improvement loop. Detected gaps flow into insight snapshots under `~/.tars/insights/` and into `RalphBridge`, which generates a loop prompt whose completion condition is "when all gaps have failure rates below 30% and all tests pass."

The project memory records that "meta-cognitive analysis found `search` capability gap (60% failure rate)." This research unit asks three questions: (1) where does the 60% number actually come from, and is it still true; (2) independent of the statistic, is there a genuine structural deficiency in TARS's search capability; and (3) what does the current literature on agentic code search and retrieval suggest as the minimal set of interventions, with what success criteria.

The relevant literature divides into two strands. The agentic code-navigation strand — SWE-agent's agent-computer interface work, Agentless's hierarchical localize-then-repair pipeline, and LocAgent's graph-guided localization — establishes that how an agent finds code dominates end-to-end repair performance, and that structured localization (files → classes/functions → edit locations, or multi-hop traversal of a repository symbol graph) substantially outperforms flat retrieval. The corrective-retrieval strand — CRAG and Self-RAG, surveyed in the Agentic RAG literature — establishes that single-shot retrieval is fragile and that robustness requires explicitly evaluating retrieval quality and triggering corrective actions (query rewriting, strategy escalation, retry) when it fails. CodeRAG-Bench adds the repo-level caveat that retriever quality, not chunking granularity, tends to be the binding constraint.

## Findings

All eight findings below were confirmed by two independent adversarial review passes against the live repository and telemetry files.

### 1. The 60% figure rests on five records, three of them seeded (`gap-origin-n5`)

The "60% search failure rate" is computed from exactly five search-tagged outcome records in `~/.tars/pattern_outcomes.json`, all timestamped between 2026-03-10 and 2026-03-12. The three failures are ReAct episodes — "search for security vulnerabilities in auth module" (2026-03-10T10:00:00Z, 15000 ms), "search codebase for deprecated API usage" (12:00:00Z, 18000 ms), and "search for performance bottlenecks in the API" (2026-03-11T09:00:00Z, 16000 ms) — and the two successes are ChainOfThought episodes for "search for and fix code smells in the codebase." Three of five is 60%. The exact-hour timestamps and round millisecond durations indicate seeded, synthetic records, and the failing goal strings appear nowhere in the repository source (a repo-wide grep finds zero matches).

The gate that admitted this gap is minimal: `GapDetection.fs:129` reads `if rate >= threshold && total >= 2`, so five samples clear the floor easily. The confidence formula at `GapDetection.fs:155`, `min 1.0 (float total / 10.0) * rate`, yields only 0.3 for this gap — the system itself rates the finding as low-confidence.

One reviewer caveat is worth recording: the file as it exists today contains a sixth record matching `extractDomainTags`' "search" keyword — the benchmark goal "inter-binary-search" (a success, 2026-06-21). A fresh recomputation over the whole file therefore gives 3/6 = 50%, not 60%. The headline figure is derivable only from the five March records; the essential claim — that the gap rests on roughly five synthetic samples admitted by a floor of two — stands, and the drift further supports re-measurement.

Code anchors: `C:/Users/spare/.tars/pattern_outcomes.json:212-231`, `C:/Users/spare/.tars/pattern_outcomes.json:261-267`, `C:/Users/spare/.tars/pattern_outcomes.json:142-155`, `v2/src/Tars.Core/MetaCognition/GapDetection.fs:129`, `v2/src/Tars.Core/MetaCognition/GapDetection.fs:155`.

### 2. The gap is stale in current telemetry (`gap-stale-in-telemetry`)

The search gap is not reproduced by any current telemetry. Every insight snapshot in `~/.tars/insights/` — 62 files, including `latest.json` dated 2026-06-24T02:14:47Z — reports `"gaps": []` (line 111 of each file), and the overall outcome summary at `latest.json:227-230` is 217 successes to 14 failures. No snapshot contains a domain entry for "search", "search-code", or "search-files". The 60% figure therefore lives only in the raw `pattern_outcomes.json` history, not in the loop's active gap register: either recency filtering excludes the March failures from the `recentOutcomes` fed to `detectGaps` at `MetaCognitionOrchestrator.fs:43-45`, or gap detection is simply not being fed them. Either way, the correct posture is to treat "search" as a structural (tooling) gap rather than a statistically live one, and to add continuous measurement rather than relying on a one-time inference from seed data.

Code anchors: `C:/Users/spare/.tars/insights/latest.json:111`, `C:/Users/spare/.tars/insights/latest.json:227-230`, `v2/src/Tars.Evolution/MetaCognitionOrchestrator.fs:43-45`.

### 3. The prescribed remedy names patterns that do not exist (`remedy-names-nonexistent-patterns`)

`GapDetection`'s suggested remedy for the search gap — `ComposePatterns ["glob-search"; "grep-search"; "semantic-search"]` at `GapDetection.fs:100`, with a sibling `ComposePatterns ["iterative-search"; "breadth-first-discovery"; "semantic-similarity"]` at line 112 — names patterns and tools that exist nowhere in v2. The identifiers `glob-search`, `grep-search`, and `iterative-search` appear only inside `GapDetection.fs` itself; no pattern definition, tool registration, or WoT clause uses those names (the closest real tools are `search_code` and `search_codebase`, registered under different names). The promotion index at `~/.tars/promotion/index.json` contains exactly nine patterns (five `ga.*` families plus `hypothesis_test_loop`, `decompose_and_solve`, `extract_test`, `verify_then_commit`) and no search pattern at any staircase level — the file contains no occurrence of "search" at all.

Two reviewer refinements: the original evidence's "exactly two lines repo-wide" undercounts slightly (`semantic-search` also appears in v1 legacy docs and a parked v1 demo command name, none of which are v2 patterns or tools), and grep-like functionality does exist under the different name `search_code` — so the remedy is unexecutable by name rather than unimplementable in spirit. The self-improvement loop's prescribed fix, as emitted, resolves to nothing it can compose.

Code anchors: `v2/src/Tars.Core/MetaCognition/GapDetection.fs:98-100`, `v2/src/Tars.Core/MetaCognition/GapDetection.fs:111-112`, `C:/Users/spare/.tars/insights/latest.json:112-226`.

### 4. `search_codebase` silently degrades to a warning string (`search-codebase-silent-degradation`)

The semantic search tool `search_codebase` depends on a mutable module-level global: `SemanticCodeTools.fs:10-11` declares `let mutable private codebaseIndex: CodebaseRAG.CodebaseIndex option = None`, annotated as "initialized by evolve/refactor commands." When the index is unset, lines 39-41 return the prose string "Codebase index not initialized. Use tars evolve or tars ingest-code first." as an ordinary tool result — type-indistinguishable from a successful search, in tension with the repo's `Result<>`-for-errors convention, and opaque to any agentic self-correction or fallback logic. Beneath it, `CodebaseRAG.SearchAsync` returns `[]` silently when not ingested (`CodebaseRAG.fs:317-318`) and swallows all exceptions into a keyword fallback (lines 339-341).

Adversarial review found the claim understated: `setCodebaseIndex` (defined at `SemanticCodeTools.fs:14`) is never called anywhere in v2, so the index is never initialized by any code path — the "evolve/refactor commands" comment is stale. Degradation is therefore the default in every context, not merely in agent runs, MCP serving, and chat. CRAG's core finding (Yan et al., 2024) is that retrieval robustness requires explicitly evaluating retrieval quality and triggering corrective actions on failure — impossible when failure is encoded as prose.

Code anchors: `v2/src/Tars.Tools/SemanticCodeTools.fs:10-19`, `v2/src/Tars.Tools/SemanticCodeTools.fs:39-41`, `v2/src/Tars.Cortex/CodebaseRAG.fs:315-341`.

### 5. Retrieval is single-shot with no corrective loop (`single-shot-retrieval-no-corrective-loop`)

`CodebaseRAG.SearchAsync` (`CodebaseRAG.fs:315-341`) embeds the query once, calls `vectorStore.SearchAsync` once, and maps identifiers to chunks. There is no score thresholding, no reranking, no query rewriting, no decompose-and-retry, and no relevance check on returned chunks; the only adaptivity is a silent fallback to a simple term-overlap keyword scorer on empty embedding or exception. This is precisely the failure mode the corrective-retrieval literature targets: CRAG shows a lightweight retrieval evaluator triggering {Correct, Incorrect, Ambiguous} actions significantly improves RAG robustness across four datasets, and Self-RAG shows adaptive retrieval with self-critique beats always-retrieve baselines. Both citations were verified as real and correctly characterized during review.

Code anchors: `v2/src/Tars.Cortex/CodebaseRAG.fs:315-341`.

### 6. Symbol-graph localization ingredients exist but are never composed (`symbol-graph-assets-unwired`)

TARS already ships both ingredients of state-of-the-art graph-guided code localization: a symbol extractor (`extract_symbols`, registered at `CodeAnalysisTools.fs:245`) and an in-memory knowledge-graph tool family (`graph_add_node` / `graph_add_edge` / `graph_get_neighborhood` / `graph_query`, `GraphTools.fs:22-144`, backed by concurrent dictionaries with JSONL persistence). No code path composes them: repo-wide grep finds `extractSymbols` referenced only at its own definition site, and the sole non-GraphTools caller of `graphAddNode` is `ChatbotClaimsBridge.fs`, which ingests chatbot claims, not code symbols. No repository symbol graph is ever built, leaving the LocAgent/Agentless class of localization — file → class/function → edit location via multi-hop graph traversal — unavailable despite the parts being on the shelf. LocAgent (ACL 2025) demonstrates that parsing a codebase into a directed heterogeneous graph and letting an agent traverse it yields large localization-accuracy gains; Agentless shows even a non-agentic hierarchical localize-then-repair pipeline outperforms many agent frameworks on SWE-bench.

Code anchors: `v2/src/Tars.Tools/CodeAnalysisTools.fs:245`, `v2/src/Tars.Tools/GraphTools.fs:22-144`.

### 7. The reputation loop is a stub, so routing cannot learn (`reputation-loop-stub`)

`CapabilityStore.TrackUsageAsync` is a no-op: its body at `CapabilityStore.fs:132-136` is `task { return () }` under the comment "TODO: Implement metrics tracking (Phase 6.5.4)." Because no reputation is ever written, `FindAgentsAsync` (lines 119-122) permanently blends the default reputation of 0.5 into routing scores via `adjustedScore = score + repScore*0.1 + confScore*0.05` with `repScore = reputation |> Option.defaultValue 0.5`. Capability-based routing therefore structurally cannot learn from the very search failures that define the gap — the credit-assignment write path does not exist. Compounding this, `CapabilityKind` (`Domain.fs:80-88`) offers `WebSearch` but no `CodeSearch`/`RepoSearch` case (only the `Custom of string` escape hatch), so codebase search cannot even be declared as a first-class capability.

Code anchors: `v2/src/Tars.Cortex/CapabilityStore.fs:132-136`, `v2/src/Tars.Cortex/CapabilityStore.fs:119-122`, `v2/src/Tars.Core/Domain.fs:80-88`.

### 8. A machine-checkable closure criterion already exists (`closure-criterion-exists`)

No new success metric needs to be invented. `RalphBridge`'s generated loop prompt formats per-gap lines as `"- **%s**: %.0f%% failure rate (%d samples) — %A"` (`RalphBridge.fs:188`) and declares completion verbatim at line 212: "When all gaps have failure rates below 30% and all tests pass," followed by the `<promise>TARS GAPS RESOLVED</promise>` marker. Combined with `GapDetection.failureRateByDomain` (`GapDetection.fs:50-89`), which computes per-domain (rate, failures, total) from goal-tagged outcomes, the metric "search-domain failure rate < 30% over >= 10 fresh outcomes" is computable today from `pattern_outcomes.json` with no new infrastructure — and closure would be observed by the same mechanism that originally reported the gap.

Code anchors: `v2/src/Tars.Cortex/RalphBridge.fs:188`, `v2/src/Tars.Cortex/RalphBridge.fs:212`, `v2/src/Tars.Core/MetaCognition/GapDetection.fs:50-89`.

### 9. Chunking is a second-order concern (`chunking-not-the-first-bottleneck`)

**Confidence: medium.** `CodebaseRAG`'s chunking configuration — `ChunkSize = 1500` characters (roughly 30 lines) and `ChunkOverlap = 200` at `CodebaseRAG.fs:69-70`, with construct-boundary heuristics (`let`/`type`/`module` prefixes) and a force-split at `ChunkSize / 50` lines in `chunkFile` (`CodebaseRAG.fs:93-160`) — is a plausible weakness, but per CodeRAG-Bench the dominant repo-level failure mode is retriever quality and context integration, not chunk granularity: current retrievers struggle to fetch useful repo-level context, and generators gain limited benefit even from gold documents. Two caveats keep this at medium confidence. First, CodeRAG-Bench did not ablate chunk size specifically, so "retrieval strategy, not chunk size, is the binding constraint" is a reasonable inference rather than a directly tested result. Second, review noted that `ChunkOverlap = 200` is defined but the chunking loop never actually applies overlap — a defect, but one that if anything reinforces the judgment that chunking is imperfect yet second-order. The recommendation is sequencing, not dismissal: revisit chunking only if the localization benchmark still shows semantic-retrieval misses after the three proposals below land.

Code anchors: `v2/src/Tars.Cortex/CodebaseRAG.fs:69-70`, `v2/src/Tars.Cortex/CodebaseRAG.fs:93-160`.

## Flagged

No claims were flagged as unverifiable in this unit. Every claim that survived drafting was either confirmed with code anchors by both review passes or refuted outright (below).

## Refuted during review

One claim was dropped after adversarial review contradicted its headline assertion:

- **"TARS's tool registry contains no lexical code-search tool: no grep, no regex content search, no glob."** Refuted. The registry does contain a lexical content-search tool: `search_code` (`v2/src/Tars.Tools/WorkflowTools.fs:78-141`) performs recursive, case-insensitive substring search over `Directory.GetFiles(path, filePattern, AllDirectories)`, returning file:line:preview results — effectively `grep -F`. `list_files` (`WorkflowTools.fs:9-76`) supports glob-style wildcard patterns with `recursive: true`, and `find_todos` (line 208) is a further recursive content scanner. The narrower observations that partially motivated the claim do hold: no regex-based search exists (`search_code` is `Contains`-only, and its ~30-result cap was found by review to be a no-op bug), `read_file` truncates at 64KB and `list_dir` is non-recursive with a 200-entry cap (`StandardTools.fs:54-95`), and the embedding-based tools (`search_codebase`, `find_similar_code`, `augment_codebase_search`) exist as described. The SWE-agent citation and its numbers (Claude 3 Opus RAG 3.79% vs SWE-agent 12.47% on SWE-bench) were verified as accurate, but the codebase premise failed, so the claim is withdrawn. The practical consequence for Proposal 1 is a reframing: the work is upgrading `search_code` to regex with sane result handling and exposing a true glob tool, not creating lexical search from nothing.

## Opportunities for TARS

Ranked by effort-to-impact, with the sequencing endorsed by finding 9 (lexical → corrective → graph):

1. **Re-measure before fixing, and harden gap detection (days).** Run at least 20 real search-goal episodes through the evolve loop to establish a current baseline failure rate; raise the `detectGaps` sample floor from 2 to ~5 and add age-decay so four-month-old seeded records cannot dominate gap ranking (finding 1). Add a search-goal probe suite to the benchmark harness so the gap is continuously measured rather than inferred once from seed data (finding 2).

2. **Proposal 1 — lexical search tools (days).** Upgrade `search_code` from substring to regex (fixing its no-op result cap), add a first-class `glob_search`, and register both under names the remedy system can reference. Then make `GapDetection`'s remedies executable: either register a composed `multi-strategy-search` pattern eligible for the promotion staircase, or have `CurriculumPlanner` validate remedy pattern names against the live tool registry and pattern store before emitting tasks (finding 3). Simultaneously fix `search_codebase` to signal machine-readable status (e.g., JSON `{status: "index_unavailable"}`) or auto-fall-back to lexical search when the index is `None`, and build the index lazily on first use — today no code path initializes it at all (finding 4).

3. **Proposal 2 — CRAG-style corrective search wrapper (~1 week).** Add `CorrectiveSearch.fs` to `Tars.Cortex` defining `type SearchStrategy = { Name: string; Run: string -> int -> Task<SearchResult list> }` and a `runCorrective` function that escalates semantic → keyword → lexical grep → query-decomposition-retry whenever a cheap relevance check (score threshold, or a one-token LLM relevance grade per CRAG) fails (finding 5). Success metric: recall@5 >= 0.8 on a 30-query labeled localization set drawn from the TARS repo (queries → known file:line answers), against the measured single-shot baseline.

4. **Proposal 3 — repository symbol graph and `locate_entity` (~2 weeks).** Add `SymbolGraph.fs` to `Tars.Cortex`: walk `v2/src`, run the existing symbol extraction per file, populate the existing graph store with contains/references edges, and expose one new tool `locate_entity {name_or_description}` returning ranked file:line candidates via neighborhood traversal (finding 6). Success metric: file-level Acc@5 and function-level Acc@10 on a 50-item localization benchmark built from TARS's own git history (bug-fix commits → changed symbols), following LocAgent's evaluation protocol.

5. **Close the credit-assignment loop (parallel, ~1 week).** Implement `TrackUsageAsync` as a Beta-Binomial update — the same machinery `PromotionPipeline` already uses for pattern weights — persisted alongside the capability store, and add `CapabilityKind.CodeSearch` so the new search tools are routable and their reputation observable (finding 7). Without this, any fix's outcomes remain invisible to routing and the 60% number stays frozen by construction.

6. **Define the gap-closed gate (free).** Adopt the conjunction: (a) search-domain failure rate < 30% on >= 10 fresh search-goal episodes — RalphBridge's native criterion; (b) recall@5 >= 0.8 on the labeled localization set (Proposal 2); (c) file-level Acc@5 >= 0.7 on the git-history benchmark (Proposal 3). Wire (a) into the evolve loop's existing gap re-detection so closure is observed by the same mechanism that reported the gap (finding 8).

## References

1. Yan, S.-Q., Gu, J.-C., Zhu, Y., Ling, Z.-H. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884.
2. Asai, A., Wu, Z., Wang, Y., Sil, A., Hajishirzi, H. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* arXiv:2310.11511.
3. Singh, A., Ehtesham, A., Kumar, S., Talaei Khoei, T. (2025). *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG.* arXiv:2501.09136.
4. Chen, Z., Tang, X., Deng, G., Wu, F., Wu, J., Jiang, Z., Prasanna, V., Cohan, A., Wang, X. (2025). *LocAgent: Graph-Guided LLM Agents for Code Localization.* arXiv:2503.09089. ACL 2025.
5. Xia, C. S., Deng, Y., Dunn, S., Zhang, L. (2024). *Agentless: Demystifying LLM-based Software Engineering Agents.* arXiv:2407.01489.
6. Wang, Z. Z., Asai, A., Yu, X. V., Xu, F. F., Xie, Y., Neubig, G., Fried, D. (2024). *CodeRAG-Bench: Can Retrieval Augment Code Generation?* arXiv:2406.14497.
7. Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., Press, O. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* arXiv:2405.15793. (Cited in the refuted-claim record; the citation and its SWE-bench numbers were verified accurate even though the associated codebase premise was refuted.)
