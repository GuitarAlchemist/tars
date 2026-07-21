---
title: "TARS as a Neuro-Symbolic System: An Architectural Audit Against the Kautz Taxonomy"
date: 2026-07-21
track: theory
unit: neuro-symbolic
status: verified
---

# TARS as a Neuro-Symbolic System: An Architectural Audit Against the Kautz Taxonomy

## Abstract

TARS presents itself as a neuro-symbolic self-improving agent system. This document audits that claim against the codebase as it actually executes, with every finding confirmed by adversarial review against source anchors. The operational picture is a Kautz Type-3 (Neuro | Symbolic) pipeline: a large language model emits symbolic artifacts — WoT DSL text and JSON — which hand-written symbolic machinery (`WotParser`, `GrammarGovernor`, `RoundtripValidation`, `dotnet fsi` compilation) post-validates and gates after generation. The infrastructure for true decode-time symbolic constraint exists in full — EBNF/regex/JSON-schema request formats, vLLM `guided_decoding` extra-body construction via xgrammar and outlines — but is dormant: no production code path issues an EBNF-constrained request, the backend router never steers constrained requests to a grammar-capable backend, and the default Ollama backend silently degrades EBNF constraints to plain JSON mode. The flagship `NeuroSymbolicIntegration` module is dead code whose constraint scoring is vacuous: stub invariants accept essentially any mutation. Symbolic feedback reaches the neural layer as prompt text and agent-selection weights, never as decode-time token constraints, and no proof-carrying certificates accompany promoted patterns. TARS's genuine learning loop — Beta-Binomial grammar weights refined by replicator dynamics — lives entirely in the symbolic layer and steers pattern selection with a capped 0.08 boost, never token distributions. Two initially attractive claims were refuted during review and are recorded here for honesty: contrary to an earlier draft, TARS does contain deliberate Type-4 scaffolding (`SelfTrain.fs` exports verifier-approved solutions as an SFT dataset), so the neural layer is not entirely cut off from symbolic outcomes at the weight level — only the gradient step itself is external.

## Background

Kautz's taxonomy of neuro-symbolic architectures distinguishes, among others, Type 3 systems — a neural front-end whose output is consumed by symbolic machinery ("Neuro | Symbolic") — from Type 5 systems, where symbolic constraints operate inside neural inference, and Type 4 systems, where symbolic knowledge is compiled into neural weights via training. The distinction matters practically. A Type-3 system must catch malformed or invalid neural output after the fact, spending retry budget on syntactic failures; a Type-5 system makes syntactically invalid output unrepresentable at decode time.

The decode-time approach is mature. Willard and Louf's Outlines work (arXiv:2307.09702) showed that regular-language and context-free constraints can be compiled to finite-state machines that mask the token vocabulary at each decoding step with negligible overhead. XGrammar (Dong et al., arXiv:2411.15100) extended this to a production-grade structured-generation engine, and vLLM exposes both backends through its `guided_decoding` request extension. Geng et al. (EMNLP 2023, arXiv:2305.13971) demonstrated that grammar-constrained decoding substantially outperforms unconstrained generation on structured NLP tasks without any fine-tuning — but also that enforcement requires backend support: a grammar the serving layer ignores is a no-op. Park et al.'s Grammar-Aligned Decoding (NeurIPS 2024) added the caveat that naive mask-based constraint distorts the language model's distribution, and that principled grammar-conditioned sampling requires production weighting — precisely the kind of learned weights a probabilistic grammar could provide.

The complementary post-hoc tradition is equally well studied. CRITIC (Gou et al., ICLR 2024, arXiv:2305.11738) formalized the verify-then-correct loop in which tool-generated critiques are fed back to the model as natural language. Kambhampati et al.'s LLM-Modulo position paper (ICML 2024, arXiv:2402.01817) argues that because LLMs cannot soundly self-verify, external sound verifiers plus critique loops are architecturally necessary. Logic-LM (Pan et al., Findings of EMNLP 2023, arXiv:2305.12295) and DeepProbLog (Manhaeve et al., NeurIPS 2018, arXiv:1805.10872) illustrate deeper couplings, where LLM output is parsed into a formal language and executed by a symbolic solver. Necula's proof-carrying code (POPL 1997) supplies the gold standard for artifact promotion: the artifact ships with a machine-checkable certificate that a small trusted verifier validates.

This audit asks where TARS actually sits in that landscape — not where its module names, file headers, and roadmap documents claim it sits. Every finding below was verified twice by independent adversarial review against the cited code anchors; two claims that failed review are documented in the "Refuted during review" section.

## Findings

### 1. Grammar-constrained decoding is fully plumbed but dormant

**Claim.** The EBNF/grammar-constrained decoding path is complete at the plumbing level — request types, vLLM `guided_decoding` extra-body construction with xgrammar and outlines backends — but no production code constructs an EBNF-, regex-, or cortex-grammar-constrained request. The only non-test consumer of `ConstrainedDecoding` is `AiFunction.withJsonSchema`, and `AiFunction` itself has no production callers outside `Tars.Llm` and tests.

**Evidence.** `OpenAiCompatibleClient.fs:122-128` builds the `{| guided_decoding = {| backend = "xgrammar"; grammar |} |}` extra body for `Constrained(Ebnf)`, and `ChatClientAdapter.fs:64-80` maps `Ebnf`/`Regex` to `guided_decoding_*` additional properties — this is exactly the request shape vLLM's structured-generation backends expect. Yet a repo-wide search for `ConstrainedDecoding` consumers finds only tests (`ConstrainedDecodingTests.fs`, `ProbabilisticGrammarIntegrationTests.fs`) plus `AiFunction.fs:76`, which uses the JSON-schema helper only; `AiFunction`/`AiFunctionIx` callers exist only inside `Tars.Llm` and tests. Every live evolution-engine LLM call uses `JsonMode=true` / `ResponseFormat.Json` (`Engine.fs:217`, `Engine.fs:457`, `Evaluation.fs:112`, `Reflection.fs:76`, `Optimizer.fs:65`, `SymbolicReflector.fs:83`), which constrains generation only to "any valid JSON" — not to a schema, and not to a grammar. Review confirmed the single production construction of a `Constrained` request anywhere is `TestGrammarCommand.fs:44`, a diagnostic CLI command using `Grammar.JsonSchema`, which does not weaken the claim as stated.

**Code anchors.** `v2/src/Tars.Llm/ConstrainedDecoding.fs:54-90`, `v2/src/Tars.Llm/OpenAiCompatibleClient.fs:122-128`, `v2/src/Tars.Llm/ChatClientAdapter.fs:64-80`, `v2/src/Tars.Llm/AiFunction.fs:75-77`, `v2/src/Tars.Evolution/Engine.fs:215-217`.

**Sources.** Willard & Louf 2023 (arXiv:2307.09702); Dong et al. 2024 (arXiv:2411.15100). Confidence: high (confirmed by both reviewers).

### 2. The router would drop the constraint even if one were issued

**Claim.** Under default configuration, an EBNF-constrained request would not be honored even if some code issued one: `chooseBackend` never inspects `ResponseFormat`, local traffic defaults to Ollama (vLLM is selected only if `VllmBaseUri` differs from `localhost:8000`), and `OllamaClient` silently degrades `Constrained(Ebnf|Regex)` to plain `"json"` format, discarding the grammar without any warning or log entry.

**Evidence.** `Routing.fs:133-228`: `chooseBackend` pattern-matches only on `req.Model` and `req.ModelHint`; `req.ResponseFormat` is never read. The `localRoute` branch (`Routing.fs:184-201`) picks Ollama unless the vLLM URI's host or port differ from the localhost default — and review noted the claim is actually understated, since a `PreferredProvider="Ollama"` setting short-circuits to Ollama even earlier. `OllamaClient.fs:186` and `OllamaClient.fs:355` both read `| Some(ResponseFormat.Constrained _) -> Some(box "json")`: the grammar payload is replaced wholesale by generic JSON mode. `Backends.fs:72-73` shows the vLLM backend — which resolves to `openAiCompatible` and would honor the grammar — is reachable but never preferentially selected for constrained requests. Together with Finding 1, this means the constraint pathway is broken at two independent layers: no producer, and a lossy default consumer.

**Code anchors.** `v2/src/Tars.Llm/Routing.fs:133-228`, `v2/src/Tars.Llm/OllamaClient.fs:184-188`, `v2/src/Tars.Llm/OllamaClient.fs:353-357`, `v2/src/Tars.Llm/Backends.fs:72-73`.

**Sources.** Geng et al., EMNLP 2023 (arXiv:2305.13971). Confidence: high (confirmed by both reviewers).

### 3. The NeuroSymbolicIntegration module is dead code

**Claim.** The module nominally embodying TARS's neuro-symbolic loop — `NeuroSymbolicIntegration` — is instantiated nowhere. `NeuroSymbolicEvolution` and `EvolutionPerformanceTracker` are constructed by no CLI command, no engine path, and no test; they appear outside the defining file only in a documentation report's prose code example.

**Evidence.** Repo-wide search for `NeuroSymbolicEvolution`, `EvolutionPerformanceTracker`, and the module name matches only the defining file (`NeuroSymbolicIntegration.fs:3,126`) and `v2/docs/3_Roadmap/3_Reports/neuro_symbolic_implementation_summary.md:279`, a code example in documentation. Review went further than the original claim: the module's free functions (`scoreMutation`, `shouldAcceptMutation`, `filterMutationsByScore`) also have zero external callers, making the module "even deader than claimed." A separate `NeuralSymbolicFeedback` module in `Tars.Symbolic` is live but is a different component; its existence does not rescue this one.

**Code anchors.** `v2/src/Tars.Evolution/NeuroSymbolicIntegration.fs:3`, `v2/src/Tars.Evolution/NeuroSymbolicIntegration.fs:126-184`.

Confidence: high (confirmed by both reviewers).

### 4. The mutation-filtering constraint scoring is vacuous

**Claim.** Even were the dead module wired in, its symbolic gate would pass essentially everything. `scoreMutation` supplies an empty context, under which the `CodeComplexityBound` invariant returns 1.0 ("assume OK if not measured") and `GrammarValidity` is a character heuristic — a string containing `|`, `*`, or `+` scores 1.0, any string longer than three characters scores 0.8 — so every non-trivial mutation averages at least 0.9, far above the 0.5 acceptance threshold. `shouldAcceptMutation` cannot reject realistic code.

**Evidence.** `NeuroSymbolicIntegration.fs:64-74` builds `context = Map.empty` and averages the `complexityLimit` and `parseableGrammar` invariants. `ConstraintScoring.fs:134-137` returns 1.0 for `CodeComplexityBound` when no `"complexity"` key is present. `ConstraintScoring.fs:50-60` — marked `TODO: Integrate with actual grammar parser` — implements the character heuristic. `defaultConfig.MinMutationScore = 0.5` (`NeuroSymbolicIntegration.fs:117-123`). Reviewers verified the arithmetic: any realistic F# code (which invariably contains `|`, `*`, or `+`) scores an average of essentially 1.0 against a 0.5 bar. The pattern generalizes: `Invariants.fs:165-181` evaluates `AlignmentThreshold` with `actual = min`, complexity with `actual = 0.0`, and resources with `actual = 0` — all trivially satisfied — and belief consistency (`Invariants.fs:82-89`) hard-codes `satisfied = true, score = 1.0`.

**Code anchors.** `v2/src/Tars.Evolution/NeuroSymbolicIntegration.fs:64-79`, `v2/src/Tars.Symbolic/ConstraintScoring.fs:50-60`, `v2/src/Tars.Symbolic/ConstraintScoring.fs:134-142`, `v2/src/Tars.Symbolic/Invariants.fs:165-181`.

Confidence: high (confirmed by both reviewers).

### 5. The live architecture post-validates; it does not constrain

**Claim.** In the executing system, the symbolic layer verifies neural output after generation rather than constraining generation. The implemented verification loops are (a) `AiFunction`'s deserialize → post-condition → textual-feedback → retry cycle, and (b) benchmark solutions compiled by `dotnet fsi` with deterministic validation. Both are instances of the CRITIC / LLM-Modulo external-verifier pattern with natural-language feedback, and neither involves any decode-time enforcement.

**Evidence.** `AiFunction.fs:60-77` builds the request in JSON mode, and the retry loop (`AiFunction.fs:91-109` and onward) injects `"Your previous output failed validation: {fb}"` as a user message on failure (`AiFunction.fs:64-65`) — critique delivered as prose, exactly the CRITIC pattern. WoT text is parsed only after generation at the `WotParser.parseFile` call sites; `GrammarGovernor` and `RoundtripValidation` run at promotion time, on artifacts that already exist (`GrammarGovernor.fs:27-52`; invoked from `PromotionPipeline.fs:340-349`). Both reviewers noted one nuance that sharpens rather than weakens the finding: per Finding 1, `AiFunction`'s loop has no production callers, so of the two loops only the `dotnet fsi` compilation loop (BenchmarkRunner, deterministic, "no LLM-as-judge") is live in production. The architectural characterization — post-hoc external verification with natural-language feedback, zero decode-time enforcement — stands as verified.

**Code anchors.** `v2/src/Tars.Llm/AiFunction.fs:60-77`, `v2/src/Tars.Llm/AiFunction.fs:91-109`, `v2/src/Tars.Evolution/GrammarGovernor.fs:27-52`.

**Sources.** Gou et al., ICLR 2024 (arXiv:2305.11738); Kambhampati et al., ICML 2024 (arXiv:2402.01817). Confidence: high (confirmed by both reviewers).

### 6. The probabilistic grammar steers selection, not decoding

**Claim.** TARS's Beta-Binomial `WeightedGrammar` and replicator-dynamics layer — its genuine, functioning learning loop — influences only which pattern or exemplar the symbolic planner selects, through a boost explicitly capped at 0.08, and never touches token-level generation. This contradicts the layered architecture documented in `ConstrainedDecoding.fs` ("Layer 2: Probabilistic policy → WeightedGrammar steers preferences") read as a description of the generation pipeline.

**Evidence.** `PatternSelector.fs:326-350`: the promotion-index boost is gated on a context signal and capped at 0.08, with the in-code comment "a tiebreaker not an override." Weights are updated from governance outcomes (`PromotionPipeline.fs:358`) and evolved by replicator dynamics (`ReplicatorDynamics.fs:90,188`), but no code anywhere converts `WeightedRule` weights into logit biases, guided-decoding grammar probabilities, or sampling parameters — reviewers confirmed zero `logit_bias` matches across `v2/src` and zero `WeightedGrammar` references in `Tars.Llm`. Consumers of the weights on the generation side are only `PatternSelector`, `PromotionIndex.fs:52`, and MCP reporting tools (`McpGrammarTools.fs:40-114`); modules such as `GrammarDistillation` and `GrammarMlBridge` produce weights rather than consume them for decoding. The relevance is more than architectural tidiness: Park et al. show that principled grammar-conditioned sampling requires exactly the production weighting that TARS's PCFG posteriors could supply — the machinery exists on both sides and no bridge connects them.

**Code anchors.** `v2/src/Tars.Cortex/PatternSelector.fs:326-350`, `v2/src/Tars.Evolution/PromotionPipeline.fs:355-375`, `v2/src/Tars.Llm/ConstrainedDecoding.fs:10-14`, `v2/src/Tars.Evolution/ReplicatorDynamics.fs:90`.

**Sources.** Park et al., NeurIPS 2024 (Grammar-Aligned Decoding); Willard & Louf 2023 (arXiv:2307.09702). Confidence: high (confirmed by both reviewers).

### 7. Promotion carries no machine-checkable certificate

**Claim.** Promoted patterns are gated by heuristics, not proofs. `GrammarGovernor`'s eight criteria are pre-computed booleans counted by `PromotionCriteria.score`, and `RoundtripValidation`'s "semantic match" is Jaccard similarity over identifier tokens with a 0.5 pass threshold, applied after a regex-based re-abstraction the code itself describes as "a heuristic stand-in for a proper re-abstraction pass." A promotion can therefore pass while changing program meaning: `if a > b then x else y` and `if b > a then y else x` have identical identifier sets, hence Jaccard similarity 1.0.

**Evidence.** `RoundtripValidation.fs:43-55` (`extractIdentifiers` + `jaccardSimilarity`), `RoundtripValidation.fs:69-74` (regex `reabstract`), `RoundtripValidation.fs:80` (`defaultThreshold = 0.5`), `GrammarGovernor.fs:15-16` (boolean count). Review added two refinements. First, `extractIdentifiers` keeps only tokens longer than two characters, so short identifiers are ignored entirely — the check is weaker than claimed. Second, the live pipeline (`PromotionPipeline.fs:341`) actually calls `quickValidate`, an identifier-coverage variant rather than Jaccard proper; it is equally a lexical token-set heuristic with the same 0.5 threshold and the same blindness to meaning. A `validateWithLlm` variant exists (`RoundtripValidation.fs:188`) but is not called by the pipeline and would in any case be an LLM judgment, not a certificate. This is precisely the gap proof-carrying-code architectures close: the artifact ships with a checkable proof validated by a small trusted verifier.

**Code anchors.** `v2/src/Tars.Evolution/RoundtripValidation.fs:43-55`, `v2/src/Tars.Evolution/RoundtripValidation.fs:66-80`, `v2/src/Tars.Evolution/GrammarGovernor.fs:15-16`.

**Sources.** Necula, POPL 1997 (DOI 10.1145/263699.263712); Kambhampati et al., ICML 2024 (arXiv:2402.01817). Confidence: high (confirmed by both reviewers).

### 8. The WoT DSL exists in three disconnected symbolic forms

**Claim.** The WoT language is defined three times with no conformance link: a hand-written regex/line-based parser (`WotParser`), an EBNF file (`grammars/wot.ebnf`) intended for constrained decoding, and the compiler/pretty-printer. Nothing verifies the parser and the EBNF accept the same language, and the EBNF is never used to constrain WoT generation in production.

**Evidence.** `WotParser.fs:38-58` parses key=value lines with `Regex.Match` and a self-described "very forgiving" bracket-list parser — it is not generated from, nor checked against, `wot.ebnf`. Review found the disconnection is total: repo-wide search for `wot.ebnf` matches only a documentation file (`dsl_unification_proposal.md`); no code or test loads it by name, despite the file's own header stating it is "Used with vLLM guided_decoding to ensure valid WoT programs." The compiler (`WotCompiler`, invoked at `WotCommand.fs:299`) is a third independent implementation. No property or conformance test links any pair of the three. This is the grammar-drift failure mode that GCD systems avoid by deriving both enforcement and parsing from a single formal grammar.

**Code anchors.** `v2/src/Tars.DSL/Wot/WotParser.fs:38-58`, `v2/grammars/wot.ebnf:1`, `v2/src/Tars.Llm/ConstrainedDecoding.fs:25-32`.

**Sources.** Geng et al., EMNLP 2023 (arXiv:2305.13971). Confidence: medium at submission, confirmed on review (both reviewers noted the finding is if anything understated — the original claim's statement that the EBNF is "loaded via generic grammar-listing paths and tests" was itself too generous; nothing loads it specifically).

### 9. Meta-cognition is statistical aggregation, not symbolic reasoning

**Claim.** `MetaCognitionOrchestrator`'s five-step cycle is neural-heavy reflection over statistics rather than symbolic reasoning: capability gaps are failure-rate thresholds, reflections are LLM calls, and `NewBeliefs` is an integer count (`gaps.Length + reflections.Length`) rather than assertions in a knowledge base. TARS's only non-LLM contradiction detection anywhere is substring matching — hard-coded antonym pairs such as `' is '`/`' is not '`, and `output.Contains("NOT") && Contains("contradiction")`. There is no entailment engine, SAT/SMT solver, or logic-programming runtime in the repository.

**Evidence.** `MetaCognitionOrchestrator.fs:36-45` (gap detection as a threshold over tag failure rates), `MetaCognitionOrchestrator.fs:104` (the integer `NewBeliefs`), `MetaCognitionOrchestrator.fs:80-97` (recommendations as sprintf-formatted strings). `ConstraintScoring.fs:16-38` is the antonym-pair substring test; `NeuroSymbolicIntegration.fs:170` is the `Contains`-based heuristic. Reviewers confirmed zero matches for Z3, SMT, Datalog, Prolog, SAT, entailment, or satisfiability across `v2/src`. The `beliefUpdateSchema` IR (`ConstrainedDecoding.fs:122-135`) defines assert/retract/revise operations, but no reasoner consumes them; its only consumers are schema-validity tests. One reviewer nuance: `Engine.fs:190-249` also performs contradiction detection via an LLM yes/no call, so "only contradiction detection anywhere is substring matching" is slightly overstated with respect to neural mechanisms — but no symbolic, solver-backed detection exists, which is the substantive point. The contrast with Logic-LM — which parses LLM output into a formal language and runs an actual symbolic solver, with large accuracy gains — locates the missing half of the loop.

**Code anchors.** `v2/src/Tars.Evolution/MetaCognitionOrchestrator.fs:36-45`, `v2/src/Tars.Evolution/MetaCognitionOrchestrator.fs:104`, `v2/src/Tars.Symbolic/ConstraintScoring.fs:16-38`, `v2/src/Tars.Llm/ConstrainedDecoding.fs:122-135`.

**Sources.** Pan et al., Findings of EMNLP 2023 (arXiv:2305.12295); Manhaeve et al., NeurIPS 2018 (arXiv:1805.10872). Confidence: high (confirmed by both reviewers).

## Flagged

No claims in this unit were flagged as unverifiable. Every claim submitted to adversarial review was either confirmed (Findings 1-9 above, each confirmed by two independent reviewers) or refuted (below).

## Refuted during review

Two claims were dropped after review found them contradicted by code the drafts had missed. Both failed on the same blind spot, and the correction materially changes the taxonomy placement, so they are recorded in some detail.

**Refuted claim 1: "Kautz Type 4 is entirely absent."** A draft claim asserted that TARS is Type 3, not Type 5 in practice, and that Type 4 (symbolic knowledge compiled into neural weights) is "entirely absent," with the supporting statement that no training or fine-tuning code exists anywhere in `v2/src`. The Type-3 characterization and the "not Type 5 in practice" conjunct verified fully and are absorbed into Findings 1, 2, and 5. But the Type-4 conjunct is false: `v2/src/Tars.Evolution/SelfTrain.fs` explicitly builds a supervised fine-tuning dataset from compiler-verified solutions plus `~/.tars/self_host_wins.jsonl`, emits an Ollama Modelfile for a fine-tuned GGUF, and is wired into the CLI (`tars self-train`, `Program.fs:269`) and the evolve loop (`Evolve.fs:797`). Only the GPU weight-update step (unsloth/llama.cpp runbook) is external to the repo. The defensible narrowing — Type-4 weight updates are not performed in-repo, so the Type-4 loop is scaffolded but not operationally closed — is what this document asserts.

**Refuted claim 2: "The neural component never learns from the symbolic layer."** A companion draft claim held that symbolic feedback reaches the neural layer exclusively as prompt text (`shapePrompt`'s "SYMBOLIC WARNINGS") and stochastic agent selection, with no pathway from verifier outcomes into model weights — no fine-tuning, LoRA, logit-bias, or STaR-style bootstrapping. Its anchors verified (`NeuralSymbolicFeedback.fs:84-106`, `:30-33`; `PromotionPipeline.fs:353-375`), but the central absence assertion is contradicted by the same `SelfTrain.fs`: it is a STaR-style bootstrap on verified outputs — collecting compiler-verified benchmark solutions (property-test failures excluded), merging verified self-hosting wins (`SelfTrain.fs:105-121`, directly contradicting the sub-claim that `self_host_wins.jsonl` is consumed only by the symbolic promotion pipeline), and emitting a chat-format SFT JSONL plus Modelfile (`SelfTrain.fs:125-141`). The claim's proposed "opportunity" was already implemented in-tree. What remains true, and is retained in Findings 5 and 6, is that within a running session symbolic feedback reaches the model only as prompt text and selection weights — logit-bias and decode-time pathways genuinely do not exist.

The honest synthesis: TARS is operationally Type 3, with a dormant Type-5 substrate (Findings 1-2) and a deliberately scaffolded but externally-completed Type-4 pathway (`SelfTrain.fs`).

## Opportunities for TARS

Ranked by leverage relative to diff size. The first three exploit the audit's central irony — every finding is an already-built component missing only its connection.

1. **Activate constrained decoding in the evolution engine** (Findings 1, 5). Wire `ConstrainedDecoding.cortexConstrained` / `intentPlanSchema` into the engine's live LLM calls (`Engine.fs`, `Evaluation.fs`), replacing bare `JsonMode=true`. The request-construction helpers exist and are tested; this is a small diff. Payoff: syntactic failures become unrepresentable, so the retry budget (Finding 5) is spent exclusively on semantic failures — the division of labor both the GCD and LLM-Modulo literatures recommend.

2. **Make the router constraint-aware and downgrades loud** (Finding 2). Add one routing rule: `ResponseFormat.Constrained(Ebnf|Regex)` requests route to a grammar-capable backend (vLLM, or llama.cpp GBNF) when configured, and otherwise fail loudly or log an explicit downgrade warning instead of Ollama's silent grammar-discard. Without this, Opportunity 1 silently does nothing on default deployments — the two must land together.

3. **Make `wot.ebnf` the single source of truth** (Finding 8). Add a property test sampling strings from `wot.ebnf` and asserting `WotParser` accepts them, and the converse over the `.wot.trsx` fixture corpus. This must precede any use of the EBNF for guided WoT generation; otherwise grammar drift means the decoder enforces a language the parser rejects.

4. **Replace token-set roundtrip validation with an executable certificate** (Finding 7). For DslClause/GrammarRule promotions, compile both the RollbackExpansion and the abstraction with `dotnet fsi` and assert behavioral equivalence on stored trace inputs — the promotion record already carries evidence traces. This upgrades a lexical heuristic that provably passes meaning-inverting edits into an observational-equivalence check, a practical approximation of proof-carrying promotion.

5. **Bridge WeightedGrammar posteriors into generation** (Finding 6). Emit a weighted GBNF/EBNF variant in which high-posterior productions are preferred, or at minimum use the Beta-Binomial weights for exemplar selection with explicit probability annotations in the prompt. This turns the one genuinely learning subsystem from a 0.08-capped selection tiebreaker into an actual probabilistic decoding policy — the direction Grammar-Aligned Decoding argues is required for distribution-faithful constrained sampling.

6. **Ground the constraint invariants in real measurements, or delete the dead module** (Findings 3, 4). Either wire `NeuroSymbolicEvolution` into the breeder loop (its SelectAgent/EvaluateMutation/PreparePrompt hooks map cleanly) with invariants fed by real data — cyclomatic complexity from BenchmarkRunner's compiled solutions, `WotParser`/`dotnet fsi` as the GrammarValidity oracle, context maps populated from execution traces — or remove it. A headline module that is both dead and vacuous is worse than absent: it misleads audits, including previous internal ones.

7. **Give the BeliefUpdate IR a real consumer** (Finding 9). Back the assert/retract/revise operations with a Datalog- or Z3-checked belief store so consistency is decided by entailment rather than substring antonyms, upgrading the `Engine.fs:197-244` LLM yes/no contradiction check into a Logic-LM-style neural-parse → symbolic-reason pipeline. This is the largest item, but it is the one that would make "neuro-symbolic" true in the reasoning sense rather than only the generation-gating sense.

## References

1. Brandon T. Willard and Rémi Louf. "Efficient Guided Generation for Large Language Models." 2023. arXiv:2307.09702.
2. Yixin Dong et al. "XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models." 2024. arXiv:2411.15100.
3. Saibo Geng, Martin Josifoski, Maxime Peyrard, and Robert West. "Grammar-Constrained Decoding for Structured NLP Tasks without Finetuning." Findings of EMNLP 2023. arXiv:2305.13971.
4. Kanghee Park, Jiayu Wang, Taylor Berg-Kirkpatrick, Nadia Polikarpova, and Loris D'Antoni. "Grammar-Aligned Decoding." NeurIPS 2024. proceedings.neurips.cc/paper_files/paper/2024.
5. Zhibin Gou et al. "CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing." ICLR 2024. arXiv:2305.11738.
6. Subbarao Kambhampati et al. "Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo Frameworks." ICML 2024, PMLR v235. arXiv:2402.01817.
7. Liangming Pan, Alon Albalak, Xinyi Wang, and William Yang Wang. "Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning." Findings of EMNLP 2023. arXiv:2305.12295.
8. Robin Manhaeve et al. "DeepProbLog: Neural Probabilistic Logic Programming." NeurIPS 2018. arXiv:1805.10872.
9. George C. Necula. "Proof-Carrying Code." POPL 1997. DOI 10.1145/263699.263712.
