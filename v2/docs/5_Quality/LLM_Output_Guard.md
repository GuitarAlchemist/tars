# LLM Output Guard: Detecting Hallucinations, Cargo-Cult Code, and Fabrication

## Goal
Add a post-generation guardrail to classify LLM outputs (text and code) for trustworthiness, then remediate (retry, request evidence, or refuse). Targets: hallucinated facts, fabricated tool/API results, and “cargo-cult” code (redundant/duplicative/over-engineered).

## Signals to Compute
- **Grounding/Provenance**
  - Retrieval coverage: % of answer tokens backed by cited chunks; require explicit citations or IDs.
  - Tool/result alignment: any claimed tool/API result must match actual tool output (hash/ID/timestamp).
- **Schema/Shape**
  - JSON/GBNF validation (GrammarDistill/GrammarValidation) for structured outputs.
  - Metascript/DSL validation: must parse and pass DSL validators.
- **Consistency**
  - Self-consistency check (low-cost): second short pass to re-evaluate key facts; flag divergence.
  - Contradiction vs context: detect when answer conflicts with provided docs/tool outputs.
- **Cargo-Cult Heuristics (code)**
  - Duplicate capability: new registry/kernel/tool layer when one already exists (ownership map).
  - Zero-usage risk: added types/files with no references/tests (static `rg` check).
  - Over-wide enums/tiers or placeholder validators without wiring.
- **Safety/Policy**
  - Forbidden patterns: invented stack traces, fake citations, fabricated benchmarks.
  - Suspicious numerical claims: unverified metrics, performance numbers without source.

## Remediation Actions
- **Retry with constraints**: tighten prompt (“cite sources”, “do not invent tool outputs”), lower temp, restrict tools.
- **Request evidence**: ask LLM to surface the supporting citations/tool call IDs; refuse if absent.
- **Fallback**: degrade to minimal/partial answer that only includes verified content.
- **Route to human**: mark “needs review” for high-risk responses.
- **Block**: refuse when violating policy (fake tool/API results).

## Integration Points in TARS
- **Response Guard stage** (after LLM call, before user/tool dispatch):
  - Run shape validation: GrammarValidation/DSL parse.
  - Run provenance check: ensure cited retrieval chunks or tool outputs cover claims.
  - Run cargo-cult code check: (a) ownership map violation (new kernel/registry/tool layers), (b) zero-usage detection stub (no refs/tests).
  - Compute a risk score; choose action (accept/trim/retry/ask/deny).
- **Epistemic Governor**: consume guard signals; escalate when entropy/drift is high.
- **Metrics**: log risk score, remediation action, and whether a retry succeeded.

## Minimal Implementation Plan
1) **Validator interface**: expose `IOutputGuard.Evaluate(response, context) -> GuardResult` (risk score + action + messages).
2) **Shape checks**: use existing GrammarDistill/GrammarValidation; add DSL parse for Metascripts.
3) **Provenance checks**: require citations/IDs in responses; match against retrieval chunks/tool outputs.
4) **Cargo-cult stub**: forbid new kernel/registry modules (ownership map), flag “no references” additions (needs static analyzer hook).
5) **Actions**: implement “retry with stricter prompt” and “respond with cannot-verify” paths; log all guard decisions.

## Future Enhancements
- Add JSON Schema/GBNF-backed validators.
- Static usage analyzer in CI to block zero-ref code.
- Lightweight self-consistency via cheap model re-ask on key claims.
- Auto-test generation for code answers (compile/run harness for snippets).
