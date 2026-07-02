---
title: Building a test-gated self-hosting improvement loop (TARS improves TARS)
date: 2026-06-22
category: integration-issues
component: Tars.Evolution / Tars.Interface.Cli / ix
problem_type: integration
status: solved
verified: live (qwen2.5-coder:7b autonomously fixed a failing test → PROMOTED, 0 regressions)
related_adrs:
  - docs/adr/0001-tars-ixql-pipeline-mesh.md
  - docs/adr/0002-tars-self-hosting-improvement.md
  - docs/adr/0003-couple-self-hosting-to-selftrain.md
tags:
  - self-improvement
  - git-worktree
  - dotnet-test
  - trx
  - ix-pipeline
  - llm-json
  - fsharp
  - goodhart
---

# Building a test-gated self-hosting improvement loop

## Problem

Make TARS improve its **own** F# source autonomously and safely: an LLM proposes
an edit to make a failing test pass, and the change is only accepted if it is
verified (builds, target test flips to pass, zero regressions) — without ever
endangering the working tree or letting the model "game" the gate.

## Solution (the reusable shape)

A four-stage loop, each stage a thin, independently-testable seam:

1. **Generate** — `SelfHostingGate.runGateGenerated llm repo proj test file`: prompt
   the model with the failing test + source file; parse a `{rationale, old_text,
   new_text}` mutation (`buildProposePrompt` / `parseProposal`, both pure).
2. **Verify (hermetic gate)** — `runGate`: `git worktree add --detach <tmp> HEAD`,
   baseline `dotnet test` (TRX), apply the single-occurrence edit, variant
   `dotnet test` (TRX), then `decide`:
   - test files are **immutable** (taken from HEAD); only the one non-test source
     file may change,
   - **test set unchanged** (blocks delete/`Skip` gaming),
   - **zero regressions** (no `Passed → not-Passed`),
   - **target test was failing and now passes**.
3. **Promote** — on Accept, `git checkout -b self-improve/<id>` + commit *in the
   worktree* (branch ref survives `git worktree remove`); never touches main.
4. **Consolidate** — `recordWin` appends the verified diff as an SFT example to
   `~/.tars/self_host_wins.jsonl`; `SelfTrain.exportDataset` merges it so verified
   source fixes become model-training data (the compounding link).

Driven by `tars self-improve --test <name> --file <relpath> --project <proj.fsproj>
[--repo <root>] [--model <name>]`.

Key files: `v2/src/Tars.Evolution/SelfHostingGate.fs`,
`v2/src/Tars.Interface.Cli/Commands/SelfImprove.fs`,
`v2/src/Tars.Evolution/SelfTrain.fs`.

## Root insight

Recursive self-improvement compounds only as far as the **verifier** is
trustworthy and cheap. The test suite is a strong, hard-to-game verifier; an MCTS
reward that returns `1.0` trivially is not. More search on a weak signal just
Goodharts faster — so the leverage is in the *fitness function*, not the search.
The hermetic invariants exist precisely because parallel best-of-N would otherwise
*find the gaming exploit faster*, not slower.

## Non-obvious gotchas (the grep-findable payload)

1. **`ix pipeline run --param NAME=@file` requires the file to be valid JSON.**
   Raw text (e.g. an EBNF grammar) must be `JsonSerializer.Serialize`'d into a
   JSON-encoded string first, or restore/run errors with "not valid JSON".
2. **`Ok`/`Error` DU collision in this repo** (`GitResult.Success/Error` and
   others shadow `FSharp.Core`). In pattern matches over `Result`, qualify:
   `Result.Ok` / `Result.Error`. Symptom: `FS0001 ... has type 'AgentState'` /
   `FS0025 incomplete pattern match`.
3. **Builds fail at restore on `NU1903`** (SQLitePCLRaw advisory) under
   `TreatWarningsAsErrors`. Add `-p:NuGetAudit=false` to every `dotnet build`/`test`.
4. **The prebuilt `ix` binary can be a `pipeline` stub.** Its `--help` says
   "stub — full impl in Week 4" even when source has the impl. Probe capability via
   `ix pipeline schema` exit 0 (not the help string); rebuild with
   `cargo build -p ix-skill`.
5. **`grammar.search` EBNF dialect is `::=` + newline-separated**, not `=` / `;`.
6. **Parse `dotnet test --logger trx`, not console output.** TRX gives per-test
   `testName`/`outcome` (needed for target-flip + regression detection); console
   only gives pass/fail counts. Build-failure (no TRX) is a distinct Reject signal.
7. **Isolate self-mutation in a git worktree**, never the live tree:
   `git worktree add --detach <tmp> HEAD` → build/test there → `git worktree
   remove --force` to roll back. (This repo already runs ~9 worktrees, so the
   workflow is proven on Windows; fresh-restore + path length are fine.)
8. **Use a coder model for structured generation, not a reasoning model.** The
   configured default `deepseek-r1:1.5b` emits `<think>` tags that pollute JSON;
   `qwen2.5-coder:7b` works. Make the JSON parser robust to markdown fences by
   extracting first-`{` … last-`}` (`parseProposal` does this).
9. **Don't reuse `SelfImprovement.analyzeAndPropose` for the gate's generation** —
   it *applies* a variant as a side effect. Keep generation (LLM) separate from
   application (worktree) so they compose cleanly.
10. **`Selection.evaluate` is too coarse for a per-test gate.** Its
    `Performance{PassRate;TotalCost;DiffCount}` can't express "this specific test
    flipped" or "test set unchanged" — write a per-test `decide` instead.

## Prevention / best practice

- Split every IO-heavy capability into **pure decision logic** (unit-tested) and a
  **thin IO shell** (verified once by a spike). Here: `isTestFile`/`parseTrx`/
  `decide`/`parseProposal` are pure and unit-tested; the worktree+dotnet mechanic
  was proven by a one-off spike, then composed.
- **Spike the riskiest mechanic before designing around it.** The whole self-hosting
  design rested on "can a worktree build+test+parse detect a flipped test on
  Windows" — proving that first (cheap) de-risked everything downstream.
- Keep the verified-only invariant end to end: only gate-Accept'd diffs become SFT
  data, so the training signal can't collapse into unverified self-reference.

## Honest limit

The loop can only fix what its tests specify — it climbs toward the spec, it can't
exceed it. A live Accept needs a genuinely-failing test; TARS's suite is green, so
fixing TARS *itself* requires authoring a red test first. Raising the ceiling is
spec-generation (proposing new tests), deliberately out of scope.
