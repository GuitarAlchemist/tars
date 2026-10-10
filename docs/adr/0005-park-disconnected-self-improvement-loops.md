# ADR 0005 — Park the disconnected self-improvement loops; one loop on verified signals

- **Status:** Accepted, 2026-10-10.
- **Context source:** grooming session "how can TARS self-improve efficiently?" It followed a GPU study, a 7B vs 14B evolve A/B and a replay of the judge's verdicts. Builds on ADR 0002 (self-hosting gate) and ADR 0003 (couple the gate to SelfTrain).
- **Question:** if the model is too weak semantically, should the self-improvement strategy change? If so, to what?

## Context

### The model is not today's bottleneck

The measurements, on October 2026 runs:

| Measurement | Result |
|---|---|
| Benchmark, 19 F# problems | qwen2.5-coder:7b 15–17, qwen2.5-coder:14b 18–19, qwen3-coder:30b 14–16. Size matters, quantization barely does. The 14B saturates the benchmark. |
| evolve, 10 tasks, Sonnet as teacher | 7B: 6/10, with 6 tasks passing their examples. 14B: 6/10, with 9 tasks passing their examples. |
| The 14B's 4 failures | 1 real bug. The other 3 broke a technique constraint (`List.fold`, `Result.bind`). Replaying the judge gave 9 identical, correct verdicts out of 9. |

The remaining failures were scaffold failures, not model failures:

- The executor prompt dropped every constraint after the 3rd. Fixed by #386.
- A judge's rejection never reached the executor. Fixed by #387.

### Persisted signals are judged, verified signals are not persisted

An inventory of every self-improvement mechanism sorted them in two groups:

- **Loops that persist something across runs, all on judged, circular or polluted signals.** Up to #389, 609 of the outcome store's 1076 rows were written by a unit test.
- **Verified signals that no loop reuses.** These are evolve's examples, which run with the code in `dotnet fsi`, the self-hosting gate's `dotnet test`, and benchmark PASS. Evolve persists no per-task result. The SFT dataset (`~/.tars/self_train/dataset.jsonl`) has never been trained on.

The four loops this ADR parks, with what feeds them (paths under `v2/src/`):

- **Promotion pipeline and index boost** (`Tars.Evolution/PromotionPipeline.fs`, `PromotionIndex.fs`).
  - `propose` receives the pattern's name as its template.
  - The Bayesian weight update counts the Grammar Governor's own `Approve` as a success, so the pipeline's approvals train its own ranking.
  - `tars promote run` feeds it synthetic artifacts (`Tars.Interface.Cli/Commands/PromoteCommand.fs`, "Create synthetic artifacts to demonstrate the pipeline").
- **Grammar-weight replicator** (`tars grammar evolve`, MCP `grammar_evolve`).
  - The CLI derives each rule's outcomes from the rule's own `SuccessRate × SelectionCount` (`GrammarCommand.fs`).
  - It then overwrites `Weight` with the replicator proportion.
  - The MCP tool passes `Map.empty` as outcomes (`Tars.Evolution/McpGrammarTools.fs`).
- **RetroactionLoop** (`Tars.Evolution/RetroactionLoop.fs`). It audits `PatternLibrary.loadAll ()`, but `PatternLibrary` has no function that writes a pattern, so the library is empty unless someone puts files in `.tars/patterns` by hand.
- **Darwin loop** (`runDarwinLoop` in `Tars.Evolution/Engine.fs`).
  - It runs only when `--focus` names a `.trsx` file; its fallback returns `None`.
  - It logs a proposed mutation as an improvement (`logImprovement ... true`) without checking that the mutated workflow works.

## Decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Park the four loops above.** Their code and tests stay, and they still build. They get no new features, no tuning and no new callers. They get fixes only when they break the build or corrupt shared data. | They give the appearance of learning without a signal that could make it real. Removing them is a separate decision. |
| D2 | **Self-improvement effort goes into one loop fed only by verified signals**: examples run with the code, `dotnet test`, benchmark PASS, and mechanical constraint checks. An LLM judge may reject an answer and say why (#387), but its verdict alone is not training data. | A verified signal cannot be talked into a pass. This is the anti-collapse anchor of ADR 0003, applied to the whole loop. |
| D3 | **Claude takes the semantic roles and the local model does the volume.** Claude writes the curriculum, judges, and supplies reference solutions for tasks the local model fails, through `claude -p` on the subscription. Budget: **about 30 Claude calls per 10-task evolve run.** | The semantic roles need the stronger model. Executing tasks does not: the 14B already passes most examples. |
| D4 | **Distillation is gated.** Verified (task, solution) pairs, from evolve and from Claude's reference solutions checked the same way, join the SFT dataset. A LoRA on the **7B** is tried first. It is kept only if it beats the current model on a held-out split of a harder benchmark, using the `self-train cycle` A/B from ADR 0003. The GPU training step is run by the owner. | It is the only path to a better model, and the gate keeps a worse one out. The 7B fits the GPU for LoRA, and the 14B is the bar to approach. |
| D5 | **A parked loop is un-parked only together with a verified signal.** For example, promotion weights could be fed by verified task outcomes instead of governor approvals. That would be decided in a later ADR. | It keeps the door open without letting judged signals back in. |

## The loop

```
Claude (teacher)  → task with examples that can run
local 14B         → answer
                    (repair from failing examples, then from mechanical constraint checks, then from the judge's remarks)
verified result   → persisted, one JSONL row per task
failure           → Claude's reference solution, verified the same way
(task, solution)  → SFT dataset → LoRA (7B first)
held-out split    → kept only if better (the gate)
```

ADR 0002/0003's self-hosting gate stays the verified loop for TARS's own source.

## Backlog this ADR orders

These are vertical slices, one PR each, each starting from a failing test. #386, #387 and #389 are the first ones.

1. Mechanical "use X / don't use X" constraint checks, which turn judged rejections into verified, repairable ones.
2. Evolve persists one verified result per task.
3. A tier-5 benchmark with hidden tests and a held-out split, because the current one is saturated.
4. The distillation experiment (D4).

## Consequences

- `CONTEXT.md`'s entry for the promotion pipeline called it "the closed self-improvement loop". It now points here.
- Constrained decoding with EBNF grammars is **not** parked. Only the replicator's weight evolution is.
- Parked code still costs build and test time. Deleting it needs its own decision, and it touches callers on the MCP surface.
- The ~30-call budget can change. A run that needs more should say so in its log, never silently exceed it.
