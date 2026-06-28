# Skill: Anti-Ball-of-Mud

## Goal
Port the spirit and workflow shape of `mattpocock/skills` into TARS as a small AFK-safe skillset for preventing agent-driven architecture entropy.

## When to use

### User-invoked candidate
`/anti-ball-of-mud`

Use when the user explicitly wants the agent to scan a change, plan, or code area for architecture entropy before implementation.

Expected behavior:
- inspect the requested change;
- produce a short entropy report;
- offer 1–3 candidate seams;
- ask which seam to explore if the choice is architectural;
- never rewrite broadly without explicit approval.

### Model-invoked candidate
`anti-ball-of-mud-guard`

Use automatically when an AFK task risks worsening architecture entropy.

Triggers:
- touching many unrelated files;
- adding logic to an already large orchestrator/module;
- duplicating contracts or scoring logic;
- mixing provider SDK calls into core runtime;
- adding a shortcut because the proper seam is unclear.

Expected behavior:
- stop and record the friction;
- propose one narrow seam;
- keep the PR reviewable;
- escalate to human/Demerzel review when the decision is architectural.

## Architecture entropy signals
Detect concrete signals:
- large files or modules with unrelated responsibilities;
- understanding one concept requires bouncing across many small shallow modules;
- interface is nearly as complex as implementation;
- feature logic mixed with orchestration, IO, provider SDK calls, or UI concerns;
- stringly typed routing where a typed contract should exist;
- repeated ad-hoc JSON shapes without schema/contract;
- shallow wrappers that only rename complexity;
- hidden global mutable state;
- tightly coupled modules leaking across their seams;
- pure functions extracted only for testability while real bugs hide in orchestration;
- direct dependency on GitHub/LLM/filesystem/network from core runtime code;
- copy-pasted algorithms or duplicate scoring/analytics logic that should belong to IX;
- Demerzel governance rules scattered as local `if` statements;
- tests that verify implementation details instead of the intended interface/seam.

## Required output from the skill

When invoked, the agent should produce an 8-point report:
1. **Entropy finding** — what looks muddy and why.
2. **Design vocabulary** — module, interface, seam, adapter, locality, leverage, deletion test, etc. (reference `codebase-design-vocabulary.md`).
3. **Proposed seam** — one boundary to introduce or strengthen.
4. **Smallest safe change** — what can fit in one reviewable PR.
5. **Files likely touched** — concrete paths if known.
6. **Test surface** — how the seam can be tested through its interface.
7. **Risk/backpressure** — why the PR should stop here.
8. **Review gate** — human/Demerzel approval needed before broader refactor.

## AFK constraints
- no broad repo rewrite;
- no auto-merge;
- no large mechanical rename unless explicitly requested;
- no silent architecture migration;
- no new provider/cloud dependency;
- no IX or Demerzel logic copied into TARS;
- no permanent repo artifacts from exploratory reports unless requested;
- stop when the seam needs product/architecture judgment;
- prefer docs/examples/contracts/tests over speculative implementation.
