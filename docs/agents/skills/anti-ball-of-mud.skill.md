# Skill: Anti-Ball-of-Mud

Adapted for TARS from `mattpocock/skills` (`improve-codebase-architecture`,
`codebase-design`, `grilling`, `domain-modeling`, `to-issues`). This is the
TARS-native, repo-readable, AFK-safe port — it teaches an agent to detect
architecture entropy, name the seam, and create **backpressure** before AFK
work makes the mud worse. It does **not** authorize broad autonomous refactors.

> Mapping to the upstream skills is recorded in
> [`matt-pocock-skills-mapping.md`](./matt-pocock-skills-mapping.md). The shared
> design vocabulary is in
> [`codebase-design-vocabulary.md`](./codebase-design-vocabulary.md).

## Two ways to use this skill

This skill has a **user-invoked** mode and a **model-invoked** mode. They share
the same vocabulary and the same output shape, but differ in *who triggers them*
and *how much latitude the agent has*.

### User-invoked — `/anti-ball-of-mud`

Use when a human explicitly asks the agent to scan a change, plan, or code area
for architecture entropy **before** implementation.

Expected behavior:

- inspect the requested change or named code area;
- produce a short entropy report (the eight-part output below);
- offer **1–3 candidate seams**, ranked, with a recommendation;
- if the choice between seams is architectural, **ask the human which to
  explore** rather than picking silently;
- never rewrite broadly without explicit approval.

### Model-invoked — `anti-ball-of-mud-guard`

Use **automatically** when an AFK task risks worsening architecture entropy. This
is the guard rail, not the orchestration command. When a trigger fires mid-task,
the agent stops feature work, records the friction, and produces the same
eight-part output as a backpressure note.

Triggers (any one is enough):

- the change is about to touch **many unrelated files**;
- you are adding logic to an **already-large orchestrator/module** instead of
  behind a seam;
- you are **duplicating** a contract, scoring rule, or analytics shape that
  already exists (especially one that belongs in `ix`);
- you are about to thread a **provider SDK call** (LLM / GitHub / network / FS)
  into core runtime code instead of through an adapter;
- you are about to add a **shortcut** because the proper seam is unclear;
- a **Demerzel governance rule** is being re-expressed as a local `if` instead of
  going through the existing gate.

Expected behavior when a trigger fires:

- **stop** and record the friction (do not push through);
- propose **one narrow seam**;
- keep the PR reviewable (respect the rewrite budget in `AGENTS.md`);
- **escalate to human/Demerzel review** when the decision is architectural.

## When to use

Reach for this skill when an issue or in-flight AFK task shows architecture
**friction** — not generic untidiness, but the specific entropy signals below.
The aim is testability and AI-navigability: deep modules behind small interfaces,
placed at clean **seams**, tested **through** their interface.

This skill is *informed by* the repo's domain language:

- read `CONTEXT.md` for the TARS ubiquitous language (it already names good seams:
  **IxSkill**, **hermetic gate**, **fitness gate**, **tars_bridge**, …) — use
  those names, don't invent new ones;
- read the relevant ADRs in `docs/adr/` and **do not re-litigate** a decision an
  ADR has already recorded; if a candidate contradicts an ADR, say so explicitly
  and only surface it when the friction is real enough to reopen the ADR.

## Allowed files/surfaces

- **Read-only, anywhere:** any source under `v2/src/`, `CONTEXT.md`, `docs/adr/`,
  `governance/`, examples — to gather evidence.
- **Writable (this skill only writes here):** `*.md` under `docs/`, example
  `*.json` / `*.md` under `examples/agents/`, and (only if explicitly requested)
  a new ADR draft under `docs/adr/`.
- **Forbidden:** editing F# (`*.fs`/`*.fsproj`), Rust (`*.rs`), CI workflows
  (`.github/workflows/*.yml`), or any **one-way-door** / **blocked** path listed
  in `AGENTS.md`. A *seam proposal* is a document, not a refactor — producing the
  refactor is a separate, human-gated PR.

## Architecture entropy signals

Look for these concrete signals. They map directly to TARS surfaces — prefer a
real example from this repo over an abstract one.

- a large module mixing **unrelated responsibilities** (e.g. an orchestrator that
  also does IO, provider calls, *and* ranking);
- understanding one concept requires **bouncing across many shallow modules**;
- the **interface is nearly as complex as the implementation** (a shallow module —
  apply the deletion test);
- **feature logic mixed with orchestration, IO, provider SDK calls, or UI**;
- **stringly-typed routing** where a typed contract should exist (e.g. capability
  dispatch by raw string instead of a discriminated union);
- repeated **ad-hoc JSON shapes** with no schema/contract (compare to the
  `examples/agents/*.example.json` and `docs/contracts/` conventions);
- **shallow wrappers** that only rename complexity (fail the deletion test);
- **hidden global mutable state**;
- tightly-coupled modules **leaking across their seams**;
- pure functions extracted **only for testability** while the real bug hides in
  the orchestration that calls them (no **locality**);
- **direct dependency on GitHub / LLM / filesystem / network from core runtime**
  (TARS rule: LLM access always via `LlmFactory.create(logger)`, never a direct
  `DefaultLlmService` — a bypass is an entropy signal);
- **copy-pasted algorithms** or duplicate scoring/analytics logic that should
  belong to **ix** (the ML/governance engine) rather than being reimplemented in
  TARS;
- **Demerzel governance rules scattered as local `if` statements** instead of
  routed through the governance gate;
- **tests that verify implementation details** instead of the intended
  interface/seam.

The full catalog, with TARS-specific examples and severity, is in
[`entropy-signal-catalog.md`](./entropy-signal-catalog.md).

## Required output (the eight-part report)

When invoked, produce exactly these eight parts. Keep it short — a report, not an
essay.

1. **Entropy finding** — what looks muddy and why, citing the signal(s) above and
   the concrete file(s).
2. **Design vocabulary** — name the concepts in play using the
   [vocabulary](./codebase-design-vocabulary.md): **module**, **interface**,
   **depth**, **seam**, **adapter**, **locality**, **leverage**, **deletion
   test**. Use the terms exactly; don't drift into "component / service / API /
   boundary".
3. **Proposed seam** — **one** boundary to introduce or strengthen. One seam, not
   a rewrite plan.
4. **Smallest safe change** — what fits in **one reviewable PR** under the rewrite
   budget.
5. **Files likely touched** — concrete paths if known.
6. **Test surface** — how the seam can be tested **through its interface**
   (replace, don't layer).
7. **Risk / backpressure** — why the PR should **stop here** rather than continue.
8. **Review gate** — the human/Demerzel approval needed before any broader
   refactor.

A worked example of this output, for a deliberately muddy change request, is in
[`../../../examples/agents/anti-ball-of-mud.expected-output.md`](../../../examples/agents/anti-ball-of-mud.expected-output.md).

## Required checks

1. **Evidence-before-refactor.** Do not recommend a refactor without naming the
   entropy signal and the file that exhibits it. A finding with no cited evidence
   is a stop.
2. **One-seam check.** The output proposes exactly one seam. If you find yourself
   listing three seams to introduce *now*, you have a rewrite — split it.
3. **Scope check.** Confirm no F#/Rust/workflow/one-way-door file is edited by
   this skill. The deliverable is docs/examples, not code.
4. **Vocabulary check.** Every architectural claim uses the
   [codebase-design vocabulary](./codebase-design-vocabulary.md) exactly.

## Stop conditions (AFK backpressure)

Stop and escalate — do **not** push through — when any of these hold:

- the seam needs **product or architecture judgment** (which of several valid
  shapes is correct);
- the smallest safe change still exceeds the **rewrite budget** in `AGENTS.md`
  (max lines / lines-per-fix / diff budget) — split into multiple PRs;
- the change would require editing a **one-way-door** path (`schemas/**`,
  `contracts/**`, `migrations/**`, `**/*.sln`, `**/*.fsproj`,
  `Directory.Build.props`) or a **blocked** path;
- the fix implies **copying ix or Demerzel logic into TARS**, or adding a **new
  provider/cloud dependency**;
- `governance/state/afk-halt.json` exists and is active (see
  [`governed-delegation.skill.md`](./governed-delegation.skill.md));
- the task's `issue_meta.afk.max_autonomy` is exceeded by what the fix would need.

When you stop, write the eight-part report as a **PR note or issue comment** and
hand the architectural decision to a human or the Demerzel tribunal.

## AFK constraints

This skill is safe for AFK delegation. It must **not**:

- do a broad repo rewrite, or auto-merge;
- perform a large mechanical rename unless explicitly requested;
- silently migrate architecture;
- add a new provider/cloud dependency;
- copy ix or Demerzel logic into TARS;
- leave permanent repo artifacts from exploratory reports unless requested
  (prefer a PR note / issue comment over committing an `architecture-review-*`
  file).

Prefer **docs / examples / contracts / tests** over speculative implementation.

## Evidence expected in the PR body

When this skill drives a PR (e.g. the docs deliverable, or a single approved
seam), the PR body should contain:

- the **eight-part report** (or a link to where it was recorded);
- a statement that **no runtime F#/Rust/workflow code was modified** beyond the
  one approved seam (if any);
- the **review gate** invoked (human reviewer or Demerzel tribunal) for any
  architectural decision;
- confirmation that `governance/state/afk-halt.json` was checked and inactive.

## Common failure modes

- **Turning into a clean-code checklist.** Generic "rename this, extract that"
  advice with no seam and no deletion test is out of scope — this skill is about
  *one boundary*, named with the design vocabulary.
- **Proposing a rewrite disguised as a seam.** Three new modules in one PR is a
  rewrite. Stop at one seam and escalate.
- **Skipping the evidence.** Recommending a refactor without citing the entropy
  signal and the file — reviewers (and the Demerzel qa-tribunal) will reject it.
- **Editing code to "just fix it".** The moment you open a `.fs` file to apply
  the refactor under AFK autonomy, you have left the skill's lane.
- **Re-litigating an ADR.** Suggesting a refactor that an existing ADR already
  decided against, without flagging the conflict.
- **Reinventing ix/Demerzel.** Re-implementing scoring/governance that already
  lives in a sibling repo instead of routing to it through the existing seam.
