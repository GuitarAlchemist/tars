# Agentic Engineering — the harness is the work

> A read-on-demand reference, **not** an always-loaded instruction block. Distilled from Matt
> Pocock's "Agentic Engineering Workflow" (aihero.dev) + Ousterhout's *A Philosophy of Software
> Design*, and mapped to **this repo's** existing machinery. Read it when you're deciding *how* to
> direct AI on a non-trivial change — not on every turn. (Mirrors `ga/docs/methodology/agentic-engineering.md`,
> adapted to tars.)

## The one idea

**Optimise the harness, not the model.** The model is the engine; the *harness* — prompts, skills,
the codebase itself, the environment the agent runs in — is roughly half the system and the half you
fully control. The load-bearing consequence:

> *"How do you optimise token spend? Have a codebase that's easier to make changes in."*

A deeper, lower-duplication, better-documented codebase lets a **cheaper model** do the same work with
fewer tokens. tars takes this further than most: its *whole point* is a self-improving harness — the
probabilistic grammar weights, the `PromotionPipeline` (Inspect → Extract → Classify → Propose →
Validate → Persist → Govern), and the `~/.tars/promotion/` recurrence + lineage stores exist to make
the agent's own reasoning measurable and promotable.

## Strategic over tactical

AI ate **tactical** programming (writing syntax, chasing bugs, making commits) — it's cheaper and
faster than you at it. Your leverage is **strategic** programming (Ousterhout):

- **Design the hard parts up front.** Decide the consequential things before delegating (the Karpathy
  4 Rules in `CLAUDE.md` make this explicit: state interpretation + assumptions, ask one question, wait).
- **Scope tasks tightly.** A well-scoped task is one an AFK agent can finish with no further context.
- **Own the interfaces / seams between modules.** This is where bugs and rework concentrate.
- **Keep just-enough docs that point agents to the right place** — not exhaustive, navigational.

"Your skills are the ceiling on what AI can do." Delegate the tactical; keep the strategic mindset.

## DX ≈ AX

Agent experience ≈ developer experience. What makes a codebase pleasant for a senior human makes it
tractable for an agent: **deep modules** (a lot of behaviour behind a small interface), **low
duplication**, **clear seams**, **guardrails** (types, tests, invariants). Improving the codebase
*is* improving the harness. The `/improve-codebase-architecture` vocabulary (module / interface /
depth / seam / **deletion test**) is the shared language for this across the ecosystem (it ships with
the aihero skill set; `setup-matt-pocock-skills` wires it in). Real deep modules already in tars:
`WotParser` (`v2/src/Tars.DSL/Wot/WotParser.fs` — full .wot.trsx grammar behind a small parse API),
`PromotionPipeline` (`v2/src/Tars.Evolution/PromotionPipeline.fs` — a 7-step loop behind save/load),
and `WoTController` (`v2/src/Tars.Cortex/WoTController.fs` — a large reasoning state machine behind one
entry point). The domain vocabulary belongs in a root `CONTEXT.md` (created lazily per
[docs/agents/domain.md](../agents/domain.md) — not yet seeded; seed it when an ambiguity is resolved).

## Procedures vs abilities (and context hygiene)

- **Procedure** — a skill *you* invoke to stay in the driver's seat. tars currently ships `digest` and
  `correct`; the aihero procedures (`/grill-me`, `/to-prd`, `/to-issues`,
  `/improve-codebase-architecture`) install via `setup-matt-pocock-skills`. Prefer procedures; keep the
  thinking in the human.
- **Ability** — a skill the *model* self-invokes. Every ability leaks its description into the context
  window. Too many = bloat; mark deliberate procedures `disable-model-invocation: true`.

Matt's blank-slate test: periodically strip skills / MCP / CLAUDE.md back toward nothing, watch what
the agent does unaided, then **layer back only the procedures you deliberately choose**. Treat a long
CLAUDE.md as a smell (tars's is a lean ~103 lines — keep it that way; push detail into read-on-demand
docs like this one).

## Queues, not loops

The unit of AFK work is a **queue** of well-scoped tasks, not an infinite prompt loop. Tasks flow
**triage → explore → implement → review → merge**, pulled off by labelled agents. tars already speaks
this: GitHub Issues + the canonical triage labels (see
[docs/agents/triage-labels.md](../agents/triage-labels.md)), `/loop 1h` gated by
`state/governance/dev-process-overseer.json` (`loop-eligible`), and the cross-repo `~/.demerzel/HALT-ALL`
marker that every `/auto-optimize` consumer reads first. Keep **human-in-the-loop checkpoints**, but
push them as far toward the final output as the work safely allows.

## Build self-improving systems

When a model finds a deep bug, the lesson is **not** "the model is great" — it's *"I should have a
system that catches this."* Prefer a cheap, scheduled review over waiting for a smarter model. *"If
someone keeps stealing your bike, buy a lock."* tars is built around this: `karpathy-cherny-discipline.yml`
(session-continuity drift gate), `agent-blackbox.yml` (harness-audit, fail-below 60), the
`state/quality/tars-harness/baseline.json` readiness baseline (oracle: `Scripts/verify.ps1`), and the
`~/.tars/promotion/` recurrence/lineage stores that promote recurring reasoning patterns. The grammar-
evolution loop *is* a self-improving system — extend it rather than one-shotting fixes.

## Make review seamless

The bottleneck is human review, so spend the harness on making review *fast*: rich PR context, AI-
assisted review passes, structured diffs. tars's edge here is its identity — it's the ecosystem's
**cross-model theory validator**. The Demerzel governance layer (vendored under `governance/demerzel`)
supplies the cross-model review machinery: a secondary model (Claude/Gemini/Codex, detected from PR
labels/commit trailers) reviews PRs, and `/demerzel-cross-review` + `/demerzel-consult` orchestrate
multi-model second opinions. You stay the gate on security and on "did the system do a good job," but
you make that gate one click.

## You own the product

AI is weak at original ideas and at deciding *what* to build. Choose the features; ask "what can I
**remove**, how do I make this **simpler**." The classic product-design fundamentals still hold — AI
just implements them faster.

## The two action steps Matt actually recommends

1. **Strip to a blank slate, then layer deliberately.** Remove the bloat; re-add only procedures you
   choose and can customise.
2. **Move work AFK.** Scope a task tightly, hand it to a sandboxed agent (a git worktree off `main`),
   review the result. Two of you, then three, then five — then you review.

---

*Pointer, not gospel: this doc is read when you're deciding how to direct a non-trivial change. It is
deliberately **not** wired into the always-loaded instruction set — that would contradict its own
context-hygiene advice.*
