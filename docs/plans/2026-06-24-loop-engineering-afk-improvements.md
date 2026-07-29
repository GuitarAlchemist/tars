# PRD — Loop-engineering improvements to the AFK delegation loop

- **Status:** Draft — 2026-06-24
- **Source:** `/ask-matt` session applying [loop-engineering](https://github.com/cobusgreyling/loop-engineering)
  to the AFK agent-delegation loop (`ready-for-agent` → Jules → PR) and its
  governance. Follows ADR 0004 (the loop itself) and PR #61 (CI Health Sweeper).
- **Frame:** loop-engineering's risk ladder — **L1 report-only → L2 assisted
  (human gate) → L3 unattended**. The AFK loop is at **L2** (Jules opens PRs,
  human merges); we deliberately refuse L3 (no auto-merge). The CI Health Sweeper
  (#61) is a new **L1** loop. This PRD adds the two remaining loop-engineering
  primitives the loop lacks: **observability (run-log)** and **multi-loop
  coordination (unified kill switch)**.

## Problem

Two gaps remain against the loop-engineering checklist:

1. **No activity detection.** The loop delegates and Jules opens PRs, but nothing
   records *what the loop did* — how many delegations, which were halted/skipped,
   what shipped. Loop-engineering: "know what your loop actually shipped."
2. **The kill switch isn't unified.** The AFK loop reads a tars-local marker
   (`governance/state/afk-halt.json`, ADR 0004), but Demerzel's ecosystem halt is
   a *separate* local marker (`~/.demerzel/HALT-ALL`) that a cloud workflow can't
   read. An ecosystem-wide HALT does not (yet) cascade to the cloud AFK loop.

## Key finding (scoping)

Investigation of the Demerzel submodule established that **kill-switch unification
is a Demerzel/contract change, not a tars edit**:

- Demerzel runs in the cloud (≈12 GitHub Actions) and **has cross-repo write** (its
  LOG shows it committing to tars).
- But `demerzel_halt.py` writes a **local** `~/.demerzel/HALT-ALL`; its docstring
  states there is **no HTTP `POST /halt` endpoint yet** (planned, per GA's
  `docs/plans/…arch-demerzel-overseer-extension-plan.md`).
- A **formal contract already exists**: GA's
  `docs/contracts/2026-05-16-overseer-halt-marker.contract.md` + schema.

Therefore tars should **plug into the contract Demerzel already owns**, not invent
a new mechanism. `afk-halt.json` is the correct interim; the cross-repo cascade is
Demerzel-side work against its existing plan.

## Scope

### In scope (tars-actionable now)

**P1 — AFK run-log / scorecard.** Append-only `governance/state/afk-runlog.jsonl`;
one record per delegation decision:
```json
{"at":"2026-06-24T12:00:00Z","issue":58,"decision":"delegated|halted|no-api-key","reason":"...","pr":null}
```
- The `jules-auto-delegate` workflow appends a record at each decision branch
  (delegated / halted-by-governance / missing-key).
- A lightweight periodic reporter (extend the CI Health Sweeper or a sibling
  scheduled job) summarizes counts + a readiness score into a comment or a
  `governance/state/afk-scorecard.md`, mirroring Demerzel's `LOG.md` health
  scoring. This is the "activity detection" + "run-logs with scores" primitive.

**P2 — Conform `afk-halt.json` to the overseer-halt contract.** Align the marker's
schema (field names, `schema_version`, `scope`, `exempt_agents`, `expires_at`) to
GA's `overseer-halt-marker.schema.json` so Demerzel can drive it unchanged. Today
the marker is an ad-hoc subset; conforming makes it the *cloud projection* of the
ecosystem HALT.

### Out of scope / non-goals

- Building a new halt mechanism (reuse the existing contract).
- The Demerzel HTTP `POST /halt` endpoint (Demerzel roadmap).
- L3 auto-merge (deliberately rejected — human merge stays the gate).

### Cross-repo follow-up (Demerzel issue, not tars)

**P3 — Demerzel halt cascades to cloud consumers.** When `demerzel_halt.py` sets a
HALT, it should also make the marker **cloud-readable** for cloud-only consumers
like the AFK loop — e.g. commit the conforming `afk-halt.json` to each consumer
repo (model "a"; feasible today given cross-repo write), or serve it via the
planned `POST /halt` endpoint (model "c"; long-term). File against Demerzel's
existing overseer-extension plan; reference this PRD.

## Success criteria

- [ ] Every AFK delegation decision appends a record to `afk-runlog.jsonl`.
- [ ] A scorecard summarizing delegations/outcomes is produced on a schedule.
- [ ] `afk-halt.json` validates against the overseer-halt-marker schema.
- [ ] A Demerzel issue (P3) exists, linked to its extension plan and this PRD.

## Open questions (need Demerzel-side answers)

1. Exact `schema_version` — the halt CLI mentions wire `v0.1` but refuses
   `schema_version != 1`; confirm the canonical value before conforming.
2. Cascade mechanism — does Demerzel commit markers to consumer repos (a), or wait
   for the HTTP endpoint (c)?
3. Halt scope semantics — should an ecosystem HALT also stop TARS's *own*
   self-improve loop (already a local consumer), or AFK only?

## Proposed issues (for `/to-issues`)

- **tars #A:** Implement AFK run-log (`afk-runlog.jsonl`) — append at each decision branch. *(agent-ready)*
- **tars #B:** AFK scorecard reporter (scheduled summary). *(agent-ready)*
- **tars #C:** Conform `afk-halt.json` to the overseer-halt schema. *(needs OQ#1)*
- **Demerzel #D:** Cascade HALT to cloud consumers (P3). *(cross-repo; needs OQ#2/#3)*
