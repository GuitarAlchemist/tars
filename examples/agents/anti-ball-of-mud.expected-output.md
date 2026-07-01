# Anti-Ball-of-Mud — Expected Output (worked example)

This is the **eight-part report** the
[`anti-ball-of-mud`](../../docs/agents/skills/anti-ball-of-mud.skill.md) skill
produces, for the muddy change request described in
[`anti-ball-of-mud-skill-invocation.example.json`](./anti-ball-of-mud-skill-invocation.example.json):
*"add retry + telemetry to `MctsBridge`."*

It is a teaching example — the file paths and code shape are illustrative, not a
committed refactor. The point is the **shape** of the output: one cited finding,
one named seam, one reviewable PR, and a hard stop with a review gate.

---

## 1. Entropy finding

Adding retry/backoff and telemetry **directly inside**
`v2/src/Tars.Evolution/MctsBridge.fs` triggers two high-severity catalog signals:

- **Row 12 (duplicated scoring/analytics that belongs to ix):** the proposed
  retry/backoff reimplements bandit-style backoff that already exists in `ix`
  (`bandit`, `grammar.weights`). TARS would own a second copy of an algorithm the
  ecosystem already has.
- **Row 11 (core depends directly on network/provider):** the telemetry hook
  would add a direct network call into core evolution runtime, bypassing any
  adapter seam.

`MctsBridge` is already a seam (Rust `ix` ↔ F# `MctsSolver` fallback, per
`CONTEXT.md`). Bolting retry + telemetry inside it mixes **three** responsibilities
into one module: MCTS dispatch, retry policy, and telemetry IO.

## 2. Design vocabulary

- **Module:** `MctsBridge` — currently a clean **deep** module (one `search` call
  hides Rust-vs-F# selection).
- **Interface:** `search(spec) -> derivation`. Adding retry/telemetry params would
  make the **interface** nearly as complex as the implementation → drift toward
  **shallow**.
- **Seam:** retry and telemetry each want their **own seam**, not to be folded
  into the MCTS seam.
- **Adapter:** telemetry should sit behind a telemetry **adapter** (a no-op
  adapter in tests, a real one in prod) — **two adapters = a real seam**.
- **Locality / leverage:** keeping retry as a policy the bridge *accepts* (rather
  than *creates*) keeps the bug surface local and reusable.

## 3. Proposed seam

**One** seam: a `Telemetry` interface that `MctsBridge` *accepts* as a dependency
(default = no-op adapter). Retry/backoff is **delegated to `IxSkill`** rather than
reimplemented — i.e. no new TARS algorithm, just route to the existing ix seam.

## 4. Smallest safe change

A single reviewable PR that:

- introduces a tiny `ITelemetry` (or `Telemetry` record-of-functions) interface
  with a no-op default;
- threads it into `MctsBridge.search` as an accepted dependency;
- leaves retry/backoff to a follow-up that calls `IxSkill`, **not** to this PR.

No new provider dependency; no copied ix logic; stays under the rewrite budget.

## 5. Files likely touched

- `v2/src/Tars.Evolution/MctsBridge.fs` (accept the telemetry dependency)
- a new small `Telemetry` module under `v2/src/Tars.Core/`
- `tests/Tars.Tests/` (a test that asserts the no-op adapter is exercised through
  the interface)

## 6. Test surface

Test **through the interface**: inject a recording telemetry adapter, run
`search`, assert the expected events crossed the seam. No inspection of
`MctsBridge` internals — the interface is the test surface.

## 7. Risk / backpressure — why this PR stops here

- Whether retry should call `ix` (and *which* ix skill) is an **architectural
  choice**, not a mechanical edit — it needs product/architecture judgment.
- Reimplementing backoff in TARS would duplicate ix (high-severity row 12) — a
  one-way-ish coupling decision that must not happen AFK.
- Folding telemetry IO into core runtime (row 11) would degrade the `MctsBridge`
  seam; the no-op-adapter approach is the minimum that avoids that.

**Decision: stop after the telemetry seam. Do not add retry in this PR.**

## 8. Review gate

The retry-routing decision (call ix vs. local backoff, and where telemetry is
emitted) is escalated to a **human reviewer / Demerzel tribunal** before any
broader change. Per `AGENTS.md`, the producing agent cannot self-certify an
architectural decision; a fresh reviewer must approve the seam direction.

---

### AFK note

`governance/state/afk-halt.json` was checked and is inactive. No runtime `.fs`
file was edited by the skill itself — this report was recorded as a PR note, and
the single approved seam (if accepted) would land as its own human-reviewed PR.
