# ADR 0004 — AFK agent-delegation loop (issue → Jules → PR), governed by Demerzel

- **Status:** Accepted — 2026-06-24
- **Context source:** Session wiring autonomous issue delegation to external
  coding agents (Codex / Jules) and bringing the resulting loop under governance.
  Relates to ADR 0002/0003 (self-hosting improvement) — this is the *external*
  agent loop, complementary to TARS's internal self-improvement loop.
- **Question:** How should TARS delegate work to an external autonomous coding
  agent from a GitHub issue, AFK (no human in the loop at trigger time), without
  the new loop escaping Demerzel's ecosystem-wide control?

## Context

Both vendor GitHub Apps are installed on `GuitarAlchemist`: **Jules**
(`google-labs-jules`, Gemini) triggered by the `jules` label, and **Codex**
(`chatgpt-codex-connector`, OpenAI) triggered by an `@codex` comment. Jules is
the chosen path (works; Codex is quota-limited). Verified end-to-end #55 → #56.

The AFK loop:

```
maintainer triages an issue → adds `ready-for-agent`
   → .github/workflows/jules-auto-delegate.yml adds `jules`
   → Jules works in cloud → opens a PR
   → PR must pass the required `build` check → human reviews + merges
```

Two governance facts shaped the design:

1. **Output is already governed.** `qa-verdict-dispatch.yml` dispatches every
   `pull_request` to Demerzel's qa-tribunal, so Jules' PRs already receive a
   governance verdict automatically. No new wiring needed for output review.
2. **The *trigger* was not under Demerzel's kill switch.** Demerzel's loop
   control is the local marker `~/.demerzel/HALT-ALL`, read at Step 0 by
   `/auto-optimize` consumers running on a dev machine. The AFK loop runs in
   GitHub's cloud (Actions + Jules cloud) and never sees that local file — so an
   ecosystem-wide HALT would **not** stop AFK delegation. By Demerzel's own
   constitution (reversibility, escalation, observability), an un-haltable
   autonomous loop is exactly what governance must prevent.

## Decision

- **Trigger seam:** `ready-for-agent` (triage label) → workflow adds `jules`.
  Decoupling the triage label from the agent trigger keeps the human triage
  vocabulary stable and centralizes the governance gate in one workflow.
- **Review gate:** Jules only *opens* PRs; merge requires the `build` status
  check (branch protection on `main`) plus a human. Review-gating holds by
  default — nothing reaches `main` unattended.
- **Governance kill switch (cloud-reachable):** the workflow reads a
  repo-committed marker `governance/state/afk-halt.json` before delegating. This
  is the AFK loop's equivalent of `~/.demerzel/HALT-ALL`, reachable from CI
  because it lives in git. Present + unexpired ⇒ delegation is paused (the
  `ready-for-agent` label is kept and the issue gets an explanatory comment);
  absent ⇒ the loop runs. A git-committed marker is chosen over a GitHub repo
  variable for **auditability** — halt/resume is visible in history, matching
  Demerzel's git-native style.

### Marker schema (`governance/state/afk-halt.json`)

Mirrors the Demerzel HALT-ALL marker. All fields optional except by convention:

```json
{
  "schema_version": "0.1",
  "halted_at": "2026-06-24T12:00:00Z",
  "halted_by": "Demerzel",
  "reason": "Investigating agent cost burn",
  "expires_at": "2026-06-25T12:00:00Z"
}
```

- `expires_at` (RFC3339, optional): once past, the workflow treats the marker as
  inactive and proceeds. Absent ⇒ indefinite halt until the file is removed.
- The marker's **absence** is the running state; it is intentionally **not**
  committed by default.

### Halt / resume procedure (Demerzel or operator)

- **Halt:** commit `governance/state/afk-halt.json` to `main` (optionally with
  `reason` / `expires_at`). Subsequent `ready-for-agent` labels are parked.
- **Resume:** delete the marker from `main`. Re-apply `ready-for-agent` to any
  issue parked during the halt to delegate it (the label was retained).

## Consequences

- **Positive:** the new autonomous loop is now both *governed at the output*
  (tribunal) and *haltable at the trigger* (marker), closing the kill-switch
  gap. Halt/resume is auditable in git.
- **Limitation:** resume does not auto-replay parked issues — re-applying the
  label is a manual (or future-automated) step. Acceptable for v1.
- **Security:** label application is restricted to write/triage collaborators, so
  the trigger is gated even on this public repo. The `@codex` comment path is
  *not* label-gated (anyone can comment) and is out of scope here.
- **Follow-up:** a resume-replay mechanism, and bringing the `@codex` comment
  path under the same marker if it is adopted for AFK use.
