---
name: supervised-loop
description: Run a supervised autonomous loop on tars under explicit edit-scope and review-independence guardrails. Each cycle: preflight, propose, verify, halt-or-continue. Author and reviewer roles stay separate (producer-reviewer); the evaluator runs in a fresh sub-agent / fresh session and cannot self-certify. Contested cycles escalate to cross-vendor / multi-LLM review (Demerzel tribunal). Edits respect a line budget (rewrite-budget) so each diff stays small.
---

# supervised-loop (tars)

Repo-local skill for running an autonomous improvement loop inside tars
without violating Agent Blackbox boundaries. Pairs with:

- `agent-blackbox.loop-policy.json` — `allow_edit` / `protected_paths`
- `agent-blackbox.policy.json` — risk thresholds and blocked paths
- `Scripts/supervised-loop-preflight.ps1` — fast pre-cycle gate
- `state/quality/tars-harness/baseline.json` — primary baseline + oracle
- `state/governance/dev-process-overseer.json` — workflow-mode snapshot

## Per-cycle protocol

1. **Preflight.** Run `pwsh Scripts/supervised-loop-preflight.ps1 -Json`.
   If `verdict != loop-eligible`, **stop**. Do not proceed.

2. **Read the baseline.** Open `state/quality/tars-harness/baseline.json`
   and the most recent `state/quality/tars-harness/last.json`. Treat the
   baseline as a one-way door: changes to the metric direction or the
   oracle command itself require an explicit sign-off PR, not a loop edit.

3. **Propose (producer).** Generate exactly one change inside the
   `allow_edit` set in `agent-blackbox.loop-policy.json`. Stay under the
   rewrite budget (≤ 200 changed lines per cycle by default; smaller is
   better). No edits to `protected_paths` — ever.

4. **Verify (oracle).** Run `pwsh Scripts/verify.ps1`. Capture the exit
   code into `state/quality/tars-harness/last.json` along with a metric
   value and oracle status. **The proposer must not edit `last.json`
   itself** — the verify script writes it.

5. **Review (fresh evaluator).** A *different* sub-agent / session reads
   the diff plus `dist/risk-report.json` (when running under the PR
   workflow) and either confirms the cycle or halts. The producer cannot
   self-certify. For contested cycles, escalate to cross-vendor review
   (Demerzel tribunal via `.github/workflows/qa-verdict-dispatch.yml`,
   or a Gemini / Codex peer review).

6. **Halt-or-continue.** If verify failed, or the reviewer halted, or a
   halt marker appeared (`~/.demerzel/HALT-ALL`, repo `.STOP`), stop the
   loop and surface findings. Otherwise commit + repeat.

## Review independence (matches AGENTS.md)

- **Producer-reviewer separation** — author of a cycle is not its reviewer.
- **Fresh evaluator** — reviewer runs in a fresh session / different
  context window and cannot self-certify.
- **Cross-vendor review** — contested PRs go through a multi-LLM /
  multi-model panel (Gemini, Codex peer, Demerzel tribunal).
- **Rewrite budget** — each cycle stays under the line budget so blast
  radius is bounded.

## What this skill is NOT

- Not a way to bypass `agent-blackbox.policy.json` blocked paths.
- Not a way to edit `.github/workflows/**`, `governance/demerzel/**`,
  `**/*.sln`, `**/*.fsproj`, or any other protected path.
- Not a substitute for the PR workflow's `enforce --report` gate.
