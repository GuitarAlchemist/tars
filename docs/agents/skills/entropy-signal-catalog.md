# Architecture Entropy Signal Catalog (TARS)

Concrete signals the [`anti-ball-of-mud`](./anti-ball-of-mud.skill.md) skill scans
for. Each is a **specific** architectural smell, not generic untidiness. For each
signal: what to look for, a TARS-grounded example, the design-vocabulary term it
violates, and a default severity for AFK backpressure.

Severity drives the stop decision:

- **high** — stop AFK work and escalate; the fix needs a named seam and review.
- **medium** — note the friction, propose one seam, keep going only if the seam
  fits one reviewable PR.
- **low** — record it; safe to leave for a dedicated micro-PR.

| # | Signal | What to look for | TARS example | Violates | Severity |
| --- | --- | --- | --- | --- | --- |
| 1 | Mixed responsibilities in one module | A file that does orchestration **and** IO **and** ranking | An orchestrator that also opens sockets and scores patterns | locality | high |
| 2 | Bouncing across shallow modules | One concept spread over many tiny files, none deep | Following a request through 6 one-function modules to understand it | depth | medium |
| 3 | Interface ≈ implementation | A wrapper whose interface is as complex as its body | A "manager" that just forwards each call with the same params | depth / deletion test | medium |
| 4 | Feature logic mixed with IO/provider/UI | Business logic interleaved with side effects | Pattern-selection logic inline with HTTP calls | locality | high |
| 5 | Stringly-typed routing | Dispatch on a raw string where a typed contract belongs | Capability dispatch by `match capName with "x" -> …` instead of a DU | interface | medium |
| 6 | Ad-hoc JSON shapes | Repeated inline JSON with no schema/contract | New `{...}` payload shapes not backed by `docs/contracts/` or an `examples/*.json` | interface | medium |
| 7 | Shallow wrapper | A rename that adds no behaviour | A `FooService` that delegates 1:1 to `Foo` | deletion test | low |
| 8 | Hidden global mutable state | A `mutable`/static cache reached from many places | A module-level mutable dictionary mutated across call sites | locality | high |
| 9 | Seam leakage | Tightly-coupled modules reaching into each other's internals | A caller pattern-matching on another module's private representation | seam | high |
| 10 | Pure-for-testability, bug-in-orchestration | A pure helper extracted only to test, while the real bug is in the caller | Well-tested scoring fn, but the orchestration that feeds it is untested | locality | medium |
| 11 | Core depends directly on GitHub/LLM/FS/network | A provider SDK call wired straight into core runtime | A direct `DefaultLlmService` (instead of `LlmFactory.create(logger)`); a raw `gh`/HTTP call in core | seam / adapter | high |
| 12 | Duplicated scoring/analytics that belongs to ix | A copy-pasted algorithm reimplemented in TARS | Reimplementing `stats`/`bandit`/`grammar.weights` in F# instead of calling `IxSkill` | leverage | high |
| 13 | Scattered Demerzel governance `if`s | Governance rules expressed as local conditionals | `if risk > x then block` inline instead of routing through the governance gate | seam | high |
| 14 | Tests assert implementation details | Tests coupled to internals, not the interface | A test that inspects private state instead of crossing the module's seam | interface / test surface | medium |

## How to use the catalog

1. While inspecting the requested change, match each touched file against the
   table. **Cite the row number and the file** in the eight-part report's
   *Entropy finding* — a finding with no cited signal is not evidence.
2. Take the **highest-severity** signal as the primary finding; secondary signals
   can be listed but the report proposes **one** seam.
3. If the highest severity is **high**, the default is **stop and escalate** —
   the fix almost always needs architecture judgment and a review gate.

## Relation to other TARS policy

- Signals 11–13 overlap with `AGENTS.md` review-independence and the Agent
  Blackbox policy. When a high-severity signal coincides with a **one-way-door**
  or **blocked** path, the stop is mandatory.
- Signal 11 directly encodes the TARS convention "LLM access always via
  `LlmFactory.create(logger)`" from `CLAUDE.md`.
