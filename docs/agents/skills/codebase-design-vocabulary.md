# Codebase Design Vocabulary (TARS)

Shared vocabulary for designing **deep modules** in TARS. Adapted from
`mattpocock/skills` `codebase-design` for this repo's F# / WoT / promotion-pipeline
surfaces. The [`anti-ball-of-mud.skill.md`](./anti-ball-of-mud.skill.md) skill uses
these terms exactly — and so should you. Consistent language is the whole point;
don't substitute "component", "service", "API", or "boundary".

A deep module is **a lot of behaviour behind a small interface, placed at a clean
seam, testable through that interface.** The aim is leverage for callers, locality
for maintainers, and testability for everyone.

## Glossary

**Module** — anything with an interface and an implementation. Scale-agnostic: an
F# function, a module, a project under `v2/src/`, or a tier-spanning slice. In
TARS, `IxSkill` is a module; so is the whole promotion pipeline.
_Avoid_: unit, component, service.

**Interface** — everything a caller must know to use the module correctly: the
type signature, but also invariants, ordering constraints, error modes
(`Result<_,_>`), required configuration, and performance characteristics.
_Avoid_: API, signature (too narrow — those refer only to the type-level surface).

**Implementation** — what's inside a module. Distinct from **adapter**: a thing
can be a small adapter with a large implementation (a Postgres-backed
`PromotionIndex`) or a large adapter with a small implementation (an in-memory
fake used in tests).

**Depth** — leverage at the interface: how much behaviour a caller (or test) can
exercise per unit of interface they must learn. A module is **deep** when a large
amount of behaviour sits behind a small interface, **shallow** when the interface
is nearly as complex as the implementation. In TARS, `IxSkill` is deep: one
`run <skill>` call hides binary-vs-`cargo` resolution, provenance, and fallback.

**Seam** _(Michael Feathers)_ — a place where you can alter behaviour **without
editing in that place**; the *location* where a module's interface lives. Where
to put the seam is its own decision, distinct from what goes behind it. TARS seams
that already exist and have names in `CONTEXT.md`: **IxSkill** (TARS → ix),
**tars_bridge** (ix → TARS), **MctsBridge** (Rust MCTS ↔ F# fallback), the
**fitness gate** / **hermetic gate** (variant → accept/reject), and
`LlmFactory.create` (TARS → any LLM provider).
_Avoid_: boundary (overloaded with DDD's bounded context).

**Adapter** — a concrete thing that satisfies an interface at a seam. Describes
*role* (which slot it fills), not substance (what's inside). The Rust `ix` binary
and the F# `MctsSolver` are two adapters behind the same MCTS seam.

**Leverage** — what callers get from depth: more capability per unit of interface
learned. One implementation pays back across N call sites and M tests.

**Locality** — what maintainers get from depth: change, bugs, knowledge, and
verification concentrate in **one place** rather than spreading across callers.
Fix once, fixed everywhere. A pure function extracted only for testability, while
the real bug lives in the orchestration that calls it, has *bad* locality.

## Deep vs shallow

```
Deep module (good)              Shallow module (avoid)
┌──────────────────┐            ┌──────────────────────────────┐
│  Small Interface │            │        Large Interface       │
├──────────────────┤            ├──────────────────────────────┤
│                  │            │   Thin Implementation        │
│  Deep            │            │   (just passes through)      │
│  Implementation  │            └──────────────────────────────┘
│                  │
└──────────────────┘
```

When shaping an interface, ask: can I reduce the number of methods? Simplify the
parameters? Hide more complexity inside?

## Principles

- **Depth is a property of the interface, not the implementation.** A deep module
  may be internally composed of small, swappable parts — they just aren't part of
  its interface.
- **The deletion test.** Imagine deleting the module. If complexity vanishes, it
  was a pass-through (a shallow wrapper — delete it). If complexity reappears
  across N callers, it was earning its keep.
- **The interface is the test surface.** Callers and tests cross the *same* seam.
  If you need to test *past* the interface, the module is the wrong shape. (TARS:
  the hermetic gate tests a variant through `dotnet test`, not by inspecting
  internals.)
- **One adapter = a hypothetical seam. Two adapters = a real one.** Don't
  introduce a seam unless something actually varies across it. MctsBridge's seam
  is real: it has two adapters (Rust `ix`, F# fallback).

## Designing for testability

1. **Accept dependencies, don't create them.** Pass the `ILlmService` /
   `IChatClient` in; don't `new` a provider inside. (TARS rule: obtain it from
   `LlmFactory.create(logger)`.)
2. **Return results, don't produce side effects.** Prefer a function that returns
   a `Selection.Performance` over one that mutates shared state.
3. **Small surface area.** Fewer methods = fewer tests; fewer params = simpler
   setup.

## Rejected framings

- **Depth as a ratio of implementation-lines to interface-lines** — rewards
  padding the implementation. Use depth-as-leverage instead.
- **"Interface" as the F#/TypeScript `interface` keyword or a type's public
  members** — too narrow; interface here includes every fact a caller must know.
- **"Boundary"** — overloaded with DDD's bounded context. Say **seam** or
  **interface**.
