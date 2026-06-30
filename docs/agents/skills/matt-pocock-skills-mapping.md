# Matt Pocock Skills → TARS Mapping

This document records how the [`anti-ball-of-mud`](./anti-ball-of-mud.skill.md)
skillset ports the **spirit and workflow shape** of
[`mattpocock/skills`](https://github.com/mattpocock/skills) into TARS. It is a
TARS-native adaptation, **not** a verbatim copy of the upstream course/repo
content.

Reference skill:
[`improve-codebase-architecture/SKILL.md`](https://github.com/mattpocock/skills/blob/main/skills/engineering/improve-codebase-architecture/SKILL.md).

## Why adapt rather than copy

`mattpocock/skills` frames skills as small, adaptable, composable practices for
real engineering. It distinguishes:

- **user-invoked skills** — explicit orchestration commands a human runs;
- **model-invoked skills** — reusable discipline the agent reaches for when the
  task fits.

The upstream skills assume an interactive human at the keyboard (HTML reports
opened in a browser, a multi-turn grilling conversation). TARS needs the same
*discipline* in an **AFK / cloud-agent** setting, where the agent reads repo
files and an issue body, cannot open a browser, and must create **backpressure**
rather than ask unbounded follow-up questions. So we keep the vocabulary and the
"detect mud → name the seam → keep the PR small" shape, and we drop the
interactive/HTML machinery.

## Concept mapping

| Upstream (`mattpocock/skills`) | TARS adaptation | Notes |
| --- | --- | --- |
| `improve-codebase-architecture` — scan for deepening opportunities, present an HTML report, grill the chosen one | [`anti-ball-of-mud.skill.md`](./anti-ball-of-mud.skill.md) — detect entropy signals, emit an **eight-part text report**, propose **one** seam | No browser/HTML; output is repo-readable Markdown a cloud agent can write into a PR note. |
| `codebase-design` — shared vocabulary (module, interface, depth, seam, adapter, locality, leverage; deletion test) | [`codebase-design-vocabulary.md`](./codebase-design-vocabulary.md) | Same terms, re-grounded in TARS seams (`IxSkill`, `MctsBridge`, hermetic gate, `LlmFactory`). |
| `grilling` / `grill-with-docs` — structured questioning before committing to a design | The skill's **stop conditions** + **review gate** | AFK substitute for an open-ended interview: when the choice is architectural, **stop and escalate** to a human / Demerzel tribunal instead of grilling autonomously. |
| `domain-modeling` — keep shared project language current | Pointers to `CONTEXT.md` + `docs/adr/` | The skill *reads* the ubiquitous language and refuses to re-litigate ADRs; it does not silently mutate the domain model AFK. |
| `to-issues` — turn a plan into independently-grabbable vertical slices | The **micro-PR plan** ("smallest safe change", one seam, under the rewrite budget) | One reviewable slice, not a backlog of refactors. |
| user-invoked vs model-invoked distinction | `/anti-ball-of-mud` (user) vs `anti-ball-of-mud-guard` (model) | Both share one vocabulary and one output shape; they differ in trigger and latitude. |

## What we deliberately did **not** port

- The **HTML report + Mermaid/CDN visualisation** machinery — an AFK cloud agent
  has no browser and should not commit `architecture-review-*.html` artifacts.
- The **multi-turn grilling loop** as an autonomous behavior — replaced by a
  hard stop + escalation when the decision is architectural.
- Any **course/marketing content** — out of scope and a stated non-goal.
- A **full architecture-governance framework** — TARS already has Demerzel and
  the Agent Blackbox policy; this skill *routes to* them, it does not replace them.

## Provenance / non-goals (from issue #135)

- Do **not** reproduce Matt Pocock course/repo content verbatim.
- Do **not** implement a full architecture-governance framework.
- Do **not** rewrite TARS modules as part of this skillset.
- Do **not** turn this into a generic clean-code checklist.
- Do **not** bypass human review for architectural decisions.
- Keep it usable by **Codex / Ollama / local agents**, not Claude-only — which is
  why the deliverable is repo-readable Markdown + JSON examples, with no
  Claude-specific tooling required.
