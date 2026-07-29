# tars Governance State

Belief state persistence directory for Demerzel governance integration.

## Contents

- `beliefs/` — Tetravalent belief states (*.belief.json)
- `pdca/` — PDCA cycle tracking (*.pdca.json)
- `knowledge/` — Knowledge transfer records (*.knowledge.json)
- `snapshots/` — Belief snapshots for reconnaissance (*.snapshot.json)
- `afk-halt.json` — **kill switch for the AFK agent-delegation loop** (absent =
  running). When present + unexpired, `.github/workflows/jules-auto-delegate.yml`
  pauses delegating `ready-for-agent` issues to Jules. The cloud-reachable
  equivalent of Demerzel's `~/.demerzel/HALT-ALL`. Schema + halt/resume procedure
  in `docs/adr/0004-afk-agent-delegation-loop.md`.

## File Naming

`{date}-{short-description}.{type}.json`

## Schema Reference

See `governance/demerzel/schemas/` and `governance/demerzel/logic/` for JSON schemas.
