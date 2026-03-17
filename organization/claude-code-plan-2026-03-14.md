# Claude Code Execution Plan

Prepared: 2026-03-14
Primary context: `C:\Users\spare\source\repos\tars\organization\claude-3pm-handoff-2026-03-14.md`

## Objective

Use Claude Code for a bounded architecture session that does three things in the correct order:

1. lock the repo split
2. define the `MachinDeOuf` to `ix` migration plan
3. bootstrap the separate governance repo

Claude should not improvise beyond that.

## Non-Negotiable Decisions

- `MachinDeOuf` is the current practical `IX` machine-forge
- personas / constitutions / alignment / tetravalent logic belong in a separate governance repo
- TARS v1 chats are extraction sources, not direct persona files
- Claude must include a rename plan for moving `MachinDeOuf` toward `ix`

## Work Order For Claude

### Phase 1. Read and restate the architecture

Claude should first read:

- `C:\Users\spare\source\repos\tars\organization\claude-3pm-handoff-2026-03-14.md`

Then Claude should produce a short confirmation of:

- repo responsibilities for `ga`, `tars`, and `MachinDeOuf` / `ix`
- why governance is separate
- what is explicitly out of scope

This is a checkpoint, not a brainstorming session.

### Phase 2. Produce the rename and restructuring plan for `MachinDeOuf`

Claude should not blindly rename files first.

Claude should produce a staged migration plan covering:

- repo rename from `MachinDeOuf` to `ix`
- README and badge updates
- folder / remote assumptions
- crate naming strategy
- CLI rename strategy
- MCP server rename strategy
- Claude skill / agent / hook rename strategy
- search-and-replace risk areas
- downstream impact on `ga` and `tars`

Claude must explicitly answer these naming questions:

- should crates move from `machin-*` to `ix-*` immediately or through aliases first?
- should the CLI remain compatible with old command names for a transition period?
- should MCP tool names preserve compatibility temporarily?

Expected output from Phase 2:

- migration stages
- impacted artifact inventory
- compatibility strategy
- rollback strategy if rename churn gets too large

### Phase 3. Define the target `ix` shape

Claude should define the target shape for the machine repo after rename:

```text
ix/
├─ README.md
├─ Cargo.toml
├─ rust-toolchain.toml
├─ crates/
│  ├─ ix-core/
│  ├─ ix-search/
│  ├─ ix-graph/
│  ├─ ix-grammar/
│  ├─ ix-eval/
│  └─ ix-cli/
├─ skills/
├─ agents/
├─ hooks/
├─ schemas/
├─ examples/
│  ├─ ga/
│  └─ tars/
└─ docs/
```

Claude should then map current `machin-*` crates into:

- keep
- merge
- rename later
- leave out of first milestone

Claude should not try to refactor all 27+ crates in the same pass.

### Phase 4. Bootstrap the governance repo

Claude should bootstrap the separate repo, codename `Demerzel`, with only the minimum coherent set:

- `README.md`
- `docs/architecture.md`
- `sources/tars-v1-chats/README.md`
- `sources/tars-v1-chats/extracted-archetypes.md`
- `personas/*.persona.yaml`
- `constitutions/default.constitution.md`
- `policies/*.yaml`
- `logic/tetravalent-logic.md`
- `logic/tetravalent-state.schema.json`
- `schemas/persona.schema.json`
- `tests/behavioral/contradiction-cases.md`
- `examples/claude-code/minimal-agent-config.md`

Constraints:

- no runtime
- no unnecessary framework
- no raw chat dump as final persona assets
- keep artifacts reusable across repos

### Phase 5. Finish with the next-smallest executable step

Claude should end with:

- files created
- files proposed but not yet created
- unresolved decisions
- the next smallest iteration for `ix`
- the next smallest iteration for `Demerzel`

## Prompt To Paste Into Claude Code

```md
Use these two files as the source of truth:

1. C:\Users\spare\source\repos\tars\organization\claude-3pm-handoff-2026-03-14.md
2. C:\Users\spare\source\repos\tars\organization\claude-code-plan-2026-03-14.md

Task:
Execute the plan in order.

Rules:
1. Do not start by freeform brainstorming.
2. First restate the architecture decisions briefly.
3. Then produce the staged rename plan for moving `MachinDeOuf` toward `ix`.
4. Then define the first-milestone target shape for `ix`.
5. Then bootstrap the separate governance repo (`Demerzel`) with the minimum coherent artifact set.
6. Treat TARS v1 chats as extraction sources, not direct persona artifacts.
7. Do not build a runtime.
8. Keep outputs concrete: files, schemas, inventories, migration stages, and next actions.

Important:
- include the repo rename to `ix` in the plan
- do not collapse governance into `tars` or `MachinDeOuf`
- keep the first milestone small enough to finish cleanly

Finish with:
- what you created
- what you intentionally deferred
- migration risks
- the next recommended step
```
