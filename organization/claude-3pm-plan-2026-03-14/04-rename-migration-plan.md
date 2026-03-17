# Staged Rename Plan: MachinDeOuf → ix

Produced: 2026-03-14, Phase 2 of execution plan.

## Impact Inventory

| Category | Count | Notes |
|----------|-------|-------|
| Crate directories | 31 | `crates/machin-*` → `crates/ix-*` |
| Cargo.toml name fields | 32 | Root + 31 crates |
| Internal dependencies | 36 | Cross-crate workspace refs |
| Source files with `use machin_*` | 154 | Import statements |
| Claude skills | 20 | `.claude/skills/machin-*` dirs + SKILL.md content |
| Config files | 3 | `.mcp.json`, `.claude/settings.local.json`, `.github/workflows/ci.yml` |
| Documentation files | 25+ | README, CLAUDE.md, docs/, plans, brainstorms |
| Example files | 10 | `examples/` with import statements |
| Binary names | 2 | `machin-mcp`, `machin` CLI |
| Total files touching "machin" | ~557 | Systematic but large |

## Naming Questions — Answers

### Should crates move from `machin-*` to `ix-*` immediately or through aliases?

**Answer: Immediate rename, no aliases.**

Rationale:
- No downstream consumers yet (not published to crates.io)
- No external code depends on `machin-*` crate names
- Aliases (re-exports, `[package] rename` hacks) add complexity for zero benefit
- Clean break is cheaper than maintaining two naming schemes

### Should the CLI remain compatible with old command names?

**Answer: No transition period needed.**

Rationale:
- CLI binary `machin` is not yet distributed or documented externally
- MCP server `machin-mcp` is only referenced in local `.mcp.json`
- New names: `ix` (CLI) and `ix-mcp` (MCP server)

### Should MCP tool names preserve compatibility temporarily?

**Answer: Rename tools immediately, with a migration note.**

Rationale:
- MCP tools are registered dynamically — tool names are strings in `tools.rs`
- No persistent state depends on old tool names
- Rename `machin_stats` → `ix_stats`, etc. in one pass
- Add a `MIGRATION.md` noting the rename for any users of the old names

## Migration Stages

### Stage 0: Pre-flight (before any rename)

- [ ] Create a `MIGRATION.md` in repo root documenting the rename
- [ ] Tag current state: `git tag v0.1.0-machin-final`
- [ ] Verify all tests pass: `cargo test --workspace`
- [ ] Ensure no uncommitted work

### Stage 1: GitHub repo rename

- [ ] Rename repo on GitHub: `MachinDeOuf` → `ix`
- [ ] GitHub auto-redirects old URL, but update local remote:
  ```bash
  git remote set-url origin https://github.com/GuitarAlchemist/ix.git
  ```
- [ ] Update CI badge URL in README
- [ ] Update any hardcoded GitHub URLs in docs

### Stage 2: Crate directory + Cargo.toml rename

This is the largest mechanical change. Execute with a script:

```bash
# Rename all crate directories
for dir in crates/machin-*; do
  new=$(echo "$dir" | sed 's/machin-/ix-/')
  mv "$dir" "$new"
done

# Update workspace Cargo.toml
sed -i 's/machin-/ix-/g' Cargo.toml
sed -i 's/machin_/ix_/g' Cargo.toml

# Update each crate's Cargo.toml
for toml in crates/ix-*/Cargo.toml; do
  sed -i 's/name = "machin-/name = "ix-/g' "$toml"
  sed -i 's/machin-/ix-/g' "$toml"
  sed -i 's/machin_/ix_/g' "$toml"
done
```

- [ ] Rename 31 directories
- [ ] Update root `Cargo.toml` workspace members and dependencies
- [ ] Update all 32 per-crate `Cargo.toml` files
- [ ] Update binary names: `machin-mcp` → `ix-mcp`, `machin` → `ix`
- [ ] Run `cargo check --workspace` — fix any missed references
- [ ] Commit: `refactor: rename crate directories machin-* → ix-*`

### Stage 3: Source code imports

```bash
# Bulk rename all Rust import paths
find crates/ -name "*.rs" -exec sed -i 's/machin_/ix_/g' {} +
find examples/ -name "*.rs" -exec sed -i 's/machin_/ix_/g' {} +
```

- [ ] Update 154+ source files: `use machin_math::` → `use ix_math::`, etc.
- [ ] Update `use machin_*` in all test modules
- [ ] Run `cargo build --workspace` — fix any remaining errors
- [ ] Run `cargo clippy --workspace -- -D warnings`
- [ ] Commit: `refactor: rename all machin_ imports to ix_`

### Stage 4: MCP + Skills + Config

- [ ] Rename `.claude/skills/machin-*` directories to `.claude/skills/ix-*`
- [ ] Update all SKILL.md content: `machin-*` → `ix-*`, `machin_*` → `ix_*`
- [ ] Update `.mcp.json`: server name and binary path
- [ ] Update `.claude/settings.local.json`: permission paths
- [ ] Rename MCP tool names in `tools.rs`: `machin_stats` → `ix_stats`, etc.
- [ ] Commit: `refactor: rename skills, MCP tools, and config to ix-*`

### Stage 5: Documentation

- [ ] Update `README.md`: project name, badge, crate table, CLI examples
- [ ] Update `CLAUDE.md`: all crate references, project description
- [ ] Update `docs/` markdown files
- [ ] Update plan/brainstorm files (historical — light touch, add note)
- [ ] Commit: `docs: update all documentation for ix rename`

### Stage 6: Verification + tag

- [ ] `cargo test --workspace` — all tests pass
- [ ] `cargo clippy --workspace -- -D warnings` — clean
- [ ] `cargo doc --workspace --no-deps` — builds
- [ ] Grep for any remaining `machin` references: `rg -i machin .`
- [ ] Tag: `git tag v0.2.0-ix`
- [ ] Push: `git push origin main --tags`

## Downstream Impact

### tars repo
- `.mcp.json` references `machin-agent` binary → update to `ix-mcp`
- Any scripts calling `machin` CLI → update to `ix`
- Low impact — tars currently has minimal direct integration

### ga repo
- Not currently consuming MachinDeOuf crates directly
- Future integration will use `ix-*` names from the start
- Zero immediate impact

## Rollback Strategy

If the rename creates unmanageable churn:

1. **Before push:** `git reset --hard v0.1.0-machin-final` restores everything
2. **After push:** GitHub repo redirect handles old URLs; revert commits if needed
3. **Partial rollback:** Since stages are committed separately, any stage can be reverted independently
4. **Risk is low:** No external consumers, no published crates, no production dependencies

## Search-and-Replace Risk Areas

| Risk | Mitigation |
|------|-----------|
| Partial replacement in strings (e.g., "machine" contains "machin") | Use word-boundary-aware sed: `\bmachin_` and `\bmachin-` |
| Binary names in shell scripts | Grep for `machin-mcp` and `machin ` (with space) specifically |
| Hardcoded paths with `MachinDeOuf` | Case-sensitive search, separate from `machin-*` rename |
| Cargo.lock inconsistency | Delete and regenerate: `rm Cargo.lock && cargo generate-lockfile` |
| Git history references | Leave as-is — history is immutable and that's fine |

## Estimated Effort

| Stage | Effort | Notes |
|-------|--------|-------|
| Stage 0 | 5 min | Tag + verify |
| Stage 1 | 10 min | GitHub rename + remote update |
| Stage 2 | 30 min | Directory + Cargo.toml (scriptable) |
| Stage 3 | 20 min | Source imports (scriptable + manual fixes) |
| Stage 4 | 15 min | Config + skills |
| Stage 5 | 20 min | Docs |
| Stage 6 | 10 min | Verify + tag |
| **Total** | **~2 hours** | Mostly mechanical, scriptable |
