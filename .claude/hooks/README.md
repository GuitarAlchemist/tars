# tars `.claude/hooks/`

Local Claude Code hooks that wrap agent edits with fast verification gates.
Mirrors **ix**'s [`.claude/hooks/`](https://github.com/GuitarAlchemist/ix/tree/main/.claude/hooks)
pattern (Pattern 3 — Cherny self-improvement loops).

Goal: catch the kind of mistake the language compiler can spot in seconds,
locally, before it becomes a failed CI run.

## Hooks

| Hook | Phase | Matcher | Purpose |
| ---- | ----- | ------- | ------- |
| `fsharp-check.sh` | PostToolUse | `Write\|Edit` | Walks up from the edited `*.fs` / `*.fsi` / `*.fsproj` to the containing `.fsproj` and runs `dotnet build --no-restore -nologo -clp:NoSummary -v:q` with a 30s cap. Emits errors/warnings to stderr on failure so the agent sees the diagnostic on the next turn. |
| `fsharp-format.sh` | PostToolUse | `Write\|Edit` | Opt-in Fantomas formatter (15s cap). No-op when `fantomas` isn't on `dotnet tool list` — graceful degradation, no install required. |
| `digest-*.ps1` | various | various | Pre-existing session-digest hooks. Not part of this PR. |

## Contract

- **Silent on success.** Emit only when something the agent should see has gone wrong.
- **Exit non-zero only on real diagnostic content** (stderr lines that name the failing file).
- **Never block.** PostToolUse hooks are advisory by design; they report, they don't gate.
- **Fast.** Hard cap via `timeout`. If `dotnet` hangs, the hook exits 0 silently — we never block agent throughput on slow tools.
- **Graceful degradation.** Missing `dotnet`, missing `fantomas`, missing project file → exit 0.
- **Skip noise.** Files under `v1/`, `v2/`, `parked_legacy/`, `autonomous_backups/`, `.tars/` are ignored (legacy / generated trees per `CLAUDE.md`).
- **Skip scripts.** `.fsx` files are eval'd separately, not built.

## Bypass

Per-hook env vars (set in the shell before launching `claude`):

| Variable | Effect |
| -------- | ------ |
| `CLAUDE_DISABLE_FSHARP_CHECK=1` | Skip `fsharp-check.sh` entirely. |
| `CLAUDE_DISABLE_FSHARP_FORMAT=1` | Skip `fsharp-format.sh` entirely. |

Use sparingly — bypassing the check defeats the purpose of the gate. Prefer
fixing the underlying issue.

## Pattern reference

Both ix and tars apply the same shape:

1. After every `Write` / `Edit`, run the language-native fast check.
2. On failure, surface the diagnostic on stderr so the model sees it on the
   next turn and self-corrects.
3. Keep CI green by catching issues at the keystroke, not on push.

ix uses `cargo check -p <crate>` (Rust). tars uses `dotnet build <project>`
(F# / .NET). Same pattern, different toolchain.

See ix's `.claude/hooks/rust-check.sh` for the reference implementation.
