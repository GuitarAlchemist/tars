# tars `.claude/hooks/`

Local Claude Code hooks: session-digest continuity, the `/correct` nudge, PR
rationale capture, and fast F# verification after agent edits. Mirrors
**ix**'s hooks (Pattern 3 — Cherny self-improvement loops).

Every hook is one Go program, [`go/claudehooks`](../../go/claudehooks)
(`tars-hooks <hook>`, Go standard library only). No hook runs a shell or a
script interpreter — no bash, sh, PowerShell, Python or Node (the user's rule:
"jamais de shell dans les hooks"). It is ix's `go/claudehooks` (ix PR #512)
for tars: the digest, correction and PR-rationale hooks come from the same
family.

## How it runs

- **SessionStart** runs `go -C "$CLAUDE_PROJECT_DIR/go/claudehooks" run ./cmd/tars-hooks session-start`.
  It installs the built program as `.claude/hooks/bin/tars-hooks[.exe]`
  (gitignored; replaced only when its content changed), then injects the digest.
  A session therefore always starts with the hooks built from the checked-out source.
- **Every other hook** calls `"$CLAUDE_PROJECT_DIR/.claude/hooks/bin/tars-hooks" <hook>`
  directly: no compile, no interpreter start-up on each prompt or tool call.
- After editing `go/claudehooks`, start a new session (or run the SessionStart
  command above) so the installed binary picks up the change.

## Hooks

| Subcommand | Phase | Matcher | Purpose |
| ---------- | ----- | ------- | ------- |
| `session-start` | SessionStart | — | Installs `.claude/hooks/bin/tars-hooks`, then runs `sessionstart-digest`. |
| `sessionstart-digest` | (via `session-start`) | — | Prints `state/digests/latest.md` for the model's context; one line instead when it is more than 24 h old. |
| `precompact-digest` | PreCompact | — | Archives `latest.md` under the session id; writes a metadata-only fallback when `/digest` has not run in the last 30 min. |
| `digest-staleness-nudge` | UserPromptSubmit | — | `additionalContext` nudge when the digest is missing, or old and the session has drifted. |
| `correction-detector` | UserPromptSubmit | — | Nudges `/correct` when the prompt opens with correction language; throttled to the 1st and every `TARS_CORRECTION_THROTTLE`-th (5) match. |
| `digest-activity-tracker` | PostToolUse | `Edit\|Write\|Bash` | Counts mutations; writes a mid-session stub digest after `TARS_DIGEST_MID_COUNT` (20) of them, or `TARS_DIGEST_MID_MIN` (30) minutes with at least 3. |
| `pr-rationale-capture` | PostToolUse | `Bash` | After `gh pr create`, saves title, body and diff stats to `state/digests/pr-<num>-<slug>.md`. |
| `fsharp-check` | PostToolUse | `Write\|Edit` | Walks up from the edited `*.fs` / `*.fsi` / `*.fsproj` to the containing `.fsproj` and runs `dotnet build --no-restore -nologo -clp:NoSummary -v:q` with a 30 s cap. Emits the error/warning lines to stderr on failure so the agent sees them on the next turn. |
| `fsharp-format` | PostToolUse | `Write\|Edit` | Opt-in Fantomas formatter (15 s cap). No-op when `fantomas` isn't in `dotnet tool list` (global or local). |
| `digest-stop-finalize` | Stop | — | Writes a metadata digest when `/digest` has not run in the last 10 min, archiving the previous one. |
| `digest-validate` | (manual, `/digest` step 7) | — | Checks a digest's frontmatter against `docs/contracts/digest-schema.json`'s required fields; exit 1 with the reason when it does not conform. |

## Contract

- **Silent on success.** Emit only when something the agent should see has gone wrong.
- **Never block.** No hook exits 2; PostToolUse hooks report, they don't gate.
  An unknown subcommand exits 1 with a usage line.
- **Fast.** The prompt and tool-call hooks start no process unless they have
  work to do (the F# hooks start `dotnet` only for an F# edit). `dotnet` runs
  are capped (30 s build, 15 s format); a hung run exits 0 silently.
- **Graceful degradation.** Missing `dotnet`, missing `fantomas`, missing project file, outside a repository → exit 0.
- **Skip noise.** Files under `v1/`, `parked_legacy/`, `autonomous_backups/`, `.tars/` are ignored (legacy / generated trees per `CLAUDE.md`). `v2/` is NOT skipped — it is the active codebase (`cd v2` per `CLAUDE.md`).
- **Skip scripts.** `.fsx` files are eval'd separately, not built.
- **Windows paths.** As the bash script did, `fsharp-check` walks to the
  project along forward slashes only: an edit reported with a backslash path
  starts no build (one heavy build at a time on the Windows machine).

## Bypass

Per-hook env vars (set in the shell before launching `claude`):

| Variable | Effect |
| -------- | ------ |
| `CLAUDE_DISABLE_FSHARP_CHECK=1` | Skip `fsharp-check` entirely. |
| `CLAUDE_DISABLE_FSHARP_FORMAT=1` | Skip `fsharp-format` entirely. |

Use sparingly — bypassing the check defeats the purpose of the gate. Prefer
fixing the underlying issue.

## Tests

```bash
cd go/claudehooks
GOPROXY=off go vet ./... && GOPROXY=off go test ./...
```

CI runs the same in `.github/workflows/karpathy-cherny-discipline.yml`.

## Pattern reference

Both ix and tars apply the same shape:

1. After every `Write` / `Edit`, run the language-native fast check.
2. On failure, surface the diagnostic on stderr so the model sees it on the
   next turn and self-corrects.
3. Keep CI green by catching issues at the keystroke, not on push.

ix uses `cargo check -p <crate>` (Rust, `ix-hooks rust-check`). tars uses
`dotnet build <project>` (F# / .NET). Same pattern, different toolchain.
