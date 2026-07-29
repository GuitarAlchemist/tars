#!/bin/bash
# Post-edit F# check — runs `dotnet build` on the containing .fsproj after
# F# file edits. Provides a fast local verification gate so type errors are
# caught before they reach CI.
#
# Pattern: ix/.claude/hooks/rust-check.sh (Pattern 3, Cherny self-improvement loop).
# Contract: silent on success; exit 0 always (PostToolUse should report, not block).
#
# Bypass: set CLAUDE_DISABLE_FSHARP_CHECK=1 to skip.

set -u

# Hard kill-switch
if [[ -n "${CLAUDE_DISABLE_FSHARP_CHECK:-}" ]]; then
    exit 0
fi

TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
FILE_PATH="${CLAUDE_FILE_PATH:-}"

# Fallback: read from stdin JSON payload if env vars unset
if [[ -z "$FILE_PATH" ]] && [[ ! -t 0 ]]; then
    PAYLOAD=$(cat 2>/dev/null || true)
    if [[ -n "$PAYLOAD" ]]; then
        # Best-effort extraction without requiring jq
        FILE_PATH=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        if [[ -z "$TOOL_NAME" ]]; then
            TOOL_NAME=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        fi
    fi
fi

# Only act on Write/Edit if tool name is provided; otherwise allow direct invocation
if [[ -n "$TOOL_NAME" ]] && [[ ! "$TOOL_NAME" =~ ^(Write|Edit|MultiEdit)$ ]]; then
    exit 0
fi

# Need a file path
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Only F# files (skip .fsx scripts per spec)
case "$FILE_PATH" in
    *.fs|*.fsi|*.fsproj) ;;
    *) exit 0 ;;
esac

# Skip legacy / generated trees (per CLAUDE.md cleanup notes)
case "$FILE_PATH" in
    */v1/*|*/parked_legacy/*|*/autonomous_backups/*|*/.tars/*)
        exit 0 ;;
esac

# Resolve project to build
PROJ=""
if [[ "$FILE_PATH" == *.fsproj ]]; then
    PROJ="$FILE_PATH"
else
    # Walk up from the file's directory looking for a .fsproj
    DIR=$(dirname "$FILE_PATH")
    while [[ -n "$DIR" && "$DIR" != "." && "$DIR" != "/" ]]; do
        CAND=$(ls "$DIR"/*.fsproj 2>/dev/null | head -1)
        if [[ -n "$CAND" ]]; then
            PROJ="$CAND"
            break
        fi
        PARENT=$(dirname "$DIR")
        # Stop if we can't ascend further
        if [[ "$PARENT" == "$DIR" ]]; then
            break
        fi
        DIR="$PARENT"
    done
fi

if [[ -z "$PROJ" ]]; then
    exit 0
fi

# Make sure dotnet is on PATH; otherwise gracefully degrade
if ! command -v dotnet >/dev/null 2>&1; then
    exit 0
fi

# 30-second cap; if dotnet hangs, kill and exit 0 (goal is fast feedback)
OUTPUT=$(timeout 30 dotnet build --no-restore -nologo -clp:NoSummary -v:q "$PROJ" 2>&1)
EXIT_CODE=$?

# 124 == timeout reached; treat as non-blocking
if [[ $EXIT_CODE -eq 124 ]]; then
    exit 0
fi

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "[fsharp-check] dotnet build failed for $(basename "$PROJ"):" >&2
    # Surface just the error/warning lines, keep context tight
    printf '%s\n' "$OUTPUT" | grep -E "(error|warning) [A-Z]+[0-9]+" | head -15 >&2
fi

# Never block — PostToolUse reports, never blocks
exit 0
