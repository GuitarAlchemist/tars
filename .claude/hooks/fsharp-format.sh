#!/bin/bash
# Optional post-edit Fantomas formatter. No-op if fantomas isn't installed.
# Contract: silent on success; exit 0 always.
#
# Bypass: set CLAUDE_DISABLE_FSHARP_FORMAT=1 to skip.

set -u

if [[ -n "${CLAUDE_DISABLE_FSHARP_FORMAT:-}" ]]; then
    exit 0
fi

TOOL_NAME="${CLAUDE_TOOL_NAME:-}"
FILE_PATH="${CLAUDE_FILE_PATH:-}"

# Fallback: read from stdin JSON payload if env vars unset
if [[ -z "$FILE_PATH" ]] && [[ ! -t 0 ]]; then
    PAYLOAD=$(cat 2>/dev/null || true)
    if [[ -n "$PAYLOAD" ]]; then
        FILE_PATH=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        if [[ -z "$TOOL_NAME" ]]; then
            TOOL_NAME=$(printf '%s' "$PAYLOAD" | sed -n 's/.*"tool_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
        fi
    fi
fi

if [[ -n "$TOOL_NAME" ]] && [[ ! "$TOOL_NAME" =~ ^(Write|Edit|MultiEdit)$ ]]; then
    exit 0
fi

if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Format only F# source files (not .fsproj / .fsx)
case "$FILE_PATH" in
    *.fs|*.fsi) ;;
    *) exit 0 ;;
esac

# Skip legacy / generated trees
case "$FILE_PATH" in
    */v1/*|*/v2/*|*/parked_legacy/*|*/autonomous_backups/*|*/.tars/*)
        exit 0 ;;
esac

# Check if fantomas is available (global or local manifest). If not, no-op.
if ! command -v dotnet >/dev/null 2>&1; then
    exit 0
fi

if ! dotnet tool list -g 2>/dev/null | grep -qi '^fantomas[[:space:]]'; then
    # Not installed globally; check local manifest too
    if ! dotnet tool list 2>/dev/null | grep -qi '^fantomas[[:space:]]'; then
        exit 0
    fi
fi

# 15s cap
timeout 15 dotnet fantomas "$FILE_PATH" >/dev/null 2>&1 || true

exit 0
