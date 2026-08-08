#!/usr/bin/env bash
# tools/agents/render-afk-board.sh - Generate the human-readable HTML and Markdown boards from JSON artifacts.

set -euo pipefail

# Ensure we run from the repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"

echo "=== Rendering AFK Live Agent Board ==="

# Define paths
GOV_LIVE_DIR="governance/agents/live"
RUNS_DIR="${GOV_LIVE_DIR}/runs"
HTML_OUT_DIR="docs/agents/live/html"
MD_OUT_FILE="docs/agents/live/afk-board.md"

# Ensure output directories exist
mkdir -p "${HTML_OUT_DIR}"

# Run Python compilation logic for robustness
python3 - << 'EOF'
import os
import json
from datetime import datetime, timezone

gov_runs_dir = "governance/agents/live/runs"
html_out_file = "docs/agents/live/html/afk-runs.json"
md_out_file = "docs/agents/live/afk-board.md"

compiled_runs = []

# 1. Gather runs from governance/agents/live/runs/*.json
if os.path.exists(gov_runs_dir):
    for filename in sorted(os.listdir(gov_runs_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(gov_runs_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    run_data = json.load(f)

                    # Normalize fields for unified schema
                    state = run_data.get("state", "queued")

                    # Synthesize flags based on state vocabulary
                    is_stale = state == "stale" or run_data.get("is_stale", False)
                    is_blocked = state == "blocked" or run_data.get("is_blocked", False)
                    is_duplicate = state in ["duplicate", "duplicate-agent-pr"] or run_data.get("is_duplicate", False)
                    ci_failing = state == "ci-failing" or run_data.get("ci_failing", False)
                    needs_human_review = state in ["needs-human-review", "human-review"] or run_data.get("needs_human_review", False)

                    # Map into unified array element
                    normalized_run = {
                        "issue": f"#{run_data.get('issue')}" if run_data.get("issue") and not str(run_data.get("issue")).startswith("#") else run_data.get("issue", ""),
                        "title": run_data.get("summary", "Untitled Agent Run"),
                        "agent": run_data.get("agent", "Unknown").capitalize(),
                        "pr": f"#{run_data.get('pr')}" if run_data.get("pr") and not str(run_data.get("pr")).startswith("#") else (run_data.get("pr") or ""),
                        "state": state,
                        "risk": run_data.get("risk", "low"),
                        "last_signal": run_data.get("last_signal_at", run_data.get("last_signal", "")),
                        "next_action": run_data.get("next_action", "None specified"),
                        "evidence": [
                            e if isinstance(e, str) else f"{e.get('type', 'link')}: {e.get('finding', 'Reference')}"
                            for e in run_data.get("evidence", [])
                        ],
                        "is_stale": is_stale,
                        "is_blocked": is_blocked,
                        "is_duplicate": is_duplicate,
                        "ci_failing": ci_failing,
                        "needs_human_review": needs_human_review
                    }
                    compiled_runs.append(normalized_run)
            except Exception as e:
                print(f"Warning: Failed to parse {filename}: {e}")

# 2. Write the compiled runs array into the HTML folder
try:
    with open(html_out_file, "w", encoding="utf-8") as f:
        json.dump(compiled_runs, f, indent=2)
    print(f"Successfully compiled {len(compiled_runs)} runs into {html_out_file}")
except Exception as e:
    print(f"Error: Failed to write compiled runs: {e}")

# 3. Generate the Markdown Board for GitHub review
try:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Calculate counters
    total_count = len(compiled_runs)
    review_count = sum(1 for r in compiled_runs if r.get("needs_human_review") or r.get("state") == "needs-human-review")
    blocked_count = sum(1 for r in compiled_runs if r.get("is_blocked") or r.get("state") == "blocked")
    failing_count = sum(1 for r in compiled_runs if r.get("ci_failing") or r.get("state") == "ci-failing")
    stale_count = sum(1 for r in compiled_runs if r.get("is_stale") or r.get("state") == "stale")
    done_count = sum(1 for r in compiled_runs if r.get("state") == "done")

    md_content = f"""# AFK Live Agent Board

*Last Updated:* {timestamp}

This is the human-readable Markdown view of the parallel cloud-agent tracking system. The static [HTML Dashboard](./html/index.html) is also available for live scanning.

## Summary Counts

| Metric | Count |
|--------|-------|
| **Total Active Runs** | {total_count} |
| **Needs Human Review** | {review_count} |
| **Blocked Runs** | {blocked_count} |
| **CI Failing Runs** | {failing_count} |
| **Stale Runs** | {stale_count} |
| **Completed (Done) Runs** | {done_count} |

## Active Runs

"""
    if compiled_runs:
        md_content += "| Issue / Task | Agent | PR | State | Risk | Last Signal | Next Action |\n"
        md_content += "|---|---|---|---|---|---|---\n"
        for run in compiled_runs:
            issue_str = run.get("issue", "N/A")
            title_str = run.get("title", "Untitled")
            agent_str = run.get("agent", "Unknown")
            pr_str = run.get("pr") or "None"
            state_str = run.get("state", "queued")
            risk_str = run.get("risk", "low").upper()
            signal_str = run.get("last_signal", "N/A")
            action_str = run.get("next_action", "N/A")

            # Add markers to State
            state_label = state_str
            if run.get("is_stale"):
                state_label += " ⚠️ STALE"
            if run.get("is_blocked"):
                state_label += " 🚫 BLOCKED"
            if run.get("is_duplicate"):
                state_label += " 📋 DUPLICATE"
            if run.get("ci_failing"):
                state_label += " ❌ FAILING"
            if run.get("needs_human_review"):
                state_label += " 👀 NEEDS REVIEW"

            md_content += f"| {issue_str} - {title_str} | {agent_str} | {pr_str} | `{state_label}` | {risk_str} | {signal_str} | {action_str} |\n"
    else:
        md_content += "*No active cloud agent work is currently registered.*\n"

    with open(md_out_file, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Successfully generated Markdown board in {md_out_file}")

except Exception as e:
    print(f"Error: Failed to write Markdown board: {e}")
EOF

echo "=== Rendering Completed ==="
