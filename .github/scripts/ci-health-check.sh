#!/usr/bin/env bash
# ci-health-check.sh: Extract the status of the canonical 'build' job from a .NET workflow run.
#
# Usage:
#   export REPO="owner/repo"
#   ./ci-health-check.sh
#
# Outputs (stdout):
#   conclusion=<success|failure|cancelled|...>
#   sha=<short-sha>
#   url=<run-url>
#   id=<run-id>

set -euo pipefail

: "${REPO:?REPO environment variable is required}"

# 1. Fetch the latest completed .NET run on main.
# We include databaseId so we can look up its constituent jobs.
RUN_JSON=$(gh run list --repo "$REPO" --workflow ".NET" --branch main --status completed --limit 1 \
             --json databaseId,headSha,url,conclusion)
RUN_JSON=$(echo "$RUN_JSON" | jq -c '.[0] // empty')

if [ -z "$RUN_JSON" ]; then
  echo "No completed .NET runs found on main." >&2
  exit 0
fi

RUN_ID=$(echo "$RUN_JSON" | jq -r '.databaseId // empty')
SHA=$(echo "$RUN_JSON" | jq -r '.headSha // empty' | cut -c1-8)
URL=$(echo "$RUN_JSON" | jq -r '.url // empty')
WF_CONCLUSION=$(echo "$RUN_JSON" | jq -r '.conclusion // empty')

# 2. Fetch the jobs for this run to find the specific 'build' gate.
# This prevents 'build-selfhosted' cancellations from marking the whole gate as red.
JOBS_JSON=$(gh run view --repo "$REPO" "$RUN_ID" --json jobs)

# Extract conclusion of the job named exactly "build".
BUILD_CONCLUSION=$(echo "$JOBS_JSON" | jq -r '.jobs[] | select(.name == "build") | .conclusion // empty')

# Fallback: if 'build' job isn't found (e.g. workflow structure changed),
# use the overall workflow conclusion to be safe.
FINAL_CONCLUSION="${BUILD_CONCLUSION:-$WF_CONCLUSION}"

echo "conclusion=$FINAL_CONCLUSION"
echo "sha=$SHA"
echo "url=$URL"
echo "id=$RUN_ID"
