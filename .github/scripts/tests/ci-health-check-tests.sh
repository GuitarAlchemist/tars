#!/usr/bin/env bash
# ci-health-check-tests.sh: Regression tests for ci-health-check.sh
# Mocks the 'gh' command to verify job-specific conclusion logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CI_HEALTH_CHECK="$SCRIPT_DIR/../ci-health-check.sh"

# Mock 'gh' command
gh() {
  case "$*" in
    "run list"*)
      echo "$MOCK_RUN_LIST"
      ;;
    "run view"*)
      echo "$MOCK_RUN_VIEW"
      ;;
    *)
      echo "Unknown gh command: $*" >&2
      exit 1
      ;;
  esac
}
export -f gh

# Test runner
run_test() {
  local name="$1"
  local expected_conclusion="$2"

  echo "Running test: $name"

  # Run the script and capture output
  # We use 'bash -c' to ensure the mocked 'gh' function is used
  output=$(export REPO="mock/repo"; export MOCK_RUN_LIST; export MOCK_RUN_VIEW; bash "$CI_HEALTH_CHECK")

  actual_conclusion=$(echo "$output" | grep "conclusion=" | cut -d'=' -f2 | tr -d "'")

  if [ "$actual_conclusion" == "$expected_conclusion" ]; then
    echo "  ✅ Pass"
  else
    echo "  ❌ Fail: expected '$expected_conclusion', got '$actual_conclusion'"
    echo "  Full output:"
    echo "$output"
    exit 1
  fi
}

# --- Case 1: build=success, build-selfhosted=cancelled -> healthy ---
MOCK_RUN_LIST='[{"databaseId": 123, "headSha": "abcdef123456", "url": "https://github.com/run/123", "conclusion": "cancelled"}]'
MOCK_RUN_VIEW='{"jobs": [{"name": "build", "conclusion": "success"}, {"name": "build-selfhosted", "conclusion": "cancelled"}]}'
run_test "build=success, build-selfhosted=cancelled" "success"

# --- Case 2: build=cancelled, build-selfhosted=success -> unhealthy ---
MOCK_RUN_LIST='[{"databaseId": 124, "headSha": "abcdef124456", "url": "https://github.com/run/124", "conclusion": "cancelled"}]'
MOCK_RUN_VIEW='{"jobs": [{"name": "build", "conclusion": "cancelled"}, {"name": "build-selfhosted", "conclusion": "success"}]}'
run_test "build=cancelled, build-selfhosted=success" "cancelled"

# --- Case 3: build=failure, build-selfhosted=success -> unhealthy ---
MOCK_RUN_LIST='[{"databaseId": 125, "headSha": "abcdef125456", "url": "https://github.com/run/125", "conclusion": "failure"}]'
MOCK_RUN_VIEW='{"jobs": [{"name": "build", "conclusion": "failure"}, {"name": "build-selfhosted", "conclusion": "success"}]}'
run_test "build=failure, build-selfhosted=success" "failure"

# --- Case 4: build job not found -> fallback to workflow conclusion ---
MOCK_RUN_LIST='[{"databaseId": 126, "headSha": "abcdef126456", "url": "https://github.com/run/126", "conclusion": "failure"}]'
MOCK_RUN_VIEW='{"jobs": [{"name": "some-other-job", "conclusion": "success"}]}'
run_test "build job not found" "failure"

echo "All tests passed!"
