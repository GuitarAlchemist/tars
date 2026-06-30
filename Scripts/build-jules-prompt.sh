#!/bin/bash
# Scripts/build-jules-prompt.sh
# Constructs the Jules cloud-agent prompt from GitHub issue data,
# including repo-specific skill documentation if declared.

set -e

ISSUE_NUMBER="$1"
REPO="$2"

if [ -z "$ISSUE_NUMBER" ] || [ -z "$REPO" ]; then
  echo "Usage: $0 <issue_number> <repo>"
  exit 1
fi

# Fetch issue data via GH CLI
ISSUE_DATA=$(gh issue view "$ISSUE_NUMBER" --repo "$REPO" --json title,body,labels)
TITLE=$(echo "$ISSUE_DATA" | jq -r '.title')
BODY=$(echo "$ISSUE_DATA" | jq -r '.body')

# 1. Extract skills from labels (skill:*)
SKILLS_FROM_LABELS=$(echo "$ISSUE_DATA" | jq -r '.labels[].name' | grep '^skill:' | sed 's/^skill://' || true)

# 2. Extract skills from body (agent_skills: YAML list)
# Robust extraction using awk to find the block starting with agent_skills:
# and capturing all subsequent lines starting with "  - ".
SKILLS_FROM_BODY=$(echo "$BODY" | awk '
  /^agent_skills:/ {found=1; next}
  found && /^  - / {print substr($0, 5); next}
  found && /^[^ ]/ {found=0}
' || true)

# Combine and deduplicate
ALL_SKILLS=$(echo -e "${SKILLS_FROM_LABELS}\n${SKILLS_FROM_BODY}" | sed '/^$/d' | sort -u)

# 3. Validate against allowlist (docs/agents/skills/*.skill.md)
VALID_SKILLS=""
if [ -n "$ALL_SKILLS" ]; then
  for skill in $ALL_SKILLS; do
    # Strict alphanumeric + hyphen check (prevents path traversal)
    if [[ "$skill" =~ ^[a-z0-9-]+$ ]]; then
      SKILL_FILE="docs/agents/skills/${skill}.skill.md"
      if [ -f "$SKILL_FILE" ]; then
        VALID_SKILLS="${VALID_SKILLS}${skill}\n"
      fi
    fi
  done
fi
VALID_SKILLS=$(echo -e "$VALID_SKILLS" | sed '/^$/d')

# 4. Build the final prompt
echo "Implement GitHub issue #${ISSUE_NUMBER}: ${TITLE}"
echo ""
echo "$BODY"
echo ""

if [ -n "$VALID_SKILLS" ]; then
  echo "The following repo-specific skills are declared for this task:"
  for skill in $VALID_SKILLS; do
    SKILL_FILE="docs/agents/skills/${skill}.skill.md"
    echo "--- SKILL: $skill ---"
    cat "$SKILL_FILE"
    echo ""
  done
  echo ""
fi

echo "Project conventions (this repo is TARS, an F# system; the working directory is v2/, not the repo root):"
echo "- Follow CLAUDE.md and CONTEXT.md; read any relevant docs/adr/ before coding."
echo "- F# functional-first: immutable types, Result<> for errors."
echo "- LLM access always goes through LlmFactory.create(logger) — never instantiate DefaultLlmService directly."
echo "- Warnings are errors (TreatWarningsAsErrors); code must compile clean."
echo "- Tests live in tests/Tars.Tests/ (xUnit). Build and test from v2/: dotnet build && dotnet test."
echo ""
echo "Open a pull request targeting main with the change. Keep it minimal and scoped to the issue. Ensure the build and tests pass before opening the PR."
