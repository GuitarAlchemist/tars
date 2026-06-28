# Skill: Docs-Only Contract

## When to use it
Use this skill when the assigned issue explicitly requires updating, creating, or refactoring documentation, Markdown files, architectural decision records (ADRs), JSON schemas, or other non-executable artifacts. It is intended for changes that must not affect runtime behavior.

## Allowed files/surfaces
- `docs/**/*.md`
- `examples/**/*.json`
- `*.md` at the repository root
- `docs/contracts/**/*.md`
- Do **not** modify `v2/**/*.fs`, `v2/**/*.cs`, or any other source code files.

## Required checks
- Ensure changes are structurally valid (e.g., proper JSON syntax, correct Markdown formatting).
- Verify that links and references to other documentation or code are accurate.
- If modifying contracts or schemas, confirm they match any examples provided in the issue.

## Stop conditions
- Halt immediately if you find yourself modifying executable code, build scripts, or CI/CD pipelines.
- Halt if you are requested to perform actions outside the allowed file surfaces.
- Halt if the `governance/state/afk-halt.json` marker is present.

## Evidence expected in the PR body
- Summary of the documentation changes made.
- Confirmation that no runtime code was modified.
- Links to relevant issues and references.

## Common failure modes
- Accidentally including unrelated source code changes in the PR.
- Introducing malformed JSON or broken Markdown links.
- Straying beyond the scope of the assigned issue.
