# Skill: Docs-Only Contract

## When to use
Use this skill when the GitHub issue's `issue_meta` explicitly states `reason: "documentation-only..."`, or when the task solely involves creating, updating, or correcting documentation, diagrams, or examples without altering runtime functionality.

## Allowed files/surfaces
- `*.md` files (e.g., `README.md`, `CLAUDE.md`, `CONTEXT.md`, files in `docs/`).
- JSON/YAML files only if they are explicitly marked as examples (e.g., `examples/**/*.json`) or non-runtime configuration.
- **Forbidden:** Any F# source files (`*.fs`, `*.fsproj`), Rust source files (`*.rs`), or CI/CD workflow definitions (`.github/workflows/*.yml`) unless explicitly requested as part of a doc update.

## Required checks
1. **Scope Check:** Verify that no runtime logic or system configuration is modified.
2. **Formatting Check:** Ensure standard Markdown formatting, broken link avoidance, and compliance with project terminology defined in `CONTEXT.md`.
3. **Build Check:** Ensure that any example code blocks provided in the documentation are syntactically valid. Run `dotnet build` from `v2/` to ensure you haven't accidentally broken the build.

## Stop conditions
- Stop when the required documentation files have been created or modified as specified in the issue.
- Stop immediately if you find yourself needing to modify an F# (`*.fs`) file to complete the documentation.

## Evidence expected in the PR body
- A checklist of the documentation files modified.
- A statement confirming that no runtime code was altered.
- Links to the rendered documentation (if applicable).

## Common failure modes
- Accidentally editing a `.fs` file to fix a typo discovered during documentation writing, thus violating the "docs-only" constraint.
- Failing to build and test the repo, assuming that documentation changes cannot break the build.
- Missing updates to `CONTEXT.md` or `CLAUDE.md` if new terminology is introduced.
