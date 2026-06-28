# Cloud-Agent PR Review Checklist

This checklist is meant for reviewers evaluating PRs submitted by cloud-agents in the TARS workflow. It ensures adherence to the Cloud-Agent PR Hygiene Policy.

## Pre-Review Checks

- [ ] **Duplicate Check:** Are there other PRs open for this same issue?
    - *If yes, determine which is most complete. Close others using the duplicate close message template.*
- [ ] **Scope Check:** Does the PR scope match the issue?
    - *Docs-only issue:* Does it touch any `.fs`, `.cs`, or test files? (If yes, ask to revise).
    - *Workflow issue:* Does it touch any runtime code in `v2/src/`? (If yes, ask to revise).
- [ ] **Staleness Check:** Is the PR cleanly mergeable against the base branch (`main`)?
    - *If there are conflicts, ask the agent to rebase.*

## Review Checks

- [ ] **CI Status:** Are all CI checks green?
- [ ] **Governance Check:** Even if CI is green, does the PR adhere to TARS governance rules?
    - e.g., No circumvention of Demerzel policies.
    - e.g., Correct use of `LlmFactory.create(logger)` instead of direct LLM instantiation.
- [ ] **Priority Check:** Does this PR modify a "contract spine" (core abstractions, DSLs)?
    - *If yes, prioritize this review.*

## Action Matrix

- **Approve & Merge:** PR passes all checks.
- **Request Changes (Revise):** PR is mostly correct but has small out-of-scope changes or minor governance violations.
- **Close & Request New:** PR is fundamentally flawed, massively out of date, or a clean duplicate.
