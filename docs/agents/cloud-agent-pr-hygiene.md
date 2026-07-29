# Cloud-Agent PR Hygiene Policy

This document defines a lightweight PR hygiene policy for cloud-agent output. It helps reviewers and future sweeper workflows classify PRs quickly.

## Policy Dimensions

### 1. Same Issue, Multiple PRs
* **Rule**: When a cloud agent opens multiple PRs for the same issue, the most recent PR should be reviewed if it is complete. If older PRs contain valuable partial work, consider asking the agent to consolidate them. Otherwise, close the older duplicate PRs immediately.
* **Detection**: Check if there are other open PRs linking to the same parent issue.

### 2. Docs-only Issue Touches Code/Test Files
* **Rule**: If an issue is explicitly scoped to documentation (e.g., creating or updating markdown files) but the PR modifies source code (`.fs`, `.cs`, `.rs`, `.js`) or test files, the PR should be flagged as out-of-scope.
* **Action**: Reject the code/test changes. Ask the agent to revise the PR to include only the docs changes, or close the PR and request a fresh attempt.

### 3. Workflow Issue Touches Runtime Files
* **Rule**: Issues scoped to CI/CD workflows (`.github/workflows/`, `.gitlab-ci.yml`, etc.) must not modify runtime code (files in `src/`, `lib/`, `v2/src/`).
* **Action**: If a workflow PR modifies runtime files, ask the agent to revert those changes before merging.

### 4. Stale Base Branch
* **Rule**: If a PR is significantly behind the base branch (e.g., `main`) and has merge conflicts, do not attempt to resolve them manually if it's complex.
* **Action**: Ask the agent to rebase or recreate the PR against the latest base branch. If it cannot, close the PR and re-delegate the issue.

### 5. CI Green but Governance Concern
* **Rule**: A PR that passes CI but violates governance policies (e.g., modifying `AGENTS.md` incorrectly, violating the TARS second-brain provenance rules, or bypassing Demerzel approvals) must be blocked.
* **Action**: Reviewers must point out the governance failure and request the agent to revise, citing the specific policy.

### 6. Duplicate Close Message
* **Rule**: When closing a duplicate PR, provide a clear, standard message indicating why it was closed and which PR supersedes it.
* **Template**: `"Closing this PR as a duplicate. A newer/more complete PR for this issue exists here: #<PR_NUMBER>. Please focus efforts there."`

### 7. Merge Priority for Contract Spine PRs
* **Rule**: PRs that modify "contract spine" files (core abstractions, API definitions, DSL schemas like `.wot.trsx`) have high merge priority. They block other dependent agent tasks.
* **Action**: Fast-track the review of contract spine PRs. Ensure strict scrutiny, but minimize wait time in the review queue.

### 8. When to Ask Agent to Revise vs. Close Duplicate
* **Revise**: Ask the agent to revise if the PR is mostly correct but has minor out-of-scope edits (e.g., touched a test file unnecessarily) or failed a specific governance check that is easily fixable.
* **Close Duplicate**: Close the PR immediately if it is a complete duplicate of another active or recently merged PR for the same issue, or if the approach is fundamentally flawed and starting over is cleaner.
