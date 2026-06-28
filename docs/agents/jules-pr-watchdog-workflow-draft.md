# Jules PR Watchdog Workflow Draft

This draft describes the workflow that should later be copied into `.github/workflows/jules-pr-watchdog.yml` after review.

## Events

- PR metadata changed
- PR review submitted
- PR conversation comment changed
- CI workflow completed
- scheduled fallback every 30 minutes
- manual dispatch

## Permissions

- read repository contents
- read pull requests
- read Actions status
- write labels and comments on issues/PRs

## Candidate selection

1. If the event is tied to a PR, inspect that PR.
2. On schedule, list open PRs whose body contains the Jules footer.
3. Skip PRs that do not contain the Jules footer.

## State labels

The workflow owns these labels:

- `jules-waiting`
- `jules-updated`
- `needs-human-review`
- `needs-agent-revision`
- `ready-to-merge`
- `stale-jules-pr`
- `duplicate-agent-pr`

Before applying a new state, remove the other state labels.

## Classification rules

| Rule | New state |
| --- | --- |
| Known requested fix still missing | `needs-agent-revision` |
| CI/checks failed or cancelled | `needs-agent-revision` |
| PR has changed since last observed state | `jules-updated` |
| Checks complete and PR looks mergeable | `ready-to-merge` |
| Checks complete but review still needed | `needs-human-review` |
| No PR movement for stale window | `stale-jules-pr` |
| PR superseded by another PR for same issue | `duplicate-agent-pr` |

## Known targeted check for #115/#129

Detect workflow code that looks for:

```text
docs/agents/skills/name.md
```

when it should prefer:

```text
docs/agents/skills/name.skill.md
```

If this mismatch is present, use `needs-agent-revision`.

## Comment policy

Comment only when the state label changes.

Comment shape:

```text
Jules PR watchdog: state changed from <old> to <new>. Reason: <reason>.
```

No comment is emitted for unchanged state.

## Safe boundaries

The watchdog must not:

- merge PRs;
- approve PRs;
- change PR branch contents;
- start new Jules tasks;
- run high-frequency polling;
- replace maintainer/Demerzel review.
