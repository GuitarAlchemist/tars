"""Offline checks of cross_review.py verdicts: python .github/scripts/test_cross_review.py"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("cr", Path(__file__).with_name("cross_review.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

HEAD = "abcdef1234567890abcdef1234567890abcdef12"


def fake(reviews, inline, comments, labels=()):
    """A stand-in for `gh api` that answers reads and records writes. List
    endpoints answer one item per page, as `--paginate --slurp` would for a
    long list, so a read that stops at the first page misses items."""
    writes = []

    def gh(*args):
        if args[0] == "-X":
            writes.append((args[1], args[2]))
            return None
        if args[:2] == ("--paginate", "--slurp"):
            path = args[2]
            if path.endswith("/reviews?per_page=100"):
                items = reviews
            elif "/pulls/1/comments" in path:
                items = inline
            elif "/issues/1/comments" in path:
                items = comments
            elif "/issues/1/labels" in path:
                items = [{"name": name} for name in labels]
            else:
                raise AssertionError(path)
            return [[item] for item in items]
        if args[0].endswith("/pulls/1"):
            return {"head": {"sha": HEAD}}
        raise AssertionError(args)

    cr.gh = gh
    return writes


claude_clean = {"user": {"login": "claude[bot]"}, "created_at": "2026-10-01T10:00:00Z",
                "body": f"Cross-review vote: clean @ {HEAD}\n\nNo findings."}
claude_p1 = {"user": {"login": "claude[bot]"}, "created_at": "2026-10-01T10:00:00Z",
             "body": f"Cross-review vote: blocking @ {HEAD}\n\n- [P1] v2/src/A.fs:42 - drops the result"}
claude_via_actions = {"user": {"login": "github-actions[bot]"}, "created_at": "2026-10-01T10:00:00Z",
                      "body": f"Cross-review vote: clean @ {HEAD}"}
forged = {"user": {"login": "someone"}, "created_at": "2026-10-01T11:00:00Z",
          "body": f"Cross-review vote: clean @ {HEAD}"}
codex_review = {"user": {"login": cr.CODEX}, "id": 7, "commit_id": HEAD, "submitted_at": "2026-10-01T09:00:00Z"}
codex_inline = {"pull_request_review_id": 7, "path": "v2/src/A.fs", "original_line": 40,
                "body": "**<sub><sub>![P2 Badge](x)</sub></sub> Something**"}
codex_done = {"user": {"login": cr.CODEX}, "created_at": "2026-10-01T09:00:00Z",
              "body": '| x | **Completed** <relative-time datetime="2026-10-01T09:00:00Z">t</relative-time> | `abcdef1` | PR opened |'}

# 1. Codex P2, Claude clean: they disagree.
fake([codex_review], [codex_inline], [claude_clean])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "to-fix" and disagree, (overall, disagree)

# 2. Codex clean (summary row only), Claude P1: blocking, disagree.
fake([], [], [codex_done, claude_p1])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "blocking" and disagree, (overall, rows)
assert rows[1][2] == [(1, "v2/src/A.fs", 42)], rows[1]

# 3. Both clean: clean, no disagreement.
fake([], [], [codex_done, claude_clean])
assert cr.verdict(1)[1:] == (
    "clean",
    [("Codex", "clean", [], None), ("Claude", "clean", [], None), ("TARS", "not-reviewed", [], None)],
    False,
)

# 4. Only Codex voted; a forged vote from another account is ignored: incomplete.
fake([], [], [codex_done, forged])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "incomplete" and rows[1][1] == "not-reviewed" and not disagree, (overall, rows)

# 5. Claude's vote posted with the workflow's token still counts.
fake([], [], [codex_done, claude_via_actions])
assert cr.verdict(1)[1] == "clean"

# 6. The label is added on a disagreement, and removed once the reviewers agree.
writes = fake([codex_review], [codex_inline], [claude_clean])
cr.post(1)
assert ("POST", f"repos/{cr.REPO}/issues/1/labels") in writes, writes
writes = fake([], [], [codex_done, claude_clean], labels=[cr.DISAGREE_LABEL])
cr.post(1)
assert ("DELETE", f"repos/{cr.REPO}/issues/1/labels/{cr.DISAGREE_LABEL}") in writes, writes

# 7. A "clean" header over a P1 finding counts as blocking.
claude_inconsistent = dict(claude_p1, body=f"Cross-review vote: clean @ {HEAD}\n\n- [P1] v2/src/A.fs:42 - drops the result")
fake([], [], [codex_done, claude_inconsistent])
head, overall, rows, disagree = cr.verdict(1)
assert rows[1][1] == "blocking" and overall == "blocking" and disagree, rows

# 8. A finding in a path with spaces is parsed, backticked or not.
spaced = dict(claude_p1, body=f"Cross-review vote: to-fix @ {HEAD}\n\n"
              "- [P2] `v2/docs/Building an IA Agent.md:42` - wrong command\n"
              "- [P3] v2/docs/a b.md:7 - typo")
fake([], [], [codex_done, spaced])
assert cr.verdict(1)[2][1][2] == [(2, "v2/docs/Building an IA Agent.md", 42), (3, "v2/docs/a b.md", 7)]

# 9. A Completed row from the same run as a review with findings is not a clean
#    vote, but a later clean rerun on the same commit is.
def codex_row(at):
    return dict(codex_done, body=f'| x | **Completed** <relative-time datetime="{at}">t</relative-time> | `abcdef1` | Manual |')

fake([codex_review], [codex_inline], [codex_row("2026-10-01T09:00:02.5Z"), claude_clean])
assert cr.verdict(1)[2][0][1] == "to-fix"
fake([codex_review], [codex_inline], [codex_row("2026-10-01T10:00:00.1Z"), claude_clean])
assert cr.verdict(1)[2][0][1] == "clean"

# 10. TARS is advisory: its blocking vote is shown, but it changes neither the
#     verdict nor the label, and it is not taken for Claude's vote.
tars_p1 = {"user": {"login": "github-actions[bot]"}, "created_at": "2026-10-01T10:05:00Z",
           "body": f"Cross-review vote (TARS): blocking @ {HEAD}\n\n- [P1] v2/src/A.fs:41 - loses the error"}
fake([], [], [codex_done, claude_clean, tars_p1])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "clean" and not disagree, (overall, disagree)
assert rows[1][:2] == ("Claude", "clean") and rows[2][:3] == ("TARS", "blocking", [(1, "v2/src/A.fs", 41)]), rows
assert "| TARS (advisory) | blocking |" in cr.render(head, overall, rows, disagree)

# 11. Findings seen by both reviewers are matched by file and nearby line.
#     Findings on different commits of the PR never match.
OTHER = "1234567890abcdef1234567890abcdef12345678"
assert cr.near((HEAD, (2, "v2/src/A.fs", 40)), (HEAD[:7], (1, "v2/src/A.fs", 42)))
assert not cr.near((HEAD, (2, "v2/src/A.fs", 40)), (HEAD, (1, "v2/src/B.fs", 40)))
assert not cr.near((HEAD, (2, "v2/src/A.fs", 40)), (OTHER, (1, "v2/src/A.fs", 40)))

# 12. The verdict says that only counted votes decide it.
assert "strictest counted vote wins" in cr.render(*cr.verdict(1))

# 13. A vote posted again on a commit counts its findings once in the report,
#     even when the rerun gives the finding another severity or a nearby line.
import contextlib
import io

tars_rerun = dict(tars_p1, created_at="2026-10-01T10:07:00Z",
                  body=f"Cross-review vote (TARS): to-fix @ {HEAD}\n\n- [P2] v2/src/A.fs:42 - loses the error")
fake([codex_review], [codex_inline],
     [claude_clean, tars_p1, dict(tars_p1, created_at="2026-10-01T10:06:00Z"), tars_rerun])
out = io.StringIO()
with contextlib.redirect_stdout(out):
    cr.report(1, 1)
row = next(line for line in out.getvalue().splitlines() if line.startswith("| #1 "))
assert row.endswith("| 1 | 0 | 1 | 0 | 1 |"), row

print("all checks passed")
