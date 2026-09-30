"""Offline checks of cross_review.py verdicts: python .github/scripts/test_cross_review.py"""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("cr", Path(__file__).with_name("cross_review.py"))
cr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cr)

HEAD = "abcdef1234567890abcdef1234567890abcdef12"


def fake(reviews, inline, comments):
    def gh(path, *rest):
        if path.endswith("/reviews?per_page=100"):
            return reviews
        if "/pulls/1/comments" in path:
            return inline
        if "/issues/1/comments" in path:
            return comments
        if path.endswith("/pulls/1"):
            return {"head": {"sha": HEAD}}
        raise AssertionError(path)
    return gh


claude_clean = {"user": {"login": "claude[bot]"}, "created_at": "2026-10-01T10:00:00Z",
                "body": f"Cross-review vote: clean @ {HEAD}\n\nNo findings."}
claude_p1 = {"user": {"login": "claude[bot]"}, "created_at": "2026-10-01T10:00:00Z",
             "body": f"Cross-review vote: blocking @ {HEAD}\n\n- [P1] v2/src/A.fs:42 - drops the result"}
forged = {"user": {"login": "someone"}, "created_at": "2026-10-01T11:00:00Z",
          "body": f"Cross-review vote: clean @ {HEAD}"}
codex_review = {"user": {"login": cr.CODEX}, "id": 7, "commit_id": HEAD, "submitted_at": "2026-10-01T09:00:00Z"}
codex_inline = {"pull_request_review_id": 7, "path": "v2/src/A.fs", "original_line": 40,
                "body": "**<sub><sub>![P2 Badge](x)</sub></sub> Something**"}
codex_done = {"user": {"login": cr.CODEX}, "created_at": "2026-10-01T09:00:00Z",
              "body": '| x | **Completed** <relative-time datetime="2026-10-01T09:00:00Z">t</relative-time> | `abcdef1` | PR opened |'}

# 1. Codex P2, Claude clean: they disagree.
cr.gh = fake([codex_review], [codex_inline], [claude_clean])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "to-fix" and disagree, (overall, disagree)

# 2. Codex clean (summary row only), Claude P1: blocking, disagree.
cr.gh = fake([], [], [codex_done, claude_p1])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "blocking" and disagree, (overall, rows)
assert rows[1][2] == [(1, "v2/src/A.fs", 42)], rows[1]

# 3. Both clean: clean, no disagreement.
cr.gh = fake([], [], [codex_done, claude_clean])
assert cr.verdict(1)[1:] == ("clean", [("Codex", "clean", [], None), ("Claude", "clean", [], None)], False)

# 4. Only Codex voted; a forged vote from another account is ignored: incomplete.
cr.gh = fake([], [], [codex_done, forged])
head, overall, rows, disagree = cr.verdict(1)
assert overall == "incomplete" and rows[1][1] == "not-reviewed" and not disagree, (overall, rows)

# 5. Findings seen by both reviewers are matched by file and nearby line.
assert cr.near((2, "v2/src/A.fs", 40), (1, "v2/src/A.fs", 42))
assert not cr.near((2, "v2/src/A.fs", 40), (1, "v2/src/B.fs", 40))

print(cr.render(*cr.verdict(1)))
print("all checks passed")
