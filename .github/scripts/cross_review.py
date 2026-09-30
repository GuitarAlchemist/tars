#!/usr/bin/env python3
"""Combine the reviewers' votes on a pull request into one verdict.

Each reviewer votes on a commit:
  blocking      at least one P0 or P1 finding
  to-fix        P2 or P3 findings only
  clean         no findings
  not-reviewed  no vote on this commit

A reviewer with no vote on the head commit is "not-reviewed". It is never
counted as clean, so a reviewer that stayed silent cannot make a PR look
approved.

Votes are read from:
  Codex   its PR reviews (inline comments carry P0-P3 badges), its
          "Didn't find any major issues" comment, and the "Completed" rows
          of its summary comment, which name the commit
  Claude  a comment whose first line is "Cross-review vote: <vote> @ <sha>"
          (see claude-code-review.yml). claude-code-action may post it with
          the Claude app or with the workflow's token, so both accounts are
          accepted. Only this repo's workflows can post as github-actions[bot].
  TARS    a comment whose first line is "Cross-review vote (TARS): <vote> @ <sha>"
          (see tars-review.yml). It is advisory: shown, and counted by
          `report`, but left out of the verdict and the label until its
          findings have been checked against the other reviewers'.

Usage:
  cross_review.py verdict <pr>            print the verdict for the head commit
  cross_review.py post <pr>               post it on the PR, and label the PR while the reviewers disagree
  cross_review.py report <first> <last>   count each reviewer's findings over a range of PRs
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime

REPO = os.environ.get("GITHUB_REPOSITORY", "GuitarAlchemist/tars")
CODEX = "chatgpt-codex-connector[bot]"
VOTE_ACCOUNTS = ("claude[bot]", "github-actions[bot]")
REVIEWERS = ("Codex", "Claude")
ADVISORY = ("TARS",)
MARKER = "<!-- cross-review -->"
DISAGREE_LABEL = "reviewers-disagree"
ORDER = ["not-reviewed", "clean", "to-fix", "blocking"]

BADGE = re.compile(r"\bP([0-3]) Badge")
CODEX_CLEAN = re.compile(r"Didn't find any major issues.*?Reviewed commit:\*\*\s*`([0-9a-f]{7,40})`", re.S)
# A row of Codex's summary table. A review that finishes with no findings
# leaves only this row (and a thumbs-up), not a comment.
CODEX_DONE = re.compile(r"\*\*Completed\*\* <relative-time datetime=\"([^\"]+)\">.*?\|\s*`([0-9a-f]{7,40})`")
VOTE = re.compile(r"Cross-review vote(?: \((TARS)\))?: (blocking|to-fix|clean|not-reviewed) @ `?([0-9a-f]{7,40})")
# The path runs up to the first ":<line>", so paths with spaces parse too.
FINDING = re.compile(r"^- \[P([0-3])\] `?(.+?)`?:(\d+)", re.M)


def gh(*args):
    out = subprocess.run(["gh", "api", *args], check=True, capture_output=True, text=True, encoding="utf-8").stdout
    return json.loads(out) if out.strip() else None


def gh_list(path):
    """Every item of a list endpoint, across all pages: a vote past the first
    page must not read as a reviewer that never voted."""
    return [item for page in gh("--paginate", "--slurp", path) for item in page]


def same_commit(a, b):
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def when(timestamp):
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))


def vote_for(severities):
    if any(p <= 1 for p in severities):
        return "blocking"
    return "to-fix" if severities else "clean"


def collect(pr):
    """Every vote cast on the PR, oldest first, as (reviewer, sha, vote, findings)."""
    reviews = gh_list(f"repos/{REPO}/pulls/{pr}/reviews?per_page=100")
    inline = gh_list(f"repos/{REPO}/pulls/{pr}/comments?per_page=100")
    comments = gh_list(f"repos/{REPO}/issues/{pr}/comments?per_page=100")
    votes = []
    for r in reviews:
        if r["user"]["login"] != CODEX:
            continue
        findings = []
        for c in inline:
            m = BADGE.search(c["body"])
            if c["pull_request_review_id"] == r["id"] and m:
                findings.append((int(m.group(1)), c["path"], c.get("original_line") or c.get("line") or 0))
        if findings:
            votes.append((r["submitted_at"], "Codex", r["commit_id"], vote_for([f[0] for f in findings]), findings))
    with_findings = [(v[2], when(v[0])) for v in votes]
    for c in comments:
        login = c["user"]["login"]
        if login == CODEX:
            m = CODEX_CLEAN.search(c["body"])
            if m:
                votes.append((c["created_at"], "Codex", m.group(1), "clean", []))
            # A Completed row is the run that posted a review with findings when
            # both are on the same commit within two minutes. Any other
            # Completed row, such as a later rerun on that commit, is clean.
            for at, sha in CODEX_DONE.findall(c["body"]):
                same_run = any(
                    same_commit(sha, s) and abs((when(at) - t).total_seconds()) <= 120 for s, t in with_findings
                )
                if not same_run:
                    votes.append((at, "Codex", sha, "clean", []))
        elif login in VOTE_ACCOUNTS:
            m = VOTE.match(c["body"].lstrip())
            if m:
                tag, stated, sha = m.groups()
                findings = [(int(p), path, int(line)) for p, path, line in FINDING.findall(c["body"])]
                # A "clean" header over a P1 finding counts as blocking: the
                # stricter of the stated vote and the findings wins.
                implied = vote_for([f[0] for f in findings]) if findings else stated
                vote = max(stated, implied, key=ORDER.index)
                votes.append((c["created_at"], tag or "Claude", sha, vote, findings))
    votes.sort(key=lambda v: when(v[0]))
    return [v[1:] for v in votes]


def verdict(pr):
    head = gh(f"repos/{REPO}/pulls/{pr}")["head"]["sha"]
    votes = collect(pr)
    rows = []
    for reviewer in REVIEWERS + ADVISORY:
        mine = [v for v in votes if v[0] == reviewer]
        here = [v for v in mine if same_commit(v[1], head)]
        if here:
            rows.append((reviewer, here[-1][2], here[-1][3], None))
        else:
            rows.append((reviewer, "not-reviewed", [], mine[-1] if mine else None))
    # Advisory votes are shown but do not count.
    counted = [row for row in rows if row[0] in REVIEWERS]
    cast = [vote for _, vote, _, _ in counted if vote != "not-reviewed"]
    worst = max((vote for _, vote, _, _ in counted), key=ORDER.index)
    if worst in ("blocking", "to-fix"):
        overall = worst
    elif len(cast) == len(counted):
        overall = "clean"
    else:
        overall = "incomplete"
    disagree = "clean" in cast and any(v != "clean" for v in cast)
    return head, overall, rows, disagree


def render(head, overall, rows, disagree):
    lines = [
        MARKER,
        f"### Cross-review @ `{head[:7]}`",
        "",
        f"**Verdict: {overall}.** The strictest vote wins. A reviewer with no vote on this commit is not counted as clean.",
        "",
        "| Reviewer | Vote on this commit | Findings |",
        "|---|---|---|",
    ]
    for reviewer, vote, findings, last in rows:
        shown = vote
        if last:
            shown += f" (last vote: {last[2]} @ `{last[1][:7]}`)"
        found = ", ".join(f"P{p} `{path}:{line}`" for p, path, line in findings)
        name = f"{reviewer} (advisory)" if reviewer in ADVISORY else reviewer
        lines.append(f"| {name} | {shown} | {found or ('none' if vote == 'clean' else '-')} |")
    if disagree:
        lines += ["", f"The reviewers disagree, so a human should decide (label `{DISAGREE_LABEL}`)."]
    return "\n".join(lines)


def post(pr):
    head, overall, rows, disagree = verdict(pr)
    body = render(head, overall, rows, disagree)
    mine = [
        c for c in gh_list(f"repos/{REPO}/issues/{pr}/comments?per_page=100")
        if c["user"]["login"] == "github-actions[bot]" and MARKER in c["body"]
    ]
    if mine:
        gh("-X", "PATCH", f"repos/{REPO}/issues/comments/{mine[-1]['id']}", "-f", f"body={body}")
    else:
        gh("-X", "POST", f"repos/{REPO}/issues/{pr}/comments", "-f", f"body={body}")
    # The label follows the current head commit: it goes once the reviewers agree.
    labelled = any(label["name"] == DISAGREE_LABEL for label in gh_list(f"repos/{REPO}/issues/{pr}/labels?per_page=100"))
    if disagree and not labelled:
        gh("-X", "POST", f"repos/{REPO}/issues/{pr}/labels", "-f", f"labels[]={DISAGREE_LABEL}")
    elif labelled and not disagree:
        gh("-X", "DELETE", f"repos/{REPO}/issues/{pr}/labels/{DISAGREE_LABEL}")
    print(body)


def near(a, b):
    return a[1] == b[1] and abs(a[2] - b[2]) <= 10


def report(first, last):
    everyone = REVIEWERS + ADVISORY
    print("| PR | Codex votes | Claude votes | TARS votes | Codex findings | Claude findings | TARS findings "
          "| Codex and Claude | TARS confirmed |")
    print("|---|---|---|---|---|---|---|---|---|")
    totals = {"Codex": 0, "Claude": 0, "TARS": 0, "both": 0, "confirmed": 0}
    for pr in range(first, last + 1):
        try:
            votes = collect(pr)
        except subprocess.CalledProcessError:
            continue  # an issue number, not a PR
        found = {r: [f for v in votes if v[0] == r for f in v[3]] for r in everyone}
        both = sum(1 for f in found["Codex"] if any(near(f, g) for g in found["Claude"]))
        # A TARS finding is confirmed when Codex or Claude found the same thing.
        others = found["Codex"] + found["Claude"]
        confirmed = sum(1 for f in found["TARS"] if any(near(f, g) for g in others))
        cast = {r: ", ".join(dict.fromkeys(f"{v[2]}@{v[1][:7]}" for v in votes if v[0] == r)) or "none" for r in everyone}
        print(f"| #{pr} | {cast['Codex']} | {cast['Claude']} | {cast['TARS']} | {len(found['Codex'])} "
              f"| {len(found['Claude'])} | {len(found['TARS'])} | {both} | {confirmed} |")
        for r in everyone:
            totals[r] += len(found[r])
        totals["both"] += both
        totals["confirmed"] += confirmed
    print()
    print(
        f"Findings: Codex {totals['Codex']}, Claude {totals['Claude']}, TARS {totals['TARS']}. "
        f"Seen by both Codex and Claude: {totals['both']}. TARS findings confirmed by Codex or Claude: "
        f"{totals['confirmed']} (same file, within 10 lines)."
    )


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["help"]
    if command == "verdict" and len(rest) == 1:
        print(render(*verdict(int(rest[0]))))
    elif command == "post" and len(rest) == 1:
        post(int(rest[0]))
    elif command == "report" and len(rest) == 2:
        report(int(rest[0]), int(rest[1]))
    else:
        sys.exit(__doc__)
