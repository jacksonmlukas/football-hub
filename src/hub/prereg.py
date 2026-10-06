"""Which scored predictions were pre-registered: the first commit with the price predates kickoff.

`docs/track-record.md` rule 1: *the git history is the pre-registration*. Until #422 nothing read
it and `publish.track_record` wrote the literal 0, so 64 predictions committed before kickoff by
the scheduled slate were labelled a backtest, and #354's "first artifact with
`n_preregistered > 0`" had no producer.

**The unit is the price, not the game.** A game's row is re-published on every refresh, and
the scored price is the one the file carries now. So a prediction is pre-registered when the
first commit whose `preds_*.json` held the same `(game_id, home_win_prob, priced_at)` was
committed before that game's kickoff. A re-priced game is a new prediction with a new first
commit; the earlier price does not vouch for it.

**Unreadable history is unknown, never 0.** `read_history` returns None for a shallow clone,
a directory outside a repository, or a failing git, and the caller publishes `n_preregistered:
null`. A 0 that means "could not tell" is the defect this module exists to remove. Both
workflows that publish the record check out with `fetch-depth: 0` for the same reason: the
default depth of 1 holds one commit and would make every run unknown.

Commit time is the *committer* date. The slate rebases before it pushes, which only moves that
date later, so the error is on the side of not counting. It is still a clock the committer
sets: the proof is the push, which git does not record, and the page claims no more than the
commit.
"""
from __future__ import annotations

import json
import subprocess
from collections.abc import Hashable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

Key = tuple[Hashable, Hashable, Hashable]


class Commit(NamedTuple):
    """One commit that touched a predictions artifact, with the rows it held (naive UTC)."""
    at: datetime
    rows: list[dict[str, Any]]


class Registration(NamedTuple):
    """The outcome for one set of scored predictions.

    `late` are scored games whose first commit came after kickoff; `unverified` are scored
    games the history cannot place (no commit ever carried that price, or no kickoff is known).
    Neither counts, and both are named rather than folded into a subtraction.
    """
    n_preregistered: int
    late: list[str]
    unverified: list[str]


def key_of(row: dict[str, Any]) -> Key:
    return (row.get("game_id"), row.get("home_win_prob"), row.get("priced_at"))


def first_seen(history: Sequence[Commit]) -> dict[Key, datetime]:
    """The earliest commit time at which each priced prediction appeared in any artifact."""
    seen: dict[Key, datetime] = {}
    for commit in history:
        for row in commit.rows:
            k = key_of(row)
            if k not in seen or commit.at < seen[k]:
                seen[k] = commit.at
    return seen


def classify(scored: Sequence[dict[str, Any]], seen: dict[Key, datetime]) -> Registration:
    """Count the scored rows whose first commit predates their kickoff.

    Each scored row carries `kickoff` (naive UTC, or None). Strictly earlier counts; a commit
    at the kickoff instant does not.
    """
    n = 0
    late: list[str] = []
    unverified: list[str] = []
    for row in scored:
        committed = seen.get(key_of(row))
        kickoff = row.get("kickoff")
        if committed is None or kickoff is None:
            unverified.append(str(row["game_id"]))
        elif committed < kickoff:
            n += 1
        else:
            late.append(str(row["game_id"]))
    return Registration(n, sorted(late), sorted(unverified))


def _git(cwd: Path, *args: str) -> str:
    got = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True,
                         check=True, timeout=120)
    return got.stdout


def read_history(out: Path) -> list[Commit] | None:
    """Every commit's view of the predictions artifacts under `out`, oldest first.

    None when it cannot be read *in full*: not a repository, git missing or failing, or a
    shallow clone, whose history stops at an arbitrary commit and would call everything older
    "uncommitted". A commit that deleted a file is skipped; its rows are in the commit before.
    """
    try:
        if _git(out, "rev-parse", "--is-shallow-repository").strip() != "false":
            return None
        listing = _git(out, "log", "--reverse", "--relative", "--name-only",
                       "--format=%x00%H %ct", "--", ":(glob)preds_*.json")
        history: list[Commit] = []
        for block in listing.split("\x00")[1:]:
            head, *names = block.split("\n")
            sha, stamp = head.split()
            at = datetime.fromtimestamp(int(stamp), UTC).replace(tzinfo=None)
            rows: list[dict[str, Any]] = []
            for name in filter(None, names):
                try:
                    body = _git(out, "show", f"{sha}:./{name}")
                except subprocess.CalledProcessError:
                    continue  # deleted or renamed away in this commit
                try:
                    rows += json.loads(body).get("rows", [])
                except ValueError:
                    continue  # a malformed artifact in one commit proves nothing about it
            history.append(Commit(at, rows))
        return history
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
