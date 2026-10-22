"""The starting-QB source: nfeloqb's per-team state, with the one timestamp that is knowable.

`qb_elos.csv` has no per-row update time, and `hub.fetch.nfeloqb`'s stamp records when *we*
pulled it. The candidates for "when did the source update", measured 2026-10-07:

* the committer date of the last commit that changed `qb_elos.csv` -- what is used. Written by
  the repository's own Action, a few seconds before the push; absent a forged commit it is
  an upper bound on when the content existed. It is the committer's claim, not GitHub's: git
  dates are whatever the committer sets, and the commits are unsigned.
* the pinned commit's own date -- NOT the same thing. The pin names a commit of the repository,
  and the repository gets commits that do not touch the file, so the pin's date can be later
  than the file's last change; it can also be *earlier* than the newest commit, so a pinned
  file is older than the source's current state however recent the pull.
* `Last-Modified` on raw.githubusercontent.com -- not sent.
* our fetch time -- not an update time, and never substituted for one.

`qb_source_at` looks the commit up (GitHub's commits API, `until=` run start) and reads the
file at that commit, not at `nfeloqb.COMMIT`; `qb_source_from_nfeloqb` takes a time handed in,
where `None` means unknown and is refused by the packet's check, not papered over.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl

from hub.fetch import cached, nfeloqb
from hub.forecasters.inputs import QbSource, SourceStamp

QB_TIME_BASIS = "github committer date of the last commit touching qb_elos.csv"


def qb_source_from_nfeloqb(state: pl.DataFrame, *, commit_time: datetime | None,
                           stamp: SourceStamp) -> QbSource:
    """`hub.fetch.nfeloqb.state`'s frame as a `QbSource`: team -> the starter on its latest row
    (the coming game's, where the source has listed it)."""
    starters = {row["team"]: row["qb"] for row in state.select("team", "qb").to_dicts()}
    return QbSource(stamp=stamp, updated_at=commit_time,
                    updated_at_basis=QB_TIME_BASIS, starters=starters)


# --- the latest commit before the run, not the pin ---------------------------------------
#
# Maintainer, 2026-10-06: the study reads the newest commit touching `qb_elos.csv` whose
# committer date precedes run start, and the file *at that commit*, rather than
# `nfeloqb.COMMIT`. The pin is the hub's reproducibility choice and was 3.5 weeks behind the
# source that day; under it "the QB source's update time precedes the run" passed on stale
# bytes. Reading the bytes at the commit whose date is the update time makes the two one fact.
# `until=` asks GitHub for the as-of commit, so an update landing mid-window is not read.

COMMITS_API = f"https://api.github.com/repos/{nfeloqb.REPO}/commits"


class QbSourceUnavailable(RuntimeError):
    """No commit of the file strictly before run start could be read. The caller serves the
    last-good source (which still has to pass the packet's check) or refuses the packet."""


def _http_get(url: str) -> bytes:
    """One GET: the commits API or the raw file. The bytes, or whatever `urllib` raises."""
    cached.refuse_live_call(f"would fetch {url}", patch="get=",
                            tests="tests/unit/test_forecasters_qb.py")
    return _transport(url)


def _transport(url: str) -> bytes:                          # pragma: no cover - network
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": nfeloqb.USER_AGENT,
                                               "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def latest_commit_before(run_start: datetime, *,
                         get: Callable[[str], bytes] = _http_get) -> tuple[str, datetime]:
    """The sha and committer date of the newest commit touching the file before `run_start`."""
    until = run_start.astimezone(UTC)
    url = (f"{COMMITS_API}?path={nfeloqb.FILE}&until={until:%Y-%m-%dT%H:%M:%SZ}&per_page=1")
    try:
        commits = json.loads(get(url))
        sha = commits[0]["sha"]
        when = datetime.fromisoformat(commits[0]["commit"]["committer"]["date"]
                                      .replace("Z", "+00:00"))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise QbSourceUnavailable(f"no commit of {nfeloqb.FILE} before {until:%Y-%m-%dT%H:%M:%SZ}"
                                  f" ({exc!r})") from exc
    # GitHub's `until` is the filter; this is the check. Strict, as the packet's is.
    if not when < run_start:
        raise QbSourceUnavailable(f"commit {sha} dated {when.isoformat()} is not before "
                                  f"run start {until.isoformat()}")
    return sha, when


def qb_source_at(run_start: datetime, *, stamp: SourceStamp,
                 get: Callable[[str], bytes] = _http_get) -> QbSource:
    """The starters from the file at the latest commit before `run_start`, timed by that commit."""
    sha, when = latest_commit_before(run_start, get=get)
    rows = nfeloqb.parse(get(nfeloqb.url(sha)))
    return qb_source_from_nfeloqb(nfeloqb.state(rows), commit_time=when, stamp=stamp)
