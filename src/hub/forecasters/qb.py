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

The caller looks the commit time up (a network call this module does not make) and passes it;
`None` means unknown and is refused by the packet's check, not papered over.
"""
from __future__ import annotations

from datetime import datetime

import polars as pl

from hub.forecasters.inputs import QbSource, SourceStamp

QB_TIME_BASIS = "github committer date of the last commit touching qb_elos.csv"


def qb_source_from_nfeloqb(state: pl.DataFrame, *, commit_time: datetime | None,
                           stamp: SourceStamp) -> QbSource:
    """`hub.fetch.nfeloqb.state`'s frame as a `QbSource`: team -> the starter on its latest row
    (the coming game's, where the source has listed it)."""
    starters = {row["team"]: row["qb"] for row in state.select("team", "qb").to_dicts()}
    return QbSource(stamp=stamp, updated_at=commit_time,
                    updated_at_basis=QB_TIME_BASIS, starters=starters)
