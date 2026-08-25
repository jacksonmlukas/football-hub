"""Keep today's ESPN ADP, because ESPN will not.

Three separate dead ends in this repo trace to one missing input, and it is not a hard one:

* `availability.fit_espn_weight` returns its prior rather than a fit, because estimating what
  share of the room drafts off ESPN's board needs ESPN ADP *as it stood before* past drafts.
* `docs/next.md` P2, the opponent model, is blocked for the same reason.
* [ADR-0010](../../../docs/adr/0010-edge-is-displayed-but-never-ranked-on.md) says `edge` is
  displayed and never ranked on, because it cannot be validated without historical ADP.

Querying ESPN for a past season returns the undrafted sentinel for every player -- verified
August 2026, when Chase, Bijan and Jefferson all came back 170.0 for 2025. The data is not
retained upstream and cannot be recovered later.

It *is* available right now, and until this module existed the repo was destroying it: ADP
lived in exactly one place, `data/processed/draft_board.parquet`, which every `make draft`
overwrites. Each rebuild silently discarded the only record of what the market looked like
that day.

So this appends a dated copy instead. It buys nothing this season -- one snapshot is not a
history -- and it is the entire ingredient for next season. The cost is a few hundred rows a
day; the alternative is arriving at August 2027 with the same three dead ends and the same
explanation.

Snapshots are keyed by date and rewritten within a day, so building the board five times on
draft morning keeps the last build rather than five copies. The day before the draft and the
morning of it are the two that matter most, because those are what the room actually drafted
against.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl

from hub.draft.board import ROOT

ARCHIVE = ROOT / "data" / "processed" / "adp_history"

# What is worth keeping. Deliberately narrow: this is an archive that has to still be
# readable in a year, not a second copy of the board. Everything else in the board is either
# derived from nflverse (recoverable) or a fitted column (reproducible from the code).
KEEP = ("player", "pos", "adp", "ecr")


def _today() -> date:
    return datetime.now(UTC).date()


def snapshot(board: pl.DataFrame, *, on: date | None = None,
             base: Path | None = None) -> Path | None:
    """Write today's ADP to the archive. Returns the path, or None if there was no ADP.

    A board built without ESPN ADP -- the documented degraded path -- must not be recorded
    as though the market had no opinion that day. Absent and zero are different claims.
    """
    if "adp" not in board.columns:
        return None
    keep = [c for c in KEEP if c in board.columns]
    rows = board.select(keep).filter(pl.col("adp").is_not_null())
    if rows.is_empty():
        return None
    day = on or _today()
    out = (base or ARCHIVE) / f"date={day.isoformat()}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "adp.parquet"
    rows.with_columns(pl.lit(day.isoformat()).alias("date")).write_parquet(path)
    return path


def history(base: Path | None = None) -> pl.DataFrame:
    """Every snapshot ever taken, oldest first. Empty frame if the archive is empty."""
    root = base or ARCHIVE
    files = sorted(root.glob("date=*/adp.parquet")) if root.exists() else []
    if not files:
        return pl.DataFrame(schema={"date": pl.Utf8, "player": pl.Utf8,
                                    "pos": pl.Utf8, "adp": pl.Float64})
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed").sort("date")


def days(base: Path | None = None) -> list[str]:
    """Which days have been captured. The thing worth checking before a draft."""
    root = base or ARCHIVE
    return sorted(p.name.removeprefix("date=")
                  for p in root.glob("date=*")) if root.exists() else []
