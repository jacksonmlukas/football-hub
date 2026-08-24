"""nflverse fetch layer: narrow at the boundary, validate on the way out.

2025 play-by-play is 48,771 rows by **372 columns**. nflreadpy offers no column selection,
so anything that wants five of those columns downloads all 372 and hands them on. That is
the exact shape of the mistake CLAUDE.md rule 1 exists to prevent, and hoping every caller
remembers to narrow is not a control.

So this module refuses. Asking for a wide source without naming columns raises, and says
how wide it would have been. Narrow sources are returned as they are, because refusing
everything just relocates the friction.

Everything returned passes a `Contract` first. The failure this guards is not a crash --
it is nflverse renaming a column between releases and downstream numbers going quietly
wrong for weeks.

    uv run python -m hub.fetch.nflverse --refresh --season 2025
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Callable, Sequence

import polars as pl

from hub import store
from hub.contracts import (FF_OPPORTUNITY, PBP, PLAYER_STATS, SCHEDULES,
                           Contract)

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw" / "nflverse"

# Sources wide enough that handing one back whole is the mistake. Anything listed here
# must be asked for by column.
WIDE: dict[str, int] = {"pbp": 372, "player_stats": 150}

# The default slice `--refresh` takes of play-by-play. Named explicitly rather than
# defaulted inside load(), so the library API stays strict while the CLI stays usable:
# a caller writing code still has to decide what they want.
PBP_COLS: tuple[str, ...] = (
    "game_id", "season", "week", "posteam", "defteam",
    "play_type", "epa", "wp", "yards_gained", "success",
)


# The default slice of weekly player stats. 150 columns of box score, of which the weekly
# spread fit wants one: what he actually scored, in this league's scoring.
PLAYER_STATS_COLS: tuple[str, ...] = (
    "player_id", "player_display_name", "position", "season", "week", "season_type",
    "fantasy_points_ppr",
)


class WideFrameRefused(Exception):
    """Asked for a frame this module will not return in the shape requested."""


def _raw_pbp(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_pbp(seasons=list(seasons))


def _clean_ff_opportunity(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that belong to no player.

    2025 ships 423 of 6,054 rows with player_id, position and full_name all null, still
    carrying up to 13.3 expected points. They are unattributed team-level residue: real
    numbers with nobody to assign them to.

    They go at the boundary rather than downstream, for two reasons. The FF_OPPORTUNITY
    contract declares player_id non-null, and it is right to -- weakening it to admit
    these would blind it to a genuine upstream break. And `expected_points()` already
    groups by player_id, so they were silently collapsing into a null bucket that joined
    to nothing. Dropping here makes a loss that was already happening visible.
    """
    return df.filter(pl.col("player_id").is_not_null())


def _raw_ff_opportunity(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_ff_opportunity(seasons=list(seasons), stat_type="weekly")
    clean = _clean_ff_opportunity(raw)
    if clean.height < raw.height:
        print(f"    ff_opportunity: dropped {raw.height - clean.height:,} unattributed "
              f"rows of {raw.height:,}")
    return clean


class UnattributedPoints(Exception):
    """Scoring rows that belong to no player -- an upstream break, not residue."""


def _clean_player_stats(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that belong to no player, and refuse if any of them scored.

    nflverse ships exactly 22 rows a season with player_id, position and name all null and
    zero fantasy points. They are residue, and the PLAYER_STATS contract declares player_id
    non-null, so they go here rather than by weakening the contract -- the same call
    `_clean_ff_opportunity` makes.

    The difference from that one: this refuses if an unattributed row carries points. There
    the residue genuinely held expected points with nobody to assign them to; here a null-id
    row that scored would mean nflverse had changed something, and silently dropping real
    points is how a projection goes quietly wrong for a month.
    """
    orphan = df.filter(pl.col("player_id").is_null())
    scoring = orphan.filter(pl.col("fantasy_points_ppr").fill_null(0.0) != 0.0)
    if scoring.height:
        raise UnattributedPoints(
            f"{scoring.height} rows with no player_id carry "
            f"{scoring['fantasy_points_ppr'].sum():.1f} fantasy points. Historically these "
            "rows are empty residue; points in them means the upstream shape changed.")
    return df.filter(pl.col("player_id").is_not_null())


def _raw_player_stats(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    raw = nfl.load_player_stats(seasons=list(seasons), summary_level="week")
    clean = _clean_player_stats(raw)
    if clean.height < raw.height:
        print(f"    player_stats: dropped {raw.height - clean.height:,} unattributed "
              f"rows of {raw.height:,}")
    return clean


def _raw_schedules(seasons: Sequence[int]) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_schedules().filter(pl.col("season").is_in(list(seasons)))


SOURCES: dict[str, Contract | None] = {
    "pbp": PBP,
    "ff_opportunity": FF_OPPORTUNITY,
    "player_stats": PLAYER_STATS,
    "schedules": SCHEDULES,
}


def _fetch(source: str, seasons: Sequence[int]) -> pl.DataFrame:
    """Dispatch by name at call time, not by binding function objects at import.

    A dict of function objects built at module scope captures whatever was defined then,
    so a test that replaces `_raw_ff_opportunity` is ignored and the call goes to the
    network instead. That is not only a testing problem: it makes the indirection a lie.
    """
    fetchers: dict[str, Callable[[Sequence[int]], pl.DataFrame]] = {
        "pbp": _raw_pbp,
        "ff_opportunity": _raw_ff_opportunity,
        "player_stats": _raw_player_stats,
        "schedules": _raw_schedules,
    }
    return fetchers[source](seasons)


def _cache_path(source: str, seasons: Sequence[int], cols: Sequence[str] | None,
                cache: Path | None) -> Path:
    """One entry per (source, seasons, columns).

    The column set is part of the key. Without that, a caller asking for four columns
    would be served an earlier caller's three and never notice.
    """
    root = cache or RAW
    key = ",".join(sorted(cols)) if cols else "all"
    digest = hashlib.sha256(key.encode()).hexdigest()[:8]
    stamp = "-".join(str(s) for s in sorted(seasons))
    return root / source / f"{stamp}-{digest}.parquet"


def load(source: str, seasons: Sequence[int], cols: Sequence[str] | None = None,
         refresh: bool = False, cache: Path | None = None) -> pl.DataFrame:
    """Fetch one nflverse source, narrowed and validated.

    Raises rather than returning a wide frame, because the caller who forgets to narrow
    is the caller this module exists for.
    """
    if source not in SOURCES:
        raise WideFrameRefused(
            f"unknown source {source!r}. Known: {', '.join(sorted(SOURCES))}")

    if source in WIDE and not cols:
        standard = {"pbp": "PBP_COLS", "player_stats": "PLAYER_STATS_COLS"}.get(source)
        raise WideFrameRefused(
            f"{source} is {WIDE[source]} columns wide; name the ones you need via cols=."
            + (f" For the standard slice use hub.fetch.nflverse.{standard}."
               if standard else ""))

    path = _cache_path(source, seasons, cols, cache)
    if path.exists() and not refresh:
        return pl.read_parquet(path)

    contract = SOURCES[source]
    df = _fetch(source, seasons)

    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise WideFrameRefused(
                f"{source} has no column {missing!r}; "
                f"got {len(df.columns)} columns from upstream")
        df = df.select(list(cols))

    if contract is not None:
        contract.validate(df)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


def _write_by_week(df: pl.DataFrame, table: str, season: int,
                   base: Path | None) -> tuple[int, int]:
    """Split a season frame into the store's week partitions.

    hub.store partitions on week, so a season-wide frame has to be broken up here rather
    than written as one blob -- otherwise every later query scans the whole season to read
    one week, which is the pruning the Hive layout exists for.
    """
    weeks = sorted(w for w in df["week"].unique().to_list() if w is not None)
    for wk in weeks:
        store.write(df.filter(pl.col("week") == wk), table, "nfl", season, int(wk),
                    base=base)
    return len(weeks), df.height


def refresh(season: int = 2025, cache: Path | None = None,
            base: Path | None = None) -> int:
    """Pull play-by-play and ff_opportunity, write both through the store.

    Prints counts only. A fetch path that prints rows is the failure it is meant to
    prevent, so the summary never names a column or a value.
    """
    print(f"  nflverse refresh: season {season}")
    total_rows = 0
    for table, cols, label in (("pbp", list(PBP_COLS), "play-by-play"),
                               ("ff_opportunity", None, "ff_opportunity")):
        df = load(table, seasons=[season], cols=cols, refresh=True, cache=cache)
        n_weeks, rows = _write_by_week(df, table, season, base)
        total_rows += rows
        print(f"    {label:<16} {rows:>7,} rows | {len(df.columns):>3} cols | "
              f"{n_weeks} week partitions")
    print(f"  wrote {total_rows:,} rows through hub.store")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.nflverse",
        description="Fetch nflverse data, narrowed at the boundary and contract-checked.")
    ap.add_argument("--refresh", action="store_true",
                    help="pull this season's pbp and ff_opportunity into the store")
    ap.add_argument("--season", type=int, default=2025)
    a = ap.parse_args(argv)
    if not a.refresh:
        ap.print_help()
        return 0
    return refresh(season=a.season, cache=RAW)


if __name__ == "__main__":
    sys.exit(main())
