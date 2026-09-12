"""Re-take `tests/golden/fixtures/panel_archive/`, at one as-of, with the trim rule in code.

The archive is seven frozen tables that let the whole Panel be built without a network. It was
captured by hand and nothing in the tree could reproduce it, which cost a day: routing the
Panel through `hub.fetch.nflverse` (#35) adds contract validation, and three captures could not
satisfy their own contracts because they had been trimmed to the columns the *Panel selects* --
a narrower set than the contract requires. `tests/golden/fixtures/README.md` already warned
about exactly this: a capture cut past the path the production reader takes "still reads as a
capture" while proving less.

So the trim is not removed, it is **stated**. Each source keeps the contract's required columns
plus what the Panel reads, and nothing else; `ff_rankings` keeps a dated window rather than
1.83M rows. What changes is that a reader can run this and get the same archive, instead of
trusting a paragraph about what someone once cut.

**One as-of for all six nflverse sources.** Sources captured on different days describe a
world that never existed -- a Panel built from today's schedules and September's player stats
is not a Panel anyone could have had. That is the subtler form of the defect above.

**The seventh file, `draft_board.json`, is not re-taken here and that is deliberate.** It is
not an nflverse capture; it is a board this repo *built*, served to `board_as_of`. Rebuilding
it would run the board through code that has since changed -- #38 bounds consensus to its own
preseason and #39 moved 138 of the first 192 picks -- so the fixture would silently acquire
today's board arithmetic while claiming to be a 2023 capture, and #50 exists to publish that
movement once, deliberately. It also fails no contract: `nflverse.load` never sees it. When it
is re-taken it should be its own change, with the before and after recorded.

    uv run python scripts/capture_panel_archive.py            # report only, writes nothing
    uv run python scripts/capture_panel_archive.py --write

Live calls, so this is not run by the suite. It is the thing the suite's fixtures come from.

**The rule is stated in terms of the readers the archive serves, and that was the first
version's defect (#234).** Run on 2026-09-11, it cut `ff_rankings` to the preseason
window `hub.draft.board.consensus` reads -- a reader this archive does not serve, because the
draft board is frozen separately as `draft_board.json` -- and so held eight scrape dates, four of
them in December, for a Panel whose `weekly_consensus` reads the in-season `weekly-op` page.
The screen's Panel shrank from 344 rows to 13 and every trend went null. It matched the cohort
by display name, which dropped `Michael Pittman Jr.` and `Patrick Mahomes II` from the
sources that spell them so, while the Panel joins on `player_key`. And it left
`ff_opportunity` at 409 rows against the 1,000 its own contract requires, so `nflverse.load`
refused the very capture that was meant to be routed through it. Each of those is now a
sentence in the code below, and every capture is validated against its contract before it is
written, so "contract-valid" is checked rather than claimed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
ARCHIVE = ROOT / "tests" / "golden" / "fixtures" / "panel_archive"

# The seasons the archive covers, read from the module that reads the archive, so the two
# cannot disagree about what was captured.
from panelarchive import SEASONS  # noqa: E402

from hub.names import player_key  # noqa: E402

# What the Panel reads, per source, beyond the contract's own required set. Both are kept:
# the contract's, because `nflverse.load` validates against it and a capture that fails its
# own contract is what blocked #35; the Panel's, because a capture that satisfies the contract
# and drops what the reader wants proves nothing about the reader.
PANEL_READS: dict[str, tuple[str, ...]] = {
    # Read off the capture this replaces, not guessed. The first attempt listed what the
    # Panel looked like it needed and dropped `total_line`, which `build_panel` reads -- so
    # every Panel test failed on a column nobody had noticed was load-bearing. The previous
    # capture is the record of what the readers actually take; the contract's required set is
    # unioned on top, and that union is the whole change #35 needed.
    "player_stats": ("player_id", "player_display_name", "position", "season", "week", "team",
                     "opponent_team", "fantasy_points_ppr", "target_share", "receiving_yards",
                     "rushing_yards", "passing_yards", "receiving_tds", "rushing_tds",
                     "passing_tds", "season_type", "targets", "receptions", "carries",
                     "attempts", "completions", "passing_interceptions",
                     "fumbles_lost_total"),
    "snap_counts": ("season", "week", "player", "offense_pct", "game_type"),
    "injuries": ("season", "week", "full_name", "report_status", "practice_status"),
    "schedules": ("game_id", "season", "week", "home_team", "away_team", "gameday", "gametime",
                  "game_type", "result", "spread_line", "total_line", "roof", "wind",
                  "home_rest", "away_rest"),
    "ff_opportunity": ("season", "week", "player_id", "full_name", "position",
                       "total_fantasy_points_exp", "receptions_exp", "rec_yards_gained_exp",
                       "rush_yards_gained_exp", "pass_yards_gained_exp"),
    "ff_rankings": ("page_type", "player", "pos", "team", "ecr", "sd", "best", "worst",
                    "scrape_date"),
}


# The cohort the archive follows, and the reason it is sixteen names rather than a league.
#
# Captured whole, these seven tables are 86 MB; the archive is 746 KB. The difference is not
# columns, it is rows -- every source is trimmed to these players over weeks 1-14 of the two
# seasons. Sixteen is enough that a six-week trend and an expanding mean are real for each of
# them, which is what the Panel's features are made of, and small enough that the fixture set
# stays something a unit suite can open on every run.
#
# Named here rather than derived from the archive this script overwrites: a rule that reads
# its own output cannot be checked against it, and the point of the file is that a reader can
# re-derive the archive rather than trust it.
COHORT: tuple[str, ...] = (
    "Bijan Robinson", "C.J. Stroud", "Cade Otton", "CeeDee Lamb", "Derrick Henry",
    "Gus Edwards", "Ja'Marr Chase", "Jalen Hurts", "Josh Jacobs", "Justin Herbert",
    "Kyle Pitts", "Michael Pittman", "Patrick Mahomes", "Sam LaPorta", "Terry McLaurin",
    "Tyreek Hill")

# Weeks 1-14: the Panel's longest lookback is six weeks, so fourteen leaves a full trend and a
# held-out tail. Whole seasons would triple the archive to prove nothing more.
WEEKS = range(1, 15)

# The column each source names its player in. `schedules` has none -- it is per game, and every
# row is needed for the join regardless of who is in the cohort.
PLAYER_COLUMN: dict[str, str] = {
    "player_stats": "player_display_name",
    "snap_counts": "player",
    "injuries": "full_name",
    "ff_opportunity": "full_name",
    "ff_rankings": "player",
}


def _in_cohort(col: str) -> pl.Expr:
    """Cohort membership, on the key the Panel joins on and not on the display name.

    The sources do not agree on a spelling: `snap_counts` and `ff_rankings` carry
    `Michael Pittman Jr.` and `ff_rankings` carries `Patrick Mahomes II`, while `player_stats`
    has neither suffix. An exact match kept both players in some sources and lost them in
    others, and the loss was invisible -- the Panel joined, found nothing, and carried the
    null. `hub.names.player_key` is what every join in the Panel collapses a name to, so it
    is what the cohort is matched on.
    """
    keys = {player_key(n) for n in COHORT}
    return pl.col(col).map_elements(player_key, return_dtype=pl.Utf8).is_in(list(keys))


def _trim(source: str, df: pl.DataFrame) -> pl.DataFrame:
    """The row rule, applied where the source has the column to apply it to."""
    col = PLAYER_COLUMN.get(source)
    if col and col in df.columns:
        df = df.filter(_in_cohort(col))
    if "week" in df.columns:
        df = df.filter(pl.col("week").cast(pl.Int64).is_in(list(WEEKS)))
    return df


def _pad_to_floor(source: str, kept: pl.DataFrame, pool: pl.DataFrame) -> pl.DataFrame:
    """Rows that join to no Panel row, kept so the capture clears its own contract's floor.

    `FF_OPPORTUNITY` declares `min_rows=1000` and `nflverse.load` validates before the Panel
    sees the frame, so the cohort's 409 rows would fail the contract rather than the test --
    which is what the first version of this rule did, and claimed otherwise. The remainder is
    every other row in the same (season, week) cells, in `(season, week, player_id)` order,
    taken until the file reaches exactly the floor. They are there for the floor and nothing
    else, and the fixtures README says so.
    """
    from hub.fetch.nflverse import SOURCES
    contract = SOURCES.get(source)
    floor = contract.min_rows if contract else 0
    if kept.height >= floor:
        return kept
    col = PLAYER_COLUMN[source]
    cells = kept.select("season", "week").unique()
    filler = (pool.join(cells, on=["season", "week"], how="semi")
                  .filter(~_in_cohort(col) & pl.col("player_id").is_not_null())
                  .sort(["season", "week", "player_id"])
                  .head(floor - kept.height))
    return pl.concat([kept, filler])


def _week_windows(schedules: pl.DataFrame) -> pl.DataFrame:
    """First and last kickoff per (season, week), the frame `hub.models.panel.assign_weeks`
    maps a scrape onto. The same three lines as `panel.week_windows`, on the frame this run
    captured rather than on a second fetch."""
    return (schedules.filter(pl.col("season").is_in(list(SEASONS))
                             & (pl.col("game_type") == "REG"))
                     .group_by(["season", "week"])
                     .agg(pl.col("gameday").str.to_date().min().alias("first_kick"),
                          pl.col("gameday").str.to_date().max().alias("last_kick"))
                     .sort("last_kick"))


def _wanted(source: str) -> list[str]:
    """The contract's required columns, then the Panel's, in a stable order and deduplicated."""
    from hub.fetch.nflverse import SOURCES
    contract = SOURCES.get(source)
    required = tuple(getattr(contract, "required", ()) or ()) if contract else ()
    out: list[str] = []
    for c in (*required, *PANEL_READS.get(source, ())):
        if c not in out:
            out.append(c)
    return out


def _dump(df: pl.DataFrame) -> dict:
    """The on-disk shape `panelarchive.frame` reads: rows plus the dtypes they were captured
    with. The dtype map is part of the capture -- inferred from JSON, a column that happened
    to be null on every row comes back `Null` and changes what the assembly does with it."""
    return {"dtypes": {c: str(t) for c, t in zip(df.columns, df.dtypes, strict=True)},
            "rows": df.to_dicts()}


def capture(source: str, keep: list[str]) -> pl.DataFrame:
    import nflreadpy as nfl
    if source == "schedules":
        got = nfl.load_schedules().filter(pl.col("season").is_in(list(SEASONS)))
    elif source == "ff_rankings":
        # The page the Panel reads, at the scrapes the Panel would assign to the captured
        # weeks. `weekly_consensus` takes `CONSENSUS_PAGE` -- the cross-position weekly page,
        # 47 pages are stacked in the full table on their own scales -- and `assign_weeks`
        # is what decides which week a scrape belongs to, so it decides here too: a scrape
        # is kept when it lands on one of `WEEKS` in one of `SEASONS`. The draft board's preseason
        # window is *not* this archive's business; `draft_board.json` serves the draft board.
        from hub.models.panel import CONSENSUS_PAGE, assign_weeks
        page = nfl.load_ff_rankings("all").filter(pl.col("page_type") == CONSENSUS_PAGE)
        scrapes = page.select(pl.col("scrape_date").str.to_date()).unique()
        assigned = assign_weeks(scrapes, _week_windows(nfl.load_schedules()))
        dates = (assigned.filter(pl.col("week").is_in(list(WEEKS)))["scrape_date"]
                         .dt.strftime("%Y-%m-%d").to_list())
        got = page.filter(pl.col("scrape_date").is_in(dates))
    elif source == "ff_opportunity":
        got = nfl.load_ff_opportunity(seasons=list(SEASONS), stat_type="weekly")
    else:
        got = getattr(nfl, f"load_{source}")(seasons=list(SEASONS))
    have = [c for c in keep if c in got.columns]
    missing = [c for c in keep if c not in got.columns]
    if missing:
        raise SystemExit(f"{source}: upstream no longer has {missing}; the contract or the "
                         f"Panel is reading a column that is gone")
    pool = got.select(have) if have else got
    return _pad_to_floor(source, _trim(source, pool), pool)


def check(source: str, df: pl.DataFrame) -> None:
    """The capture, asked the question `nflverse.load` will ask of it. A capture that fails
    its own contract blocks the routing it exists to serve, and the first run of this script
    wrote one while its docstring said it had not."""
    from hub.contracts import ContractViolation
    from hub.fetch.nflverse import SOURCES
    contract = SOURCES.get(source)
    if contract is None:
        return
    try:
        contract.validate(df)
    except ContractViolation as e:
        raise SystemExit(f"{source}: the capture fails its own contract -- {e}") from e


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capture_panel_archive", description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the files; otherwise report")
    a = ap.parse_args(argv)

    total = 0
    for source in sorted(set(PANEL_READS)):
        keep = _wanted(source)
        df = capture(source, keep)
        check(source, df)
        blob = json.dumps(_dump(df), separators=(",", ":"))
        total += len(blob)
        print(f"  {source:16} {df.height:>7,} rows x {df.width:>3} cols  "
              f"{len(blob) / 1e6:6.2f} MB")
        if a.write:
            (ARCHIVE / f"{source}.json").write_text(blob + "\n")
    print(f"  {'':16} {'':>7}       {'':>3}       {total / 1e6:6.2f} MB total"
          + ("" if a.write else "   (nothing written; pass --write)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
