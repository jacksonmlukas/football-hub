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

**Not yet adopted, and the reason is the ticket's own point.** Run on 2026-09-06 this produces
a contract-valid archive at 0.62 MB, against the committed 0.75 MB -- but eleven Panel tests
fail against it, on content rather than columns: `snap_trend` goes null where the tests expect
a trend, and the row counts do not match what is committed (`ff_opportunity` 409 against 1000,
`ff_rankings` 835 against 456, `snap_counts` 385 against 410).

That is the unwritten trim rule biting. The committed archive was cut by hand and the cuts
were never recorded, so this script can state *a* rule but not *the* rule, and a different
rule is a different world -- one where a six-week trend has fewer weeks to be computed from.
Adopting it therefore means reconciling those eleven tests against the new archive, deciding
in each case whether the assertion was about the Panel or about the fixture. That is real work
and it is not a re-capture, so it is not smuggled in here.

What this file buys today is that the rule is finally written down and executable. What it
costs is that the first person to run `--write` owns those eleven tests.
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

# Scrape dates kept per preseason for `ff_rankings`. One would serve `consensus` and prove
# nothing about the as-of filter; the whole window is four hundred dates and 54 MB.
SCRAPES_PER_SEASON = 4

# The column each source names its player in. `schedules` has none -- it is per game, and every
# row is needed for the join regardless of who is in the cohort.
PLAYER_COLUMN: dict[str, str] = {
    "player_stats": "player_display_name",
    "snap_counts": "player",
    "injuries": "full_name",
    "ff_opportunity": "full_name",
    "ff_rankings": "player",
}


def _trim(source: str, df: pl.DataFrame) -> pl.DataFrame:
    """The row rule, applied where the source has the column to apply it to."""
    col = PLAYER_COLUMN.get(source)
    if col and col in df.columns:
        df = df.filter(pl.col(col).is_in(list(COHORT)))
    if "week" in df.columns:
        df = df.filter(pl.col("week").is_in(list(WEEKS)))
    return df


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
        # The archive, narrowed to the preseasons the captured seasons drafted from. The full
        # table is 1.83M rows from 2019 and has no business in a fixture directory; the window
        # is the one `hub.draft.board.consensus` reads for these seasons.
        # `scrape_date` is an ISO string and sorts correctly as text, which is the comparison
        # `hub.draft.board.consensus` makes; casting it here just to compare would be waste
        # and would differ from the reader this fixture serves.
        window = nfl.load_ff_rankings("all").filter(
            (pl.col("scrape_date") >= f"{min(SEASONS)}-07-01")
            & (pl.col("scrape_date") <= f"{max(SEASONS)}-09-01"))
        # The last few scrapes of each preseason, not all of them. `consensus(as_of)` takes
        # the latest scrape per player at or before its date, so one date would serve the
        # reader and prove nothing about the as-of filter; a handful exercises it while
        # keeping four hundred rows rather than four hundred dates.
        keep_dates = []
        for yr in SEASONS:
            dates = sorted(window.filter(pl.col("scrape_date").str.starts_with(str(yr)))
                                 ["scrape_date"].unique().to_list())
            keep_dates += dates[-SCRAPES_PER_SEASON:]
        got = window.filter(pl.col("scrape_date").is_in(keep_dates))
    elif source == "ff_opportunity":
        got = nfl.load_ff_opportunity(seasons=list(SEASONS), stat_type="weekly")
    else:
        got = getattr(nfl, f"load_{source}")(seasons=list(SEASONS))
    have = [c for c in keep if c in got.columns]
    missing = [c for c in keep if c not in got.columns]
    if missing:
        raise SystemExit(f"{source}: upstream no longer has {missing}; the contract or the "
                         f"Panel is reading a column that is gone")
    return _trim(source, got.select(have) if have else got)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capture_panel_archive", description=__doc__)
    ap.add_argument("--write", action="store_true", help="write the files; otherwise report")
    a = ap.parse_args(argv)

    total = 0
    for source in sorted(set(PANEL_READS)):
        keep = _wanted(source)
        df = capture(source, keep)
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
