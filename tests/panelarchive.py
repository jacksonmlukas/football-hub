"""The frozen archive the **Panel** assembly is driven against, with no network.

`hub.models.panel.build_panel` is the sole enforcer of what makes a Panel a Panel -- one row
per (player, season, week), every feature measured strictly before its outcome -- and eleven of
its sources are marked `pragma: no cover - network`, so until now the only thing any test did
with it was read its source text back. A greppable string survives a rewrite that does
something else entirely; the rule does not. This module is the adapter that lets the assembly
*run*.

**The seam is nflreadpy, not this repo's own functions.** Faking `snap_share` or
`weekly_consensus` would leave the per-source narrowing -- the REG filter, the position
filter, the name key, the as-of window -- untested, and those narrowings are half of where a
join goes wrong. So every substitution here is at the last call before the wire, and
everything above it is the production code path.

**Anything not frozen raises.** `load_participation` and `load_ftn_charting` are installed as
refusals rather than left alone, because a source that quietly reached the real nflverse from
a unit test would be a test that passes on a laptop and hangs in CI -- and one whose result
nobody froze. See `tests/golden/fixtures/README.md` for what each capture holds, what was
trimmed, and the two spec flags this archive deliberately cannot drive.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

ARCHIVE = Path(__file__).resolve().parent / "golden" / "fixtures" / "panel_archive"

# What the capture covers. Stated here rather than in each test, because these are properties
# of the frozen files -- a test that assumed a third season would silently measure nothing.
SEASONS: tuple[int, ...] = (2023, 2024)
WEEKS: tuple[int, ...] = tuple(range(1, 15))

# The sixteen. Four at each position, six of them changing team between the two seasons or
# missing weeks, and two -- Michael Pittman and Gus Edwards -- carrying a real week where the
# consensus page did not list them and they scored anyway. That last pair is what makes the
# gate's join failure a thing the assembly can actually produce.
PLAYERS: tuple[str, ...] = (
    "patrick mahomes", "jalen hurts", "justin herbert", "cj stroud",
    "bijan robinson", "josh jacobs", "derrick henry", "gus edwards",
    "ceedee lamb", "jamarr chase", "tyreek hill", "michael pittman", "terry mclaurin",
    "kyle pitts", "sam laporta", "cade otton",
)


class NotFrozen(Exception):
    """A source the archive does not hold was asked for. Loud, rather than a live fetch."""


def frame(name: str) -> pl.DataFrame:
    """One captured table, in the dtypes it was captured with.

    The dtype map is part of the capture and not decoration. JSON has three scalar types and
    nflverse has a dozen; inferred, a column of whole numbers comes back `Int64` and a column
    that happened to be null on every captured row comes back `Null`, and either one changes
    what the assembly does with it. Recording the dtypes keeps the file a statement about the
    frame nflverse returned rather than about what JSON could carry.
    """
    blob = json.loads((ARCHIVE / f"{name}.json").read_text())
    schema = {c: getattr(pl, t) for c, t in blob["dtypes"].items()}
    return pl.DataFrame(blob["rows"], schema=schema)


Edit = Callable[[pl.DataFrame], pl.DataFrame]


def _seasons(arg: object) -> list[int] | None:
    """nflreadpy takes `seasons` as an int, a list, True for all, or None."""
    if arg is None or arg is True:
        return None
    return [int(arg)] if isinstance(arg, int) else [int(s) for s in arg]  # type: ignore[arg-type]


def install(monkeypatch, tmp_path: Path, *, edits: dict[str, Edit] | None = None,
            board: bool = False) -> None:
    """Serve the archive to every nflverse read the assembly makes, and nothing else.

    `edits` maps a captured table to a transform applied on the way out. That is how the
    leakage rule is asserted behaviourally rather than read: change one player-week's realised
    play in the archive, rebuild, and see which feature values move. A test that only ran the
    assembly and checked it returned rows would prove nothing about the rule.

    `board` additionally serves the frozen board to `hub.draft.board.board_as_of`, which
    `hub.season.weekly_gate_data.assemble_universe` calls. The board is a capture of that
    function's own output rather than of its inputs: rebuilding it offline would mean freezing
    a whole season of `ff_opportunity` and the draft archive to test a seam that belongs to
    `hub.draft.board` and has six test files of its own. What `assemble_universe` owns is
    everything from the board onward, and that runs here for real.
    """
    import nflreadpy as nfl

    import hub.fetch.nflverse as nv

    got = edits or {}

    def served(name: str) -> pl.DataFrame:
        df = frame(name)
        return got[name](df) if name in got else df

    def by_season(name: str):
        def load(seasons=None, **_kw):
            df = served(name)
            want = _seasons(seasons)
            # nflreadpy picks the season *files* to read, so its `seasons` argument never
            # meets the column's dtype. This filter does, and `ff_opportunity` ships `season`
            # as a string -- the capture records that, as it should, and `expected_weekly`
            # casts it -- so the comparison is made on the integer the argument names.
            return df if want is None else df.filter(
                pl.col("season").cast(pl.Int64).is_in(want))
        return load

    def refuse(source: str):
        def load(*_a, **_kw):
            raise NotFrozen(
                f"{source} is not in tests/golden/fixtures/panel_archive. It is play-level, "
                f"and a capture deep enough to make a six-week trend real would be larger "
                f"than the rest of the archive together -- see that directory's README for "
                f"what the two opt-in spec flags therefore cannot be driven against here.")
        return load

    monkeypatch.setattr(nfl, "load_player_stats", by_season("player_stats"))
    monkeypatch.setattr(nfl, "load_snap_counts", by_season("snap_counts"))
    monkeypatch.setattr(nfl, "load_injuries", by_season("injuries"))
    monkeypatch.setattr(nfl, "load_ff_opportunity", by_season("ff_opportunity"))
    monkeypatch.setattr(nfl, "load_schedules", lambda *_a, **_kw: served("schedules"))
    monkeypatch.setattr(nfl, "load_ff_rankings", lambda *_a, **_kw: served("ff_rankings"))
    monkeypatch.setattr(nfl, "load_participation", refuse("participation"))
    monkeypatch.setattr(nfl, "load_ftn_charting", refuse("ftn_charting"))

    # The rankings archive is the one source read through the pinning loader, which writes a
    # dated parquet and a pin beside it. Left alone it writes into the developer's own
    # data/raw and the next test naming the same as-of is served this archive instead of its
    # own -- the reason `test_panel.py::_routed` redirects it too.
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")

    if board:
        import hub.draft.board as brd
        from hub.draft.board import BuildReport
        monkeypatch.setattr(brd, "board_as_of",
                            lambda season: (frame("draft_board"), BuildReport()))


def play_derived_columns() -> set[str]:
    """The columns the archive hands the panel already measured on the outcome week itself.

    Derived from the captures rather than listed by hand: every column of `player_stats` and
    `snap_counts` is a fact about week *w*'s play, plus the two totals `build_panel` adds from
    them. Written out as a list, this would be the same read-the-source assertion in a
    different costume -- a column added to the panel would quietly join the exempt set instead
    of being tested.

    **This is now a second opinion rather than the only one.** Which of the Panel's columns
    are features is `hub.models.panel.feature_columns`' answer, because #203 moved it onto the
    interface -- it used to live here, which made it a property of the tests rather than of
    the thing under test, and a caller could not consult it at all. What this keeps is
    independence: it reads the captured frames and the module reads its own declarations, so
    `test_the_modules_outcome_set_agrees_with_what_the_captures_supply` is two derivations
    meeting rather than one restated.
    """
    return set(frame("player_stats").columns) | set(frame("snap_counts").columns) | {
        # `tds` and `yds` are week-w sums `build_panel` forms from the columns above, and
        # `offense_pct` is the snap capture's own column under the name the join gives it.
        "tds", "yds", "offense_pct",
    }


def rows_up_to(panel: pl.DataFrame, season: int, week: int,
               keys: Sequence[str] = ("player_id", "season", "week")) -> pl.DataFrame:
    """Every Panel row at or before `(season, week)`, sorted, for a frame-to-frame compare.

    Lexicographic on (season, week) and not on week alone -- week 3 of 2024 is later than week
    14 of 2023, and the naive comparison would call half the archive "earlier".
    """
    earlier = (pl.col("season") < season) | (
        (pl.col("season") == season) & (pl.col("week") <= week))
    return panel.filter(earlier).sort(list(keys))
