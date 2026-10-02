"""The frozen archive the **Panel** assembly is driven against, with no network.

`hub.models.panel.build_panel` is the sole enforcer of what makes a Panel a Panel -- one row
per (player, season, week), every feature measured strictly before its outcome -- and eleven of
its sources are marked `pragma: no cover - network`, so until now the only thing any test did
with it was read its source text back. A greppable string survives a rewrite that does
something else entirely; the rule does not. This module is the adapter that lets the assembly
*run*.

**The seam is the nflverse adapter, not this repo's own functions.** Faking `snap_share` or
`weekly_consensus` would leave the per-source narrowing -- the REG filter, the position
filter, the name key, the as-of window -- untested, and those narrowings are half of where a
join goes wrong. So the archive is served as a `hub.fetch.replay.Replay`, the last thing
before the wire, and everything above it is the production code path.

**Anything not frozen raises.** `participation` and `ftn_charting` are not in the recorded
set, and a Replay raises `NotRecorded` for what it does not hold, because a source that
quietly reached the real nflverse from a unit test would be a test that passes on a laptop and
hangs in CI -- and one whose result nobody froze. See `tests/golden/fixtures/README.md` for what each capture holds, what was
trimmed, and the two spec flags this archive deliberately cannot drive.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

import hub.fetch.nflverse as nv
from hub.fetch.replay import Replay, read_recording

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


def frame(name: str) -> pl.DataFrame:
    """One captured table, in the dtypes it was captured with."""
    return read_recording(ARCHIVE / f"{name}.json")


Edit = Callable[[pl.DataFrame], pl.DataFrame]

# The sources the archive holds. `participation` and `ftn_charting` are deliberately absent:
# a Replay raises `NotRecorded` for them, which is what keeps a unit test from quietly
# reaching the real nflverse for a source nobody froze.
RECORDED = ("player_stats", "snap_counts", "injuries", "ff_opportunity", "schedules",
            "ff_rankings")
NOT_FROZEN = ("It is play-level, and a capture deep enough to make a six-week trend real "
              "would be larger than the rest of the archive together -- see "
              "tests/golden/fixtures/README.md for what the two opt-in spec flags therefore "
              "cannot be driven against here.")


def replay(*, edits: dict[str, Edit] | None = None) -> Replay:
    """The archive as the nflverse seam's second adapter."""
    return Replay.recorded(ARCHIVE, RECORDED, edits=edits,
                           absent={"participation": NOT_FROZEN, "ftn_charting": NOT_FROZEN})


def install(monkeypatch, tmp_path: Path, *, edits: dict[str, Edit] | None = None,
            board: bool = False) -> Replay:
    """Select the archive as the nflverse adapter, with its cache under `tmp_path`.

    `edits` maps a captured table to a transform applied on the way out. That is how the
    leakage rule is asserted behaviourally rather than read: change one player-week's realised
    play in the archive, rebuild, and see which feature values move. A test that only ran the
    assembly and checked it returned rows would prove nothing about the rule.

    The seam is `hub.fetch.nflverse`'s adapter, not nflreadpy and not this repo's own
    functions: everything above the adapter -- the REG filter, the position filter, the name
    key, the as-of window, the contract, the cache and the pin -- is the production path.
    `tests/conftest.py` puts the network adapter back after the test.

    `board` additionally serves the frozen board to `hub.draft.board.board_as_of`, which
    `hub.season.weekly_gate_data.assemble_universe` calls. The board is a capture of that
    function's own output rather than of its inputs: rebuilding it offline would mean freezing
    a whole season of `ff_opportunity` and the draft archive to test a seam that belongs to
    `hub.draft.board` and has six test files of its own. What `assemble_universe` owns is
    everything from the board onward, and that runs here for real.
    """
    rep = replay(edits=edits)
    nv.select(rep, cache=tmp_path / "raw")
    if board:
        import hub.draft.board as brd
        from hub.draft.board import Board
        # The frozen frame under the report its columns derive (#295): what `board_as_of`
        # returns is a Board, and the frame with an empty report beside it is a frame fuller
        # than its report and no longer a Board.
        monkeypatch.setattr(brd, "board_as_of",
                            lambda season: Board.served(frame("draft_board")))
    return rep


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
