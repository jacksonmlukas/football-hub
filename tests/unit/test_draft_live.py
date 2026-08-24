"""The draft-day poller.

`draft-day-ops/SKILL.md:25`, `SETUP.md:227` and the Makefile all reference this and it did
not exist. The draft is 2026-09-03.

Everything here is shaped by one constraint the skill states outright: roughly 90 seconds
per pick. That makes latency a correctness property, not a nicety -- a board that is right
but arrives after the clock expires is wrong. So the refresh is budgeted and the output is
bounded, and both are tested rather than hoped for.

The other thing worth pinning is that replacement level *moves during a draft*. VOR
computed against a preseason baseline quietly overvalues a position after a run on it, and
a run is precisely when the number is being consulted.
"""
import time

import polars as pl
import pytest

from hub.draft import live
from hub.draft.state import DraftState, take


def _board(n=200):
    cycle = ["RB", "WR", "WR", "TE", "QB"]
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [cycle[i % len(cycle)] for i in range(n)],
        "team": ["KC"] * n,
        "ecr": [float(i + 1) for i in range(n)],
        "sd": [3.0] * n,
        "worst": [float(i + 20) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "edge": [float((i % 21) - 10) for i in range(n)],
        "games": [16] * n,
        "xfp_per_game": [max(0.5, 22.0 - i * 0.1) for i in range(n)],
        "vor": [max(0.0, 12.0 - i * 0.05) for i in range(n)],
    })


# --- positional runs ------------------------------------------------------

def test_three_of_the_last_five_is_a_run():
    """The skill's definition, exactly: 3+ at one position inside the last 5."""
    assert live.detect_run(["RB", "WR", "RB", "TE", "RB"]) == "RB"


def test_two_of_the_last_five_is_not_a_run():
    assert live.detect_run(["RB", "WR", "RB", "TE", "QB"]) is None


def test_only_the_last_five_count():
    """Four RBs ago is history; the market has already re-priced."""
    assert live.detect_run(["RB", "RB", "RB", "WR", "TE", "QB", "WR"]) is None


def test_an_early_draft_with_fewer_than_five_picks_can_still_run():
    assert live.detect_run(["RB", "RB", "RB"]) == "RB"


def test_no_picks_is_not_a_run():
    assert live.detect_run([]) is None


# --- the board moves as players go ---------------------------------------

def test_drafted_players_leave_the_board():
    view = live.refresh(_board(), take(DraftState(), "P0", "P1"), my_slot=3)
    assert "P0" not in view["available"]["player"].to_list()


def test_replacement_level_is_recomputed_from_what_is_left():
    """A preseason baseline overvalues a position right after a run on it -- which is
    exactly when someone is looking at the number."""
    board = _board()
    before = live.refresh(board, DraftState(), my_slot=3)["replacement"]
    # take the top 30 RBs off the board
    rbs = board.filter(pl.col("pos") == "RB")["player"].to_list()[:30]
    after = live.refresh(board, take(DraftState(), *rbs), my_slot=3)["replacement"]
    assert after["RB"] < before["RB"], "RB replacement must fall as the pool empties"


def test_vor_is_recomputed_against_the_live_replacement():
    board = _board()
    rbs = board.filter(pl.col("pos") == "RB")["player"].to_list()[:30]
    view = live.refresh(board, take(DraftState(), *rbs), my_slot=3)
    live_vor = view["available"].filter(pl.col("pos") == "RB")["vor_live"]
    assert live_vor.null_count() == 0


# --- the pick in front of you --------------------------------------------

def test_it_reports_the_mode_for_the_next_pick():
    """Slot 3 alternates scarcity and value; that alternation should drive the pick."""
    view = live.refresh(_board(), DraftState(), my_slot=3)
    assert view["mode"] in ("scarcity", "value")
    assert view["next_pick"] == 3


def test_the_next_pick_advances_as_the_draft_moves():
    state = take(DraftState(), *[f"P{i}" for i in range(10)])
    assert live.refresh(_board(), state, my_slot=3)["next_pick"] == 22


def test_a_run_is_surfaced_in_the_view():
    state = take(DraftState(), "P0", "P5", "P10")  # three RBs on this board
    assert live.refresh(_board(), state, my_slot=3)["run"] == "RB"


def test_the_draft_being_over_does_not_crash():
    state = take(DraftState(), *[f"P{i}" for i in range(192)])
    view = live.refresh(_board(), state, my_slot=3, rounds=16)
    assert view["next_pick"] is None


# --- latency is a correctness property -----------------------------------

def test_a_refresh_fits_inside_the_budget():
    """90 seconds per pick. The plan's ceiling is 2s; anything slower is unusable."""
    board = _board(450)
    state = take(DraftState(), *[f"P{i}" for i in range(60)])
    start = time.perf_counter()
    live.refresh(board, state, my_slot=3)
    assert time.perf_counter() - start < live.REFRESH_BUDGET_S


def test_the_rendered_view_is_short_enough_to_read_on_the_clock():
    view = live.refresh(_board(), DraftState(), my_slot=3)
    assert len(live.render(view)) <= live.MAX_LINES


def test_the_render_leads_with_the_run_warning_when_one_fires():
    state = take(DraftState(), "P0", "P5", "P10")
    text = "\n".join(live.render(live.refresh(_board(), state, my_slot=3)))
    assert "RUN" in text and "RB" in text


# --- replay ---------------------------------------------------------------

def test_replay_reproduces_the_pick_sequence():
    """The plan's done-when. Replay is also the only way to rehearse before Sep 3."""
    picks = [f"P{i}" for i in range(48)]
    seen = live.replay(_board(), picks, my_slot=3, quiet=True)
    assert seen == picks


def test_replay_surfaces_every_run_that_occurred():
    picks = ["P0", "P5", "P10", "P3", "P8"]  # RB, RB, RB, TE, TE on this board
    runs = live.replay(_board(), picks, my_slot=3, quiet=True, collect_runs=True)
    assert "RB" in runs


def test_replay_of_an_empty_draft_is_not_an_error():
    assert live.replay(_board(), [], my_slot=3, quiet=True) == []


# --- the round-3 rule -----------------------------------------------------

def test_early_rounds_do_not_rank_on_edge():
    """draft-day-ops: consensus is tight at the top, so a large edge in round 1 is
    usually a name-matching failure. Ranking by it surfaced three negative-VOR players
    as the top of a first-round board."""
    key, _ = live._sort_key(3, teams=12)
    assert key == "vor_live"


def test_edge_takes_over_from_round_four():
    key, _ = live._sort_key(46, teams=12)   # round 4
    assert key == "edge"


def test_the_boundary_is_where_the_skill_puts_it():
    assert live._sort_key(36, teams=12)[0] == "vor_live"   # round 3
    assert live._sort_key(37, teams=12)[0] == "edge"       # round 4


def test_the_view_says_why_it_ranked_that_way():
    text = "\n".join(live.render(live.refresh(_board(), DraftState(), my_slot=3)))
    assert "unreliable this early" in text


def test_a_first_round_board_is_not_topped_by_negative_vor():
    """The actual failure this rule prevents."""
    view = live.refresh(_board(), DraftState(), my_slot=3)
    key, _ = live._sort_key(view["next_pick"], 12)
    top = view["available"].sort(key, descending=True).head(5)
    assert (top["vor_live"] > 0).all()
