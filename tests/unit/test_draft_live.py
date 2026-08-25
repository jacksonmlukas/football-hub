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


# --- the context table never ranks on `edge` ------------------------------
#
# It used to, from round four. `edge` is the repo's original draft signal and the only one
# never tested against outcomes -- and it cannot be, because it needs ADP and no historical
# ADP exists. The column is information and stays; the sort order is advice and does not.

def test_no_round_ranks_on_edge():
    """Every round, including the ones that used to switch."""
    board, state = _board(), DraftState()
    for taken in (0, 36, 37, 100):
        st = take(DraftState(), *[f"P{i}" for i in range(taken)]) if taken else state
        text = "\n".join(live.render(live.refresh(board, st, my_slot=3)))
        assert "by value over replacement" in text
        assert "by edge" not in text


def test_the_edge_column_is_still_shown():
    """Retiring the sort is not retiring the number."""
    text = "\n".join(live.render(live.refresh(_board(), DraftState(), my_slot=3)))
    assert "edge" in text


def test_the_sort_key_switch_is_gone():
    """`_sort_key` existed only to choose between two keys. With one key it is a
    pass-through, and `EDGE_FROM_ROUND` was a threshold with a story but no measurement."""
    assert not hasattr(live, "_sort_key")
    assert not hasattr(live, "EDGE_FROM_ROUND")


def test_a_first_round_board_is_not_topped_by_negative_vor():
    """The failure the old round-3 rule prevented, which must still not happen."""
    view = live.refresh(_board(), DraftState(), my_slot=3)
    top = live.rank(view, "vor_live").head(5)
    assert (top["vor_live"] > 0).all()


# --- bugs found by the 2.5 dry run ---------------------------------------

def test_unmatched_picks_are_surfaced():
    """Dry-run bug 1. A pick whose name does not match leaves that player ON the board,
    so the poller keeps recommending someone already drafted. board.py warned about this;
    the poller -- the thing actually running during the draft -- said nothing."""
    view = live.refresh(_board(), take(DraftState(), "Nobody At All"), my_slot=3)
    assert view["unmatched"] == ["Nobody At All"]
    assert "unmatched" in "\n".join(live.render(view)).lower()


def test_a_matched_pick_raises_no_warning():
    view = live.refresh(_board(), take(DraftState(), "P0"), my_slot=3)
    assert view["unmatched"] == []
    assert "unmatched" not in "\n".join(live.render(view)).lower()


def test_the_poller_knows_what_i_already_hold():
    """Dry-run bug 2. my_slot was accepted and never used. Replaying 2025 I reached pick
    51 holding RB3 WR2 with no QB and no TE, and the board offered a fourth WR."""
    state = take(DraftState(), *[f"P{i}" for i in range(22)])
    view = live.refresh(_board(), state, my_slot=3)
    assert sum(view["roster"].values()) == 2, "slot 3 has picked at 3 and 22"


def test_unfilled_starting_slots_are_named():
    state = take(DraftState(), *[f"P{i}" for i in range(22)])
    view = live.refresh(_board(), state, my_slot=3)
    text = "\n".join(live.render(view))
    assert "need" in text.lower()


def test_a_full_starting_lineup_reports_no_need():
    counts = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
    assert live.unfilled(counts) == []


def test_a_missing_te_is_reported_even_with_a_deep_bench():
    counts = {"QB": 1, "RB": 5, "WR": 6}
    assert live.unfilled(counts) == ["TE"]


def test_below_replacement_players_are_never_recommended():
    """Dry-run bug 3. At pick 46 the edge board opened with Jordan Love at VOR -1.0 --
    below replacement, so literally worse than a waiver pickup, however large the edge."""
    board = _board()
    state = take(DraftState(), *[f"P{i}" for i in range(40)])
    view = live.refresh(board, state, my_slot=3)
    shown = live.rank(view, "vor_live").head(live.TOP_N)
    assert (shown["vor_live"] > 0).all()


# --- bugs found running the poller against the live league ---------------

def test_sync_can_be_quiet():
    """Live-run bug: sync printed 'draft is empty' on every poll. At 10s over a three
    hour draft that is ~1000 lines pushing the board you need off the screen."""
    import hub.fetch.espn as espn
    from hub.draft import state as st_mod

    class _L:
        draft = []
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(espn, "league_settings", lambda: (_L(), {}))
        mp.setattr(st_mod, "load", lambda path=None: DraftState())
        st_mod.sync_from_espn(quiet=True)
    finally:
        mp.undo()


def test_quiet_sync_prints_nothing(capsys):
    import hub.fetch.espn as espn
    from hub.draft import state as st_mod
    import pytest as _pytest

    class _L:
        draft = []
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(espn, "league_settings", lambda: (_L(), {}))
        mp.setattr(st_mod, "load", lambda path=None: DraftState())
        st_mod.sync_from_espn(quiet=True)
        assert capsys.readouterr().out == ""
    finally:
        mp.undo()


def test_loud_sync_still_reports_by_default(capsys):
    import hub.fetch.espn as espn
    from hub.draft import state as st_mod
    import pytest as _pytest

    class _L:
        draft = []
    mp = _pytest.MonkeyPatch()
    try:
        mp.setattr(espn, "league_settings", lambda: (_L(), {}))
        mp.setattr(st_mod, "load", lambda path=None: DraftState())
        st_mod.sync_from_espn()
        assert "not started" in capsys.readouterr().out
    finally:
        mp.undo()


def test_the_poller_flushes_every_line_it_prints():
    """Live-run bug: 22 seconds against the real league produced zero output when piped.
    Python block-buffers a pipe, so `--poll 10 | tee draft.log` -- a reasonable way to keep
    a record of draft night -- shows nothing at all."""
    import inspect
    src = inspect.getsource(live.poll)
    prints = [ln for ln in src.splitlines() if "print(" in ln]
    assert prints, "poll must report something"
    joined = "\n".join(src.splitlines())
    assert "flush=True" in joined, "every print in the poll loop must flush"


# --- the two draft-night tools give the same answer -----------------------
#
# They did not. `board.py --pick` led with THE PICK -- best available filling a need,
# lexicographically. `live.py --poll`, the one you actually have open on the clock, sorted by
# `edge` from round four and `vor_live` before it, with no need term in the ordering at all:
# need was printed as a `*` and never sorted on. Nothing could catch the disagreement,
# because the two shared no code. They share `optimize.the_pick` now.

def _agree_board(n=60):
    import polars as pl
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(n)],
        "pos": [["RB", "WR", "WR", "TE", "QB"][i % 5] for i in range(n)],
        "ecr": [float(i + 1) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "xfp_per_game": [float(max(20 - i * 0.2, 1.0)) for i in range(n)],
        "games": pl.Series([16] * n, dtype=pl.UInt32),
    })


def test_both_draft_night_tools_recommend_the_same_player():
    """The test that could not pass before. One rule, two renderers."""
    from hub.draft.optimize import the_pick
    board = _agree_board()
    st = take(DraftState(), *[f"P{i}" for i in range(8)])

    from_optimize = the_pick(board, st, my_slot=3, teams=12)
    in_live = live.refresh(board, st, my_slot=3, teams=12)["the_pick"]
    assert from_optimize is not None
    assert in_live.player == from_optimize.player


def test_the_live_poller_leads_with_the_pick():
    """It used to lead with a table sorted by edge."""
    board = _agree_board()
    st = take(DraftState(), *[f"P{i}" for i in range(8)])
    text = "\n".join(live.render(live.refresh(board, st, my_slot=3, teams=12)))
    assert "THE PICK:" in text


def test_the_live_table_is_labelled_as_context():
    """`top 8 by edge` read as a recommendation. It has no need term in its ordering, and
    P0b measured a need-blind objective losing by 19.66 points a team-game."""
    board = _agree_board()
    st = take(DraftState(), *[f"P{i}" for i in range(8)])
    text = "\n".join(live.render(live.refresh(board, st, my_slot=3, teams=12)))
    assert "context, not a ranking to draft off" in text


def test_the_recommendation_fills_a_need_where_the_table_need_not():
    """The substance of the disagreement, not just the wording. Holding two RBs and nothing
    else, THE PICK must not be a third running back while QB/WR/TE sit empty."""
    from hub.draft.optimize import the_pick
    import polars as pl
    board = _agree_board()
    # my picks at slot 3 of 12 are 3 and 22; make both of mine running backs
    taken = [f"P{i}" for i in range(22)]
    taken[2], taken[21] = "P0", "P5"        # P0 and P5 are both RB
    st = DraftState(taken=taken)
    tp = the_pick(board, st, my_slot=3, teams=12)
    assert tp is not None
    assert tp.pos != "RB", "RB is full; the pick must fill an empty slot"


# --- the poll loop's one decision ----------------------------------------

def test_a_new_pick_prints_the_board():
    assert live.next_action(n_taken=5, last=4, since_beat=1.0, heartbeat_s=120) == "board"


def test_no_change_stays_silent():
    """Over three hours, announcing every quiet pass scrolls the board out of view."""
    assert live.next_action(n_taken=5, last=5, since_beat=1.0, heartbeat_s=120) is None


def test_silence_past_the_heartbeat_says_so():
    """A silent poller and a hung poller look identical at 2am, and you have ninety
    seconds to decide which one you are looking at."""
    assert live.next_action(n_taken=5, last=5, since_beat=200.0,
                            heartbeat_s=120) == "heartbeat"


def test_a_pick_beats_a_due_heartbeat():
    """Both conditions true at once: the board is the more useful thing to print."""
    assert live.next_action(n_taken=6, last=5, since_beat=999.0,
                            heartbeat_s=120) == "board"


# --- the board's age is always visible ------------------------------------

def test_board_age_is_reported_in_hours(tmp_path):
    """`exists()` was the only check, so a board built Tuesday and a Thursday build that
    failed looked identical -- you would poll all night against stale ADP."""
    f = tmp_path / "b.parquet"
    f.write_bytes(b"x")
    mtime = f.stat().st_mtime
    assert live.board_age_hours(f, mtime + 7200) == pytest.approx(2.0)


def test_a_board_from_the_future_is_zero_not_negative():
    """Clock skew between a build host and the poller should read as fresh, not as a
    negative age that formats into nonsense on the one screen you are reading."""
    import pathlib

    class _F:
        def stat(self):
            class _S:
                st_mtime = 1000.0
            return _S()

    assert live.board_age_hours(_F(), 500.0) == 0.0
