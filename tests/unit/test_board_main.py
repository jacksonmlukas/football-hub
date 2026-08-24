"""The board's CLI, and the decisions that used to live inside its print blocks.

`board.main()` was the only entry point in the repo that did not take `argv` -- the other
sixteen CLIs all did -- so the module with the most churn also had the least reachable
entry point, and all three bugs found in the draft-night rehearsal lived below it.

Two things are tested here that were previously unreachable:

  * `main(argv)` itself, on the paths that do not need the network.
  * The decisions extracted out of the report layer. `held_positions` is what makes THE
    PICK fill a need; `pick_notes` decides what is worth interrupting a drafter with. Both
    were expressions inside f-string blocks, which is why the dead guard in the first one
    survived so long.
"""
import collections

import polars as pl
import pytest

from hub.draft import board
from hub.draft.state import DraftState


def _board(names, pos=None, **cols):
    n = len(names)
    df = pl.DataFrame({"player": names, "pos": pos or ["RB"] * n})
    return df.with_columns(**cols) if cols else df


# --- main() takes argv -----------------------------------------------------

def test_show_slots_prints_the_league_and_exits_clean(capsys):
    """A path that touches no network: the shape, the pick schedule, and out."""
    assert board.main(["--show-slots"]) == 0
    out = capsys.readouterr().out
    assert "teams 12" in out and "slot 3" in out
    assert "QB1" in out and "WR3" in out


def test_show_slots_reports_the_shape_the_config_declares():
    """It used to print `board.SLOTS`, which was its own declaration of the roster."""
    from hub.config import RosterConfig, starters
    assert board.SLOTS == starters(RosterConfig())


def test_main_returns_an_exit_code():
    """`sys.exit(main())` at the bottom of the module needs one."""
    assert board.main(["--show-slots"]) == 0


# --- build reports what it did rather than being sniffed for it ------------

def test_a_fresh_report_has_nothing_and_says_so():
    r = board.BuildReport()
    assert set(r.degraded()) == {"sos", "td_luck", "durability", "adp",
                                 "scoring_checked", "roster_checked"}


def test_a_stage_that_ran_leaves_the_degraded_list():
    r = board.BuildReport()
    r.td_luck = True
    assert "td_luck" not in r.degraded()
    assert "sos" in r.degraded()


def test_the_report_distinguishes_a_stage_that_ran_from_one_that_did_not():
    """The case sniffing gets wrong.

    A stage that ran and returned an all-null column is indistinguishable from a stage that
    never ran, if all you have is `"td_luck" in board.columns`. Both frames below carry the
    column and neither carries a value; only the report can tell you which happened, and on
    draft night that difference is "nflverse was down" versus "nobody got lucky".
    """
    ran_but_empty = pl.DataFrame({"player": ["A"]}).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("td_luck"))
    never_ran = pl.DataFrame({"player": ["A"]}).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("td_luck"))
    assert ran_but_empty.columns == never_ran.columns      # sniffing cannot separate them

    ran = board.BuildReport()
    ran.td_luck = True
    assert "td_luck" not in ran.degraded()
    assert "td_luck" in board.BuildReport().degraded()


# --- held_positions: the decision behind "filling a need" ------------------

def test_held_positions_counts_your_roster_by_position():
    b = _board(["A", "B", "C", "D"], pos=["RB", "RB", "WR", "TE"])
    st = DraftState(taken=["A", "B", "C"])
    got = board.held_positions(b, st, my_slot=1, teams=1)
    assert got == {"RB": 2, "WR": 1}


def test_held_positions_is_empty_before_you_have_picked():
    assert board.held_positions(_board(["A"]), DraftState(taken=[])) == {}


def test_held_positions_ignores_players_drafted_by_other_teams():
    """`my_roster` walks the snake; only your own picks count toward your need."""
    b = _board(["A", "B", "C", "D"], pos=["RB", "WR", "TE", "QB"])
    st = DraftState(taken=["A", "B", "C", "D"])
    got = board.held_positions(b, st, my_slot=1, teams=2)
    assert sum(got.values()) < 4


def test_a_pick_not_on_the_board_does_not_become_a_null_position():
    """Kickers and defences are taken but never on the board. Counting a null `pos` would
    put a phantom position into the need calculation."""
    b = _board(["A"], pos=["RB"])
    st = DraftState(taken=["A", "Some Kicker"])
    assert board.held_positions(b, st, my_slot=1, teams=1) == {"RB": 1}


def test_the_dead_guard_is_gone():
    """It read `(remaining(board, st).is_empty() and []) or [...]`, and `(x and []) or y`
    is `y` for every x -- so the guard never fired and `remaining()` ran for nothing. The
    behaviour it *looked* like it wanted, an empty count once the board is exhausted, was
    never what it did; this pins what it actually does."""
    b = _board(["A"], pos=["RB"])
    st = DraftState(taken=["A"])           # board fully exhausted
    assert board.held_positions(b, st, my_slot=1, teams=1) == {"RB": 1}


# --- pick_notes: what is worth interrupting a drafter with -----------------

def test_no_notes_for_a_clean_player():
    assert board.pick_notes({"td_luck": 0.1, "missed": 0, "injury_status": "ACTIVE"}) == []


def test_touchdown_luck_is_noted_in_both_directions():
    """An `avoid` tag that ignored sign is a bug this repo has already had once."""
    up = board.pick_notes({"td_luck": 1.2})
    down = board.pick_notes({"td_luck": -1.2})
    assert "+1.20" in up[0]
    assert "-1.20" in down[0]


def test_touchdown_luck_below_the_threshold_is_not_worth_saying():
    assert board.pick_notes({"td_luck": 0.4}) == []
    assert board.pick_notes({"td_luck": -0.4}) == []


def test_a_null_touchdown_luck_is_not_an_extreme_one():
    """`abs(None)` raises; a degraded board carries nulls here."""
    assert board.pick_notes({"td_luck": None}) == []


def test_missed_games_are_noted():
    assert "missed 6 last season" in board.pick_notes({"missed": 6})


def test_a_player_who_missed_nothing_gets_no_note():
    assert board.pick_notes({"missed": 0}) == []


def test_only_flagworthy_designations_are_surfaced():
    """ACTIVE is not news. QUESTIONABLE is, even though it is not priced."""
    assert board.pick_notes({"injury_status": "ACTIVE"}) == []
    assert board.pick_notes({"injury_status": "QUESTIONABLE"}) == ["QUESTIONABLE"]
    assert board.pick_notes({"injury_status": "OUT"}) == ["OUT"]


def test_an_empty_row_is_not_a_crash():
    """A board built with every optional stage degraded still has to reach THE PICK."""
    assert board.pick_notes({}) == []


def test_notes_stack_in_a_stable_order():
    got = board.pick_notes({"td_luck": 1.5, "missed": 4, "injury_status": "OUT"})
    assert len(got) == 3
    assert got[0].startswith("td luck") and got[2] == "OUT"
