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

import polars as pl

from hub.draft import board


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


# `held_positions` and `pick_notes` moved to `hub.draft.optimize` with the code they test:
# they are decisions, and both draft-night tools need them. See tests/unit/test_optimize.py.
