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
import pytest

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


# --- the offline fallback --------------------------------------------------
#
# `build` degrades stage by stage, but only for advisory stages. The two spine fetches have
# no fallback inside it, so with no network the CLI raised ConnectionError and printed a
# traceback instead of a pick -- on the one night the repo exists for.

def _stub_board(tmp_path):
    """A minimal board on disk, standing in for the last good one."""
    p = tmp_path / "draft_board.parquet"
    pl.DataFrame({"player": ["A"], "pos": ["RB"], "vor": [1.0]}).write_parquet(p)
    return p


def test_a_failed_build_serves_the_last_good_board(monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise ConnectionError("ffopportunity is down")
    monkeypatch.setattr(board, "build", boom)
    got, _, age = board.build_or_last_good(path=_stub_board(tmp_path))
    assert got.height == 1
    assert age is not None and age >= 0.0
    out = capsys.readouterr().out
    assert "BUILD FAILED" in out and "ConnectionError" in out
    assert "ffopportunity is down" in out, "the operator has to know what actually broke"


def test_a_successful_build_reports_no_staleness(monkeypatch, tmp_path):
    """`age is None` is how the caller tells the two apart, so it has to mean fresh."""
    fresh = pl.DataFrame({"player": ["B"], "pos": ["WR"]})
    monkeypatch.setattr(board, "build", lambda *a, **k: (fresh, board.BuildReport()))
    got, _, age = board.build_or_last_good(path=_stub_board(tmp_path))
    assert age is None and got["player"].to_list() == ["B"]


def test_serving_last_good_does_not_rewrite_the_file(monkeypatch, tmp_path):
    """Rewriting resets the mtime, so the age just printed becomes a lie and every later
    run reports a fresh board that is as stale as the first failure."""
    p = _stub_board(tmp_path)
    import os
    os.utime(p, (1000.0, 1000.0))
    monkeypatch.setattr(board, "build", lambda *a, **k: (_ for _ in ()).throw(OSError("no net")))
    board.build_or_last_good(path=p)
    assert p.stat().st_mtime == 1000.0


def test_no_board_at_all_says_what_to_run(tmp_path):
    """The genuinely unrecoverable case still has to be a sentence, not a traceback."""
    import pytest
    with pytest.raises(FileNotFoundError, match="make draft"):
        board.last_good(path=tmp_path / "absent.parquet")


def test_the_age_is_reported_in_hours(tmp_path):
    import os
    p = _stub_board(tmp_path)
    os.utime(p, (1000.0, 1000.0))
    _, age = board.last_good(path=p, now=1000.0 + 7200)
    assert age == 7200 / 3600.0


def test_persisting_creates_both_parents(tmp_path):
    """`make draft` had never worked on a fresh clone: `data/processed/` is created by
    `hub.store`, the board does not go through `hub.store`, and every developer machine
    already had the directory. The first command in the README."""
    out = tmp_path / "site" / "data"
    path = tmp_path / "data" / "processed" / "draft_board.parquet"
    board._persist(_board(["A"], pos=["RB"]), out=out, path=path)
    assert path.exists(), "the board parquet's parent must be created"
    assert (out / "draft_board.json").exists()


def test_persisting_is_safe_to_repeat(tmp_path):
    out, path = tmp_path / "s", tmp_path / "d" / "b.parquet"
    df = _board(["A"], pos=["RB"])
    board._persist(df, out=out, path=path)
    board._persist(df, out=out, path=path)
    assert path.exists()


# --- main(), driven offline -------------------------------------------------
#
# `main` was the last uncovered block in this module: state transitions, persistence, the ADP
# archive and the report composition, none of which need a network once `build` is stubbed.

def _full_board(n=320):
    pos = ["QB", "RB", "WR", "WR", "TE", "RB", "WR"]
    return pl.DataFrame({
        "player": [f"Player {i:03d}" for i in range(n)],
        "pos": [pos[i % len(pos)] for i in range(n)],
        "team": ["KC"] * n,
        "ecr": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) / 10 for i in range(n)],
        "xfp_per_game": [20.0 - i * 0.05 for i in range(n)],
        "fp_over_expected": [float(i % 11) - 5 for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "proj_blend": [18.0 - i * 0.04 for i in range(n)],
        "consensus_rank": [float(i + 1) for i in range(n)],
        "games": pl.Series([14] * n, dtype=pl.UInt32),
    })


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """`main` with the network, the state file and the output paths all redirected."""
    b = _full_board()
    monkeypatch.setattr(board, "build_or_last_good",
                        lambda *a, **k: (b, board.BuildReport(adp=True), None))
    monkeypatch.setattr(board, "BOARD_PARQUET", tmp_path / "board.parquet")
    monkeypatch.setattr(board, "OUT", tmp_path / "site")
    from hub.draft import adp_history, state as state_mod
    monkeypatch.setattr(adp_history, "ARCHIVE", tmp_path / "adp")
    monkeypatch.setattr(state_mod, "STATE", tmp_path / "state.json")
    return b


def test_main_writes_the_board_and_the_site_copy(cli, tmp_path, capsys):
    assert board.main([]) == 0
    assert (tmp_path / "board.parquet").exists()
    assert (tmp_path / "site" / "draft_board.json").exists()
    assert "320 players" in capsys.readouterr().out


def test_main_archives_the_day_s_adp(cli, tmp_path, capsys):
    """ESPN does not retain historical ADP, so the build is the only chance to record it."""
    assert board.main([]) == 0
    assert "adp archived" in capsys.readouterr().out
    assert list((tmp_path / "adp").glob("date=*/adp.parquet"))


def test_a_recorded_pick_updates_the_state(cli, tmp_path, capsys):
    assert board.main(["--taken", "Player 000, Player 001"]) == 0
    assert "2 picks recorded" in capsys.readouterr().out


def test_a_mistyped_pick_is_warned_about(cli, capsys):
    """The misspelt player stays on the board as available, and the next recommendation can
    hand back someone already drafted."""
    assert board.main(["--taken", "Playr 000"]) == 0
    assert "NOT ON THE BOARD" in capsys.readouterr().out


def test_undo_removes_the_last_picks(cli, capsys):
    board.main(["--taken", "Player 000, Player 001"])
    capsys.readouterr()
    assert board.main(["--undo", "1"]) == 0
    assert "1 picks recorded" in capsys.readouterr().out


def test_reset_clears_the_state(cli, capsys):
    board.main(["--taken", "Player 000"])
    capsys.readouterr()
    assert board.main(["--reset"]) == 0
    assert "0 picks recorded" in capsys.readouterr().out


def test_the_pick_path_renders(cli, capsys):
    assert board.main(["--pick", "3"]) == 0
    out = capsys.readouterr().out
    assert "THE PICK" in out and "Also close" in out


def test_the_sos_path_renders(cli, capsys):
    b = cli.with_columns(pl.Series("wk15_17_sos", [1.0 + (i % 5) * 0.1 for i in range(cli.height)]))
    from hub.draft import board as bm
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(bm, "build_or_last_good", lambda *a, **k: (b, bm.BuildReport(adp=True), None))
    assert board.main(["--sos"]) == 0
    assert "strength of schedule" in capsys.readouterr().out
    mp.undo()


def test_a_stale_board_is_not_rewritten(monkeypatch, tmp_path, capsys):
    """Rewriting resets the mtime, so the age just printed becomes a lie."""
    b = _full_board()
    monkeypatch.setattr(board, "build_or_last_good",
                        lambda *a, **k: (b, board.BuildReport(), 3.5))
    monkeypatch.setattr(board, "BOARD_PARQUET", tmp_path / "board.parquet")
    monkeypatch.setattr(board, "OUT", tmp_path / "site")
    from hub.draft import state as state_mod
    monkeypatch.setattr(state_mod, "STATE", tmp_path / "state.json")
    assert board.main([]) == 0
    assert not (tmp_path / "board.parquet").exists(), "serving last-good must not rewrite it"
