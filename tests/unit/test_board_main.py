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


# --- the ECR-only path, which is a documented fallback and used to crash ----
#
# `docs/draft-night.md` lists `by consensus (ECR) -- no ADP today` as "a good fallback, not a
# degraded one". It was unreachable: two report sections read `adp` while guarding on a
# different flag, so a board with no ESPN ADP raised ColumnNotFoundError before THE PICK.

def _ecr_only():
    """What `build` returns with no ESPN key: everything but ADP."""
    return pl.DataFrame({"player": ["A", "B"], "pos": ["RB", "WR"],
                         "td_luck": [1.0, -1.0], "missed": [5, 6],
                         "injury_status": ["QUESTIONABLE", None]})


def test_the_injury_report_is_skipped_without_adp():
    """Both halves are scoped to `inside ADP 120` and both print an ADP, so there is nothing
    to show -- and the filter used to raise."""
    r = board.BuildReport(durability=True, adp=False)
    board._print_injuries(_ecr_only(), r)          # must not raise


def test_the_touchdown_luck_report_is_skipped_without_adp():
    """It reaches nflverse, so it can succeed while ESPN ADP fails."""
    r = board.BuildReport(td_luck=True, adp=False)
    board._print_td_luck(_ecr_only(), r)           # must not raise


def test_those_reports_still_run_when_adp_is_present():
    """The guard must not have turned them off altogether."""
    df = _ecr_only().with_columns(pl.Series("adp", [10.0, 20.0]))
    board._print_injuries(df, board.BuildReport(durability=True, adp=True))
    board._print_td_luck(df, board.BuildReport(td_luck=True, adp=True))


def test_the_corrections_note_is_route_aware():
    """"bounded at 20% of ADP" printed against a board with no ADP is the kind of sentence
    that gets believed at 9pm. Corrections move a player relative to ADP; ECR has none."""
    from hub.draft.optimize import ThePick

    def _tp(via, notes=("missed 1 last season",)):
        return ThePick(player="A", pos="WR", via=via, rank=1.5, rank_label="ECR",
                       notes=list(notes))

    corrected = "\n".join(board.corrections_note(_tp("draft market, corrected")))
    assert "bounded at 20% of ADP" in corrected

    ecr = "\n".join(board.corrections_note(_tp("consensus (ECR) -- no ADP today")))
    assert "bounded at 20% of ADP" not in ecr
    assert "NOT folded into this" in ecr


def test_the_uncorrected_adp_route_does_not_claim_corrections_either():
    """`by draft market (ADP)` means the board predates corrected ADP -- nothing was folded
    in there either, and saying otherwise would misdescribe the ranking."""
    from hub.draft.optimize import ThePick
    tp = ThePick(player="A", pos="WR", via="draft market (ADP)", rank=3.0,
                 rank_label="ADP", notes=["td luck +2.0/gm"])
    assert "bounded at 20% of ADP" not in "\n".join(board.corrections_note(tp))


def test_a_pick_with_no_notes_says_nothing():
    from hub.draft.optimize import ThePick
    tp = ThePick(player="A", pos="WR", via="consensus (ECR) -- no ADP today",
                 rank=1.5, rank_label="ECR", notes=[])
    assert board.corrections_note(tp) == []
