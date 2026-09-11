"""`build()` assembled offline, which is where 141 of the repo's uncovered statements were.

It is the largest single block of untested code here, and everything it does that is not a
fetch is assembly: the consensus/xFP join, imputation for players with no prior season,
replacement level, VOR, and six optional stages that each degrade on their own.

The fetches are stubbed at module level rather than mocked deep, so what these exercise is the
assembly and the degradation branches -- the parts a network outage does not change and a
schema change does.

All offline.
"""
import json

import numpy as np
import polars as pl
import pytest
from polars.exceptions import ColumnNotFoundError

from hub.draft import board

# The DRAFT_BOARD contract requires 300 rows, so these are league-sized rather than toy.
# `games` is UInt32 because the contract declares it and now checks it.
N = 320
POS = ["QB", "RB", "WR", "WR", "TE", "RB", "WR"]          # roughly a real board's mix
NAMES = [f"Player {i:03d}" for i in range(N)]


def _xp(names=None, games=12):
    names = list(names) if names is not None else NAMES
    n = len(names)
    return pl.DataFrame({
        "player_id": [f"id{i}" for i in range(n)],
        "full_name": names,
        "position": [POS[i % len(POS)] for i in range(n)],
        "rec_exp": [50.0] * n, "rec_act": [48.0] * n,
        "xfp": [300.0 - 0.5 * i for i in range(n)],
        "fp": [290.0 - 0.5 * i for i in range(n)],
        "games": pl.Series([games] * n, dtype=pl.UInt32),
    }).with_columns([(pl.col("fp") - pl.col("xfp")).alias("fp_over_expected"),
                     (pl.col("xfp") / pl.col("games")).alias("xfp_per_game")])


def _ecr(names=None, rookie="Rookie"):
    """Everyone in `_xp`, plus one player with a rank and no prior season."""
    names = (list(names) if names is not None else NAMES) + ([rookie] if rookie else [])
    n = len(names)
    return pl.DataFrame({
        "player": names,
        "pos": [POS[i % len(POS)] for i in range(n)],
        "team": ["KC", "SF", "BUF", "PHI"][:1] * n,
        "ecr": [float(i + 1) for i in range(n)],
        "ecr_sd": [2.0] * n, "best": [1.0] * n, "worst": [20.0] * n,
    })


@pytest.fixture
def offline(monkeypatch):
    """Every fetch stubbed. Each optional stage raises unless a test says otherwise."""
    monkeypatch.setattr(board, "expected_points", lambda season: _xp())
    monkeypatch.setattr(board, "consensus", lambda as_of=None: _ecr())
    monkeypatch.setattr(board, "espn_adp", lambda *a, **k: None)

    def _boom(*a, **k):
        raise RuntimeError("stage unavailable")
    monkeypatch.setattr(board, "playoff_sos", _boom)
    monkeypatch.setattr(board, "bye_weeks", _boom)
    from hub.draft import durability
    from hub.draft import regression as td
    monkeypatch.setattr(td, "prior_season", _boom)
    monkeypatch.setattr(durability, "prior_season", _boom)
    monkeypatch.setattr(durability, "appearances", _boom)
    from hub.fetch import espn as espn_fetch
    monkeypatch.setattr(espn_fetch, "scoring_settings", _boom)
    monkeypatch.setattr(espn_fetch, "league_settings", _boom)
    return monkeypatch


# --- the spine assembles ----------------------------------------------------

def test_a_board_builds_with_every_optional_stage_failing(offline, capsys):
    """The ECR-only path. Six stages down and it still produces a board -- that is the
    property `build` exists to have, and it had no test."""
    b, report = board.build(league_size=12, season=2025)
    assert b.height > 0
    assert {"player", "pos", "ecr", "xfp_per_game", "vor", "consensus_rank"} <= set(b.columns)
    assert set(report.degraded()) >= {"sos", "td_luck", "durability", "adp", "bye"}


def test_vor_is_points_over_the_position_replacement(offline):
    b, _ = board.build()
    row = b.filter(pl.col("player") == NAMES[0]).to_dicts()[0]
    assert row["vor"] == pytest.approx(row["xfp_per_game"] - _replacement(b, b.filter(pl.col("player") == NAMES[0])["pos"][0]))


def _replacement(b, pos):
    r = b.filter(pl.col("pos") == pos).to_dicts()[0]
    return r["xfp_per_game"] - r["vor"]


def test_a_player_with_no_prior_season_is_imputed_not_dropped(offline):
    """A rookie has a consensus rank and no xFP. Leaving him null propagates a zero all the
    way into the season simulation; dropping him removes a draftable player from the board."""
    b, _ = board.build()
    rookie = b.filter(pl.col("player") == "Rookie")
    assert rookie.height == 1
    assert rookie["xfp_per_game"][0] is not None


def test_the_report_names_what_ran_rather_than_sniffing_columns(offline):
    """BuildReport exists because the report layer used to infer what had happened by
    checking whether a column was present -- and a stage that ran and returned all-nulls is
    indistinguishable from one that never ran."""
    _, report = board.build()
    assert report.adp is False and report.sos is False


# --- the optional stages, one at a time -------------------------------------

def test_sos_attaches_when_it_is_available(offline):
    offline.setattr(board, "playoff_sos", lambda **k: pl.DataFrame(
        {"team": ["KC"], "pos": ["RB"], "wk15_17_sos": [1.1],
         "sos_games": pl.Series([3], dtype=pl.UInt32)}))
    b, report = board.build()
    assert report.sos is True
    assert "wk15_17_sos" in b.columns


def test_adp_attaches_and_is_reported(offline):
    offline.setattr(board, "espn_adp", lambda *a, **k: pl.DataFrame(
        {"player": [NAMES[0], NAMES[1]], "adp": [1.5, 2.5], "proj_ppg": [18.0, 16.0],
         "injury_status": ["ACTIVE", "QUESTIONABLE"]}))
    b, report = board.build()
    assert report.adp is True
    assert b.filter(pl.col("player") == NAMES[0])["adp"][0] == pytest.approx(1.5)


def test_proj_blend_falls_back_when_espn_has_no_projection(offline):
    """`proj_blend` is a coalesce, so an all-null ESPN projection silently becomes xFP. That
    is the shape of the 2027 season bug -- worth pinning that the fallback is intentional."""
    offline.setattr(board, "espn_adp", lambda *a, **k: pl.DataFrame(
        {"player": [NAMES[0]], "adp": [1.5],
         "proj_ppg": pl.Series([None], dtype=pl.Float64),
         "injury_status": ["ACTIVE"]}))
    b, _ = board.build()
    row = b.filter(pl.col("player") == NAMES[0]).to_dicts()[0]
    assert row["proj_blend"] == pytest.approx(row["xfp_per_game"])


def test_a_historical_board_skips_the_espn_stages(offline, capsys):
    """ESPN publishes ADP, scoring and roster slots for the current season only, so asking
    for a 2022 board and reading 2026 ADP onto it is a contradiction, not a config choice."""
    _, report = board.build(season=2022, as_of="2022-09-01")
    assert report.adp is False and report.scoring_checked is False
    out = capsys.readouterr().out
    assert "current season only" in out


def test_replacement_level_needs_a_real_games_sample(offline):
    """A one-game player outranks every genuine starter on a per-game rate and then defines
    replacement level -- which is how WR replacement came out above RB."""
    offline.setattr(board, "expected_points", lambda season: _xp(games=1))
    b, _ = board.build()
    assert b.height > 0, "a board of short-sample players still builds"


# --- the stages that succeed, and the two mismatch warnings -----------------

def test_touchdown_luck_and_durability_attach_when_available(offline):
    from hub.draft import durability
    from hub.draft import regression as td
    # `attach` is patched rather than `prior_season`: this test is about `build` wiring the
    # stage in and flagging it, and both modules have their own tests for the join itself.
    offline.setattr(td, "prior_season", lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(durability, "prior_season", lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(durability, "appearances", lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(td, "attach", lambda b, s: b.with_columns(
        pl.lit(1.0).alias("td_luck")))
    offline.setattr(durability, "attach", lambda b, s, **kw: b.with_columns(
        pl.lit(2).cast(pl.Int64).alias("missed"), pl.lit(False).alias("sat_out")))
    b, report = board.build()
    assert report.td_luck is True and report.durability is True
    assert "td_luck" in b.columns and "missed" in b.columns


def test_a_served_report_names_the_columns_build_actually_leaves(offline):
    """What keeps `BuildReport.of_served` in step with the builder it reads after.

    A board recovered from disk is described by reading four columns off it, and that list
    is a second declaration of what the stages leave behind -- the kind that goes quietly
    wrong. So it is checked against a board `build` really produced: every stage that flags
    itself here, apart from the two checks that write no column at all, has to be
    recoverable from the frame. A stage that leaves a column and is missing from
    `STAGE_COLUMN` is a panel the served path silently drops.
    """
    from hub.draft import durability
    from hub.draft import regression as td
    offline.setattr(board, "playoff_sos", lambda **k: pl.DataFrame(
        {"team": ["KC"], "pos": ["RB"], "wk15_17_sos": [1.1],
         "sos_games": pl.Series([3], dtype=pl.UInt32)}))
    offline.setattr(td, "prior_season", lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(durability, "prior_season",
                    lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(durability, "appearances",
                    lambda season: pl.DataFrame({"player": NAMES[:3]}))
    offline.setattr(td, "attach", lambda b, s: b.with_columns(pl.lit(1.0).alias("td_luck")))
    offline.setattr(durability, "attach", lambda b, s, **kw: b.with_columns(
        pl.lit(2).cast(pl.Int64).alias("missed"), pl.lit(False).alias("sat_out")))
    offline.setattr(board, "espn_adp", lambda *a, **k: pl.DataFrame(
        {"player": [NAMES[0], NAMES[1]], "adp": [1.5, 2.5], "proj_ppg": [18.0, 16.0],
         "injury_status": ["ACTIVE", "QUESTIONABLE"]}))
    offline.setattr(board, "bye_weeks", lambda season: {"KC": 10})

    b, built = board.build()
    assert set(built.carried()) == set(board.STAGE_COLUMN), (
        "the five column-leaving stages all ran; the two checks write nothing and are "
        "correctly absent from STAGE_COLUMN")
    assert set(board.BuildReport.of_served(b).carried()) == set(built.carried())


# --- what each stage leaves, against the declaration that says so (issue #143) ----------
#
# `STAGE_COLUMN` answers "did this stage run", which is one column per stage. Consumers ask
# the other question -- which columns leave with an absorbed stage -- and every one of them
# used to answer it in its own prose. `STAGE_COLUMNS` is that answer once, and prose is
# exactly the thing that cannot be held to a build, so this is.


def _prior_td_season():
    """`regression.prior_season`'s frame, so the real `attach` has real work to do."""
    n = 8
    return pl.DataFrame({
        "player": NAMES[:n],
        "pos": [POS[i % len(POS)] for i in range(n)],
        "g": pl.Series([16] * n, dtype=pl.UInt32),
        "receiving_yards": [800.0] * n, "receiving_tds": [6.0] * n,
        "rushing_yards": [200.0] * n, "rushing_tds": [2.0] * n,
        "passing_yards": [0.0] * n, "passing_tds": [0.0] * n,
    })


def _prior_missed_season():
    """`durability.prior_season`'s frame. `ppg` has to clear `MIN_PPG` or nothing survives."""
    n = 8
    return pl.DataFrame({
        "player": NAMES[:n],
        "pos": [POS[i % len(POS)] for i in range(n)],
        "g": pl.Series([13] * n, dtype=pl.UInt32),
        "ppg": [14.0] * n,
    })


def _build_with_every_stage(offline, absorb: str | None = None):
    """Build with all four column-leaving stages running, bar the one named.

    Both prior-season modules keep their **real** `attach`; only the fetch under it is
    stubbed. A stand-in that writes the column the declaration names would prove that the
    stand-in agrees with the declaration, which is not the claim.

    Every stage is set either way round on every call, so two builds in one test do not
    inherit each other's patches.
    """
    from hub.draft import durability
    from hub.draft import regression as td
    offline.setattr(board, "playoff_sos",
                    _source_is_down if absorb == "sos" else lambda **k: pl.DataFrame(
                        {"team": ["KC"], "pos": ["RB"], "wk15_17_sos": [1.1],
                         "sos_games": pl.Series([3], dtype=pl.UInt32)}))
    offline.setattr(td, "prior_season",
                    _source_is_down if absorb == "td_luck"
                    else lambda season: _prior_td_season())
    offline.setattr(durability, "prior_season",
                    _source_is_down if absorb == "durability"
                    else lambda season: _prior_missed_season())
    # The sat-out cohort (#86): last preseason's consensus is the stubbed one, and one of its
    # names has no stats row, so the real `sat_out` has someone to flag.
    offline.setattr(durability, "appearances",
                    lambda season: pl.DataFrame({"player": NAMES[1:8]}))
    # The ADP stage absorbs nothing, so its outage is `espn_adp` returning None -- the one
    # path by which this stage is legitimately absent, and the one `build` is written for.
    offline.setattr(board, "espn_adp",
                    (lambda *a, **k: None) if absorb == "adp" else lambda *a, **k: _live_adp())
    # Byes come off the schedule, and the real `attach_bye` does the join (#226).
    offline.setattr(board, "bye_weeks",
                    _source_is_down if absorb == "bye" else lambda season: {"KC": 10})
    return board.build()


@pytest.mark.parametrize("stage", ["sos", "td_luck", "durability", "adp", "bye"])
def test_a_stage_leaves_exactly_the_columns_declared_for_it(offline, stage):
    """Absorb one stage and diff the board against the whole one. What went missing is what
    that stage leaves, measured rather than asserted, and `STAGE_COLUMNS` has to name it.

    Both directions fail here, which is the point of an equality rather than a subset: a
    column the stage leaves and the declaration omits is a consumer told an absorbed stage
    cost it less than it did, and a column the declaration names and the stage does not
    leave is one told it cost more. The ADP stage's nine are where this earns its keep --
    `injury_status` and `proj_ppg` reach the board through ESPN's ADP payload, so they
    leave with this stage and not with durability, and no reader guesses that.
    """
    whole, report = _build_with_every_stage(offline)
    assert set(report.carried()) == set(board.STAGE_COLUMNS), (
        "the baseline has to be a board with every column-leaving stage on it")

    thin, thin_report = _build_with_every_stage(offline, absorb=stage)
    assert getattr(thin_report, stage) is False, f"{stage} was meant to be absorbed"

    assert set(whole.columns) - set(thin.columns) == set(board.STAGE_COLUMNS[stage]), (
        f"the columns {stage} really leaves are not the ones STAGE_COLUMNS names for it")
    assert not set(thin.columns) - set(whole.columns), (
        f"absorbing {stage} added a column, so the diff is not only what it leaves")


def test_the_sentinel_is_one_of_the_columns_its_stage_leaves(offline):
    """`STAGE_COLUMN` is derived from the first column of each entry above, so this cannot
    drift -- but the *choice* can. A sentinel has to be a column the stage cannot finish
    without, since `of_served` reads its presence as the stage having run, and the entry
    above says the ordering carries that. This holds the derivation to a real build.
    """
    whole, _ = _build_with_every_stage(offline)
    for flag, col in board.STAGE_COLUMN.items():
        assert col == board.STAGE_COLUMNS[flag][0]
        assert col in whole.columns, f"{flag}'s sentinel is not on a board that ran it"
        thin, _report = _build_with_every_stage(offline, absorb=flag)
        assert col not in thin.columns, (
            f"{col} survives {flag} being absorbed, so a served board would claim the stage "
            "ran when it did not")


def test_a_scoring_mismatch_is_shouted_not_swallowed(offline, capsys):
    """Every projection is scored on the wrong weights until it is fixed, so this is one of
    the few things allowed to interrupt the operator."""
    from hub.fetch import espn as espn_fetch
    offline.setattr(espn_fetch, "scoring_settings", lambda: {"rec": 0.5})
    _, report = board.build()
    assert report.scoring_checked is True
    out = capsys.readouterr().out
    assert "SCORING MISMATCH" in out and "wrong weights" in out


def test_a_matching_scoring_setting_says_nothing(offline, capsys):
    from hub.fetch import espn as espn_fetch
    from hub.models.components import SCORING
    offline.setattr(espn_fetch, "scoring_settings", lambda: dict(SCORING))
    _, report = board.build()
    assert report.scoring_checked is True
    assert "SCORING MISMATCH" not in capsys.readouterr().out


def test_a_roster_mismatch_is_shouted(offline, capsys):
    """Replacement level and every VOR below assume this repo's roster shape."""
    from hub.fetch import espn as espn_fetch
    offline.setattr(espn_fetch, "league_settings",
                    lambda *a, **k: espn_fetch.LeagueView(None, {"RB": 4, "WR": 1}))
    _, report = board.build()
    assert report.roster_checked is True
    assert "ROSTER MISMATCH" in capsys.readouterr().out


def test_a_matching_roster_says_nothing(offline, capsys):
    from hub.config import RosterConfig, starters
    from hub.fetch import espn as espn_fetch
    slots = dict(starters(RosterConfig()))
    offline.setattr(espn_fetch, "league_settings",
                    lambda *a, **k: espn_fetch.LeagueView(None, slots))
    _, report = board.build()
    assert report.roster_checked is True
    assert "ROSTER MISMATCH" not in capsys.readouterr().out


# --- the degradation policy, now in one place ------------------------------
#
# This rule was written out five times, 48 lines, 31% of build(). The cost was never the
# duplication: `BuildReport` exists because consumers used to sniff for columns, and with
# five producers and no shared shape the consumers' guards drifted anyway -- one read
# `durability or adp`, another read `td_luck`, and a board built without an ESPN key raised
# ColumnNotFoundError before printing THE PICK.

def _rep():
    return board.BuildReport()


def test_a_failing_stage_leaves_the_board_and_the_flag_alone(capsys):
    b = pl.DataFrame({"player": ["A"]})
    rep = _rep()

    def _boom(_):
        raise RuntimeError("nope")
    out = board._stage(b, rep, "sos", "weeks 15-17 SoS", _boom)
    assert out is b and rep.sos is False
    assert capsys.readouterr().out == \
        "  weeks 15-17 SoS unavailable (RuntimeError); board built without it.\n"


def test_a_succeeding_stage_returns_the_new_board_and_flags_it(capsys):
    b = pl.DataFrame({"player": ["A"]})
    rep = _rep()
    out = board._stage(b, rep, "td_luck", "touchdown luck",
                       lambda x: x.with_columns(pl.lit(1.0).alias("td_luck")))
    assert "td_luck" in out.columns and rep.td_luck is True
    assert capsys.readouterr().out == ""


def test_a_check_stage_returns_none_and_keeps_the_board(capsys):
    """The two league checks warn rather than transform, and must not blank the board."""
    b = pl.DataFrame({"player": ["A"]})
    rep = _rep()
    out = board._stage(b, rep, "scoring_checked", "scoring check", lambda _: None)
    assert out is b and rep.scoring_checked is True


def test_a_historical_stage_is_announced_and_skipped(capsys):
    """ESPN publishes settings for the current season only. Outside a live build the stage is
    skipped rather than attempted and caught -- and the flag stays false either way, which is
    the correct record: it genuinely did not run."""
    b = pl.DataFrame({"player": ["A"]})
    rep = _rep()
    ran = []
    out = board._stage(b, rep, "scoring_checked", "scoring check", lambda x: ran.append(1),
                       live=False, skip_note="ESPN publishes settings for the current "
                                             "season only.")
    assert out is b and rep.scoring_checked is False and ran == []
    assert capsys.readouterr().out == (
        "  scoring check skipped: ESPN publishes settings for the current season only.\n")


def test_the_same_stage_runs_on_a_live_build(capsys):
    b = pl.DataFrame({"player": ["A"]})
    rep = _rep()
    board._stage(b, rep, "roster_checked", "roster check", lambda _: None,
                 live=True, skip_note="anything", on_fail="assuming this repo's shape.")
    assert rep.roster_checked is True


def test_the_failure_note_is_the_caller_s(capsys):
    """`assuming full PPR` and `board built without it` say different things to an operator
    on the clock, and the wording is draft-night output."""
    b = pl.DataFrame({"player": ["A"]})

    def _boom(_):
        raise KeyError("k")
    board._stage(b, _rep(), "scoring_checked", "scoring check", _boom,
                 on_fail="assuming full PPR.")
    assert capsys.readouterr().out == \
        "  scoring check unavailable (KeyError); assuming full PPR.\n"


def test_build_has_no_hand_rolled_degrade_blocks_left():
    """The property, not the instance: a sixth stage should be a declaration, not a block."""
    import ast
    import inspect
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(board)))
              if isinstance(n, ast.FunctionDef) and n.name == "build")
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]


# --- one policy, two kinds of failure --------------------------------------
#
# Folding the ADP stage into `_stage` stopped it bypassing the degradation policy, and put a
# stage that is *not* advisory under a handler written for stages that are. An advisory
# stage adds a signal the board is better for having: source down, board thinner, board
# still correct. This one computes the blended projection, both corrections and Corrected
# ADP -- what THE PICK ranks on -- so a board without it is not a thinner board, it is a
# board ranking on something else, while the run reports itself as having fallen back to
# consensus. True for an outage. False for a `ColumnNotFoundError` out of a refactor.
#
# Raised rather than asserted: the three below drive each kind of failure through and read
# the two outcomes. A test that inspected the handler's shape instead would pass on a
# handler widened back by one keyword, which is the failure the whole guard habit exists
# for. Issue #106.


def _live_adp():
    """ESPN's frame as `_parse_market` types it, so the stage has real work to do."""
    return pl.DataFrame({"player": [NAMES[0], NAMES[1]], "adp": [1.5, 2.5],
                         "proj_ppg": [18.0, 16.0],
                         "injury_status": ["ACTIVE", "QUESTIONABLE"]})


def _source_is_down(*a, **k):
    raise ConnectionError("nflverse is down")


def _a_refactor_renamed_a_column(*a, **k):
    raise ColumnNotFoundError("proj_ppg")


def test_an_advisory_stage_whose_source_is_down_still_degrades(offline, capsys):
    """The half that must not move. A board is better for having weeks 15-17 SoS on it and
    correct without it, so an unreachable source is a thinner board and a printed reason."""
    offline.setattr(board, "playoff_sos", _source_is_down)
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    b, report = board.build()
    assert b.height > 0 and report.sos is False
    assert "wk15_17_sos" not in b.columns, "thinner, and really thinner"
    assert "adp_corrected" in b.columns, "and still ranking on what THE PICK ranks on"
    assert "  weeks 15-17 SoS unavailable (ConnectionError); board built without it." \
        in capsys.readouterr().out


def test_a_built_board_carries_null_and_never_nan_where_a_player_is_undrafted(offline):
    """Two players priced, three hundred not (#243). On the served board of 2026-09-11 the
    unpriced ones carried NaN in `adp_corrected` -- a null turned NaN on a numpy round trip
    -- and `DRAFT_BOARD` now refuses that at the end of `build`, so this board building at
    all is half the claim; the other half is that the null lands exactly where `adp` is."""
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    b, report = board.build()
    assert report.adp
    assert b["adp"].null_count() == b.height - 2
    assert b["adp_corrected"].is_null().to_list() == b["adp"].is_null().to_list()
    for col in ("adp_corrected", "proj_correction", "edge", "vor_proj"):
        assert not b[col].is_nan().fill_null(False).any(), col


def test_a_defect_in_the_stage_the_pick_ranks_on_is_not_absorbed(offline, capsys):
    """The other half. The same policy, a failure it was never proved against: nothing in
    `_attach_market` reaches a source -- the outage is `espn_adp`'s, one layer up -- so a
    missing column here is a defect, and a defect that degrades is a board ranking on raw
    consensus with a line underneath saying that was the intention."""
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    offline.setattr(board, "_attach_market", _a_refactor_renamed_a_column)
    with pytest.raises(ColumnNotFoundError):
        board.build()
    out = capsys.readouterr().out
    assert "market corrections FAILED (ColumnNotFoundError" in out, \
        "and the operator is told which stage, which `BUILD FAILED` alone cannot say"
    assert "unavailable" not in out.split("market corrections")[-1]


def test_the_two_failures_are_two_different_nights(offline, tmp_path, capsys):
    """Both kinds, one fixture, read the way the operator reads them.

    The outage builds a board and prints what it was built without; the defect builds no
    board at all and reaches `build_or_last_good`, which serves the last good one and says
    so. Under one blanket handler these two printed the same line and returned boards that
    differed only in a column nobody was told about.
    """
    from hub.draft import report as report_mod
    last_good = tmp_path / "draft_board.parquet"
    pl.DataFrame({"player": ["Yesterday"], "pos": ["RB"], "vor": [1.0], "adp": [3.0]}
                 ).write_parquet(last_good)
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    offline.setattr(board, "playoff_sos", _source_is_down)

    built, outage_report, fresh = board.build_or_last_good(path=last_good)
    printed = capsys.readouterr().out
    reads = "\n".join(report_mod.built_or_served(outage_report, fresh))
    assert fresh is None and outage_report.served is False
    assert built.height > 1 and "adp_corrected" in built.columns
    assert "BUILD FAILED" not in printed
    assert "built without: sos" in reads and "SERVED BOARD" not in reads

    offline.setattr(board, "_attach_market", _a_refactor_renamed_a_column)
    served, defect_report, age = board.build_or_last_good(path=last_good)
    printed = capsys.readouterr().out
    reads = "\n".join(report_mod.built_or_served(defect_report, age))
    assert age is not None and defect_report.served is True
    assert served["player"].to_list() == ["Yesterday"], \
        "the last good board, not a board quietly missing Corrected ADP"
    assert "BUILD FAILED" in printed and "ColumnNotFoundError" in printed
    assert "SERVED BOARD" in reads and "built without" not in reads


def test_every_optional_stage_goes_through_the_helper():
    """Five did and the sixth did not, which is how the flag two renderers read ended up
    being the one set by hand."""
    import ast
    import inspect
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(board)))
              if isinstance(n, ast.FunctionDef) and n.name == "build")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_stage"]
    assert len(calls) == 7, f"expected seven staged stages, found {len(calls)}"

    # The AST, not the prose: `_attach_market`'s docstring quotes `report.adp = True` while
    # explaining why it no longer exists, and a string search matches the explanation.
    tree = ast.parse(inspect.getsource(board))
    by_hand = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
               for t in n.targets
               if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
               and t.value.id == "report"]
    assert not by_hand, "a stage flag is being set outside _stage"


def test_the_board_is_reproducible():
    """improvements.md #18. Two `board_as_of` calls returned the same 1,103 players in a
    different row order, and the draft indexes the board by row -- so every measurement
    drafting from it wobbled by ~0.04 points a team-week between identical runs.

    Two causes, both fixed: `playoff_sos._dvp_from_stats` handed a hash-ordered frame to a
    mean, and floating-point addition is not associative, so the same input gave answers
    differing at 7.1e-15; and `build` ended with `sort("ecr")`, whose ties ordered arbitrarily.

    Tested at the level that matters -- the final sort -- so it does not need a network.
    """
    import polars as pl
    frame = pl.DataFrame({
        "player": ["Zeta", "Alpha", "Mid", "Beta"],
        "ecr": [1.0, 1.0, 2.0, 1.0],
        "pos": ["WR"] * 4})
    runs = [frame.sample(fraction=1.0, shuffle=True, seed=s).sort(["ecr", "player"])
            for s in range(5)]
    assert all(r["player"].to_list() == runs[0]["player"].to_list() for r in runs), \
        "a tied ECR must order the same way whatever order the rows arrive in"
    assert runs[0]["player"].to_list() == ["Alpha", "Beta", "Zeta", "Mid"], \
        "and the tiebreaker never reorders across different ECRs"


def test_every_build_is_archived_immutably(tmp_path):
    """improvements.md #8. `build()` writes one flat `draft_board.parquet` and overwrites it,
    so every `make draft` destroyed the previous day's board -- and with it that day's ADP,
    which ESPN does not retain and which `fit_espn_weight`, the opponent model and validating
    `edge` all need. The fetch layer had already solved this: `hub.store.write` is "immutable
    dated partitions; corrections write a new file, nothing is overwritten".
    """
    import datetime as dt

    import polars as pl

    from hub import store
    from hub.draft import board as B

    frame = pl.DataFrame({"player": ["A"], "ecr": [1.0]})
    first = B._archive(frame, now=dt.datetime(2026, 9, 1, 12, 0, tzinfo=dt.UTC), base=tmp_path)
    second = B._archive(frame, now=dt.datetime(2026, 9, 2, 12, 0, tzinfo=dt.UTC), base=tmp_path)
    assert first is not None and second is not None
    assert first != second, "a second build must not overwrite the first"
    assert first.exists() and second.exists()
    assert store.tables(tmp_path) == {B.BOARD_TABLE}
    assert store.sql(f"SELECT count(*) AS n FROM {B.BOARD_TABLE}", base=tmp_path)["n"][0] == 2


def test_a_failed_archive_never_breaks_the_build(tmp_path, monkeypatch, capsys):
    """An archival side effect, not the product. A board that will not print because an
    archive write failed is exactly the operator-dependence CLAUDE.md warns about."""
    import polars as pl

    from hub import store
    from hub.draft import board as B

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(store, "write", boom)
    assert B._archive(pl.DataFrame({"player": ["A"]}), base=tmp_path) is None
    assert "archive skipped" in capsys.readouterr().out


def test_the_flat_board_is_still_where_everything_expects_it():
    """Additive on purpose: `last_good` reads this path, `adherence` copies it, `hub.inspect`
    special-cases it and `docs/draft-night.md` names it as the draft-night fallback."""
    from hub.draft.board import BOARD_PARQUET
    from hub.paths import BOARD_PARQUET as leaf
    assert BOARD_PARQUET is leaf
    assert BOARD_PARQUET.name == "draft_board.parquet"
    assert BOARD_PARQUET.parent.name == "processed"


def test_the_components_ride_alongside_the_total_without_moving_it():
    """`expected_points` gained eleven columns and must have changed none. The rebuild from
    components reproduces the published total to ~0.017 points a player-week, which would move
    `proj_blend` by half that -- immaterial as a projection, but improvements.md #18 records a
    Board change of that order reordering simulated drafts and moving the frozen weekly gate
    from +0.711 to +0.215. So the parts ride alongside; they do not replace.

    Constructed rather than fetched: comparing two live rebuilds cannot separate this change
    from upstream data moving underneath it, which is exactly what a first attempt at this
    check did.
    """
    import polars as pl

    from hub.models import components

    weekly = pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "full_name": ["A", "A", "B"],
        "position": ["WR", "WR", "RB"],
        "receptions_exp": [4.0, 6.0, 1.0],
        "receptions": [3.0, 7.0, 2.0],
        "total_fantasy_points_exp": [10.0, 14.0, 5.0],
        "total_fantasy_points": [9.0, 16.0, 6.0],
    })
    # the aggregation exactly as it stood before components were carried
    old = (weekly.group_by(["player_id", "full_name", "position"]).agg([
                pl.col("receptions_exp").sum().alias("rec_exp"),
                pl.col("receptions").sum().alias("rec_act"),
                pl.col("total_fantasy_points_exp").sum().alias("xfp"),
                pl.col("total_fantasy_points").sum().alias("fp"),
                pl.len().alias("games")])
            .with_columns([(pl.col("fp") - pl.col("xfp")).alias("fp_over_expected"),
                           (pl.col("xfp") / pl.col("games")).alias("xfp_per_game")]))

    comps = components.from_opportunity(weekly, by="player_id")
    comps = comps.rename({c: f"exp_{c}" for c in components.EXPECTED if c in comps.columns}
                         | {"xfp_per_game": "xfp_components_per_game"}).drop("games")
    new = old.join(comps, on="player_id", how="left")

    assert new.height == old.height, "the join must not fan a player out"
    for c in old.columns:
        assert old.sort("player_id")[c].equals(new.sort("player_id")[c]), f"{c} moved"
    assert "exp_receptions" in new.columns
    assert "xfp_components_per_game" in new.columns


def test_a_build_failure_with_no_parquet_falls_back_to_the_published_board(
        tmp_path, offline, capsys):
    """Issue #120. The fallback's own read could raise, and then named the wrong input.

    Narrowing the market stage's absorption made this reachable: the build now fails where it
    used to absorb, and on a machine with no parquet `last_good` raised `FileNotFoundError`
    whose remedy sentence is "Run `make draft` first" -- the command that just failed. Not an
    exotic state. It is a fresh clone, and it is every CI runner, because `data/processed/` is
    gitignored as redistributed third-party data.

    The committed artifact is what survives that, `docs/draft-night.md` already calls it the
    fallback, and nothing fell back to it.
    """
    from hub.draft import report as report_mod
    site = tmp_path / "site"
    site.mkdir()
    (site / "draft_board.json").write_text(json.dumps(
        [{"player": "Published", "pos": "RB", "vor": 1.0, "adp": 3.0}]))
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    offline.setattr(board, "_attach_market", _a_refactor_renamed_a_column)

    served, report, age = board.build_or_last_good(
        path=tmp_path / "nothing.parquet", site=site)
    printed = capsys.readouterr().out

    assert served["player"].to_list() == ["Published"]
    assert report.served and age is None, "the committed artifact carries no age to report"
    assert "BUILD FAILED" in printed and "serving the published one" in printed
    # The renderer already had this branch and nothing could reach it.
    assert "age unknown" in "\n".join(report_mod.built_or_served(report, age))


def test_with_neither_board_the_build_failure_is_what_escapes(tmp_path, offline):
    """The third criterion, and the one that decides whether a reader is helped.

    Both fallbacks are gone, so something has to be raised. Raising the missing file puts
    "Run `make draft` first" on the last line of the traceback -- an instruction to repeat the
    command that just failed -- and demotes the arithmetic defect that actually broke to a
    context line readers skip. Raised the other way round: the build failure escapes, and the
    missing file is its cause.
    """
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    offline.setattr(board, "_attach_market", _a_refactor_renamed_a_column)

    with pytest.raises(Exception) as caught:
        board.build_or_last_good(path=tmp_path / "nothing.parquet", site=tmp_path / "empty")
    assert not isinstance(caught.value, FileNotFoundError), (
        "the missing file is not what the operator has to act on")
    assert isinstance(caught.value.__cause__, FileNotFoundError), (
        "and it is not discarded either -- it is why the fallback could not cover")


def test_a_served_published_board_is_not_written_to_the_parquet(tmp_path, offline, capsys):
    """`main` used to persist on `stale_h is None`, which meant "built just now". The age is
    `None` on two paths now, and writing the published board to the parquet would date a
    top-300 artifact as a freshly built board -- the mtime lie issue #122 fixed for the
    roster, arriving here through a new path."""
    site = tmp_path / "site"
    site.mkdir()
    (site / "draft_board.json").write_text(json.dumps(
        [{"player": "Published", "pos": "RB", "vor": 1.0, "adp": 3.0}]))
    offline.setattr(board, "espn_adp", lambda *a, **k: _live_adp())
    offline.setattr(board, "_attach_market", _a_refactor_renamed_a_column)

    parquet = tmp_path / "nothing.parquet"
    _served, report, _age = board.build_or_last_good(path=parquet, site=site)
    assert report.served, "the guard `main` reads is the report, not the age"
    assert not parquet.exists(), "the published board must not become the built one"


# --- the two advisory stages that feed the one that is not (issue #121) ----

def test_absorbing_a_correction_stage_says_so_beside_the_ranking():
    """Issue #121. `built without: durability` was read as "the board is thinner". For a
    Correction it means something else: the correction never applied, so Corrected ADP --
    what THE PICK ranks on -- is a different order. The flags were already right; nothing
    connected them to the ranking that consumed their absence.
    """
    from hub.draft import report as report_mod
    ran = board.BuildReport(adp=True, td_luck=True, durability=True)
    assert ran.corrections_missing() == ()
    assert "CORRECTED ADP" not in "\n".join(report_mod.built_or_served(ran, None))

    without = board.BuildReport(adp=True, td_luck=True, durability=False)
    assert without.corrections_missing() == ("durability",)
    said = "\n".join(report_mod.built_or_served(without, None))
    assert "CORRECTED ADP is missing durability" in said
    assert "different order" in said and "not a thinner board" in said


def test_absorbing_the_touchdown_luck_stage_no_longer_claims_the_ranking_moved():
    """The report's half of #48, and the reason the term had to leave `CORRECTION_COLUMN`
    rather than merely stop being multiplied.

    A term left declared would print "CORRECTED ADP is missing touchdown luck" over a
    ranking that is identical either way, which is the report saying more than it knows --
    `BuildReport`'s own defect. The stage still ran and is still reported as degraded when
    it fails; what it no longer does is warn about a ranking.
    """
    from hub.draft import report as report_mod
    absorbed = board.BuildReport(adp=True, td_luck=False, durability=True)
    assert absorbed.corrections_missing() == ()
    said = "\n".join(report_mod.built_or_served(absorbed, None))
    assert "CORRECTED ADP" not in said
    assert "td_luck" in said, "it is still named as a stage that did not run"


def test_a_board_with_no_adp_claims_no_missing_corrections():
    """An ECR-only board did not compute a corrected ranking at all, so it has none to be
    missing terms from. Saying "missing touchdown luck" there would point at the wrong
    thing -- the ranking is raw consensus, and the board already says so."""
    ecr_only = board.BuildReport(adp=False, td_luck=False, durability=False)
    assert ecr_only.corrections_missing() == ()


def test_the_corrections_a_report_names_are_the_ones_declared():
    """`CORRECTION_STAGE` is derived from the two dicts that already exist rather than being
    a third list of the same terms. This holds the derivation to them."""
    assert set(board.CORRECTION_STAGE) == set(board.CORRECTION_COLUMN)
    for term, flag in board.CORRECTION_STAGE.items():
        assert board.STAGE_COLUMN[flag] == board.CORRECTION_COLUMN[term], (
            f"{term} reads {board.CORRECTION_COLUMN[term]}, which no stage leaves")
    short = board.BuildReport(adp=True)
    assert set(short.corrections_missing()) == set(board.CORRECTION_COLUMN), (
        "a board that ran none of the correction stages is missing every declared term")


def test_a_third_correction_needs_no_second_declaration_to_reach_a_gate(monkeypatch):
    """A **Correction** is a shape and not a module (ADR-0021), so these two dicts are the
    only place the repo says how many there are.

    `corrections_missing` used to name "touchdown luck" and "durability" in its own body,
    which made `CORRECTION_COLUMN` -- written to say which column each term reads --
    consumed by nothing at all. A third Correction wired into a stage and a column would
    then have been absent from every Gate's view of the board while the flag beside it said
    its stage had run: `BuildReport`'s own defect, one level up. Adding a term here rather
    than asserting the two the repo has today is what tells the derivation from the
    coincidence that the hand-written pair matched it.
    """
    monkeypatch.setitem(board.CORRECTION_STAGE, "snap share", "sos")
    named = board.BuildReport(adp=True, td_luck=True, durability=True, sos=False)
    assert named.corrections_missing() == ("snap share",)
    ran = board.BuildReport(adp=True, td_luck=True, durability=True, sos=True)
    assert ran.corrections_missing() == ()


def _corrected_ranking(frame: pl.DataFrame) -> list[str]:
    """The board's own arithmetic from `proj_blend` to the order THE PICK reads."""
    from hub.draft import durability
    from hub.draft.optimize import corrected_adp

    raw = frame["proj_blend"]
    f = durability.correct_projection(frame)
    f = f.with_columns((pl.col("proj_blend") - raw).alias("proj_correction"))
    return (f.with_columns(corrected_adp(f).alias("adp_corrected"))
             .sort("adp_corrected")["player"].to_list())


def _synthetic_board(n: int = 120) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "player": [f"p{i}" for i in range(n)],
        "pos": ["QB", "WR", "RB", "TE"] * (n // 4),
        "adp": np.sort(rng.uniform(1, 180, n)),
        "proj_blend": np.sort(rng.uniform(60, 320, n))[::-1],
        "games": [17] * n,
        "td_luck": rng.normal(0, 3, n),
        "missed": rng.integers(0, 6, n).astype(float),
    })


def test_a_missing_correction_actually_moves_the_corrected_ranking():
    """The demonstration the ticket asks for, as numbers rather than as an argument.

    `correct_projection` returns the frame untouched when its column is absent, so this is
    the arithmetic an absorbed stage produces. If this ever stops moving the ranking,
    durability really is advisory and the note above should go.
    """
    b = _synthetic_board()
    assert _corrected_ranking(b.drop("missed")) != _corrected_ranking(b)


def test_the_touchdown_luck_stage_no_longer_moves_the_corrected_ranking():
    """The other half of the same claim, and the one #48 created.

    This is the test the removal is measurable by: the same frame with and without the
    `td_luck` column must now produce the *same* order, because nothing multiplies it. Until
    #48 these two rankings differed -- 346 of 457 players moved on the live board of
    2026-09-06 -- and `corrections_missing` printed a warning about the difference. Both
    facts had to stop being true together, or the report would be describing a board that no
    longer exists.
    """
    b = _synthetic_board()
    assert _corrected_ranking(b.drop("td_luck")) == _corrected_ranking(b)


def test_a_routed_board_reads_one_archive_entry_twice_and_gets_the_same_frame(
        tmp_path, monkeypatch):
    """#36's third criterion. The board's two archive reads go through the fetch layer now,
    so a build at a fixed as-of has to answer from the pinned entry rather than from whatever
    the wire says this minute -- twice, identically.

    Asserted at the loader rather than through a whole board build, because a build needs
    ESPN and a scoring check besides, and a test that can only run with a network is a test
    nobody re-runs. What routing changed is which path the archive reads take; that is what
    is checked.
    """
    from hub.fetch import nflverse as nv

    # The nine columns `FF_RANKINGS` declares. Building a narrower frame here would be
    # testing the contract's refusal rather than the routing, and it did on the first run.
    frame = pl.DataFrame({
        "page_type": ["redraft-overall"] * 2,
        "player": ["A", "B"], "pos": ["WR", "RB"], "team": ["PHI", "DAL"],
        "ecr": [1.0, 2.0], "best": [1.0, 1.0], "worst": [3.0, 4.0], "sd": [0.5, 0.6],
        "scrape_date": ["2026-08-20", "2026-08-20"],
    })
    calls = []
    monkeypatch.setattr(nv, "RAW", tmp_path / "raw")
    monkeypatch.setattr(nv, "_raw_ff_rankings", lambda pages: calls.append(pages) or frame)

    first = nv.load_rankings("all", as_of="2026-09-01")
    second = nv.load_rankings("all", as_of="2026-09-01")
    assert first.equals(second), "a pinned as-of did not reproduce"
    assert len(calls) == 1, "the second read went to the wire, so the as-of is not pinned"
    # And the run can say what it read, which is the point of routing it at all.
    assert any(p.source == "ff_rankings" for p in nv.pins_this_run())


def test_the_bye_stage_places_every_player_by_his_team(offline):
    """#226. The schedule says when each team sits; the board carries it per player, and a
    schedule outage leaves the board without the column rather than without a board."""
    offline.setattr(board, "bye_weeks", lambda season: {"KC": 10})
    b, report = board.build()
    assert report.bye is True
    placed = b.filter(pl.col("team") == "KC")
    assert placed.height > 0 and (placed["bye_week"] == 10).all()
