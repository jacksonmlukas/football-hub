"""`build()` assembled offline, which is where 141 of the repo's uncovered statements were.

It is the largest single block of untested code here, and everything it does that is not a
fetch is assembly: the consensus/xFP join, imputation for players with no prior season,
replacement level, VOR, and six optional stages that each degrade on their own.

The fetches are stubbed at module level rather than mocked deep, so what these exercise is the
assembly and the degradation branches -- the parts a network outage does not change and a
schema change does.

All offline.
"""
import polars as pl
import pytest

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
    from hub.draft import durability
    from hub.draft import regression as td
    monkeypatch.setattr(td, "prior_season", _boom)
    monkeypatch.setattr(durability, "prior_season", _boom)
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
    assert set(report.degraded()) >= {"sos", "td_luck", "durability", "adp"}


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
    offline.setattr(td, "attach", lambda b, s: b.with_columns(
        pl.lit(1.0).alias("td_luck")))
    offline.setattr(durability, "attach", lambda b, s: b.with_columns(
        pl.lit(2).cast(pl.Int64).alias("missed")))
    b, report = board.build()
    assert report.td_luck is True and report.durability is True
    assert "td_luck" in b.columns and "missed" in b.columns


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


def test_the_market_block_degrades_instead_of_serving_stale(offline, capsys):
    """The ADP stage was the one stage outside `_stage`: `report.adp = True` set by hand, and
    ~40 lines after it under no `try`. So a failure in the corrections did not degrade -- it
    propagated to `build_or_last_good`, which served yesterday's board. A far bigger hammer
    than ECR-only mode, for a stage that is advisory by design."""
    offline.setattr(board, "espn_adp", lambda *a, **k: pl.DataFrame(
        {"player": [NAMES[0]], "adp": [1.5], "proj_ppg": [18.0],
         "injury_status": ["ACTIVE"]}))

    def _boom(*a, **k):
        raise RuntimeError("market maths broke")
    offline.setattr(board, "_attach_market", _boom)
    b, report = board.build()
    assert report.adp is False, "a failed stage must not leave its flag set"
    assert "adp" not in b.columns, "the board reverts, so it really is ECR-only"
    assert "running ECR-only mode" in capsys.readouterr().out


def test_every_optional_stage_goes_through_the_helper():
    """Five did and the sixth did not, which is how the flag two renderers read ended up
    being the one set by hand."""
    import ast
    import inspect
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(board)))
              if isinstance(n, ast.FunctionDef) and n.name == "build")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_stage"]
    assert len(calls) == 6, f"expected six staged stages, found {len(calls)}"

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
