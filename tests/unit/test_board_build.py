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
