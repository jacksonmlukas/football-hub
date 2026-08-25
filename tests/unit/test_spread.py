"""Whether weekly spread is a property of the player or only of his mean.

`sd = K[position] * sqrt(mu)` makes spread a deterministic function of the mean, which is
why ADR-0012 measured the lineup optimiser at +0.00: ordering by mu and ordering by any
mean-variance objective are the same ordering. These tests pin the gate that decides
whether anything richer earns its way in.

All offline.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import spread
from hub.models.predict import WEEKLY_K


def _stats(rows):
    """(season, week, player_id, position, points)."""
    cols = ["season", "week", "player_id", "position", "fantasy_points_ppr"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"season": pl.Int32, "week": pl.Int32, "player_id": pl.Utf8,
                                "position": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _season(pid, pts, season=2023, pos="WR"):
    return [(season, w, pid, pos, p) for w, p in enumerate(pts, start=1)]


# --- the observation set --------------------------------------------------

def test_k_is_the_realised_spread_over_root_mean():
    pts = [10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 12.0, 18.0, 22.0, 8.0]
    ps = spread.player_seasons(_stats(_season("a", pts)))
    assert ps.height == 1
    mu, sd = float(np.mean(pts)), float(np.std(pts, ddof=1))
    assert ps["mu"][0] == pytest.approx(mu)
    assert ps["k"][0] == pytest.approx(sd / np.sqrt(mu))


def test_a_short_season_is_excluded():
    """Below eight games the sd is a handful of numbers."""
    assert spread.player_seasons(_stats(_season("a", [10.0, 20.0, 5.0]))).is_empty()


def test_a_fringe_scorer_is_excluded():
    """Below 3 ppg, sd/sqrt(mu) is dominated by whether he happened to score once."""
    assert spread.player_seasons(_stats(_season("a", [0.0, 1.0] * 6))).is_empty()


def test_a_player_with_no_variance_is_excluded():
    """k = 0 goes to -inf in logs and poisons the shrinkage fit."""
    assert spread.player_seasons(_stats(_season("a", [10.0] * 12))).is_empty()


def test_the_postseason_is_excluded():
    rows = _season("a", [10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 12.0, 18.0])
    df = _stats(rows).with_columns(pl.lit("REG").alias("season_type"))
    post = _stats([(2023, 19, "a", "WR", 99.0)]).with_columns(
        pl.lit("POST").alias("season_type"))
    ps = spread.player_seasons(pl.concat([df, post]))
    assert ps["games"][0] == 8, "a playoff explosion is not part of the regular-season spread"


def test_kickers_and_defences_are_excluded():
    assert spread.player_seasons(_stats(_season("k", [10.0, 2.0] * 6, pos="K"))).is_empty()


def test_missing_columns_raise():
    with pytest.raises(ValueError, match="fantasy_points_ppr"):
        spread.player_seasons(pl.DataFrame({"season": [2023], "week": [1],
                                            "player_id": ["a"], "position": ["WR"]}))


# --- snap share and the crosswalk ----------------------------------------

def _snaps(rows):
    cols = ["season", "week", "pfr_player_id", "position", "offense_pct"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"season": pl.Int32, "week": pl.Int32,
                                "pfr_player_id": pl.Utf8, "position": pl.Utf8,
                                "offense_pct": pl.Float64})


_XW = pl.DataFrame({"pfr_id": ["P1"], "gsis_id": ["a"]})


def test_snap_share_arrives_as_a_fraction_however_nflverse_ships_it():
    """nflverse has shipped this both ways. Detecting beats assuming."""
    frac = spread.snap_usage(_snaps([(2023, w, "P1", "WR", 0.5) for w in range(1, 9)]), _XW)
    pct = spread.snap_usage(_snaps([(2023, w, "P1", "WR", 50.0) for w in range(1, 9)]), _XW)
    assert frac["snap_pct"][0] == pytest.approx(0.5)
    assert pct["snap_pct"][0] == pytest.approx(0.5)


def test_a_growing_role_has_positive_drift():
    """The term docs/weekly-spread.md accuses of masquerading as spread."""
    rows = [(2023, w, "P1", "WR", 0.1 * w) for w in range(1, 11)]
    assert spread.snap_usage(_snaps(rows), _XW)["drift"][0] == pytest.approx(0.1)


def test_an_unmatched_player_gets_null_snap_share_not_zero():
    """Zero would assert he never played; null says we do not know."""
    ps = spread.player_seasons(
        _stats(_season("a", [10.0, 20.0, 5.0, 15.0, 25.0, 12.0, 18.0, 22.0])),
        _snaps([(2023, w, "P9", "WR", 0.5) for w in range(1, 9)]), _XW)
    assert ps["snap_pct"][0] is None


# --- pairing --------------------------------------------------------------

def _two_seasons(pid="a", pos="WR", a_pts=None, b_pts=None):
    a_pts = a_pts or [10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 12.0, 18.0]
    b_pts = b_pts or [11.0, 21.0, 31.0, 6.0, 16.0, 26.0, 13.0, 19.0]
    return _stats(_season(pid, a_pts, 2023, pos) + _season(pid, b_pts, 2024, pos))


def test_a_pair_carries_last_years_role_and_this_years_outcome():
    pr = spread.pairs(spread.player_seasons(_two_seasons()))
    assert pr.height == 1 and pr["season"][0] == 2024
    assert set(pr.columns) >= {"k_prev", "mu_next", "sd_next"}


def test_no_outcome_season_feature_reaches_the_features():
    """Every predictor is suffixed `_prev`. `mu_next` is shared by both arms by design."""
    pr = spread.pairs(spread.player_seasons(_two_seasons()))
    leaked = [c for c in pr.columns
              if c.endswith("_next") and c not in ("mu_next", "sd_next")]
    assert leaked == []


def test_a_player_missing_either_season_is_not_paired():
    assert spread.pairs(spread.player_seasons(
        _stats(_season("a", [10.0, 20.0, 30.0, 5.0, 15.0, 25.0, 12.0, 18.0])))).is_empty()


# --- the candidates -------------------------------------------------------

def _pairs(rows):
    """(season, position, k_prev, mu_next, sd_next), other features null."""
    df = pl.DataFrame(
        {"season": [r[0] for r in rows], "position": [r[1] for r in rows],
         "k_prev": [r[2] for r in rows], "mu_next": [r[3] for r in rows],
         "sd_next": [r[4] for r in rows],
         "player_id": [str(i) for i in range(len(rows))]})
    return df.with_columns([pl.lit(None, pl.Float64).alias(f"{f}_prev")
                            for f in spread.FEATURES] + [pl.lit(1.0).alias("mu_prev")])


def test_the_positional_arm_is_exactly_the_shipped_formula():
    pr = _pairs([(2024, "WR", 3.0, 16.0, 9.0)])
    got = spread._predict(pr, "positional")[0]
    assert got == pytest.approx(WEEKLY_K["WR"] * np.sqrt(16.0))


def test_shrinkage_of_zero_is_the_positional_arm():
    pr = _pairs([(2024, "RB", 5.0, 9.0, 20.0)])
    assert spread._predict(pr, "own_k", w=0.0)[0] == pytest.approx(
        spread._predict(pr, "positional")[0])


def test_shrinkage_of_one_uses_the_players_own_k():
    pr = _pairs([(2024, "RB", 5.0, 9.0, 20.0)])
    assert spread._predict(pr, "own_k", w=1.0)[0] == pytest.approx(5.0 * 3.0)


def test_shrinkage_is_geometric_not_arithmetic():
    """Averaging k=0.5 and k=2.0 to 1.25 rather than 1.0 biases every prediction up."""
    pr = _pairs([(2024, "WR", 4.0, 1.0, 1.0)])
    base = WEEKLY_K["WR"]
    assert spread._predict(pr, "own_k", w=0.5)[0] == pytest.approx(np.sqrt(4.0 * base))


def test_a_player_with_no_prior_k_falls_back_to_his_position():
    pr = _pairs([(2024, "TE", None, 16.0, 8.0)])
    assert spread._predict(pr, "own_k", w=1.0)[0] == pytest.approx(
        WEEKLY_K["TE"] * 4.0)


def test_the_usage_arm_with_no_coefficients_reproduces_the_shipped_model():
    """It is fitted as a *residual* from `positional`, so it can only be adopted by
    earning it -- a zero coefficient vector is exactly the model it must beat."""
    pr = _pairs([(2024, "WR", 3.0, 16.0, 9.0)])
    assert spread._predict(pr, "usage", coef={})[0] == pytest.approx(
        spread._predict(pr, "positional")[0])


def test_an_unknown_candidate_raises():
    with pytest.raises(ValueError, match="unknown candidate"):
        spread._predict(_pairs([(2024, "WR", 3.0, 16.0, 9.0)]), "vibes")


# --- fitting --------------------------------------------------------------

def test_shrinkage_goes_high_when_own_k_is_perfectly_persistent():
    rows = [(2024, "WR", 1.0 + 0.05 * i, 16.0, (1.0 + 0.05 * i) * 4.0) for i in range(40)]
    assert spread.fit_shrinkage(_pairs(rows)) >= 0.9


def test_shrinkage_goes_to_zero_when_own_k_is_noise():
    rng = np.random.default_rng(0)
    rows = [(2024, "WR", float(rng.uniform(1.0, 4.0)), 16.0, WEEKLY_K["WR"] * 4.0)
            for _ in range(200)]
    assert spread.fit_shrinkage(_pairs(rows)) <= 0.1


def test_a_thin_training_set_fits_no_usage_coefficients():
    """Six features on twenty rows fits noise. Returning nothing falls back to shipped."""
    rows = [(2024, "WR", 3.0, 16.0, 9.0) for _ in range(20)]
    train = _pairs(rows).with_columns(
        (pl.col("sd_next") / pl.col("mu_next").sqrt()).alias("k_next_implied"))
    assert spread.fit_usage(train) == {}


def test_held_out_nulls_are_filled_from_the_training_mean_not_their_own():
    """Filling from the held-out set would leak the evaluation set into the features."""
    coef = {"_intercept": 0.0, "tgt_share": 1.0, "_mean_tgt_share": 0.25}
    pr = _pairs([(2024, "WR", 3.0, 16.0, 9.0)])          # tgt_share_prev is null
    got = spread._predict(pr, "usage", coef=coef)[0]
    assert got == pytest.approx(WEEKLY_K["WR"] * 4.0 * np.exp(0.25))


# --- the gate -------------------------------------------------------------

def _errs(gain, noise, seasons=(2024, 2025), n=400, seed=0):
    """Per-observation errors where `own_k` beats `positional` by exactly `gain` per season.

    The noise is antithetic -- every draw appears as +e and -e -- so each season's mean
    difference is exactly `gain` while its spread is `noise`. That separates the two halves
    of the gate cleanly: `gain` decides which seasons win, `noise` decides whether the win
    is distinguishable from nothing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for si, season in enumerate(seasons):
        e = rng.normal(0, noise, n // 2)
        for d in np.concatenate([e, -e]):
            base = 5.0
            rows.append((season, base, base - gain[si] + float(d), base + 0.1))
    return pl.DataFrame(
        {"season": [r[0] for r in rows], "w": [0.5] * len(rows),
         "err_positional": [r[1] for r in rows], "err_own_k": [r[2] for r in rows],
         "err_usage": [r[3] for r in rows]})


def test_a_candidate_that_wins_every_season_and_clears_two_se_is_adopted():
    winner, text = spread.verdict(_errs([0.5, 0.5], noise=1.0))
    assert winner == "own_k" and text.startswith("ADOPT")


def test_winning_on_average_but_losing_a_season_is_not_enough():
    """That is how a fit gets adopted on one lucky year."""
    winner, text = spread.verdict(_errs([2.0, -0.5], noise=1.0))
    assert winner == "positional" and text.startswith("KEEP")


def test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted():
    """The half of the gate the first version of `verdict` was missing: it checked only
    that every season won, so a consistently-signed but meaningless gain passed."""
    errs = _errs([0.2, 0.2], noise=3.0)   # 1.9 se pooled, positive every season
    per = spread.summarise(errs)
    assert bool((per["mae_own_k"].to_numpy() < per["mae_positional"].to_numpy()).all()), \
        "the setup must win every season, or it is not testing the se half"
    winner, text = spread.verdict(errs)
    assert winner == "positional" and "se" in text


def test_the_gate_reports_both_halves_for_every_candidate():
    _, text = spread.verdict(_errs([0.5, 0.5], noise=1.0))
    assert "own_k" in text and "usage" in text
    assert "se" in text and "seasons" in text


def test_no_held_out_season_reports_nothing_measured():
    winner, text = spread.verdict(pl.DataFrame(schema={"season": pl.Int32}))
    assert winner == "positional" and "nothing measured" in text


def test_the_walk_forward_fits_only_on_earlier_seasons():
    rows = ([(2024, "WR", 3.0, 16.0, 9.0)] * 60) + ([(2025, "WR", 3.0, 16.0, 9.0)] * 60)
    errs = spread.walk_forward(_pairs(rows))
    assert errs["season"].unique().to_list() == [2025], "2024 has nothing earlier to fit on"


def test_the_walk_forward_scores_every_candidate_on_the_same_rows():
    """A paired test is only paired if the arms see identical observations."""
    rows = ([(2024, "WR", 3.0, 16.0, 9.0)] * 60) + ([(2025, "WR", 3.0, 16.0, 9.0)] * 60)
    errs = spread.walk_forward(_pairs(rows))
    assert all(f"err_{c}" in errs.columns for c in spread.CANDIDATES)
    assert errs.height == 60


def test_the_shipped_model_is_always_one_of_the_arms():
    assert "positional" in spread.CANDIDATES


# --- the CLI --------------------------------------------------------------

def test_help_needs_no_network():
    assert spread.main([]) == 0


def test_the_fit_path_runs_offline(monkeypatch, capsys, tmp_path):
    import nflreadpy as nfl
    rng = np.random.default_rng(1)
    rows = []
    for season in (2023, 2024, 2025):
        for pid in range(80):
            base = 12.0 + rng.normal(0, 2)
            rows += _season(f"p{pid}", [max(0.0, base + rng.normal(0, 6))
                                        for _ in range(12)], season)
    monkeypatch.setattr(nfl, "load_player_stats", lambda *a, **k: _stats(rows))
    monkeypatch.setattr(nfl, "load_snap_counts", lambda *a, **k: _snaps(
        [(s, w, "P1", "WR", 0.5) for s in (2023, 2024, 2025) for w in range(1, 13)]))
    monkeypatch.setattr(nfl, "load_ff_playerids", lambda *a, **k: _XW)
    out = tmp_path / "s.parquet"
    assert spread.main(["--fit", "--seasons", "2023,2024,2025", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "consecutive-season pairs" in text
    assert out.exists()
