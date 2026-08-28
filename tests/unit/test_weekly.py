"""The Weekly projection: a multiplier on Usage and a touchdown regression.

Offline. Two of these are regression tests for defects that inverted the module's own headline
result -- before them the walk-forward said the whole thing was a null at 1/4 seasons.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import weekly as W


def _rows(n=200, week=10, season=2024, seed=0, **over):
    rng = np.random.default_rng(seed)
    base = {
        "season": [season] * n, "week": [week] * n,
        "position": ["WR"] * n, "player_id": [f"p{i}" for i in range(n)],
        "games_before": [6.0] * n, "snap_trend": list(rng.normal(0, 0.1, n)),
        "targets_prior": [6.0] * n, "receptions_prior": [4.0] * n,
        "carries_prior": [0.5] * n, "attempts_prior": [0.0] * n,
        "receiving_yards_prior": [52.0] * n, "rushing_yards_prior": [4.0] * n,
        "passing_yards_prior": [0.0] * n,
        "passing_interceptions_prior": [0.0] * n, "fumbles_lost_total_prior": [0.0] * n,
        "fantasy_points_ppr": list(rng.normal(11, 5, n)),
        "ppg_before": [11.0] * n,
        # the realised counts, which `fit_multiplier` regresses against their priors
        "targets": list(rng.poisson(6, n).astype(float)),
        "carries": list(rng.poisson(1, n).astype(float)),
        "attempts": [0.0] * n,
    }
    base.update({k: [v] * n if not isinstance(v, list) else v for k, v in over.items()})
    return pl.DataFrame(base)


# --- the multiplier ---------------------------------------------------------

def test_a_zero_coefficient_is_the_identity():
    """The design the plan fixed: `f = 1` recovers the incumbent exactly, so the null is the
    identity and this cannot be much worse than the projection it adjusts."""
    x = np.array([-0.3, 0.0, 0.4])
    assert np.allclose(W.multiplier(x, 0.0), 1.0)


def test_a_zero_feature_is_the_identity_whatever_the_coefficient():
    assert W.multiplier(np.zeros(3), 5.0).tolist() == [1.0, 1.0, 1.0]


def test_a_missing_trend_leaves_the_projection_alone():
    """A player with no snap history must get the flat projection, not a multiplier of zero."""
    assert W.multiplier(np.array([np.nan, np.nan]), 0.5).tolist() == [1.0, 1.0]


def test_the_multiplier_is_bounded_in_both_directions():
    assert W.multiplier(np.array([10.0]), 1.0)[0] == W.MULTIPLIER_HI
    assert W.multiplier(np.array([-10.0]), 1.0)[0] == W.MULTIPLIER_LO


def test_fit_recovers_a_coefficient_it_was_given():
    rng = np.random.default_rng(3)
    n = 4000
    trend = rng.normal(0, 0.15, n)
    prior = np.full(n, 6.0)
    count = (prior + 1.0) * np.exp(0.5 * trend) - 1.0
    d = pl.DataFrame({"week": [10] * n, "targets": count, "targets_prior": prior,
                      "snap_trend": trend})
    assert W.fit_multiplier(d, "targets") == pytest.approx(0.5, abs=0.05)


def test_a_thin_sample_returns_the_identity_rather_than_a_coefficient():
    d = pl.DataFrame({"week": [10] * 20, "targets": [5.0] * 20,
                      "targets_prior": [5.0] * 20, "snap_trend": [0.1] * 20})
    assert W.fit_multiplier(d, "targets") == 0.0


def test_a_constant_feature_returns_the_identity():
    d = pl.DataFrame({"week": [10] * 200, "targets": list(np.arange(200.0)),
                      "targets_prior": [5.0] * 200, "snap_trend": [0.1] * 200})
    assert W.fit_multiplier(d, "targets") == 0.0


def test_the_multiplier_is_off_before_the_week_the_trend_exists():
    """`snap_trend` is non-null from week 7 -- it needs six prior weeks -- but the screen only
    ever established it from week 8. docs/snap-trend-signal.md: anchors 4 and 6 are null and
    flip sign between seasons."""
    coefs = dict.fromkeys(W.VOLUME, 1.0)
    early = W.project(_rows(week=W.TREND_MIN_WEEK - 1, snap_trend=[0.3] * 200), coefs)
    late = W.project(_rows(week=W.TREND_MIN_WEEK, snap_trend=[0.3] * 200), coefs)
    assert early["targets_hat"].to_list() == pytest.approx([6.0] * 200)
    assert late["targets_hat"][0] > 6.5, "and it is on from the week it was measured in"


# --- efficiency, and the floor that was measured against the wrong thing ----

def test_a_player_with_enough_accumulated_volume_keeps_his_own_rate():
    d = _rows(n=50, receptions_prior=5.0, receiving_yards_prior=75.0, games_before=10.0)
    assert W.efficiency(d, "receiving_yards", "receptions") == pytest.approx(15.0)


def test_the_volume_floor_is_a_total_not_a_per_game_figure():
    """The defect that under-projected receivers by 0.66 points a week. `MIN_UNITS` is 8
    accumulated units and it was being compared against a per-game mean, so every receiver
    failed it -- nobody catches eight passes a game -- and got the pooled rate instead.

    A mixed population, so the pooled rate is visibly not any one player's own.
    """
    lo = _rows(n=100, receptions_prior=1.0, receiving_yards_prior=8.0, games_before=10.0)
    hi = _rows(n=100, receptions_prior=6.0, receiving_yards_prior=90.0, games_before=10.0)
    both = pl.concat([lo, hi])
    eff = W.efficiency(both, "receiving_yards", "receptions")
    assert eff[0] == pytest.approx(8.0), "10 games x 1 catch clears the floor: his own rate"
    assert eff[-1] == pytest.approx(15.0), "and so does the busier one, at his own"

    one_week = both.with_columns(pl.lit(1.0).alias("games_before"))
    pooled = W.efficiency(one_week, "receiving_yards", "receptions")
    assert len(set(np.round(pooled, 6))) == 1, "under the floor everyone gets one rate"
    assert pooled[0] == pytest.approx((8.0 + 90.0) / (1.0 + 6.0)), "and it is the pooled one"


# --- the projection ---------------------------------------------------------

def test_points_are_rebuilt_from_counts_never_projected_directly():
    p = W.project(_rows(), dict.fromkeys(W.VOLUME, 0.0))
    for c in ("targets_hat", "receptions_hat", "rec_yards_hat", "tds_hat", "mu"):
        assert c in p.columns
    row = p.to_dicts()[0]
    assert row["receptions_hat"] == pytest.approx(4.0), "targets x his own catch rate"
    assert row["rec_yards_hat"] == pytest.approx(52.0), "receptions x his own yards per catch"


def test_turnovers_are_priced():
    """Omitting them over-projected quarterbacks by +1.44 points a week -- an interception a
    game, almost exactly. A projection that leaves out two of the scoring components is not
    projecting fantasy points."""
    clean = W.project(_rows(position="QB"), dict.fromkeys(W.VOLUME, 0.0))
    picky = W.project(_rows(position="QB", passing_interceptions_prior=1.0),
                      dict.fromkeys(W.VOLUME, 0.0))
    assert picky["mu"][0] == pytest.approx(clean["mu"][0] - 2.0)
    fumbler = W.project(_rows(fumbles_lost_total_prior=1.0), dict.fromkeys(W.VOLUME, 0.0))
    assert fumbler["mu"][0] == pytest.approx(
        W.project(_rows(), dict.fromkeys(W.VOLUME, 0.0))["mu"][0] - 2.0)


def test_touchdowns_come_from_the_position_rate_not_the_player_s_own():
    """component-projection.md measured a player's own touchdown rate as carrying no
    information beyond his yardage: year-over-year r of -0.004 receiving, -0.030 rushing."""
    from hub.models.components import td_rate
    p = W.project(_rows(), dict.fromkeys(W.VOLUME, 0.0)).to_dicts()[0]
    assert p["tds_hat"] == pytest.approx(
        p["rec_yards_hat"] * td_rate("WR", "rec") + p["rush_yards_hat"] * td_rate("WR", "rush"))


def test_a_quarterback_gets_passing_touchdowns_and_a_receiver_does_not():
    qb = W.project(_rows(position="QB", attempts_prior=32.0, passing_yards_prior=240.0),
                   dict.fromkeys(W.VOLUME, 0.0)).to_dicts()[0]
    assert qb["pass_yards_hat"] == pytest.approx(240.0)
    wr = W.project(_rows(attempts_prior=0.0), dict.fromkeys(W.VOLUME, 0.0)).to_dicts()[0]
    assert wr["pass_yards_hat"] == pytest.approx(0.0)


# --- the walk-forward, and the arm it was missing ---------------------------

def _panel(seasons=(2022, 2023, 2024)):
    return pl.concat([_rows(n=300, season=s, week=w, seed=s + w)
                      for s in seasons for w in (8, 9, 10)])


def test_the_walk_forward_carries_three_arms_not_two():
    """The first version carried two -- fitted against `ppg_before` -- so it compared a
    component rebuild with a points mean and buried the multiplier under every difference
    between two whole estimators. It reported -0.0025 MAE at 1/4 seasons, about the rebuild."""
    errs = W.walk_forward(_panel())
    assert {"err_flat", "err_component", "err_weekly"} <= set(errs.columns)


def test_the_component_arm_is_the_fitted_arm_with_the_multiplier_off():
    """`f = 1` is the incumbent, so the two arms differ by exactly the week."""
    errs = W.walk_forward(_panel())
    assert errs.height > 0
    coef = errs["coef_targets"].to_list()[0]
    if coef == 0.0:
        assert (errs["err_component"] == errs["err_weekly"]).all()


def test_nothing_is_scored_on_the_season_it_was_fitted_on():
    errs = W.walk_forward(_panel(seasons=(2022, 2023, 2024)))
    assert sorted(errs["season"].unique().to_list()) == [2023, 2024], \
        "the earliest season is training data only"


def test_the_diagnostic_reports_all_three_contrasts_and_says_it_decides_nothing():
    text = "\n".join(W.diagnostic(W.walk_forward(_panel())))
    for label in ("the week", "the rebuild", "both together"):
        assert label in text
    assert "DIAGNOSTIC ONLY" in text and "ADR-0015" in text


def test_an_empty_walk_forward_reports_rather_than_crashing():
    assert "nothing measured" in "\n".join(W.diagnostic(pl.DataFrame()))
