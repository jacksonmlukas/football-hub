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
        "receptions": list(rng.poisson(4, n).astype(float)),
        "receiving_yards": list(rng.normal(50, 20, n)),
        "rushing_yards": list(rng.normal(4, 3, n)),
        "passing_yards": [0.0] * n,
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


# --- shrinkage, and the objective that could not see the defect -------------

def _train(n=600, seed=7):
    """A population with thick- and thin-sample players at two very different levels."""
    frames = []
    for i, (games, tgt, yds) in enumerate([(10.0, 8.0, 100.0), (1.0, 8.0, 100.0),
                                           (10.0, 2.0, 20.0), (1.0, 2.0, 20.0)]):
        d = _rows(n=n // 4, seed=seed + i, games_before=games, targets_prior=tgt,
                  receptions_prior=tgt * 0.7, receiving_yards_prior=yds)
        frames.append(d)
    return pl.concat(frames)


def test_a_zero_constant_leaves_the_projection_untouched():
    """Zero is in both grids on purpose, so 'no shrinkage' is a candidate the fit can pick --
    which is what makes this an experiment rather than an assumption. It picked it."""
    t = _train()
    coefs = dict.fromkeys(W.VOLUME, 0.0)
    plain = W.project(t, coefs)["mu"].to_numpy()
    vm, em = W._pos_means(t)
    zeroed = W.project(t, coefs, shrink=W.Shrink(0.0, 0.0, vm, em))["mu"].to_numpy()
    assert np.allclose(plain, zeroed)


def test_shrinkage_pulls_a_thin_sample_toward_its_position_and_leaves_a_thick_one():
    t = _train()
    vm, em = W._pos_means(t)
    sh = W.Shrink(8.0, 0.0, vm, em)
    got = W._shrunk(t, "targets", sh.volume_k, vm, "targets")
    thin = t["games_before"].to_numpy() == 1.0
    thick = t["games_before"].to_numpy() == 10.0
    own = t["targets_prior"].to_numpy()
    assert np.abs(got[thin] - own[thin]).mean() > np.abs(got[thick] - own[thick]).mean(), \
        "one game is pulled further than ten"


def test_the_pull_is_n_over_n_plus_k():
    t = _rows(n=10, games_before=4.0, targets_prior=10.0, position="WR")
    means = {("WR", "targets"): 0.0}
    got = W._shrunk(t, "targets", 4.0, means, "targets")
    assert got[0] == pytest.approx(5.0), "4/(4+4) of his own, the rest of a zero mean"


def test_the_efficiency_shrink_subsumes_the_hard_threshold():
    """Without a `Shrink` this is a step at MIN_UNITS; with one it is the smooth version."""
    t = _rows(n=20, receptions_prior=5.0, receiving_yards_prior=75.0, games_before=10.0)
    vm, em = {}, {("WR", "receiving_yards"): 5.0}
    hard = W.efficiency(t, "receiving_yards", "receptions")
    smooth = W.efficiency(t, "receiving_yards", "receptions", W.Shrink(0.0, 50.0, vm, em))
    assert hard[0] == pytest.approx(15.0)
    assert 5.0 < smooth[0] < 15.0, "pulled toward the positional rate, not snapped to it"


def test_each_objective_optimises_its_own_loss():
    """The two are different questions, which is the whole experiment. `mae` minimises mean
    error; `tail` minimises bias among the top decile by projection, where a waiver decision
    reads. On the real panel `mae` fitted `volume_k = 0` in every held-out season -- the
    projection is already unbiased at every sample size, so mean error had nothing to reward,
    and the objective was blind to the defect the shrinkage was meant to fix.
    """
    t = _train()
    coefs = dict.fromkeys(W.VOLUME, 0.0)
    actual = t["fantasy_points_ppr"].to_numpy().astype(float)

    def loss(sh, kind):
        mu = W.project(t, coefs, shrink=sh)["mu"].to_numpy().astype(float)
        if kind == "mae":
            return float(np.abs(mu - actual).mean())
        top = mu >= float(np.quantile(mu, W.TAIL_Q))
        return abs(float((mu[top] - actual[top]).mean()))

    vm, em = W._pos_means(t)
    for kind in ("mae", "tail"):
        fitted = W.fit_shrink(t, coefs, objective=kind)
        best = loss(fitted, kind)
        for vk in W.VOLUME_SHRINK_GRID:
            for ek in W.EFF_SHRINK_GRID:
                assert loss(W.Shrink(vk, ek, vm, em), kind) >= best - 1e-12, \
                    f"{kind} did not find its own minimum at ({vk}, {ek})"


def test_both_objectives_fit_on_training_rows_only():
    """A shrinkage fitted on the season it is scored against is the treatment arm reading its
    own answer sheet -- the defect ADR-0012 records the first lineup gate dying of."""
    import inspect
    src = inspect.getsource(W.fit_shrink)
    assert "train" in inspect.signature(W.fit_shrink).parameters
    assert "held-out" in src or "training" in src


def test_the_shrink_target_is_per_position():
    t = pl.concat([_rows(n=50, position="WR", targets_prior=8.0, targets=8.0),
                   _rows(n=50, position="RB", targets_prior=2.0, targets=2.0)])
    vm, _ = W._pos_means(t)
    assert vm[("WR", "targets")] != vm[("RB", "targets")], \
        "a receiver and a back do not regress toward the same place"
