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


# --- the Usage multiplier, which is the identity ----------------------------

def test_the_usage_multiplier_is_exactly_one_for_every_player_week():
    """#248, implementing the decision on #233: `snap_trend` leaves the Usage multiplier.

    On the settled `(yds_prior, ecr)` basis the trend is +0.014 at permutation p 0.24 at
    the published anchor, so its licence is revoked and the multiplier is the `f = 1`
    incumbent the module was built around. Exactly one -- not a coefficient of zero that
    something could refit -- at every week, whatever the trend says, and where it is
    missing. Byte equality rather than `allclose`, because `x * 1.0` is `x` in IEEE
    arithmetic and anything short of that is a multiplier.
    """
    from hub.models.panel import TREND_MIN_WEEK
    trend = [3.0, -3.0, float("nan"), 0.25] * 50
    for week in (TREND_MIN_WEEK - 1, TREND_MIN_WEEK, 14):
        rows = _rows(week=week, snap_trend=trend, carries_prior=2.5, attempts_prior=1.5)
        out = W.project(rows)
        for c in W.VOLUME:
            hat, prior = out[f"{c}_hat"].to_numpy(), rows[f"{c}_prior"].to_numpy()
            assert hat.tobytes() == prior.tobytes(), f"{c} moved in week {week}"
            assert (hat / prior == 1.0).all(), f"the {c} multiplier is not exactly 1.0"


def test_the_touchdown_term_is_untouched_by_the_multiplier_leaving():
    """The other coefficient's fate, side by side. The touchdown term is the position's
    rate applied to projected yards, and it is the same bytes it was under the `f = 1` arm
    the walk-forward already carried: `components.td_rate` does not move, and neither does
    the yardage it multiplies now that the Usage it is built from is the prior itself."""
    from hub.models.components import td_rate
    rows = _rows(snap_trend=[2.0] * 200, carries_prior=2.5, rushing_yards_prior=12.0)
    out = W.project(rows)
    rec_y = rows["receptions_prior"].to_numpy() / rows["targets_prior"].to_numpy() \
        * rows["targets_prior"].to_numpy() \
        * (rows["receiving_yards_prior"].to_numpy() / rows["receptions_prior"].to_numpy())
    rush_y = rows["carries_prior"].to_numpy() \
        * (rows["rushing_yards_prior"].to_numpy() / rows["carries_prior"].to_numpy())
    want = rec_y * td_rate("WR", "rec") + rush_y * td_rate("WR", "rush")
    assert out["tds_hat"].to_numpy().tobytes() == want.tobytes()
    assert out["tds_hat"][0] > 0.0, "and the term is live, not zero on both sides"


def test_the_projection_takes_no_coefficient_at_all():
    """A `coefs` argument that is always zero is a hook, and a hook is what the four rescue
    variants were built on. The projection has nowhere to put a fitted multiplier."""
    import inspect
    assert "coefs" not in inspect.signature(W.project).parameters
    assert not hasattr(W, "fit_multiplier") and not hasattr(W, "multiplier"), (
        "the fit is gone with the licence; the record of what was tried is the screen's "
        "(`weekly_screen --permute snap_trend`, `--usage`), per ADR-0007")


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
    p = W.project(_rows())
    for c in ("targets_hat", "receptions_hat", "rec_yards_hat", "tds_hat", "mu"):
        assert c in p.columns
    row = p.to_dicts()[0]
    assert row["receptions_hat"] == pytest.approx(4.0), "targets x his own catch rate"
    assert row["rec_yards_hat"] == pytest.approx(52.0), "receptions x his own yards per catch"


def test_turnovers_are_priced():
    """Omitting them over-projected quarterbacks by +1.44 points a week -- an interception a
    game, almost exactly. A projection that leaves out two of the scoring components is not
    projecting fantasy points."""
    clean = W.project(_rows(position="QB"))
    picky = W.project(_rows(position="QB", passing_interceptions_prior=1.0))
    assert picky["mu"][0] == pytest.approx(clean["mu"][0] - 2.0)
    fumbler = W.project(_rows(fumbles_lost_total_prior=1.0))
    assert fumbler["mu"][0] == pytest.approx(
        W.project(_rows())["mu"][0] - 2.0)


def test_touchdowns_come_from_the_position_rate_not_the_player_s_own():
    """component-projection.md measured a player's own touchdown rate as carrying no
    information beyond his yardage: year-over-year r of -0.004 receiving, -0.030 rushing."""
    from hub.models.components import td_rate
    p = W.project(_rows()).to_dicts()[0]
    assert p["tds_hat"] == pytest.approx(
        p["rec_yards_hat"] * td_rate("WR", "rec") + p["rush_yards_hat"] * td_rate("WR", "rush"))


def test_a_quarterback_gets_passing_touchdowns_and_a_receiver_does_not():
    qb = W.project(_rows(position="QB", attempts_prior=32.0, passing_yards_prior=240.0)).to_dicts()[0]
    assert qb["pass_yards_hat"] == pytest.approx(240.0)
    wr = W.project(_rows(attempts_prior=0.0)).to_dicts()[0]
    assert wr["pass_yards_hat"] == pytest.approx(0.0)


# --- the walk-forward, and the arm it no longer carries ---------------------

def _panel(seasons=(2022, 2023, 2024)):
    return pl.concat([_rows(n=300, season=s, week=w, seed=s + w)
                      for s in seasons for w in (8, 9, 10)])


def test_the_walk_forward_carries_two_arms_and_no_fitted_one():
    """It carried three until #248: `flat`, `component` (`f = 1`) and `weekly` (the fitted
    multiplier), because the first version had two and could not see its own subject. The
    subject is gone -- the weekly projection *is* `f = 1` now -- so the fitted arm and the
    coefficient columns beside it go with it, rather than staying as a third arm that is
    the second arm again under another name."""
    errs = W.walk_forward(_panel())
    assert {"err_flat", "err_weekly", "crps_weekly"} <= set(errs.columns)
    assert "err_component" not in errs.columns
    assert not [c for c in errs.columns if c.startswith("coef_")], (
        "a coefficient column with no multiplier to carry it")


def test_the_weekly_arm_is_the_rebuild_with_nothing_fitted():
    """`f = 1` is the incumbent and it is also the projection, so the arm the walk-forward
    scores is `project` on the held-out rows and nothing else -- the same number a caller
    with no training seasons at all would get."""
    panel = _panel()
    errs = W.walk_forward(panel)
    held = panel.filter(pl.col("season") > 2022)
    mu = W.project(held)["mu"].to_numpy()
    actual = held["fantasy_points_ppr"].to_numpy()
    assert np.array_equal(errs["err_weekly"].to_numpy(), np.abs(mu - actual))


def test_nothing_is_scored_on_the_season_it_was_fitted_on():
    """Nothing is fitted any more, and the window is kept anyway: the rebuild's published
    +0.0634 MAE against flat was taken on the seasons `expanding_seasons` holds out, and a
    re-run that widened the window would not be comparable with it."""
    errs = W.walk_forward(_panel(seasons=(2022, 2023, 2024)))
    assert sorted(errs["season"].unique().to_list()) == [2023, 2024], \
        "the earliest season is training data only"


def test_the_diagnostic_reports_the_rebuild_and_says_it_decides_nothing(tmp_path, monkeypatch):
    # The artifact `hub.models.coverage` commits under `state/` is part of the tree now (#273),
    # and its sentence says "the week it scores". Pin the tree to one with no artifact so the
    # text is the diagnostic's own, and look for the retired contrast by its label.
    from hub.models import coverage
    monkeypatch.setattr(coverage, "ARTIFACT", tmp_path / "none.json")
    text = "\n".join(W.diagnostic(W.walk_forward(_panel())))
    assert "the rebuild" in text
    assert "(weekly vs f=1)" not in text and "both together" not in text, (
        "a contrast against an arm that no longer exists")
    assert "--measure --write" in text, "degrades to the command that produces the artifact"
    assert "f = 1" in text and "#248" in text, "the diagnostic says what the arm is"
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
    plain = W.project(t)["mu"].to_numpy()
    vm, em = W._pos_means(t)
    zeroed = W.project(t, shrink=W.Shrink(0.0, 0.0, vm, em))["mu"].to_numpy()
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
    actual = t["fantasy_points_ppr"].to_numpy().astype(float)

    def loss(sh, kind):
        mu = W.project(t, shrink=sh)["mu"].to_numpy().astype(float)
        if kind == "mae":
            return float(np.abs(mu - actual).mean())
        top = mu >= float(np.quantile(mu, W.TAIL_Q))
        return abs(float((mu[top] - actual[top]).mean()))

    vm, em = W._pos_means(t)
    for kind in ("mae", "tail"):
        fitted = W.fit_shrink(t, objective=kind)
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


# --- the market-implied target ----------------------------------------------

def _ranked(n=400, seed=13):
    """Players spread across preseason ranks, with volume falling as rank rises."""
    frames = []
    for r in (2.0, 10.0, 40.0, 120.0):
        d = _rows(n=n // 4, seed=int(r), targets=float(max(11 - r / 12, 1.0)),
                  targets_prior=float(max(11 - r / 12, 1.0)))
        frames.append(d.with_columns(pl.lit(r).alias("preseason_ecr")))
    return pl.concat(frames)


def test_the_consensus_prior_falls_with_rank():
    """Log-log in the pick, the shape volume-model.md fitted: volume is roughly a power law
    in the market's opinion and strictly non-negative."""
    t = _ranked()
    m = W.fit_consensus_prior(t)
    assert ("WR", "targets") in m
    lo = W.consensus_target(_rows(n=1, preseason_ecr=3.0), "targets", m)[0]
    hi = W.consensus_target(_rows(n=1, preseason_ecr=150.0), "targets", m)[0]
    assert lo > hi > 0, "a third-ranked receiver is projected more volume than a 150th"


def test_the_rank_is_clamped_to_the_fitted_range():
    """What stops a 599th-ranked player being extrapolated off the end of a log curve."""
    t = _ranked()
    m = W.fit_consensus_prior(t)
    edge = W.consensus_target(_rows(n=1, preseason_ecr=120.0), "targets", m)[0]
    beyond = W.consensus_target(_rows(n=1, preseason_ecr=5000.0), "targets", m)[0]
    assert beyond == pytest.approx(edge)


def test_a_player_with_no_rank_falls_back_to_his_position():
    t = _rows(n=4).with_columns(pl.lit(None, dtype=pl.Float64).alias("preseason_ecr"))
    means = {("WR", "targets"): 3.0}
    got = W._shrunk(t, "targets", 1e9, means, "targets", prior={("WR", "targets"): (1., -1., 1., 9.)})
    assert got[0] == pytest.approx(3.0), "no rank, so the positional mean is the target"


def test_the_consensus_target_reads_the_preseason_rank_and_ignores_a_weekly_one():
    """The distinction the whole variant rests on. Shrinking toward the *weekly* ranking would
    make the arm partly be the incumbent Gate B measures it against; the preseason rank is a
    different quantity, published four months earlier. Tested by handing it both and checking
    which one moves the answer."""
    m = W.fit_consensus_prior(_ranked())
    a = _rows(n=1, preseason_ecr=3.0).with_columns(pl.lit(400.0).alias("ecr"))
    b = _rows(n=1, preseason_ecr=3.0).with_columns(pl.lit(1.0).alias("ecr"))
    assert W.consensus_target(a, "targets", m)[0] == W.consensus_target(b, "targets", m)[0], \
        "a weekly rank must not move it"
    c = _rows(n=1, preseason_ecr=120.0).with_columns(pl.lit(400.0).alias("ecr"))
    assert W.consensus_target(c, "targets", m)[0] != W.consensus_target(a, "targets", m)[0], \
        "and the preseason rank must"


def test_the_curves_are_refitted_not_imported():
    """`volume.VOLUME_CURVE` is frozen at a 2022-25 fit, which is the span held out here, so
    importing it would evaluate a prior on the seasons it was fitted on."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(W))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names}
    assert "VOLUME_CURVE" not in imported and "volume" not in imported
    assert "train" in inspect.signature(W.fit_consensus_prior).parameters

    # and two different training sets give two different curves
    one = W.fit_consensus_prior(_ranked(seed=1))
    two = W.fit_consensus_prior(_ranked(seed=2).with_columns(pl.col("targets") * 2))
    assert one[("WR", "targets")] != two[("WR", "targets")]


def test_pure_market_ignores_the_players_own_history():
    """The probe that asks how much of a gain is the market's rather than the model's."""
    t = _ranked()
    m = W.fit_consensus_prior(t)
    own = t.with_columns(pl.lit(99.0).alias("targets_prior"))
    got = W._shrunk(own, "targets", W.PURE_MARKET_K, {("WR", "targets"): 0.0},
                    "targets", prior=m)
    assert got.max() < 20.0, "a 99-target prior is ignored entirely"


# --- parameter uncertainty, which is not outcome volatility -----------------

def test_the_standard_error_falls_with_games_played():
    """+36% predictive sd at one game against twelve, and it is just `n`. ADR-0012 measured
    per-player *volatility* beyond the positional constant and called it not estimable; this
    is how well we know his mean, which is a different quantity."""
    t = pl.concat([_rows(n=5, games_before=float(g)) for g in (1, 4, 16)])
    se = W.standard_error(t, {"WR": 5.0, "__pooled__": 5.0})
    assert se[0] == pytest.approx(5.0)
    assert se[5] == pytest.approx(2.5)
    assert se[10] == pytest.approx(1.25)


def test_a_player_with_no_games_is_not_divided_by_zero():
    t = _rows(n=3, games_before=0.0)
    assert np.isfinite(W.standard_error(t, {"__pooled__": 5.0})).all()


def test_the_positional_sd_is_per_position_and_falls_back():
    rng = np.random.default_rng(4)
    frames = []
    for pos, spread in (("QB", 9.0), ("TE", 3.0)):
        for i in range(12):
            frames.append(_rows(n=8, position=pos, seed=i).with_columns(
                pl.lit(f"{pos}{i}").alias("player_id"),
                pl.Series("fantasy_points_ppr", rng.normal(10, spread, 8))))
    sig = W.positional_sd(pl.concat(frames))
    assert sig["QB"] > sig["TE"], "quarterbacks vary more week to week than tight ends"
    assert "__pooled__" in sig, "and an unseen position falls back rather than raising"
    assert W.standard_error(_rows(n=1, position="K", games_before=1.0), sig)[0] == \
        pytest.approx(sig["__pooled__"]), "an unseen position falls back to the pool"


# --- the distribution, not just the centre (#177) ---------------------------
#
# MAE is what decides, and it is minimised by the median: it reads the centre and is blind
# to the spread and the skew, which are two thirds of what `predict.moments` produces. These
# check the wiring, not the mathematics -- the rule itself is proved against its own
# definition and against a hand-computed closed form in `tests/unit/test_scoring_rules.py`.

def test_the_walk_forward_scores_the_distribution_beside_the_error():
    """One CRPS column, on the same rows and the same window as the three error columns."""
    errs = W.walk_forward(_panel())
    assert "crps_weekly" in errs.columns
    assert errs["crps_weekly"].null_count() == 0
    assert (errs["crps_weekly"] >= 0).all(), "a score that can go negative is not this one"


def test_the_published_spread_earns_its_place_in_the_score():
    """The distribution scores better than the same projection published as a point mass.

    Not a tautology -- a distribution that is too wide or in the wrong place scores *worse*
    than its own centre, which is the whole reason CRPS is worth reporting. This says the
    shipped one is not that, on the walk-forward window.
    """
    errs = W.walk_forward(_panel())
    assert errs["crps_weekly"].to_numpy().mean() < errs["err_weekly"].to_numpy().mean()


def test_the_crps_column_reads_the_published_laws_rather_than_a_copy_of_them():
    """`shipped_quantiles` must grade what `hub.models.predict` serves, not a second
    implementation of `sd = K*sqrt(mu)` that can drift away from it.

    Proved by making the published spread zero at its source: with every `WEEKLY_K` at zero
    the served distribution *is* a point mass, and CRPS must then equal the absolute error
    exactly -- the identity `test_a_point_mass_is_scored_as_the_absolute_error` states in
    the other file. A copied law would not notice the patch and the two would disagree.
    """
    mu = np.array([12.0, 3.0, 21.5])
    actual = np.array([4.0, 9.0, 21.5])
    got = W.crps_from_quantiles(W.shipped_quantiles(mu, ["WR", "RB", "QB"]), actual)
    assert (got > 0.0).all(), "a week scored at zero is a forecast with no spread at all"
    # The third row is the outcome landing exactly on the projection, where a point mass
    # scores zero and any honest distribution scores more. Per row the two are not ordered;
    # the ordering is an expectation, and `test_the_published_spread_earns_its_place_in_the
    # _score` is where it is asserted.
    assert got[2] > abs(mu[2] - actual[2])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(W.predict, "WEEKLY_K", dict.fromkeys(W.predict.WEEKLY_K, 0.0))
        mp.setattr(W.predict, "WEEKLY_K_POOLED", 0.0)
        flat_dist = W.crps_from_quantiles(W.shipped_quantiles(mu, ["WR", "RB", "QB"]), actual)
    assert flat_dist == pytest.approx(np.abs(mu - actual), abs=1e-9)

    # And the third moment, which the spread law says nothing about: weekly scoring is
    # right-skewed -- the median week is about 0.90 of the mean, because the mean is carried
    # by touchdown spikes -- so the quantiles being graded must sit below their own centre.
    # A normal fitted to the same two moments would put the median exactly on it, and every
    # assertion above would still hold.
    wr = W.shipped_quantiles(np.array([12.0]), ["WR"])[0]
    assert float(np.median(wr)) < 12.0 - 0.5, (
        "the graded distribution is symmetric about its mean, so it is not the one "
        "`predict.skewed` serves")


def test_the_diagnostic_reports_crps_beside_mae_over_the_same_window():
    """#177's third criterion. Both columns are means over the rows `expanding_seasons`
    left, per season and pooled, with the ratio a point mass would score 1.000 on."""
    errs = W.walk_forward(_panel())
    text = "\n".join(W.diagnostic(errs))
    assert "CRPS" in text and "crps/mae" in text
    assert "pooled" in text
    for season in errs["season"].unique().to_list():
        assert str(season) in text
    ratio = float(errs["crps_weekly"].to_numpy().mean() / errs["err_weekly"].to_numpy().mean())
    assert f"{ratio:.3f}" in text, text
    assert f"{W.CALIBRATED_RATIO:.3f}" in text, "the calibrated reference is not printed"


def test_an_empty_walk_forward_still_reports_rather_than_dividing_by_zero():
    """The degraded path: no held-out season means no rows to score, and the distribution
    report must not be reached at all rather than meaning an empty column."""
    assert "nothing measured" in "\n".join(W.diagnostic(pl.DataFrame()))


def test_the_report_cites_the_coverage_measurement_rather_than_a_figure_typed_here():
    """#177's fourth criterion, in the direction the ticket cares about.

    The argument that deferred this ticket was that the moments MAE cannot see were "already
    within a point of nominal". That is now a measurement rather than a sentence, and the
    report reads its artifact -- so when the measurement is re-run this line moves with it,
    which a number typed into a print statement would not.
    """
    from hub.models import coverage

    published = {"centre": "prior", "lookahead": False, "n": 16061,
                 "gate_subset": "strictly positive", "gate_n": 10536, "gate_cov80": 0.774,
                 "band": 0.02, "verdict": "UNDER-COVERS", "generated_at": "2026-09-07"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(coverage, "published_summary", lambda *a, **k: published)
        text = "\n".join(W.diagnostic(W.walk_forward(_panel())))
    assert "UNDER-COVERS" in text
    assert "77.4%" in text and "80%" in text and "10,536" in text
    assert "2026-09-07" in text, "a measurement with no date is not one that can go stale"


def test_a_tree_with_no_measurement_published_says_how_to_run_it():
    """Graceful degradation, `CLAUDE.md`'s standing rule: a fresh clone has no
    `data/processed/`, and a report that raised there would make the diagnostic unrunnable
    for the case the suite is written around."""
    from hub.models import coverage

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(coverage, "published_summary", lambda *a, **k: None)
        text = "\n".join(W.diagnostic(W.walk_forward(_panel())))
    assert "coverage --measure --write" in text
    assert "crps/mae" in text, "the score itself still reports without the measurement"
