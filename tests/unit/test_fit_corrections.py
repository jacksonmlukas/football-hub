"""The refitting harness for the five board corrections.

Nothing here touches the network. The panel is planted -- a known coefficient, a known
season-level component -- so every assertion is against an answer that was decided before the
harness ran, which is the only way a fitter can be tested at all: a fit on real data has no
right answer to check against, and a test that asserted today's number would be a snapshot
rather than a test.

**Where the assertions sit.** At the seam, not through the CLI. `fit_one` returns the
coefficient, the interval and the counts as a `Fit`, and every property below is asserted on
that object rather than on a printed line, because a report that formatted the wrong field
would print something plausible and a string test would pass.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub.draft.fit_corrections import (
    COEFFICIENTS,
    DISPOSITIONS,
    HEADLINE_BASELINE,
    SHRINKS,
    Fit,
    _cluster_key,
    _contributions,
    _design,
    best_shrink,
    disposition,
    fit_one,
    refit_interval,
    report_lines,
    shrink_curve,
    walk_forward,
)
from hub.models.experiment import SEASON_CLUSTER, summarise

SEASONS = tuple(range(2018, 2026))
QB = COEFFICIENTS[0]              # td_luck.QB, shipped -0.540


def fitted(*args, **kwargs) -> Fit:
    """`fit_one` where the test's premise is that it fits at all.

    `fit_one` returns `None` for a slice too thin or a signal that never varies, and both are
    tested for directly below. Everywhere else a `None` means the fixture stopped exercising
    the thing under test, so it fails here loudly rather than as an attribute error twenty
    lines later.
    """
    got = fit_one(*args, **kwargs)
    assert got is not None, "the panel this test is built on no longer fits at all"
    return got


def mae(wf: pl.DataFrame, arm: str) -> float:
    """One walk-forward arm's mean held-out error, as a number not a polars scalar."""
    got = wf[f"mae_{arm}"].mean()
    assert isinstance(got, float)
    return got


def planted(beta: float, *, slope_sd: float = 0.0, seed: int = 7, per_season: int = 200,
            seasons: tuple[int, ...] = SEASONS, signal: str = "td_luck",
            pos: str = "QB") -> pl.DataFrame:
    """A panel with a known coefficient on `signal` and a known season structure.

    `slope_sd` is the part that matters for the clustering claim. At zero the effect is the
    same number every year and rows inside a season share nothing about it; above zero the
    coefficient is **one realisation of the year**, which is exactly what
    `hub.models.experiment.SEASON_CLUSTER` says a season is. The two cases are kept separate
    below rather than averaged into one fixture, because they are different claims.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        slope = beta + rng.normal(0.0, slope_sd)
        shock = rng.normal(0.0, 1.2)
        for p in range(per_season):
            base = float(rng.uniform(2.0, 20.0))
            sig = float(rng.normal(0.0, 2.0))
            rows.append({
                "season": season, "player_key": f"p{p}", "pos": pos,
                "base_xfp": base, "base_carry": base, "base_blend": base,
                signal: sig,
                "target": 1.0 + 0.8 * base + slope * sig + shock + float(rng.normal(0, 2.0)),
            })
    return pl.DataFrame(rows)


# --- the estimator ------------------------------------------------------------


def test_the_harness_recovers_a_planted_coefficient():
    """The acceptance criterion, and the only thing that makes the rest of the file mean
    anything: a fitter that could not find an effect it was handed cannot be trusted to
    report one it was not."""
    fit = fitted(planted(-0.6), QB)
    assert fit is not None
    assert fit.beta == pytest.approx(-0.6, abs=0.08), fit.beta
    assert fit.lo < -0.6 < fit.hi, (fit.lo, fit.hi)


def test_a_planted_coefficient_of_the_other_sign_comes_back_the_other_sign():
    """Recovery of one number is also satisfied by a fitter that always returns it. The
    sign is what every disposition here turns on, so it is planted both ways."""
    assert fitted(planted(+0.9), QB).beta == pytest.approx(+0.9, abs=0.08)
    assert fitted(planted(-0.9), QB).beta == pytest.approx(-0.9, abs=0.08)


def test_the_interval_is_centred_on_the_ols_coefficient():
    """`_contributions` claims the estimator is written as a mean *exactly*, not
    approximately, so `summarise` resamples the OLS coefficient itself. That is an algebraic
    identity and it is asserted as one, to a tolerance no approximation would meet.

    It is also what ties the two groupings together: `_cluster_key` builds the partition to
    scale the contributions and `summarise` builds it again to average them, and if those two
    ever disagreed the mean would drift off the coefficient. This is the line that would
    notice."""
    panel = planted(-0.4, slope_sd=0.3)
    df = _design(panel, QB, HEADLINE_BASELINE)
    y = df["target"].to_numpy().astype(float)
    X = np.column_stack([np.ones(df.height),
                         df[HEADLINE_BASELINE].to_numpy().astype(float),
                         df[QB.signal].to_numpy().astype(float)])
    ols = float(np.linalg.pinv(X.T @ X) @ X.T @ y)if False else float(
        (np.linalg.pinv(X.T @ X) @ X.T @ y)[2])
    keys = _cluster_key(df, SEASON_CLUSTER)
    contrib, _ = _contributions(y, X, 2, keys)
    got = summarise(df.select(list(SEASON_CLUSTER)).with_columns(pl.Series("diff", contrib)),
                    cluster=SEASON_CLUSTER)
    assert got["mean"] == pytest.approx(ols, abs=1e-9), (got["mean"], ols)
    assert fitted(panel, QB).beta == pytest.approx(ols, abs=1e-9)


def test_the_identity_holds_when_the_seasons_are_different_sizes():
    """The balanced case cannot see the scaling, and the live panel is never balanced.

    `_contributions` multiplies each row's share by `K * n_k`. With every season the same
    size that factor is a constant, so dropping it entirely leaves the cluster means
    *identical* -- a mutant that deletes it survives every balanced fixture in this file,
    which is how it was found. Unbalanced, `n_k` varies and the grand mean over cluster means
    stops being the OLS coefficient unless each cluster's share is scaled by its own size.

    The live panel is 340 QB rows over eight seasons and 4,595 rows over the designation's,
    none of them equal, so this is the case that runs in production and the balanced one is
    the special case.
    """
    panel = pl.concat([planted(-0.5, seasons=(y,), per_season=n, seed=20 + i)
                       for i, (y, n) in enumerate(zip(SEASONS, (40, 300, 55, 210, 90, 25,
                                                               400, 130), strict=True))])
    sizes = panel.group_by("season").len()["len"].to_list()
    assert len(set(sizes)) > 1, "the fixture stopped being unbalanced"
    df = _design(panel, QB, HEADLINE_BASELINE)
    y = df["target"].to_numpy().astype(float)
    X = np.column_stack([np.ones(df.height),
                         df[HEADLINE_BASELINE].to_numpy().astype(float),
                         df[QB.signal].to_numpy().astype(float)])
    ols = float((np.linalg.pinv(X.T @ X) @ X.T @ y)[2])
    fit = fitted(panel, QB, cluster=SEASON_CLUSTER)
    assert fit.beta == pytest.approx(ols, abs=1e-9)
    contrib, _ = _contributions(y, X, 2, _cluster_key(df, SEASON_CLUSTER))
    got = summarise(df.select(list(SEASON_CLUSTER)).with_columns(pl.Series("diff", contrib)),
                    cluster=SEASON_CLUSTER)
    assert got["mean"] == pytest.approx(ols, abs=1e-9), (got["mean"], ols)


def test_a_season_clustered_interval_is_wider_than_a_player_clustered_one():
    """The acceptance criterion, planted with the structure the claim is about.

    #45's argument is that rows inside a season share "one realisation of the year", so the
    honest replication count is the season. That is a statement about a season-level
    component *in the effect*, and it is planted here as one: `slope_sd` makes the
    coefficient a draw per season. Clustering on the season then has to admit there are eight
    observations rather than 1,600, and the interval widens.

    Planting it is not rigging it. With `slope_sd=0` there is no season-level component and
    season clustering correctly buys nothing -- asserted directly below, so this test cannot
    pass by a `cluster` argument that is quietly ignored."""
    panel = planted(-0.6, slope_sd=0.25)
    by_season = fitted(panel, QB, cluster=SEASON_CLUSTER)
    by_player = fitted(panel, QB, cluster=("player_key",))
    assert by_season.clusters == len(SEASONS)
    assert by_player.clusters == 200
    assert (by_season.hi - by_season.lo) > 2.0 * (by_player.hi - by_player.lo), (
        f"season {by_season.hi - by_season.lo:.4f} against "
        f"player {by_player.hi - by_player.lo:.4f}")


def test_the_cluster_argument_is_read_rather_than_assumed():
    """The other half of the test above. A `fit_one` that hard-coded the season would pass
    that one and fail this: with no season-level component the two groupings are estimating
    the same variance, and the widths come out together."""
    panel = planted(-0.6, slope_sd=0.0)
    by_season = fitted(panel, QB, cluster=SEASON_CLUSTER)
    by_player = fitted(panel, QB, cluster=("player_key",))
    assert (by_season.hi - by_season.lo) < 1.5 * (by_player.hi - by_player.lo)


def test_the_mde_comes_back_and_answers_the_power_question():
    """A disposition without an MDE beside it is #45's whole complaint. `resolved` is what
    reads it, and it has to move with the evidence rather than being a constant."""
    # Both are planted at -1.5 against a shipped -0.540, so both disagree with the board by
    # about a point. Only the evidence differs: 3,200 rows with a near-constant effect
    # against 200 rows whose effect is a wild draw per season.
    thick = fitted(planted(-1.5, slope_sd=0.02, per_season=400), QB)
    thin = fitted(planted(-1.5, slope_sd=2.5, per_season=10), QB)
    assert thin.mde > thick.mde, (thin.mde, thick.mde)
    # The thin fit disagrees with the shipped constant by *more* and still cannot resolve the
    # disagreement, while the thick one disagrees by less and can. That ordering is the whole
    # point of printing an MDE beside a disposition, and it is unreachable for a `resolved`
    # that reads only the gap -- which would have to call the thin one resolved first.
    assert thin.gap > thick.gap, (thin.gap, thick.gap)
    assert thick.resolved is True, (thick.gap, thick.mde)
    assert thin.resolved is False, (thin.gap, thin.mde)


def test_the_refit_cross_check_agrees_with_the_linearised_interval():
    """`_contributions` argues its one approximation is second order. This is the measurement
    that argument is not allowed to replace.

    `docs/talent-cv.md` records a two-stage estimator where refitting *inside* the resample
    moved the interval by 54%, so "the difference is small" is exactly the claim this repo has
    been wrong about before. Here the estimator is one OLS and the two do agree -- asserted
    loosely, because they are different estimators and pinning them tight would be asserting
    the bootstrap's seed rather than the property."""
    panel = planted(-0.6, slope_sd=0.25)
    fit = fitted(panel, QB)
    lo, hi = refit_interval(panel, QB, draws=400)
    assert lo < -0.6 < hi, (lo, hi)
    assert lo == pytest.approx(fit.lo, abs=0.15), (lo, fit.lo)
    assert hi == pytest.approx(fit.hi, abs=0.15), (hi, fit.hi)


def test_the_refit_cross_check_declines_a_panel_too_thin_to_fit():
    lo, hi = refit_interval(planted(-0.6, per_season=1, seasons=(2020, 2021)), QB)
    assert lo != lo and hi != hi, "a panel that cannot be fitted has no interval, not a zero"


def test_no_cluster_at_all_treats_the_row_as_the_unit():
    """`summarise`'s documented default, reachable here so the harness can be *shown* to
    cost precision by clustering rather than asserted to. The row-level interval on a panel
    with a season-level effect is the too-narrow one #45 is about."""
    panel = planted(-0.6, slope_sd=0.25)
    by_row = fitted(panel, QB, cluster=None)
    by_season = fitted(panel, QB, cluster=SEASON_CLUSTER)
    assert by_row.clusters == panel.height
    assert (by_season.hi - by_season.lo) > 2.0 * (by_row.hi - by_row.lo)


# --- dispositions --------------------------------------------------------------


@pytest.mark.parametrize("inside,beta,shipped,want", [
    (True, -0.1, -0.286, "reproduced"),
    (True, +0.5, -0.540, "reproduced"),     # containment first: noise is not a reversal
    (False, +0.5, -0.540, "sign-reversed"),
    (False, -0.9, -0.286, "unreproduced"),  # same sign, outside: a level disagreement
    (False, +0.9, +0.286, "unreproduced"),
    (False, 0.0, -0.286, "unreproduced"),   # zero is not a sign
])
def test_every_input_resolves_to_exactly_one_disposition(inside, beta, shipped, want):
    got = disposition(inside, beta, shipped)
    assert got == want
    assert got in DISPOSITIONS


def test_a_sign_flip_inside_the_interval_is_not_called_a_reversal():
    """The ordering argument, asserted rather than left in the docstring. A point estimate
    on the other side of zero whose interval still covers the shipped value is a run that
    cannot tell them apart, and calling that `sign-reversed` reads a sign off noise --
    which is the mistake `docs/td-luck.md` warns about in its own pooled row."""
    assert disposition(True, +0.5, -0.540) == "reproduced"
    assert disposition(False, +0.5, -0.540) == "sign-reversed"


def test_the_disposition_on_a_fit_uses_that_fit_s_own_interval():
    """`Fit.disposition` must read the interval it shipped with. A property that recomputed
    containment against some other fit's numbers is the shape of defect this repo keeps
    finding, and it is invisible from the outside."""
    fit = fitted(planted(-0.540), QB)          # planted *at* the shipped QB value
    assert fit.shipped_inside is True
    assert fit.disposition == "reproduced"
    reversed_fit = fitted(planted(+1.4), QB)   # far the other side, interval nowhere near
    assert reversed_fit.shipped_inside is False
    assert reversed_fit.disposition == "sign-reversed"


def test_a_shipped_value_on_an_interval_edge_is_flagged():
    """#155's shape, mechanised. Nothing in this harness is a constrained fit, so this does
    not fire on the live panel -- but a number sitting exactly on a bound is what
    `docs/pick-noise.md` found last time, and a check that only exists in prose is one
    nobody runs."""
    fit = fitted(planted(-0.6), QB)
    on_edge = fit._replace(coefficient=QB._replace(shipped=fit.lo))
    assert on_edge.at_a_bound is True
    assert fit._replace(coefficient=QB._replace(shipped=fit.lo - 0.05)).at_a_bound is False


# --- the walk-forward split ----------------------------------------------------


def test_the_walk_forward_never_scores_a_season_on_its_own_data():
    """The acceptance criterion, asserted on the split rather than on the scores.

    A leak shows up downstream as a *better* number, which is why it has to be caught here:
    nothing about a held-out MAE says whether the fit had seen the year. `expanding_seasons`
    is the one statement of the rule and this walks its output to prove the walk actually
    used it."""
    panel = planted(-0.6)
    wf = walk_forward(panel, QB)
    assert not wf.is_empty()
    years = wf["season"].to_list()
    assert years == sorted(years) and years[0] > min(SEASONS), (
        "the earliest season must be training data only")
    for row in wf.iter_rows(named=True):
        earlier = panel.filter(pl.col("season") < row["season"]).height
        assert row["n_past"] == earlier, (
            f"season {row['season']} was fitted on {row['n_past']} rows where only "
            f"{earlier} are strictly earlier")
        assert row["n"] == panel.filter(pl.col("season") == row["season"]).height


def test_the_split_guard_refuses_a_leaking_season(monkeypatch):
    """The guard is proved by handing it the thing it exists to refuse.

    **No panel can make it fire.** `expanding_seasons` derives `past` and `now` from the
    season column itself, so however the column is shuffled the two never overlap -- which is
    the point of routing through it, and is why the first version of this test could not fail.
    The only thing that leaks is a *splitter* that leaks, so that is what is substituted: one
    extra year of rows in `past`, the mistake a hand-rolled loop makes and the one
    `docs/method.md` rule #2 records this repo reading at 7.4 se.

    `tests/contracts/test_guards_are_load_bearing.py` deletes the marked block and requires
    this file to go red. This is the test that makes it do so.
    """
    import hub.draft.fit_corrections as mod

    def leaky(df, *, min_past=1, season_col="season"):
        for yr in sorted(df[season_col].unique().to_list())[1:]:
            yield (int(yr), df.filter(pl.col(season_col) <= yr),   # <= is the leak
                   df.filter(pl.col(season_col) == yr))

    monkeypatch.setattr(mod, "expanding_seasons", leaky)
    with pytest.raises(ValueError, match="leaks"):
        walk_forward(planted(-0.6), QB)


def test_the_walk_forward_scores_the_shipped_constant_as_its_own_arm():
    """The arm #48 acts on. `shipped` must be the shipped number applied to the
    no-correction fit, so a harness that quietly refitted the constant would score a model
    the board does not run."""
    panel = planted(-0.6)
    wf = walk_forward(panel, QB)
    assert {"mae_plain", "mae_fitted", "mae_shipped"} <= set(wf.columns)
    # Planted at -0.6 against a shipped -0.540, so the fitted arm must beat the plain one and
    # the shipped arm must land between them: close, because the constant is nearly right.
    assert mae(wf, "fitted") < mae(wf, "plain")
    assert mae(wf, "shipped") < mae(wf, "plain")


def test_the_shipped_arm_is_the_shipped_number_and_not_a_refit():
    """The distinguishing case: plant the coefficient far from the shipped value. A harness
    that refitted inside the `shipped` arm would still score well; one that truly applies
    -0.540 to data generated at +1.5 must score *worse* than no correction at all."""
    wf = walk_forward(planted(+1.5), QB)
    assert mae(wf, "shipped") > mae(wf, "plain"), (
        "applying -0.540 to a panel where the effect is +1.5 must hurt")


# --- the five, and their two counts --------------------------------------------


def test_the_five_coefficients_are_the_five_the_board_applies():
    """The registry against the modules. A coefficient renamed or re-keyed in
    `regression`/`durability` with this list left behind would refit a number nothing
    applies, and report a disposition about it."""
    from hub.draft import durability, regression
    live = {("hub.draft.regression", "TD_LUCK_BETA"): regression.TD_LUCK_BETA,
            ("hub.draft.durability", "BETA"): durability.BETA,
            ("hub.draft.durability", "INJURY_BETA"): durability.INJURY_BETA}
    assert len(COEFFICIENTS) == 5
    for coef in COEFFICIENTS:
        table = live[(coef.declared_in, coef.constant)]
        assert coef.key in table, f"{coef.name} keys {coef.constant} on a key it does not have"
        assert table[coef.key] == pytest.approx(coef.shipped), (
            f"{coef.name} is recorded here as {coef.shipped} and shipped as "
            f"{table[coef.key]}. The disposition is a claim about a specific number; update "
            f"it deliberately, which is issue #48's job and not a refit's.")


def test_the_designation_reports_both_of_its_sample_counts():
    """The acceptance criterion, and the defect it comes from: `docs/durability.md`
    publishes "1,263 player-seasons" for the regression and "n = 27" for the Out/Doubtful
    row, and nothing says those are one fit counted two ways. `n` and `n_signal` are both on
    the `Fit`, so a reader cannot see one without the other."""
    rng = np.random.default_rng(3)
    rows = []
    for season in SEASONS:
        for p in range(200):
            base = float(rng.uniform(2, 20))
            flag = 1.0 if p < 4 else 0.0        # 32 carriers in 1,600 rows
            rows.append({"season": season, "player_key": f"p{p}", "pos": "WR",
                         "base_xfp": base, "base_carry": base, "base_blend": base,
                         "designation": flag,
                         "target": 0.8 * base - 1.2 * flag + float(rng.normal(0, 2))})
    panel = pl.DataFrame(rows)
    coef = next(c for c in COEFFICIENTS if c.signal == "designation")
    fit = fitted(panel, coef)
    assert fit.n == 1600
    assert fit.n_signal == 32, "the carriers are the evidence, and they are not the row count"
    assert fit.n_signal < fit.n
    assert "1600" in fit.population and "every drafted position" in fit.population


def test_the_designation_is_fitted_at_every_position_and_the_traits_are_not():
    """`INJURY_BETA` applies at every position and `BETA`/`TD_LUCK_BETA` apply at two, and
    the harness has to fit each on the population its coefficient is applied to. A
    designation fitted per position would be a different number from the one shipped."""
    designation = next(c for c in COEFFICIENTS if c.signal == "designation")
    assert designation.position is None
    assert {c.position for c in COEFFICIENTS if c.signal != "designation"} == {"QB", "WR"}
    mixed = pl.concat([planted(-0.5, pos="QB"), planted(-0.5, pos="WR", seed=11)])
    assert _design(mixed, QB, HEADLINE_BASELINE).height == mixed.height / 2
    assert _design(mixed, designation._replace(signal="td_luck"),
                   HEADLINE_BASELINE).height == mixed.height


def test_every_fit_names_the_population_it_was_fitted_on():
    """The repo rule: a fitted number ships with its interval and its population. A blank
    or generic sentence is the failure this asserts against."""
    fit = fitted(planted(-0.6), QB)
    assert str(min(SEASONS)) in fit.population and str(max(SEASONS)) in fit.population
    assert "8 season clusters" in fit.population
    assert HEADLINE_BASELINE in fit.population
    assert "QB" in fit.population


def test_a_slice_too_thin_to_fit_returns_nothing_rather_than_a_number():
    assert fit_one(planted(-0.6, per_season=1, seasons=(2020, 2021)), QB) is None
    flat = planted(-0.6).with_columns(pl.lit(0.0).alias("td_luck"))
    assert fit_one(flat, QB) is None, "a signal that never varies carries no coefficient"


def test_the_report_names_every_coefficient_and_its_disposition():
    """The rendering, over the seam that produced it. Deliberately shallow: what a line says
    is per-line and changes, what it must not do is drop one of the five."""
    panel = pl.concat([
        planted(-0.6, pos="QB").with_columns(pl.lit(0.0).alias("missed"),
                                             pl.lit(0.0).alias("designation")),
        planted(-0.2, pos="WR", seed=11).with_columns(pl.lit(0.0).alias("missed"),
                                                      pl.lit(0.0).alias("designation"))])
    text = "\n".join(report_lines(panel, baselines=(HEADLINE_BASELINE,)))
    for coef in COEFFICIENTS:
        assert coef.name in text
        assert coef.documented_in in text
    assert any(d in text for d in DISPOSITIONS)


# --- the shrink sweep (#186) ---------------------------------------------------


def test_the_sweep_has_both_ends_of_the_line_on_it():
    """A sweep that omitted zero or one would be scoring the middle of a line against
    nothing: those two are the arms `walk_forward` already reports."""
    assert 0.0 in SHRINKS and 1.0 in SHRINKS
    assert list(SHRINKS) == sorted(SHRINKS)


def test_no_shrink_reproduces_the_no_correction_arm_exactly():
    """The equality the docstring claims, held rather than trusted. `lambda = 0` is not
    *approximately* the `plain` arm -- it is the same prediction on the same held-out rows
    from the same split, and a sweep that re-walked the panel could differ."""
    panel = planted(-0.6)
    curve = shrink_curve(panel, QB)
    at_zero = curve.filter(pl.col("shrink") == 0.0)["mae"][0]
    assert at_zero == pytest.approx(mae(walk_forward(panel, QB), "plain"), rel=1e-12)


def test_full_shrink_reproduces_the_shipped_arm_exactly():
    panel = planted(-0.6)
    curve = shrink_curve(panel, QB)
    at_one = curve.filter(pl.col("shrink") == 1.0)["mae"][0]
    assert at_one == pytest.approx(mae(walk_forward(panel, QB), "shipped"), rel=1e-12)


def test_a_constant_of_the_right_sign_and_size_is_kept_whole():
    """The sweep has to be able to say *keep it*, or its saying "remove it" means nothing.

    Planted at the shipped value exactly, so applying the constant whole is applying the
    truth and every shrink towards zero throws signal away.
    """
    curve = shrink_curve(planted(QB.shipped), QB)
    assert best_shrink(curve) == 1.0, curve
    assert (curve.sort("shrink")["mae"].to_list()
            == sorted(curve["mae"].to_list(), reverse=True)), (
        "MAE must fall monotonically towards the shrink that matches the truth")


def test_a_sign_reversed_constant_is_shrunk_to_nothing():
    """#186's case, on planted data where the answer is known before the run. The panel's
    effect is +0.9 and the shipped constant is -0.540, so every part of it applied is a
    markdown on players who should be marked up -- and no shrink between zero and one can
    rescue that. This is the shape `docs/fitted-corrections.md` reports for `td_luck.QB`."""
    curve = shrink_curve(planted(+0.9), QB)
    assert best_shrink(curve) == 0.0, curve
    worst = curve.sort("mae")["shrink"][-1]
    assert worst == 1.0, "applying a reversed constant whole must be the worst arm"


def test_a_constant_twice_the_size_of_the_truth_is_shrunk_by_half():
    """The middle of the sweep is the part `walk_forward` could not reach. Planted at half
    the shipped constant, so the measured answer is a *partial* shrink -- neither end."""
    curve = shrink_curve(planted(QB.shipped / 2.0), QB)
    assert best_shrink(curve) == 0.5, curve


def test_a_tie_goes_to_the_smaller_shrink():
    """Two shrinks the run cannot tell apart are two models it cannot tell apart, and the
    one that applies less of an unreproduced constant is the one that claims less."""
    tied = pl.DataFrame({"shrink": [1.0, 0.25, 0.5], "mae": [2.0, 2.0, 2.0],
                         "seasons": [7, 7, 7], "beats_none": [0, 0, 0]})
    assert best_shrink(tied) == 0.25


def test_the_season_count_is_reported_beside_the_mean():
    """Either alone misleads: a mean can be carried by one season, and a count says nothing
    about size. Both are on every row."""
    curve = shrink_curve(planted(-0.6), QB)
    assert set(curve.columns) == {"shrink", "mae", "seasons", "beats_none"}
    for row in curve.iter_rows(named=True):
        assert row["seasons"] == len(SEASONS) - 1, "every season but the first is scored"
        assert 0 <= row["beats_none"] <= row["seasons"]
    assert curve.filter(pl.col("shrink") == 0.0)["beats_none"][0] == 0, (
        "no correction cannot beat itself on any season")


def test_an_unfittable_panel_returns_an_empty_curve_rather_than_a_number():
    thin = planted(-0.6, per_season=1, seasons=(2020, 2021))
    curve = shrink_curve(thin, QB)
    assert curve.is_empty()
    assert best_shrink(curve) != best_shrink(curve), "an empty curve has no best, so NaN"


def test_the_report_prints_the_sweep_beside_the_walk_forward():
    panel = pl.concat([
        planted(-0.6, pos="QB").with_columns(pl.lit(0.0).alias("missed"),
                                             pl.lit(0.0).alias("designation")),
        planted(-0.2, pos="WR", seed=11).with_columns(pl.lit(0.0).alias("missed"),
                                                      pl.lit(0.0).alias("designation"))])
    text = "\n".join(report_lines(panel, baselines=(HEADLINE_BASELINE,)))
    assert "shrink sweep" in text
    assert "x0.00" in text and "x1.00" in text
