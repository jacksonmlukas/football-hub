"""One prediction object: what a player does in a week.

`docs/next.md`. The pieces existed but as three implementations of one idea --
`weekly_moments` handed back two moments, the skewed correlated draw lived inside
`simulate_weeks`, and `lineup.py` computed its own group spread. Three copies is how they
drift apart, and drift is silent.

The split is by subject rather than by convenience: **this module is what a player does, and
`hub.draft.season` is how a league works.** Dispersion laws, skew, talent and correlation
live here; rosters, schedules, brackets and lineups live there.

A constraint the unification has to respect. Component-derived spread was measured *worse*
than the fitted square-root law at predicting a player's actual weekly sd -- mean error 1.365
against 1.140, P(better) 0.0% (`docs/component-projection.md`). So the object keeps the
fitted laws for its moments and exposes components alongside them. Unifying must not quietly
swap a validated number for a tidier one.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import predict


def _frame():
    return pl.DataFrame({"proj_ppg": [18.0, 12.0, 5.0],
                         "position": ["QB", "RB", "WR"]})


# --- the object owns the fitted laws --------------------------------------

def test_moments_carry_mean_spread_and_skew():
    got = predict.moments(_frame())
    assert {"mu", "sd", "skew"} <= set(got.columns)


def test_spread_still_follows_the_square_root_law():
    """The law is fitted (exponent 0.498 +/- 0.012) and must survive the move."""
    got = predict.moments(pl.DataFrame({"proj_ppg": [4.0, 16.0],
                                        "position": ["WR", "WR"]}))
    assert got["sd"][1] / got["sd"][0] == pytest.approx(2.0, rel=0.01)


def test_skew_is_carried_per_position():
    got = predict.moments(_frame())
    by = dict(zip(got["position"].to_list(), got["skew"].to_list()))
    assert by["QB"] < by["RB"]


def test_the_numbers_are_unchanged_by_the_move():
    """A refactor that changes a number is not a refactor. `hub.draft.season` re-exports
    these, and both routes must agree exactly."""
    from hub.draft import season
    a = predict.moments(_frame())
    b = season.weekly_moments(_frame())
    assert a["mu"].to_list() == b["mu"].to_list()
    assert a["sd"].to_list() == b["sd"].to_list()


def test_season_still_re_exports_for_back_compat():
    """Callers in calibrate, leverage and optimize import these from season today. Moving
    the definition must not break them."""
    from hub.draft import season
    assert season.TALENT_CV == predict.TALENT_CV
    assert season.WEEKLY_K == predict.WEEKLY_K
    assert season.talent_cv_for(np.array(["RB"]))[0] == predict.TALENT_CV_BY_POS["RB"]


# --- the draw ---------------------------------------------------------------

def test_a_draw_has_the_mean_spread_and_skew_it_was_asked_for():
    rng = np.random.default_rng(0)
    z = predict.correlated_normal(rng, (60000,), np.array(["WR"]), None)
    got = predict.skewed(12.0, 7.0, 0.66, z)
    assert got.mean() == pytest.approx(12.0, rel=0.03)
    assert got.std() == pytest.approx(7.0, rel=0.06)


def test_teammates_correlate_and_others_do_not():
    rng = np.random.default_rng(1)
    pos = np.array(["QB", "WR"])
    same = predict.correlated_normal(rng, (40000, 2), pos, np.array(["KC", "KC"]))
    apart = predict.correlated_normal(rng, (40000, 2), pos, np.array(["KC", "DEN"]))
    assert np.corrcoef(same[:, 0], same[:, 1])[0, 1] > 0.15
    assert abs(np.corrcoef(apart[:, 0], apart[:, 1])[0, 1]) < 0.05


# --- components live on the same object -----------------------------------

def test_the_component_line_is_reachable_from_the_same_object():
    """The reason to unify: a consumer that wants stats rather than points -- the props
    audit, a future volume model -- reads them from here instead of reaching into
    hub.models.volume separately."""
    got = predict.components(pick=12, position="WR", proj_ppg=15.5)
    assert got["receiving_yards"] > 0
    from hub.models.components import points
    assert points(got) == pytest.approx(15.5, rel=1e-6)


def test_components_are_empty_for_a_position_with_no_fitted_curve():
    assert predict.components(pick=50, position="K", proj_ppg=8.0) == {}


# --- correlation is owned here, not by the scoring module -----------------

def test_teammate_correlation_is_owned_by_the_prediction_module():
    """It was defined in `hub.models.components` and used two ways -- analytically by
    `lineup.py` for a closed-form spread, and by sampling in the league simulator. Two uses
    of one table is fine; the table living in the *scoring* module is not. Scoring is how
    stats become points, and who moves together is a prediction."""
    assert predict.teammate_rho("QB", "WR") > 0.2
    assert predict.teammate_rho("WR", "WR") == 0.0


def test_group_spread_counts_teammate_covariance():
    lone = predict.group_sd([(7.0, "QB", "KC"), (6.0, "WR", "DEN")])
    stack = predict.group_sd([(7.0, "QB", "KC"), (6.0, "WR", "KC")])
    assert stack > lone


def test_components_still_re_exports_correlation_for_back_compat():
    from hub.models import components as C
    assert C.teammate_rho("QB", "TE") == predict.teammate_rho("QB", "TE")
    assert C.group_sd([(5.0, "QB", None)]) == predict.group_sd([(5.0, "QB", None)])
