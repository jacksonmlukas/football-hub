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
    """`sd = 0.55 * mu` assumed spread is proportional to the mean. Fitted against 1,174
    player-seasons of nflverse weekly scoring, the exponent is 0.498 +/- 0.012 -- flat on
    the Poisson value and about 42 standard errors from 1.

    That is not a curve-fitting accident, it is what aggregating component stats produces:
    weekly points are a sum of count-driven pieces (receptions, carries, touchdowns) whose
    variance grows with their mean, so the spread of the total grows with its square root.
    See docs/weekly-spread.md.
    """
    got = predict.moments(pl.DataFrame({"proj_ppg": [4.0, 16.0],
                                        "position": ["WR", "WR"]}))
    # four times the mean is twice the spread, not four times
    assert got["sd"][1] / got["sd"][0] == pytest.approx(2.0, rel=0.01)


def test_a_boom_bust_profile_is_a_property_of_the_level_not_a_setting():
    """The consequence that matters for the draft board: relative volatility falls as
    projection rises. A 5-point-a-game flier really is nearly twice the lottery, per point,
    that a 20-point-a-game starter is -- and the old constant said they were identical."""
    got = predict.moments(pl.DataFrame({"proj_ppg": [5.0, 20.0],
                                        "position": ["RB", "RB"]}))
    cv = (got["sd"] / got["mu"]).to_list()
    assert cv[0] > 1.8 * cv[1]


def test_positions_keep_their_own_coefficient():
    got = predict.moments(pl.DataFrame({"proj_ppg": [10.0, 10.0],
                                        "position": ["QB", "WR"]}))
    assert got["sd"][0] < got["sd"][1]
    assert predict.WEEKLY_K["WR"] > predict.WEEKLY_K["QB"]


def test_a_missing_position_falls_back_to_the_pooled_coefficient():
    xp = pl.DataFrame({"proj_ppg": [9.0]},
                      schema={"proj_ppg": pl.Float64}).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("position"))
    got = predict.moments(xp)
    assert got["sd"][0] == pytest.approx(predict.WEEKLY_K_POOLED * 3.0, rel=1e-6)


def test_a_player_projected_at_nothing_has_no_spread():
    """Under the old floor a zero-projection player still carried sd = 2.0, which the
    best-lineup rule turned into free points. sqrt(0) is 0."""
    got = predict.moments(pl.DataFrame({"proj_ppg": [0.0], "position": ["WR"]}))
    assert got["sd"][0] == 0.0


def test_skew_is_carried_per_position():
    got = predict.moments(_frame())
    by = dict(zip(got["position"].to_list(), got["skew"].to_list(), strict=True))
    assert by["QB"] < by["RB"]


def test_the_numbers_are_pinned():
    """A refactor that changes a number is not a refactor.

    This assertion used to compare `predict.moments` against `season.weekly_moments`, which
    was an *alias for the same object* -- it compared a function with itself and could not
    fail. Pin the values instead, so a change to the law has to be deliberate.
    """
    got = predict.moments(_frame())
    assert got["mu"].to_list() == [18.0, 12.0, 5.0]
    assert got["sd"].to_list() == pytest.approx(
        [7.976164491784255, 7.170690343335151, 4.762824792074552], rel=1e-12)
    assert got["skew"].to_list() == [0.15, 0.67, 0.66]


def test_there_is_only_one_implementation_of_the_weekly_moments():
    """`predict.weekly_moments` was a dead twin of `predict.moments` -- same law, no skew,
    zero callers. Two functions for one quantity is how they drift, and the drift is silent
    because whichever one a caller picked still returned plausible numbers."""
    assert not hasattr(predict, "weekly_moments")


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


# --- a block that will not factor ------------------------------------------
#
# Issue #171. A block that is not positive semi-definite used to be caught and skipped, so
# that team's players were drawn independently -- the exact model the correlation structure
# exists to replace -- with no counter, no warning and nothing recorded. The draw it produces
# has the shape and dtype of a correlated one, so the count is the only evidence there is.


def _star(pass_catchers: int) -> tuple[np.ndarray, np.ndarray]:
    """One team: a quarterback and `pass_catchers` receivers, as (pos, nfl_team).

    Only the quarterback's edges are non-zero, so the block is a star and is PSD exactly
    while the receivers' squared correlations sum below one. At +0.232 that turns over
    between 18 and 19 receivers -- an absurd roster, and the point: this is a numerical
    property of `TEAMMATE_RHO` rather than a mock, so a refit that makes real blocks fail
    is caught by the same code path these tests drive.
    """
    pos = np.array(["QB"] + ["WR"] * pass_catchers)
    return pos, np.array(["ONE"] * (pass_catchers + 1))


def _teams(good: int, bad_pass_catchers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`good` two-man teams that factor, plus optionally one that does not."""
    pos: list[str] = []
    team: list[str] = []
    for i in range(good):
        pos += ["QB", "WR"]
        team += [f"T{i}", f"T{i}"]
    if bad_pass_catchers:
        p, t = _star(bad_pass_catchers)
        pos += p.tolist()
        team += t.tolist()
    return np.array(pos), np.array(team)


def test_a_block_that_will_not_factor_is_counted_rather_than_skipped():
    pos, team = _teams(good=20, bad_pass_catchers=19)
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, pos.size), pos, team,
                              report=report)
    assert report.blocks == 21, "every block carrying a correlation should be counted"
    assert report.independent == 1
    assert report.share == pytest.approx(1 / 21)


def test_a_run_that_lost_a_block_is_distinguishable_from_one_that_lost_none():
    """The whole defect: both runs return an array of the same shape and dtype, and before
    this there was nothing else to tell them apart."""
    clean_pos, clean_team = _teams(good=21)
    lost_pos, lost_team = _teams(good=20, bad_pass_catchers=19)
    clean = predict.CorrelationReport()
    lost = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, clean_pos.size),
                              clean_pos, clean_team, report=clean)
    predict.correlated_normal(np.random.default_rng(0), (4, lost_pos.size),
                              lost_pos, lost_team, report=lost)
    assert not clean.degraded()
    assert lost.degraded()
    assert clean.note() != lost.note()
    assert "independently" in lost.note()


def test_a_run_that_loses_more_than_the_floor_refuses():
    """It does not hand back a correlated simulation it did not perform. A single team,
    entirely independent, is 100% of the blocks."""
    pos, team = _star(19)
    with pytest.raises(predict.CorrelationVoid, match="would not factor"):
        predict.correlated_normal(np.random.default_rng(0), (4, pos.size), pos, team)


def test_the_floor_is_a_share_and_not_a_count():
    """One lost block in twenty-one is inside the floor and reported; two in twenty-two is
    outside it and refused. A guard keyed to the count alone would treat these the same."""
    inside_pos, inside_team = _teams(good=20, bad_pass_catchers=19)
    predict.correlated_normal(np.random.default_rng(0), (4, inside_pos.size),
                              inside_pos, inside_team)
    report = predict.CorrelationReport(blocks=22, independent=2)
    assert report.share > report.floor
    with pytest.raises(predict.CorrelationVoid):
        report.check()


def test_the_accounting_does_not_move_a_run_with_nothing_to_report():
    """A clean run is unchanged. The counters read the block the Cholesky already built,
    so nothing about the draw depends on whether a report was passed."""
    pos, team = _teams(good=8)
    kw = {"pos": pos, "nfl_team": team}
    with_report = predict.correlated_normal(np.random.default_rng(3), (200, pos.size),
                                            report=predict.CorrelationReport(), **kw)
    without = predict.correlated_normal(np.random.default_rng(3), (200, pos.size), **kw)
    assert np.array_equal(with_report, without)


def test_a_team_with_nothing_to_correlate_is_not_counted_as_a_block():
    """Two receivers correlate at zero, so their block is the identity and independence
    costs it nothing. Counting it would make the share read low exactly when the teams that
    did lose something were few."""
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, 2), np.array(["WR", "WR"]),
                              np.array(["ONE", "ONE"]), report=report)
    assert report.blocks == 0
    assert report.note() == "correlation: no team block carried a correlation to apply."


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


def test_the_scoring_module_does_not_re_export_correlation():
    """It briefly did, for one caller in `lineup.py`. That re-export made `predict` import
    its own symbol back out of `components` -- a cycle that only stayed legal because the
    import sat inside a function body. One owner, no round trip."""
    from hub.models import components as C
    assert not hasattr(C, "teammate_rho")
    assert not hasattr(C, "group_sd")
