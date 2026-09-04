"""Where the component projection's error lives, priced in points.

All offline. The fetch is one function and it is the only network-bound thing here; the
statistics take a paired frame and are exercised directly, which is the split `hub.models.panel`
made for the same reason.
"""
import numpy as np
import polars as pl

from hub.models import component_error as CE
from hub.models.components import SCORING


def _paired(n=400, seed=0, **over):
    """A paired frame with a known relationship, so the statistics have a right answer.

    One noise draw shared by every component, deliberately: that makes the raw error identical
    across all seven, so anything that separates them in the scorecard can only be the scoring
    weight. Drawing fresh noise per component would leave the ranking confounded.
    """
    rng = np.random.default_rng(seed)
    p = rng.gamma(3.0, 2.0, n)
    noise = rng.normal(0, 1.0, n)
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:
        d[f"p_{k}"] = p
        d[f"a_{k}"] = p + noise
    d.update(over)
    return pl.DataFrame(d)


def test_components_are_ranked_by_what_the_league_pays_for_them():
    """The point of the module. A yard of error and a touchdown of error are not comparable
    until both are multiplied by the scoring weight, and ranked raw they come out backwards."""
    got = CE.scorecard(_paired())
    pts = dict(zip(got["component"].to_list(), got["points"].to_list(), strict=True))
    mae = dict(zip(got["component"].to_list(), got["mae"].to_list(), strict=True))
    # identical error in every component, so the ranking must be the scoring weight alone
    assert got["component"][0] == max(pts, key=lambda k: abs(SCORING[k]))
    assert mae["receiving_yards"] == mae["receiving_tds"]
    assert pts["receiving_tds"] > pts["receiving_yards"], "6 points beats 0.1 a yard"


def test_the_scorecard_is_sorted_by_points_not_by_raw_error():
    got = CE.scorecard(_paired())
    assert got["points"].to_list() == sorted(got["points"].to_list(), reverse=True)


def test_a_perfect_projection_scores_zero_error_and_unit_slope():
    n = 300
    p = np.linspace(1, 20, n)
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:
        d[f"p_{k}"] = p
        d[f"a_{k}"] = p
    got = CE.scorecard(pl.DataFrame(d))
    assert got["mae"].max() == 0.0
    assert all(abs(s - 1.0) < 1e-9 for s in got["slope"].to_list())


def test_an_over_dispersed_projection_reports_a_slope_below_one():
    """The finding the module exists to pin: 28 of 28 season-components came back below 1."""
    n, rng = 400, np.random.default_rng(1)
    p = rng.normal(10, 3, n)
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:                      # realised deviates only half as far as projected
        d[f"p_{k}"] = p
        d[f"a_{k}"] = 10 + 0.5 * (p - 10) + rng.normal(0, 0.1, n)
    got = CE.scorecard(pl.DataFrame(d))
    assert all(s < 1.0 for s in got["slope"].to_list())
    assert all(abs(s - 0.5) < 0.05 for s in got["slope"].to_list())


def test_a_component_the_source_did_not_provide_is_skipped_not_zeroed():
    """A missing component must not enter the budget as a zero-error one, which would make the
    projection look better the less of it there is."""
    d = _paired().drop("p_receiving_tds", "a_receiving_tds")
    got = CE.scorecard(d)
    assert "receiving_tds" not in got["component"].to_list()
    assert got.height == len(CE.COMPONENTS) - 1


def test_an_empty_frame_yields_the_right_empty_shape():
    got = CE.scorecard(pl.DataFrame({"season": []}))
    assert got.height == 0
    assert set(got.columns) >= {"component", "corr", "slope", "mae", "points"}


# --- the calibration, and the null it returns ---

def test_calibration_is_fitted_on_the_past_only():
    """Leak-free by construction. Fitting on the season being scored would manufacture a gain."""
    past = _paired(n=300, seed=2).with_columns(pl.lit(2023).alias("season"))
    now = _paired(n=300, seed=3)
    got = CE.calibrated(past, now)
    assert got["n"] == 300
    assert got["raw_mae"] > 0 and got["cal_mae"] > 0


def test_calibration_helps_when_the_projection_really_is_mis_scaled():
    """The control. If a projection is genuinely off by a constant factor, fitting that factor
    on the past and applying it forward must help -- otherwise the harness proves nothing."""
    rng = np.random.default_rng(4)
    def frame(season):
        p = rng.normal(10, 3, 500)
        d: dict[str, object] = {"season": [season] * 500}
        for k in CE.COMPONENTS:
            d[f"p_{k}"] = p * 2.0                # projection is double the truth, every season
            d[f"a_{k}"] = p + rng.normal(0, 0.2, 500)
        return pl.DataFrame(d)
    got = CE.calibrated(frame(2023), frame(2024))
    assert got["cal_mae"] < got["raw_mae"], "a real mis-scaling must be correctable"


def test_the_verdict_refuses_a_calibration_that_worsens_the_linear_loss():
    """Fantasy points are linear, so MAE decides. A correction that buys RMSE at MAE's expense
    is the shape least squares produces by definition and is not an improvement."""
    rounds = [{"raw_mae": 3.5, "cal_mae": 3.6, "raw_rmse": 6.6, "cal_rmse": 6.5},
              {"raw_mae": 3.2, "cal_mae": 3.3, "raw_rmse": 5.7, "cal_rmse": 5.5}]
    status, note = CE.verdict(rounds)
    assert status == "NULL"
    assert "MAE" in note and "RMSE" in note


def test_the_verdict_adopts_only_when_every_held_out_season_improves():
    every = [{"raw_mae": 3.5, "cal_mae": 3.4, "raw_rmse": 6.6, "cal_rmse": 6.5},
             {"raw_mae": 3.2, "cal_mae": 3.1, "raw_rmse": 5.7, "cal_rmse": 5.6}]
    assert CE.verdict(every)[0] == "ADOPT"
    one_bad = [*every[:1], {"raw_mae": 3.2, "cal_mae": 3.3, "raw_rmse": 5.7, "cal_rmse": 5.6}]
    assert CE.verdict(one_bad)[0] == "NULL", "one season against it is enough"


def test_nothing_measured_is_reported_rather_than_crashing():
    assert CE.verdict([])[0] == "NULL"


def test_the_report_names_the_receiving_share_of_the_budget():
    lines = "\n".join(CE.report(CE.scorecard(_paired()), []))
    assert "error budget" in lines and "receiving game is" in lines
