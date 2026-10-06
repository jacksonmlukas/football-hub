"""Where the component projection's error lives, priced in points.

All offline. The fetch is one function and it is the only network-bound thing here; the
statistics take a paired frame and are exercised directly, which is the split `hub.models.panel`
made for the same reason.
"""
import numpy as np
import polars as pl
import pytest

from hub.ledger import Ledger
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


SEASONS = (2022, 2023, 2024, 2025, 2026)


def _seasons(scale=2.0, seed=4, n=300, seasons=SEASONS, noise=0.2):
    """Several seasons of pairs, `player_id` carried: the projection is `scale` times the truth
    in every season, so the calibration fitted on earlier seasons really does help."""
    rng = np.random.default_rng(seed)
    frames = []
    for season in seasons:
        p = rng.normal(10, 3, n)
        d: dict[str, object] = {"season": [season] * n,
                                "player_id": [f"p{season}-{i}" for i in range(n)]}
        for k in CE.COMPONENTS:
            d[f"p_{k}"] = p * scale
            d[f"a_{k}"] = p + rng.normal(0, noise, n)
        frames.append(pl.DataFrame(d))
    return pl.concat(frames)


def test_a_real_mis_scaling_is_taken_by_the_gate_in_every_held_out_season():
    """The control that can fail: a projection that is double the truth in every season, so the
    calibration fitted on earlier seasons improves the points error in every held-out season by
    a margin far above its noise, and the in-sample ceiling is the same size. ADOPT, via the
    shared rule."""
    label, note = CE.verdict(CE.calibration_frame(_seasons()))
    assert label == "ADOPT", note


def test_a_calibration_that_worsens_the_linear_loss_is_not_taken():
    """Fantasy points are linear, so MAE decides. A projection that is already calibrated has
    nothing for a fitted line to add: the held-out error does not improve, and the Gate says
    SHOW or NOT-RUNNABLE, never ADOPT."""
    label, _ = CE.verdict(CE.calibration_frame(_seasons(scale=1.0, noise=1.0)))
    assert label != "ADOPT"


def test_one_season_against_it_is_enough_to_withhold_adoption():
    frame = CE.calibration_frame(_seasons())
    worse = frame.with_columns(
        pl.when(pl.col("season") == 2026).then(-pl.col("diff").abs()).otherwise(pl.col("diff"))
        .alias("diff"))
    run = CE.gate_run(worse)
    assert run.verdict[0] != "ADOPT"
    assert run.summary["clusters"] == 4, "the season is the cluster: 2022 is only training"


def test_a_ceiling_the_design_cannot_resolve_is_not_runnable_not_a_null():
    """Stage 2, read for the first time on this comparison (#343): with the in-sample ceiling
    this small and three seasons, the design could not have seen a gain, and says so."""
    frame = CE.calibration_frame(_seasons(seasons=(2023, 2024, 2025, 2026), scale=1.0, noise=1.0))
    run = CE.gate_run(frame)
    assert run.verdict[0] == "NOT-RUNNABLE"
    assert "not planned" in run.verdict[1]


def test_the_calibration_frame_is_one_row_per_held_out_player_season():
    frame = CE.calibration_frame(_seasons())
    assert frame.height == 4 * 300 and set(frame["season"].to_list()) == {2023, 2024, 2025, 2026}
    assert {"season", "player_id", "diff", "ceiling_diff"} == set(frame.columns)


def test_the_per_season_gain_is_the_aggregate_the_published_table_reports():
    """Equivalence with the retired reading: the mean of the row-level points error across a
    held-out season is `calibrated`'s own `raw_mae - cal_mae` for that season."""
    paired = _seasons(scale=1.3, noise=0.5)
    frame = CE.calibration_frame(paired)
    for target, past, now in CE.expanding_seasons(paired):
        agg = CE.calibrated(past, now)
        row_gain = float(frame.filter(pl.col("season") == target)["diff"].to_numpy().mean())
        assert row_gain == pytest.approx(agg["raw_mae"] - agg["cal_mae"], abs=1e-9)


def test_the_verdict_carries_the_rmse_context_when_it_is_not_adopted():
    rounds = [{"raw_mae": 3.5, "cal_mae": 3.6, "raw_rmse": 6.6, "cal_rmse": 6.5},
              {"raw_mae": 3.2, "cal_mae": 3.3, "raw_rmse": 5.7, "cal_rmse": 5.5}]
    label, note = CE.verdict(CE.calibration_frame(_seasons(scale=1.0, noise=1.0)), rounds)
    assert label != "ADOPT" and "RMSE in 2 of 2" in note and "MAE is the loss" in note


def test_nothing_measured_is_reported_rather_than_crashing():
    label, note = CE.verdict(pl.DataFrame())
    assert label == "SHOW" and "Nothing measured" in note


def test_the_published_run_prints_stage_2_and_the_stamps_and_writes_one_entry():
    ledger = Ledger(path=None)
    run = CE.gate_run(CE.calibration_frame(_seasons()), publish=True, seasons=SEASONS,
                      ledger=ledger)
    text = "\n".join(run.lines)
    assert "MDE at 80% power" in text and "ceiling (" in text
    assert "data:" in text and "board:" in text and "commit:" in text
    assert [e.name for e in ledger._entries] == ["component_calibration"]


def test_the_report_names_the_receiving_share_of_the_budget():
    lines = "\n".join(CE.report(CE.scorecard(_paired()), []))
    assert "error budget" in lines and "receiving game is" in lines


# --- the decomposition, whose whole value is that it adds up ---

def test_the_components_sum_to_the_gap_in_the_total():
    """The property that makes this an explanation rather than a decoration. If the parts do
    not add to the whole, "we have him at two more receptions" is a story, not an account."""
    rng = np.random.default_rng(5)
    n = 200
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:
        d[f"p_{k}"] = rng.gamma(2.0, 3.0, n)
        d[f"a_{k}"] = rng.gamma(2.0, 3.0, n)
    f = pl.DataFrame(d)
    got = CE.attribution(f)
    parts = float(got["points"].sum())
    # the gap in the total, computed independently
    whole = sum(float((f[f"p_{k}"] - f[f"a_{k}"]).to_numpy().mean()) * SCORING[k]
                for k in CE.COMPONENTS)
    assert parts == pytest.approx(whole)


def test_a_component_we_get_exactly_right_contributes_nothing():
    n = 50
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:
        v = np.linspace(1, 5, n)
        d[f"p_{k}"], d[f"a_{k}"] = v, v
    got = CE.attribution(pl.DataFrame(d))
    assert got["points"].abs().max() == pytest.approx(0.0)


def test_over_projecting_a_component_contributes_positively():
    """Sign convention: projected minus realised, so over-projection is a positive contribution
    and the reader can tell which way we are wrong without consulting a docstring."""
    n = 40
    d: dict[str, object] = {"season": [2024] * n}
    for k in CE.COMPONENTS:
        d[f"p_{k}"] = np.full(n, 2.0)
        d[f"a_{k}"] = np.full(n, 1.0)
    got = CE.attribution(pl.DataFrame(d))
    by = dict(zip(got["component"].to_list(), got["points"].to_list(), strict=True))
    assert by["receiving_tds"] == pytest.approx(SCORING["receiving_tds"])
    assert all(v > 0 for v in by.values())


def test_grouping_splits_the_account_without_breaking_it():
    """Each group's parts must still sum to that group's gap -- otherwise grouping would be a
    different statistic wearing the same name."""
    rng = np.random.default_rng(6)
    n = 120
    d: dict[str, object] = {"pos": ["WR"] * 60 + ["RB"] * 60}
    for k in CE.COMPONENTS:
        d[f"p_{k}"] = rng.gamma(2.0, 3.0, n)
        d[f"a_{k}"] = rng.gamma(2.0, 3.0, n)
    f = pl.DataFrame(d)
    got = CE.attribution(f, by="pos")
    assert set(got["pos"].to_list()) == {"WR", "RB"}
    for pos in ("WR", "RB"):
        sub = f.filter(pl.col("pos") == pos)
        parts = float(got.filter(pl.col("pos") == pos)["points"].sum())
        whole = sum(float((sub[f"p_{k}"] - sub[f"a_{k}"]).to_numpy().mean()) * SCORING[k]
                    for k in CE.COMPONENTS)
        assert parts == pytest.approx(whole), pos


def test_an_empty_frame_decomposes_into_nothing():
    got = CE.attribution(pl.DataFrame({"season": []}))
    assert got.height == 0


def test_the_report_prints_the_held_out_table_and_the_gate_sentence_when_given_them():
    rounds = [{"season": 2024.0, "raw_mae": 3.5, "cal_mae": 3.4, "raw_rmse": 6.6, "cal_rmse": 6.5}]
    text = "\n".join(CE.report(CE.scorecard(_paired()), rounds, "ADOPT: taken"))
    assert "held out" in text and "2024" in text and "ADOPT: taken" in text


def test_a_component_with_no_spread_or_no_column_is_left_out_of_both_arms():
    """`_fits` skips a constant projection and `_points_error` skips what has no fit or no
    column, so the raw and calibrated errors are always summed over the same components."""
    frame = _paired(n=50).drop("p_receptions", "a_receptions").with_columns(
        pl.lit(1.0).alias("p_rushing_tds"))
    fits = CE._fits(frame)
    assert "receptions" not in fits and "rushing_tds" not in fits
    assert "receiving_yards" in fits
    err = CE._points_error(frame, dict.fromkeys(CE.COMPONENTS, (1.0, 0.0)))
    assert err.shape == (50,) and (err >= 0).all()
