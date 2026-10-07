"""#309: the published interval carries estimation error, is calibrated empirically within
position, and the gate grades the whole board it scored.

Each test here is a positive control in the sense of `docs/method.md` rule 18: it plants the
condition the check exists to detect and the check has been seen red on it (the mutation is
named in the test's docstring and was applied, observed red, and reverted before the test was
trusted).
"""
import math

import numpy as np
import polars as pl
import pytest

from hub.models import conformal, coverage, predict


def _stats(rows):
    """nflverse-shaped weekly stats: what `player_weeks` is handed."""
    return pl.DataFrame(rows, schema={
        "player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
        "week": pl.Int32, "season_type": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _player(pid, pos, season, points):
    return [{"player_id": pid, "position": pos, "season": season, "week": w + 1,
             "season_type": "REG", "fantasy_points_ppr": float(p)}
            for w, p in enumerate(points)]


def _drawn(n_players=400, weeks=17, pos="WR", mu=12.0, seed=0, spread=1.0, seasons=(2024,)):
    """Weeks drawn through the shipped law around a fixed true mean, so the centre built from
    earlier weeks is the only estimated thing; `spread` above 1 is a model whose intervals are
    too narrow. Same construction as `test_coverage._drawn`."""
    rng = np.random.default_rng(seed)
    sd = predict.WEEKLY_K[pos] * math.sqrt(mu) * spread
    sk = predict.WEEKLY_SKEW[pos]
    rows = []
    for season in seasons:
        for p in range(n_players):
            z = rng.standard_normal(weeks)
            rows += _player(f"p{p}", pos, season, predict.skewed(mu, sd, sk, z))
    return _stats(rows)


# --- the order statistics, taken by rank ---------------------------------


def test_the_upper_rank_is_ceil_of_p_times_n_plus_one_and_survives_float_dust():
    """`0.07 * 100` is 7.000000000000001, so a bare `ceil` reads rank 8 where the rank is 7.
    Mutation: drop the `round(..., 9)` in `order_statistic` and this reads 8."""
    s = np.arange(1.0, 100.0)               # n = 99, values equal their ranks
    assert conformal.order_statistic(s, 0.07, upper=True) == 7.0


def test_the_lower_rank_is_floor_of_p_times_n_plus_one_and_survives_float_dust():
    """`0.29 * 100` is 28.999999999999996, so a bare `floor` reads rank 28 where it is 29.
    Mutation: `floor` -> `ceil` in the lower branch, or dropping the rounding, reads 28 or 30."""
    s = np.arange(1.0, 100.0)
    assert conformal.order_statistic(s, 0.29, upper=False) == 29.0


def test_a_rank_past_either_end_is_the_infinity_that_covers_everything():
    s = np.arange(1.0, 6.0)                 # n = 5: ceil(0.9 * 6) = 6 > 5; floor(0.1 * 6) = 0
    assert conformal.order_statistic(s, 0.90, upper=True) == float("inf")
    assert conformal.order_statistic(s, 0.10, upper=False) == float("-inf")
    assert conformal.order_statistic([], 0.90, upper=True) == float("inf")


def test_the_two_one_sided_ranks_cover_at_least_the_nominal_in_fresh_draws():
    """The guarantee itself, by Monte Carlo: calibrate on 250 skewed scores, score fresh ones.
    Mutation: swap `lower_p`/`upper_p` in `mondrian` and the interval inverts, coverage ~0."""
    rng = np.random.default_rng(11)
    cover = []
    for _ in range(400):
        cal = rng.gamma(2.0, 1.0, 250) - 2.0
        fresh = rng.gamma(2.0, 1.0, 400) - 2.0
        c = conformal.mondrian(cal, np.array(["x"] * 250), "x", lower_p=0.10, upper_p=0.90)
        assert c is not None
        cover.append(np.mean((fresh >= c.q_lo) & (fresh <= c.q_hi)))
    assert float(np.mean(cover)) == pytest.approx(0.80, abs=0.012)
    assert float(np.mean(cover)) >= 0.80 - 0.004, "finite-sample correction: never below nominal"


# --- Mondrian, with a named fallback -------------------------------------


def test_a_group_with_enough_rows_is_calibrated_on_its_own():
    scores = np.concatenate([np.linspace(-1, 1, 250), np.linspace(-5, 5, 300)])
    groups = np.array(["QB"] * 250 + ["WR"] * 300)
    c = conformal.mondrian(scores, groups, "QB", lower_p=0.10, upper_p=0.90)
    assert c is not None and not c.fell_back and c.n_cal == 250
    assert -1.0 <= c.q_lo < 0 < c.q_hi <= 1.0, "QB's own scores, not the pooled spread"


def test_a_group_under_the_floor_borrows_the_pool_and_says_so():
    """Mutation: make `fell_back` always False, or drop the pooled branch; either is red."""
    scores = np.concatenate([np.linspace(-1, 1, 50), np.linspace(-5, 5, 300)])
    groups = np.array(["QB"] * 50 + ["WR"] * 300)
    c = conformal.mondrian(scores, groups, "QB", lower_p=0.10, upper_p=0.90)
    assert c is not None and c.fell_back and c.n_cal == 50, "n_cal is the group's own count"
    assert c.q_hi > 1.0, "the pooled quantile, which is wider than QB's own"


def test_a_pool_under_the_floor_is_not_a_calibration():
    scores = np.linspace(-1, 1, 120)
    groups = np.array(["QB"] * 120)
    assert conformal.mondrian(scores, groups, "QB", lower_p=0.10, upper_p=0.90) is None


def test_history_is_strictly_earlier_in_time_season_first_and_bounded():
    cells = [(2024, 1), (2023, 17), (2023, 18), (2024, 2), (2024, 3)]   # arrives unsorted
    assert conformal.history(cells, (2024, 2)) == [(2023, 17), (2023, 18), (2024, 1)]
    assert conformal.history(cells, (2024, 2), window=2) == [(2023, 18), (2024, 1)]
    assert conformal.history(cells, (2023, 17)) == [], "a later season never reaches an earlier"


# --- estimation error enters the interval --------------------------------


def test_the_predictive_sd_is_the_shape_laws_variance_plus_the_means_estimation_error():
    sd = np.array([5.0, 5.0, 5.0])
    got = predict.predictive_sd(sd, np.array([1.0, 4.0, 16.0]))
    assert got == pytest.approx(5.0 * np.sqrt(np.array([2.0, 1.25, 1.0625])))
    assert predict.predictive_sd(5.0, float("nan")) == pytest.approx(5.0 * math.sqrt(2.0))
    assert predict.predictive_sd(5.0, 0.0) == pytest.approx(5.0 * math.sqrt(2.0))


def test_moments_sd_is_not_inflated_the_simulators_carry_their_own_mean_uncertainty():
    """The estimation term is the interval's, not `moments`': a draw simulator that read it
    would count the mean's uncertainty twice."""
    frame = pl.DataFrame({"proj_ppg": [12.0], "position": ["WR"], "n_prior": [4]})
    got = predict.moments(frame)
    assert float(got["sd"][0]) == pytest.approx(predict.WEEKLY_K["WR"] * math.sqrt(12.0))


def test_graded_carries_the_estimation_aware_scale_only_where_the_centre_was_estimated():
    stats = _drawn(n_players=6, weeks=10, seed=1)
    prior = coverage.graded(coverage.centred(coverage.player_weeks(stats), "prior"))
    want = prior["sd"].to_numpy() * np.sqrt(1.0 + 1.0 / prior["n_prior"].to_numpy())
    assert prior["sd_pred"].to_numpy() == pytest.approx(want)
    real = coverage.graded(coverage.centred(coverage.player_weeks(stats), "realised"))
    assert real["sd_pred"].to_numpy() == pytest.approx(real["sd"].to_numpy()), (
        "the lookahead centre has no estimation error to add; that is what it flatters")


def _bucket_cov(g: pl.DataFrame, lo: str, hi: str, low: int, high: int) -> float:
    n = g["n_prior"].to_numpy()
    m = (n >= low) & (n <= high)
    y = g["points"].to_numpy()
    return float(np.mean(((y >= g[lo].to_numpy()) & (y <= g[hi].to_numpy()))[m]))


def test_an_interval_that_omits_estimation_error_under_covers_where_the_term_matters(
        monkeypatch):
    """The planted case. Weeks are drawn through the shipped law around a fixed true mean, so
    a centre built from four or five earlier weeks is the only thing wrong with the interval;
    the parametric interval at the estimation-aware scale must cover near 80% there, and the
    same interval with the term removed must not.

    Mutation: `predictive_sd` returning `sd` unchanged -- this test reads red, and the
    `monkeypatch` below is that mutation applied from outside.
    """
    stats = _drawn(n_players=250, weeks=17, seasons=(2022, 2023, 2024), seed=3)
    sample = coverage.centred(coverage.player_weeks(stats), "prior")
    with_term = _bucket_cov(coverage.graded(sample), "pe10", "pe90", 4, 5)
    assert with_term > 0.76, f"estimation error did not close the thin end: {with_term:.3f}"
    monkeypatch.setattr(predict, "predictive_sd", lambda sd, games: np.asarray(sd, dtype=float))
    without = _bucket_cov(coverage.graded(sample), "pe10", "pe90", 4, 5)
    assert without < 0.75, f"the planted omission should under-cover at 4-5 games: {without:.3f}"
    assert with_term - without > 0.03


# --- conformalisation, and what it is for --------------------------------


def test_the_published_interval_covers_at_nominal_when_the_shape_law_is_wrong():
    """Weeks drawn 30% wider than the law believes: the parametric interval under-covers, the
    conformal one does not -- that is the whole of what the construction promises.

    Mutation: replace `calibrate`'s empirical bounds with the parametric `pe10`/`pe90` and the
    published coverage reads ~0.67."""
    stats = _drawn(n_players=250, weeks=17, spread=1.3, seasons=(2022, 2023, 2024), seed=3)
    got = coverage.measure(stats, "prior")
    assert got["parametric"]["gate_cov80"] < 0.70, "the planted miss is real"
    assert got["gate_cov80"] == pytest.approx(0.80, abs=0.02)


def test_a_shift_the_window_has_not_seen_is_a_miss_not_a_pass():
    """Exchangeability fails when a later season is wider than the one it calibrates on, and
    the gate must say so (UNDER-COVERS against a claim of 0.80) rather than absorb it. A
    conformal gate that could not fail would be rule 18's decorative check.

    The first season is drawn at the model's own width, the second 60% wider."""
    a = _drawn(n_players=250, weeks=17, seasons=(2023,), seed=5)
    b = _drawn(n_players=250, weeks=17, spread=1.6, seasons=(2024,), seed=6)
    got = coverage.measure(pl.concat([a, b]), "prior", claim=0.80)
    pool = next(r for r in got["by_position"] if r["group"] == "all")
    assert pool["cov80"] < 0.78
    assert got["verdict"] == "UNDER-COVERS"


# --- the gate measures the whole board -----------------------------------


def test_the_gate_reads_the_whole_scored_board_not_a_subset():
    """The control for a gate that measures only part of the board (#309). A table whose pool
    covers 0.60 and whose unclipped half covers 0.80 must gate at 0.60.

    Mutation (observed): `measure` building the pool from `raw_lo > 0` rows only -- the
    2026-09-13 gate, a subset of the board -- is red in the next test, which counts both
    halves; this one pins that the gate reads the pool row and that `GATE_SUBSET` is "all"."""
    rows = [{"group": "QB", "cov80": 0.79, "n": 10}, {"group": "all", "cov80": 0.60, "n": 30}]
    assert coverage.gate_population(rows)["cov80"] == 0.60
    assert coverage.GATE_SUBSET == "all"


def test_the_gate_counts_the_clipped_weeks_and_reports_their_share():
    """Mutation: filter the scored rows to `raw_lo > 0` before the pool is built and `gate_n`
    stops being the sum of the two halves."""
    stats = _drawn(n_players=200, seed=5)
    low = _drawn(n_players=200, mu=2.6, seed=6, pos="TE")           # clipped board: mu near 2
    got = coverage.measure(pl.concat([stats, low.with_columns(
        pl.col("player_id") + "_low")]), "prior")
    halves = {r["group"]: r["n"] for r in got["floor_split"]}
    assert set(halves) == {"clipped at zero", "strictly positive"}
    assert got["gate_n"] == sum(halves.values()), "the gate graded both halves of the board"
    assert got["gate_clipped_share"] == pytest.approx(halves["clipped at zero"] / got["gate_n"])
    assert 0.05 < got["gate_clipped_share"] < 0.95, "the fixture has both kinds of week"


def test_the_gate_names_the_rows_it_could_not_score():
    """The first weeks of the first season have no calibration; they are counted and named, so
    `gate_n` is never read as the board.

    Mutation: report `n_uncalibrated = 0`, or drop `uncalibrated_cells`."""
    got = coverage.measure(_drawn(n_players=250, weeks=17, seed=2), "prior")
    assert got["n"] == 250 * 13, "weeks 5-17 of one season are the board"
    assert got["n_uncalibrated"] == got["n"] - got["gate_n"] > 0
    cells = got["uncalibrated_cells"]
    assert cells[0] == [2024, 5] and len(cells) == 1 + (got["n_uncalibrated"] // 250 - 1)


def test_a_position_under_the_floor_names_the_rows_that_fell_back():
    """QB has 30 rows a week and a floor of 200, so for most of the window it is calibrated
    on the pool and the row says so, naming the cells. WR is never thin.

    Mutation: set every `fell_back` False in `calibrate`, or drop `fallback_cells`."""
    wr = _drawn(n_players=300, weeks=17, pos="WR", seed=1)
    qb = _drawn(n_players=30, weeks=17, pos="QB", mu=18.0, seed=2).with_columns(
        pl.col("player_id") + "_qb")
    got = coverage.measure(pl.concat([wr, qb]), "prior")
    rows = {r["group"]: r for r in got["by_position"]}
    assert rows["QB"]["pooled_calibration"] and rows["QB"]["n_fallback"] > 0
    assert rows["QB"]["fallback_cells"], "which rows fell back is named, not just counted"
    assert rows["QB"]["n_cal"] < conformal.MIN_GROUP_CALIBRATION, "n_cal beside the group"
    assert not rows["WR"]["pooled_calibration"] and rows["WR"]["fallback_cells"] == []


def test_every_group_row_carries_its_position_n_n_cal_deviation_and_sigma():
    got = coverage.measure(_drawn(n_players=250, weeks=17, seed=2), "prior")
    wr = got["by_position"][0]
    assert wr["position"] == "WR" and wr["group"] == "WR"
    for key in ("n", "n_cal", "cov80", "deviation", "se", "sigma"):
        assert key in wr
    assert wr["deviation"] == pytest.approx(wr["cov80"] - 0.80)
    assert wr["se"] == pytest.approx(math.sqrt(2.0) * math.sqrt(0.16 / wr["n"]))
    assert wr["sigma"] == pytest.approx(wr["deviation"] / wr["se"])
    pool = got["by_position"][-1]
    assert pool["group"] == "all" and pool["position"] is None


def test_the_parametric_interval_is_kept_beside_the_published_one():
    """The prior values stay in the file (rule 13)."""
    got = coverage.measure(_drawn(n_players=250, weeks=17, seed=2), "prior")
    par = got["parametric"]
    assert par["gate_subset"] == "unclipped" and par["by_position"][-1]["group"] == "all"
    assert 0.0 <= par["clipped_share"] < 1.0 and "verdict" in par


def test_a_board_with_nothing_calibrated_is_reported_not_returned_empty():
    with pytest.raises(coverage.NotEnoughWeeks, match="calibration"):
        coverage.measure(_stats(_player("a", "WR", 2024, [10.0] * 17)), "prior")
