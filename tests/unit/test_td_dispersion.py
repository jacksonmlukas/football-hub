"""The TD_DISPERSION refit gate (#472, docs/td-dispersion.md).

Rule 18: every check is run against the condition it exists to detect. The estimator recovers a
planted dispersion (and a planted estimator that forgets the degrees of freedom does not); the
log pmf is the law the sampler draws (and a pmf in the wrong family is not); the gate adopts a
planted improvement and does not adopt a null; the walk-forward reads nothing from the season it
scores. All offline, all against planted data.
"""
import numpy as np
import polars as pl
import pytest

from hub.ledger import Ledger
from hub.models import components as C
from hub.models import td_dispersion as T
from hub.models.experiment import Actions, Harness

PER_PHASE = {"rec": 40, "rush": 20, "pass": 10}


def _table(counts: dict[str, np.ndarray], *, phase: str = "rec", season: int = 2022,
           position: str = "WR") -> pl.DataFrame:
    """A `phase_table` from {player: weekly counts}."""
    rows = []
    for pid, x in counts.items():
        for w, c in enumerate(x, start=1):
            rows.append({"phase": phase, "season": season, "player_id": pid, "week": w,
                         "position": position, "x": float(c), "n": float(len(x)),
                         "s": float(np.sum(x))})
    return pl.DataFrame(rows)


def _sum(frame: pl.DataFrame, col: str) -> float:
    return float(frame[col].sum())


def _mean(frame: pl.DataFrame, col: str) -> float:
    return float(np.mean(frame[col].to_numpy()))


def _planted(draw, n_players: int, n_games: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return _table({f"p{i}": draw(rng, i, n_games) for i in range(n_players)})


# --- the estimator recovers a planted dispersion ---------------------------------------

def test_a_poisson_count_has_variance_over_mean_one():
    t = _planted(lambda rng, i, n: rng.poisson(0.6, n), 4000, 12)
    assert T.fit_dispersion(t)["rec"] == pytest.approx(1.0, abs=0.03)


def test_a_binomially_thinned_count_has_variance_over_mean_point_eight():
    """Binomial(5, 0.2): variance over mean is 1 - p = 0.8, drawn by a mechanism that is not
    the sampler's own, so the estimator is not being tested against the thing it estimates."""
    t = _planted(lambda rng, i, n: rng.binomial(5, 0.2, n), 4000, 12)
    assert T.fit_dispersion(t)["rec"] == pytest.approx(0.8, abs=0.03)


def test_a_planted_estimator_without_the_degrees_of_freedom_misses_a_short_season():
    """The control that lets the test above tell: dividing by `n` instead of `n - 1` is low by
    `(n - 1) / n`, which at eight games is 12%."""
    t = _planted(lambda rng, i, n: rng.poisson(0.6, n), 4000, 8)
    ps = T._player_seasons(t)
    wrong = (_sum(ps, "ss") / _sum(ps, "n")) / (_sum(ps, "s") / _sum(ps, "n"))
    assert abs(wrong - 1.0) > 0.08
    assert T.fit_dispersion(t)["rec"] == pytest.approx(1.0, abs=0.03)


def test_the_estimator_is_the_pooled_ratio_not_a_mean_of_ratios():
    t = _table({"a": np.array([0, 0, 0, 0, 0, 0, 0, 4]), "b": np.array([1, 0, 1, 0, 1, 0, 1, 0])})
    ps = T._player_seasons(t)
    want = (_sum(ps, "ss") / _sum(ps, "dof")) / (_sum(ps, "s") / _sum(ps, "n"))
    assert T.fit_dispersion(t)["rec"] == pytest.approx(want)


def test_a_phase_with_no_touchdown_is_absent_not_nan():
    t = _table({"a": np.zeros(10)})
    assert T.fit_dispersion(t) == {}


def test_by_position_is_keyed_by_phase_and_position():
    t = pl.concat([_table({"a": np.array([1, 0, 2, 0, 1, 0, 0, 1])}, position="WR"),
                   _table({"b": np.array([0, 0, 1, 0, 0, 0, 0, 1])}, position="TE")])
    assert set(T.fit_dispersion(t, by_position=True)) == {("rec", "WR"), ("rec", "TE")}


def test_the_interval_brackets_the_estimate():
    t = _planted(lambda rng, i, n: rng.binomial(5, 0.2, n), 600, 12)
    lo, hi = T.fit_interval(t, "rec", resamples=300)
    assert lo < T.fit_dispersion(t)["rec"] < hi
    assert hi - lo < 0.2


def test_the_training_rate_is_touchdowns_per_game():
    t = _table({"a": np.array([1, 0, 0, 0]), "b": np.array([2, 1, 0, 1])})
    assert T.training_rate(t) == {"rec": pytest.approx(5 / 8)}


# --- the qualifying table ----------------------------------------------------------------

def _stats(**over) -> pl.DataFrame:
    rows = []
    for pid, pos, n, rec, car, att, tds in (
            ("wr", "WR", 10, 5, 0, 0, 1), ("few", "WR", 7, 5, 0, 0, 1),
            ("bench", "WR", 12, 1, 0, 0, 1), ("rb", "RB", 12, 3, 12, 0, 1),
            ("qb", "QB", 12, 0, 3, 30, 2), ("k", "K", 12, 5, 0, 0, 1)):
        for w in range(1, n + 1):
            rows.append({"fantasy_points_ppr": 1.0, "season": 2024, "week": w, "player_id": pid,
                         "position": pos, "season_type": over.get("st", "REG"),
                         "attempts": att, "carries": car, "receptions": rec,
                         "passing_tds": tds if pos == "QB" else 0,
                         "rushing_tds": tds if pos == "RB" else 0,
                         "receiving_tds": tds if pos in ("WR", "K") else 0})
    return pl.DataFrame(rows)


def test_the_table_keeps_a_player_season_at_its_games_and_volume():
    t = T.phase_table(_stats())
    got = {(p, pid) for p, pid in t.select("phase", "player_id").unique().rows()}
    assert got == {("rec", "wr"), ("rec", "rb"), ("rush", "rb"), ("pass", "qb")}
    # below 8 games, below the volume floor, and a kicker, are all out; a QB's 3 carries are
    # under the rushing floor and an RB's 3 receptions a game are over the receiving one
    assert t.filter(pl.col("player_id") == "wr")["n"].unique().to_list() == [10.0]


def test_the_table_is_the_regular_season_only():
    assert T.phase_table(_stats(st="POST")).is_empty()


def test_the_table_names_what_it_needs():
    with pytest.raises(ValueError, match="phase_table needs"):
        T.phase_table(_stats().drop("receptions"))


# --- the log pmf is the law the sampler draws -------------------------------------------

@pytest.mark.parametrize("phi", [0.83, 1.0, 1.3])
def test_the_log_pmf_matches_the_draws_of_the_sampler(phi):
    d = C.draw_counts(np.random.default_rng(0), 0.7, phi, 600_000)
    emp = np.bincount(d, minlength=8)[:8] / d.size
    got = np.exp(T.count_logpmf(np.arange(8), np.full(8, 0.7), phi))
    assert np.abs(emp - got).max() < 2e-3


def test_the_log_pmf_below_one_is_the_conway_maxwell_poisson_table():
    pmf = C.cmp_pmf(0.7, C.cmp_nu(0.7, 0.83))
    got = np.exp(T.count_logpmf(np.arange(5), np.full(5, 0.7), 0.83))
    assert np.allclose(got, pmf[:5])


def test_a_pmf_in_the_wrong_family_does_not_match_the_draws():
    """The control: Poisson probabilities against under-dispersed draws differ by more than the
    tolerance the test above holds the right family to."""
    d = C.draw_counts(np.random.default_rng(0), 0.7, 0.83, 600_000)
    emp = np.bincount(d, minlength=8)[:8] / d.size
    wrong = np.exp(T.count_logpmf(np.arange(8), np.full(8, 0.7), 1.0))
    assert np.abs(emp - wrong).max() > 4e-3


def test_a_count_the_law_calls_impossible_is_scored_at_the_floor():
    """A two-touchdown week at a mean of 0.05, where `phi = 0.83` is below what any count of
    that mean can reach: the narrowest law gives it ~0, and the score is the floor."""
    raw = T.count_logpmf(np.array([2]), np.array([0.05]), 0.83)
    floored = T.count_logpmf(np.array([2]), np.array([0.05]), 0.83, T.SCORE_FLOOR)
    assert raw[0] < -30.0
    assert floored[0] == pytest.approx(np.log(T.SCORE_FLOOR))


def test_the_log_pmf_over_the_support_sums_to_one():
    for phi in (0.83, 1.0, 1.4):
        assert np.exp(T.count_logpmf(np.arange(40), np.full(40, 1.3), phi)).sum() == \
            pytest.approx(1.0, abs=1e-6)


# --- no look-ahead -------------------------------------------------------------------------

def _seasons(truth, seasons=(2019, 2020, 2021, 2022), seed=0) -> pl.DataFrame:
    return T.planted_table(truth, seasons, PER_PHASE, seed=seed)


TRUTH = {"rec": 0.96, "rush": 0.96, "pass": 0.84}


def test_a_later_season_changes_no_earlier_held_out_row():
    """The walk-forward scores season `s` from seasons before `s`: rewriting the last season's
    counts leaves every earlier held-out row exactly as it was."""
    t = _seasons(TRUTH)
    base = T.walk_forward(t, ceiling=TRUTH)
    last = t["season"].max()
    flipped = t.with_columns(pl.when(pl.col("season") == last).then(pl.col("x") + 1.0)
                             .otherwise(pl.col("x")).alias("x"),
                             pl.when(pl.col("season") == last).then(pl.col("s") + pl.col("n"))
                             .otherwise(pl.col("s")).alias("s"))
    after = T.walk_forward(flipped, ceiling=TRUTH)
    keep = base.filter(pl.col("season") < last)
    assert keep.height and keep.equals(after.filter(pl.col("season") < last))
    assert not base.filter(pl.col("season") == last).equals(after.filter(pl.col("season") == last))


def test_the_fit_reads_only_the_seasons_it_is_given():
    t = _seasons(TRUTH, seasons=(2019, 2020))
    only = T.fit_dispersion(t, [2019])
    assert only != T.fit_dispersion(t, [2020])
    assert only == T.fit_dispersion(t.filter(pl.col("season") == 2019))


def test_the_week_scored_does_not_enter_its_own_mean():
    """One player, all of whose touchdowns came in the scored week: his mean for that week is
    the pseudo-game's alone, `r / n`, not `s / n`."""
    past = _table({f"p{i}": np.array([i % 2, 0, 1, 0, 0, 1, 0, 0, 1]) for i in range(30)},
                  season=2021)
    now = _table({"star": np.array([0, 0, 0, 0, 0, 0, 0, 5])}, season=2022)
    t = pl.concat([past, now])
    rate = T.training_rate(past)["rec"]
    got = T.walk_forward(t, ceiling={"rec": 0.9, "rush": 0.9, "pass": 0.9})
    row = got.filter((pl.col("player_id") == "star") & (pl.col("week") == 8))
    m = np.array([(5 - 5 + T.PRIOR_GAMES * rate) / 8])
    shipped = C.TD_DISPERSION["rec"]
    fit = T.fit_dispersion(past)["rec"]
    want = (T.count_logpmf(np.array([5]), m, fit, T.SCORE_FLOOR)
            - T.count_logpmf(np.array([5]), m, shipped, T.SCORE_FLOOR))[0]
    assert row["diff"][0] == pytest.approx(float(want))


def test_a_planted_in_sample_mean_would_have_scored_differently():
    """The control for the test above: the same week at its in-sample mean, `s / n`, is a
    different number, so the test could tell."""
    in_sample = T.count_logpmf(np.array([2]), np.array([2 / 8]), 0.83, T.SCORE_FLOOR)
    leave_one_out = T.count_logpmf(np.array([2]), np.array([0.04]), 0.83, T.SCORE_FLOOR)
    assert abs(in_sample[0] - leave_one_out[0]) > 1.0


def test_an_empty_table_has_no_held_out_season():
    out = T.walk_forward(_seasons(TRUTH, seasons=(2019,)), ceiling=TRUTH)
    assert out.is_empty() and out.columns == ["season", "player_id", "week", "diff",
                                              "ceiling_diff"]


# --- the gate ---------------------------------------------------------------------------

BAD = {"rec": 0.5, "rush": 0.5, "pass": 0.5}


def test_the_gate_adopts_a_planted_real_improvement():
    """Truth at the measured figures, an incumbent planted far from it: the refit wins every
    season by more than the noise."""
    t = _seasons(TRUTH, seasons=(2019, 2020, 2021, 2022, 2023), seed=3)
    paired = T.walk_forward(t, incumbent=BAD, ceiling=TRUTH)
    run = T.HARNESS.decide(paired, seed=1, bootstrap=200)
    assert run.verdict[0] == "ADOPT", run.verdict
    assert run.summary["mean"] > 0 and run.summary["t_lo"] > 0


def test_the_gate_does_not_adopt_a_null():
    """Truth is the incumbent's own constants: there is nothing to find, and the refit, which
    has to estimate them, cannot be better than them."""
    truth = dict(C.TD_DISPERSION)
    t = _seasons(truth, seasons=(2019, 2020, 2021, 2022, 2023), seed=4)
    paired = T.walk_forward(t, ceiling=truth)
    run = T.HARNESS.decide(paired, seed=1, bootstrap=200)
    assert run.verdict[0] != "ADOPT", run.verdict


def test_the_gate_removes_a_challenger_that_loses_in_every_season():
    """The branch where the elaborate thing loses (rule 1): a paired frame in which the refit
    is worse in every season, by more than its noise, with a ceiling above the design's MDE."""
    rng = np.random.default_rng(0)
    rows = [{"season": s, "player_id": f"p{i}", "week": 1, "diff": -0.01 + rng.normal(0, 0.001),
             "ceiling_diff": 0.05} for s in range(2020, 2026) for i in range(40)]
    run = T.HARNESS.decide(pl.DataFrame(rows), seed=0, bootstrap=200)
    assert run.verdict[0] == "REMOVE", run.verdict


def test_the_ceiling_is_the_dispersion_fitted_to_the_held_out_season():
    """Left to search, the ceiling arm finds the planted truth's neighbourhood, and it is at
    least what any constant earns on those rows."""
    t = _seasons(TRUTH, seasons=(2019, 2020, 2021), seed=5)
    searched = T.walk_forward(t)
    planted = T.walk_forward(t, ceiling=TRUTH)
    assert _mean(searched, "ceiling_diff") >= _mean(planted, "ceiling_diff") - 1e-3
    assert _mean(searched, "ceiling_diff") >= _mean(searched, "diff") - 1e-3


def test_the_harness_is_season_clustered_and_names_its_own_module():
    assert T.HARNESS.arm_roots == ("hub.models.td_dispersion",)
    assert T.HARNESS.within == ("player_id",)
    assert isinstance(T.HARNESS, Harness) and isinstance(T.ACTIONS, Actions)


def test_a_published_run_writes_one_ledger_entry():
    t = _seasons(TRUTH, seasons=(2019, 2020, 2021), seed=6)
    paired = T.walk_forward(t, ceiling=TRUTH)
    ledger = Ledger(path=None)
    T.gate_run(paired, publish=True, seasons=(2019, 2020, 2021), ledger=ledger)
    entries = ledger._read() or []
    assert len(entries) == 1 and entries[0].name == "td_dispersion"


# --- the simulation that sizes the gate ----------------------------------------------------

def test_the_simulation_adopts_a_real_improvement_and_not_a_null():
    per_phase = {"rec": 30, "rush": 15, "pass": 8}
    seasons = (2019, 2020, 2021, 2022, 2023)
    null = T.simulate(dict(C.TD_DISPERSION), per_phase, seasons, 3, seed=10)
    wide = T.simulate({"rec": 1.4, "rush": 1.4, "pass": 1.4}, per_phase, seasons, 3, seed=10)
    assert null["adopt"] <= 1 / 3
    assert wide["adopt"] >= 2 / 3
    assert wide["mean"] > null["mean"]


def test_the_simulation_can_be_mapped_by_a_caller():
    seen = []

    def fake(fn, jobs):
        seen.append(len(jobs))
        return [{"adopt": 1.0, "not_runnable": 0.0, "remove": 0.0, "mean": 0.1, "mde": 0.2,
                 "ceiling": 0.3, "se": 0.01} for _ in jobs]

    out = T.simulate(TRUTH, PER_PHASE, (2019, 2020, 2021), 4, map_fn=fake)
    assert seen == [4] and out["adopt"] == 1.0 and out["mde"] == pytest.approx(0.2)


# --- the descriptive tail table and the CLI ---------------------------------------------

def _full_stats(seed=0, seasons=(2022, 2023, 2024)) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(30):
            pos = "WR" if i % 3 else "RB"
            for w in range(1, 15):
                rows.append({"fantasy_points_ppr": float(rng.gamma(4.0, 3.0)), "season": season,
                             "week": w, "player_id": f"{pos}{i}", "position": pos,
                             "season_type": "REG", "attempts": 0, "carries": 6 if pos == "RB" else 0,
                             "receptions": 4, "passing_tds": 0,
                             "rushing_tds": int(rng.poisson(0.3)) if pos == "RB" else 0,
                             "receiving_tds": int(rng.poisson(0.3)),
                             "passing_yards": 0.0, "passing_interceptions": 0,
                             "rushing_yards": 25.0 if pos == "RB" else 0.0,
                             "receiving_yards": 45.0, "rushing_fumbles_lost": 0.0,
                             "receiving_fumbles_lost": 0.0, "sack_fumbles_lost": 0.0})
    return pl.DataFrame(rows)


def test_the_tail_table_counts_weeks_at_each_line_and_restores_the_constants():
    before = dict(C.TD_DISPERSION)
    r = T.tail_calibration(_full_stats(), (2022, 2023, 2024),
                           {"shipped": before, "wide": {"rec": 1.2, "rush": 1.2, "pass": 1.2}},
                           draws=300)
    assert r["player_seasons"] == 90 and r["player_weeks"] == 90 * 14
    assert r["observed"][20] >= r["observed"][40]
    assert r["shipped"][20] >= r["shipped"][40]
    assert dict(C.TD_DISPERSION) == before


def test_the_cli_prints_help_without_a_network():
    assert T.main([]) == 0


def test_the_fit_path_runs_offline_and_writes_one_entry(capsys):
    stats = _full_stats(seasons=(2021, 2022, 2023, 2024))
    ledger = Ledger(path=None)
    assert T.main(["--fit", "--seasons", "2021,2022,2023,2024"],
                  load=lambda *a, **k: stats, ledger=ledger) == 0
    text = capsys.readouterr().out
    assert "The estimator on every season loaded" in text and "By position" in text
    assert len(ledger._read() or []) == 1


def test_the_tail_path_runs_offline(capsys):
    stats = _full_stats()
    assert T.main(["--tails", "--seasons", "2022,2023,2024"],
                  load=lambda *a, **k: stats) == 0
    assert ">= 30: observed" in capsys.readouterr().out


def test_an_unreadable_source_is_reported_not_raised(capsys):
    def boom(*a, **k):
        raise OSError("no network")

    assert T.main(["--fit"], load=boom) != 0
