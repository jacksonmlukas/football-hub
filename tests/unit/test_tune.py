"""Tuning projection_lambda against a holdout.

`projection_lambda` was 0.08 by judgment. The plan asks for it to be set by evidence.

Before trusting what a sweep says about real data, the harness has to be shown to work on
data where the answer is known. So the load-bearing tests here build a world where the
regression signal genuinely predicts next season, and check the sweep finds a positive
lambda -- and a world where the signal is pure noise, and check it finds zero. A tuner that
cannot recover a planted answer cannot be believed about a real one.
"""
import numpy as np
import polars as pl
import pytest

from hub.draft import tune
from hub.draft.projection import adjusted


def _holdout(n=300, *, signal_strength=0.0, seed=0):
    """A synthetic season pair.

    `signal_strength` is how much last season's underperformance (z) actually predicts
    this season's finish. At 0 the signal is noise and the right lambda is 0.
    """
    rng = np.random.default_rng(seed)
    ecr = np.arange(1, n + 1, dtype=float)
    z = rng.normal(0, 1, n)
    # True skill is ECR order, perturbed by the signal and by noise.
    true_rank = ecr - signal_strength * z * 20 + rng.normal(0, 8, n)
    actual = 300.0 - true_rank          # more points is better
    return pl.DataFrame({"player": [f"P{i}" for i in range(n)],
                         "ecr": ecr, "z_regress": z, "actual_points": actual})


# --- the harness recovers a planted answer -------------------------------

def test_zero_lambda_is_exactly_the_consensus_board():
    """The baseline has to be the untouched board, or every comparison is meaningless."""
    h = _holdout()
    scored = tune.score(h, lam=0.0)
    assert scored["spearman"] == pytest.approx(tune.score(h, lam=0.0)["spearman"])
    ranked = adjusted(h, lam=0.0)
    assert ranked["adj_ecr"].to_list() == h["ecr"].to_list()


def test_a_real_signal_produces_a_positive_best_lambda():
    """Plant a signal that genuinely predicts, and the sweep must find it."""
    h = _holdout(signal_strength=1.0, seed=1)
    best = tune.best_lambda(tune.sweep(h, lams=[0.0, 0.05, 0.1, 0.2, 0.4]))
    assert best > 0.0


def test_pure_noise_produces_a_best_lambda_of_zero():
    """The honest outcome when the signal is worthless -- and the one most likely on real
    data. A tuner that never returns zero is fitting noise."""
    h = _holdout(signal_strength=0.0, seed=2)
    best = tune.best_lambda(tune.sweep(h, lams=[0.0, 0.05, 0.1, 0.2, 0.4]))
    assert best == 0.0


def test_an_inverted_signal_is_not_rewarded():
    """If last season's underperformers do worse, lambda must not go positive."""
    h = _holdout(signal_strength=-1.0, seed=3)
    best = tune.best_lambda(tune.sweep(h, lams=[0.0, 0.05, 0.1, 0.2]))
    assert best == 0.0


# --- the adjustment itself ------------------------------------------------

def test_an_underperformer_moves_up_the_board():
    h = pl.DataFrame({"player": ["a", "b"], "ecr": [50.0, 50.0],
                      "z_regress": [2.0, -2.0], "actual_points": [0.0, 0.0]})
    out = adjusted(h, lam=0.1)
    up = out.filter(pl.col("player") == "a")["adj_ecr"][0]
    down = out.filter(pl.col("player") == "b")["adj_ecr"][0]
    assert up < 50.0 < down, "positive z means underperformed, which is a buy"


def test_the_nudge_is_larger_deep_in_the_board():
    """Multiplicative on purpose: one sd of evidence should move a player further at pick
    150 than at pick 5, because consensus is tight at the top and loose at the bottom."""
    h = pl.DataFrame({"player": ["top", "deep"], "ecr": [5.0, 150.0],
                      "z_regress": [1.0, 1.0], "actual_points": [0.0, 0.0]})
    out = adjusted(h, lam=0.1)
    moved = (out["ecr"] - out["adj_ecr"]).to_list()
    assert moved[1] > moved[0] * 5


def test_players_without_a_signal_keep_their_rank():
    """Rookies have no prior season. Inventing a nudge for them is worse than none."""
    h = pl.DataFrame({"player": ["rook"], "ecr": [40.0], "z_regress": [None],
                      "actual_points": [0.0]})
    assert adjusted(h, lam=0.2)["adj_ecr"][0] == 40.0


# --- the sweep ------------------------------------------------------------

def test_sweep_returns_one_row_per_lambda():
    got = tune.sweep(_holdout(), lams=[0.0, 0.1, 0.2])
    assert got.height == 3
    assert got["lam"].to_list() == [0.0, 0.1, 0.2]


def test_sweep_reports_the_metrics_a_decision_needs():
    got = tune.sweep(_holdout(), lams=[0.0, 0.1])
    assert {"lam", "spearman", "top50_points", "delta_vs_consensus"} <= set(got.columns)


def test_delta_is_measured_against_the_untouched_board():
    got = tune.sweep(_holdout(), lams=[0.0, 0.1])
    at_zero = got.filter(pl.col("lam") == 0.0)["delta_vs_consensus"][0]
    assert at_zero == pytest.approx(0.0), "lambda 0 must be exactly the baseline"


def test_sweep_is_deterministic():
    h = _holdout(signal_strength=0.5, seed=7)
    assert (tune.sweep(h, lams=[0.1])["spearman"][0]
            == tune.sweep(h, lams=[0.1])["spearman"][0])


def test_an_empty_holdout_does_not_crash():
    empty = pl.DataFrame(schema={"player": pl.Utf8, "ecr": pl.Float64,
                                 "z_regress": pl.Float64, "actual_points": pl.Float64})
    assert tune.sweep(empty, lams=[0.0]).height == 1


# --- selecting on rank correlation ----------------------------------------

def test_spearman_selection_finds_a_planted_signal():
    """Whole-board Spearman reads every player, so it should recover a signal the
    top-50 metric can only see through two or three boundary swaps."""
    h = _holdout(signal_strength=1.0, seed=11)
    assert tune.best_lambda_spearman(tune.sweep(h, lams=[0.0, 0.05, 0.1, 0.2])) > 0.0


def test_spearman_selection_returns_zero_on_noise():
    h = _holdout(signal_strength=0.0, seed=12)
    assert tune.best_lambda_spearman(tune.sweep(h, lams=[0.0, 0.05, 0.1, 0.2])) == 0.0


def test_spearman_selection_is_not_fooled_by_a_single_boundary_swap():
    """The 2023-24 failure mode, in miniature. One player at the top-50 edge scoring
    enormously must not move a whole-board criterion."""
    h = _holdout(signal_strength=0.0, seed=13)
    spiked = h.with_columns(
        pl.when(pl.col("player") == "P51").then(5000.0)
          .otherwise(pl.col("actual_points")).alias("actual_points"))
    assert tune.best_lambda_spearman(tune.sweep(spiked, lams=[0.0, 0.05, 0.1])) == 0.0
