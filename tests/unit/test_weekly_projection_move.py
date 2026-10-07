"""#430: the Weekly projection moved to `hub.exhibits.weekly_projection` and its parametric
interval was renamed; the projection did not change.

`docs/weekly-forward.md` pins the arm under test by git blob and refuses a reading on any other
blob. A move and a rename change the blob and nothing about what is computed, and the re-pin
is legitimate only if that is *shown*, so this file shows it: `FROZEN` was computed on the
pre-move code (`hub.models.weekly` at blob f6de17b2ca26a6dbaa2e606f567f8227456bc015) from the
fixture below, before any file moved, and the moved module has to reproduce every figure. The
fixture has four positions, games played from 1 to 15 so the shrinkage and the efficiency
threshold both bite, and a training frame the shrinkage is fitted on, so it reaches `project`
with and without a `Shrink`, `fit_shrink` under both objectives, `positional_sd` and
`standard_error` -- everything the forward measurement calls -- and the renamed quantiles.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from hub.models import weekly as W

# name -> (sum, position-weighted sum). The weighted sum is what a reordering would move.
FROZEN: dict[str, tuple[float, float]] = {
    'project.mu': (1303.3730855848667, 76730.9567485606),
    'project.targets_hat': (580.8723306939436, 34297.809405848544),
    'project.receptions_hat': (307.08520498234134, 18129.37594719237),
    'project.carries_hat': (652.7048189195596, 42168.31299917896),
    'project.attempts_hat': (983.4306406775578, 58651.381336710314),
    'project.rec_yards_hat': (3110.8772029536813, 191089.44200656295),
    'project.rush_yards_hat': (1632.6103636716487, 93072.36562422948),
    'project.pass_yards_hat': (6356.262216857482, 366346.6815440145),
    'project.tds_hat': (71.03991373379222, 4164.418026361786),
    'project.turnovers_hat': (40.18441093485661, 2474.455599325468),
    'fit_shrink.mae.ks': (1.0, 64.0),
    'fit_shrink.mae.volume_mean': (80.03333333333333, 478.8833333333333),
    'fit_shrink.mae.eff_mean': (75.16120562822674, 449.33860855410825),
    'project.shrunk.mae.mu': (1468.4908087778394, 87454.81844156396),
    'fit_shrink.tail.ks': (0.0, 128.0),
    'fit_shrink.tail.volume_mean': (80.03333333333333, 478.8833333333333),
    'fit_shrink.tail.eff_mean': (75.16120562822674, 449.33860855410825),
    'project.shrunk.tail.mu': (1466.3835596701147, 87559.5549178031),
    'project.fixed.mu': (1515.5956874949666, 89121.72189561365),
    'positional_sd': (29.02807120032057, 88.2806276903199),
    'standard_error': (298.60514055344197, 17513.85842297929),
    'quantiles': (524865.664901114, 12298522217.791828),
}


def _frame(n: int = 240, seed: int = 7) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    pos = (["WR", "RB", "QB", "TE"] * n)[:n]
    games = (np.arange(n) % 15 + 1).astype(float)
    qb = np.array([p == "QB" for p in pos])
    return pl.DataFrame({
        "season": [2024] * n, "week": [10] * n, "position": pos,
        "player_id": [f"p{i % 4}-{(i // 4) % 10}" for i in range(n)],
        "games_before": games,
        "targets_prior": np.where(qb, 0.0, rng.gamma(3.0, 2.0, n)),
        "receptions_prior": np.where(qb, 0.0, rng.gamma(3.0, 1.3, n)),
        "carries_prior": np.where(qb, rng.gamma(2.0, 1.0, n), rng.gamma(2.0, 3.0, n)),
        "attempts_prior": np.where(qb, rng.gamma(8.0, 4.0, n), 0.0),
        "receiving_yards_prior": rng.gamma(4.0, 10.0, n),
        "rushing_yards_prior": rng.gamma(2.0, 8.0, n),
        "passing_yards_prior": np.where(qb, rng.gamma(8.0, 30.0, n), 0.0),
        "passing_interceptions_prior": np.where(qb, rng.gamma(2.0, 0.4, n), 0.0),
        "fumbles_lost_total_prior": rng.gamma(1.0, 0.1, n),
        "fantasy_points_ppr": rng.gamma(3.0, 4.0, n),
        "ppg_before": rng.gamma(3.0, 4.0, n),
        "snap_trend": rng.normal(0, 0.1, n),
        "targets": rng.poisson(5, n).astype(float),
        "carries": rng.poisson(4, n).astype(float),
        "attempts": np.where(qb, rng.poisson(30, n), 0).astype(float),
        "receptions": rng.poisson(3, n).astype(float),
        "receiving_yards": rng.gamma(4.0, 10.0, n),
        "rushing_yards": rng.gamma(2.0, 8.0, n),
        "passing_yards": np.where(qb, rng.gamma(8.0, 30.0, n), 0.0),
    })


def _digest(a) -> tuple[float, float]:
    v = np.asarray(a, dtype=float).ravel()
    return float(v.sum()), float(v @ np.arange(1, v.size + 1))


def _computed() -> dict[str, tuple[float, float]]:
    train, now = _frame(240, 7), _frame(120, 11)
    out: dict[str, tuple[float, float]] = {}
    base = W.project(now)
    for col in ("mu", "targets_hat", "receptions_hat", "carries_hat", "attempts_hat",
                "rec_yards_hat", "rush_yards_hat", "pass_yards_hat", "tds_hat",
                "turnovers_hat"):
        out[f"project.{col}"] = _digest(base[col].to_numpy())
    for objective in ("mae", "tail"):
        sh = W.fit_shrink(train, objective=objective)
        out[f"fit_shrink.{objective}.ks"] = (sh.volume_k, sh.eff_k)
        out[f"fit_shrink.{objective}.volume_mean"] = _digest(
            [sh.volume_mean[k] for k in sorted(sh.volume_mean)])
        out[f"fit_shrink.{objective}.eff_mean"] = _digest(
            [sh.eff_mean[k] for k in sorted(sh.eff_mean)])
        out[f"project.shrunk.{objective}.mu"] = _digest(W.project(now, shrink=sh)["mu"].to_numpy())
    # a fixed non-trivial Shrink as well, so the fitted constants being (0, 0) cannot hide a
    # shrinkage path that moved
    vm, em = W._pos_means(train)
    fixed = W.Shrink(4.0, 8.0, vm, em)
    out["project.fixed.mu"] = _digest(W.project(now, shrink=fixed)["mu"].to_numpy())
    sigma = W.positional_sd(train)
    out["positional_sd"] = _digest([sigma[k] for k in sorted(sigma)])
    out["standard_error"] = _digest(W.standard_error(now, sigma))
    mu = base["mu"].to_numpy()
    out["quantiles"] = _digest(W.shipped_quantiles(mu, now["position"].to_list()))
    return out


def test_the_moved_projection_computes_exactly_what_the_pinned_arm_computed():
    got = _computed()
    assert set(got) == set(FROZEN), f"the digest changed shape: {set(got) ^ set(FROZEN)}"
    for name, want in FROZEN.items():
        assert got[name] == pytest.approx(want, rel=1e-12, abs=1e-12), \
            f"{name}: the moved projection no longer computes what the pinned arm computed"
