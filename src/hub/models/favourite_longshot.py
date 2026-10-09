"""The favourite-longshot correction, as a gate (#465, from #367).

`normal_cdf(spread / MARGIN_SD)` prices every game in this repo, and `docs/margin-sd.md`
measured where it misses: favourites of 7+ are priced at 0.774 and win 0.806, pick'ems trail
the number, the middle of the range beats it. That was a screen and a bucket cut after
looking. ADR-0014 lets a provisional rule act only where no gate can run, and "no gate can
run" has to be shown by ADR-0019's MDE against the ceiling, not asserted. So this module runs
the gate, as a **measurement**: it wires no consumer and changes no price. It only *reads*
`hub.models.margin` and `hub.models.market`, which the weekly arm pins (#456), and edits
neither.

**The pre-registration is `docs/margin-sd.md`, "Pre-registered 2026-10-09 ... (#465)", and
was committed before any number here existed.** What follows is that text in code.

* **Incumbent**: `Phi(s / MARGIN_SD)`, the live constant, held fixed in every season.
* **Shifted**: `Phi((s + sign(s) * delta_b(|s|)) / sd)`, a favourite-relative location shift in
  five buckets of `|s|` (`EDGES`), fitted **jointly** with the sd by ridge-penalised maximum
  likelihood (prior sd `SHIFT_PRIOR_SD` points), on the **trailing ten seasons strictly before**
  the scored one (rule 2; `experiment.expanding_seasons` is the one statement of it). Joint,
  because bucket residuals against one fixed sd cannot separate location from scale (#367).
* **Unit**: the season, one row per held-out season; **metric**: walk-forward log loss on
  `home_won`, the paired difference `LL(incumbent) - LL(shifted)`, positive when the
  correction is better.
* **Ceiling arm**: a perfect `P(win | spread)` by one-point bucket of `|s|`, fitted on all
  held-out games pooled and scored on each season. Pooled, not refitted per season, because a
  per-season oracle on ~270 games inflates itself by roughly `buckets / (2 n)` nats and would
  make NOT-RUNNABLE unreachable by construction.
* **Held-out seasons** are completed seasons (`MIN_SCORED_GAMES`) with at least
  `MIN_PAST_SEASONS` earlier seasons of history.

The fit is Fisher scoring on the probit with an analytic information matrix, so it needs no
optimiser dependency and is deterministic.

    uv run python -m hub.models.favourite_longshot --run
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from math import erf, sqrt
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.declare import not_an_input
from hub.models.experiment import Actions, Harness, expanding_seasons

# Imported rather than restated, so the incumbent and the live price cannot drift apart.
from hub.models.margin import TRAILING, residuals
from hub.models.market import MARGIN_SD
from hub.models.scoring_rules import log_loss

# Bucket edges of |spread| in points, closed below and open above; the last bucket is open at
# the top. The ones `margin.calibration_by_spread` already publishes: the only edges tried.
EDGES: tuple[int, ...] = (0, 3, 6, 9, 14)
N_BUCKETS = len(EDGES)

# A ridge toward zero on each bucket's shift, prior sd in points. Fixed in the pre-registration,
# about the size of the largest published bucket residual; it only ever hurts the arm under test.
SHIFT_PRIOR_SD = not_an_input(
    1.5,
    "the pre-registered prior sd of a bucket's location shift, in points; it shapes a "
    "gate's challenger arm and no price anything ships reads")

MIN_PAST_SEASONS = 5
MIN_SCORED_GAMES = 240

_MAX_ITER = 60
_TOL = not_an_input(
    1e-9, "the step size below which the fit stops iterating; it ends a search and sets no price")
_EPS = not_an_input(
    1e-9, "the clip that keeps the fit's logarithms finite; it bounds a search and sets no price")
_SQRT2 = not_an_input(
    sqrt(2.0), "the mathematical constant that turns erf into the normal CDF; not a choice")
_SQRT2PI = not_an_input(
    sqrt(2.0 * np.pi), "the normal density's normalising constant; mathematics and not a choice")

CEILING_ARM = "a perfect P(win | spread), pooled over every held-out game"

ACTIONS = Actions(
    adopt="ADOPT: the location correction becomes eligible for a consumer -- the survivor and "
          "pool price in grid_from_schedule only, held behind #379 and #204; nothing is wired.",
    remove="REMOVE: the correction prices games worse; disclose it, no consumer.",
    show="SHOW: keep as a measurement, no consumer.",
)

HARNESS = Harness(
    name="favourite_longshot", arm_a="location-shifted price", arm_b="normal_cdf(s / MARGIN_SD)",
    within=("season",), ceiling_arm=CEILING_ARM, actions=ACTIONS,
    unit="nats of log loss per game", places=5,
    arm_roots=("hub.models.favourite_longshot",))


class Fit(NamedTuple):
    """A fitted correction: one favourite-relative shift per bucket, in points, and the sd."""

    delta: tuple[float, ...]
    sd: float


def _phi_cdf(z: np.ndarray) -> np.ndarray:
    """The standard normal CDF, exact (`math.erf`), as `margin.home_win_prob` computes it."""
    return 0.5 * (1.0 + np.array([erf(v / _SQRT2) for v in z]))


def bucket(spread: np.ndarray) -> np.ndarray:
    """Index of each spread's |spread| bucket, 0 .. N_BUCKETS-1."""
    return np.searchsorted(np.asarray(EDGES[1:], dtype=float), np.abs(spread), side="right")


def _nll(p: np.ndarray, won: np.ndarray, delta: np.ndarray) -> float:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.sum(won * np.log(q) + (1.0 - won) * np.log(1.0 - q))
                 + np.sum(delta ** 2) / (2.0 * SHIFT_PRIOR_SD ** 2))


def fit_shift(spread: np.ndarray, won: np.ndarray, *, shifts: bool = True) -> Fit:
    """Penalised maximum likelihood of `(delta_1..delta_5, sd)` on `home_won`.

    `shifts=False` pins every shift at zero and fits the sd alone: the descriptive "scale only"
    arm, which says how much of any gain is the sd and how much is location. Fisher scoring
    with backtracking, so the objective never rises. A bucket with no games keeps its prior
    (a zero shift).
    """
    s = np.asarray(spread, dtype=float)
    y = np.asarray(won, dtype=float)
    sg = np.sign(s)
    b = bucket(s)
    delta = np.zeros(N_BUCKETS)
    ls = float(np.log(MARGIN_SD))
    ridge = 1.0 / SHIFT_PRIOR_SD ** 2
    k = N_BUCKETS + 1
    active = np.ones(k, dtype=bool)
    if not shifts:
        active[:N_BUCKETS] = False

    def objective(d: np.ndarray, lsd: float) -> float:
        z = (s + sg * d[b]) / np.exp(lsd)
        return _nll(_phi_cdf(z), y, d)

    cur = objective(delta, ls)
    for _ in range(_MAX_ITER):
        sd = np.exp(ls)
        z = (s + sg * delta[b]) / sd
        p = np.clip(_phi_cdf(z), _EPS, 1.0 - _EPS)
        dens = np.exp(-0.5 * z * z) / _SQRT2PI
        var = p * (1.0 - p)
        score = (y - p) * dens / var            # d log-lik / dz
        w = dens * dens / var                   # expected information in z
        jb = sg / sd                            # dz / d delta_b (zero at s = 0)
        jl = -z                                 # dz / d log sd
        grad = np.zeros(k)
        info = np.zeros((k, k))
        grad[:N_BUCKETS] = -np.bincount(b, weights=score * jb, minlength=N_BUCKETS) \
            + ridge * delta
        grad[N_BUCKETS] = -float(np.sum(score * jl))
        diag = np.bincount(b, weights=w * jb * jb, minlength=N_BUCKETS) + ridge
        cross = np.bincount(b, weights=w * jb * jl, minlength=N_BUCKETS)
        info[np.arange(N_BUCKETS), np.arange(N_BUCKETS)] = diag
        info[np.arange(N_BUCKETS), N_BUCKETS] = cross
        info[N_BUCKETS, np.arange(N_BUCKETS)] = cross
        info[N_BUCKETS, N_BUCKETS] = float(np.sum(w * jl * jl))
        idx = np.flatnonzero(active)
        step = np.zeros(k)
        step[idx] = np.linalg.solve(info[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx)),
                                    grad[idx])
        scale = 1.0
        while scale > 1e-6:
            nd = delta - scale * step[:N_BUCKETS]
            nl = ls - scale * step[N_BUCKETS]
            new = objective(nd, nl)
            if new <= cur + 1e-12:
                break
            scale *= 0.5
        else:
            break
        moved = float(np.max(np.abs(scale * step)))
        delta, ls, cur = nd, nl, new
        if moved < _TOL:
            break
    return Fit(tuple(float(v) for v in delta), float(np.exp(ls)))


def incumbent_prob(spread: np.ndarray) -> np.ndarray:
    """`normal_cdf(s / MARGIN_SD)`, the price every consumer computes."""
    return _phi_cdf(np.asarray(spread, dtype=float) / MARGIN_SD)


def shifted_prob(spread: np.ndarray, fit: Fit) -> np.ndarray:
    """P(home win) under the location-shifted price."""
    s = np.asarray(spread, dtype=float)
    d = np.asarray(fit.delta)
    return _phi_cdf((s + np.sign(s) * d[bucket(s)]) / fit.sd)


def oracle_rates(spread: np.ndarray, won: np.ndarray) -> dict[float, float]:
    """The favourite's realised win rate by one-point bucket of |s| (`margin.ceiling`'s oracle)."""
    s = np.asarray(spread, dtype=float)
    y = np.asarray(won, dtype=float)
    fav = np.where(s > 0, y, 1.0 - y)
    bk = np.rint(np.abs(s))
    return {float(v): float(fav[bk == v].mean()) for v in np.unique(bk)}


def oracle_prob(spread: np.ndarray, rates: dict[float, float]) -> np.ndarray:
    """Home-win probability under the oracle: the favourite's bucket rate, 0.5 at a pick'em."""
    s = np.asarray(spread, dtype=float)
    bk = np.rint(np.abs(s))
    r = np.array([rates[float(v)] for v in bk])
    out = np.where(s > 0, r, 1.0 - r)
    return np.where(s == 0, 0.5, np.clip(out, 1e-6, 1.0 - 1e-6))


def walk_forward(resid: pl.DataFrame, *, trailing: int = TRAILING) -> pl.DataFrame:
    """One row per held-out season: both arms' log loss, the paired `diff`, and the ceiling.

    `resid` is `margin.residuals`'s frame (`season`, `spread_line`, `home_won`, ...). The
    fit for season `t` sees only the `trailing` seasons strictly before it, and a season is
    scored only when it is complete (`MIN_SCORED_GAMES`) and has `MIN_PAST_SEASONS` of
    history. `ll_scale_only` is descriptive and outside the gate. `ceiling_diff` needs every
    held-out game, so it is attached after the loop.
    """
    rows: list[dict] = []
    held: list[pl.DataFrame] = []
    for yr, past, now in expanding_seasons(resid):
        window = past.filter(pl.col("season") >= yr - trailing)
        if past["season"].n_unique() < MIN_PAST_SEASONS or now.height < MIN_SCORED_GAMES:
            continue
        s_tr = window["spread_line"].to_numpy().astype(float)
        y_tr = window["home_won"].to_numpy().astype(float)
        full = fit_shift(s_tr, y_tr)
        scale = fit_shift(s_tr, y_tr, shifts=False)
        s = now["spread_line"].to_numpy().astype(float)
        y = now["home_won"].to_numpy().astype(float)
        ll_inc = log_loss(incumbent_prob(s), y)
        ll_sh = log_loss(shifted_prob(s, full), y)
        ll_sc = log_loss(shifted_prob(s, scale), y)
        rows.append({"season": yr, "n": now.height, "sd": full.sd, "ll_incumbent": ll_inc,
                     "ll_shifted": ll_sh, "ll_scale_only": ll_sc, "diff": ll_inc - ll_sh})
        held.append(now)
    if not rows:
        return pl.DataFrame(schema={"season": pl.Int64, "n": pl.Int64, "sd": pl.Float64,
                                    "ll_incumbent": pl.Float64, "ll_shifted": pl.Float64,
                                    "ll_scale_only": pl.Float64, "diff": pl.Float64,
                                    "ceiling_diff": pl.Float64})
    pooled = pl.concat(held)
    rates = oracle_rates(pooled["spread_line"].to_numpy().astype(float),
                         pooled["home_won"].to_numpy().astype(float))
    ceilings = []
    for now in held:
        s = now["spread_line"].to_numpy().astype(float)
        y = now["home_won"].to_numpy().astype(float)
        ceilings.append(log_loss(incumbent_prob(s), y) - log_loss(oracle_prob(s, rates), y))
    return pl.DataFrame(rows).with_columns(pl.Series("ceiling_diff", ceilings, dtype=pl.Float64))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.favourite_longshot",
        description="Gate the favourite-longshot location correction (#465). Measurement only.")
    ap.add_argument("--run", action="store_true", help="run the walk-forward gate once")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 0

    from hub.fetch import nflverse

    print("  loading schedules ...")
    try:
        resid = residuals(nflverse.load("schedules", nflverse.every_season()))
    except Exception as e:
        return unavailable("hub.models.favourite_longshot", "nflverse schedules", e)
    paired = walk_forward(resid)
    if paired.is_empty():
        print("  no held-out season scored; nothing to gate.")
        return 1
    print(f"\n  Walk-forward, {paired.height} held-out seasons "
          f"({int(paired['season'].to_numpy().min())}-"
          f"{int(paired['season'].to_numpy().max())}), fitted on the "
          f"trailing {TRAILING} seasons before each.")
    print(f"  {'season':>6} {'n':>4} {'sd':>7} {'ll_incumbent':>13} {'ll_shifted':>11} "
          f"{'diff':>10} {'scale_only':>11} {'ceiling':>9}")
    for r in paired.iter_rows(named=True):
        print(f"  {r['season']:>6} {r['n']:>4} {r['sd']:>7.3f} {r['ll_incumbent']:>13.5f} "
              f"{r['ll_shifted']:>11.5f} {r['diff']:>+10.5f} "
              f"{r['ll_incumbent'] - r['ll_scale_only']:>+11.5f} {r['ceiling_diff']:>+9.5f}")
    latest = int(resid["season"].to_numpy().max())
    last = resid.filter((pl.col("season") >= latest - TRAILING) & (pl.col("season") < latest))
    final = fit_shift(last["spread_line"].to_numpy().astype(float),
                      last["home_won"].to_numpy().astype(float))
    print(f"\n  Last fit (seasons {latest - TRAILING}-{latest - 1}), descriptive: sd "
          f"{final.sd:.3f}, shifts by bucket {EDGES} "
          + ", ".join(f"{d:+.2f}" for d in final.delta))
    run = HARNESS.run(paired[["season", "diff", "ceiling_diff"]])
    print()
    for line in run.lines:
        print(line)
    print(f"\n  {run.verdict[0]}: {run.verdict[1]}")
    print(f"  resolved {run.resolved}, abstained {run.abstained}; "
          f"MDE {run.summary.get('mde', float('nan')):+.5f} against ceiling "
          f"{run.summary.get('ceiling', float('nan')):+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
