"""P(win | spread) fitted on wins, with sd allowed to depend on the total, as a gate (#502).

`MARGIN_SD = 12.741` turns every spread in this repo into a win probability. It was fitted to
the dispersion of *margins*, while the product is `P(home win)`, and it is one number for every
game, though a game expected to have more scoring should scatter further around its spread.
This module asks whether a probit of `home_won` on the spread, its sd fitted **on wins** and
allowed to grow with the closing total, prices held-out games better than the live mapping.
It is a **measurement**: it wires no consumer and changes no price. It only *reads*
`hub.models.margin` and `hub.models.market`, which the weekly arm pins (#456), and edits
neither; every constant here is `not_an_input`, so no model version moves.

**The pre-registration is `docs/margin-sd.md`, "Pre-registered 2026-10-18 ... (#502)", and was
committed before any number here existed.** What follows is that text in code.

* **Incumbent**: `Phi(s / MARGIN_SD)`, the live constant, held fixed in every season.
* **Challenger**: `Phi(s / sd(T))`, `sd(T) = sd0 · (T / T_REF) ^ β`, `T` the closing total. `sd0`
  and `β` are fitted jointly by maximum likelihood on `home_won`, on the **trailing ten seasons
  strictly before** the scored one (rule 2; `experiment.expanding_seasons`). The form nests
  both forms #502 names: a constant sd fitted on wins at β = 0, sd ∝ √T at β = ½.
* **Pairing**: a game enters a fit or a score only with a spread *and* a total, and both arms
  are scored on the same games.
* **Unit**: the season; **metric**: walk-forward log loss, the paired difference
  `LL(incumbent) - LL(challenger)`, positive when the challenger is better.
* **Ceiling arm**: a perfect `P(win | spread, total)`, the favourite's realised rate in cells of
  one-point `|s|` buckets by three total bands (`TOTAL_BANDS`), pooled over every held-out game.
* **Descriptive, outside the gate**: `scale_only` (β pinned at zero: the arm #465 already
  looked at, post hoc) and `increment` (what the total adds over it). Never ranked on.

The fit is Fisher scoring on the probit with an analytic information matrix, as
`favourite_longshot.fit_shift` is, so it needs no optimiser dependency and is deterministic.

    uv run python -m hub.models.total_sd --run
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from math import erf, sqrt
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.declare import not_an_input
from hub.ledger import Ledger
from hub.ledger import recipe as _recipe
from hub.models.experiment import Actions, Harness, expanding_seasons, t_interval

# Imported rather than restated, so the incumbent and the live price cannot drift apart.
from hub.models.margin import TRAILING, home_won
from hub.models.market import MARGIN_SD
from hub.models.scoring_rules import log_loss

T_REF = not_an_input(
    44.0, "the total sd0 is quoted at; it reparametrises sd0 and the fitted price does not "
          "depend on it")

# The ceiling's total bands, edges in points: below 41, 41 to under 47, 47 and over. Fixed in
# the pre-registration from the usual range of NFL totals, without looking.
TOTAL_BANDS: tuple[float, ...] = (41.0, 47.0)

# Unchanged from #465, so the two gates score the same seasons.
MIN_PAST_SEASONS = 5
MIN_SCORED_GAMES = 240

# A held-out season losing more than this share of its spread-priced games to a missing total
# voids the run: the pairing would be scoring a different population from the one fitted.
MAX_MISSING_TOTAL = not_an_input(
    0.10, "the pre-registered VOID threshold on games dropped for a missing total; it guards a "
          "gate's inputs and no price reads it")

_MAX_ITER = 60
_TOL = not_an_input(
    1e-9, "the step size below which the fit stops iterating; it ends a search and sets no price")
_EPS = not_an_input(
    1e-9, "the clip that keeps the fit's logarithms finite; it bounds a search and sets no price")
_SQRT2 = not_an_input(
    sqrt(2.0), "the mathematical constant that turns erf into the normal CDF; not a choice")
_SQRT2PI = not_an_input(
    sqrt(2.0 * np.pi), "the normal density's normalising constant; mathematics and not a choice")

CEILING_ARM = "a perfect P(win | spread, total), pooled over every held-out game"

ACTIONS = Actions(
    adopt="ADOPT: the fitted sigma(total) becomes eligible to replace Phi(s / MARGIN_SD); "
          "nothing is wired -- hub.models.market is pinned (#456) and MARGIN_SD is "
          "grandfathered for 2026, so adoption is the maintainer's, for the offseason.",
    remove="REMOVE: the fitted sigma(total) prices games worse; disclose it, no consumer.",
    show="SHOW: keep as a measurement, no consumer.",
)

HARNESS = Harness(
    name="total_sd", arm_a="sigma(total) fitted on wins", arm_b="normal_cdf(s / MARGIN_SD)",
    within=("season",), ceiling_arm=CEILING_ARM, actions=ACTIONS,
    unit="nats of log loss per game", places=5,
    arm_roots=("hub.models.total_sd",))


class Fit(NamedTuple):
    """A fitted mapping: sd at the reference total, and its elasticity in the total."""

    sd0: float
    beta: float


def _phi_cdf(z: np.ndarray) -> np.ndarray:
    """The standard normal CDF, exact (`math.erf`), as `margin.home_win_prob` computes it."""
    return 0.5 * (1.0 + np.array([erf(v / _SQRT2) for v in np.asarray(z, dtype=float)]))


def games(schedules: pl.DataFrame) -> pl.DataFrame:
    """`(season, spread_line, total_line, result, home_won)` per scored game with a spread.

    A missing total is kept here, as a null, so the walk-forward can count what the pairing
    drops before it drops it; `margin.home_won` decides what is scorable, one answer for the
    repo."""
    keep = ("season", "spread_line", "total_line", "result")
    missing = sorted(set(keep) - set(schedules.columns))
    if missing:
        raise ValueError(f"schedules is missing {missing}")
    return home_won(schedules.select(keep).drop_nulls("spread_line"))


def sigma(total: np.ndarray, fit: Fit) -> np.ndarray:
    """sd(T) = sd0 · (T / T_REF) ^ β."""
    return fit.sd0 * (np.asarray(total, dtype=float) / T_REF) ** fit.beta


def challenger_prob(spread: np.ndarray, total: np.ndarray, fit: Fit) -> np.ndarray:
    """P(home win) under the fitted mapping."""
    return _phi_cdf(np.asarray(spread, dtype=float) / sigma(total, fit))


def incumbent_prob(spread: np.ndarray) -> np.ndarray:
    """`normal_cdf(s / MARGIN_SD)`, the price every consumer computes."""
    return _phi_cdf(np.asarray(spread, dtype=float) / MARGIN_SD)


def _nll(p: np.ndarray, won: np.ndarray) -> float:
    q = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.sum(won * np.log(q) + (1.0 - won) * np.log(1.0 - q)))


def fit_sigma(spread: np.ndarray, total: np.ndarray, won: np.ndarray, *,
              by_total: bool = True) -> Fit:
    """Maximum likelihood of `(sd0, β)` on `home_won`, unpenalised.

    `by_total=False` pins β at zero and fits a constant sd alone: the descriptive `scale_only`
    arm. Fisher scoring on `(log sd0, β)` with backtracking, so the objective never rises; it
    starts at the incumbent (`MARGIN_SD`, β = 0), which is never where a fixture plants it.
    """
    s = np.asarray(spread, dtype=float)
    y = np.asarray(won, dtype=float)
    u = np.log(np.asarray(total, dtype=float) / T_REF)
    theta = np.array([np.log(MARGIN_SD), 0.0])
    active = np.array([True, by_total])

    def objective(th: np.ndarray) -> float:
        return _nll(_phi_cdf(s / np.exp(th[0] + th[1] * u)), y)

    cur = objective(theta)
    for _ in range(_MAX_ITER):
        z = s / np.exp(theta[0] + theta[1] * u)
        p = np.clip(_phi_cdf(z), _EPS, 1.0 - _EPS)
        dens = np.exp(-0.5 * z * z) / _SQRT2PI
        var = p * (1.0 - p)
        score = (y - p) * dens / var            # d log-lik / dz
        w = dens * dens / var                   # expected information in z
        jac = np.stack([-z, -z * u])            # dz / d(log sd0, β)
        grad = -(jac @ score)
        info = (jac * w) @ jac.T
        idx = np.flatnonzero(active)
        step = np.zeros(2)
        step[idx] = np.linalg.solve(info[np.ix_(idx, idx)] + 1e-10 * np.eye(len(idx)),
                                    grad[idx])
        scale = 1.0
        while scale > 1e-6:
            nxt = theta - scale * step
            new = objective(nxt)
            if new <= cur + 1e-12:
                break
            scale *= 0.5
        else:
            break
        moved = float(np.max(np.abs(scale * step)))
        theta, cur = nxt, new
        if moved < _TOL:
            break
    return Fit(float(np.exp(theta[0])), float(theta[1]))


def band(total: np.ndarray) -> np.ndarray:
    """Index of each total's ceiling band, 0 .. len(TOTAL_BANDS)."""
    return np.searchsorted(np.asarray(TOTAL_BANDS), np.asarray(total, dtype=float), side="right")


def oracle_rates(spread: np.ndarray, total: np.ndarray,
                 won: np.ndarray) -> dict[tuple[float, int], float]:
    """The favourite's realised win rate by (one-point |s| bucket, total band)."""
    s = np.asarray(spread, dtype=float)
    y = np.asarray(won, dtype=float)
    fav = np.where(s > 0, y, 1.0 - y)
    keys = list(zip(np.rint(np.abs(s)).tolist(), band(total).tolist(), strict=True))
    sums: dict[tuple[float, int], list[float]] = {}
    for k, f in zip(keys, fav.tolist(), strict=True):
        acc = sums.setdefault((float(k[0]), int(k[1])), [0.0, 0.0])
        acc[0] += f
        acc[1] += 1.0
    return {k: v[0] / v[1] for k, v in sums.items()}


def oracle_prob(spread: np.ndarray, total: np.ndarray,
                rates: dict[tuple[float, int], float]) -> np.ndarray:
    """Home-win probability under the oracle: the favourite's cell rate, 0.5 at a pick'em."""
    s = np.asarray(spread, dtype=float)
    r = np.array([rates[(float(a), int(b))]
                  for a, b in zip(np.rint(np.abs(s)), band(total), strict=True)])
    out = np.where(s > 0, r, 1.0 - r)
    return np.where(s == 0, 0.5, np.clip(out, 1e-6, 1.0 - 1e-6))


_COLUMNS = {"season": pl.Int64, "n": pl.Int64, "missing_total": pl.Float64,
            "sd0": pl.Float64, "beta": pl.Float64, "ll_incumbent": pl.Float64,
            "ll_challenger": pl.Float64, "ll_scale_only": pl.Float64, "diff": pl.Float64,
            "increment": pl.Float64, "ceiling_diff": pl.Float64}


def _arrays(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (frame["spread_line"].to_numpy().astype(float),
            frame["total_line"].to_numpy().astype(float),
            frame["home_won"].to_numpy().astype(float))


def walk_forward(frame: pl.DataFrame, *, trailing: int = TRAILING) -> pl.DataFrame:
    """One row per held-out season: both arms' log loss, the paired `diff`, and the ceiling.

    `frame` is `games`'s (a null `total_line` allowed). The fit for season `t` sees only the
    `trailing` seasons strictly before it; a season is scored only when it is complete
    (`MIN_SCORED_GAMES` spread-priced games) and has `MIN_PAST_SEASONS` of history.
    `missing_total` is the share of the season's games the pairing drops; `ll_scale_only` and
    `increment` are descriptive and outside the gate. `ceiling_diff` needs every held-out game,
    so it is attached after the loop.
    """
    rows: list[dict] = []
    held: list[pl.DataFrame] = []
    for yr, past, now in expanding_seasons(frame):
        if past["season"].n_unique() < MIN_PAST_SEASONS or now.height < MIN_SCORED_GAMES:
            continue
        window = past.filter(pl.col("season") >= yr - trailing).drop_nulls("total_line")
        scored = now.drop_nulls("total_line")
        s_tr, t_tr, y_tr = _arrays(window)
        full = fit_sigma(s_tr, t_tr, y_tr)
        scale = fit_sigma(s_tr, t_tr, y_tr, by_total=False)
        s, t, y = _arrays(scored)
        ll_inc = log_loss(incumbent_prob(s), y)
        ll_ch = log_loss(challenger_prob(s, t, full), y)
        ll_sc = log_loss(challenger_prob(s, t, scale), y)
        rows.append({"season": yr, "n": scored.height,
                     "missing_total": 1.0 - scored.height / now.height,
                     "sd0": full.sd0, "beta": full.beta, "ll_incumbent": ll_inc,
                     "ll_challenger": ll_ch, "ll_scale_only": ll_sc, "diff": ll_inc - ll_ch,
                     "increment": ll_sc - ll_ch})
        held.append(scored)
    if not rows:
        return pl.DataFrame(schema=_COLUMNS)
    rates = oracle_rates(*_arrays(pl.concat(held)))
    ceilings = []
    for now in held:
        s, t, y = _arrays(now)
        ceilings.append(log_loss(incumbent_prob(s), y) - log_loss(oracle_prob(s, t, rates), y))
    return pl.DataFrame(rows).with_columns(pl.Series("ceiling_diff", ceilings, dtype=pl.Float64))


def void_reason(paired: pl.DataFrame) -> str | None:
    """The pre-registered VOID conditions on the inputs, phrased; `None` when they hold."""
    if paired.height < MIN_PAST_SEASONS:
        return (f"VOID: {paired.height} held-out season(s) scored, fewer than "
                f"{MIN_PAST_SEASONS}; repair the inputs and re-run.")
    bad = paired.filter(pl.col("missing_total") > MAX_MISSING_TOTAL)["season"].to_list()
    if bad:
        return (f"VOID: season(s) {bad} lose more than {MAX_MISSING_TOTAL:.0%} of their "
                f"games to a missing total; repair the inputs and re-run.")
    return None


def increment_summary(paired: pl.DataFrame) -> tuple[float, float, float, int]:
    """Descriptive: mean of the per-season `increment`, its t interval over the seasons, and
    how many seasons it is positive in. Outside the gate; never ranked on."""
    x = paired["increment"].to_numpy().astype(float)
    k = x.size
    mean = float(x.mean())
    se = float(x.std(ddof=1) / sqrt(k)) if k > 1 else float("nan")
    lo, hi = t_interval(mean, se, k)
    return mean, lo, hi, int((x > 0).sum())


def main(argv: Sequence[str] | None = None, *,
         load: Callable[[], pl.DataFrame] | None = None,
         ledger: Ledger | None = None) -> int:
    """`load` and `ledger` are #403's seam: the one network-bound read (every season's
    schedules, `nflverse.load` by default) and the ledger the run is written to."""
    ap = argparse.ArgumentParser(
        prog="hub.models.total_sd",
        description="Gate P(win | spread) fitted on wins with sigma by the total (#502). "
                    "Measurement only.")
    ap.add_argument("--run", action="store_true", help="run the walk-forward gate once")
    a = ap.parse_args(argv)
    if not a.run:
        ap.print_help()
        return 0

    from hub.fetch import nflverse

    print("  loading schedules ...")
    try:
        frame = games(load() if load is not None
                      else nflverse.load("schedules", nflverse.every_season()))
    except Exception as e:
        return unavailable("hub.models.total_sd", "nflverse schedules", e)
    paired = walk_forward(frame)
    if paired.is_empty():
        print("  no held-out season scored; nothing to gate.")
        return 1
    print(f"\n  Walk-forward, {paired.height} held-out seasons "
          f"({int(paired['season'].to_numpy().min())}-"
          f"{int(paired['season'].to_numpy().max())}), fitted on the "
          f"trailing {TRAILING} seasons before each.")
    print(f"  {'season':>6} {'n':>4} {'miss':>5} {'sd0':>7} {'beta':>6} {'ll_incumbent':>13} "
          f"{'ll_challenger':>14} {'diff':>10} {'scale_only':>11} {'increment':>10} "
          f"{'ceiling':>9}")
    for r in paired.iter_rows(named=True):
        print(f"  {r['season']:>6} {r['n']:>4} {r['missing_total']:>5.1%} {r['sd0']:>7.3f} "
              f"{r['beta']:>+6.2f} {r['ll_incumbent']:>13.5f} {r['ll_challenger']:>14.5f} "
              f"{r['diff']:>+10.5f} {r['ll_incumbent'] - r['ll_scale_only']:>+11.5f} "
              f"{r['increment']:>+10.5f} {r['ceiling_diff']:>+9.5f}")
    mean, lo, hi, pos = increment_summary(paired)
    print(f"\n  Descriptive, outside the gate: what the total adds over a constant sigma fitted "
          f"on wins, {mean:+.5f} a game, t interval [{lo:+.5f}, {hi:+.5f}], positive in "
          f"{pos} of {paired.height} seasons.")
    complete = (frame.group_by("season").len().filter(pl.col("len") >= MIN_SCORED_GAMES)
                     .sort("season")["season"].to_list())
    window = complete[-TRAILING:]
    last = frame.filter(pl.col("season").is_in(window)).drop_nulls("total_line")
    final = fit_sigma(*_arrays(last))
    print(f"  Last fit (seasons {window[0]}-{window[-1]}), descriptive: sd0 {final.sd0:.3f} at "
          f"a total of {T_REF:g}, beta {final.beta:+.3f}")
    harness = HARNESS if ledger is None else HARNESS._replace(ledger=ledger)
    run = harness.run(paired[["season", "diff", "ceiling_diff"]], void=void_reason(paired),
                      recipe=_recipe(bands=TOTAL_BANDS, t_ref=T_REF, min_games=MIN_SCORED_GAMES,
                                     min_past=MIN_PAST_SEASONS, trailing=TRAILING,
                                     max_missing=MAX_MISSING_TOTAL))
    print()
    for line in run.lines:
        print(line)
    print(f"\n  {run.verdict[1]}")
    print(f"  resolved {run.resolved}, abstained {run.abstained}; "
          f"MDE {run.summary.get('mde', float('nan')):+.5f} against ceiling "
          f"{run.summary.get('ceiling', float('nan')):+.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
