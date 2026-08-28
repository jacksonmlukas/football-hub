"""Fantasy points as an aggregate of component stats, not a quantity projected directly.

Points are a sum of counts -- receptions, carries, touchdowns -- and modelling the sum
throws away everything the counts know. Three things follow from building it up instead,
all measured in `docs/component-projection.md`:

**Touchdowns regress, volume does not.** Swapping a player's own touchdown rate for his
position's, applied to his own yardage, beats carrying his points forward: RMSE 3.30 against
3.38 across 1,049 player-season pairs, 99.7% on a paired bootstrap. The fitted optimal
shrink is 1.0 -- a player's own touchdown rate carries no information beyond his yards. Be
clear about the size though: that is a ~2% gain on the mean, concentrated in RB and WR, and
quarterbacks come out marginally worse.

**The distribution comes out right for free.** Real weekly scoring is right-skewed (1.04
pooled) with a median at 0.90 of the mean. A normal says 0.00 and 1.00, so the old model
believed the typical week was the projection. It is not -- the mean is carried by touchdown
spikes, and sampling counts reproduces that rather than assuming it away.

**The square-root spread law is derived rather than fitted.** `docs/weekly-spread.md`
measured `sd = k*sqrt(mu)` at an exponent of 0.498 +/- 0.012. Here it is not an input: it
falls out of counts having variance that grows with their mean.

Distributions are chosen from measured dispersion, not assumed. Weekly variance-to-mean
ratios come out at 1.11 for receptions and 0.83-0.86 for each kind of touchdown -- all close
enough to Poisson to use it -- while yards are continuous and right-skewed, so Gamma with a
fitted coefficient of variation.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import polars as pl

from hub.config import drafted_positions

# Full PPR, matching the league. Reconstructs nflverse's own fantasy_points_ppr to within
# 0.01 for 99.4% of player-weeks; the rest are return and special-teams scores, which no
# drafted skill player is rostered for.
SCORING: dict[str, float] = {
    "passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receiving_yards": 0.1, "receiving_tds": 6.0, "receptions": 1.0,
    "fumbles_lost": -2.0, "two_point_conversions": 2.0,
}

# Touchdowns per yard, 2022-25. Only phases where the position has real volume: a
# quarterback's raw receiving rate is 0.125 per yard, twenty times any true rate, because
# quarterbacks catch a pass a season and it is usually a trick play that scores. Applying
# that to a quarterback who caught one would invent points from a sample of nothing.
TD_RATE: dict[str, dict[str, float]] = {
    "QB": {"pass": 0.00615, "rush": 0.01045},
    "RB": {"rec": 0.00521, "rush": 0.00707},
    "WR": {"rec": 0.00605, "rush": 0.00756},
    "TE": {"rec": 0.00705},
}
FALLBACK_TD_RATE: dict[str, float] = {"rec": 0.00600, "rush": 0.00720, "pass": 0.00615}

# Yardage is compound, not a fixed-spread quantity: a week's receiving yards are the sum
# of one gain per catch. Modelling it as a Gamma with a constant coefficient of variation
# makes its variance grow with the square of the mean, which quietly reimposes the
# proportional spread law this layer exists to replace -- the first version of this file
# did exactly that and produced spread growing as mu^1 instead of sqrt(mu).
#
# So sample the units and the yards-per-unit separately. These are per-unit dispersions,
# not per-week ones, and the aggregate weekly CV falls out of them.
PER_UNIT_CV: dict[str, float] = {"pass": 1.00, "rush": 1.47, "rec": 0.76}

# Typical yards per unit, used to infer a volume count when the caller supplies yards
# without one. Real usage should pass the count.
YARDS_PER_UNIT: dict[str, float] = {"pass": 7.1, "rush": 4.3, "rec": 11.6}

# Volume counts are overdispersed relative to Poisson -- measured variance-to-mean, 2022-25.
# Usage itself moves week to week and a Poisson around a fixed rate cannot say so: a team
# that is ahead runs, a quarterback who is behind throws. Carries and attempts are twice
# Poisson; receptions barely. Sampling counts as Poisson understated weekly spread by about
# 13% and correspondingly overstated skew, because too much of the variance was left to come
# from the touchdown term.
COUNT_DISPERSION: dict[str, float] = {"pass": 2.45, "rush": 2.39, "rec": 1.20}

# Touchdowns are *under*dispersed -- variance-to-mean 0.83 to 0.86, measured the same way.
# Poisson would be 1.0, and using it overstated weekly skew (0.80 simulated against 0.60
# observed) because the lumpiest term was drawn lumpier than it really is. Under 1.0 the
# right family is binomial: var/mean = 1 - p, so p follows directly from the measurement.
TD_DISPERSION: dict[str, float] = {"pass": 0.85, "rush": 0.86, "rec": 0.83}

# Teammate correlation is owned by `hub.models.predict` -- who moves together is a
# prediction, where this module is about how stats become points. It used to be re-exported
# from here, which made `predict` import its own symbol back through this module; callers
# now take it from `predict` directly.


_PHASE_YARDS = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}
_PHASE_TDS = {"pass": "passing_tds", "rush": "rushing_tds", "rec": "receiving_tds"}
# The volume count behind each phase's yardage.
_PHASE_UNITS = {"pass": "attempts", "rush": "carries", "rec": "receptions"}


def _counts(rng: np.random.Generator, mean: float, phi: float, n: int) -> np.ndarray:
    """Draw `n` counts with mean `mean` and variance `phi * mean`.

    Three families, picked by what the measurement says rather than by convention:
    negative binomial above Poisson, binomial below it, Poisson at it. Volume counts are
    overdispersed because usage moves week to week; touchdowns are underdispersed because
    they are bounded by a small number of scoring chances.
    """
    if mean <= 0:
        return np.zeros(n, dtype=int)
    if phi > 1.0:
        # var/mean = 1/p, and mean = r(1-p)/p
        p = 1.0 / phi
        return rng.negative_binomial(max(mean * p / (1.0 - p), 1e-9), p, n)
    if phi < 1.0:
        # var/mean = 1 - p for a binomial, so p is fixed by the dispersion and the number
        # of trials by the mean.
        p = 1.0 - phi
        trials = round(mean / p)
        return rng.binomial(max(trials, 1), min(mean / max(trials, 1), 1.0), n)
    return rng.poisson(mean, n)


def scoring_mismatch(league: Mapping[str, float]) -> dict[str, tuple]:
    """Where a league's own scoring disagrees with `SCORING`, as {item: (league, ours)}.

    Fantasy points are an aggregate of real stats, so the weights belong to the league
    rather than to this repo. `SCORING` was hardcoded and assumed to be full PPR -- it does
    match this league on all nine items, but that was unverified until it was checked, and a
    commissioner moving to half-PPR would otherwise mis-score every projection, every
    simulation and every pick without a word.

    An item the league does not set is not a disagreement: ESPN omits anything worth zero.
    An item the league scores and this module does not is reported, since that is the same
    failure pointed the other way.
    """
    out: dict[str, tuple] = {}
    for k, v in league.items():
        ours = SCORING.get(k)
        if ours is None:
            out[k] = (v, None)
        elif abs(float(v) - float(ours)) > 1e-9:
            out[k] = (float(v), float(ours))
    return out


def td_rate(position: str, phase: str) -> float:
    """Touchdowns per yard for a position and phase, falling back where volume is too thin."""
    return TD_RATE.get(position, {}).get(phase, FALLBACK_TD_RATE[phase])


def points(c: Mapping[str, float]) -> float:
    """Full-PPR points from a component line. Absent components are zero."""
    return float(sum(SCORING[k] * float(v) for k, v in c.items() if k in SCORING))


def points_expr(available: Iterable[str] | None = None) -> pl.Expr:
    """The same arithmetic as a polars expression, for whole frames.

    `available` names the columns actually present; anything absent scores zero rather than
    raising, which matches `points` treating a missing component as zero.
    """
    keys = SCORING if available is None else [k for k in SCORING if k in set(available)]
    return sum((pl.col(k).fill_null(0.0) * SCORING[k] for k in keys), start=pl.lit(0.0))


def regress_touchdowns(c: Mapping[str, float], position: str) -> dict[str, float]:
    """Replace a player's own touchdown rate with his position's, on his own yardage.

    A full swap rather than a partial shrink, because that is what the fit says: sweeping
    the shrink weight from 0 to 1 improves monotonically and flattens at 0.75-1.0. Volume
    is untouched -- it is the half of the signal that does carry forward.
    """
    out = dict(c)
    for phase, yards_key in _PHASE_YARDS.items():
        yards = float(out.get(yards_key) or 0.0)
        if yards:
            out[_PHASE_TDS[phase]] = yards * td_rate(position, phase)
    return out


def sample_weeks(c: Mapping[str, float], position: str, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Draw `n` weeks of fantasy points for a player with these per-game components.

    Counts are Poisson and yards are Gamma, both from measured dispersion. Aggregating the
    draws rather than drawing points directly is what produces the right skew, and what
    makes the square-root spread law an output instead of an assumption.
    """
    total = np.zeros(n)
    counted_receptions = False
    for phase, yards_key in _PHASE_YARDS.items():
        mu_y = float(c.get(yards_key) or 0.0)
        units_key = _PHASE_UNITS[phase]
        mu_u = float(c.get(units_key) or 0.0)
        if mu_y > 0 and mu_u <= 0:
            # No count supplied: infer one so the compound structure still holds.
            mu_u = mu_y / YARDS_PER_UNIT[phase]
        if mu_u > 0:
            units = _counts(rng, mu_u, COUNT_DISPERSION[phase], n)
            if mu_y > 0:
                # Sum of `units` iid Gamma(1/cv^2, m*cv^2) is Gamma(units/cv^2, m*cv^2),
                # so a week's yardage is drawn in one call and its variance grows with the
                # number of units -- linearly in the mean, which is the whole point.
                cv = PER_UNIT_CV[phase]
                per_unit = mu_y / mu_u
                shape = units / cv ** 2
                yards = np.zeros(n)
                nz = shape > 0
                yards[nz] = rng.gamma(shape[nz], per_unit * cv ** 2)
                total += SCORING[yards_key] * yards
            if phase == "rec":
                total += SCORING["receptions"] * units
                counted_receptions = True
        mu_td = float(c.get(_PHASE_TDS[phase]) or 0.0)
        if mu_td > 0:
            total += SCORING[_PHASE_TDS[phase]] * _counts(
                rng, mu_td, TD_DISPERSION[phase], n)

    mu_rec = float(c.get("receptions") or 0.0)
    if mu_rec > 0 and not counted_receptions:
        total += SCORING["receptions"] * rng.poisson(mu_rec, n)
    for k in ("interceptions", "fumbles_lost"):
        mu_k = float(c.get(k) or 0.0)
        if mu_k > 0:
            total += SCORING[k] * rng.poisson(mu_k, n)
    return total


def moments(c: Mapping[str, float], position: str, n: int = 20000,
            rng: np.random.Generator | None = None) -> dict[str, float]:
    """Mean, spread, skew and quantiles of a weekly line, by sampling it.

    `p50 < mu` for anyone whose points come partly from touchdowns, which is the number the
    two-moment normal was getting wrong.
    """
    draws = sample_weeks(c, position, n, rng or np.random.default_rng(0))
    sd = float(draws.std())
    return {"mu": float(draws.mean()), "sd": sd,
            "skew": float(((draws - draws.mean()) ** 3).mean() / sd ** 3) if sd else 0.0,
            "p10": float(np.percentile(draws, 10)),
            "p50": float(np.percentile(draws, 50)),
            "p90": float(np.percentile(draws, 90))}


def project(prior: pl.DataFrame) -> pl.DataFrame:
    """Per-game component projection from a prior season's totals.

    Volume carries forward per game; touchdowns are regressed to the positional rate on
    the player's own yardage. Deliberately naive on volume -- this is the aggregation
    layer, and a real volume model (target share, backfield share, depth chart) is a
    separate piece of work that this makes possible rather than replaces.
    """
    per_game = [c for c in
                ("receptions", "receiving_yards", "receiving_tds", "rushing_yards",
                 "rushing_tds", "passing_yards", "passing_tds", "interceptions")
                if c in prior.columns]
    out = prior.with_columns(
        [(pl.col(c) / pl.col("g")).alias(c) for c in per_game])

    for phase, yards_key in _PHASE_YARDS.items():
        if yards_key not in out.columns:
            continue
        rate = pl.col("position").replace_strict(
            {p: td_rate(p, phase) for p in drafted_positions()},
            default=FALLBACK_TD_RATE[phase], return_dtype=pl.Float64)
        out = out.with_columns((pl.col(yards_key) * rate).alias(_PHASE_TDS[phase]))

    return out.with_columns(points_expr(out.columns).alias("proj_ppg"))
