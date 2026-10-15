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

**The touchdown count (#307, 2026-10-15).** Under-dispersed counts are drawn from the
Conway-Maxwell-Poisson, whose `nu` is solved so the variance over mean is the fitted constant
at the mean being drawn, with no ceiling on the support; and a phase's touchdowns are paired
with its yards at a fitted correlation (`TD_YARDS_RHO`) rather than drawn apart. Both are in
`docs/td-count-sampler.md`, with what each moved.

**The expected columns are not point-in-time (named 2026-10-07, #313, audit IV R11).** The
`_exp` columns `EXPECTED` maps are the output of a model nflverse fitted over its whole history,
so a 2022 board reading 2021 values is reading numbers produced with sight of 2022 through 2025.
The weeks averaged are point-in-time; the values are not, and everything downstream of the
expected components (xFP, the Board, the Panel's `expected` priors) inherits it. The direction
is not known, and no figure here measures it.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache

import numpy as np
import polars as pl
from scipy.special import ndtri
from scipy.stats import rankdata

from hub.config import drafted_positions
from hub.declare import chosen, fitted

# Full PPR, matching the league. Reconstructs nflverse's own fantasy_points_ppr to within
# 0.01 for 99.4% of player-weeks; the rest are return and special-teams scores, which no
# drafted skill player is rostered for.
SCORING: dict[str, float] = chosen({
    "passing_yards": 0.04, "passing_tds": 4.0, "interceptions": -2.0,
    "rushing_yards": 0.1, "rushing_tds": 6.0,
    "receiving_yards": 0.1, "receiving_tds": 6.0, "receptions": 1.0,
    "fumbles_lost": -2.0, "two_point_conversions": 2.0,
})

# Where each scored stat's *expected* value comes from upstream, beside the weights that
# price it. One owner: it was declared twice, in `hub.models.panel` for the weekly screen and
# again in `hub.models.component_error`, and two modules holding the same vocabulary is how
# they come to disagree about which upstream column means "expected receiving yards".
#
# A tuple because the mapping is not one-to-one. Two-point conversions are scored once and
# arrive in three columns -- passing, receiving, rushing -- and folding them into a single
# name would either drop two thirds of them or invent a column that does not exist. Including
# them takes the rebuild's mean absolute error against the pre-summed total from 0.056 to
# 0.017 points a player-week, so they are small and not nothing.
#
# **Consumers take subsets, and should.** The Panel deliberately uses four of these: counts
# have no expected version and should not, and the touchdowns are left out for a reason that
# has moved twice and is now this: the Panel wants the priors `hub.exhibits.weekly_projection`
# divides by a count to hold an efficiency, and touchdown *rate* is the one efficiency this repo has
# specifically declined to project. The Board needs the touchdowns because six points is a heavy
# weight on a small error. Same vocabulary, different questions.
#
# This comment used to say the touchdown rate had been *measured* to regress -- `td_rate_prior`
# at -0.040 across five of five seasons, a broken pre-stated null. That was true on the control
# basis it was taken on, `(ppg_before, ecr)`, and `ppg_before` is PPR points, which contain
# the feature's own numerator. On the basis #229 settled, `(yds_prior, ecr)`, it is **-0.012
# across four of five seasons with 2023 positive**, which `docs/method.md` rule 4 reads as a
# bug and not a finding. So it is not cited here as a finding. The subset does not move --
# #179 checked that either reading leaves the touchdowns out -- and `docs/weekly-screen.md`
# carries the figure on every basis it was run on.
EXPECTED: dict[str, tuple[str, ...]] = {
    "receptions": ("receptions_exp",),
    "receiving_yards": ("rec_yards_gained_exp",),
    "receiving_tds": ("rec_touchdown_exp",),
    "rushing_yards": ("rush_yards_gained_exp",),
    "rushing_tds": ("rush_touchdown_exp",),
    "passing_yards": ("pass_yards_gained_exp",),
    "passing_tds": ("pass_touchdown_exp",),
    "interceptions": ("pass_interception_exp",),
    "two_point_conversions": ("pass_two_point_conv_exp", "rec_two_point_conv_exp",
                              "rush_two_point_conv_exp"),
}

# Scored, but with no expected column upstream at all. Listed rather than omitted so that
# "unmapped" and "forgotten" are different states: a scoring item missing from both this and
# `EXPECTED` fails a test rather than silently contributing zero.
#
# `ff_opportunity` publishes realised fumbles (`rec_fumble_lost`, `rush_fumble_lost`) and no
# expectation of them, which is defensible -- a fumble is an event, not a rate anyone models.
NO_EXPECTED_SOURCE: frozenset[str] = frozenset({"fumbles_lost"})

# Touchdowns per yard, 2022-25. Only phases where the position has real volume: a
# quarterback's raw receiving rate is 0.125 per yard, twenty times any true rate, because
# quarterbacks catch a pass a season and it is usually a trick play that scores. Applying
# that to a quarterback who caught one would invent points from a sample of nothing.
TD_RATE: dict[str, dict[str, float]] = fitted({
    "QB": {"pass": 0.00615, "rush": 0.01045},
    "RB": {"rec": 0.00521, "rush": 0.00707},
    "WR": {"rec": 0.00605, "rush": 0.00756},
    "TE": {"rec": 0.00705},
})
FALLBACK_TD_RATE: dict[str, float] = fitted({"rec": 0.00600, "rush": 0.00720, "pass": 0.00615})

# Yardage is compound, not a fixed-spread quantity: a week's receiving yards are the sum
# of one gain per catch. Modelling it as a Gamma with a constant coefficient of variation
# makes its variance grow with the square of the mean, which quietly reimposes the
# proportional spread law this layer exists to replace -- the first version of this file
# did exactly that and produced spread growing as mu^1 instead of sqrt(mu).
#
# So sample the units and the yards-per-unit separately. These are per-unit dispersions,
# not per-week ones, and the aggregate weekly CV falls out of them.
#
# The four dispersions below are in the one digest since #253. They were argued out of it
# -- "no points prediction reads it" -- while `hub.models.props`, which prices a published
# prop from them, carried them in a private hash of its own; one model version that says
# less than the predictions stamped with it is the failure ADR-0006 exists to prevent, so
# the fold moved `fitted_digest` as a coverage correction, recorded in the pin.
PER_UNIT_CV: dict[str, float] = chosen(
    {"pass": 1.00, "rush": 1.47, "rec": 0.76})

# Typical yards per unit, used to infer a volume count when the caller supplies yards
# without one. Real usage should pass the count.
YARDS_PER_UNIT: dict[str, float] = chosen(
    {"pass": 7.1, "rush": 4.3, "rec": 11.6})

# Volume counts are overdispersed relative to Poisson -- measured variance-to-mean, 2022-25.
# Usage itself moves week to week and a Poisson around a fixed rate cannot say so: a team
# that is ahead runs, a quarterback who is behind throws. Carries and attempts are twice
# Poisson; receptions barely. Sampling counts as Poisson understated weekly spread by about
# 13% and correspondingly overstated skew, because too much of the variance was left to come
# from the touchdown term.
COUNT_DISPERSION: dict[str, float] = fitted(
    {"pass": 2.45, "rush": 2.39, "rec": 1.20})

# Touchdowns are *under*dispersed -- variance-to-mean 0.83 to 0.86, measured the same way.
# Poisson would be 1.0, and using it overstated weekly skew (0.80 simulated against 0.60
# observed) because the lumpiest term was drawn lumpier than it really is. Under 1.0 the
# family is the Conway-Maxwell-Poisson, whose `nu` is solved so the variance over mean is this
# number at the mean being drawn (#307; it was a binomial of `round(mean / (1 - phi))` trials,
# which has a ceiling a few counts above the mean and drifts off the number it holds).
TD_DISPERSION: dict[str, float] = fitted(
    {"pass": 0.85, "rush": 0.86, "rec": 0.83})

# The largest Conway-Maxwell-Poisson `nu` the solver will return: where a dispersion is below
# what any count of that mean can reach (0.83 at a mean under 0.17), `nu` runs to this and the
# law is the two-point one on the integers either side of the mean. Past about 10 the law no
# longer moves at any mean a touchdown count has, so the value is a stop and not a fit.
CMP_NU_MAX: float = chosen(64.0)

# How tightly a phase's touchdowns follow its yards, week to week within a player: the
# Gaussian-copula correlation `draw_touchdowns` pairs the drawn counts with the yards' ranks at
# (#307). Fitted so the sampler's own within-player Pearson correlation of yards and touchdowns
# on each 2022-25 player-season's line equals the observed one; `docs/td-count-sampler.md`.
TD_YARDS_RHO: dict[str, float] = fitted(
    {"pass": 0.506, "rush": 0.465, "rec": 0.456})

# Teammate correlation is owned by `hub.models.predict` -- who moves together is a
# prediction, where this module is about how stats become points. It used to be re-exported
# from here, which made `predict` import its own symbol back through this module; callers
# now take it from `predict` directly.


_PHASE_YARDS = {"pass": "passing_yards", "rush": "rushing_yards", "rec": "receiving_yards"}
_PHASE_TDS = {"pass": "passing_tds", "rush": "rushing_tds", "rec": "receiving_tds"}
# The volume count behind each phase's yardage.
_PHASE_UNITS = {"pass": "attempts", "rush": "carries", "rec": "receptions"}


def _log_factorials(k_max: int) -> np.ndarray:
    """`log(k!)` for k = 0..k_max."""
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, k_max + 1)))])


def _cmp_support(mean: float) -> int:
    """The last count the Conway-Maxwell-Poisson tables carry for a mean. Ten Poisson
    standard deviations and twenty-five counts past the mean: at `nu >= 1` the tail is
    lighter than the Poisson's, whose mass there is below 1e-20, so the cut is not a
    ceiling anything can see. It is wide on purpose -- the defect this replaced was a
    ceiling at `round(mean / (1 - phi))`, a few counts above the mean."""
    return int(mean + 10.0 * np.sqrt(mean) + 25.0)


def _cmp_log_lambda(mean: np.ndarray, nu: np.ndarray, k_max: int) -> np.ndarray:
    """The `log(lambda)` at which the Conway-Maxwell-Poisson `P(k) ~ lambda^k / (k!)^nu`
    has this mean, elementwise: bisection on a mean that rises with `lambda`, all pairs at
    once. Sixty halvings of a bracket a few hundred wide leave it at machine precision."""
    ks = np.arange(k_max + 1)
    log_fact = _log_factorials(k_max)
    lo = np.full(mean.shape, -60.0)
    hi = nu * np.log(k_max + 1.0) + 10.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        logp = ks[None, :] * mid[:, None] - nu[:, None] * log_fact[None, :]
        p = np.exp(logp - logp.max(axis=1, keepdims=True))
        rising = (p @ ks) / p.sum(axis=1) < mean
        lo = np.where(rising, mid, lo)
        hi = np.where(rising, hi, mid)
    return (lo + hi) / 2.0


def _cmp_dispersion(mean: float, nu: np.ndarray) -> np.ndarray:
    """Variance over mean of the Conway-Maxwell-Poisson at this mean, for each `nu`."""
    k_max = _cmp_support(mean)
    ks = np.arange(k_max + 1)
    log_fact = _log_factorials(k_max)
    log_lam = _cmp_log_lambda(np.full(nu.shape, mean), nu, k_max)
    logp = ks[None, :] * log_lam[:, None] - nu[:, None] * log_fact[None, :]
    p = np.exp(logp - logp.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    m = p @ ks
    return ((p @ ks ** 2) - m ** 2) / m


@lru_cache(maxsize=8192)
def cmp_nu(mean: float, phi: float) -> float:
    """The Conway-Maxwell-Poisson `nu` whose variance over mean is `phi` at this mean.

    `nu = 1` is the Poisson (`phi = 1`) and a larger `nu` is a narrower distribution, so the
    dispersion index falls monotonically from one and a bracket search finds it: five passes
    of a 17-point grid, each narrowing the bracket sixteen-fold. **A target below what any
    count distribution of this mean can reach returns `CMP_NU_MAX`**, the two-point law on the
    integers either side of the mean, which is the narrowest there is: a count with mean
    m < 1 cannot have variance below `m(1-m)`, so `phi` under `1 - m` is unreachable -- 0.83 at
    a mean under 0.17 -- and the shipped binomial's answer there was not 0.83 either.
    """
    if not 0.0 < phi < 1.0:
        raise ValueError(f"the Conway-Maxwell-Poisson narrows a count: phi={phi} is not in (0, 1)")
    if mean <= 0.0:
        raise ValueError(f"mean={mean}: there is no count to narrow at a mean of zero")
    if _cmp_dispersion(mean, np.array([CMP_NU_MAX]))[0] >= phi:
        return CMP_NU_MAX
    lo, hi = 1.0, CMP_NU_MAX
    for _ in range(5):
        grid = np.linspace(lo, hi, 17)
        i = min(max(int((_cmp_dispersion(mean, grid) > phi).sum()), 1), 16)
        lo, hi = float(grid[i - 1]), float(grid[i])
    return (lo + hi) / 2.0


def cmp_pmf(mean: float, nu: float) -> np.ndarray:
    """The Conway-Maxwell-Poisson probabilities of 0..K at this mean and `nu`, summing to one."""
    k_max = _cmp_support(mean)
    ks = np.arange(k_max + 1)
    log_lam = _cmp_log_lambda(np.array([mean]), np.array([nu]), k_max)[0]
    logp = ks * log_lam - nu * _log_factorials(k_max)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def draw_counts(rng: np.random.Generator, mean: float, phi: float, n: int) -> np.ndarray:
    """Draw `n` counts with mean `mean` and variance `phi * mean`.

    Three families, picked by what the measurement says rather than by convention:
    negative binomial above Poisson, Conway-Maxwell-Poisson below it, Poisson at it. Volume
    counts are overdispersed because usage moves week to week; touchdowns are underdispersed
    because they are bounded by a small number of scoring chances.

    **The family below one is the Conway-Maxwell-Poisson, and why it is not the binomial
    (#307).** A binomial of `round(mean / (1 - phi))` trials has a ceiling at that many
    counts -- two touchdowns at a WR1's rate and one below a mean of 0.3, where the
    Poisson's chance of three is 0.6% -- and the rounding moves the realised dispersion off
    the constant it was fitted to hold. The Katz family has the same ceiling (`b < 0` is
    its under-dispersed side, a binomial in a continuous parameter, with support ending at
    `-a/b`). The Conway-Maxwell-Poisson has none and `nu` solves the dispersion exactly at
    every mean the dispersion is reachable at (`cmp_nu`).
    """
    if mean <= 0:
        return np.zeros(n, dtype=int)
    if phi > 1.0:
        # var/mean = 1/p, and mean = r(1-p)/p
        p = 1.0 / phi
        return rng.negative_binomial(max(mean * p / (1.0 - p), 1e-9), p, n)
    if phi < 1.0:
        mean = float(mean)
        cdf = np.cumsum(cmp_pmf(mean, cmp_nu(mean, float(phi))))
        return np.minimum(np.searchsorted(cdf, rng.random(n), side="left"), cdf.size - 1)
    return rng.poisson(mean, n)


def draw_touchdowns(rng: np.random.Generator, mean: float, phi: float, yards: np.ndarray,
                    rho: float) -> np.ndarray:
    """A phase's touchdowns, drawn *given the week's yards* (#307).

    The marginal count is `draw_counts` -- the measured mean and dispersion, untouched. What
    changes is which weeks get which count: a Gaussian copula pairs them with the yards at
    correlation `rho`, by giving the weeks a latent score `rho * z + sqrt(1 - rho^2) * e`
    (`z` the normal score of the week's yards rank, `e` a fresh normal) and handing the largest
    counts to the largest scores. A big week is big because the yards and the touchdowns
    happened together; drawn apart, as this module did, the weeks that are high on both are
    missing, and the upper tail of every total is thinned a second time.

    The coupling is a reordering of the counts drawn, so the marginal law is the one drawn
    and no family needs its own conditional. A phase with no yards drawn (touchdowns supplied
    without them) has nothing to condition on and is drawn marginally."""
    tds = draw_counts(rng, mean, phi, yards.size)
    if mean <= 0 or rho == 0.0 or not yards.any():
        return tds
    z = ndtri((rankdata(yards) - 0.5) / yards.size)
    latent = rho * z + np.sqrt(1.0 - rho ** 2) * rng.standard_normal(yards.size)
    out = np.empty_like(tds)
    out[np.argsort(latent, kind="stable")] = np.sort(tds)
    return out


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


def expected_columns() -> list[str]:
    """Every upstream column `EXPECTED` names, flattened. What a fetch has to ask for."""
    return [c for cols in EXPECTED.values() for c in cols]


def from_opportunity(weekly: pl.DataFrame, *, by: str = "player_id") -> pl.DataFrame:
    """Per-game expected components, and the points they add up to under this league's scoring.

    The Board reads a pre-summed expected-points column and drops the twenty-one component
    columns that sit beside it, so a disagreement with any other source is a scalar and cannot
    be attributed to a stat. This turns the same frame into the components *and* the total, so
    the total stays comparable while the parts become inspectable.

    **Touchdowns are not regressed here and must not be.** The upstream touchdown columns are
    already an expectation -- that is what `_exp` means -- and `td_luck` is defined elsewhere in
    this repo as actual minus expected. `regress_touchdowns` replaces a player's *own realised*
    rate with his position's and belongs on a frame of realised totals, which this is not.
    Applying it here would regress an expectation that has already been regressed, and would
    leave `td_luck` with nothing to explain.

    Missing columns contribute zero and are named in the `missing` attribute of the result
    rather than dropped, because a projection that quietly gets smaller as its inputs vanish
    looks exactly like a projection of a worse player.
    """
    present = {k: [c for c in cols if c in weekly.columns] for k, cols in EXPECTED.items()}
    missing = sorted(c for cols in EXPECTED.values() for c in cols if c not in weekly.columns)
    usable = {k: cols for k, cols in present.items() if cols}
    if not usable or weekly.is_empty():
        out = weekly.head(0).select(by) if by in weekly.columns else pl.DataFrame({by: []})
        return out.with_columns(pl.lit(0.0).alias("games"), pl.lit(0.0).alias("xfp_per_game"))

    totals = weekly.group_by(by).agg(
        [pl.sum_horizontal([pl.col(c).cast(pl.Float64).fill_null(0.0) for c in cols])
           .sum().alias(k) for k, cols in usable.items()]
        + [pl.len().alias("games")])
    per_game = totals.with_columns(
        [(pl.col(k) / pl.col("games")).alias(k) for k in usable])
    return per_game.with_columns(
        points_expr(usable.keys()).alias("xfp_per_game")).with_columns(
        pl.lit(", ".join(missing) or None).alias("missing_components"))


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


def phase_draws(rng: np.random.Generator, c: Mapping[str, float], phase: str,
                n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One phase's (units, yards, touchdowns) over `n` weeks.

    Units are counts with measured dispersion. Yards are a Gamma compounded on the drawn count
    -- a week's yardage is the sum of one gain per unit -- so their variance grows with the
    count rather than with the square of the mean, which is the square-root spread law arriving
    as an output. A line that carries yards and no count infers one from the phase's typical
    yards per unit, so the compound structure holds either way. **Touchdowns are drawn last,
    given the yards (#307)**: the units and the yards of a seed are the draws they were, and
    only the touchdowns, which follow the yards, are new.
    """
    mu_y = float(c.get(_PHASE_YARDS[phase]) or 0.0)
    mu_u = float(c.get(_PHASE_UNITS[phase]) or 0.0)
    if mu_y > 0 and mu_u <= 0:
        # No count supplied: infer one so the compound structure still holds.
        mu_u = mu_y / YARDS_PER_UNIT[phase]
    units = draw_counts(rng, mu_u, COUNT_DISPERSION[phase], n)
    yards = np.zeros(n)
    if mu_y > 0 and mu_u > 0:
        # Sum of `units` iid Gamma(1/cv^2, m*cv^2) is Gamma(units/cv^2, m*cv^2),
        # so a week's yardage is drawn in one call and its variance grows with the
        # number of units -- linearly in the mean, which is the whole point.
        cv = PER_UNIT_CV[phase]
        shape = units / cv ** 2
        nz = shape > 0
        yards[nz] = rng.gamma(shape[nz], (mu_y / mu_u) * cv ** 2)
    tds = draw_touchdowns(rng, float(c.get(_PHASE_TDS[phase]) or 0.0), TD_DISPERSION[phase],
                          yards, TD_YARDS_RHO[phase])
    return units, yards, tds


def sample_weeks(c: Mapping[str, float], position: str, n: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Draw `n` weeks of fantasy points for a player with these per-game components.

    Counts are negative binomial above Poisson and Conway-Maxwell-Poisson below it, yards are
    Gamma compounded on the count, and touchdowns are drawn given the yards -- all from measured
    dispersion. Aggregating the draws rather than drawing points directly is what produces the
    right skew, and what makes the square-root spread law an output instead of an assumption.
    """
    total = np.zeros(n)
    counted_receptions = False
    for phase, yards_key in _PHASE_YARDS.items():
        units, yards, tds = phase_draws(rng, c, phase, n)
        total += SCORING[yards_key] * yards + SCORING[_PHASE_TDS[phase]] * tds
        if phase == "rec" and (float(c.get(_PHASE_UNITS[phase]) or 0.0) > 0
                               or float(c.get(yards_key) or 0.0) > 0):
            total += SCORING["receptions"] * units
            counted_receptions = True

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
