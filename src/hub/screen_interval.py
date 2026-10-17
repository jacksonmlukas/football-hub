"""The interval a preseason screen reports instead of the word "null" (#505).

A screen that finds partial r near zero has not shown r is zero; it has shown r is inside
some interval. `docs/detectable-effects.md` rows 1-5 put the smallest effect these designs
could resolve at partial r 0.09-0.21, so "null" overstated them. The accurate reading is
*inconclusive, consistent with r no larger than the interval's ends*.

**Method.** Two, by what the record stored:

* `fisher_interval(r, n)` -- the textbook Fisher-z interval for a partial correlation with
  `controls` control variables: `z = atanh(r)`, `se_z = 1 / sqrt(n - 3 - controls)`, interval
  `tanh(z -/+ z_{0.975} * se_z)`. For the age screen `se_z` is 0.0216 at n = 2,140, the "se
  ~0.022" `docs/signal-screens.md` printed before this existed.
* `t_interval(mean, se, seasons)` -- for a screen whose record keeps only the mean of the
  per-season r's and the between-season standard error of that mean (the rest-of-season
  depth-chart screen): `mean -/+ t_{0.975, k-1} * se`.

Neither is invented where the record does not hold its inputs: a screen with no stored sample
size and no stored se gets no interval from this module, and the docs say so.
"""
from __future__ import annotations

import math
from statistics import NormalDist

from hub.declare import not_an_input
from hub.models.experiment import t_quantile

LEVEL = not_an_input(
    0.95, "the coverage a screen's interval is printed at; it reaches no prediction or model")


def fisher_interval(r: float, n: int, controls: int = 1,
                    level: float = LEVEL) -> tuple[float, float]:
    """The Fisher-z interval for a partial correlation `r` from `n` rows and `controls` controls."""
    dof = n - 3 - controls
    if dof < 1:
        raise ValueError(f"n = {n} with {controls} controls leaves no degrees of freedom")
    if not -1.0 < r < 1.0:
        raise ValueError(f"r = {r} is not a correlation strictly inside (-1, 1)")
    half = NormalDist().inv_cdf(0.5 + level / 2) / math.sqrt(dof)
    z = math.atanh(r)
    return math.tanh(z - half), math.tanh(z + half)


def t_interval(mean: float, se: float, seasons: int,
               level: float = LEVEL) -> tuple[float, float]:
    """`mean` -/+ t(k-1) * `se`; `se` is the between-season se of the mean of `seasons` r's."""
    if seasons < 2:
        raise ValueError("a between-season se needs at least two seasons")
    half = t_quantile(0.5 + level / 2, seasons - 1) * se
    return mean - half, mean + half


def render(r: float, interval: tuple[float, float]) -> str:
    """`partial r = +0.008, 95% interval [-0.049, +0.066]` -- the form `publish_ledger` reads."""
    lo, hi = interval
    return f"partial r = {r:+.4f}, 95% interval [{lo:+.3f}, {hi:+.3f}]"
