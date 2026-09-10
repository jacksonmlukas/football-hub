"""Proper scoring rules, and the reliability diagram that reads them.

Three of the four rules here are **binary** -- `log_loss`, `brier` and the reliability
diagram all grade a probability against an outcome that happened or did not. `crps` is the
continuous one, and it exists because the Weekly projection does not emit a probability: it
emits a mean, a spread and a skew, and every gate scored it with mean absolute error, which
is minimised by the median and cannot see either of the other two (issue #177).

These lived in `hub.publish` -- the module that writes `site/data/*.json` -- so
`hub.models.eval` imported its metrics from the site writer, and could not be read or
imported without dragging in `nflreadpy`, the survivor solver and the manifest machinery.
The depth was real (two callers, three test files); it was in the wrong file.

Named `scoring_rules` rather than `scoring` on purpose: `hub.models.components.SCORING` is
already the league's fantasy point weights, and this repo does not reuse a word for two
things (see `CONTEXT.md`).

`hub.models.margin` carried a second `log_loss` with `eps=1e-12` against this one's `1e-15`
-- two clipping constants, two test files, one concept. It now imports this.
"""
from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import polars as pl

# The standard normal, both ways, from the standard library rather than scipy -- the trade
# `hub.models.coverage._z` and `hub.models.market.normal_cdf` already declined to make. The
# vectorised wrapper is a Python loop over `math.erf`, which costs nothing beside the array
# arithmetic around it at the tens of thousands of rows this grades.
_erf = np.vectorize(math.erf, otypes=[float])


def normal_quantile(p: Sequence[float] | np.ndarray) -> np.ndarray:
    """The standard normal quantile, vectorised. The grid `crps_from_quantiles` reads."""
    nd = statistics.NormalDist()
    return np.array([nd.inv_cdf(float(x)) for x in np.asarray(p, dtype=float)])


def log_loss(probs: Sequence[float] | np.ndarray,
             outcomes: Sequence[int] | np.ndarray, eps: float = 1e-15) -> float:
    """Mean negative log likelihood.

    Clipped at eps because a model that said 1.0 and was wrong would otherwise put an
    infinity on the page. Clipping bounds the penalty at ~34 per game, which is still
    ruinous and still renders.
    """
    # len() rather than truthiness: `if not probs` raises on a numpy array, which went
    # unnoticed while every caller passed lists. Vectorised because hub.models.eval
    # bootstraps this thousands of times per comparison.
    q = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    if q.size == 0:
        return float("nan")
    y = np.asarray(outcomes, dtype=float)
    return float(np.mean(-(y * np.log(q) + (1.0 - y) * np.log(1.0 - q))))


def brier(probs: Sequence[float] | np.ndarray,
          outcomes: Sequence[int] | np.ndarray) -> float:
    q = np.asarray(probs, dtype=float)
    if q.size == 0:
        return float("nan")
    return float(np.mean((q - np.asarray(outcomes, dtype=float)) ** 2))


# --- the continuous rule --------------------------------------------------
#
# **What CRPS is.** For a predictive distribution `F` and an outcome `y`,
#
#     CRPS(F, y) = INTEGRAL (F(x) - 1{x >= y})^2 dx
#
# -- the squared area between the forecast's CDF and the step function the outcome turns out
# to be. It is *proper*: in expectation it is minimised only by the distribution the outcome
# is actually drawn from, so a forecast cannot score better by lying about its spread, which
# is the whole reason for preferring it to a rule that reads one moment.
#
# **It is on the same scale as MAE, and that is not a coincidence.** A point forecast is a
# point mass, whose CDF is itself a step function, and the integral above then collapses to
# `|y - mu|`. So MAE *is* CRPS -- the score this rule gives the same projection published
# without a distribution -- and the two can be printed in adjacent columns of the same table
# and subtracted. The difference is what the published spread and skew earn or cost.

# Levels for the quantile form below. The error is in the tails, where a midpoint grid stops
# at `1 - 1/(2m)` and scores the mass beyond that level at that quantile; measured against
# the closed form over five (mu, sd, y) cases, worst absolute error 1.0e-2 points at m=100,
# 4.6e-3 at 200, **2.0e-3 at 400** and 5.0e-4 at 1000, and in every one of those the worst
# case is an outcome four standard deviations out. On a weekly fantasy score reported to
# three decimals, at 400 the quadrature is two orders of magnitude below the last digit
# printed, and the cost is one array of `rows x 400` floats. Raise it if a caller ever quotes
# a fifth decimal.
CRPS_QUANTILES = 400


def quantile_levels(m: int = CRPS_QUANTILES) -> np.ndarray:
    """`m` equiprobable levels at the midpoint of each `1/m` slice.

    Midpoints rather than `i/m` because the endpoints of a distribution with unbounded
    support are infinite, and a grid that asks for them scores every observation as
    infinitely bad.
    """
    return (np.arange(m, dtype=float) + 0.5) / m


def crps_normal(mu: Sequence[float] | np.ndarray, sd: Sequence[float] | np.ndarray,
                outcomes: Sequence[float] | np.ndarray) -> np.ndarray:
    """CRPS of a normal predictive distribution, per observation. Exact.

    The closed form (Gneiting & Raftery 2007), with `z = (y - mu)/sd`:

        CRPS = sd * [ z * (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]

    Two consequences worth naming because the tests turn on them. At `y = mu` it is
    `sd * (sqrt(2) - 1)/sqrt(pi)`, a constant times the spread and nothing else. And where
    `y` is drawn from the same normal the expected score is `sd/sqrt(pi)`, against
    `sd*sqrt(2/pi)` for the point mass at the same centre -- so a *calibrated* distribution
    scores exactly `1/sqrt(2)` of what publishing only its centre scores. That ratio is the
    payoff a distributional model is claiming, and it is why the two columns are worth
    printing side by side.

    **`sd = 0` returns `|y - mu|`.** Not a special case bolted on to avoid dividing by zero:
    it is the point-mass limit, and it is what makes MAE a column of this same rule.

    Per observation rather than meaned, unlike `log_loss` and `brier` above, because this has
    the shape of an error column -- it is reported beside one, and grouped the same way.
    """
    m = np.asarray(mu, dtype=float)
    s = np.asarray(sd, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    safe = np.where(s > 0.0, s, 1.0)
    z = (y - m) / safe
    phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
    scored = safe * (z * (2.0 * cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))
    return np.where(s > 0.0, scored, np.abs(y - m))


def crps_from_quantiles(quantiles: np.ndarray, outcomes: Sequence[float] | np.ndarray,
                        levels: np.ndarray | None = None) -> np.ndarray:
    """CRPS of a distribution given by its quantiles, per observation.

    For the distribution this repo actually publishes, which has no closed form: the weekly
    moments are pushed through a Cornish-Fisher skew and clipped at zero
    (`hub.models.predict.skewed`), and a normal CRPS would grade a distribution nobody
    serves.

    Uses the quantile identity rather than integrating the CDF numerically:

        CRPS(F, y) = 2 * INTEGRAL_0^1 QL_tau(F^-1(tau), y) dtau

    with the pinball loss `QL_tau(q, y) = (y - q)*tau` when `y >= q` and `(q - y)*(1 - tau)`
    otherwise. The CDF integral has a step discontinuity at `y` that quadrature handles
    badly; this one has a kink at the same place but the grid is in *probability*, where the
    forecast's own quantiles put the points where the mass is.

    `quantiles` is `(rows, m)`, ascending along the second axis, at `levels` -- which default
    to `quantile_levels(m)`, the grid they should have been built on.
    """
    q = np.asarray(quantiles, dtype=float)
    y = np.asarray(outcomes, dtype=float).reshape(-1, 1)
    tau = (quantile_levels(q.shape[1]) if levels is None
           else np.asarray(levels, dtype=float)).reshape(1, -1)
    pinball = np.where(y >= q, (y - q) * tau, (q - y) * (1.0 - tau))
    return 2.0 * pinball.mean(axis=1)


def reliability_by(df: pl.DataFrame, edges: Sequence[float], *, on: str,
                   prob: str = "home_win_prob", outcome: str = "home_won",
                   places: int = 1) -> list[dict[str, Any]]:
    """Predicted versus actual, binned on `on` rather than on the probability itself.

    `reliability` bins a probability against its own value, which answers "when the model
    said 70%, did it happen 70% of the time". That is the right question when the model is
    the thing under test and the wrong one when the *input* is: a survivor pick is chosen by
    spread, so a miss concentrated in one spread range is invisible in a diagram whose bins
    are probabilities, because every game in a probability bin came from roughly one spread
    anyway. Binning on the input is how a miss gets attributed to the range it lives in.

    `edges` are bin boundaries, low to high; the last bin includes its upper edge so nothing
    at the top of the range falls out of the diagram. Empty bins are kept, for the reason
    `reliability` keeps them.

    `gap` is actual minus predicted, so a positive gap is a model that was *under*-confident.
    """
    if df.is_empty():
        return []
    out = []
    for i, (lo, hi) in enumerate(itertools.pairwise(edges)):
        last = i == len(edges) - 2
        sel = df.filter((pl.col(on) >= lo)
                        & ((pl.col(on) <= hi) if last else (pl.col(on) < hi)))
        n = sel.height
        p = float(cast(float, sel[prob].mean())) if n else None
        a = float(cast(float, sel[outcome].mean())) if n else None
        out.append({"bin": f"{lo:.{places}f}-{hi:.{places}f}", "n": n,
                    "predicted": p, "actual": a,
                    "gap": (a - p) if (p is not None and a is not None) else None})
    return out


def reliability(df: pl.DataFrame, n_bins: int = 10) -> list[dict[str, Any]]:
    """Reliability diagram: predicted versus actual, with the count in each bin.

    Counts are not decoration. `docs/track-record.md` asks for them because a bin holding
    four games says nothing, and a diagram that hides its bin sizes invites exactly the
    over-reading the page exists to prevent.

    The equal-width probability case of `reliability_by`, rather than a second copy of the
    binning loop. The bins and their labels are unchanged: the last one used to admit its
    upper edge by comparing against 1.01, which for a probability validated into [0, 1] is
    the same set of rows as including the edge.
    """
    edges = [i / n_bins for i in range(n_bins + 1)]
    return reliability_by(df, edges, on="home_win_prob")
