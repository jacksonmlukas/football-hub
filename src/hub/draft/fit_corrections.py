"""Refit the five coefficients that mark the draft board down, against the column they mark.

`hub.draft.regression.TD_LUCK_BETA` (QB, WR), `hub.draft.durability.BETA` (QB, WR) and
`hub.draft.durability.INJURY_BETA` (OUT/DOUBTFUL/IR) are the five numbers that move
`proj_blend`, and through it every championship equity `hub.draft.optimize` reports. Until
this module they had no committed fitting code at all: the tables in `docs/td-luck.md` and
`docs/durability.md` are the output of a fit nobody can re-run.

**This module fits. It does not ship.** The constants stay exactly where they are; changing
them is issue #48's job and the disposition below is its input. A fit that disagrees with a
shipped number is a finding, not a licence.

## The axis, which is the whole point

Both shipped comments name their fit as `ppg_next ~ proj_ppg + signal`. **`proj_ppg` is not
what either correction touches.** `correct_projection` adds `beta * signal` to `proj_blend`,
and `hub.models.predict.blend` defines that as the *mean* of `proj_ppg` and `xfp_per_game` --
two columns that disagree about touchdowns by construction. `xfp_per_game` is an
**expectation**: `hub.models.components` says in as many words that it is already regressed
and that `td_luck` is defined as actual minus expected. `proj_ppg` carries a player's own
realised touchdowns forward. So a touchdown-luck residual measured against one half has the
opposite sign to the same residual measured against the other, and the coefficient applied to
their average is neither.

That is `docs/pick-noise.md`'s defect one layer over: a coefficient fitted on one axis and
applied on another. It was fitted on ranks and applied to picks there; here it is fitted on
one summand and applied to the sum.

## What can be reconstructed, and what cannot

`proj_ppg` for a past season is **gone**. `hub.draft.adp_history` records why -- ESPN does not
retain a past season's draft-time view, verified in August 2026 -- and the same is true of the
projection beside the ADP. A fit that needs it is not reproducible, which is the deeper reason
the shipped numbers have no committed harness.

So the panel carries three baselines, all built from nflverse, and every fit is reported
against all three:

  * `base_xfp`   -- prior-season `xfp_per_game`, exactly `hub.draft.board.expected_points`
                    builds it. This is the half of `proj_blend` that *is* recoverable.
  * `base_carry` -- prior-season realised points per game played. The stand-in for
                    `proj_ppg`: not a market projection, but the only reconstructible column
                    that carries a player's own touchdowns forward the way one does.
  * `base_blend` -- their mean, which is `predict.blend()`'s shape with the stand-in in
                    `proj_ppg`'s place. The headline, because it is the shape of the column
                    actually corrected.

`base_carry` is a proxy and is named as one. What it buys is a **bracket**: a coefficient
with the same sign at both ends is resolved whatever ESPN's projection would have said, and
one that changes sign between them is not resolvable from this repo's data at all. Three of
the five are resolved and two are bracketed; see `docs/fitted-corrections.md`.

## The target, which was already right

`total_next / 17` -- points per **team** game. Not a choice made here: `hub.draft.season`
draws every one of `REG_SEASON_WEEKS + PLAYOFF_ROUNDS` weeks from `mu`, with no games-played
term anywhere, so `proj_blend` is *consumed* as a per-team-game rate. `hub.draft.calibrate`
fixes the same convention for `TALENT_CV` and gives the reason: "a player who misses ten weeks
really did deliver close to nothing".

A player with no outcome row scores zero rather than dropping out, for `calibrate`'s other
reason -- the vanished players are the busts, and dropping them reads as a more predictable
season than the season was.

## The interval

Season-clustered, through `hub.models.experiment.summarise(cluster=SEASON_CLUSTER)`, which
also produces the `se` and the `minimum_detectable_effect` under it. Not a second bootstrap
and not a t approximation of one -- #45's rule, and at these cluster counts the t quantile in
`minimum_detectable_effect` is 1.44x the normal one it replaced, which is large enough to
change a disposition on its own.

A regression coefficient is not a mean of a paired difference, so it reaches `summarise`
through `_contributions` -- the estimator written as a mean over rows, exactly. See that
function for why this is the cluster-robust standard error rather than an approximation of
one, and for the one place it differs from resampling clusters and refitting.

## The shrink sweep, added by #186

`walk_forward` scores two points of one line: the constant applied whole, and not applied at
all. #186 asks a third thing of the same data -- whether a correction that loses at full
strength wins at *part* of it -- and two points cannot answer that without assuming the shape
between them. `shrink_curve` scores the line, on the same splits and the same held-out rows,
so "keep it at a measured shrink" is an arm rather than an option nobody scored. Its ends
reproduce `plain` and `shipped` exactly, which is what makes the middle readable.

    uv run python -m hub.draft.fit_corrections --fit
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.models.experiment import SEASON_CLUSTER, expanding_seasons, summarise

NOT_FITTED_BECAUSE = (
    "every number in it is an output rather than an input -- this module refits the five "
    "coefficients that hub.draft.regression and hub.draft.durability declare, and ships "
    "none of its own. The thresholds here are the span of seasons and the bootstrap seed, "
    "which are settings. See docs/fitted-corrections.md "
)

# Points per team game is the currency, so a season total divides by the games the *team*
# played rather than the games the player did. `hub.draft.durability` and
# `hub.draft.calibrate` both declare this; it is stated a third time rather than imported
# from either, because importing `durability` for a 17 would make the panel builder depend on
# the module it is measuring.
TEAM_GAMES = 17

# The outcome seasons the headline is fitted on. Eight, which is eight season clusters --
# `hub.models.experiment.SMALL_CLUSTERS` is 8, so this sits exactly at the line where the
# percentile bootstrap is still reported beside a t interval rather than alone.
#
# The floor is nflverse's `ff_opportunity`, which is what `xfp_per_game` is built from, and
# the ceiling is the last completed season. Widening it is a `--seasons` away and costs
# nothing but relevance: a 2011 draft board is not this one.
DEFAULT_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025)

# The three baselines, and the one the headline is published on. `base_blend` because that is
# the shape of `proj_blend`, which is the column `correct_projection` writes to.
BASELINES: tuple[str, ...] = ("base_xfp", "base_carry", "base_blend")
HEADLINE_BASELINE = "base_blend"

# The shrink factors `shrink_curve` scores the shipped constant at, held out. A setting, not
# a measurement: the *output* is which of these wins, and widening the grid costs a run.
#
# Zero and one have to be on it, and are asserted to be. They are the two arms
# `walk_forward` already reports -- no correction at all, and the constant applied whole --
# so a sweep that omitted either would be scoring the middle of a line against nothing.
SHRINKS: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

# Designations `hub.draft.durability.INJURY_BETA` prices at -1.631. IR is not on this list and
# cannot be: `INJURY_BETA` borrows the Out coefficient for it precisely because nobody on
# injured reserve appears on a practice report, so a week-1 report has no IR rows to fit. The
# fitted population is therefore Out and Doubtful, which is what the shipped comment says its
# own was.
PRICED_DESIGNATIONS: frozenset[str] = frozenset({"OUT", "DOUBTFUL"})


class Coefficient(NamedTuple):
    """One shipped constant, and where the harness has to look to refit it.

    `shipped` is copied rather than imported so that #48 moving the constant makes this
    module's own tests fail loudly instead of silently refitting against a moved target --
    the disposition is a claim about a *specific* number, and a claim that quietly re-points
    at whatever is current is not a claim. `tests/unit/test_fit_corrections.py` holds the two
    against each other.
    """
    name: str
    declared_in: str        # dotted module path holding the constant
    constant: str           # the name of the dict
    key: str                # the key inside it
    signal: str             # the panel column the coefficient multiplies
    position: str | None    # None means every position, which is what INJURY_BETA does
    shipped: float
    documented_in: str


COEFFICIENTS: tuple[Coefficient, ...] = (
    Coefficient("td_luck.QB", "hub.draft.regression", "TD_LUCK_BETA", "QB",
                "td_luck", "QB", -0.540, "docs/td-luck.md"),
    Coefficient("td_luck.WR", "hub.draft.regression", "TD_LUCK_BETA", "WR",
                "td_luck", "WR", -0.286, "docs/td-luck.md"),
    Coefficient("missed.QB", "hub.draft.durability", "BETA", "QB",
                "missed", "QB", -0.457, "docs/durability.md"),
    Coefficient("missed.WR", "hub.draft.durability", "BETA", "WR",
                "missed", "WR", -0.151, "docs/durability.md"),
    Coefficient("designation.OUT", "hub.draft.durability", "INJURY_BETA", "OUT",
                "designation", None, -1.631, "docs/durability.md"),
)

DISPOSITIONS: tuple[str, ...] = ("reproduced", "unreproduced", "sign-reversed")


class Fit(NamedTuple):
    """A refitted coefficient, its interval, and the population both are about.

    `n` and `n_signal` are two different counts and both are reported, because the shipped
    designation number is published as two that were never reconciled: `docs/durability.md`
    says "1,263 player-seasons" for the regression and "n = 27" for the Out/Doubtful row, and
    a reader has no way to tell those are the same fit counted twice. `n` is the rows the
    regression ran on; `n_signal` is the rows where the signal is non-zero -- the ones
    carrying the whole of the coefficient's evidence.
    """
    coefficient: Coefficient
    baseline: str
    beta: float             # the OLS coefficient on the signal
    beta_baseline: float    # the coefficient on the projection, a calibration read
    lo: float
    hi: float
    se: float
    mde: float
    n: int
    n_signal: int
    clusters: int
    population: str

    @property
    def disposition(self) -> str:
        return disposition(self.shipped_inside, self.beta, self.coefficient.shipped)

    @property
    def shipped_inside(self) -> bool:
        return bool(self.lo <= self.coefficient.shipped <= self.hi)

    @property
    def gap(self) -> float:
        """How far the shipped number sits from this refit, in the units both are in."""
        return abs(self.coefficient.shipped - self.beta)

    @property
    def resolved(self) -> bool:
        """Whether this run had the power to tell the shipped number from the refit.

        The disposition says which side of the interval the shipped value fell; this says
        whether the run could have found the difference at all. They are different questions
        and #45 exists because the second one used to go unasked: an `unreproduced` whose gap
        is under the MDE is a run that could not resolve the disagreement it is reporting, and
        a `reproduced` whose gap is under the MDE is agreement by width rather than by
        evidence. Both are printed, because either one alone misleads.

        The MDE is `minimum_detectable_effect`'s, on `summarise`'s own bootstrap SE, with the
        t quantile #45 corrected it to -- at eight clusters `t(0.975, 7)` is 2.365 against
        `z`'s 1.960, so a disposition taken against the old MDE would call two of these five
        resolved that are not.
        """
        return bool(math.isfinite(self.mde) and self.gap >= self.mde)

    @property
    def at_a_bound(self) -> bool:
        """Whether the shipped number sits exactly on this fit's interval edge.

        `docs/pick-noise.md`'s third finding was that a shipped intercept was exactly
        `MIN_SIGMA` -- the constraint speaking rather than the data -- and #155 is open on it.
        Nothing here is a constrained fit, so the analogue is weaker and is reported rather
        than acted on: a shipped value landing on an interval edge to the precision it is
        published at is a coincidence worth naming.
        """
        s = self.coefficient.shipped
        return any(abs(s - edge) < 5e-4 for edge in (self.lo, self.hi))


def disposition(shipped_inside: bool, beta: float, shipped: float) -> str:
    """Exactly one of `DISPOSITIONS`, for every input. Total by construction.

    The order is the argument. Containment is asked **first**, so a fit whose point estimate
    lands on the other side of zero but whose interval still covers the shipped value is
    `unreproduced` and not `sign-reversed`: the data cannot tell the two apart, and calling
    that a reversal would read a sign off noise. `sign-reversed` is reserved for the case
    where the fit is on the other side of zero *and* the shipped value is outside the
    interval, which is the pair that says something.

    Zero is not a sign. A fitted coefficient of exactly zero reverses nothing.
    """
    if shipped_inside:
        return "reproduced"
    if beta != 0.0 and shipped != 0.0 and (beta > 0) != (shipped > 0):
        return "sign-reversed"
    return "unreproduced"


# --- the estimator, written as a mean so `summarise` can resample it ----------


def _as_int(v: object) -> int:
    """A polars aggregate as an `int`. `Series.max()` is typed as every scalar polars has."""
    return int(v) if isinstance(v, (int, float)) else -1


def _as_float(v: object) -> float:
    """A polars aggregate as a `float`, NaN when the column was empty."""
    return float(v) if isinstance(v, (int, float)) else float("nan")


def _design(panel: pl.DataFrame, coef: Coefficient, baseline: str) -> pl.DataFrame:
    """The rows this coefficient is fitted on: its position, and both columns present."""
    df = panel.filter(pl.col(baseline).is_not_null() & pl.col(coef.signal).is_not_null()
                      & pl.col("target").is_not_null())
    if coef.position is not None:
        df = df.filter(pl.col("pos") == coef.position)
    return df


def _contributions(y: np.ndarray, X: np.ndarray, j: int,
                   cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficient `j` rewritten as a mean over rows, so a bootstrap of means resamples it.

    `b = (X'X)^-1 X'y`, so `b_j = b_j + [(X'X)^-1 X'e]_j` for the fitted residuals `e`, and
    that bracket is **exactly zero** -- `X'e = 0` is the normal equation. Writing `h_i` for
    row `i`'s share of it, `[(X'X)^-1 x_i e_i]_j`, the numbers

        d_i = b_j + K * n_k(i) * h_i

    have a cluster mean of `b_j + K * S_k` for each cluster's `S_k = sum_{i in k} h_i`, and a
    grand mean over the `K` clusters of exactly `b_j`. So `summarise` handed these returns the
    OLS coefficient as its `mean`, not an approximation of it, and
    `tests/unit/test_fit_corrections.py` asserts that equality rather than trusting this
    paragraph.

    **The interval is then the cluster-robust one.** Resampling `K` cluster means with
    replacement gives a spread of `sum_k (S_k - Sbar)^2`, which is the CR0 sandwich variance
    for this coefficient -- the same quantity a `cluster()` standard error reports, arrived at
    by resampling rather than by a formula. With no cluster the units are rows and it is the
    HC0 robust one, by the same algebra with `n_k = 1`.

    **Where this differs from resampling clusters and refitting.** A refit varies `(X'X)`
    between draws and this does not; the estimator is linear in `y` so the difference is
    second order, but it is not nothing, and `docs/talent-cv.md` records a two-stage estimator
    where refitting inside the resample moved an interval by 54%. That one was a curve fitted
    and then a dispersion taken around it -- nonlinear, where linearising throws away the
    first-order term. This is one OLS. `refit_interval` runs the refit anyway and the CLI
    prints both, because a reader should not have to take the previous sentence on trust.

    Returns `(d, cluster)`, the second unchanged, so a caller can hand both to one frame.
    """
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ X.T @ y
    resid = y - X @ b
    h = (X * resid[:, None]) @ xtx_inv[:, j]
    ids, inverse, counts = np.unique(cluster, return_inverse=True, return_counts=True)
    return b[j] + len(ids) * counts[inverse] * h, ids


def _cluster_key(df: pl.DataFrame, cluster: Sequence[str] | None) -> np.ndarray:
    """One opaque label per cluster, or one per row when there is no cluster."""
    if not cluster:
        return np.arange(df.height)
    joined = df.select(pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cluster],
                                     separator="␟"))
    return joined.to_series().to_numpy()


def fit_one(panel: pl.DataFrame, coef: Coefficient, *, baseline: str = HEADLINE_BASELINE,
            cluster: Sequence[str] | None = SEASON_CLUSTER, seed: int = 0,
            population: str = "") -> Fit | None:
    """Refit one coefficient. `None` when the panel holds too little to fit at all.

    The model is the shipped comment's, with the axis corrected: `target ~ baseline + signal`,
    where `target` is points per team game and `baseline` is the column the correction is
    applied to rather than one of the two it is built from.
    """
    df = _design(panel, coef, baseline)
    if df.height < 10:
        return None
    y = df["target"].to_numpy().astype(float)
    sig = df[coef.signal].to_numpy().astype(float)
    base = df[baseline].to_numpy().astype(float)
    X = np.column_stack([np.ones(df.height), base, sig])
    if np.linalg.matrix_rank(X) < X.shape[1]:
        # A signal that never varies inside this slice carries no coefficient, and `pinv`
        # would hand back a number for it anyway. Refusing is the honest answer.
        return None
    b = np.linalg.pinv(X.T @ X) @ X.T @ y
    keys = _cluster_key(df, cluster)
    d, ids = _contributions(y, X, 2, keys)
    # The cluster columns travel to `summarise` by name, so what it groups on is the argument
    # this function was given -- `SEASON_CLUSTER` unless a caller says otherwise -- rather
    # than a label this module folded them into first. `_cluster_key` builds the same grouping
    # a second time only to scale the contributions, and the two agreeing is asserted at
    # `test_the_interval_is_centred_on_the_ols_coefficient`.
    paired = (df.select(list(cluster)).with_columns(pl.Series("diff", d)) if cluster
              else pl.DataFrame({"diff": d}))
    s = summarise(paired, cluster=cluster, seed=seed)
    return Fit(coefficient=coef, baseline=baseline, beta=float(b[2]),
               beta_baseline=float(b[1]), lo=float(s["lo"]), hi=float(s["hi"]),
               se=float(s["se"]), mde=float(s["mde"]), n=int(df.height),
               n_signal=int((sig != 0.0).sum()), clusters=len(ids),
               population=population or _population(df, coef, baseline))


def _population(df: pl.DataFrame, coef: Coefficient, baseline: str) -> str:
    """The sentence every fitted number ships with. Named, not implied."""
    years = sorted(df["season"].unique().to_list())
    span = f"{years[0]}-{years[-1]}" if years else "no seasons"
    who = coef.position or "every drafted position"
    return (f"{df.height} player-seasons ({who}), outcome seasons {span}, "
            f"{len(years)} season clusters, baseline {baseline}")


def refit_interval(panel: pl.DataFrame, coef: Coefficient, *,
                   baseline: str = HEADLINE_BASELINE,
                   cluster: Sequence[str] | None = SEASON_CLUSTER,
                   draws: int = 1000, seed: int = 0) -> tuple[float, float]:
    """The cross-check `_contributions` promises: resample clusters and refit inside each draw.

    Not the published interval -- #45 fixed that as `summarise`'s, and a second bootstrap that
    happened to agree would be exactly the coincidence `minimum_detectable_effect` warns about
    when it insists the SE under the MDE is the interval's own. This exists so the linearised
    interval's one approximation is measured rather than argued, which is what
    `docs/talent-cv.md` says the repo learned to do the hard way.
    """
    df = _design(panel, coef, baseline)
    if df.height < 10:
        return float("nan"), float("nan")
    keys = _cluster_key(df, cluster)
    ids = np.unique(keys)
    blocks = [np.flatnonzero(keys == i) for i in ids]
    y = df["target"].to_numpy().astype(float)
    X = np.column_stack([np.ones(df.height), df[baseline].to_numpy().astype(float),
                         df[coef.signal].to_numpy().astype(float)])
    rng = np.random.default_rng(seed)
    out = []
    for pick in rng.integers(0, len(blocks), size=(draws, len(blocks))):
        rows = np.concatenate([blocks[p] for p in pick])
        xs, ys = X[rows], y[rows]
        if np.linalg.matrix_rank(xs) < xs.shape[1]:
            continue
        out.append((np.linalg.pinv(xs.T @ xs) @ xs.T @ ys)[2])
    if not out:
        return float("nan"), float("nan")
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# --- walk-forward -------------------------------------------------------------


class HeldOut(NamedTuple):
    """One season scored by a fit that never saw it.

    Split out from `walk_forward` so that `shrink_curve` scores the *same* held-out rows
    from the *same* splits rather than walking the panel a second time. Two walks would be
    two chances to disagree about which season was held out, and the arms would then not be
    paired -- `docs/method.md` rule 7, which is the whole reason the arms below share
    `plain`.
    """
    season: int
    n_past: int
    plain: np.ndarray       # the no-correction prediction on this season's rows
    fitted: np.ndarray      # target ~ baseline + signal, both fitted on strictly earlier rows
    signal: np.ndarray
    actual: np.ndarray

    def mae(self, pred: np.ndarray) -> float:
        return float(np.abs(pred - self.actual).mean())

    def applied(self, beta: float) -> np.ndarray:
        """`plain`, plus `beta` times the signal. What the board does, at any `beta`."""
        return self.plain + beta * self.signal


def _held_out(panel: pl.DataFrame, coef: Coefficient, baseline: str):
    """Every season in turn, fitted only on strictly earlier ones."""
    df = _design(panel, coef, baseline)
    for yr, past, now in expanding_seasons(df, min_past=30):
        latest_fitted = _as_int(past["season"].max())
        scored = (_as_int(now["season"].min()), _as_int(now["season"].max()))
        # GUARD walk-forward-never-sees-its-own-season: `past` is what the coefficient is
        # fitted on and `now` is what it is scored on, and a single row of `now`'s season
        # inside `past` is the leak that `docs/method.md` rule #2 records this repo reading
        # at 7.4 se. Asserted on the split itself rather than on the numbers downstream,
        # because a leak shows up there as a *better* result and nothing looks wrong.
        if latest_fitted >= yr or scored != (yr, yr):
            raise ValueError(
                f"walk-forward split leaks: fitting on seasons up to {latest_fitted} "
                f"to score {yr}, whose rows span {scored}")
        # /GUARD
        yp = past["target"].to_numpy().astype(float)
        bp = past[baseline].to_numpy().astype(float)
        sp = past[coef.signal].to_numpy().astype(float)
        plain = np.linalg.lstsq(np.column_stack([np.ones(past.height), bp]), yp,
                                rcond=None)[0]
        full = np.linalg.lstsq(np.column_stack([np.ones(past.height), bp, sp]), yp,
                               rcond=None)[0]
        bn = now[baseline].to_numpy().astype(float)
        sn = now[coef.signal].to_numpy().astype(float)
        yield HeldOut(season=yr, n_past=past.height,
                      plain=plain[0] + plain[1] * bn,
                      fitted=full[0] + full[1] * bn + full[2] * sn,
                      signal=sn, actual=now["target"].to_numpy().astype(float))


def walk_forward(panel: pl.DataFrame, coef: Coefficient, *,
                 baseline: str = HEADLINE_BASELINE) -> pl.DataFrame:
    """Held-out error per season, fitting only on strictly earlier ones.

    Three arms on the same held-out rows, which is what makes the comparison worth reading:

      * `plain`   -- `target ~ baseline`, no correction term at all. The null.
      * `fitted`  -- `target ~ baseline + signal`, both coefficients fitted on `past`.
      * `shipped` -- `plain` plus the *shipped* constant times the signal. The arm that asks
                     the question #48 will act on: does the number the board already applies
                     help on a season it never saw?

    `shipped` is scored on `plain`'s intercept and slope deliberately. The board applies its
    constant to a projection that was not refitted around it, so an arm that re-estimated the
    intercept to suit the constant would be scoring a model the board does not run.
    """
    rows = []
    for h in _held_out(panel, coef, baseline):
        preds = {"plain": h.plain, "fitted": h.fitted,
                 "shipped": h.applied(coef.shipped)}
        rows.append({"season": h.season, "n": h.actual.size, "n_past": h.n_past,
                     **{f"mae_{k}": h.mae(v) for k, v in preds.items()}})
    return pl.DataFrame(rows)


def shrink_curve(panel: pl.DataFrame, coef: Coefficient, *,
                 baseline: str = HEADLINE_BASELINE,
                 shrinks: Sequence[float] = SHRINKS) -> pl.DataFrame:
    """Held-out MAE of the shipped constant applied at each shrink factor.

    **This is the arm #186 needed and `walk_forward` did not have.** That function scores
    two points of one line -- the constant applied whole (`shipped`) and not applied at all
    (`plain`) -- and a ticket asking whether a correction should be *kept at a measured
    shrink* cannot be answered from two points without assuming the shape between them.
    `lambda = 0` reproduces `plain` exactly and `lambda = 1` reproduces `shipped` exactly, by
    construction rather than by coincidence, and `tests/unit/test_fit_corrections.py` holds
    both equalities.

    The shrink is on the **shipped** constant, not on a refit, for `walk_forward`'s reason:
    the question is what the board should apply, and the board applies a constant to a
    projection that was not refitted around it. A negative shrink is deliberately not on the
    sweep -- flipping a coefficient's sign is a refit, not a shrinkage, and #48's rule for a
    sign-reversed coefficient is that it is zeroed rather than reversed.

    Returns one row per shrink: the mean held-out MAE, the seasons it was averaged over, and
    how many of them it beat `lambda = 0` on. Both are reported because either alone
    misleads -- a mean can be carried by one season, and a season count says nothing about
    size.
    """
    held = list(_held_out(panel, coef, baseline))
    if not held:
        return pl.DataFrame(schema={"shrink": pl.Float64, "mae": pl.Float64,
                                    "seasons": pl.Int64, "beats_none": pl.Int64})
    none = [h.mae(h.applied(0.0)) for h in held]
    rows = []
    for lam in shrinks:
        per_season = [h.mae(h.applied(lam * coef.shipped)) for h in held]
        rows.append({"shrink": float(lam),
                     "mae": float(np.mean(per_season)),
                     "seasons": len(per_season),
                     "beats_none": int(sum(a < b for a, b in zip(per_season, none, strict=True)))})
    return pl.DataFrame(rows)


def best_shrink(curve: pl.DataFrame) -> float:
    """The shrink factor with the lowest held-out MAE. NaN for an empty curve.

    Ties go to the *smaller* shrink, which is the conservative direction: two shrinks that
    score the same are two models the run cannot tell apart, and the one that applies less
    of an unreproduced constant is the one that claims less.
    """
    if curve.is_empty():
        return float("nan")
    ordered = curve.sort(["mae", "shrink"])
    return float(ordered["shrink"][0])


# --- the panel ----------------------------------------------------------------


def _keyed(col: str) -> pl.Expr:
    from hub.names import player_key
    return pl.col(col).map_elements(player_key, return_dtype=pl.Utf8).alias("player_key")


def realised_per_team_game(season: int, cache=None) -> pl.DataFrame:  # pragma: no cover - network
    """What a player actually delivered across `season`, per team game."""
    from hub.fetch import nflverse
    cols = ("player_id", "player_display_name", "position", "season", "week",
            "season_type", "fantasy_points_ppr")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache).filter(
        pl.col("season_type") == "REG")
    return (w.group_by("player_display_name")
             .agg(pl.col("fantasy_points_ppr").sum().alias("total"))
             .select(_keyed("player_display_name"),
                     (pl.col("total") / TEAM_GAMES).alias("target")))


def week_one_designation(season: int, cache=None) -> pl.DataFrame:  # pragma: no cover - network
    """A 0/1 Out-or-Doubtful flag off `season`'s week-1 injury report.

    The week-1 report is the shipped comment's own instrument and its own caveat: it is the
    closest historical analogue to an August designation, and `docs/durability.md` measures
    the population gap that makes QUESTIONABLE untransferable (12.6% of an August board
    against 2.9% at week 1). Nothing here re-opens that. QUESTIONABLE is measured and left
    out of `PRICED_DESIGNATIONS` for the same reason it is left out of `INJURY_BETA`.
    """
    from hub.fetch import nflverse
    inj = nflverse.load("injuries", [season], cache=cache).filter(pl.col("week") == 1)
    return (inj.select(_keyed("full_name"),
                       pl.col("report_status").fill_null("").str.to_uppercase().alias("st"))
               .with_columns(pl.col("st").is_in(sorted(PRICED_DESIGNATIONS))
                               .cast(pl.Float64).alias("designation"))
               .group_by("player_key").agg(pl.col("designation").max()))


def season_rows(outcome: int, cache=None) -> pl.DataFrame:  # pragma: no cover - network
    """One outcome season's rows: prior-season signals and baselines, this season's result.

    Every signal comes from the **shipped** function that computes it on the live board --
    `regression.td_luck` and `durability.games_missed`, eligibility filters and all -- so the
    fit runs on the population the correction is applied to rather than on a wider one
    assembled here. A harness that rebuilt the signal would be fitting a second definition of
    it, which is the shape `docs/next.md` names as how two implementations drift.
    """
    from hub.config import drafted_positions
    from hub.draft import board, durability, regression
    prior = outcome - 1
    xp = (board.expected_points(prior)
          .filter(pl.col("position").is_in(list(drafted_positions())))
          .select(_keyed("full_name"), pl.col("position").alias("pos"),
                  pl.col("xfp_per_game").alias("base_xfp"),
                  (pl.col("fp") / pl.col("games")).alias("base_carry"))
          .unique("player_key"))
    tl = (regression.td_luck(regression.prior_season(prior, cache=cache))
          .select(_keyed("player"), "td_luck").unique("player_key"))
    gm = (durability.games_missed(durability.prior_season(prior, cache=cache))
          .select(_keyed("player"), "missed").unique("player_key"))
    return (xp.join(tl, on="player_key", how="left")
              .join(gm, on="player_key", how="left")
              .join(week_one_designation(outcome, cache=cache), on="player_key", how="left")
              .join(realised_per_team_game(outcome, cache=cache), on="player_key", how="left")
              .with_columns(
                  pl.lit(outcome, dtype=pl.Int64).alias("season"),
                  # Absent from the outcome season means he scored nothing, which is the
                  # truth about him and the busts. `hub.draft.calibrate` records what
                  # dropping them does: the season reads as more predictable than it was.
                  pl.col("target").fill_null(0.0),
                  # Absent from a week-1 report means he was not designated. Unlike the two
                  # trait columns, whose null is "no prior season" and must stay a null.
                  pl.col("designation").fill_null(0.0),
                  ((pl.col("base_xfp") + pl.col("base_carry")) / 2.0).alias("base_blend")))


def build_panel(seasons: Sequence[int] = DEFAULT_SEASONS,
                cache=None) -> pl.DataFrame:  # pragma: no cover - network
    """The panel every fit here runs on, one row per (player, outcome season)."""
    return pl.concat([season_rows(int(y), cache=cache) for y in seasons])


# --- reporting ----------------------------------------------------------------


def report_lines(panel: pl.DataFrame, *, baselines: Sequence[str] = BASELINES,
                 seed: int = 0) -> list[str]:
    """The whole finding, as lines. Pure, so a test can read it without a CLI."""
    out: list[str] = []
    for coef in COEFFICIENTS:
        out.append(f"\n  {coef.name} -- shipped {coef.shipped:+.3f} "
                   f"in {coef.declared_in}.{coef.constant}[{coef.key!r}], "
                   f"{coef.documented_in}")
        for baseline in baselines:
            fit = fit_one(panel, coef, baseline=baseline, seed=seed)
            if fit is None:
                out.append(f"    {baseline:<11} not fittable on this panel")
                continue
            mark = "  <- headline" if baseline == HEADLINE_BASELINE else ""
            power = "resolved" if fit.resolved else "UNDERPOWERED"
            out.append(
                f"    {baseline:<11} {fit.beta:+.4f} "
                f"[{fit.lo:+.4f}, {fit.hi:+.4f}]  se {fit.se:.4f}  MDE {fit.mde:.4f}  "
                f"beta(proj) {fit.beta_baseline:+.3f}  {fit.disposition}, "
                f"gap {fit.gap:.4f} {power}{mark}")
            if baseline == HEADLINE_BASELINE:
                lo, hi = refit_interval(panel, coef, baseline=baseline, seed=seed)
                out.append(f"    {'':<11} population: {fit.population}; "
                           f"n {fit.n}, carrying the signal {fit.n_signal}")
                out.append(f"    {'':<11} refit-in-the-resample cross-check "
                           f"[{lo:+.4f}, {hi:+.4f}]")
                if fit.at_a_bound:
                    out.append(f"    {'':<11} NOTE: the shipped value sits on an interval "
                               f"edge -- read docs/pick-noise.md and #155 before quoting it")
        wf = walk_forward(panel, coef)
        if not wf.is_empty():
            means = {c: _as_float(wf[f"mae_{c}"].mean())
                     for c in ("plain", "fitted", "shipped")}
            best = min(means, key=lambda c: means[c])
            wins = int((wf["mae_shipped"] < wf["mae_plain"]).sum())
            out.append(f"    walk-forward  held-out MAE over {wf.height} seasons: "
                       + ", ".join(f"{c} {means[c]:.4f}" for c in means)
                       + f"; best {best}; shipped beats no-correction in "
                         f"{wins}/{wf.height} seasons")
        curve = shrink_curve(panel, coef)
        if not curve.is_empty():
            out.append("    shrink sweep  "
                       + ", ".join(f"x{r['shrink']:.2f} {r['mae']:.4f} "
                                   f"({r['beats_none']}/{r['seasons']})"
                                   for r in curve.iter_rows(named=True))
                       + f"; best x{best_shrink(curve):.2f}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.fit_corrections",
        description="Refit the five board corrections against the column they correct.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS),
                    help="outcome seasons; each is scored against the one before it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit", action="store_true", help="run the fits (needs nflverse)")
    a = ap.parse_args(argv)
    if not a.fit:
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    try:
        panel = build_panel(seasons)
    except Exception as e:                       # pragma: no cover - network
        return unavailable("hub.draft.fit_corrections",
                           "nflverse ff_opportunity, player stats and injuries", e)
    print(f"  panel: {panel.height} player-seasons over {len(seasons)} outcome seasons")
    print("  target: points per team game (total/17), the rate hub.draft.season draws every "
          "week from")
    for line in report_lines(panel, seed=a.seed):
        print(line)
    print("\n  Dispositions are inputs to #48. Nothing here changes a shipped constant.")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
