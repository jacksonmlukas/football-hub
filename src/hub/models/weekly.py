"""The **Weekly projection**: a multiplier on **Usage**, and a touchdown regression.

Everything this repo projects today is one season-long per-game mean applied flat to all
seventeen weeks. This is the week-specific layer, and its shape was fixed by measurement
rather than chosen -- see `docs/weekly-screen.md`.

    weekly Usage = season-to-date Usage x exp(coef . snap_trend)
    weekly TDs   = weekly yards x the POSITION's touchdown rate
    weekly points = league scoring applied to the counts

**`f = 1` is the incumbent.** With `coef = 0` the multiplier is one and the projection is
exactly the flat one, so the null is the identity and this cannot be much worse than what it
adjusts. It also means a failed gate leaves one interpretable object -- a multiplier printable
per player per week -- rather than a parallel model to debug.

**The multiplier acts on counts, never on points.** Points are rebuilt from adjusted Usage
through `components.SCORING`, which is the premise of the whole approach end to end.

WHY THESE TWO TERMS AND NOTHING ELSE. Nine features were screened; two survived a joint screen,
and screening them against the counts rather than the total showed they do not overlap:

> **Restated 2026-09-10 -- the family is eight.** #170 removed `wind` from the screen: it is
> game-time observed weather, read off the schedule as recorded conditions and screened as a
> week-w feature with a pre-stated sign, which is `docs/method.md` rule 2. It was killed at
> 4/5 seasons and never reached the joint screen, so **neither of the two terms below moves**
> and no figure in this table is affected -- what moves is the count of features tried, which
> is the size of the multiple-comparison family the two survivors came out of.
> `docs/weekly-screen.md` carries the restatement.


                  targets  receptions  carries  attempts  touchdowns
    snap_trend     +0.094      +0.077   +0.062    +0.032   +0.018 killed
    td_rate_prior  +0.008      -0.000   -0.005    +0.005   -0.120

The snap trend moves every volume count and no touchdowns; the prior touchdown rate moves
touchdowns at -14 se and no volume count. So one goes in the Usage multiplier and the other in
the touchdown term, and nothing crosses between them.

WHAT IS DELIBERATELY ABSENT. **Efficiency is not projected.** Yards per carry persists at
r = 0.108 year over year and touchdowns per yard at ~0, so a model predicting this week's
efficiency is predicting noise. Each player's efficiency is held at his own recent rate and
only his *opportunity* moves. **Spread is not projected either**: `sd = k*sqrt(mu)` stands,
per ADR-0012, which measured per-player volatility beyond the positional constant at +/-9.3%
and not estimable. This module produces a mean; `predict.moments` turns it into a distribution.

    uv run python -m hub.models.weekly --fit
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NamedTuple, cast

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.config import FANTASY_WEEKS
from hub.models import predict
from hub.models.components import SCORING, td_rate
from hub.models.experiment import expanding_seasons
from hub.models.panel import (
    MIN_GAMES_BEFORE,
    SEASONS,
    TREND_MIN_WEEK,
    PanelSpec,
    build_panel,
)
from hub.models.scoring_rules import crps_from_quantiles, normal_quantile, quantile_levels

NOT_FITTED_BECAUSE = (
    "the Weekly projection. MULTIPLIER_LO/HI bound a fitted multiplier, MIN_UNITS is a volume "
    "floor below which a per-unit efficiency rate is noise, TAIL_Q is the slice the shrinkage "
    "experiment scores and the two SHRINK_GRIDs are search grids -- all settings. The "
    "multiplier's coefficients are fitted at run time from the panel and never frozen into "
    "the module, the same shape as hub.models.injury. See docs/weekly-screen.md "
)

# The multiplier is bounded. A snap share that doubled is real information; a multiplier of 4
# on a player's target count is an extrapolation past anything in the fit, and the cost of
# being wrong upward on a lineup is asymmetric -- you start him.
MULTIPLIER_LO, MULTIPLIER_HI = 0.6, 1.6

# Usage counts the multiplier applies to, and the phase each one scores through.
VOLUME: tuple[str, ...] = ("targets", "carries", "attempts")

# (yards, units) pairs whose ratio is the efficiency held at the player's own rate.
EFFICIENCY_PAIRS: tuple[tuple[str, str], ...] = (
    ("receiving_yards", "receptions"), ("rushing_yards", "carries"),
    ("passing_yards", "attempts"))

# Accumulated units -- games played times units per game -- below which a per-unit efficiency
# rate is a handful of plays and the ratio is noise. It is a *total*, not a per-game figure:
# comparing a per-game mean against this sent almost every receiver to the pooled rate, since
# nobody catches eight passes a game, and under-projected the good ones by 0.66 points a week.
MIN_UNITS = 8.0


def fit_multiplier(train: pl.DataFrame, component: str, *,
                   feature: str = "snap_trend") -> float:
    """`coef` in `count = expected * exp(coef * feature)`, by least squares on the log ratio.

    The log ratio rather than the raw difference so the term is multiplicative and scale-free:
    a snap-share jump should move a twelve-target receiver by more targets than a four-target
    one, and by the same *fraction*. Fitting the difference instead would apply a receiver's
    absolute gain to a tight end.

    Returns 0.0 -- the identity -- when there is nothing to fit, which keeps the incumbent
    rather than inventing a coefficient from a handful of rows.
    """
    d = train.drop_nulls([component, f"{component}_prior", feature]).filter(
        (pl.col(f"{component}_prior") > 0) & (pl.col("week") >= TREND_MIN_WEEK))
    if d.height < 100:
        return 0.0
    y = np.log((d[component].to_numpy().astype(float) + 1.0)
               / (d[f"{component}_prior"].to_numpy().astype(float) + 1.0))
    x = d[feature].to_numpy().astype(float)
    if x.std() == 0:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def multiplier(feature: np.ndarray, coef: float, *, lo: float = MULTIPLIER_LO,
               hi: float = MULTIPLIER_HI) -> np.ndarray:
    """`exp(coef * feature)`, clipped, and exactly 1.0 wherever the feature is missing."""
    x = np.nan_to_num(np.asarray(feature, dtype=float), nan=0.0)
    return np.clip(np.exp(coef * x), lo, hi)


# Grids for the two shrinkage constants, searched on TRAINING seasons only. Both are in the
# units of the thing they temper: games for a volume prior, accumulated units for an
# efficiency rate. Zero is in each grid so "no shrinkage" is a candidate the fit can pick,
# which is what makes this an experiment rather than an assumption.
VOLUME_SHRINK_GRID: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EFF_SHRINK_GRID: tuple[float, ...] = (0.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)


class Shrink(NamedTuple):
    """How hard to pull a thin sample toward the target, and what the target is."""
    volume_k: float
    eff_k: float
    volume_mean: dict[tuple[str, str], float]     # (position, component) -> mean per game
    eff_mean: dict[tuple[str, str], float]        # (position, yards) -> yards per unit
    # (position, component) -> (a, b, lo, hi) in log(1 + count) = a + b*log(rank). When
    # present the pull is toward what the player's *preseason* rank implies rather than
    # toward his position's average, which is the market-implied variant.
    consensus_prior: dict[tuple[str, str], tuple[float, float, float, float]] | None = None


def fit_consensus_prior(train: pl.DataFrame,
                     ) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """`log(1 + count per game) = a + b*log(preseason rank)`, per position, per Usage count.

    The same functional form as `hub.models.volume.VOLUME_CURVE` -- log-log in the pick, which
    `docs/volume-model.md` fitted because volume is roughly a power law in the market's
    opinion and strictly non-negative -- but **refitted here on training seasons only**. The
    shipped constants are frozen at a 2022-25 fit, which is exactly the span held out, so
    importing them would evaluate a prior on the seasons it was fitted on.

    Against the board's preseason `ecr` rather than an actual draft pick, because that is what
    a historical board carries. The rank is clamped to the fitted range, which is what stops a
    599th-ranked player being extrapolated off the end of a log curve.
    """
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {}
    d = train.drop_nulls("preseason_ecr").filter(pl.col("preseason_ecr") > 0)
    for pos in sorted(set(d["position"].to_list())):
        sub = d.filter(pl.col("position") == pos)
        if sub.height < 100:
            continue
        rank = sub["preseason_ecr"].to_numpy().astype(float)
        lo, hi = float(rank.min()), float(rank.max())
        x = np.log(rank)
        for c in (*VOLUME, "receptions"):
            y = np.log1p(np.clip(sub[c].to_numpy().astype(float), 0.0, None))
            if x.std() == 0:
                continue
            a, b = float(np.polyfit(x, y, 1)[1]), float(np.polyfit(x, y, 1)[0])
            out[(pos, c)] = (a, b, lo, hi)
    return out


def consensus_target(df: pl.DataFrame, col: str,
                  prior: dict[tuple[str, str], tuple[float, float, float, float]],
                  ) -> np.ndarray:
    """What each player's preseason rank implies for `col`, per game. NaN where unfitted."""
    ranks = df["preseason_ecr"].to_numpy().astype(float)
    out = np.full(len(ranks), np.nan)
    for i, (pos, r) in enumerate(zip(df["position"].to_list(), ranks, strict=True)):
        fit = prior.get((pos, col))
        if fit is None or not np.isfinite(r) or r <= 0:
            continue
        a, b, lo, hi = fit
        out[i] = max(float(np.expm1(a + b * np.log(min(max(r, lo), hi)))), 0.0)
    return out


def _pos_means(train: pl.DataFrame) -> tuple[dict, dict]:
    """Positional means for every Usage count and every efficiency rate."""
    vol, eff = {}, {}
    for pos in sorted(set(train["position"].to_list())):
        d = train.filter(pl.col("position") == pos)
        for c in (*VOLUME, "receptions"):
            v = d[c].mean()
            vol[(pos, c)] = float(cast(float, v)) if v is not None else 0.0
        for yards, units in EFFICIENCY_PAIRS:
            u, y = float(d[units].sum() or 0.0), float(d[yards].sum() or 0.0)
            eff[(pos, yards)] = y / u if u > 0 else 0.0
    return vol, eff


# The share of the projection's top end the tail objective scores. A waiver pick is roughly
# the best of a few hundred, so a decile is the coarsest slice that is still about the top.
TAIL_Q = 0.90


def fit_shrink(train: pl.DataFrame, coefs: dict[str, float], *,
               objective: str = "mae", target: str = "position") -> Shrink:
    """Fit both constants by grid search on `train`. Zero is in both grids.

    Fitted rather than chosen, and on training seasons only, because a shrinkage picked after
    seeing a held-out result is the thing `docs/method.md` rule #1 exists to stop.

    **Two objectives, and which one you pick is the whole experiment.**

    `mae` minimises the projection's mean absolute error. It is what
    `docs/weekly-projection-plan.md` pre-registered, and it fits `volume_k = 0` in every
    season -- no shrinkage at all -- because the projection is *already unbiased at every
    sample size*. Mean error cannot see a winner's curse: the curse lives at the maximum over
    hundreds of candidates and the mean is dominated by the bulk. The pre-registered objective
    was blind to the defect the shrinkage was meant to fix.

    `target="market"` regresses toward what each player's **preseason** consensus rank implies
    rather than toward his position's average -- refitted per fold, never imported from the
    frozen `volume.VOLUME_CURVE`, which was fitted on the seasons held out here.

    `tail` minimises the absolute bias among the **top decile by projection** in training,
    which is where a waiver decision reads. It was specified *after* seeing `mae` return a
    null, so it is exploratory and is reported as such -- but it is still fitted on training
    seasons only and never on the held-out ones.
    """
    vol_mean, eff_mean = _pos_means(train)
    prior = fit_consensus_prior(train) if target.startswith("market") else None
    if target == "market-only":
        return Shrink(PURE_MARKET_K, PURE_MARKET_K, vol_mean, eff_mean, prior)
    actual = train["fantasy_points_ppr"].to_numpy().astype(float)
    best = Shrink(0.0, 0.0, vol_mean, eff_mean, prior)
    best_loss = float("inf")
    for vk in VOLUME_SHRINK_GRID:
        for ek in EFF_SHRINK_GRID:
            cand = Shrink(vk, ek, vol_mean, eff_mean, prior)
            mu = project(train, coefs, shrink=cand)["mu"].to_numpy().astype(float)
            if objective == "mae":
                loss = float(np.abs(mu - actual).mean())
            else:
                cut = float(np.quantile(mu, TAIL_Q))
                top = mu >= cut
                loss = abs(float((mu[top] - actual[top]).mean())) if top.any() else np.inf
            if loss < best_loss:
                best, best_loss = cand, loss
    return best


# A `volume_k` this large makes `n/(n+k)` ~ 0 for any real sample, so the projection becomes
# the market prior and nothing else. Not a candidate in any grid -- it exists so a run can ask
# "how much of this gain is ours and how much is the market's?" and get a number.
PURE_MARKET_K = 1e9


def _shrunk(df: pl.DataFrame, col: str, k: float, means: dict, key: str,
            prior: dict | None = None) -> np.ndarray:
    """`n/(n+k)` of his own, the rest of the target. `k = 0` returns his own untouched.

    The target is his position's average, or -- in the market-implied variant -- what his
    preseason rank implies, falling back to the positional average where the rank is missing
    or the position was never fitted. A WR1 and a WR5 do not regress toward the same place,
    which is the diagnosis `component-projection.md` made of the positional-mean version.
    """
    own = np.nan_to_num(df[f"{col}_prior"].to_numpy().astype(float), nan=0.0)
    if k <= 0:
        return own
    n = np.nan_to_num(df["games_before"].to_numpy().astype(float), nan=0.0)
    target = np.array([means.get((p, key), 0.0) for p in df["position"].to_list()])
    if prior is not None and "preseason_ecr" in df.columns:
        implied = consensus_target(df, key, prior)
        target = np.where(np.isnan(implied), target, implied)
    w = n / (n + k)
    return w * own + (1.0 - w) * target


def efficiency(df: pl.DataFrame, yards: str, units: str,
               shrink: Shrink | None = None) -> np.ndarray:
    """Yards per unit from the player's own prior weeks, tempered toward the sample.

    Held rather than projected: year over year, yards per carry persists at r = 0.108 and
    yards per target at 0.369, so there is nothing here a week-level model could add.

    Without `shrink` this is a hard threshold at `MIN_UNITS` accumulated units -- his own rate
    above it, the pooled rate below. With one it is the smooth version, `n/(n+k)` of his own,
    which subsumes the threshold and is what the shrinkage experiment tests.
    """
    y = df[f"{yards}_prior"].to_numpy().astype(float)
    u = df[f"{units}_prior"].to_numpy().astype(float)
    games = df["games_before"].to_numpy().astype(float)
    acc = np.nan_to_num(u * games, nan=0.0)
    pooled = float(np.nansum(y) / np.nansum(u)) if np.nansum(u) > 0 else 0.0
    own = np.nan_to_num(np.divide(y, np.maximum(u, 1e-9)), nan=pooled)
    if shrink is None or shrink.eff_k <= 0:
        # `eff_k = 0` must be the *shipped* estimator, not "always his own rate", or the zero
        # point of the grid would be a third model and the fit could not decline to change
        # anything.
        return np.where(acc >= MIN_UNITS, own, pooled)
    target = np.array([shrink.eff_mean.get((p, yards), pooled)
                       for p in df["position"].to_list()])
    w = acc / (acc + shrink.eff_k)
    return w * own + (1.0 - w) * target


def project(now: pl.DataFrame, coefs: dict[str, float],
            *, shrink: Shrink | None = None) -> pl.DataFrame:
    """Weekly Usage, yards and touchdowns, and the points they add up to.

    Touchdowns come from projected yards times the **position's** rate, never the player's
    own: `docs/component-projection.md` measured a player's own touchdown rate as carrying no
    information beyond his yardage (year-over-year r of -0.004 receiving, -0.030 rushing), and
    the weekly screen found the same thing from the other direction -- a high prior rate
    predicts *fewer* touchdowns at -14 se, which is what over-shrinking would not do.
    """
    # The trend is dark before TREND_MIN_WEEK, so the multiplier is exactly 1 there. Gating
    # it here as well as in the fit matters: `snap_trend` is non-null from week 7 (it needs
    # six prior weeks), and week 7 is a week the screen never established it in.
    early = (now["week"].to_numpy().astype(int) < TREND_MIN_WEEK)
    trend = np.where(early, 0.0, np.nan_to_num(now["snap_trend"].to_numpy(), nan=0.0))
    m = {c: multiplier(trend, coefs.get(c, 0.0)) for c in VOLUME}
    pos = now["position"].to_list()

    def prior(col: str) -> np.ndarray:
        if shrink is None:
            return np.nan_to_num(now[f"{col}_prior"].to_numpy().astype(float), nan=0.0)
        return _shrunk(now, col, shrink.volume_k, shrink.volume_mean, col,
                       shrink.consensus_prior)

    tgt = prior("targets") * m["targets"]
    car = prior("carries") * m["carries"]
    att = prior("attempts") * m["attempts"]

    catch = np.clip(np.nan_to_num(
        prior("receptions") / np.maximum(prior("targets"), 1e-9), nan=0.0), 0.0, 1.0)
    rec = tgt * catch

    rec_y = rec * efficiency(now, "receiving_yards", "receptions", shrink)
    rush_y = car * efficiency(now, "rushing_yards", "carries", shrink)
    pass_y = att * efficiency(now, "passing_yards", "attempts", shrink)

    rec_td = rec_y * np.array([td_rate(p, "rec") for p in pos])
    rush_td = rush_y * np.array([td_rate(p, "rush") for p in pos])
    pass_td = pass_y * np.array([td_rate(p, "pass") if p == "QB" else 0.0 for p in pos])

    # Turnovers are carried at the player's own recent rate rather than modelled: they are
    # rare, and `components.SCORING` prices both at -2. Omitting them is not a rounding error --
    # it over-projected quarterbacks by +1.44 points a week, which is what an interception a
    # game costs.
    ints = np.nan_to_num(now["passing_interceptions_prior"].to_numpy().astype(float), nan=0.0)
    fum = np.nan_to_num(now["fumbles_lost_total_prior"].to_numpy().astype(float), nan=0.0)

    mu = (SCORING["interceptions"] * ints + SCORING["fumbles_lost"] * fum
          + SCORING["receptions"] * rec
          + SCORING["receiving_yards"] * rec_y + SCORING["receiving_tds"] * rec_td
          + SCORING["rushing_yards"] * rush_y + SCORING["rushing_tds"] * rush_td
          + SCORING["passing_yards"] * pass_y + SCORING["passing_tds"] * pass_td)

    return now.with_columns(
        pl.Series("targets_hat", tgt), pl.Series("receptions_hat", rec),
        pl.Series("carries_hat", car), pl.Series("attempts_hat", att),
        pl.Series("rec_yards_hat", rec_y), pl.Series("rush_yards_hat", rush_y),
        pl.Series("pass_yards_hat", pass_y),
        pl.Series("tds_hat", rec_td + rush_td + pass_td),
        pl.Series("turnovers_hat", ints + fum),
        pl.Series("mu", np.nan_to_num(mu, nan=0.0)))


def positional_sd(train: pl.DataFrame) -> dict[str, float]:
    """Median within-player weekly sd, per position, from training seasons only.

    The *outcome* spread of a week, which is the numerator of the standard error below. Median
    rather than mean because a handful of player-seasons with two games produce sd estimates
    that are themselves noise.
    """
    d = (train.group_by(["player_id", "season", "position"])
              .agg(pl.col("fantasy_points_ppr").std().alias("sd"), pl.len().alias("n"))
              .filter(pl.col("n") >= 6).drop_nulls("sd"))
    out = {}
    for pos in sorted(set(d["position"].to_list())):
        v = d.filter(pl.col("position") == pos)["sd"].median()
        if v is not None:
            out[pos] = float(cast(float, v))
    pooled = d["sd"].median()
    out["__pooled__"] = float(cast(float, pooled)) if pooled is not None else 0.0
    return out


def standard_error(now: pl.DataFrame, sigma: dict[str, float]) -> np.ndarray:
    """`sigma_pos / sqrt(n)`: how well we know his mean, not how much his weeks vary.

    ADR-0012 measured per-player *volatility* beyond the positional constant at +/-9.3% and
    called it not estimable. This is the other quantity, it is +36% at one game played against
    twelve, and it is estimable because it is just `n`. See docs/parameter-uncertainty.md.
    """
    n = np.clip(np.nan_to_num(now["games_before"].to_numpy().astype(float), nan=0.0), 1.0, None)
    pooled = sigma.get("__pooled__", 0.0)
    s = np.array([sigma.get(p, pooled) for p in now["position"].to_list()])
    return s / np.sqrt(n)


def flat(now: pl.DataFrame) -> np.ndarray:
    """The incumbent: his season-to-date points per game, applied unchanged to this week."""
    return np.nan_to_num(now["ppg_before"].to_numpy().astype(float), nan=0.0)


def shipped_quantiles(mu: np.ndarray, position: Sequence[str]) -> np.ndarray:
    """The distribution this repo *publishes* around a projected mean, as its quantiles.

    `predict.moments` and `predict.skewed` rather than the two laws written out again:
    `sd = K[position] * sqrt(mu)` and the fitted Cornish-Fisher skew are one implementation
    each, and a second copy here is how the thing being graded stops being the thing being
    served. `moments` reads a mean off `proj_ppg`, so that is the column the arm's mean is
    handed to it under.

    **Including the zero clip**, which is part of the published forecast rather than an
    artefact of grading it: `skewed` clips at zero because nobody scores negative points
    often enough to matter, so a week projected at or below zero is served as a point mass
    at zero and is scored as one. A grader that unclipped it would be scoring a distribution
    this repo does not serve, which is the defect `docs/weekly-coverage.md` found in its own
    first measurement from the other side.
    """
    m = predict.moments(pl.DataFrame({"proj_ppg": np.asarray(mu, dtype=float),
                                      "position": list(position)}))
    z = normal_quantile(quantile_levels())[None, :]
    return predict.skewed(m["mu"].to_numpy()[:, None], m["sd"].to_numpy()[:, None],
                          m["skew"].to_numpy()[:, None], z)


def walk_forward(panel: pl.DataFrame) -> pl.DataFrame:
    """One row per held-out player-week, carrying **three** arms' absolute error.

    Three, not two, because the first version carried two and could not see its own subject.
    It compared the fitted projection against `ppg_before` -- a component rebuild against a
    points mean -- so the multiplier's contribution was buried under every difference between
    two whole estimators, and the answer it gave (-0.0025 MAE, 1/4 seasons) was about the
    rebuild rather than about the week.

    The plan's design is that **`f = 1` is the incumbent**. So:

      * `flat`      -- his season-to-date points per game. What the repo does today.
      * `component` -- the same rebuild with the multiplier switched off. `f = 1`.
      * `weekly`    -- the rebuild with the fitted multiplier.

    `weekly` against `component` isolates the week. `component` against `flat` is a separate
    question about the rebuild, and conflating them is how a null gets attributed to the wrong
    half. Fitting is on strictly earlier seasons only, through `expanding_seasons`.

    **And one column that is not an absolute error** (#177). `crps_weekly` is the continuous
    ranked probability score of the *distribution* the deployed arm publishes -- mean, spread
    and skew -- against the same outcome, on the same rows. `err_weekly` is what that same
    rule returns for the same projection published as a point mass, so the two are on one
    scale and their difference is what the spread and the skew earn.

    Only the deployed arm carries one, deliberately. Scoring all three under CRPS would be
    re-running the contrasts under a second rule, which is a decision about what the gate is
    (ADR-0015) and not what this ticket asks for; `flat` and `component` are alternative
    *means* and the repo publishes a distribution for neither.
    """
    frames = []
    for season, past, now in expanding_seasons(panel):
        coefs = {c: fit_multiplier(past, c) for c in VOLUME}
        fitted = project(now, coefs)
        base = project(now, dict.fromkeys(VOLUME, 0.0))
        actual = fitted["fantasy_points_ppr"].to_numpy().astype(float)
        mu = fitted["mu"].to_numpy().astype(float)
        frames.append(pl.DataFrame({
            "season": [season] * fitted.height, "week": fitted["week"],
            "err_weekly": np.abs(mu - actual),
            "err_component": np.abs(base["mu"].to_numpy().astype(float) - actual),
            "err_flat": np.abs(flat(fitted) - actual),
            "crps_weekly": crps_from_quantiles(
                shipped_quantiles(mu, fitted["position"].to_list()), actual),
            **{f"coef_{c}": [coefs[c]] * fitted.height for c in VOLUME}}))
    return pl.concat(frames) if frames else pl.DataFrame()


def _contrast(errs: pl.DataFrame, base: str, arm: str, label: str) -> list[str]:
    from hub.models.experiment import paired_gain
    per = (errs.group_by("season")
               .agg(pl.col(f"err_{base}").mean().alias("b"),
                    pl.col(f"err_{arm}").mean().alias("a"))
               .sort("season"))
    g = paired_gain(errs[f"err_{base}"].to_numpy(), errs[f"err_{arm}"].to_numpy(),
                    base_mae=per["b"].to_numpy(), arm_mae=per["a"].to_numpy())
    return [f"  {label:34} {g.mean:+.4f} MAE at {g.t:+5.1f} se, "
            f"wins {g.wins}/{g.seasons} seasons"]


# What `crps_weekly / err_weekly` would be if the published distribution were exactly right.
# `hub.models.scoring_rules.crps_normal` derives it: over outcomes drawn from the forecast
# itself the expected CRPS is `sd/sqrt(pi)` and the expected absolute error is
# `sd*sqrt(2/pi)`, so a calibrated forecast scores `1/sqrt(2)` of what publishing only its
# centre scores. It is a *reference*, not a bar: it holds for a normal, and the shipped
# distribution is skewed and clipped, so the figure printed beside it says how much of the
# available gain the published spread is collecting rather than whether it passed anything.
CALIBRATED_RATIO = 1.0 / np.sqrt(2.0)


def distribution_report(errs: pl.DataFrame) -> list[str]:
    """CRPS beside MAE for the deployed arm, on the rows the table above was built from.

    #177. MAE is what decides today and it is minimised by the median: it reads the centre
    and is blind to the spread and the skew, which are two thirds of what `predict.moments`
    produces and what `season/lineup.py` chooses a lineup with. CRPS reads the whole
    distribution and is proper, so a forecast cannot score better by misstating its spread.

    The two columns are comparable because MAE *is* CRPS -- the score the same rule gives the
    same projection published as a point mass -- so `crps/mae` is a pure number: 1.000 is a
    distribution that has added nothing over its own centre, and above 1.000 is one that has
    made the forecast worse.

    **The same window, and the same rows.** Both columns are means over the held-out
    player-weeks in `errs`, which is what `expanding_seasons` left. This is not the weekly
    *gate* -- that one scores lineups, restricts both arms to what both can price (#206), and
    has no points-error to take (ADR-0015). CRPS could not be reported "beside MAE" there
    because there is no MAE there to report it beside.
    """
    per = (errs.group_by("season")
               .agg(pl.len().alias("n"),
                    pl.col("err_weekly").mean().alias("mae"),
                    pl.col("crps_weekly").mean().alias("crps"))
               .sort("season"))
    lines = ["", "  The distribution, not just the centre (#177). CRPS of the published "
                 "(mu, sd, skew)",
             "  against MAE, which is the same rule applied to the same projection as a "
             "point mass:", "",
             f"  {'season':>7} {'n':>6} {'MAE':>8} {'CRPS':>8} {'crps/mae':>9}"]
    for r in per.iter_rows(named=True):
        ratio = r["crps"] / r["mae"] if r["mae"] else float("nan")
        lines.append(f"  {r['season']:>7} {r['n']:>6} {r['mae']:>8.3f} "
                     f"{r['crps']:>8.3f} {ratio:>9.3f}")
    mae = float(cast(float, errs["err_weekly"].mean()))
    crps = float(cast(float, errs["crps_weekly"].mean()))
    lines.append(f"  {'pooled':>7} {errs.height:>6} {mae:>8.3f} {crps:>8.3f} "
                 f"{(crps / mae if mae else float('nan')):>9.3f}")
    lines.append(f"\n  A point mass scores 1.000 by construction; a perfectly calibrated "
                 f"forecast would score {CALIBRATED_RATIO:.3f}.")
    lines += _what_the_coverage_measurement_says()
    return lines


def _what_the_coverage_measurement_says() -> list[str]:
    """Why the ratio above is worth reading, from the measurement rather than the document.

    The re-prioritisation that deferred this ticket argued the expected payoff was low
    because "the moments MAE cannot see are already within a point of nominal". That premise
    is now committed code and it did not survive being run on a centre that cannot see the
    week it scores -- which is the sequence ADR-0007 exists to produce, and the reason this
    reads `hub.models.coverage`'s own artifact instead of quoting a figure a person wrote
    down. A figure restated here would go stale the next time the measurement is re-run, and
    would say so to nobody.

    Degrades to the command that produces it, per `CLAUDE.md`: a tree with no
    `data/processed/` is a fresh clone, not a defect.
    """
    from hub.models.coverage import published_summary

    got = published_summary()
    if got is None:
        return ["  Whether that gap is worth closing is what the interval measurement says: "
                "run `uv run python -m\n  hub.models.coverage --measure --write` for it. "
                "Nothing is published in this tree yet."]
    nominal = 0.80
    return [f"  The published interval, measured on a centre that cannot see the week it "
            f"scores: {got['verdict']}\n  at {got['gate_cov80']:.1%} against a nominal "
            f"{nominal:.0%} over {got['gate_n']:,} {got['gate_subset']} weeks "
            f"(+/- {got['band']:.0%}), measured\n  {got['generated_at']}. That is what the "
            f"gap between this ratio and {CALIBRATED_RATIO:.3f} is made of."]


def diagnostic(errs: pl.DataFrame) -> list[str]:
    """Gate A, which is a **diagnostic and not a gate** -- ADR-0015.

    Both contrasts are reported because they answer different questions and only one of them
    is about the week. Neither decides anything: what decides is the lineup, and its incumbent
    is a consensus ranking with no points to take an error against.
    """
    if errs.is_empty():
        return ["  nothing measured -- no held-out season"]
    per = (errs.group_by("season")
               .agg(pl.len().alias("n"),
                    pl.col("err_flat").mean().alias("flat"),
                    pl.col("err_component").mean().alias("component"),
                    pl.col("err_weekly").mean().alias("weekly"),
                    *[pl.col(f"coef_{c}").first().alias(c) for c in VOLUME])
               .sort("season"))
    lines = [f"  {'season':>7} {'n':>6} {'flat':>8} {'f=1':>8} {'weekly':>8}"
             f" {'coef tgt':>9} {'coef car':>9}"]
    for r in per.iter_rows(named=True):
        lines.append(f"  {r['season']:>7} {r['n']:>6} {r['flat']:>8.3f} "
                     f"{r['component']:>8.3f} {r['weekly']:>8.3f} "
                     f"{r['targets']:>9.4f} {r['carries']:>9.4f}")
    lines.append("")
    lines += _contrast(errs, "component", "weekly", "the week (weekly vs f=1)")
    lines += _contrast(errs, "flat", "component", "the rebuild (f=1 vs flat)")
    lines += _contrast(errs, "flat", "weekly", "both together (weekly vs flat)")
    lines.append("\n  DIAGNOSTIC ONLY -- none of these is the gate. The flat projection has no "
                 "week-level term, so beating it is nearly free; the gate is the "
                 "lineup (ADR-0015).")
    return lines + distribution_report(errs)


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.models.weekly",
        description="Fit the Weekly projection and report the Gate A diagnostic.")
    ap.add_argument("--fit", action="store_true", help="build the panel, fit, walk forward")
    ap.add_argument("--expected", action="store_true",
                    help="use ff_opportunity's expected receptions and yardage in the priors")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.fit:
        ap.print_help()
        return 0
    try:
        panel = build_panel(SEASONS, PanelSpec(expected=a.expected))
    except Exception as e:
        return unavailable("hub.models.weekly", "the sources the Panel is built from", e)
    # Outside the guard deliberately. The filter is this repo's own logic over the Panel's own
    # columns, so a `ColumnNotFoundError` on `week` or `games_before` is a defect here, not a
    # source being unreachable -- and reporting a defect as an outage is what the board's
    # market stage was just hardened against. `weekly_screen` guards the build alone; this
    # matches it.
    panel = panel.filter(pl.col("week").is_in(list(FANTASY_WEEKS))
                         & (pl.col("games_before") >= MIN_GAMES_BEFORE))
    print(f"  {panel.height} player-weeks over {panel['season'].n_unique()} seasons")
    errs = walk_forward(panel)
    print("\n".join(diagnostic(errs)))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
