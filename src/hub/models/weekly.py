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

import numpy as np
import polars as pl

from hub.models.components import SCORING, td_rate
from hub.models.experiment import expanding_seasons

# The trend is dark before this: docs/snap-trend-signal.md finds anchors 4 and 6 null with the
# sign flipping between seasons, so before week 8 the multiplier is 1 and the projection is
# the flat one. That is not a fallback, it is the measurement.
TREND_MIN_WEEK = 8

# The multiplier is bounded. A snap share that doubled is real information; a multiplier of 4
# on a player's target count is an extrapolation past anything in the fit, and the cost of
# being wrong upward on a lineup is asymmetric -- you start him.
MULTIPLIER_LO, MULTIPLIER_HI = 0.6, 1.6

# Usage counts the multiplier applies to, and the phase each one scores through.
VOLUME: tuple[str, ...] = ("targets", "carries", "attempts")

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


def efficiency(df: pl.DataFrame, yards: str, units: str) -> np.ndarray:
    """Yards per unit from the player's own prior weeks, falling back to the sample rate.

    Held rather than projected: year over year, yards per carry persists at r = 0.108 and
    yards per target at 0.369, so there is nothing here a week-level model could add.
    """
    y = df[f"{yards}_prior"].to_numpy().astype(float)
    u = df[f"{units}_prior"].to_numpy().astype(float)
    games = df["games_before"].to_numpy().astype(float)
    pooled = float(np.nansum(y) / np.nansum(u)) if np.nansum(u) > 0 else 0.0
    enough = np.nan_to_num(u * games, nan=0.0) >= MIN_UNITS
    out = np.where(enough, np.divide(y, np.maximum(u, 1e-9)), pooled)
    return np.nan_to_num(out, nan=pooled)


def project(now: pl.DataFrame, coefs: dict[str, float]) -> pl.DataFrame:
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

    tgt = now["targets_prior"].to_numpy().astype(float) * m["targets"]
    car = now["carries_prior"].to_numpy().astype(float) * m["carries"]
    att = now["attempts_prior"].to_numpy().astype(float) * m["attempts"]

    catch = np.clip(np.nan_to_num(
        now["receptions_prior"].to_numpy().astype(float)
        / np.maximum(now["targets_prior"].to_numpy().astype(float), 1e-9), nan=0.0), 0.0, 1.0)
    rec = tgt * catch

    rec_y = rec * efficiency(now, "receiving_yards", "receptions")
    rush_y = car * efficiency(now, "rushing_yards", "carries")
    pass_y = att * efficiency(now, "passing_yards", "attempts")

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


def flat(now: pl.DataFrame) -> np.ndarray:
    """The incumbent: his season-to-date points per game, applied unchanged to this week."""
    return np.nan_to_num(now["ppg_before"].to_numpy().astype(float), nan=0.0)


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
    """
    frames = []
    for season, past, now in expanding_seasons(panel):
        coefs = {c: fit_multiplier(past, c) for c in VOLUME}
        fitted = project(now, coefs)
        base = project(now, dict.fromkeys(VOLUME, 0.0))
        actual = fitted["fantasy_points_ppr"].to_numpy().astype(float)
        frames.append(pl.DataFrame({
            "season": [season] * fitted.height, "week": fitted["week"],
            "err_weekly": np.abs(fitted["mu"].to_numpy().astype(float) - actual),
            "err_component": np.abs(base["mu"].to_numpy().astype(float) - actual),
            "err_flat": np.abs(flat(fitted) - actual),
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
    return lines


def main(argv: Sequence[str] | None = None) -> int:      # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.models.weekly",
        description="Fit the Weekly projection and report the Gate A diagnostic.")
    ap.add_argument("--fit", action="store_true", help="build the panel, fit, walk forward")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.fit:
        ap.print_help()
        return 0
    from hub.models.weekly_screen import (
        GATE_WEEKS,
        MIN_GAMES_BEFORE,
        SEASONS,
        build_panel,
    )
    panel = build_panel(SEASONS).filter(
        pl.col("week").is_in(list(GATE_WEEKS))
        & (pl.col("games_before") >= MIN_GAMES_BEFORE))
    print(f"  {panel.height} player-weeks over {panel['season'].n_unique()} seasons")
    errs = walk_forward(panel)
    print("\n".join(diagnostic(errs)))
    return 0


if __name__ == "__main__":                                # pragma: no cover
    sys.exit(main())
