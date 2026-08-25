"""What a weekly injury designation costs, this week.

**This is not `INJURY_BETA`, and the difference matters.** `hub.draft.durability.INJURY_BETA`
prices OUT/DOUBTFUL/IR at -1.631 and is applied to `proj_blend`, a *season-long per-game*
projection — so it answers "what does a preseason designation cost across the whole season",
which is a draft question. A player ruled out in week 1 misses week 1 and plays the other
sixteen, so the season-average cost is small.

This module answers the *lineup* question: what does the designation cost **in the week it is
issued**. Those numbers are not comparable, and treating a within-week penalty as a correction
to a season-long one would look like a 5x discrepancy and be nonsense.

The repo has no weekly injury input. This is it.

## The estimand, and why it is points rather than P(plays)

`docs/next.md` framed this as `P(plays week N | ...)`. Measured directly, that runs into a data
problem: `player_stats` carries a row only for a player who *recorded a stat*, so ~18% of
demonstrably healthy players on the injury report have no row — a WR3 who dressed and was never
targeted looks identical to one who was inactive. Separating them needs a `gsis_id`-to-
`pfr_player_id` crosswalk into `snap_counts`.

It also would not help. For fantasy, a player who dressed and scored nothing is the same as one
who did not dress, and the quantity every consumer needs is points. So the estimand is the
**points delta against the player's own healthy baseline**, which subsumes P(plays) and is
directly usable.

## What it finds

Monotone in both dimensions, independently — the designation and the practice report each
carry information the other does not:

    Doubtful     + DNP      -8.88 +/-0.59   n=96
    Out          + DNP      -8.38 +/-0.20   n=779
    Questionable + DNP      -5.24 +/-0.46   n=206
    Questionable + Limited  -3.91 +/-0.25   n=809
    Questionable + Full     -2.48 +/-0.39   n=256
    (none)       + Full     -1.23 +/-0.15   n=2088

**A Questionable player who did not practise costs twice what one who practised fully does.**
`INJURY_BETA` prices QUESTIONABLE at zero because a preseason Questionable is a much healthier
group than a week-1 one. In-season, with the practice report attached, it is clearly not zero.

    uv run python -m hub.models.injury --fit
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

# Positions this league drafts.
POSITIONS = ("QB", "RB", "WR", "TE")

# Weeks of healthy play needed before a player's own baseline means anything. Below this, one
# good game defines the level everything else is measured against.
MIN_HEALTHY_WEEKS = 6

# Cells thinner than this are folded into their status's pooled value rather than reported.
MIN_CELL = 60

# Practice status arrives as long prose ("Did Not Participate In Practice"); the first seven
# characters separate the three cases and nothing else collides.
_PS_WIDTH = 7


def observations(injuries: pl.DataFrame, stats: pl.DataFrame, *,
                 min_healthy: int = MIN_HEALTHY_WEEKS) -> pl.DataFrame:
    """One row per designated player-week: (status, practice, points, baseline, delta).

    A player-week with an injury row and no stat row scores zero, not null — he is exactly the
    outcome this is trying to price, and dropping him would measure the cost of an injury
    among players who played through it.
    """
    inj = (injuries.filter(pl.col("position").is_in(POSITIONS))
                   .unique(["season", "week", "gsis_id"])
                   .select("season", "week", "gsis_id",
                           pl.col("report_status").fill_null("None").alias("status"),
                           pl.col("practice_status").fill_null("None")
                             .str.slice(0, _PS_WIDTH).alias("practice")))
    st = (stats.filter(pl.col("position").is_in(POSITIONS))
               .select("season", "week", pl.col("player_id").alias("gsis_id"),
                       pl.col("fantasy_points_ppr").fill_null(0.0).alias("pts")))

    full = (st.join(inj, on=["season", "week", "gsis_id"], how="full", coalesce=True)
              .with_columns(pl.col("pts").fill_null(0.0),
                            pl.col("status").fill_null("Healthy"),
                            pl.col("practice").fill_null("Healthy")))
    base = (full.filter(pl.col("status") == "Healthy")
                .group_by(["gsis_id", "season"])
                .agg(pl.len().alias("healthy_weeks"), pl.col("pts").mean().alias("baseline"))
                .filter(pl.col("healthy_weeks") >= min_healthy))
    return (full.filter(pl.col("status") != "Healthy")
                .join(base, on=["gsis_id", "season"])
                .with_columns((pl.col("pts") - pl.col("baseline")).alias("delta")))


def retention_table(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """Fraction of his healthy production a designated player keeps, per cell.

    **Ratio of totals, not mean of ratios.** A player whose healthy baseline is near zero
    produces a ratio near infinity, and averaging those measures nothing; summing points over
    summing baselines is what "fraction of production retained" actually means, and it is
    defined when a baseline is zero.

    Multiplicative because the shape of the thing demands it: a player ruled Out scores
    exactly zero, which `retention = 0` expresses exactly and no additive penalty can. The
    additive table lost its own gate for precisely this reason -- it predicted
    `baseline - 8.35` for an Out player whose true score was 0, so a 12-point player got a
    3.65-point prediction and a 3.65-point error.
    """
    if obs.is_empty():
        return pl.DataFrame(schema={"status": pl.Utf8, "practice": pl.Utf8, "n": pl.UInt32,
                                    "retention": pl.Float64})
    return (obs.group_by(["status", "practice"])
               .agg(pl.len().alias("n"), pl.col("pts").sum().alias("p"),
                    pl.col("baseline").sum().alias("b"))
               .filter((pl.col("n") >= min_cell) & (pl.col("b") > 0))
               .with_columns((pl.col("p") / pl.col("b")).alias("retention"))
               .drop("p", "b")
               .sort("retention"))


def predict_retention(obs: pl.DataFrame, table: pl.DataFrame, *,
                      fallback: float) -> np.ndarray:
    """Predicted points: the player's healthy baseline scaled by his cell's retention."""
    look = dict(zip(zip(table["status"].to_list(), table["practice"].to_list(), strict=True),
                    table["retention"].to_list(), strict=True))
    keep = np.array([look.get((s, p), fallback) for s, p
                     in zip(obs["status"].to_list(), obs["practice"].to_list(), strict=True)])
    return obs["baseline"].to_numpy().astype(float) * keep


def penalty_table(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """Mean points delta per (status, practice) cell, with a standard error.

    The additive form. Kept because it is the readable one -- "a Questionable player who did
    not practise costs five points" is a sentence -- and because its failure against
    `retention_table` is the finding, not a dead end.
    """
    if obs.is_empty():
        return pl.DataFrame(schema={"status": pl.Utf8, "practice": pl.Utf8, "n": pl.UInt32,
                                    "penalty": pl.Float64, "se": pl.Float64})
    return (obs.group_by(["status", "practice"])
               .agg(pl.len().alias("n"), pl.col("delta").mean().alias("penalty"),
                    pl.col("delta").std().alias("sd"))
               .filter(pl.col("n") >= min_cell)
               .with_columns((pl.col("sd") / pl.col("n").sqrt()).alias("se"))
               .drop("sd")
               .sort("penalty"))


def predict(obs: pl.DataFrame, table: pl.DataFrame, *, fallback: float) -> np.ndarray:
    """Predicted points for each designated player-week: baseline plus the cell's penalty.

    A cell absent from the table — too thin, or unseen in the fitting seasons — falls back to
    a single pooled penalty rather than to zero. Zero would silently assert that an unseen
    designation costs nothing, which is the opposite of what a designation means.
    """
    look = dict(zip(zip(table["status"].to_list(), table["practice"].to_list(), strict=True),
                    table["penalty"].to_list(), strict=True))
    pen = np.array([look.get((s, p), fallback) for s, p
                    in zip(obs["status"].to_list(), obs["practice"].to_list(), strict=True)])
    return obs["baseline"].to_numpy().astype(float) + pen


def _mean(df: pl.DataFrame, col: str) -> float:
    v = df[col].mean()
    return float(v) if isinstance(v, (int, float)) else float("nan")


CANDIDATES = ("baseline", "out_zero", "table", "retention")


def walk_forward(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """Held-out mean absolute error per season, table against two simpler rules.

    Three candidates, and the two baselines are the ones a person would actually use:

      * `baseline`  -- ignore the designation; predict his healthy average. The null.
      * `out_zero`  -- bench anyone listed Out or Doubtful, otherwise ignore it. The rule
                       every fantasy manager already follows without a model.
      * `table`     -- the fitted (status, practice) penalties.

    Fitting is on strictly earlier seasons only.
    """
    seasons = sorted(obs["season"].unique().to_list())
    rows = []
    for yr in seasons[1:]:
        past = obs.filter(pl.col("season") < yr)
        now = obs.filter(pl.col("season") == yr)
        if past.is_empty() or now.is_empty():
            continue
        table = penalty_table(past, min_cell=min_cell)
        ret = retention_table(past, min_cell=min_cell)
        pooled = _mean(past, "delta")
        pb = float(past["baseline"].sum() or 0.0)
        pooled_ret = float(past["pts"].sum() or 0.0) / pb if pb > 0 else 1.0
        actual = now["pts"].to_numpy().astype(float)
        base = now["baseline"].to_numpy().astype(float)
        ruled_out = np.array([s in ("Out", "Doubtful") for s in now["status"].to_list()])
        preds = {
            "baseline": base,
            "out_zero": np.where(ruled_out, 0.0, base),
            "table": predict(now, table, fallback=pooled),
            "retention": predict_retention(now, ret, fallback=pooled_ret),
        }
        rows.append({"season": yr, "n": now.height,
                     **{f"mae_{k}": float(np.abs(v - actual).mean()) for k, v in preds.items()}})
    return pl.DataFrame(rows)


def verdict(wf: pl.DataFrame) -> tuple[str, str]:
    """Pre-registered. The table must beat BOTH simpler rules on held-out mean absolute error.

    Both, not either: beating "ignore the injury report" only shows that injuries matter, which
    nobody doubts. The rule worth beating is the one a manager already follows for free — bench
    anyone ruled out — and a table that cannot beat that is a lookup nobody needs.
    """
    if wf.is_empty():
        return "baseline", "no held-out seasons; nothing measured."
    m = {c: _mean(wf, f"mae_{c}") for c in CANDIDATES}
    best = min(m, key=lambda c: m[c])
    if best in ("table", "retention"):
        # Every candidate is reported, not just the winner and the baselines. The additive
        # table losing to a one-line rule is the finding that motivated the multiplicative
        # one, and a verdict that printed only the winner would bury it.
        return best, (f"ADOPT '{best}': held-out MAE "
                      + ", ".join(f"{c} {m[c]:.4f}" for c in CANDIDATES)
                      + f". '{best}' beats bench-the-ruled-out by "
                      f"{m['out_zero'] - m[best]:.4f}.")
    return best, (f"KEEP '{best}': no fitted table beat both simpler rules "
                  f"(retention {m['retention']:.4f}, table {m['table']:.4f}, "
                  f"out_zero {m['out_zero']:.4f}, baseline {m['baseline']:.4f}). A lookup "
                  f"that cannot beat benching the ruled-out is a lookup nobody needs.")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.injury",
        description="What a weekly injury designation costs, and whether a table beats a rule.")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not a.fit:
        ap.print_help()
        return 0

    import nflreadpy as nfl

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    print(f"  loading injuries and weekly stats for {seasons} ...")
    obs = observations(
        nfl.load_injuries(seasons=seasons),
        nfl.load_player_stats(seasons=seasons, summary_level="week"))
    print(f"  {obs.height} designated player-weeks with a usable healthy baseline")

    table = penalty_table(obs)
    print(f"\n  {'status':<14} {'practice':<9} {'n':>5} {'penalty':>9} {'se':>7}")
    for r in table.iter_rows(named=True):
        print(f"  {r['status']:<14} {r['practice']:<9} {r['n']:>5} "
              f"{r['penalty']:>+9.2f} {r['se']:>7.2f}")

    wf = walk_forward(obs)
    if not wf.is_empty():
        print(f"\n  Held-out MAE, {wf.height} seasons, fitting only on earlier ones:")
        print(f"  {'season':>6} {'n':>5}  " + "  ".join(f"{c:>10}" for c in CANDIDATES))
        for r in wf.iter_rows(named=True):
            print(f"  {r['season']:>6} {r['n']:>5}  "
                  + "  ".join(f"{r['mae_' + c]:>10.4f}" for c in CANDIDATES))
        print(f"  {'mean':>6} {'':>5}  "
              + "  ".join(f"{_mean(wf, 'mae_' + c):>10.4f}" for c in CANDIDATES))
    print(f"\n  {verdict(wf)[1]}")
    if a.out:
        table.write_parquet(a.out)
        print(f"  wrote {table.height} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
