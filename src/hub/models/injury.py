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
from pathlib import Path

import numpy as np
import polars as pl

from hub import atomic
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.declare import not_an_input
from hub.ledger import Ledger, recipe
from hub.models.experiment import (
    Actions,
    GateRun,
    Harness,
    expanding_seasons,
)
from hub.names import practice_key

# Positions this league drafts.

# Weeks of healthy play needed before a player's own baseline means anything. Below this, one
# good game defines the level everything else is measured against.
MIN_HEALTHY_WEEKS = 6

# The version of the baseline every gate here is measured on, carried in the ledger's recipe
# (#436) so a width on the strictly-prior baseline never compares against one on the
# within-season lookahead (#361): the config and data digests are the same under both, since
# neither sees the code. The systemic fix -- a code digest in the key -- is #435's; until it
# lands, CHANGE THIS STRING whenever `observations` changes what a baseline is.
BASELINE = "strictly-prior"

# Cells thinner than this are folded into their status's pooled value rather than reported.
MIN_CELL = 60

# The weekly-stats columns `observations` reads: `PLAYER_STATS`' five required, which is also
# everything it asks of the frame.
STAT_COLS = ("season", "week", "player_id", "position", "fantasy_points_ppr")



def _injury_type(injuries: pl.DataFrame) -> pl.Expr:
    """`report_primary_injury`, normalised, because the raw field is three fields.

    nflverse passes the club's own wording straight through, so `Shoulder`, `Right Shoulder`
    and `left Shoulder` are three separate categories for one injury -- 110 distinct values
    across 2022-25, collapsing to 74 on casefolding and stripping laterality, which roughly
    doubles the evidence behind each cell. One "category" is a free-text sentence beginning
    "Player was ill this morning".

    Laterality is dropped rather than kept because nothing here is lateral: this prices what a
    designation costs in fantasy points, and a right hamstring costs what a left one costs.

    53% of rows carry no type at all, and those become "unknown" -- a real category. Dropping
    them would price injuries among the players whose injury happened to be reported.
    """
    if "report_primary_injury" not in injuries.columns:
        return pl.lit("unknown", pl.Utf8)
    return (pl.col("report_primary_injury").fill_null("unknown")
              .str.strip_chars().str.to_lowercase()
              .str.replace(r"^(right|left|r\.|l\.)\s+", "")
              .replace("", "unknown"))


def observations(injuries: pl.DataFrame, stats: pl.DataFrame, *,
                 min_healthy: int = MIN_HEALTHY_WEEKS) -> pl.DataFrame:
    """One row per designated player-week: (status, practice, points, baseline, delta).

    A player-week with an injury row and no stat row scores zero, not null — he is exactly the
    outcome this is trying to price, and dropping him would measure the cost of an injury
    among players who played through it.
    """
    inj = (injuries.filter(pl.col("position").is_in(DRAFTED_POSITIONS))
                   .unique(["season", "week", "gsis_id"])
                   .select("season", "week", "gsis_id",
                           pl.col("report_status").fill_null("None").alias("status"),
                           practice_key().alias("practice"),
                           # What is actually wrong with him, which the (status, practice)
                           # table throws away. 53% of rows are null upstream, so "Unknown"
                           # is a real category here rather than a drop: dropping them would
                           # price injuries among players whose injury was reported.
                           _injury_type(injuries).alias("injury")))
    st = (stats.filter(pl.col("position").is_in(DRAFTED_POSITIONS))
               .select("season", "week", pl.col("player_id").alias("gsis_id"),
                       pl.col("fantasy_points_ppr").fill_null(0.0).alias("pts")))

    full = (st.join(inj, on=["season", "week", "gsis_id"], how="full", coalesce=True)
              .with_columns(pl.col("pts").fill_null(0.0),
                            pl.col("status").fill_null("Healthy"),
                            pl.col("practice").fill_null("Healthy")))
    # #361 (S4): the baseline is what a Sunday has -- the mean of his healthy weeks STRICTLY
    # BEFORE the designated one, in the same season. It was the mean over the whole season's
    # healthy weeks, so a week-5 designation was measured against weeks 6-18: the held-out arm
    # handed the realised in-season level. `healthy_weeks` is the prior count, and the
    # `min_healthy` floor is on it, so a designation needs `min_healthy` weeks of history
    # *behind* it and not merely in the season.
    healthy = (full.filter(pl.col("status") == "Healthy")
                   .sort("gsis_id", "season", "week")
                   .with_columns(pl.col("pts").cum_sum().over(["gsis_id", "season"])
                                   .alias("_cum"),
                                 pl.col("pts").cum_count().over(["gsis_id", "season"])
                                   .alias("healthy_weeks"))
                   .select("gsis_id", "season", "week", "healthy_weeks", "_cum"))
    designated = full.filter(pl.col("status") != "Healthy").sort("week")
    return (designated
            .join_asof(healthy.sort("week"), on="week", by=["gsis_id", "season"],
                       strategy="backward", allow_exact_matches=False,
                       check_sortedness=False)
            .filter(pl.col("healthy_weeks") >= min_healthy)
            .with_columns((pl.col("_cum") / pl.col("healthy_weeks")).alias("baseline"))
            .with_columns((pl.col("pts") - pl.col("baseline")).alias("delta"))
            .drop("_cum"))


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

# --- does what is wrong with him add anything? ------------------------------
#
# `retention` prices a designation by (status, practice) and ignores `report_primary_injury`
# entirely. A hamstring is not an ankle is not a concussion: hamstrings re-injure, concussions
# are protocol-governed and resolve on a schedule that has little to do with Friday practice.
#
# THE GATE, FIXED BEFORE THIS WAS RUN. The incumbent is `retention` -- the thing that already
# won -- not `out_zero`. To be adopted, the type-adjusted model must beat it on held-out MAE
# in EVERY held-out season AND clear the pooled interval on the paired difference. Beating it
# on the mean while losing a season is how a fit gets adopted on one lucky year, and this repo
# has eleven nulls behind it precisely because that bar is kept where it is.
#
# **Since #343 the bar is `experiment.run_gate`'s and not this module's.** It was written out
# here by hand (`g.wins == g.seasons and g.t >= MIN_SE`) with no stage 2 and no stamps -- the
# second copy of the rule ADR-0019 exists to end. The comparison is one `Harness.run`.

# The within-season unit of both comparisons here is the player: `gsis_id`, `docs/method.md`
# rule 3's repeated-measure unit (a player appears in many designated weeks of one season) and
# the unit #335 adopted for this gate. #343 declared the type comparison with `season` instead,
# one cluster a season and a declared no-op that read every season on its sign alone, because
# this module's unit was #360's to choose; #360 chose it (2026-10-07), which discharges the
# no-op ADR-0019's amendment recorded.
WITHIN: tuple[str, ...] = ("gsis_id",)

# The declared ceiling arm (S6, #363), by name where it prints. Pre-registered for *this*
# comparison at #343's adoption (not the component calibration's, which was the lane's own
# choice -- see `component_error.py`), the `hub.models.margin.ceiling` analogue: per held-out
# season, a per-injury-type
# retention multiplier fitted *in sample on that season's own designated rows* (unshrunk,
# `k = 0`: the oracle that knows each type's realised multiplier) and scored on the same
# rows. The ceiling gain is `retention`'s error minus this arm's, row by row. Flattered by
# construction, which is what a ceiling is; it is a forecast bound on the functional form
# under test, never a bound read off the outcome.
CEILING_ARM = "the per-type multiplier fitted in sample on the held-out season's own rows"

ACTIONS = Actions(
    adopt="The type-adjusted table replaces retention as the weekly injury price.",
    remove="The type adjustment is worse than retention: it stays out.",
    show="Retention stays; what is wrong with him adds nothing the Gate could establish.")

HARNESS = Harness(name="injury_type", arm_a="type-adjusted", arm_b="retention", within=WITHIN,
                  ceiling_arm=CEILING_ARM, actions=ACTIONS,
                  unit="MAE points per designated player-week", places=4,
                  arm_roots=("hub.models.injury",))

# --- is retention the availability model, or is zeroing a designated player enough? (#360) ---
#
# THE GATE, FIXED BEFORE IT WAS RUN (docs/weekly-injury.md, *Pre-registration, 2026-10-07*).
# `verdict` used to be an argmin over four candidates that adopted whichever fitted table had
# the lowest mean MAE: no interval, no every-season half, the lowest bar in the repo. It is
# now `experiment.run_gate` over two arms, `retention` against `out_zero`; `table` and
# `baseline` are diagnostics, measured and printed and never gated. The baseline stays
# within-season (the strictly-prior expanding mean, `BASELINE`).
RETENTION_CEILING_ARM = "the per-cell retention fitted in sample on the held-out season's own rows"

RETENTION_ACTIONS = Actions(
    adopt="Retention is the availability model the product wires in place of zeroing a "
          "designated player.",
    remove="Retention is worse than zeroing a designated player; it is dropped from the "
           "model surface.",
    show="Retention is kept as a measurement and not wired; the product keeps ESPN's "
         "availability.")

RETENTION_HARNESS = Harness(
    name="injury_retention", arm_a="retention", arm_b="out_zero", within=WITHIN,
    ceiling_arm=RETENTION_CEILING_ARM, actions=RETENTION_ACTIONS,
    unit="MAE points per designated player-week", places=4,
    arm_roots=("hub.models.injury",))

# Shrinkage grid for the per-type multiplier, chosen on TRAINING rows only. Shrinking toward
# 1.0 rather than imposing a cell minimum is what lets a thin type (Groin, n=450 across four
# seasons, far fewer inside any one training window) contribute in proportion to its evidence
# instead of being either trusted outright or dropped outright.
SHRINK_GRID = not_an_input(
    (25.0, 50.0, 100.0, 200.0, 400.0),
    "a search grid; the retention table is fitted at run time from nflverse and never "
    "frozen into this module (docs/weekly-injury.md)")


def type_adjustment(obs: pl.DataFrame, table: pl.DataFrame, *, fallback: float,
                    k: float) -> dict[str, float]:
    """Multiplier per injury type on what the base table already predicts.

    Fitted as a ratio of totals for the same reason `retention_table` is -- a per-row ratio
    blows up wherever the base prediction is near zero, and an Out player's prediction is
    exactly zero by construction.

    Shrunk toward 1.0 by `k`, so a type with little evidence changes nothing.
    """
    if obs.is_empty() or "injury" not in obs.columns:
        return {}
    pred = predict_retention(obs, table, fallback=fallback)
    df = obs.select("injury").with_columns(
        pl.Series("pred", pred),
        pl.Series("act", obs["pts"].to_numpy().astype(float)))
    g = (df.group_by("injury")
           .agg(pl.len().alias("n"), pl.col("pred").sum().alias("p"),
                pl.col("act").sum().alias("a"))
           .filter(pl.col("p") > 0))
    return {r["injury"]: float((r["n"] * (r["a"] / r["p"]) + k) / (r["n"] + k))
            for r in g.iter_rows(named=True)}


def predict_with_type(obs: pl.DataFrame, table: pl.DataFrame,
                      adj: dict[str, float], *, fallback: float) -> np.ndarray:
    """The base prediction, scaled by the player's injury type. Empty `adj` is a no-op."""
    base = predict_retention(obs, table, fallback=fallback)
    if not adj or "injury" not in obs.columns:
        return base
    mult = np.array([adj.get(i, 1.0) for i in obs["injury"].to_list()])
    return base * mult


def fit_shrink(train: pl.DataFrame, table: pl.DataFrame, *, fallback: float) -> float:
    """Pick `k` on training rows only. Never sees a held-out season."""
    actual = train["pts"].to_numpy().astype(float)
    best, best_mae = SHRINK_GRID[-1], np.inf
    for k in SHRINK_GRID:
        adj = type_adjustment(train, table, fallback=fallback, k=k)
        mae = float(np.abs(predict_with_type(train, table, adj, fallback=fallback)
                           - actual).mean())
        if mae < best_mae:
            best, best_mae = float(k), mae
    return best


def walk_forward_type(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """One row per held-out observation: `retention`'s error and the type-adjusted one.

    Per observation rather than per season so the gate can be paired -- the same player-week
    scored by both arms, which is a far tighter comparison than two independent means.

    **`err_oracle` (#343, S6):** the declared ceiling arm's error on the same row -- the
    per-type multiplier fitted *in sample on this held-out season's own rows* (`k = 0`) and
    applied to them. It sees the outcome it is scored on, so it bounds what any walk-forward
    fit of this functional form could earn; it is never an arm.
    """
    frames = []
    for yr, past, now in expanding_seasons(obs):
        ret = retention_table(past, min_cell=min_cell)
        pb = float(past["baseline"].sum() or 0.0)
        pooled = float(past["pts"].sum() or 0.0) / pb if pb > 0 else 1.0
        k = fit_shrink(past, ret, fallback=pooled)
        adj = type_adjustment(past, ret, fallback=pooled, k=k)
        oracle = type_adjustment(now, ret, fallback=pooled, k=0.0)
        actual = now["pts"].to_numpy().astype(float)
        frames.append(pl.DataFrame({
            "season": [yr] * now.height, "gsis_id": now["gsis_id"], "k": [k] * now.height,
            "err_retention": np.abs(predict_retention(now, ret, fallback=pooled) - actual),
            "err_type": np.abs(predict_with_type(now, ret, adj, fallback=pooled) - actual),
            "err_oracle": np.abs(predict_with_type(now, ret, oracle, fallback=pooled) - actual),
        }))
    return pl.concat(frames) if frames else pl.DataFrame()


def type_frame(errs: pl.DataFrame) -> pl.DataFrame:
    """The paired rows for the Gate: `diff` is `retention` minus the type-adjusted error
    (positive when type-adjusting helps), `ceiling_diff` is `retention` minus the declared
    ceiling arm's. A frame with no `err_oracle` carries no `ceiling_diff` column at all, so the
    Gate reads it as a ceiling never measured -- NOT-RUNNABLE -- and not as a ceiling of zero."""
    cols = [pl.col("season"), pl.col("gsis_id"),
            (pl.col("err_retention") - pl.col("err_type")).alias("diff")]
    if "err_oracle" in errs.columns:
        cols.append((pl.col("err_retention") - pl.col("err_oracle")).alias("ceiling_diff"))
    return errs.select(cols)


def type_run(errs: pl.DataFrame, *, publish: bool = False, seasons: Sequence[int] = (),
             seed: int = 0, ledger: Ledger | None = None) -> GateRun:
    """`type` against `retention` through the one Gate (#343): `Harness.decide` for the pure
    verdict, `Harness.run` -- the report with stage 2 and the four stamps, and a Ledger write --
    when `publish`."""
    harness = HARNESS
    paired = type_frame(errs)
    if publish:
        return harness.run(paired, seed=seed, ledger=ledger,
                           recipe=recipe(baseline=BASELINE, seasons=list(seasons)))
    return harness.decide(paired, seed=seed)


def type_verdict(errs: pl.DataFrame, *, run: GateRun | None = None) -> tuple[str, str]:
    """Does what is wrong with him add to how he practised? The Gate's answer.

    **Not a rule of this module's own since #343.** It was `g.wins == g.seasons and g.t >=
    MIN_SE`, a hand-built every-season-and-2-se check beside `experiment.gate`, with no stage 2
    and no ceiling -- and before that, per #335, a `within=errs["season"]` no-op passed to a
    function that no longer decides here. The type adjustment is adopted exactly when
    `experiment.gate` says ADOPT for it; every other verdict -- SHOW, REMOVE, and
    NOT-RUNNABLE, which says the design could not have found the effect rather than that there
    was none -- keeps `retention`, with the Gate's own sentence beside it.

    `run` is what `main` hands in so the gate is run once, published, and read here.
    """
    if errs.is_empty() or "err_type" not in errs.columns:
        return "retention", "nothing measured -- no held-out season"
    got = run if run is not None else type_run(errs)
    label, sentence = got.verdict
    line = (f"  type-adjusted: {label}: mean gain {float(got.summary['mean']):+.4f} MAE -- "
            f"{sentence}")
    if label == "ADOPT":
        return "type", (f"ADOPT 'type': the injury type adds to (status, practice).\n{line}")
    return "retention", (f"KEEP 'retention': what is wrong with him adds nothing the Gate "
                         f"could establish.\n{line}")


ERR_COLS = ("baseline", "out_zero", "table", "retention")


def walk_forward_rows(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """One row per held-out designated player-week: every candidate's absolute error on it.

    Four candidates, and the two simple ones are the ones a person would actually use:

      * `baseline`  -- ignore the designation; predict his healthy average. The null.
      * `out_zero`  -- bench anyone listed Out or Doubtful, otherwise ignore it. The rule
                       every fantasy manager already follows without a model. **The
                       incumbent the gate measures `retention` against.**
      * `table`     -- the fitted (status, practice) additive penalties. A diagnostic.
      * `retention` -- the fitted multiplicative table. The arm under test.

    plus `err_ceiling` (#360): the declared ceiling arm, the (status, practice) retention table
    fitted *in sample on this held-out season's own designated rows* (same `min_cell`, thin
    cells folded to that season's own pooled ratio) and scored on them. It sees the outcome it
    is scored on, so it bounds what any walk-forward fit of this functional form could earn;
    it is never an arm.

    Fitting is on strictly earlier seasons only, except `err_ceiling`. Per observation rather
    than per season so the Gate can pair the arms (the same player-week scored by both) and
    cluster on the player (`gsis_id`).
    """
    frames = []
    for yr, past, now in expanding_seasons(obs):
        table = penalty_table(past, min_cell=min_cell)
        ret = retention_table(past, min_cell=min_cell)
        pooled = _mean(past, "delta")
        pooled_ret = _pooled_retention(past)
        actual = now["pts"].to_numpy().astype(float)
        base = now["baseline"].to_numpy().astype(float)
        ruled_out = np.array([s in ("Out", "Doubtful") for s in now["status"].to_list()])
        in_sample = retention_table(now, min_cell=min_cell)
        preds = {
            "baseline": base,
            "out_zero": np.where(ruled_out, 0.0, base),
            "table": predict(now, table, fallback=pooled),
            "retention": predict_retention(now, ret, fallback=pooled_ret),
            "ceiling": predict_retention(now, in_sample, fallback=_pooled_retention(now)),
        }
        frames.append(pl.DataFrame({
            "season": [yr] * now.height, "gsis_id": now["gsis_id"],
            **{f"err_{k}": np.abs(v - actual) for k, v in preds.items()}}))
    return pl.concat(frames) if frames else pl.DataFrame()


def _pooled_retention(df: pl.DataFrame) -> float:
    """Ratio of totals over every row: what a cell absent from the table falls back to."""
    b = float(df["baseline"].sum() or 0.0)
    return float(df["pts"].sum() or 0.0) / b if b > 0 else 1.0


def walk_forward(obs: pl.DataFrame, *, min_cell: int = MIN_CELL) -> pl.DataFrame:
    """Held-out mean absolute error per season: the diagnostic table `main` prints.

    The four candidates' means over `walk_forward_rows`, one row per held-out season. Only
    `retention` against `out_zero` is gated (`verdict`); `table` and `baseline` are reported.
    """
    rows = walk_forward_rows(obs, min_cell=min_cell)
    if rows.is_empty():
        return pl.DataFrame()
    return (rows.group_by("season")
                .agg(pl.len().alias("n"),
                     *[pl.col(f"err_{c}").mean().alias(f"mae_{c}") for c in ERR_COLS])
                .sort("season"))


def retention_frame(errs: pl.DataFrame) -> pl.DataFrame:
    """The paired rows for the Gate: `diff` is `out_zero`'s error minus `retention`'s (positive
    when retention helps), `ceiling_diff` is `out_zero`'s minus the declared ceiling arm's. A
    frame with no `err_ceiling` carries no `ceiling_diff` column at all, so the Gate reads it
    as a ceiling never measured -- NOT-RUNNABLE -- and not as a ceiling of zero."""
    cols = [pl.col("season"), pl.col("gsis_id"),
            (pl.col("err_out_zero") - pl.col("err_retention")).alias("diff")]
    if "err_ceiling" in errs.columns:
        cols.append((pl.col("err_out_zero") - pl.col("err_ceiling")).alias("ceiling_diff"))
    return errs.select(cols)


def retention_run(errs: pl.DataFrame, *, publish: bool = False, seasons: Sequence[int] = (),
                  seed: int = 0, ledger: Ledger | None = None) -> GateRun:
    """`retention` against `out_zero` through the one Gate: `Harness.decide` for the pure
    verdict, `Harness.run` -- the report with stage 2 and the four stamps, and a Ledger write --
    when `publish`."""
    paired = retention_frame(errs)
    if publish:
        return RETENTION_HARNESS.run(paired, seed=seed, ledger=ledger,
                                     recipe=recipe(baseline=BASELINE, seasons=list(seasons)))
    return RETENTION_HARNESS.decide(paired, seed=seed)


def verdict(errs: pl.DataFrame, *, run: GateRun | None = None) -> tuple[str, str]:
    """Is retention the availability model? The Gate's answer, `(model, sentence)`.

    **Not an argmin since #360.** It took the lowest of four mean MAEs and adopted whichever
    fitted table that was: no interval, no every-season half, no ceiling. The bar is now
    `experiment.gate`'s, the same two halves every other comparison faces, and the rule is
    pre-registered (`docs/weekly-injury.md`): `retention` is the availability model exactly when
    the Gate says ADOPT for it against `out_zero`. REMOVE, SHOW and NOT-RUNNABLE (the design
    could not have found the effect, which is not that there was none) each leave `out_zero`
    standing, with the Gate's own sentence beside it. What the product wires is not this
    function's to change.

    `run` is what `main` hands in so the gate is run once, published, and read here.
    """
    if errs.is_empty() or "err_retention" not in errs.columns:
        return "baseline", "no held-out seasons; nothing measured."
    got = run if run is not None else retention_run(errs)
    label, sentence = got.verdict
    line = (f"  retention vs out_zero: {label}: mean gain {float(got.summary['mean']):+.4f} MAE"
            f" -- {sentence}")
    if label == "ADOPT":
        return "retention", f"ADOPT 'retention': it beats bench-the-ruled-out.\n{line}"
    return "out_zero", (f"KEEP 'out_zero': retention did not clear the Gate against "
                        f"bench-the-ruled-out.\n{line}")


def main(argv: Sequence[str] | None = None, *, ledger: Ledger | None = None) -> int:
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

    from hub.fetch import nflverse

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    print(f"  loading injuries and weekly stats for {seasons} ...")
    try:
        obs = observations(
            nflverse.load("injuries", seasons),
            nflverse.load("player_stats", seasons, cols=STAT_COLS))
    except Exception as e:
        return unavailable("hub.models.injury", "nflverse injuries and weekly player stats", e)
    print(f"  {obs.height} designated player-weeks with a usable healthy baseline")

    table = penalty_table(obs)
    print(f"\n  {'status':<14} {'practice':<9} {'n':>5} {'penalty':>9} {'se':>7}")
    for r in table.iter_rows(named=True):
        print(f"  {r['status']:<14} {r['practice']:<9} {r['n']:>5} "
              f"{r['penalty']:>+9.2f} {r['se']:>7.2f}")

    rows = walk_forward_rows(obs)
    wf = walk_forward(obs)
    if not wf.is_empty():
        print(f"\n  Held-out MAE, {wf.height} seasons, fitting only on earlier ones:")
        print(f"  {'season':>6} {'n':>5}  " + "  ".join(f"{c:>10}" for c in CANDIDATES))
        for r in wf.iter_rows(named=True):
            print(f"  {r['season']:>6} {r['n']:>5}  "
                  + "  ".join(f"{r['mae_' + c]:>10.4f}" for c in CANDIDATES))
        print(f"  {'mean':>6} {'':>5}  "
              + "  ".join(f"{_mean(wf, 'mae_' + c):>10.4f}" for c in CANDIDATES))
    # Is retention the availability model? The Gate (#360), not an argmin.
    if not rows.is_empty():
        gate_run = retention_run(rows, publish=True, seasons=seasons, ledger=ledger)
        print("\n  === retention against out_zero ===")
        print("  (the MAE table above is a diagnostic: `table` and `baseline` are reported, "
              "never gated)")
        print("\n".join(gate_run.lines))
        print(f"\n  {verdict(rows, run=gate_run)[1]}")
    else:
        print(f"\n  {verdict(rows)[1]}")

    # Does what is wrong with him add to how he practised? The Gate (#343).
    errs = walk_forward_type(obs)
    if not errs.is_empty():
        top = (obs.group_by("injury").agg(pl.len().alias("n"))
                  .sort("n", descending=True).head(8))
        print("\n  Injury types by volume: "
              + ", ".join(f"{r['injury']} {r['n']}" for r in top.iter_rows(named=True)))
        run = type_run(errs, publish=True, seasons=seasons, ledger=ledger)
        print("\n  === type-adjusted against retention ===")
        print("\n".join(run.lines))
        print(f"\n  {type_verdict(errs, run=run)[1]}")
    if a.out:
        atomic.write_parquet(table, Path(a.out))
        print(f"  wrote {table.height} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
