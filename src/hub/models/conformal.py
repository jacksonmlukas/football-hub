"""Rolling conformal calibration: intervals that cover whether or not the model is right.

`docs/foundation-plan.md` 3.3. `Conformalized` already existed in `hub.models.base` and
could calibrate against a window handed to it, but nothing ever decided what that window
was, so it was never used on anything.

What conformal buys is a coverage guarantee that does not depend on the model being right.
The intervals are built from the model's *observed* errors rather than its beliefs, so a
model badly overconfident about its own spread still gets covered at the nominal rate. It
fixes coverage, not bias -- a biased model is still covered, and pays for the bias in width.

The calibration window is strictly earlier weeks. Calibrating on the week being predicted
manufactures coverage out of nothing, and is the same leak `docs/track-record.md` rule 1
forbids -- and the same one this repo caught in its own depth-chart screen, where measuring
the predictor over the outcome window turned a true -0.005 into a reported +0.19 at 7.4
sigma. It has its own test rather than a comment.

    uv run python -m hub.models.conformal --recalibrate --model market_baseline
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from hub.config import ModelConfig

# conf/ owns this: `model.conformal_alpha` in the dataclass, overridable from conf/config.yaml.
DEFAULT_ALPHA = ModelConfig().conformal_alpha
DEFAULT_MIN_CALIBRATION = 40


class NotEnoughCalibration(Exception):
    """Never accumulated enough past residuals to calibrate against."""


def interval(residuals: pl.Series, alpha: float) -> float:
    """Split-conformal half-width: the (1-alpha)(n+1)/n quantile of absolute residuals.

    The finite-sample correction is not decoration. With the plain (1-alpha) quantile,
    coverage sits just below nominal for small windows -- which is exactly the regime a
    rolling weekly window runs in.
    """
    n = residuals.len()
    k = min(1.0, (1.0 - alpha) * (n + 1) / n)
    return float(residuals.abs().quantile(k) or 0.0)


def rolling_coverage(df: pl.DataFrame, alpha: float = DEFAULT_ALPHA,
                     min_calibration: int = DEFAULT_MIN_CALIBRATION,
                     window: int | None = None) -> dict:
    """Walk the season forward, calibrating on past weeks and scoring the next one.

    `df` needs week, margin_mean, margin_actual. `window` caps how many weeks of history
    feed the calibration -- a season is not stationary, and an unbounded window drags
    January's errors into September.
    """
    df = df.with_columns(
        (pl.col("margin_actual") - pl.col("margin_mean")).alias("residual"))
    weeks = sorted({int(w) for w in df["week"].to_list()})

    by_week, covered, total, widths, cal_ns = [], 0, 0, [], []
    for w in weeks:
        past_weeks = [x for x in weeks if x < w]
        if window is not None:
            past_weeks = past_weeks[-window:]
        past = df.filter(pl.col("week").is_in(past_weeks))
        if past.height < min_calibration:
            continue
        q = interval(past["residual"], alpha)
        cur = df.filter(pl.col("week") == w)
        hit = (cur["residual"].abs() <= q).sum()
        by_week.append({"week": w, "n": cur.height, "half_width": q,
                        "coverage": float(hit) / cur.height,
                        "calibration_n": past.height})
        covered += int(hit)
        total += cur.height
        widths.append(q)
        cal_ns.append(past.height)

    if not total:
        raise NotEnoughCalibration(
            f"never reached {min_calibration} calibration points; the data has "
            f"{df.height} rows across {len(weeks)} weeks")

    return {
        "alpha": alpha, "nominal": 1.0 - alpha, "empirical": covered / total,
        "n_scored": total, "n_weeks_scored": len(by_week),
        "first_scored_week": by_week[0]["week"],
        "mean_width": 2.0 * sum(widths) / len(widths),
        "mean_calibration_n": sum(cal_ns) / len(cal_ns),
        "window": window, "by_week": by_week,
    }


def load_scored(model: str, base: Path | None = None,
                schedules: pl.DataFrame | None = None) -> pl.DataFrame:
    """Predictions with realised margins, for one model.

    This used to select `margin_actual` straight out of `preds`, and there is no such column
    -- the CLI in this module died on a DuckDB binder error every time it was run, which is
    the practical reason nothing consumes conformal intervals.

    `preds` records what was predicted and never what happened: `hub.publish` writes a row
    before kickoff and does not revisit it. The realised margin lives in nflverse's schedules
    as `result`, home score minus away. So scoring a prediction is a join, not a column, and
    an unplayed game simply does not survive it.
    """
    from hub import store
    preds = store.sql(
        "SELECT game_id, week, margin_mean FROM preds WHERE model = ?",
        params=[model], base=base)
    empty = preds.head(0).with_columns(pl.lit(None, pl.Float64).alias("margin_actual"))
    if preds.is_empty():
        return empty.select("week", "margin_mean", "margin_actual")
    if schedules is None:                                       # pragma: no cover
        import nflreadpy as nfl
        schedules = nfl.load_schedules()
    if "result" not in schedules.columns:
        raise ValueError("schedules is missing `result`, the realised margin")
    actual = (schedules.select("game_id",
                               pl.col("result").cast(pl.Float64).alias("margin_actual"))
              .drop_nulls("margin_actual"))
    return preds.join(actual, on="game_id", how="inner").select(
        "week", "margin_mean", "margin_actual")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.conformal",
        description="Rolling conformal calibration; reports empirical vs nominal coverage.")
    ap.add_argument("--recalibrate", action="store_true",
                    help="walk the season forward and report coverage")
    ap.add_argument("--model", default="market_baseline")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--window", type=int, default=None,
                    help="weeks of history to calibrate on; unbounded if omitted")
    ap.add_argument("--min-calibration", type=int, default=DEFAULT_MIN_CALIBRATION)
    a = ap.parse_args(argv)

    if not a.recalibrate:
        ap.print_help()
        return 0
    try:
        got = rolling_coverage(load_scored(a.model), alpha=a.alpha,
                               min_calibration=a.min_calibration, window=a.window)
    except NotEnoughCalibration as e:
        print(f"hub.models.conformal: {e}", file=sys.stderr)
        return 1

    print(f"  {a.model}, alpha {a.alpha} -> nominal coverage {got['nominal']:.1%}")
    print(f"    empirical  {got['empirical']:.1%} over {got['n_scored']} games, "
          f"weeks {got['first_scored_week']}+")
    print(f"    mean interval width {got['mean_width']:.1f} points, "
          f"calibrated on {got['mean_calibration_n']:.0f} games on average")
    gap = got["empirical"] - got["nominal"]
    print(f"    gap {gap:+.1%} -- "
          + ("covering as promised" if abs(gap) < 0.05
             else "under-covering" if gap < 0 else "over-covering, intervals too wide"))
    print("    calibration uses earlier weeks only; the week being scored is never in it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
