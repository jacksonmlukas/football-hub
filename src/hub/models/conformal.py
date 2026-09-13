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

**Earlier in time, not earlier in number (#262).** The walk was over week numbers with no
season on the row, so once the store held two seasons, week 6 of the earlier one was
calibrated on weeks 1-5 of *every* season -- the later one included, which is the
within-season split wearing a temporal split's name that `eval._holdout_window` fixed for
its sibling (#168). The walk is over `(season, week)` cells in time order now: a later
season's rows never reach an earlier season's window, and the later season's first weeks
calibrate on the whole of the season before, bounded by `window` like any other history.
`--season` scores one season alone.

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


def _cell(season: int, week: int) -> pl.Expr:
    """The rows of one (season, week). An explicit pair of equalities rather than a struct
    membership test, for `eval.compare`'s reason: `season` and `week` arrive as Int32 from
    the store and as python ints from the walk, and #25 was a filter comparing a column
    against the wrong dtype."""
    return (pl.col("season") == season) & (pl.col("week") == week)


def rolling_coverage(df: pl.DataFrame, alpha: float = DEFAULT_ALPHA,
                     min_calibration: int = DEFAULT_MIN_CALIBRATION,
                     window: int | None = None, season: int | None = None) -> dict:
    """Walk the seasons forward, calibrating on past (season, week) cells and scoring the
    next one.

    `df` needs week, margin_mean, margin_actual, and season where it spans more than one --
    a frame without the column is one season, which is what every frame built before #262
    was and the case the week-number walk got right. `window` caps how many cells of
    history feed the calibration -- a season is not stationary, and an unbounded window
    drags January's errors into September. Cells are ordered season first, so the window
    crosses a season boundary the way time does: the later season's week 1 calibrates on
    the earlier season's last weeks, not on nothing.

    `season` scores that season alone, calibrated only on itself; the walk keyed on the
    pair is right without it, and a reader asking about one season should not have to
    subtract the other from the report.
    """
    if "season" not in df.columns:
        df = df.with_columns(pl.lit(0, dtype=pl.Int32).alias("season"))
    if season is not None:
        df = df.filter(pl.col("season") == season)
    df = df.with_columns(
        (pl.col("margin_actual") - pl.col("margin_mean")).alias("residual"))
    cells = sorted({(int(s), int(w)) for s, w in
                    zip(df["season"].to_list(), df["week"].to_list(), strict=True)})

    by_week, covered, total, widths, cal_ns = [], 0, 0, [], []
    for s, w in cells:
        # Strictly earlier in time: season first, then week. A tuple comparison is the
        # timeline, and this is the whole of #262 -- `x < w` over week numbers let a later
        # season's September into an earlier season's window.
        past_cells = [c for c in cells if c < (s, w)]
        if window is not None:
            past_cells = past_cells[-window:]
        past = (df.filter(pl.any_horizontal(*[_cell(*c) for c in past_cells]))
                if past_cells else df.clear())
        if past.height < min_calibration:
            continue
        q = interval(past["residual"], alpha)
        cur = df.filter(_cell(s, w))
        hit = (cur["residual"].abs() <= q).sum()
        by_week.append({"season": s, "week": w, "n": cur.height, "half_width": q,
                        "coverage": float(hit) / cur.height,
                        "calibration_n": past.height})
        covered += int(hit)
        total += cur.height
        widths.append(q)
        cal_ns.append(past.height)

    if not total:
        raise NotEnoughCalibration(
            f"never reached {min_calibration} calibration points; the data has "
            f"{df.height} rows across {len(cells)} season-weeks")

    return {
        "alpha": alpha, "nominal": 1.0 - alpha, "empirical": covered / total,
        "n_scored": total, "n_weeks_scored": len(by_week),
        "first_scored_season": by_week[0]["season"],
        "first_scored_week": by_week[0]["week"],
        "mean_width": 2.0 * sum(widths) / len(widths),
        "mean_calibration_n": sum(cal_ns) / len(cal_ns),
        "window": window, "season": season, "by_week": by_week,
    }


def load_scored(model: str, base: Path | None = None,
                schedules: pl.DataFrame | None = None, season: int | None = None) -> pl.DataFrame:
    """Predictions with realised margins, for one model, carrying the season (#262).

    This used to select `margin_actual` straight out of `preds`, and there is no such column
    -- the CLI in this module died on a DuckDB binder error every time it was run, which is
    the practical reason nothing consumes conformal intervals.

    `preds` records what was predicted and never what happened: `hub.publish` writes a row
    before kickoff and does not revisit it. The realised margin lives in nflverse's schedules
    as `result`, home score minus away. So scoring a prediction is a join, not a column, and
    an unplayed game simply does not survive it.
    """
    from hub import store
    empty_shape = pl.DataFrame(schema={"season": pl.Int32, "week": pl.Int64,
                                       "margin_mean": pl.Float64, "margin_actual": pl.Float64})
    # One row per game. The calibration quantile is a rank over the window, so counting each
    # residual once per fitted version does not merely inflate `n` -- it reweights the
    # quantile toward whichever games happen to have been re-fitted most often.
    #
    # The family, not the exact name (#284): `market_baseline-qb` is the row the quarterback
    # layer moved, and the interval being calibrated is the module's published number for
    # every game. On the exact name the window would score the unadjusted subset -- every
    # game but the backup-quarterback ones -- and call the coverage the model's.
    got = store.predictions(model=model, base=base, family=True, season=season)
    if got.is_empty():
        return empty_shape
    # The season rides along so `rolling_coverage` can walk time rather than week numbers.
    preds = got.select("game_id", "season", "week", "margin_mean")
    if schedules is None:                                       # pragma: no cover
        import nflreadpy as nfl
        schedules = nfl.load_schedules()
    if "result" not in schedules.columns:
        raise ValueError("schedules is missing `result`, the realised margin")
    actual = (schedules.select("game_id",
                               pl.col("result").cast(pl.Float64).alias("margin_actual"))
              .drop_nulls("margin_actual"))
    return preds.join(actual, on="game_id", how="inner").select(
        "season", "week", "margin_mean", "margin_actual")


def _root(v: str | None) -> Path | None:
    return Path(v) if v else None


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
    ap.add_argument("--season", type=int, default=None,
                    help="score one season alone, calibrated on itself; every season in "
                         "the store, walked in time order, if omitted")
    ap.add_argument("--store", default=None,
                    help="processed-store root; defaults to this repo's. Overridable "
                         "so the CLI can be driven against an empty or backup store")
    a = ap.parse_args(argv)

    if not a.recalibrate:
        ap.print_help()
        return 0
    try:
        got = rolling_coverage(load_scored(a.model, base=_root(a.store), season=a.season),
                               alpha=a.alpha, min_calibration=a.min_calibration,
                               window=a.window, season=a.season)
    except NotEnoughCalibration as e:
        print(f"hub.models.conformal: {e}", file=sys.stderr)
        return 1

    print(f"  {a.model}, alpha {a.alpha} -> nominal coverage {got['nominal']:.1%}")
    print(f"    empirical  {got['empirical']:.1%} over {got['n_scored']} games, "
          f"from {got['first_scored_season']} week {got['first_scored_week']}+")
    print(f"    mean interval width {got['mean_width']:.1f} points, "
          f"calibrated on {got['mean_calibration_n']:.0f} games on average")
    gap = got["empirical"] - got["nominal"]
    print(f"    gap {gap:+.1%} -- "
          + ("covering as promised" if abs(gap) < 0.05
             else "under-covering" if gap < 0 else "over-covering, intervals too wide"))
    print("    calibration uses earlier season-weeks only; the week being scored is never "
          "in it, and a later season never reaches an earlier one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
