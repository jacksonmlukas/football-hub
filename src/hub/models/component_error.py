"""Where the component projection's error actually lives, priced in fantasy points.

`docs/component-projection.md` established the shape of this model: volume carries forward,
touchdowns regress to the positional rate, and a volume/efficiency shrinkage toward positional
means came back null. What it did not establish is **which component the remaining error is
in**, and that is the question a projection needs answered before anyone works on it.

The unit matters. A yard of receiving error and a touchdown of receiving error are not
comparable until both are multiplied by what the league pays for them, so every figure here is
weighted by `components.SCORING`. Ranked that way the receiving game is roughly two thirds of
the budget and passing volume is nearly noise-free, which is not what the raw error columns
suggest.

Committed rather than left in a notebook because the numbers are cited as a reason -- ADR-0007's
trigger. Two of them are load-bearing:

  * **Every component is over-dispersed.** Regressing realised on projected gives a slope below
    one for all seven components in all four season pairs, 28 of 28. Deviations from the mean
    are worth less than the projection claims.
  * **Correcting that does not help.** Applying each component's own calibration, fitted on
    earlier pairs only, *improves* RMSE and *worsens* MAE in both held-out seasons. Fantasy
    points are linear, so MAE is the loss that matters and the correction is not taken. The
    null is recorded here rather than rediscovered.

The comparison is prior-season **expected** components against next-season **realised** ones.
Expected rather than realised on the projection side because that is what the Board already
carries, and because ff_opportunity's expected touchdowns are an expectation already -- the
thing `docs/component-projection.md` found a player's own rate carries no information beyond.

    uv run python -m hub.models.component_error --run
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Sequence

import numpy as np
import polars as pl

from hub.models import components
from hub.models.components import SCORING
from hub.models.experiment import expanding_seasons

# The components ff_opportunity prices and this league scores. `interceptions` is available
# upstream but is a quarterback-only term whose error is a rounding difference beside the rest;
# fumbles have no expected column at all, only a realised one.
COMPONENTS: tuple[str, ...] = (
    "receptions", "receiving_yards", "receiving_tds",
    "rushing_yards", "rushing_tds", "passing_yards", "passing_tds",
)

# Upstream's expected-stat column for each. The Board keeps only the pre-summed total and drops
# these; see issue #2, which gives the mapping one owner.
EXPECTED: dict[str, str] = {k: components.EXPECTED[k][0] for k in COMPONENTS}

# Below this a per-game rate is a handful of snaps and the pairing is noise on both sides.
MIN_GAMES = 6


def scorecard(paired: pl.DataFrame) -> pl.DataFrame:
    """Per-component accuracy, with the error priced in points.

    `paired` carries `p_<component>` and `a_<component>` per-game columns. Returns one row per
    component: correlation, the slope of realised on projected, the mean signed bias, the mean
    absolute error in the component's own unit, and that error multiplied by what the league
    pays for the component -- which is the only column the seven can be ranked on.
    """
    rows = []
    for k in COMPONENTS:
        if f"p_{k}" not in paired.columns or f"a_{k}" not in paired.columns:
            continue
        p, a = paired[f"p_{k}"].to_numpy(), paired[f"a_{k}"].to_numpy()
        m = np.isfinite(p) & np.isfinite(a)
        p, a = p[m], a[m]
        if len(p) < 2 or p.std() == 0:
            continue
        mae = float(np.abs(a - p).mean())
        rows.append({
            "component": k, "n": len(p),
            "corr": float(np.corrcoef(p, a)[0, 1]),
            "slope": float(np.polyfit(p, a, 1)[0]),
            "bias": float((a - p).mean()),
            "mae": mae,
            "points": mae * abs(SCORING[k]),
        })
    if not rows:
        return pl.DataFrame(schema={"component": pl.Utf8, "n": pl.Int64, "corr": pl.Float64,
                                    "slope": pl.Float64, "bias": pl.Float64, "mae": pl.Float64,
                                    "points": pl.Float64})
    return pl.DataFrame(rows).sort("points", descending=True)


def calibrated(train: pl.DataFrame, test: pl.DataFrame) -> dict[str, float]:
    """Fit each component's calibration on `train`, apply to `test`, score both losses.

    Leak-free by construction: the line applied to a season is fitted only on pairs that ended
    before it. Both losses are reported because they disagree, and the disagreement is the
    finding -- least squares minimises RMSE by definition, so a calibration improving RMSE and
    worsening MAE is the expected shape rather than a surprise, and MAE is the one fantasy
    points are linear in.
    """
    out = {"raw_mae": 0.0, "cal_mae": 0.0, "raw_rmse": 0.0, "cal_rmse": 0.0, "n": 0.0}
    for k in COMPONENTS:
        if f"p_{k}" not in train.columns:
            continue
        p0, a0 = train[f"p_{k}"].to_numpy(), train[f"a_{k}"].to_numpy()
        m0 = np.isfinite(p0) & np.isfinite(a0)
        if m0.sum() < 2 or p0[m0].std() == 0:
            continue
        slope, intercept = np.polyfit(p0[m0], a0[m0], 1)
        p, a = test[f"p_{k}"].to_numpy(), test[f"a_{k}"].to_numpy()
        m = np.isfinite(p) & np.isfinite(a)
        p, a = p[m], a[m]
        if not len(p):
            continue
        w = abs(SCORING[k])
        c = slope * p + intercept
        out["raw_mae"] += float(np.abs(a - p).mean()) * w
        out["cal_mae"] += float(np.abs(a - c).mean()) * w
        out["raw_rmse"] += float(np.sqrt(((a - p) ** 2).mean())) * w
        out["cal_rmse"] += float(np.sqrt(((a - c) ** 2).mean())) * w
        out["n"] = float(len(p))
    return out


def verdict(rounds: Sequence[dict[str, float]]) -> tuple[str, str]:
    """Whether the per-component calibration is worth taking. Pre-stated: it must improve the
    loss the product is linear in, in every held-out season."""
    if not rounds:
        return "NULL", "nothing measured -- no held-out season"
    mae_gain = [r["raw_mae"] - r["cal_mae"] for r in rounds]
    rmse_gain = [r["raw_rmse"] - r["cal_rmse"] for r in rounds]
    if all(g > 0 for g in mae_gain):
        return "ADOPT", (f"calibration improves MAE in all {len(rounds)} held-out seasons "
                         f"(mean {np.mean(mae_gain):+.3f} points a game)")
    won = sum(g > 0 for g in mae_gain)
    return "NULL", (
        f"NOT TAKEN: calibration improves MAE in {won} of {len(mae_gain)} held-out seasons "
        f"(mean {np.mean(mae_gain):+.3f} points a game -- a wash), while improving RMSE in "
        f"{sum(g > 0 for g in rmse_gain)} of {len(rmse_gain)} (mean {np.mean(rmse_gain):+.3f}). "
        f"Least squares minimises RMSE by definition, so the split is the expected shape rather "
        f"than a surprise; fantasy points are linear, so MAE is the loss that decides. The "
        f"over-dispersion is real and this correction for it does not pay for itself.")


def attribution(paired: pl.DataFrame, by: str | None = None) -> pl.DataFrame:
    """Split the points gap into the components that produce it, so that they sum to it.

    "We are a point high on him" becomes "we have him at two more receptions and eight more
    receiving yards". Each component contributes `(projected - realised) * what the league pays
    for it`, and the contributions add up to the gap in the total exactly -- which is the
    property that makes the split an explanation rather than a decoration, and which a test
    pins.

    **This is against what happened, not against another projection, and that is a limit rather
    than a choice.** Decomposing a disagreement needs parts on both sides, and ESPN publishes a
    total and no parts -- `proj_ppg` is one column. So the gap against ESPN can be decomposed on
    our side only: it says what our number is made of, not which stat we disagree about. Against
    the realised outcome both sides have components and the split is complete.

    `by` groups the result -- position, a board tier, anything carried on the frame -- because
    "where is the error" is usually a question about a kind of player rather than one player.
    """
    if by is not None and by not in paired.columns:
        raise ValueError(
            f"cannot group the decomposition by {by!r}: the paired frame carries "
            f"{sorted(paired.columns)[:6]}... Add it to `pairs` if it should be there.")
    have = [k for k in COMPONENTS
            if f"p_{k}" in paired.columns and f"a_{k}" in paired.columns]
    if not have or paired.is_empty():
        return pl.DataFrame(schema={"component": pl.Utf8, "points": pl.Float64})

    contrib = paired.with_columns(
        [((pl.col(f"p_{k}") - pl.col(f"a_{k}")) * SCORING[k]).alias(f"d_{k}") for k in have])
    keys = [by] if by else []
    agg = (contrib.group_by(keys).agg(
               [pl.col(f"d_{k}").mean().alias(k) for k in have] + [pl.len().alias("n")])
           if keys else
           contrib.select([pl.col(f"d_{k}").mean().alias(k) for k in have]
                          + [pl.len().alias("n")]))
    out = agg.unpivot(index=[*keys, "n"], variable_name="component", value_name="points")
    return out.sort([*keys, "points"], descending=[*([False] * len(keys)), True])


def pairs(seasons: Sequence[int]) -> pl.DataFrame:  # pragma: no cover - network
    """Prior-season expected components against next-season realised ones, per game."""
    import nflreadpy as nfl
    out = []
    for prev, nxt in itertools.pairwise(seasons):
        o = nfl.load_ff_opportunity(seasons=[prev], stat_type="weekly")
        have = {k: v for k, v in EXPECTED.items() if v in o.columns}
        pos = (o.select("player_id", "position").drop_nulls()
                 .unique(subset="player_id", keep="first"))
        proj = (o.group_by("player_id")
                 .agg([pl.col(v).sum().alias(f"p_{k}") for k, v in have.items()]
                      + [pl.len().alias("pg")])
                 .with_columns([(pl.col(f"p_{k}") / pl.col("pg")).alias(f"p_{k}") for k in have])
                 .filter(pl.col("pg") >= MIN_GAMES))
        s = nfl.load_player_stats(seasons=[nxt])
        act = (s.group_by("player_id")
                .agg([pl.col(k).sum().alias(f"a_{k}") for k in have] + [pl.len().alias("ag")])
                .with_columns([(pl.col(f"a_{k}") / pl.col("ag")).alias(f"a_{k}") for k in have])
                .filter(pl.col("ag") >= MIN_GAMES))
        out.append(proj.join(act, on="player_id", how="inner")
                       .join(pos, on="player_id", how="left")
                       .with_columns(pl.lit(nxt).alias("season")))
    return pl.concat(out) if out else pl.DataFrame()


def report(card: pl.DataFrame, rounds: Sequence[dict[str, float]]) -> list[str]:
    """Lines, not prints -- the reason `hub.draft.report` exists."""
    lines = [f"\n  {'component':17} {'n':>5} {'corr':>6} {'slope':>7} {'bias':>9} {'MAE':>8}"
             f" {'pts/gm':>8}"]
    for r in card.iter_rows(named=True):
        lines.append(f"  {r['component']:17} {r['n']:>5} {r['corr']:>6.3f} {r['slope']:>7.3f} "
                     f"{r['bias']:>+9.3f} {r['mae']:>8.3f} {r['points']:>8.3f}")
    total = float(card["points"].sum()) if card.height else 0.0
    rec = float(card.filter(pl.col("component").str.starts_with("rec"))["points"].sum())
    if total:
        lines += ["", f"  total error budget {total:.2f} points a game; the receiving game is "
                      f"{rec / total:.0%} of it"]
    under = card.filter(pl.col("slope") < 1.0).height
    lines.append(f"  over-dispersed components (slope < 1): {under} of {card.height}")
    if rounds:
        lines += ["", f"  {'held out':>9} {'raw MAE':>9} {'cal MAE':>9} {'raw RMSE':>10}"
                      f" {'cal RMSE':>10}"]
        for r in rounds:
            lines.append(f"  {int(r['season']):>9} {r['raw_mae']:>9.3f} {r['cal_mae']:>9.3f} "
                         f"{r['raw_rmse']:>10.3f} {r['cal_rmse']:>10.3f}")
    lines += ["", f"  {verdict(rounds)[1]}"]
    return lines


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(
        prog="hub.models.component_error",
        description="Where the component projection's error lives, priced in fantasy points.")
    ap.add_argument("--run", action="store_true", help="fetch and measure")
    ap.add_argument("--seasons", default="2021,2022,2023,2024,2025")
    ap.add_argument("--by", default=None,
                    help="decompose the gap by a grouping column, e.g. position")
    a = ap.parse_args(list(argv) if argv is not None else None)
    if not a.run:
        ap.print_help()
        return 0

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    d = pairs(seasons)
    if d.is_empty():
        print("  nothing paired -- need at least two consecutive seasons", file=sys.stderr)
        return 1
    card = scorecard(d)
    got = sorted(set(d["season"].to_list()))
    # `expanding_seasons`, not a hand-written `season < target`: docs/method.md rule #2 has one
    # statement in code and a guard test that fails when a second one appears. It found this.
    rounds = []
    for target, past, now in expanding_seasons(d):
        r = calibrated(past, now)
        r["season"] = float(target)
        rounds.append(r)
    print("\n  prior-season expected components vs next-season realised, per game")
    print(f"  {d.height:,} player-seasons over {len(got)} pairs, >= {MIN_GAMES} games both sides")
    print("\n".join(report(card, rounds)))

    # The gap, split into the components that produce it. Against what happened, not against
    # another projection: decomposing a disagreement needs parts on both sides, and ESPN
    # publishes a total and no parts.
    att = attribution(d, by=a.by)
    if att.height:
        print(f"\n  the gap, decomposed -- projected minus realised, in points a game"
              f"{f' by {a.by}' if a.by else ''}\n")
        for r in att.iter_rows(named=True):
            lead = f"{r[a.by]:<6} " if a.by else ""
            print(f"  {lead}{r['component']:20} {r['points']:+8.3f}")
        if not a.by:
            print(f"  {'':20} {'':>8}\n  {'total':20} {att['points'].sum():+8.3f}")
    return 0


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
