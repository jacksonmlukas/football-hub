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

**Since #343 the decision is the Gate's.** `verdict` used to read the per-season aggregates by
hand -- "MAE improves in every held-out season" over three numbers, no interval, no stage 2,
no stamps -- which is a bare every-season check beside `experiment.gate`, the copy ADR-0019
exists to end. It is one `Harness.run` over the paired rows now, and what the page above
calls a null is whatever the shared rule says it is: see `docs/component-projection.md`'s
restatement box.

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

from hub.cli import unavailable
from hub.ledger import Ledger, recipe
from hub.models import components
from hub.models.components import SCORING
from hub.models.experiment import Actions, GateRun, Harness, expanding_seasons

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

# `PLAYER_STATS`' five required columns, which any load of it must carry; the realised
# components are named beside them from the expected ones the opportunity table has.
_STAT_KEYS = ("season", "week", "player_id", "position", "fantasy_points_ppr")


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


# #343, this comparison's declaration. The unit that is paired is the player-season (one row
# of `pairs`, the points error summed over the seven components -- what the aggregate
# `calibrated` averaged per component), the cluster is the season, and the within-season
# repeated-measure unit is the player (`docs/method.md` rule 3).
WITHIN: tuple[str, ...] = ("player_id",)

# **Not pre-registered: the #343 lane's own choice.** `docs/gate-power.md`'s #343 section
# pre-registers the spread and injury-type ceilings and says nothing about this comparison,
# which joined the batch later (2026-10-02). This construction was declared in the commit that
# routed it, before any run, and it is flagged for the maintainer to ratify or replace -- the
# verdict does not depend on it (see the restatement in `docs/component-projection.md`).
# The declared ceiling arm (S6, #363), by name where it prints: each component's calibration
# line fitted *in sample on the held-out season's own pairs* and scored on them -- the
# `coverage` and `injury` ceilings' construction, flattered by construction, which bounds what
# any calibration of this functional form could earn on these rows.
CEILING_ARM = "each component's calibration fitted in sample on the held-out season's own pairs"

ACTIONS = Actions(
    adopt="The per-component calibration is taken: it improves the loss the product is "
          "linear in.",
    remove="The calibration worsens the loss the product is linear in: it stays out.",
    show="The calibration is not taken; the Gate could not establish that it pays.")

HARNESS = Harness(name="component_calibration", arm_a="calibrated", arm_b="raw projection",
                  within=WITHIN, ceiling_arm=CEILING_ARM, actions=ACTIONS,
                  unit="points per game of MAE", places=3,
                  arm_roots=("hub.models.component_error",))


def _points_error(frame: pl.DataFrame, fits: dict[str, tuple[float, float]]) -> np.ndarray:
    """Per-row error in points, summed over the components in `fits`: the projection passed
    through each component's own `(slope, intercept)`. The raw projection is the identity line
    `(1.0, 0.0)`. A component with no fit, or a non-finite value, contributes nothing to that
    row -- `calibrated`'s own mask, so raw and calibrated are always summed over the same set."""
    total = np.zeros(frame.height)
    for k in COMPONENTS:
        if f"p_{k}" not in frame.columns or f"a_{k}" not in frame.columns or k not in fits:
            continue
        p = frame[f"p_{k}"].to_numpy().astype(float)
        a = frame[f"a_{k}"].to_numpy().astype(float)
        c = fits[k][0] * p + fits[k][1]
        err = np.abs(a - c) * abs(SCORING[k])
        total += np.where(np.isfinite(p) & np.isfinite(a), err, 0.0)
    return total


def _fits(frame: pl.DataFrame) -> dict[str, tuple[float, float]]:
    """Each component's `(slope, intercept)` of realised on projected over `frame`, as
    `calibrated` fits them: finite rows only, and none for a component with no spread."""
    out: dict[str, tuple[float, float]] = {}
    for k in COMPONENTS:
        if f"p_{k}" not in frame.columns or f"a_{k}" not in frame.columns:
            continue
        p, a = frame[f"p_{k}"].to_numpy(), frame[f"a_{k}"].to_numpy()
        m = np.isfinite(p) & np.isfinite(a)
        if m.sum() < 2 or p[m].std() == 0:
            continue
        slope, intercept = np.polyfit(p[m], a[m], 1)
        out[k] = (float(slope), float(intercept))
    return out


def calibration_frame(paired: pl.DataFrame) -> pl.DataFrame:
    """The paired rows the Gate reads: one per held-out player-season.

    For each held-out season (`expanding_seasons`: `past` strictly earlier), the calibration is
    fitted on `past` alone and applied to `now`. `diff` is the raw projection's error minus the
    calibrated one's, in points (positive when the calibration helps, the sign `gate` adopts
    on); `ceiling_diff` is the raw error minus the in-sample calibration's -- the same lines
    fitted on `now` itself -- the declared ceiling arm. `player_id` is the within-season unit.
    """
    frames = []
    for target, past, now in expanding_seasons(paired):
        fit_past, fit_now = _fits(past), _fits(now)
        identity = dict.fromkeys(COMPONENTS, (1.0, 0.0))
        frames.append(pl.DataFrame({
            "season": [target] * now.height,
            "player_id": now["player_id"] if "player_id" in now.columns
            else [f"row{i}" for i in range(now.height)],
            "diff": (_points_error(now, {k: identity[k] for k in fit_past})
                     - _points_error(now, fit_past)),
            "ceiling_diff": (_points_error(now, {k: identity[k] for k in fit_now})
                             - _points_error(now, fit_now)),
        }))
    return pl.concat(frames) if frames else pl.DataFrame()


def gate_run(frame: pl.DataFrame, *, publish: bool = False, seasons: Sequence[int] = (),
             seed: int = 0, ledger: Ledger | None = None) -> GateRun:
    """The calibration against the raw projection, through the one Gate (#343).

    `Harness.decide` is the pure half; `publish` is `Harness.run` -- stage 2, the width review
    and the four stamps in `.lines`, and a Ledger entry.
    """
    harness = HARNESS
    if publish:
        return harness.run(frame, seed=seed, ledger=ledger,
                           recipe=recipe(seasons=list(seasons)))
    return harness.decide(frame, seed=seed)


def verdict(frame: pl.DataFrame, rounds: Sequence[dict[str, float]] = (), *,
            run: GateRun | None = None) -> tuple[str, str]:
    """Whether the per-component calibration is worth taking: the Gate's answer.

    **Not a rule of this module's own since #343.** It was "MAE improves in every held-out
    season" over the per-season aggregates -- the bare every-season check, no interval, no
    stage 2, no stamps -- and `docs/component-projection.md`'s calibration null rested on it.
    The label is `experiment.gate`'s own (`ADOPT`, `REMOVE`, `SHOW`, `NOT-RUNNABLE`): a
    `NOT-RUNNABLE` says the design cannot tell a real gain from a perfect one over this many
    seasons, which is not the claim "the correction does not pay" this used to print.

    `rounds` is optional context: the RMSE the calibration buys, which is the half of the
    finding the points-error Gate does not read -- least squares minimises it by definition,
    so a calibration improving RMSE while MAE does not is the expected shape.

    `run` is what `main` hands in so the gate is run once, published, and read here.
    """
    got = run if run is not None else gate_run(frame)
    label, sentence = got.verdict
    note = sentence
    if rounds and label != "ADOPT":
        rmse_gain = [r["raw_rmse"] - r["cal_rmse"] for r in rounds]
        note += (f" The same calibration improves RMSE in {sum(g > 0 for g in rmse_gain)} of "
                 f"{len(rmse_gain)} held-out seasons (mean {np.mean(rmse_gain):+.3f}); least "
                 f"squares minimises RMSE by definition, so the split is the expected shape "
                 f"rather than a surprise, and fantasy points are linear, so MAE is the loss "
                 f"that decides.")
    return label, note


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
    from hub.fetch import nflverse
    out = []
    for prev, nxt in itertools.pairwise(seasons):
        o = nflverse.load("ff_opportunity", [prev])
        have = {k: v for k, v in EXPECTED.items() if v in o.columns}
        pos = (o.select("player_id", "position").drop_nulls()
                 .unique(subset="player_id", keep="first"))
        proj = (o.group_by("player_id")
                 .agg([pl.col(v).sum().alias(f"p_{k}") for k, v in have.items()]
                      + [pl.len().alias("pg")])
                 .with_columns([(pl.col(f"p_{k}") / pl.col("pg")).alias(f"p_{k}") for k in have])
                 .filter(pl.col("pg") >= MIN_GAMES))
        s = nflverse.load("player_stats", [nxt], cols=[*_STAT_KEYS, *have])
        act = (s.group_by("player_id")
                .agg([pl.col(k).sum().alias(f"a_{k}") for k in have] + [pl.len().alias("ag")])
                .with_columns([(pl.col(f"a_{k}") / pl.col("ag")).alias(f"a_{k}") for k in have])
                .filter(pl.col("ag") >= MIN_GAMES))
        out.append(proj.join(act, on="player_id", how="inner")
                       .join(pos, on="player_id", how="left")
                       .with_columns(pl.lit(nxt).alias("season")))
    return pl.concat(out) if out else pl.DataFrame()


def report(card: pl.DataFrame, rounds: Sequence[dict[str, float]],
           decision: str | None = None) -> list[str]:
    """Lines, not prints -- the reason `hub.draft.report` exists. `decision` is the verdict's
    sentence when a Gate run produced one (#343); the descriptive tables stand without it."""
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
    if decision is not None:
        lines += ["", f"  {decision}"]
    return lines


def main(argv: Sequence[str] | None = None, *,
         ledger: Ledger | None = None) -> int:  # pragma: no cover - network
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
    try:
        d = pairs(seasons)
    except Exception as e:
        return unavailable("hub.models.component_error", "the nflverse player seasons", e)
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
    frame = calibration_frame(d)
    run = gate_run(frame, publish=True, seasons=got, ledger=ledger)
    label, note = verdict(frame, rounds, run=run)
    print("\n".join(report(card, rounds, note if label == "NOT-RUNNABLE" else f"{label}: {note}")))
    print("\n  the Gate, through `experiment.run_gate` (#343):")
    print("\n".join(run.lines))

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
