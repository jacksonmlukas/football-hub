"""Does the shipped weekly interval cover, and does the survivor price hold by spread?

Two distributions that no gate has ever scored, graded here as committed code because
[ADR-0007](../../../docs/adr/0007-measurements-that-steer-the-product-are-committed-code.md)
requires it of a number that steers anything -- and these now do, through `--gate`.

**What is graded is the deployed moment function.** `hub.models.predict.moments` is what
`draft/optimize.py`, `season/roster.py` and `season/lineup_gate.py` actually call, and this
harness calls the same function rather than restating `WEEKLY_K[pos] * sqrt(mu)` beside it.
The interval bounds come from `predict.skewed` at `z = Phi^-1(p)` -- the transform is
monotone in `z`, so that is the quantile -- which is the same function `draft/season.py`
draws through. Nothing about the distribution is re-implemented here; a restatement that
agreed today is how the grader and the graded drift apart later.

**The centre is the whole argument.** `docs/weekly-coverage.md` centred on each
player-season's own realised mean, and said so: that is a *lookahead*, because the number
being used to build the interval was computed from the very weeks the interval is then
scored against. It flatters the result in one specific way -- the centre is exactly right by
construction, so the only spread the interval has to cover is the week-to-week noise, and
none of the error in knowing where the centre is.

So `--centre prior` is the default and is the real measurement: for each week, the centre is
the mean of that player's **strictly earlier** weeks in the same season. That is the rule
`hub.models.conformal` already states for its calibration window and the one
`docs/track-record.md` rule 1 forbids breaking, applied here. `--centre realised` reproduces
the document, and is kept for exactly that -- a superseded number you cannot reproduce is a
number nobody can check.

The two answer different questions and the difference is large; see the 2026-09-07
restatement in `docs/weekly-coverage.md`.

**The survivor half.** `season/survivor.py` prices every pick as
`normal_cdf(close_spread / MARGIN_SD)` and then multiplies those numbers into a survival
probability. Nothing has ever asked whether that price holds *at the spreads survivor
actually picks at*, which are the big favourites and nowhere near the middle of the
distribution. Graded by spread bucket rather than by probability bucket, because a miss
concentrated in one spread range is what a pick rule would walk into.

    uv run python -m hub.models.coverage --measure
    uv run python -m hub.models.coverage --measure --centre realised
    uv run python -m hub.models.coverage --survivor
    uv run python -m hub.models.coverage --gate          # exits 1 on a refusal
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import polars as pl

from hub import jsonio
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.models import predict
from hub.models.scoring_rules import reliability_by
from hub.paths import PROCESSED

# Nothing here is fitted. Every number below is either a filter this measurement inherits
# from the document it supersedes, a nominal level, or a threshold pre-registered before the
# answer was looked at -- and none of them reaches a prediction, because this module grades
# `hub.models.predict` and never calls it to forecast anything. Registering them in
# `config_digest` would stamp a prediction with the settings of its own grader.
NOT_FITTED_BECAUSE = (
    "grading harness; its thresholds are pre-registered filters, not measured constants, "
    "and no prediction is made through this module"
)

Centre = Literal["prior", "realised"]

# The window `docs/weekly-coverage.md` measured over, so `--centre realised` reproduces it
# without the caller having to know which five seasons those were.
SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

# The positions the weekly laws are fitted for, read from the league rather than written out
# again: `WEEKLY_K` and `WEEKLY_SKEW` fall back to a pooled value for anything else, and
# grading a fallback tells you about the fallback. A superflex league would move
# `RosterConfig` and this follows it.
POSITIONS: tuple[str, ...] = DRAFTED_POSITIONS

# A player-season is in the sample if it has this many scoring weeks and averages this many
# points. Both from the document, unchanged, so the two centres are compared on one filter.
MIN_WEEKS = 8
MIN_MU = 2.0

# Under `--centre prior` a week is scored only once the player has this many earlier weeks
# behind him. Below it the centre is mostly noise and the measurement becomes a statement
# about small-sample means rather than about the interval. Four is the smallest number at
# which the centre's own standard error is under half the weekly spread.
MIN_PRIOR = 4

# The two nominal levels, as (lower p, upper p). 80% is the interval `season/lineup.py`'s
# win probability is effectively asserting; 68% is one sigma, reported because a miss that
# is about the *shape* rather than the width shows up as the two disagreeing.
LEVELS: tuple[tuple[float, float], ...] = ((0.10, 0.90), (0.16, 0.84))

# Pre-registered before the prior-centre numbers were looked at, and the only thing `--gate`
# reads: empirical coverage must sit within this of nominal. Two points is roughly three
# standard errors at n = 10,000, so it is a band a calibrated model clears comfortably and a
# real miss does not.
BAND = 0.02

# The gate is read off the weeks whose interval is *not* pinned at the zero floor. A clipped
# lower bound cannot be fallen below, so those weeks report a coverage the model did not
# earn, and pooling them in hides the miss -- which is the document's own diagnosis, used
# here as the reason to exclude them rather than as a footnote.
GATE_SUBSET = "unclipped"

# Spread buckets for the survivor price, home-relative and in points. The top bucket is where
# survivor lives: `season/survivor.py` picks the biggest favourite on the board, so a bucket
# that pools a 3-point favourite with a 13-point one answers a question nobody asks of it.
SPREAD_EDGES: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 14.0, 30.0)

# What counts as "the favourites survivor actually picks", for the headline the gate reads.
SURVIVOR_SPREAD = 7.0

# Where `--measure --write` leaves its answer and where `hub.publish` reads it from. Under
# `data/processed/` rather than `site/data/` because the site copy is published output and
# this is the measurement behind it; the publisher joins the two.
ARTIFACT = PROCESSED / "interval_coverage.json"


class NotEnoughWeeks(Exception):
    """No player-season survived the filters, so there is nothing to grade."""


def _z(p: float) -> float:
    """The standard normal quantile, from the standard library rather than scipy.

    `hub.models.market.normal_cdf` is this function's inverse and is written out the same
    way, with `math`. Adding a dependency to invert a normal is not a trade this repo makes.
    """
    return statistics.NormalDist().inv_cdf(p)


def player_weeks(stats: pl.DataFrame) -> pl.DataFrame:
    """(player_id, position, season, week, points) for regular-season skill players.

    Weeks are weeks he recorded stats. Byes and missed games are therefore out of the sample
    entirely, which is a real limit and the same one the document names: this grades the
    distribution of a week he played, not the distribution of a week.
    """
    need = {"player_id", "position", "season", "week", "season_type",
            "fantasy_points_ppr"}
    missing = sorted(need - set(stats.columns))
    if missing:
        raise ValueError(f"player_stats is missing {missing}")
    return (stats.filter(pl.col("season_type") == "REG")
                 .filter(pl.col("position").is_in(list(POSITIONS)))
                 .select("player_id", "position", "season", "week",
                         pl.col("fantasy_points_ppr").fill_null(0.0)
                           .cast(pl.Float64).alias("points"))
                 .sort(["player_id", "season", "week"]))


def centred(pw: pl.DataFrame, centre: Centre = "prior", *, min_weeks: int = MIN_WEEKS,
            min_prior: int = MIN_PRIOR, min_mu: float = MIN_MU) -> pl.DataFrame:
    """Attach the centre each week's interval is built around.

    `"realised"` is the document's: one number per player-season, that season's own mean.
    Every week is inside its own centre, which is the lookahead.

    `"prior"` is the honest one: the running mean of the weeks *before* this one. A week is
    kept once `min_prior` earlier weeks exist, so the first few weeks of every player-season
    leave the sample rather than being graded against a centre built from one game.

    The `min_mu` filter applies to whichever centre is in use, so the two are compared on the
    same rule and not on the same *rows* -- they cannot be, and pretending otherwise by
    fixing the row set to the realised-centre sample would smuggle the lookahead back in
    through the sample definition.
    """
    if centre not in ("prior", "realised"):
        raise ValueError(f"centre must be 'prior' or 'realised', got {centre!r}")
    if centre == "realised":
        by = ["player_id", "season"]
        g = pw.group_by(by).agg(pl.len().alias("weeks"),
                                pl.col("points").mean().alias("centre"))
        keep = g.filter((pl.col("weeks") >= min_weeks) & (pl.col("centre") >= min_mu))
        out = (pw.join(keep.select(*by, "centre"), on=by, how="inner")
                 .with_columns(pl.lit(None, dtype=pl.Int64).alias("n_prior")))
    else:
        over = ["player_id", "season"]
        out = (pw.with_columns(
                    pl.col("points").cum_sum().over(over).alias("_cs"),
                    pl.col("points").cum_count().over(over).alias("_cn"))
                 .with_columns(
                    ((pl.col("_cs") - pl.col("points"))
                     / (pl.col("_cn") - 1)).alias("centre"),
                    (pl.col("_cn") - 1).cast(pl.Int64).alias("n_prior"))
                 .drop("_cs", "_cn")
                 .filter((pl.col("n_prior") >= min_prior) & (pl.col("centre") >= min_mu)))
    if out.is_empty():
        raise NotEnoughWeeks(
            f"no player-week survived centre={centre!r} with min_weeks={min_weeks}, "
            f"min_prior={min_prior}, min_mu={min_mu}")
    return out


def graded(sample: pl.DataFrame) -> pl.DataFrame:
    """The sample with the deployed moments and the interval bounds attached.

    `predict.moments` reads its mean off `proj_blend`/`proj_ppg`/`xfp_per_game`, so the
    centre is handed to it as `proj_ppg` -- the same door a projection comes through. What
    comes back is the shipped `mu`, `sd` and `skew` and not a local copy of the laws behind
    them.

    `p10_raw` is the lower bound *before* `skewed()`'s clip at zero, and is the only piece of
    that function reproduced here. It has to be: the clip is what the floor split tests, and
    there is no way to ask whether a bound was clipped from the clipped bound alone.
    """
    m = predict.moments(sample.with_columns(pl.col("centre").alias("proj_ppg")))
    mu, sd, sk = (m["mu"].to_numpy(), m["sd"].to_numpy(), m["skew"].to_numpy())
    cols = {f"q{int(p * 100):02d}": predict.skewed(mu, sd, sk, _z(p))
            for pair in LEVELS for p in pair}
    a = np.maximum(sk, predict.MIN_SKEW) / 6.0
    z10 = _z(LEVELS[0][0])
    cols["p10_raw"] = mu + sd * (z10 + a * (z10 ** 2 - 1.0)) / np.sqrt(1.0 + 2.0 * a ** 2)
    # The counterfactual for "does the skew earn its place": the same mu and sd read through
    # a plain normal. Deliberately *unclipped*, which is what the document compared against
    # -- the clip is a separate effect and the floor split is where it is isolated.
    cols["n10"] = mu + sd * z10
    cols["n90"] = mu + sd * _z(LEVELS[0][1])
    return m.with_columns(**{k: pl.Series(k, v) for k, v in cols.items()})


def _row(label: str, g: pl.DataFrame) -> dict[str, Any]:
    """One line of the coverage table."""
    y = g["points"].to_numpy()
    lo, hi = g["q10"].to_numpy(), g["q90"].to_numpy()
    lo68, hi68 = g["q16"].to_numpy(), g["q84"].to_numpy()
    n10, n90 = g["n10"].to_numpy(), g["n90"].to_numpy()
    # Realised spread against model spread, one reading per player-season so that a player
    # with nineteen weeks does not count nineteen times, then averaged over player-seasons.
    # The residual is taken around each row's own mu, which is what makes this work for a
    # moving centre as well as a fixed one.
    per = (g.with_columns((pl.col("points") - pl.col("mu")).alias("_r"))
            .group_by(["player_id", "season"])
            .agg(pl.col("_r").std().alias("_sd"), pl.col("sd").mean().alias("_m"),
                 pl.len().alias("_n"))
            .filter((pl.col("_n") >= 2) & (pl.col("_m") > 0)))
    ratio = (float(cast(float, (per["_sd"] / per["_m"]).mean()))
             if per.height else float("nan"))
    return {
        "group": label, "n": int(g.height),
        "cov80": float(np.mean((y >= lo) & (y <= hi))),
        "cov68": float(np.mean((y >= lo68) & (y <= hi68))),
        "below_p10": float(np.mean(y < lo)), "above_p90": float(np.mean(y > hi)),
        "cov80_no_skew": float(np.mean((y >= n10) & (y <= n90))),
        "sd_ratio": ratio,
    }


def table(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The coverage table: one row per position, then the pool."""
    rows = [_row(p, g.filter(pl.col("position") == p)) for p in POSITIONS
            if g.filter(pl.col("position") == p).height]
    rows.append(_row("all", g))
    return rows


def floor_split(g: pl.DataFrame) -> list[dict[str, Any]]:
    """The same coverage, split on whether the model's own p10 survived the zero clip.

    `skewed()` clips at zero, so for a low-`mu` player the 10th percentile lands at or below
    zero and *nothing can fall below it*. Those weeks report a coverage the interval did not
    earn. This is the document's diagnosis and it survives the centre change; what does not
    survive is the conclusion drawn from it.
    """
    return [_row(name, g.filter(sel)) for name, sel in
            (("clipped at zero", pl.col("p10_raw") <= 0.0),
             ("strictly positive", pl.col("p10_raw") > 0.0))
            if g.filter(sel).height]


def verdict(row: dict[str, Any], nominal: float = 0.80, band: float = BAND) -> str:
    """COVERS, UNDER-COVERS or OVER-COVERS, against the pre-registered band.

    Three answers rather than a boolean because the two failures have opposite fixes: an
    interval that is too narrow makes a lineup rule overconfident, and one that is too wide
    makes it refuse to distinguish players it could.
    """
    gap = row["cov80"] - nominal
    if abs(gap) <= band:
        return "COVERS"
    return "UNDER-COVERS" if gap < 0 else "OVER-COVERS"


def measure(stats: pl.DataFrame, centre: Centre = "prior", *,
            min_weeks: int = MIN_WEEKS, min_prior: int = MIN_PRIOR,
            min_mu: float = MIN_MU, band: float = BAND) -> dict[str, Any]:
    """The whole weekly-interval measurement, as one dict."""
    g = graded(centred(player_weeks(stats), centre, min_weeks=min_weeks,
                       min_prior=min_prior, min_mu=min_mu))
    rows = table(g)
    split = floor_split(g)
    gated = next((r for r in split if r["group"] == "strictly positive"), rows[-1])
    return {
        "centre": centre, "lookahead": centre == "realised",
        "n": int(g.height), "nominal": {"cov80": 0.80, "cov68": 0.68},
        "band": band, "by_position": rows, "floor_split": split,
        "gate_subset": GATE_SUBSET, "gate_n": gated["n"], "gate_cov80": gated["cov80"],
        "verdict": verdict(gated, band=band),
    }


# --- the survivor price ---------------------------------------------------


def survivor_price(schedules: pl.DataFrame, edges: Sequence[float] = SPREAD_EDGES,
                   *, sd: float | None = None) -> dict[str, Any]:
    """Survivor's win probability, graded by spread bucket.

    Both sides of every game go in, home-relative spread negated for the away row, which is
    exactly the grid `survivor.grid_from_schedule` builds -- a diagram over home rows alone
    would grade one half of the pick space and survivor picks from both.

    The price and the tie convention are read from the modules that own them
    (`hub.models.margin`), not restated: a survivor pick and a weekly prediction are not
    allowed to disagree about a game they both price, and a grader holding a third copy of
    the formula is how they would.
    """
    from hub.models.margin import home_win_prob, home_won
    from hub.models.market import MARGIN_SD

    need = {"spread_line", "result"}
    missing = sorted(need - set(schedules.columns))
    if missing:
        raise ValueError(f"schedules is missing {missing}")
    scored = home_won(schedules.drop_nulls("spread_line").select(
        pl.col("spread_line").cast(pl.Float64), pl.col("result").cast(pl.Float64)))
    if scored.is_empty():
        raise NotEnoughWeeks("no completed game carries both a spread and a result")
    s = scored["spread_line"].to_numpy()
    won = scored["home_won"].to_numpy()
    margin_sd = MARGIN_SD if sd is None else sd
    p = home_win_prob(s, margin_sd)
    sides = pl.DataFrame({"spread": np.concatenate([s, -s]),
                          "win_prob": np.concatenate([p, 1.0 - p]),
                          "won": np.concatenate([won, 1 - won])})
    fav = sides.filter(pl.col("spread") >= SURVIVOR_SPREAD)
    n = fav.height
    pred = float(cast(float, fav["win_prob"].mean())) if n else float("nan")
    act = float(cast(float, fav["won"].mean())) if n else float("nan")
    se = math.sqrt(act * (1.0 - act) / n) if n else float("nan")
    return {
        "margin_sd": margin_sd, "n_games": scored.height, "n_sides": sides.height,
        "buckets": reliability_by(sides, list(edges), on="spread", prob="win_prob",
                                  outcome="won"),
        "favourite_spread": SURVIVOR_SPREAD, "favourite_n": n,
        "favourite_predicted": pred, "favourite_actual": act,
        "favourite_gap": act - pred, "favourite_se": se,
        "favourite_sigma": ((act - pred) / se) if se else float("nan"),
        "verdict": ("HOLDS" if not se or abs(act - pred) <= 2.0 * se
                    else "UNDER-CONFIDENT" if act > pred else "OVER-CONFIDENT"),
    }


# --- what reads the answer ------------------------------------------------


def write_summary(result: dict[str, Any], path: Path | None = None) -> Path:
    """Leave the measurement where a publisher can read it.

    ADR-0007's second consequence: the result is written rather than printed, so a later
    disagreement is between two recorded runs instead of between two memories.
    """
    p = path or ARTIFACT
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(jsonio.dumps({"name": "interval_coverage",
                               "generated_at": jsonio.stamp(), **result}, indent=2))
    return p


def published_summary(path: Path | None = None) -> dict[str, Any] | None:
    """The last measurement, small enough to publish, or None if none has been made.

    None rather than a raise, and last-good rather than a fresh run: the publisher runs
    unattended on a Sunday, and a page that cannot render because a research measurement is
    missing is the failure mode `CLAUDE.md` names. An unreadable file is the same nothing as
    an absent one -- what the page needs is a field or no field, not a traceback.
    """
    p = path or ARTIFACT
    try:
        got = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or "verdict" not in got:
        return None
    return {k: got.get(k) for k in
            ("centre", "lookahead", "n", "gate_subset", "gate_n", "gate_cov80", "band",
             "verdict", "generated_at")}


def _print_table(rows: list[dict[str, Any]]) -> None:
    print(f"    {'':<18}{'n':>7}{'80%':>8}{'68%':>8}{'<p10':>8}{'>p90':>8}"
          f"{'80% no skew':>13}{'sd r/m':>8}")
    for r in rows:
        print(f"    {r['group']:<18}{r['n']:>7,}{r['cov80']:>8.1%}{r['cov68']:>8.1%}"
              f"{r['below_p10']:>8.1%}{r['above_p90']:>8.1%}"
              f"{r['cov80_no_skew']:>13.1%}{r['sd_ratio']:>8.2f}")


def _stats(seasons: Sequence[int], cache: Path | None) -> pl.DataFrame:
    from hub.fetch import nflverse
    return nflverse.load("player_stats", seasons=list(seasons),
                         cols=list(nflverse.PLAYER_STATS_COLS), cache=cache)


def _schedules(seasons: Sequence[int], cache: Path | None) -> pl.DataFrame:
    from hub.fetch import nflverse
    return nflverse.load("schedules", seasons=list(seasons), cache=cache)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.coverage",
        description="Grade the shipped weekly interval, and the survivor price by spread.")
    ap.add_argument("--measure", action="store_true", help="the weekly interval table")
    ap.add_argument("--survivor", action="store_true",
                    help="survivor win probability by spread bucket")
    ap.add_argument("--gate", action="store_true",
                    help="measure, then exit 1 if the interval leaves the band")
    ap.add_argument("--centre", default="prior", choices=("prior", "realised"),
                    help="'prior' uses only earlier weeks; 'realised' is the document's "
                         "lookahead centre and is kept to reproduce it")
    ap.add_argument("--seasons", default=",".join(str(s) for s in SEASONS))
    ap.add_argument("--min-prior", type=int, default=MIN_PRIOR)
    ap.add_argument("--band", type=float, default=BAND)
    ap.add_argument("--write", action="store_true",
                    help=f"write the result to {ARTIFACT.name} for the publisher")
    ap.add_argument("--cache", default=None, help="raw-cache root; defaults to this repo's")
    a = ap.parse_args(argv)

    if not (a.measure or a.survivor or a.gate):
        ap.print_help()
        return 0
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    cache = Path(a.cache) if a.cache else None

    if a.survivor:
        try:
            sched = _schedules(seasons, cache)
        except Exception as e:
            # Broad on `hub.cli.unavailable`'s reasoning: an empty cache with no network is
            # what a fresh clone hands this command, and the answer to it is a sentence and
            # a non-zero exit rather than nflreadpy's traceback.
            return unavailable("hub.models.coverage", "nflverse schedules", e)
        try:
            got = survivor_price(sched)
        except (ValueError, NotEnoughWeeks) as e:
            print(f"hub.models.coverage: {e}", file=sys.stderr)
            return 1
        print(f"  survivor price, {got['n_games']:,} games over seasons "
              f"{seasons[0]}-{seasons[-1]}, margin sd {got['margin_sd']}")
        print(f"    {'spread':<14}{'n':>7}{'predicted':>12}{'actual':>10}{'gap':>10}")
        for b in got["buckets"]:
            if not b["n"]:
                continue
            print(f"    {b['bin']:<14}{b['n']:>7,}{b['predicted']:>12.3f}"
                  f"{b['actual']:>10.3f}{b['gap']:>+10.3f}")
        print(f"    favourites of {got['favourite_spread']:.0f}+ : predicted "
              f"{got['favourite_predicted']:.3f}, actual {got['favourite_actual']:.3f}, "
              f"gap {got['favourite_gap']:+.3f} at {got['favourite_sigma']:+.1f} se "
              f"over {got['favourite_n']:,} sides -> {got['verdict']}")

    if a.measure or a.gate:
        try:
            stats = _stats(seasons, cache)
        except Exception as e:
            return unavailable("hub.models.coverage", "nflverse player_stats", e)
        try:
            got = measure(stats, a.centre, min_prior=a.min_prior, band=a.band)
        except (ValueError, NotEnoughWeeks) as e:
            print(f"hub.models.coverage: {e}", file=sys.stderr)
            return 1
        lookahead = "  LOOKAHEAD CENTRE" if got["lookahead"] else ""
        print(f"  weekly interval, centre={got['centre']}, {got['n']:,} player-weeks"
              f"{lookahead}")
        _print_table(got["by_position"])
        print("    -- split on whether the model's own p10 survived the zero clip --")
        _print_table(got["floor_split"])
        print(f"    gate reads the {got['gate_subset']} weeks: {got['gate_cov80']:.1%} "
              f"against 80.0% +/- {got['band']:.0%} over {got['gate_n']:,} "
              f"-> {got['verdict']}")
        if a.write:
            print(f"    written to {write_summary(got)}")
        if a.gate and got["verdict"] != "COVERS":
            print("    the deployed interval does not cover; sd = K*sqrt(mu) carries no "
                  "term for the error in the centre.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
