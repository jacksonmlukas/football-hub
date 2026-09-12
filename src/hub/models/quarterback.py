"""The quarterback adjustment: relative, decaying, and only where no live price exists (#218).

The game layer had no quarterback awareness of any kind. Ratings pass the betting market's
number through, and where the staleness field (#210) says that number has stood untouched
for weeks, a starting quarterback ruled out reaches survivor and the weekly prediction only
through whatever the betting market has already done to it -- which, for most of the
survivor horizon, is nothing the poller can see.

**What licenses it, and how far.** `docs/qb-adjustment.md`: 538 shipped both a base and a
quarterback-adjusted win probability for every game, and the Brier difference between the
two columns over 2013-2022 is +0.0057 with a 95% t interval excluding zero, positive in 9 of
10 seasons. That is the value of the adjustment out of sample, built by nobody here. It
licenses using one **only where the staleness field marks no live price**: the best published
quarterback-adjusted Elo is +0.01 MAE against the close after fifteen years and blends
65% betting market, so where a live price exists this module changes nothing and a test holds
the row byte for byte.

**The construction is relative, and that is the design point that is easy to get wrong.**
The adjustment is this quarterback *minus what the team rating already embeds*, not how good
this quarterback is. It sits near zero for an established starter and goes sharply negative
for a backup, and it decays at 10% per game of the new starter's tenure, because after three
or four games the team rating has absorbed most of the downgrade. A model adding a full
starter-versus-backup gap in week six of a backup's tenure is double-counting. The recipe is
the research artifact's `qb_adjustment_recipe` (2026-09-07), and the numbers below are its.

The quarterback layer itself is not built here. `hub.fetch.nfeloqb` consumes
`greerreNFL/nfeloqb`'s published ratings -- 538's method on nflfastR data -- and hands over
a per-team state: the starter, his value, the row he arrived on, and his tenure. This
module is the team layer only: it turns that state into spread points and decides which
rows may receive them.

    points = POINTS_PER_VALUE x (qb_value - embedded) x DECAY_PER_GAME ** tenure
    embedded = arrival_value - arrival_adj / ELO_PER_VALUE

`embedded` is the team's rolling quarterback value on the row the starter arrived, recovered
from the source's own adjustment on that row (3.3 x the gap, in Elo). A game's spread moves by
the home side's points minus the away side's, and only when both sides are known.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

# STATED CHOICE, not a fitted constant, and in `FITTED_MODULES` because it is an input to a
# published prediction and the digest is owed coverage of anything that is. Its provenance:
# 538 converted a quarterback value to Elo at 3.3 per unit and Elo to spread points at 25 per
# point, and 3.3 / 25 = 0.132. The recipe records that the other published conversion, Elway's
# 21.5 Elo per point, disagrees with 538's by 16%, "which is a fair statement of the precision
# available". No interval; this repo has not fitted it and does not claim to have.
ELO_PER_VALUE = 3.3
ELO_PER_POINT = 25
POINTS_PER_VALUE = 0.132

# The share of the relative gap that survives each game the new starter plays. 538's rolling
# quarterback value is 0.9 x previous + 0.1 x this game, so a tenth of the gap is absorbed
# into the team rating per game and the remainder is what is still unpriced. Stated choice,
# same provenance as above.
DECAY_PER_GAME = 0.9

# A snapshot quote that has stood still longer than this is not a live price. STATED CHOICE
# from #210's measurement on 2026-09-11: the current week's median run was 1.8 days unmoved
# and every week from 2 out had stood the full 12.2 days of the archive, so a week separates
# the two populations with margin on both sides. A quote no book has touched across a whole
# slate's worth of news is a posted lookahead, not a price responding to it. An unpriced
# game and one priced from the moving field, which nothing polls, are not live prices either.
STALE_AFTER_DAYS = 7

# The two columns `apply` writes on a row it moved, and leaves null on one it did not.
# `adjusted_by` names the source whose state moved it; `qb_adjustment` is the points added
# to the home spread. A row a live price protects carries null in both, which is how a
# reader tells "not adjusted" from "adjusted by nothing".
ADJUSTMENT_COLUMNS = ("qb_adjustment", "adjusted_by")
SOURCE = "nfeloqb"


def points(state: pl.DataFrame) -> pl.DataFrame:
    """Per team, the spread points its current starter is worth relative to what the team
    rating embeds, decayed by his tenure. `team` and `points`."""
    embedded = pl.col("arrival_value") - pl.col("arrival_adj") / ELO_PER_VALUE
    decayed = (pl.col("qb_value") - embedded) * (pl.lit(DECAY_PER_GAME) ** pl.col("tenure"))
    return state.select(pl.col("team"), (POINTS_PER_VALUE * decayed).alias("points"))


def no_live_price(games: pl.DataFrame, at: datetime) -> pl.Expr:
    """The rows the staleness field marks as having no live price, and that carry a number
    to adjust.

    A snapshot is live while the run of polls returning its quote began within
    `STALE_AFTER_DAYS` of `at`. The moving field is never live: nothing polls it, so nothing
    can show it is. A snapshot row on a frame that carries no staleness columns is not
    *marked* either way and is left alone -- the rule reaches only what the field says.
    """
    priced = pl.col("close_spread").is_not_null()
    moving = pl.col("price_source") == "schedule"
    if "unmoved_since" not in games.columns:
        return priced & moving
    stood = pl.col("unmoved_since") < pl.lit(at - timedelta(days=STALE_AFTER_DAYS))
    frozen = (pl.col("price_source") == "snapshot") & pl.col("unmoved_since").is_not_null() & stood
    return priced & (moving | frozen)


def apply(games: pl.DataFrame, state: pl.DataFrame | None, *, at: datetime) -> pl.DataFrame:
    """The slate with its ratings adjusted where no live price exists, and said so.

    Every column the frame arrived with is returned as it arrived on every row a live price
    protects; the two `ADJUSTMENT_COLUMNS` are appended, null there and filled on the rows
    moved. With no state -- a fresh clone, a pull that never succeeded -- nothing moves and
    the columns are still appended, so a reader downstream meets one schema either way.

    A game moves only when both its teams are in the state. Half an adjustment is a number
    with no meaning, and the row would have to say which half it was.
    """
    empty = [pl.lit(None, dtype=pl.Float64).alias("qb_adjustment"),
             pl.lit(None, dtype=pl.Utf8).alias("adjusted_by")]
    if state is None or not state.height:
        return games.with_columns(empty)
    pts = points(state)
    home = pts.rename({"team": "home_team", "points": "_home_points"})
    away = pts.rename({"team": "away_team", "points": "_away_points"})
    joined = games.join(home, on="home_team", how="left").join(away, on="away_team", how="left")
    touched = (no_live_price(games, at) & pl.col("_home_points").is_not_null()
               & pl.col("_away_points").is_not_null())
    delta = pl.col("_home_points") - pl.col("_away_points")
    return (joined.with_columns(
                pl.when(touched).then(delta).otherwise(None).cast(pl.Float64)
                  .alias("qb_adjustment"),
                pl.when(touched).then(pl.lit(SOURCE)).otherwise(None).cast(pl.Utf8)
                  .alias("adjusted_by"))
                  .with_columns(
                      pl.when(touched).then(pl.col("close_spread") + delta)
                        .otherwise(pl.col("close_spread")).cast(pl.Float64)
                        .alias("close_spread"))
                  .drop("_home_points", "_away_points"))


def report_line(adjusted: pl.DataFrame) -> str:
    """The one line #218's last criterion asks for: how many priced games the adjustment
    touched and the mean absolute change, in spread points and in home win probability.

    Computed off the frame `apply` returned, so a caller with a slate and a season's grid
    can each say it about what they published. The win-probability change is `MarketBaseline`'s
    conversion of the same spread, before and after, so the two numbers describe
    one move.
    """
    from hub.models.market import MARGIN_SD, normal_cdf

    priced = adjusted.filter(pl.col("close_spread").is_not_null())
    moved = priced.filter(pl.col("adjusted_by").is_not_null())
    if not moved.height:
        return (f"quarterback adjustment: 0 of {priced.height} priced games touched; every "
                f"priced game had a live price or no state from {SOURCE}")
    spreads = moved["close_spread"].to_list()
    deltas = moved["qb_adjustment"].to_list()
    d_points = sum(abs(d) for d in deltas) / len(deltas)
    d_prob = sum(abs(normal_cdf(s / MARGIN_SD) - normal_cdf((s - d) / MARGIN_SD))
                 for s, d in zip(spreads, deltas, strict=True)) / len(deltas)
    return (f"quarterback adjustment: {moved.height} of {priced.height} priced games touched "
            f"(no live price); mean |change| {d_points:.2f} points, {d_prob:.3f} home win "
            f"probability; source {SOURCE}")
