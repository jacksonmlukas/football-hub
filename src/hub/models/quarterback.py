"""The quarterback adjustment: the source's own, in spread points, only where no live price
exists (#218; the construction restated under #268).

The game layer had no quarterback awareness of any kind. Ratings pass the betting market's
number through, and where no poll has returned that number for a week, a starting
quarterback ruled out reaches survivor and the weekly prediction only through whatever the
betting market has already done to it -- which, for most of the survivor horizon, is
nothing the poller can see.

Since #281 the row arrives already labelled. `hub.schedule.priced_games` applies
`live_price` below to write `price_source` as `live`, `stale` or `schedule`, and `apply`
reads that label: the one cut, declared here and applied there, so the coalesce that
chooses the number and the layer that may move it agree row by row. Since #297 the cut
reads the age of the capture that priced the row, not how long the quote has stood.

**What licenses it, and how far.** `docs/qb-adjustment.md`: 538 shipped both a base and a
quarterback-adjusted win probability for every game, and the Brier difference between the
two columns over 2013-2022 is +0.0057 with a 95% t interval excluding zero, positive in 9 of
10 seasons. That is the value of the adjustment out of sample, built by nobody here. It
licenses using one **only where the staleness field marks no live price**: the best published
quarterback-adjusted Elo is +0.01 MAE against the close after fifteen years and blends
65% betting market, so where a live price exists this module changes nothing and a test holds
the row byte for byte.

**The construction is the source's, and the design point is to add nothing to it.** The
adjustment is this quarterback *minus what the team rating already embeds*, not how good
this quarterback is -- near zero for an established starter, sharply negative for a backup
-- and the source publishes exactly that quantity on every row as `qbN_adj`: relative
already, in Elo, and already decayed by nfelo's own rolling update. The team layer reads it
off the latest row and converts it to spread points.

    points = qb_adj / ELO_PER_POINT

Until #268 this module rebuilt the gap itself, subtracting an arrival-time baseline from the
current value and decaying the difference by tenure. The difference carried the starter's
own value drift since he arrived, which has nothing to do with the gap being priced, and it
was right only when the arrival row *was* the latest row -- the one condition every fixture
set. Measured on the 32 live teams, 2026-09-12: mean absolute error 0.4 spread points, worst
3.6, four sign flips; Baltimore, an established starter with no quarterback change, at +4.018
where the source says +0.404. `qb_adj / 25` over the cached file reproduces the published 538
and nfelo magnitudes (n = 5,554: median +0.08, p1 -5.37, p95 +1.19).

The quarterback layer itself is not built here. `hub.fetch.nfeloqb` consumes
`greerreNFL/nfeloqb`'s published ratings -- 538's method on nflfastR data -- and hands over
a per-team state: the starter, his value and adjustment on the latest row, and his tenure.
This module is the team layer only: it turns that state into spread points and decides which
rows may receive them. A game's spread moves by the home side's points minus the away side's,
and only when both sides are known.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from hub.declare import chosen, not_an_input

# STATED CHOICE, not a fitted constant, and declared `chosen` because it is an input to a
# published prediction and the digest is owed coverage of anything that is. Its provenance:
# 538 converted Elo to spread points at 25 per point. The recipe records that the other
# published conversion, Elway's 21.5 Elo per point, disagrees with 538's by 16%, "which is a
# fair statement of the precision available". No interval; this repo has not fitted it and
# does not claim to have. The source's 3.3 Elo per value unit is inside `qb_adj` already and
# is not restated here.
ELO_PER_POINT = chosen(25)

# A snapshot whose capture is older than this is not a live price: the poll age past which
# a snapshot is `stale`. STATED CHOICE from #210's measurement on 2026-09-11: the current
# week's median run was 1.8 days unmoved and every week from 2 out had stood the full 12.2
# days of the archive, so a week separates the two populations with margin on both sides.
# Until #297 the clock ran from the quote's last move; it now runs from the last capture
# (#251: a liquid line nobody has had reason to move in a week is still a live price when
# the poller returned it this morning, and on the Actions runner every capture is minutes
# old). The number is unchanged; what it is measured against is not. An unpriced game and
# one priced from the moving field, which nothing polls, are not live prices either.
STALE_AFTER_DAYS = chosen(7)

# The two columns `apply` writes on a row it moved, and leaves null on one it did not.
# `adjusted_by` names the source whose state moved it; `qb_adjustment` is the points added
# to the home spread. A row a live price protects carries null in both, which is how a
# reader tells "not adjusted" from "adjusted by nothing".
ADJUSTMENT_COLUMNS = not_an_input(
    ("qb_adjustment", "adjusted_by"),
    "the names of the columns this module writes; renaming a column moves no rating, "
    "and a digest that moved on a rename would be a version claiming a difference "
    "that does not exist")
SOURCE = not_an_input(
    "nfeloqb",
    "the provenance label written into `adjusted_by`; not a number, and renaming the "
    "source moves no rating -- ADR-0006's objection from the other side")


def points(state: pl.DataFrame) -> pl.DataFrame:
    """Per team, the spread points its current starter is worth relative to what the team
    rating embeds: the source's adjustment on the latest row over `ELO_PER_POINT`, and
    nothing else. `team` and `points`."""
    return state.select(pl.col("team"), (pl.col("qb_adj") / ELO_PER_POINT).alias("points"))


def live_price(at: datetime) -> pl.Expr:
    """Whether a snapshot is a live price at `at`: the capture that priced the row -- the
    last poll at or before `at`, on the row as `priced_at` -- is within `STALE_AFTER_DAYS`.

    The one declaration of the cut, and it is applied in exactly one place --
    `hub.schedule.priced_games`, which labels every row `live`, `stale` or `schedule` and
    lets a stale snapshot lose to the moving field (#281). This module then reads the label
    rather than the clock, so the coalesce and the adjustment cannot disagree about a row.

    **The clock is the poll's, not the quote's (#297, decided in #251).** Until #297 this
    read `unmoved_since`, the start of the run of polls returning the same number. That
    measured the wrong thing for this consumer: a liquid line nobody has had reason to move
    in ten days is still a live price when the poller returned it this morning, and on the
    Actions runner every capture is minutes old, so the cut could never fire there and fired
    on exactly those lines on the laptop. It also made the label depend on what
    `hub.fetch.odds._quote_moved` counts as a move -- a book joining an otherwise frozen
    quote restarted the run and switched the adjustment off. The staleness columns still
    ride on the row for a reader; nothing here reads them.

    A row with no capture is not a snapshot -- every snapshot is dated -- and cannot be
    shown to be a live price, so it is not one. `priced_games` never asks about such a
    row: it applies this only where a snapshot priced the candidate.
    """
    captured = pl.col("priced_at")
    return (captured >= pl.lit(at - timedelta(days=STALE_AFTER_DAYS))).fill_null(False)


def no_live_price(games: pl.DataFrame) -> pl.Expr:
    """The rows `price_source` marks as having no live price, and that carry a number to
    adjust: a stale snapshot, or the moving field, which nothing polls. The label is
    `hub.schedule.priced_games`'s, decided by `live_price` above; nothing here re-derives
    it, which is what keeps the two consumers of the cut in agreement (#281).
    """
    priced = pl.col("close_spread").is_not_null()
    return priced & pl.col("price_source").is_in(["stale", "schedule"])


def apply(games: pl.DataFrame, state: pl.DataFrame | None) -> pl.DataFrame:
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
    touched = (no_live_price(games) & pl.col("_home_points").is_not_null()
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
