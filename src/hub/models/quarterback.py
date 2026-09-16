"""The quarterback adjustment: the source's own, in spread points, on rows labelled as
having no live price (#218; the construction restated under #268). **Pulled from the
published path under #299**; read by `hub.models.starter_change`, the harness that gates
it, and by nothing the product ships.

**Why it left (#299, decided 2026-09-16).** Two facts landed in one week. The nfeloqb file
names a new starter only *after* his first game (#291), and after #297 the adjustment
fires only on a stale poll -- so when it fires it adds the current `qb_adj` to a line that
already priced that same quarterback. The one case it was built for, a starter change the
frozen line predates, cannot reach it from this source. Marked-and-shipping (#270) was
harmless on the Actions runner, where it never fires, and wrong in expectation on the
laptop. `hub.models.ratings` and `hub.season.survivor` no longer call `apply`; no written
prediction or survivor pick carries `adjusted_by` or `qb_adjustment`, and no row is filed
under `market_baseline-qb`. `tests/contracts/test_the_quarterback_adjustment_is_not_a_dependency.py`
holds that the harness is the only reader. The way back is a starter source that is timely
before kickoff (#221's depth-chart construction) and #291's gate clearing;
`docs/qb-adjustment.md` records both.

**What it was, kept re-runnable.** The game layer had no quarterback awareness of any kind.
Ratings pass the betting market's number through, and where no poll has returned that
number for a week, a starting quarterback ruled out reaches survivor and the weekly
prediction only through whatever the betting market has already done to it -- which, for
most of the survivor horizon, is nothing the poller can see.

Since #281 the row arrives already labelled. `hub.schedule.priced_games` applies
`hub.schedule.live_price` to write `price_source` as `live`, `stale` or `schedule`, and
`apply` reads that label: the one cut, declared and applied there, so the coalesce that
chooses the number and the layer that may move it agree row by row. Until #299 the cut and
`STALE_AFTER_DAYS` were declared here; they moved to `hub.schedule` with the pull, so the
product's schedule does not import this module to label a row.

**What licenses it, and how far.** `docs/qb-adjustment.md`: 538 shipped both a base and a
quarterback-adjusted win probability for every game, and the Brier difference between the
two columns over 2013-2022 is +0.0057 with a 95% t interval excluding zero, positive in 9 of
10 seasons. That is the value of the adjustment out of sample, built by nobody here. It
licenses using one **only where the staleness field marks no live price**: the best published
quarterback-adjusted Elo is +0.01 MAE against the close after fifteen years and blends
65% betting market, so where a live price exists this module changes nothing and a test holds
the row byte for byte. What it does not license is this estimator against a frozen line,
which is what #291's gate measures and #299 pulled on.

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

import polars as pl

from hub.declare import not_an_input

# 538's conversion of Elo to spread points, 25 per point. The recipe records that the other
# published conversion, Elway's 21.5 Elo per point, disagrees with 538's by 16%, "which is a
# fair statement of the precision available". No interval; this repo has not fitted it and
# does not claim to have. The source's 3.3 Elo per value unit is inside `qb_adj` already and
# is not restated here. Declared `chosen` from #218 to #299, while a published prediction
# read it; since #299 no product path reaches this module, so no prediction can, and a
# digest that still hashed it would be claiming a difference between two runs that compute
# the same thing.
ELO_PER_POINT = not_an_input(
    25,
    "the harness's Elo-to-points conversion since #299: read by hub.models.starter_change "
    "and by no published prediction, so it identifies no model version")

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


def no_live_price(games: pl.DataFrame) -> pl.Expr:
    """The rows `price_source` marks as having no live price, and that carry a number to
    adjust: a stale snapshot, or the moving field, which nothing polls. The label is
    `hub.schedule.priced_games`'s, decided by `hub.schedule.live_price`; nothing here
    re-derives it, which is what keeps the consumers of the cut in agreement (#281).
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
