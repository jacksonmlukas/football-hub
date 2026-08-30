"""Availability as a per-player trait, and today's injury news as a separate thing.

`TALENT_CV` already carried availability, but only as a population average -- it was fitted
on points per *team* game, so missed time sits inside it at the positional level. Every
running back therefore carried identical durability risk, and a back with a long injury
history looked to the simulation exactly like one who has never missed a snap.

Two measured facts make a per-player version worth having (`docs/durability.md`):

**Availability persists.** Games missed correlates +0.407 year over year, which is stronger
than the folklore that injury proneness is hindsight. A player who missed six or more games
last season misses three or more this season 76% of the time against a 55% base rate; one
who missed none does it 41% of the time.

**The projection does not price it.** Regressing next-season points per team game on the
projection *plus* prior games missed leaves -0.186 per game, P(<0) = 100%. ESPN projects
per-game scoring for a healthy player and discounts expected absence by less than it should.

The per-position cut is the surprising part and the reason this is not applied uniformly.
Running backs come back at **-0.065 (71%, nothing)** -- the market already discounts running
back durability, because everyone knows running backs break. The inefficiency is at
quarterback and receiver, not at the position people worry about.

One season is enough. A two-year history scores R2 0.4597 against 0.4614 for last season
alone, and the second year's coefficient is -0.040 against -0.159, so almost all of the
signal is in the most recent season.
"""
from __future__ import annotations

import polars as pl

from hub.config import drafted_positions
from hub.draft import prior_signal

TEAM_GAMES = 17

# Points per team game lost per prior-season game missed, beyond what the projection already
# prices. From `ppg_next ~ proj_ppg + missed` on historical ESPN projections.
#
#   QB  -0.457  [-0.645, -0.257]  100.0%   applied
#   WR  -0.151  [-0.266, -0.045]   99.6%   applied
#   TE  -0.097  [-0.242, +0.062]   89.0%   not applied
#   RB  -0.065  [-0.293, +0.167]   71.2%   not applied -- the market already prices it
BETA: dict[str, float] = {"QB": -0.457, "WR": -0.151}

# Below this a player was not a starter, and his missed games are being a backup rather than
# being hurt. Without it the signal is mostly roster churn.
MIN_PPG = 5.0

# Designations worth putting in front of a drafter. ACTIVE is not news.
FLAG_STATUS = frozenset({"OUT", "DOUBTFUL", "QUESTIONABLE", "INJURY_RESERVE"})

# Points per team game lost by carrying a designation *now*, beyond what the projection
# prices. Fitted against week-1 injury reports, which are the closest historical analogue to
# an August designation: `total_next/17 ~ proj_ppg + designation`, 1,263 player-seasons.
#
#   Out / Doubtful  -1.631  [-2.554, -0.736]  100.0%   applied
#   Questionable    -0.949  [-2.495, +0.459]   90.2%   NOT applied
#
# Questionable is left out for two reasons, and the second is the stronger one. It does not
# clear significance at n=36. And an August QUESTIONABLE is a different population from the
# week-1 one this would be fitted on: 12.6% of the August board carries it against 2.9% at
# week 1, **4.4x more common**. Pricing an eighth of the board on a coefficient estimated
# from a much sicker group would be worse than leaving it for judgment.
#
# Out, Doubtful and IR transfer far better -- 4.1% of the August board against 2.1% at week
# 1. IR has no coefficient of its own, since nobody on IR appears on a practice report;
# starting a season there means missing at least four games by rule, so it is at least as
# severe as Out and borrowing that number understates it.
INJURY_BETA: dict[str, float] = {
    "OUT": -1.631, "DOUBTFUL": -1.631, "INJURY_RESERVE": -1.631,
}


def is_flagworthy(status: str | None) -> bool:
    """Whether a current injury designation is worth surfacing."""
    return bool(status) and str(status).upper() in FLAG_STATUS


def games_missed(season: pl.DataFrame) -> pl.DataFrame:
    """Games a player missed last season, for players who had a real role.

    `season` needs player, pos, g and ppg.
    """
    return (season.filter(pl.col("ppg") > MIN_PPG)
                  .with_columns((pl.lit(TEAM_GAMES) - pl.col("g")).clip(0).alias("missed")))


def prior_season(season: int, cache=None) -> pl.DataFrame:
    """Games played and scoring rate per player, last season."""
    from hub.fetch import nflverse
    cols = ("player_id", "player_display_name", "position", "season", "week",
            "season_type", "fantasy_points_ppr")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache).filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(list(drafted_positions())))
    return (w.group_by(["player_display_name", "position"])
             .agg(pl.len().alias("g"),
                  pl.col("fantasy_points_ppr").mean().alias("ppg"))
             .rename({"player_display_name": "player", "position": "pos"}))


def attach(board: pl.DataFrame, season: pl.DataFrame) -> pl.DataFrame:
    """Add a `missed` column. Players with no prior role keep a null rather than a zero.

    The join itself is `prior_signal.join_by_player`, which `hub.draft.regression` also uses:
    it was the same twelve lines in both files -- improvements.md #15.
    """
    return prior_signal.join_by_player(board, games_missed(season), "missed")


def correct_projection(board: pl.DataFrame, column: str = "proj_blend") -> pl.DataFrame:
    """Mark a projection down for a player's own availability history.

    Applied where the market leaves a residual and nowhere else. `hub.draft.optimize` scores
    seasons against this column, so this is where a durability signal has to land to reach a
    pick -- one that is only printed does not change a decision.

    Today's designation is priced separately and only where it transfers -- Out, Doubtful
    and IR. QUESTIONABLE is carried for judgment and not priced; see `INJURY_BETA`.
    """
    if column not in board.columns:
        return board
    adjustment = pl.lit(0.0)
    if "missed" in board.columns:
        adjustment = adjustment + prior_signal.priced("missed", BETA)
    if "injury_status" in board.columns:
        # Applied at every position, unlike the durability trait: being ruled out is news,
        # not a trait the market has had years to discount.
        adjustment = adjustment + pl.col("injury_status").fill_null("").str.to_uppercase(
        ).replace_strict(INJURY_BETA, default=0.0, return_dtype=pl.Float64)
    return board.with_columns((pl.col(column) + adjustment).clip(0.0).alias(column))
