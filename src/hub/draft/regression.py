"""Touchdown luck: last season's actuals held against the current board.

Three separately measured facts make this worth computing:

1. **Touchdown rate per yard does not persist.** Year over year it is +0.004 receiving and
   -0.030 rushing -- indistinguishable from zero (`docs/component-projection.md`). A
   player's touchdown rate says nothing about his next one.
2. **Fully regressing it beats carrying his points forward**, and the fitted optimal shrink
   is 1.0 rather than a partial one (`docs/volume-model.md`).
3. **The draft market does not fully regress it.** Weighting prior touchdown points against
   prior yardage points, the room prices touchdowns at 1.02 relative to volume for
   quarterbacks where their true predictive weight is -0.05. See `docs/td-luck.md` for the
   per-position table and, more importantly, for how much of this is established and how
   much is directional.

So a player whose prior season carried more touchdowns than his yardage supports is being
priced on something that will not repeat, and one whose touchdowns lagged his yardage is
being marked down for the same reason in reverse.

This is not the board's existing `fp_over_expected`. That measures realised points against
expected points from opportunity; this measures realised touchdowns against the yardage that
produced them. On the live board they correlate at +0.16 -- and a genuine overlap would show
as a strong *negative*, since the two are signed in opposite directions.
"""
from __future__ import annotations

import polars as pl

from hub.draft.state import _norm
from hub.models.components import td_rate

# Fewer games than this and the number is noise wearing a number's clothes: a two-game
# sample of touchdown luck is one red-zone target either way.
MIN_GAMES = 6

_PHASES = (("receiving_yards", "receiving_tds", "rec", 6.0),
           ("rushing_yards", "rushing_tds", "rush", 6.0),
           ("passing_yards", "passing_tds", "pass", 4.0))


def prior_season(season: int, cache=None) -> pl.DataFrame:
    """Per-player season totals for the phases touchdown luck needs.

    Regular season only: a playoff run inflates totals for a handful of players and none of
    it is in the fantasy season anyway.
    """
    from hub.fetch import nflverse
    cols = ("player_id", "player_display_name", "position", "season", "week",
            "season_type", "fantasy_points_ppr", "receiving_yards", "receiving_tds",
            "rushing_yards", "rushing_tds", "passing_yards", "passing_tds")
    w = nflverse.load("player_stats", seasons=[season], cols=cols, cache=cache).filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(["QB", "RB", "WR", "TE"]))
    return (w.group_by(["player_display_name", "position"])
             .agg(pl.len().alias("g"),
                  *[pl.col(c).sum().alias(c) for c in
                    ("receiving_yards", "receiving_tds", "rushing_yards", "rushing_tds",
                     "passing_yards", "passing_tds")])
             .rename({"player_display_name": "player", "position": "pos"}))


def td_luck(season: pl.DataFrame) -> pl.DataFrame:
    """Fantasy points per game a player scored above the touchdowns his yardage supports.

    `season` needs player, pos, g and the per-phase yardage and touchdown totals. Positive
    means he outscored his own yardage -- the part least likely to repeat.
    """
    df = season.filter(pl.col("g") >= MIN_GAMES)
    if df.is_empty():
        return df.select("player", "pos").with_columns(
            pl.lit(None, dtype=pl.Float64).alias("td_luck"))

    # Scored in fantasy points, not touchdown counts: a passing touchdown is worth four and
    # a rushing one six, so counting touchdowns understates quarterbacks by a third -- and
    # quarterbacks are where the market misprices most.
    actual = sum((pts * pl.col(td) for _, td, _, pts in _PHASES), start=pl.lit(0.0))
    expected = sum(
        (pts * pl.col(yd) * pl.col("pos").replace_strict(
            {p: td_rate(p, phase) for p in ("QB", "RB", "WR", "TE")},
            default=0.0, return_dtype=pl.Float64)
         for yd, _, phase, pts in _PHASES), start=pl.lit(0.0))

    return df.with_columns(((actual - expected) / pl.col("g")).alias("td_luck"))


def attach(board: pl.DataFrame, season: pl.DataFrame) -> pl.DataFrame:
    """Add a `td_luck` column to a draft board.

    Joined on a normalised name because nflverse and ESPN disagree about punctuation --
    `A.J. Brown` against `AJ Brown` -- and an exact join drops him silently. Players with no
    prior season keep a null rather than being dropped: rookies are half an early board.
    """
    luck = td_luck(season)
    if "td_luck" not in luck.columns or luck.is_empty():
        return board.with_columns(pl.lit(None, dtype=pl.Float64).alias("td_luck"))

    keyed = (luck.with_columns(
                pl.col("player").map_elements(_norm, return_dtype=pl.Utf8).alias("_k"))
             .select("_k", "td_luck").unique(subset=["_k"], keep="first"))
    return (board.drop("td_luck", strict=False)
                 .with_columns(
                     pl.col("player").map_elements(_norm, return_dtype=pl.Utf8).alias("_k"))
                 .join(keyed, on="_k", how="left").drop("_k"))
