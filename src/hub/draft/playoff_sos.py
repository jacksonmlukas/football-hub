"""Weeks 15-17 strength of schedule.

The fantasy playoffs are three known games against known defences, and nobody drafting
off ESPN's app prices them. A season-long projection says what a player does on average
against an average defence; it says nothing about whether his week 16 is against the
softest secondary in the league or the hardest.

Two ingredients, both bulk pulls:

  * defence vs position -- PPR points each defence allowed to each position per game,
    from last season. A defence soft against WRs is not necessarily soft against RBs, so
    this is never collapsed to a single "good defence" number.
  * the weeks 15-17 schedule for the season ahead, which is already published.

The output is a ratio: 1.10 means this player's playoff opponents allowed 10% more to his
position than a league-average defence did. It is a tiebreaker, not a ranking -- it moves
players inside a tier, and last year's defence is a noisy guide to this year's.
"""
from __future__ import annotations

import polars as pl

from hub.config import DRAFTED_POSITIONS, SEASON_AHEAD, SEASON_COMPLETED

PLAYOFF_WEEKS = (15, 16, 17)

# FantasyPros and nflverse disagree on three codes. FA is genuinely teamless.
TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA", "LV": "LV", "WSH": "WAS", "ARZ": "ARI"}


def _canon_team(team: str | None) -> str | None:
    if not team or team in ("FA", ""):
        return None
    return TEAM_ALIASES.get(team, team)


def _dvp_from_stats(stats: pl.DataFrame) -> pl.DataFrame:
    """PPR points allowed per game, by defence and position, indexed to league average.

    Summed per (defence, week) first: a defence that faces a committee backfield allowed
    those points whether they came from one back or three, and averaging player-level
    rows instead would reward defences that face deeper rotations.
    """
    # Sorted before the second aggregation, and that is not tidiness. A `group_by` emits its
    # rows in a hash-dependent order that varies between calls, the mean below sums them, and
    # floating-point addition is not associative -- so without this the same input gives
    # answers differing at 7.1e-15, the board sorted on that float lands in a different ROW
    # ORDER, and the draft indexes the board by row. Two identical `board_as_of` calls
    # returned different boards, and every measurement drafting from them wobbled by ~0.04
    # points a team-week. improvements.md #18.
    per_week = (stats
                .filter(pl.col("position").is_in(DRAFTED_POSITIONS))
                .group_by(["opponent_team", "position", "week"])
                .agg(pl.col("fantasy_points_ppr").sum().alias("allowed"))
                .sort(["opponent_team", "position", "week"]))
    dvp = (per_week.group_by(["opponent_team", "position"])
                   .agg(pl.col("allowed").mean().alias("ppg_allowed"))
                   .rename({"opponent_team": "defense", "position": "pos"}))
    # `.mean().over("pos")` is a third aggregation over the second's output, so it needs the
    # same treatment; and the final sort carries `defense` as a tiebreaker, because two
    # defences with an identical ratio would otherwise order arbitrarily.
    return (dvp.sort(["defense", "pos"])
               .with_columns(
                   (pl.col("ppg_allowed") / pl.col("ppg_allowed").mean().over("pos"))
                   .alias("dvp_ratio"))
               .sort(["pos", "dvp_ratio", "defense"], descending=[False, True, False]))


def _opponents_from_schedule(sched: pl.DataFrame,
                             weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> pl.DataFrame:
    """One row per (team, week, opponent) -- both sides of every game."""
    g = sched.filter(pl.col("week").is_in(weeks))
    home = g.select(pl.col("home_team").alias("team"), "week",
                    pl.col("away_team").alias("opponent"))
    away = g.select(pl.col("away_team").alias("team"), "week",
                    pl.col("home_team").alias("opponent"))
    return pl.concat([home, away]).sort(["team", "week"])


def _sos_from(dvp: pl.DataFrame, opponents: pl.DataFrame) -> pl.DataFrame:
    """Mean opponent dvp_ratio over the playoff weeks, per team and position."""
    joined = opponents.join(dvp, left_on="opponent", right_on="defense", how="inner")
    return (joined.group_by(["team", "pos"])
                  .agg(pl.col("dvp_ratio").mean().alias("wk15_17_sos"),
                       pl.len().alias("sos_games"))
                  .sort(["pos", "wk15_17_sos"], descending=[False, True]))


def attach_sos(board: pl.DataFrame, sos: pl.DataFrame) -> pl.DataFrame:
    """Add wk15_17_sos to the board. Teamless or unmatched players get null, never a default.

    A silent 1.0 would read as "average playoff schedule" for a player we simply could
    not place, which is the kind of quiet wrong answer this repo keeps finding.
    """
    keyed = board.with_columns(
        pl.col("team").map_elements(_canon_team, return_dtype=pl.Utf8).alias("_team"))
    return (keyed.join(sos, left_on=["_team", "pos"], right_on=["team", "pos"], how="left")
                 .drop("_team"))


def playoff_sos(season_ahead: int = SEASON_AHEAD, dvp_season: int = SEASON_COMPLETED,
                weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> pl.DataFrame:
    """Fetch both inputs and build the table. Two bulk pulls, no per-team loop."""
    import nflreadpy as nfl

    stats = nfl.load_player_stats(seasons=[dvp_season])
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    sched = nfl.load_schedules().filter(pl.col("season") == season_ahead)
    return _sos_from(_dvp_from_stats(stats), _opponents_from_schedule(sched, weeks))
