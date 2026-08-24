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

PLAYOFF_WEEKS = (15, 16, 17)
SCORING_POSITIONS = ("QB", "RB", "WR", "TE")

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
    per_week = (stats
                .filter(pl.col("position").is_in(SCORING_POSITIONS))
                .group_by(["opponent_team", "position", "week"])
                .agg(pl.col("fantasy_points_ppr").sum().alias("allowed")))
    dvp = (per_week.group_by(["opponent_team", "position"])
                   .agg(pl.col("allowed").mean().alias("ppg_allowed"))
                   .rename({"opponent_team": "defense", "position": "pos"}))
    return (dvp.with_columns(
                (pl.col("ppg_allowed") / pl.col("ppg_allowed").mean().over("pos"))
                .alias("dvp_ratio"))
               .sort(["pos", "dvp_ratio"], descending=[False, True]))


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


def playoff_sos(season_ahead: int = 2026, dvp_season: int = 2025,
                weeks: tuple[int, ...] = PLAYOFF_WEEKS) -> pl.DataFrame:
    """Fetch both inputs and build the table. Two bulk pulls, no per-team loop."""
    import nflreadpy as nfl

    stats = nfl.load_player_stats(seasons=[dvp_season])
    if "season_type" in stats.columns:
        stats = stats.filter(pl.col("season_type") == "REG")
    sched = nfl.load_schedules().filter(pl.col("season") == season_ahead)
    return _sos_from(_dvp_from_stats(stats), _opponents_from_schedule(sched, weeks))
