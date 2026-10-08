"""The synthetic world `test_starter_events.py`, `test_starter_study.py` and
`test_starter_change.py` share (#346): the source's rows for two seasons of KC, DEN and the Rams,
the play-by-play and schedule they imply, the snapshot archive the readers price off, and the
offline `Replay` the events source is read through. Every fixture is synthetic; nothing here
touches `data/` or the network.
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from hub.fetch import replay
from hub.models import starter_events as se

# --- fixtures ------------------------------------------------------------------------------


def source_rows(games):
    """nfeloqb-shaped rows. Each game: (date, season, week, home, away, home_qb, away_qb,
    home_value, away_value, home_adj, away_adj, home_score | None, away_score | None,
    elo_prob1 | None, qbelo_prob1 | None). Home is team1, the source's convention."""
    cols = ("date", "season", "week", "team1", "team2", "qb1", "qb2", "qb1_value_pre",
            "qb2_value_pre", "qb1_adj", "qb2_adj", "score1", "score2", "elo_prob1",
            "qbelo_prob1")
    frame = pl.DataFrame({c: [g[i] for g in games] for i, c in enumerate(cols)},
                         schema={"date": pl.Utf8, "season": pl.Int64, "week": pl.Utf8,
                                 "team1": pl.Utf8, "team2": pl.Utf8, "qb1": pl.Utf8,
                                 "qb2": pl.Utf8, "qb1_value_pre": pl.Float64,
                                 "qb2_value_pre": pl.Float64, "qb1_adj": pl.Float64,
                                 "qb2_adj": pl.Float64, "score1": pl.Int64,
                                 "score2": pl.Int64, "elo_prob1": pl.Float64,
                                 "qbelo_prob1": pl.Float64})
    return frame.with_columns(pl.lit("REG").alias("game_type"))


def polls(rows):
    """The archive as `hub.fetch.odds._archive` returns it: (game_id, close_spread,
    captured_at, week). `captured_at` is naive UTC."""
    return pl.DataFrame({"game_id": [r[0] for r in rows],
                         "close_spread": [float(r[1]) for r in rows],
                         "captured_at": [r[2] for r in rows],
                         "week": [r[3] for r in rows]},
                        schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                "captured_at": pl.Datetime("us"), "week": pl.Int64})


def utc(day, hour=16):
    return dt.datetime(2026, 9, day, hour)


# A season for one team, KC: Mahomes starts weeks 1-2, a backup takes week 3, Mahomes is
# back for week 4. The opponent DEN keeps one starter throughout; the Rams, spelled the
# source's way, arrive in week 3 with a starter who differs from their last 2025 start --
# an offseason change, flagged and not an event.
SEASON = [
    ("2026-09-13", 2026, "1.0", "KC", "DEN", "Mahomes", "Nix", 200.0, 120.0, 40.0, 2.0,
     27, 20, 0.70, 0.75),
    ("2026-09-20", 2026, "2.0", "DEN", "KC", "Nix", "Mahomes", 121.0, 205.0, 2.5, 42.0,
     17, 24, 0.40, 0.35),
    ("2026-09-27", 2026, "3.0", "KC", "LAR", "Gabbert", "Bethard", 60.0, 55.0, -110.0, -90.0,
     10, 20, 0.65, 0.52),
    ("2026-10-04", 2026, "4.0", "KC", "DEN", "Mahomes", "Nix", 199.0, 122.0, 38.0, 3.0,
     None, None, 0.71, 0.76),
]
PRIOR = [("2025-12-28", 2025, "17.0", "KC", "DEN", "Mahomes", "Nix", 210.0, 100.0, 45.0,
          -5.0, 30, 10, 0.8, 0.85),
         ("2025-12-28", 2025, "17.0", "LAR", "SF", "Stafford", "Purdy", 150.0, 160.0, 10.0,
          12.0, 21, 24, 0.5, 0.49)]


def pbp_for(rows_df, *, drop=()):
    """nflverse play-by-play, `STARTER_PBP_COLS`-shaped, whose first pass of every (game, team)
    the source's rows hold was thrown by the source's own starter -- so the two definitions of
    "starter" agree by construction and a test of the *study* is not a test of the reconciliation.
    A game with a starter and no score is an in-flight one: its first pass is thrown, no
    result. `drop` names (game_id, team) pairs left out, a game with no pass play yet. A
    run play precedes each first pass, and a later pass by a backup follows it, so the rule
    is read off the first *pass* and not the first play or the last passer."""
    tg = se.team_games(rows_df)
    frames = []
    for play_id, kind, who in ((1, "run", None), (2, "pass", "first"), (3, "pass", "later")):
        frames.append(tg.select(
            pl.col("game_id"), pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
            pl.lit("REG").alias("season_type"), pl.lit(float(play_id)).alias("play_id"),
            pl.col("team").alias("posteam"), pl.lit(kind).alias("play_type"),
            (pl.col("qb") if who == "first" else
             pl.lit("00-backup") if who == "later" else pl.lit(None, dtype=pl.Utf8))
            .alias("passer_player_id"), pl.col("team")))
    # `nflverse.PBP`'s contract refuses fewer than 1,000 rows, so a recording of a handful of
    # plays would not reach the reader at all; the padding is kickoffs of no game on the
    # schedule, which name no passer and join nothing.
    # One block per season, because the reader asks for one season at a time and the contract
    # holds each of those reads to the floor.
    pad = pl.concat([pl.DataFrame({
        "game_id": ["pad"] * 1000, "season": [year] * 1000, "week": [1] * 1000,
        "season_type": ["REG"] * 1000, "play_id": [float(i) for i in range(1000)],
        "posteam": [None] * 1000, "play_type": ["kickoff"] * 1000,
        "passer_player_id": [None] * 1000, "team": [None] * 1000},
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "season_type": pl.Utf8, "play_id": pl.Float64, "posteam": pl.Utf8,
                "play_type": pl.Utf8, "passer_player_id": pl.Utf8, "team": pl.Utf8})
        for year in range(2020, 2031)])
    out = pl.concat([*frames, pad]).sort("game_id", "team", "play_id", nulls_last=True)
    for game_id, team in drop:
        out = out.filter(~((pl.col("game_id") == game_id) & (pl.col("team") == team)))
    return out.drop("team")


def schedule_for(rows_df):
    """nflverse's schedule for the source's rows: the game day, and the result where there is
    one -- null until played, as the wire's is."""
    tg = se.team_games(rows_df)
    home = tg.filter(pl.col("home"))
    return home.select(
        pl.col("game_id"), pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32),
        pl.lit("REG").alias("game_type"), pl.col("date").alias("gameday"),
        pl.col("game_id").str.split("_").list.get(3).alias("home_team"),
        pl.col("game_id").str.split("_").list.get(2).alias("away_team"),
        pl.col("score").alias("home_score"), pl.col("opp_score").alias("away_score"))


def serve_world(rows_df, *, drop=(), pbp=None):
    """Select a Replay holding the play-by-play and schedule `rows_df` implies, so the CLI's
    events source is read through the nflverse seam offline. `pbp` replaces the derived one."""
    frame = pbp_for(rows_df, drop=drop) if pbp is None else pbp
    return replay.serve(
        pbp=lambda keys: frame.filter(pl.col("season").is_in([int(k) for k in keys])),
        schedules=schedule_for(rows_df))




# --- the gate ----------------------------------------------------------------------------


ARCHIVE = polls([
    # KC-LA (week 3): a lookahead frozen at +3 through the week-2 game day, then repriced
    # after Gabbert is named. Game day 2026-09-27; the previous KC game day 2026-09-20.
    ("2026_03_LA_KC", 3.0, utc(15), 3), ("2026_03_LA_KC", 3.0, utc(19), 3),
    ("2026_03_LA_KC", 3.0, utc(20, 12), 3),    # still the week-2 game day: not before it
    ("2026_03_LA_KC", -1.0, utc(23), 3), ("2026_03_LA_KC", -2.0, utc(26), 3),
    # KC-DEN (week 4): Mahomes back; frozen before 2026-09-27, close before 2026-10-04.
    ("2026_04_DEN_KC", -3.0, utc(24), 4), ("2026_04_DEN_KC", 4.0, utc(30), 4),
    # Another week-3 game, no change: the week's mean move between the same two poll days.
    ("2026_03_SF_DEN", 1.0, utc(19), 3), ("2026_03_SF_DEN", 2.0, utc(26), 3),
])


# --- the event source is play-by-play (#420, ADOPTED (B) on #421) -----------------------------
#
# Rule 18: each check below is run against the condition it exists to detect. The mutation
# each one was seen red under is named in its docstring.


def pbp_events(rows_df, pbp=None):
    """The in-season events the study reads, built as `study_events` builds them."""
    frame = pbp_for(rows_df) if pbp is None else pbp
    tg = se.team_games_from_pbp(frame, schedule_for(rows_df))
    return tg, se.in_season_events(se.events(se.observed(tg)))


def played_rows(rows_df):
    """The fixture with its week-4 game played, so a game with a result and no observed
    starter is something a test can make."""
    return rows_df.with_columns(pl.col("score1").fill_null(24), pl.col("score2").fill_null(17))
