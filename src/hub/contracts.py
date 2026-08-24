"""Data contracts. The Week 7 failure mode is not a crash -- it is ESPN silently
renaming a field and your projections going quietly wrong for three weeks.

Every fetch function asserts its contract at the boundary. Violations raise loudly
and the pipeline serves last-good state rather than propagating bad data.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import cast
import polars as pl


class ContractViolation(Exception):
    pass


@dataclass(frozen=True)
class Contract:
    name: str
    required: dict[str, type]                      # column -> polars dtype family
    non_null: tuple[str, ...] = ()
    unique: tuple[str, ...] = ()
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    min_rows: int = 1

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        problems = []
        if df.height < self.min_rows:
            problems.append(f"{df.height} rows < min {self.min_rows}")
        missing = set(self.required) - set(df.columns)
        if missing:
            problems.append(f"missing columns: {sorted(missing)}")
        for c in self.non_null:
            if c in df.columns and df[c].null_count():
                problems.append(f"{c} has {df[c].null_count()} nulls")
        for c in self.unique:
            if c in df.columns and df[c].n_unique() != df.height:
                problems.append(f"{c} not unique ({df[c].n_unique()}/{df.height})")
        for c, (lo, hi) in self.ranges.items():
            if c in df.columns:
                mn, mx = df[c].min(), df[c].max()
                if mn is not None and (cast(float, mn) < lo or cast(float, mx) > hi):
                    problems.append(f"{c} range [{mn}, {mx}] outside [{lo}, {hi}]")
        if problems:
            raise ContractViolation(f"{self.name}: " + "; ".join(problems))
        return df


DRAFT_BOARD = Contract(
    name="draft_board",
    required={"player": pl.Utf8, "pos": pl.Utf8, "ecr": pl.Float64},
    non_null=("player", "ecr"),
    unique=("player",),
    ranges={"ecr": (1, 1000)},
    min_rows=300,
)

FF_OPPORTUNITY = Contract(
    name="ff_opportunity",
    required={"player_id": pl.Utf8, "position": pl.Utf8, "total_fantasy_points_exp": pl.Float64},
    non_null=("player_id",),
    ranges={"total_fantasy_points_exp": (-10, 80)},
    min_rows=1000,
)

PBP = Contract(
    name="pbp",
    required={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32},
    non_null=("game_id", "season", "week"),
    # Ranges from three observed seasons (2023-25, 147,928 plays) widened ~20%, per the
    # rule in the data-contracts skill: set them from history, not theory. Observed epa
    # ran -12.69..8.88 and yards_gained -34..98.
    #
    # Deliberately NOT non_null: posteam. It is null on 8,080 of those plays -- kickoffs,
    # timeouts, end-of-quarter rows -- so requiring it would fail every honest refresh.
    ranges={"week": (1, 22), "epa": (-16, 12), "wp": (0, 1), "yards_gained": (-45, 120)},
    min_rows=1000,
)

SCHEDULES = Contract(
    name="nflverse_schedules",
    required={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
              "home_team": pl.Utf8, "away_team": pl.Utf8},
    non_null=("game_id", "season", "week", "home_team", "away_team"),
    unique=("game_id",),
    # spread_line is the home side, positive when the home team is favoured. The widest
    # NFL closing spreads on record sit around 26.5; 40 leaves room without admitting a
    # sign flip, which would show as a plausible number on the wrong team.
    ranges={"week": (1, 22), "spread_line": (-40, 40), "total_line": (20, 80)},
    min_rows=1,
)

CFBD_GAMES = Contract(
    name="cfbd_games",
    required={"id": pl.Int64, "season": pl.Int64, "week": pl.Int64,
              "homeTeam": pl.Utf8, "awayTeam": pl.Utf8},
    non_null=("id", "season", "week", "homeTeam", "awayTeam"),
    unique=("id",),
    # CFB runs longer than the NFL: 15 regular-season weeks plus postseason.
    ranges={"week": (1, 20), "homePoints": (0, 120), "awayPoints": (0, 120)},
    min_rows=1,
)

CFBD_LINES = Contract(
    name="cfbd_lines",
    required={"id": pl.Int64, "season": pl.Int64, "week": pl.Int64,
              "homeTeam": pl.Utf8, "awayTeam": pl.Utf8},
    non_null=("id", "homeTeam", "awayTeam"),
    unique=("id",),
    # College spreads reach much further than NFL ones -- 50+ happens in September.
    ranges={"week": (1, 20)},
    min_rows=1,
)

ODDS_SNAPSHOT = Contract(
    name="odds_snapshot",
    required={"game_id": pl.Utf8, "close_spread": pl.Float64,
              "captured_at": pl.Datetime},
    non_null=("game_id", "close_spread", "captured_at"),
    # Deliberately NOT unique on game_id: several snapshots per game is the entire point,
    # and it is what makes AS_OF_LINES more than a normal join.
    ranges={"close_spread": (-40, 40)},
    min_rows=1,
)

ESPN_SCOREBOARD = Contract(
    name="espn_scoreboard",
    required={"id": pl.Utf8, "state": pl.Utf8, "home": pl.Utf8, "away": pl.Utf8},
    non_null=("id", "state", "home", "away"),
    unique=("id",),
    min_rows=1,
)
