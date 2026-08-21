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

ESPN_SCOREBOARD = Contract(
    name="espn_scoreboard",
    required={"id": pl.Utf8, "state": pl.Utf8, "home": pl.Utf8, "away": pl.Utf8},
    non_null=("id", "state", "home", "away"),
    unique=("id",),
    min_rows=1,
)
