"""Measure how players' weekly scores move together.

`hub.models.predict.TEAMMATE_RHO` carries three within-team edges, measured this way.
`docs/correlation.md` names what it does not cover:

> **Opponent correlation is not modelled.** [...] Shared game exposure shrinks *margin*
> variance, which helps whoever is favoured. Nobody prices it here either.

This module measures it, so that sentence can be replaced with a number. It is deliberately a
*measurement* and not a model: nothing in the repo consumes opponent correlation yet, and
wiring an unused coefficient into the simulator would be adding a parameter to buy nothing.

**Method, identical to the teammate measurement so the two are comparable.** Standardise each
player's weekly PPR points within his own season — requiring at least `MIN_WEEKS` weeks and a
non-zero spread, so a player with two appearances cannot define a correlation — then pair
players who appeared in the same game and correlate the standardised values by position pair.

Same game, opposite teams is the opponent set. Same game, same team is the teammate set, and
running this module over it reproduces `TEAMMATE_RHO` — which is the check that the method here
is the method there.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import polars as pl

# A player needs a real sample before his own mean and spread mean anything. Eight weeks is
# half a season; below it, one big game defines the standardisation.
MIN_WEEKS = 8

# Positions this league drafts. K and DST are excluded for the same reason they are excluded
# from the board.
POSITIONS = ("QB", "RB", "WR", "TE")

# A correlation on fewer pairs than this is not reported. At n=200 the standard error is
# already 0.071, which is wider than every effect the teammate measurement found.
MIN_PAIRS = 200


def standardised(stats: pl.DataFrame, *, min_weeks: int = MIN_WEEKS) -> pl.DataFrame:
    """Weekly points as a z-score within each player-season.

    Within-season, not within-career: a player's role changes between years, and pooling
    across them would mix a starter's weeks with a backup's and call the difference variance.
    """
    need = {"season", "week", "game_id", "team", "player_id", "position", "fantasy_points_ppr"}
    missing = sorted(need - set(stats.columns))
    if missing:
        raise ValueError(f"player stats missing {missing}")

    d = (stats.filter(pl.col("position").is_in(POSITIONS))
              .select("season", "week", "game_id", "team", "player_id", "position",
                      pl.col("fantasy_points_ppr").fill_null(0.0).alias("pts")))
    if "season_type" in stats.columns:
        d = d.join(stats.select("player_id", "week", "season", "season_type"),
                   on=["player_id", "week", "season"], how="left")
        d = d.filter((pl.col("season_type") == "REG") | pl.col("season_type").is_null())
        d = d.drop("season_type")

    per = (d.group_by(["player_id", "season"])
             .agg(pl.len().alias("n"), pl.col("pts").mean().alias("m"),
                  pl.col("pts").std().alias("s"))
             .filter((pl.col("n") >= min_weeks) & (pl.col("s") > 0)))
    return (d.join(per, on=["player_id", "season"])
             .with_columns(((pl.col("pts") - pl.col("m")) / pl.col("s")).alias("z")))


def pair_correlations(z: pl.DataFrame, *, same_team: bool,
                      min_pairs: int = MIN_PAIRS) -> pl.DataFrame:
    """Correlation of standardised points by position pair, within a game.

    `same_team=True` reproduces the teammate measurement; `False` gives the opponent one.
    Returns (pos_a, pos_b, n, rho, se), sorted by n.
    """
    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in z.select("game_id", "team", "position", "z").iter_rows(named=True):
        by_game[r["game_id"]].append(r)

    xs: dict[tuple[str, str], tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    for players in by_game.values():
        for a, b in itertools.combinations(players, 2):
            if (a["team"] == b["team"]) is not same_team:
                continue
            key = (a["position"], b["position"])
            key = key if key[0] <= key[1] else (key[1], key[0])
            xs[key][0].append(a["z"])
            xs[key][1].append(b["z"])

    rows = []
    for (pa, pb), (u, v) in xs.items():
        if len(u) < min_pairs:
            continue
        r = float(np.corrcoef(np.array(u), np.array(v))[0, 1])
        rows.append({"pos_a": pa, "pos_b": pb, "n": len(u), "rho": r,
                     "se": 1.0 / np.sqrt(len(u) - 3)})
    if not rows:
        return pl.DataFrame(schema={"pos_a": pl.Utf8, "pos_b": pl.Utf8, "n": pl.Int64,
                                    "rho": pl.Float64, "se": pl.Float64})
    return pl.DataFrame(rows).sort("n", descending=True)


def significant(table: pl.DataFrame, se_threshold: float = 2.0) -> pl.DataFrame:
    """Rows whose correlation clears `se_threshold` standard errors of zero."""
    if table.is_empty():
        return table
    return (table.with_columns((pl.col("rho").abs() / pl.col("se")).alias("se_away"))
                 .filter(pl.col("se_away") >= se_threshold)
                 .sort("se_away", descending=True))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.models.correlate",
        description="Within-game correlation of standardised weekly points.")
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    ap.add_argument("--teammates", action="store_true",
                    help="same-team pairs instead of opponents; reproduces TEAMMATE_RHO")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import nflreadpy as nfl

    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    print(f"  loading weekly player stats for {seasons} ...")
    z = standardised(nfl.load_player_stats(seasons=seasons, summary_level="week"))
    print(f"  {z.height} player-weeks after standardising "
          f"(>= {MIN_WEEKS} weeks, non-zero spread)")

    table = pair_correlations(z, same_team=a.teammates)
    label = "TEAMMATE" if a.teammates else "OPPONENT"
    print(f"\n  {label} correlation, within game\n")
    print(f"  {'pair':<10} {'n':>7} {'rho':>9} {'se':>8} {'se away':>9}")
    for r in table.iter_rows(named=True):
        print(f"  {r['pos_a']}-{r['pos_b']:<8} {r['n']:>7} {r['rho']:>+9.4f} "
              f"{r['se']:>8.4f} {abs(r['rho']) / r['se']:>9.1f}")
    sig = significant(table)
    print(f"\n  {sig.height} of {table.height} pairs clear two standard errors.")
    if a.out:
        table.write_parquet(a.out)
        print(f"  wrote {table.height} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
