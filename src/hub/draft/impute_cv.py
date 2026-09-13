"""IMPUTE_CV measured on the players the board actually imputes: rookies (#277).

`hub.models.predict.IMPUTE_CV` / `IMPUTE_CV_BY_POS` say how wrong `board._impute_xfp`'s
rank curve is, as a share of the projection it invents. The shipped values were measured
leave-one-out on observed **veterans** inside the top 200: each was blanked and re-imputed
from the rest. That is a lower bound on the error the flag is carrying, because the players
the board imputes are **rookies** -- no prior NFL season, so no prior-season xFP -- and a
veteran's rank was set partly by the very production the curve predicts, where a rookie's
rank carries no such information.

**The population, as #277's disposition fixed it.** Rookies are players with no prior NFL
season who appear inside the top 200 by consensus on that season's board; the realised
season is scored against the curve's prediction, and the residual CV is taken per position
and pooled -- the same statistic as the shipped constant, `sd(realised / imputed - 1)`, so
the two are comparable. Two departures from the disposition's wording, each stated:

* *Top 200 by consensus rank, not by ADP.* ESPN publishes ADP for the current season only
  (ADR-0010; `backtest.LIMITATIONS`), so a past board has no ADP to rank on. Consensus is
  the rank the curve reads, which is the rank the residual is a function of.
* *"No prior NFL season" is read as "first season with any weekly line in nflverse player
  stats".* nflverse's `rookie_year` lives on its rosters table, which no loader in this
  repo carries; the weekly stats are cached 2019 onward, so a 2021 board looks back two
  seasons and a 2025 board six. A player who was rostered but recorded nothing for a whole
  season before his first line would be mis-read as a rookie; inside the top 200 that is
  rare and it is said rather than hidden.

Realised production comes from `board.expected_points(season)` -- the season's own
`ff_opportunity` rows -- as **both** realised PPR points per game (`ppg`, the disposition's
quantity) and the season's own xFP per game (`xfp_pg`, the quantity the curve
imputes and the one the veteran measurement compared against). Both CVs are printed; the
first is what the ticket asked for, the second is the like-for-like beside the shipped
number.

Nothing here writes a constant. The shipped values keep shipping until a decision moves
them; this module and `scripts/fit_impute_cv.py` are the committed fitting code the
constant's docstring names as its successor.

    uv run python scripts/fit_impute_cv.py
    uv run python scripts/fit_impute_cv.py --seasons 2021,2022,2023,2024,2025 --min-games 8
    uv run python scripts/fit_impute_cv.py --exclude-season 2024
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.declare import not_an_input
from hub.names import player_key

# The board depth the disposition named and the veteran measurement used: inside rank 200.
TOP = not_an_input(
    200,
    "the board depth this measurement is taken inside, the same depth the shipped "
    "veteran measurement used; a rule of the fit and not an input to a prediction")
# Fewest games before a per-game rate is a rate. Eight is half a season and the sample rule
# docs/weekly-spread.md fitted under; the board's own MIN_GAMES (10) is a replacement-level
# rule and would drop a rookie hurt in November whose ten weeks are a measurement.
MIN_GAMES = not_an_input(
    8,
    "a sample threshold of this measurement: fewest games before a rookie's per-game "
    "rate is read, and nothing a board or a simulation computes from")
# The boards a rookie residual can be measured on from the archive: the consensus archive
# starts 2019-12-27, so no 2019 preseason board exists, and 2020's as-of is not cached.
DEFAULT_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025)

FIRST_STATS_SEASON = not_an_input(
    2019,
    "the first season the cached weekly stats reach: the look-back a rookie proxy has, "
    "a fact about the archive and not an input to any prediction")


def first_seasons(stats: pl.DataFrame) -> pl.DataFrame:
    """`(player, first_season)`: the first season each player has any weekly line, keyed the
    way the board joins. The proxy for nflverse `rookie_year` -- see the module docstring."""
    key = pl.col("player_display_name").map_elements(player_key, return_dtype=pl.Utf8)
    return (stats.select(key.alias("player"), pl.col("season").cast(pl.Int64))
                 .group_by("player").agg(pl.col("season").min().alias("first_season")))


def rookie_rows(board: pl.DataFrame, stats: pl.DataFrame, realised: pl.DataFrame, *,
                season: int, top: int = TOP,
                min_games: int = MIN_GAMES) -> tuple[pl.DataFrame, dict[str, int]]:
    """One row per rookie the board imputed and the season measured, and the counts on the way.

    The counts are the population rules applied in order: inside the top `top` by consensus;
    a rookie (first season line is `season`); imputed by the board -- a rookie carrying a
    prior xFP is a join collision and is counted under `collisions`, not measured; holding a
    season line at all; and at least `min_games` games played. `no_line` counts the imputed
    players inside the top `top` with no weekly line in any season, whom the proxy cannot
    classify either way.
    """
    # The five columns the rules read, and no others: a served board also carries a prior
    # season's `games`, `fp` and `xfp`, null for every rookie, and a join that let them
    # through would filter on the wrong `games`.
    keyed = (board.select("player", "pos", "ecr", "xfp_per_game", "xfp_imputed")
                  .with_columns(pl.col("player").map_elements(player_key, return_dtype=pl.Utf8)
                                .alias("player")))
    top_n = keyed.filter((pl.col("ecr") <= top) & pl.col("pos").is_in(DRAFTED_POSITIONS))
    first = first_seasons(stats)
    rookies = (top_n.join(first, on="player", how="inner")
                    .filter(pl.col("first_season") == season))
    collisions = rookies.filter(~pl.col("xfp_imputed"))
    imputed = rookies.filter(pl.col("xfp_imputed"))
    real = (realised.select(
                pl.col("full_name").map_elements(player_key, return_dtype=pl.Utf8).alias("player"),
                pl.col("games").cast(pl.Int64),
                (pl.col("fp") / pl.col("games")).alias("ppg"),
                (pl.col("xfp") / pl.col("games")).alias("xfp_pg"))
            .filter(pl.col("games") > 0)
            .unique(subset=["player"], keep="first"))
    lined = imputed.join(real, on="player", how="inner")
    played = lined.filter(pl.col("games") >= min_games)
    # Imputed inside the top `top` and absent from the weekly stats in every season: the
    # proxy cannot classify these, so they are counted rather than folded into either side.
    no_line = (top_n.filter(pl.col("xfp_imputed"))
                    .join(first, on="player", how="anti"))
    rows = played.select(
        pl.lit(season, dtype=pl.Int64).alias("season"), "player", "pos", "ecr",
        pl.col("xfp_per_game").alias("imputed"), "games", "ppg", "xfp_pg",
    ).sort("ecr")
    counts = {"top200": top_n.height, "rookies": rookies.height, "imputed": imputed.height,
              "with_line": lined.height, "played": played.height,
              "collisions": collisions.height, "no_line": no_line.height}
    return rows, counts


def residual_cv(rows: pl.DataFrame, against: str) -> dict[str, dict[str, Any]]:
    """`sd(realised / imputed - 1)` per position and pooled, with n and the median.

    The shipped constant's statistic, so the rookie number reads beside it. A position with
    fewer than two rows has no spread and carries `None` rather than a zero that would read
    as a measurement.
    """
    out: dict[str, dict[str, Any]] = {}
    ratio = (pl.col(against) / pl.col("imputed") - 1.0)
    for pos in (*DRAFTED_POSITIONS, "pooled"):
        sub = rows if pos == "pooled" else rows.filter(pl.col("pos") == pos)
        r = sub.select(ratio.alias("r"))["r"].to_numpy()
        out[pos] = {
            "n": int(r.size),
            "cv": float(np.std(r, ddof=1)) if r.size >= 2 else None,
            "median": float(np.median(r)) if r.size else None,
        }
    return out


def hold_out(rows: pl.DataFrame, exclude: int | None) -> tuple[pl.DataFrame, str]:
    """The rows with one season held out, and the sentence that says so (#294)."""
    seasons = sorted(rows["season"].unique().to_list())
    if exclude is None:
        return rows, f"  seasons {seasons}, nothing held out"
    kept = rows.filter(pl.col("season") != exclude)
    remain = sorted(kept["season"].unique().to_list())
    return kept, f"  season {exclude} held out; fitted on {remain}"


def player_bootstrap_se(rows: pl.DataFrame, against: str, *, draws: int = 2000,
                        seed: int = 0) -> float | None:
    """Standard error of the pooled CV over players, the way the shipped number quoted its
    se (0.010 over players). Rookies are one season each, so the player is the row; the
    seasons are the clusters and are counted rather than resampled at this n."""
    r = rows.select((pl.col(against) / pl.col("imputed") - 1.0).alias("r"))["r"].to_numpy()
    if r.size < 3:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, r.size, size=(draws, r.size))
    return float(np.std(np.std(r[idx], axis=1, ddof=1), ddof=1))


def measure(seasons: Sequence[int], *, exclude: int | None = None, min_games: int = MIN_GAMES,
            build_board=None, load_stats=None, load_realised=None) -> tuple[dict, list[str]]:
    """The measurement over `seasons`, and the lines a script prints. The three loaders
    are seams for a test; the defaults read the archive through the repo's own readers."""
    from hub.draft.board import board_as_of, expected_points
    from hub.fetch import nflverse
    from hub.models.predict import IMPUTE_CV, IMPUTE_CV_BY_POS

    build_board = build_board or (lambda yr: board_as_of(yr)[0])
    load_realised = load_realised or expected_points
    if load_stats is None:
        def load_stats():
            span = list(range(FIRST_STATS_SEASON, max(seasons) + 1))
            return nflverse.load("player_stats", span, cols=list(nflverse.PLAYER_STATS_COLS))
    stats = load_stats()
    lines: list[str] = []
    frames, tallies = [], {}
    for yr in seasons:
        rows, counts = rookie_rows(build_board(yr), stats, load_realised(yr),
                                   season=yr, min_games=min_games)
        frames.append(rows)
        tallies[yr] = counts
        lines.append(f"  {yr}: top {TOP} {counts['top200']}, rookies {counts['rookies']}, "
                     f"imputed {counts['imputed']} (collisions {counts['collisions']}, "
                     f"imputed with no line in any season {counts['no_line']}), "
                     f"with a season line {counts['with_line']}, "
                     f">= {min_games} games {counts['played']}")
    rows = pl.concat(frames)
    kept, said = hold_out(rows, exclude)
    lines.append(said)
    result: dict[str, Any] = {"seasons": list(seasons), "exclude": exclude,
                              "n": kept.height, "clusters": kept["season"].n_unique(),
                              "tallies": tallies, "min_games": min_games}
    for against, label in (("ppg", "realised PPR points per game"),
                           ("xfp_pg", "the season's own xFP per game")):
        cv = residual_cv(kept, against)
        se = player_bootstrap_se(kept, against)
        result[against] = {"by_position": cv, "se": se}
        lines.append(f"\n  residual CV against {label}: n = {kept.height} rookies over "
                     f"{result['clusters']} seasons (the clusters)")
        lines.append(f"  {'pos':>6} {'n':>4} {'rookie cv':>10} {'median':>8} {'shipped':>8}")
        for pos in (*DRAFTED_POSITIONS, "pooled"):
            c = cv[pos]
            shipped = IMPUTE_CV if pos == "pooled" else IMPUTE_CV_BY_POS.get(pos, IMPUTE_CV)
            cv_s = f"{c['cv']:.3f}" if c["cv"] is not None else "n/a"
            md_s = f"{c['median']:+.2f}" if c["median"] is not None else "n/a"
            lines.append(f"  {pos:>6} {c['n']:>4} {cv_s:>10} {md_s:>8} {shipped:>8.3f}")
        if se is not None:
            lines.append(f"  pooled se over players {se:.3f}")
    return result, lines


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts/fit_impute_cv.py",
        description="Measure IMPUTE_CV on rookies -- the players the board imputes (#277).")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    ap.add_argument("--exclude-season", type=int, default=None,
                    help="hold one season out of the measurement (#294)")
    ap.add_argument("--min-games", type=int, default=MIN_GAMES)
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    try:
        _result, lines = measure(seasons, exclude=a.exclude_season, min_games=a.min_games)
    except Exception as e:
        return unavailable("scripts/fit_impute_cv.py", "the boards and seasons of the archive", e)
    print("\n".join(lines))
    print("\n  the shipped IMPUTE_CV was measured on blanked veterans and is unchanged by this "
          "run; moving it is a decision, not a script's side effect")
    return 0


if __name__ == "__main__":                       # pragma: no cover - entry point
    sys.exit(main())
