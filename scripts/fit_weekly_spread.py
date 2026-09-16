"""Fit the weekly law `sd = k * sqrt(mu)` and the weekly skew, with a season held out (#294).

`predict.WEEKLY_K` / `WEEKLY_K_POOLED` and `WEEKLY_SKEW` / `WEEKLY_SKEW_POOLED` were fitted
on 2026-08-23 (docs/weekly-spread.md, docs/component-projection.md) and no script in the
tree reproduced them until #294. This is `hub.models.spread.fit_weekly_law` on the
qualifying player-seasons `spread.player_seasons` selects -- regular season, the four
drafted positions, at least 8 games, over 3 points a game -- read season by season through
`hub.fetch.nflverse.load`, so a hold-out never keys a new cache entry.

    uv run python scripts/fit_weekly_spread.py
    uv run python scripts/fit_weekly_spread.py --exclude-season 2024 --record

**What it reproduces, measured 2026-09-13 on the pinned archive.** With nothing excluded:
1,174 qualifying player-seasons, the doc's count; `k` = 1.880 / 2.068 / 2.125 / 1.994,
pooled 2.042, the shipped 1.88 / 2.07 / 2.13 / 1.99 / 2.04 to the precision they ship at;
exponents 0.161 / 0.469 / 0.518 / 0.622, pooled 0.498, the doc's table exactly. So
`WEEKLY_K` is reproduced and is what `--record` writes.

**The skew it does not reproduce, and does not record.** The shipped `WEEKLY_SKEW` is the
observed column of a 760-player-season validation in docs/component-projection.md whose
code was never committed; the within-player-season population skew over these 1,174 reads
0.13 / 0.78 / 0.73 / 0.78, pooled 0.68, against the shipped 0.15 / 0.67 / 0.66 / 0.72,
pooled 0.60. It is printed beside the shipped value and recorded as *not refitted*, because
a hold-out set carrying a different estimator's skew would confound the estimator with the
season it holds out. Nothing here writes a constant into `src/`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import polars as pl

from hub import holdout
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.fetch import nflverse
from hub.models.predict import WEEKLY_K, WEEKLY_K_POOLED, WEEKLY_SKEW, WEEKLY_SKEW_POOLED
from hub.models.spread import fit_weekly_law, player_seasons, without_season

DEFAULT_SEASONS = (2022, 2023, 2024, 2025)
SCRIPT = "scripts/fit_weekly_spread.py"
# The one cache entry every run reads, filtered locally: the standard slice over every
# season the archive is pinned for. A hold-out, or a shorter `--seasons`, must not key a new
# entry -- a fetch is the network, and a per-season entry can be a stub nothing pinned.
ARCHIVE_SEASONS = tuple(range(2019, 2026))
SKEW_NOT_REFITTED = ("WEEKLY_SKEW is not refitted: the shipped estimator (a 760-player-season "
                     "validation, docs/component-projection.md) is not in the tree, and this "
                     "script's reads 0.68 pooled against the shipped 0.60 on the full sample, "
                     "so a set carrying it would confound the estimator with the season")


def weekly_stats(seasons: Sequence[int]) -> pl.DataFrame:
    """The standard weekly slice for `seasons`, read from the pinned multi-season entry."""
    whole = nflverse.load("player_stats", list(ARCHIVE_SEASONS),
                          cols=list(nflverse.PLAYER_STATS_COLS))
    return whole.filter(pl.col("season").is_in(list(seasons)))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=SCRIPT, description="Fit WEEKLY_K and WEEKLY_SKEW.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    holdout.add_arguments(ap)
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    note = holdout.recording(a, holdout.command_line(SCRIPT, a))

    try:
        stats = weekly_stats(seasons)
    except Exception as e:
        return unavailable(SCRIPT, "nflverse weekly player stats", e)
    ps = without_season(player_seasons(stats), a.exclude_season)
    kept = sorted(ps["season"].unique().to_list())
    print(f"  {ps.height} qualifying player-seasons over {kept}"
          + (f" (season {a.exclude_season} held out)" if a.exclude_season else ""))
    got = fit_weekly_law(ps)
    print(f"\n  {'pos':>6} {'n':>5} {'k':>6} {'shipped':>8} {'exponent':>9} {'skew':>6} "
          f"{'shipped':>8}")
    for pos in (*DRAFTED_POSITIONS, "pooled"):
        if pos not in got["k"]:
            continue
        k_ship = WEEKLY_K_POOLED if pos == "pooled" else WEEKLY_K.get(pos, WEEKLY_K_POOLED)
        s_ship = WEEKLY_SKEW_POOLED if pos == "pooled" else WEEKLY_SKEW.get(pos, WEEKLY_SKEW_POOLED)
        print(f"  {pos:>6} {got['n'][pos]:>5} {got['k'][pos]:>6.3f} {k_ship:>8.2f} "
              f"{got['exponent'][pos]:>9.3f} {got['skew'][pos]:>6.2f} {s_ship:>8.2f}")
    k_pos = {p: round(got["k"][p], 2) for p in DRAFTED_POSITIONS if p in got["k"]}
    note("predict.WEEKLY_K", k_pos)
    note("predict.WEEKLY_K_POOLED", round(got["k"]["pooled"], 2))
    # The skew is printed and not recorded: this estimator does not reproduce the shipped
    # one on the full sample, so a set carrying it would confound the estimator with the
    # season it holds out. The replay keeps the shipped skew and the run line says so.
    for key in ("predict.WEEKLY_SKEW", "predict.WEEKLY_SKEW_POOLED"):
        note(key, None, why_not=SKEW_NOT_REFITTED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
