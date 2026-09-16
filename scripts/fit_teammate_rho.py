"""Fit TEAMMATE_RHO -- the within-game correlation of teammates -- with a season held out (#294).

`predict.TEAMMATE_RHO` carries three quarterback edges measured by the method in
`hub.models.correlate` (`python -m hub.models.correlate --teammates` reproduces the table
from a live nflreadpy load). This is that measurement on the one **cached** slice that
carries teams -- the Panel's `WEEK_STATS_COLS` entry over `panel.SEASONS`, read once and
filtered locally, so a hold-out never keys a new cache entry -- with the game keyed from the
two teams (`correlate.with_game_key`), since that slice carries no `game_id`.

    uv run python scripts/fit_teammate_rho.py
    uv run python scripts/fit_teammate_rho.py --exclude-season 2024 --record

**What it reproduces, measured 2026-09-13.** With nothing excluded, on 2022-25: QB-WR
+0.222 (n 7,998), QB-TE +0.205 (3,887), QB-RB +0.052 (4,927), against the shipped +0.232 /
+0.225 / +0.054 (docs/correlation.md: 6,352 / 3,057 / 4,077). Each is inside 0.02 of the
shipped value and inside two of its own standard errors (0.011-0.016); the shipped run
predates this module, read nflreadpy's own `game_id` on a smaller archive, and its exact
sample is not in the tree. The gap is stated in the set it records. Nothing here writes a
constant into `src/`.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import polars as pl

from hub import holdout
from hub.cli import unavailable
from hub.fetch import nflverse
from hub.models import panel
from hub.models.correlate import (
    pair_correlations,
    standardised,
    teammate_rho_from,
    with_game_key,
)
from hub.models.predict import TEAMMATE_RHO

DEFAULT_SEASONS = (2022, 2023, 2024, 2025)
SCRIPT = "scripts/fit_teammate_rho.py"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=SCRIPT, description="Fit TEAMMATE_RHO.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    holdout.add_arguments(ap)
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    if a.exclude_season is not None:
        seasons = [s for s in seasons if s != a.exclude_season]
    note = holdout.recording(a, holdout.command_line(SCRIPT, a))

    try:
        whole = nflverse.load("player_stats", list(panel.SEASONS),
                              cols=list(panel.WEEK_STATS_COLS))
    except Exception as e:
        return unavailable(SCRIPT, "nflverse weekly player stats (the Panel's slice)", e)
    z = standardised(with_game_key(whole.filter(pl.col("season").is_in(seasons))))
    print(f"  {z.height} standardised player-weeks over {seasons}"
          + (f" (season {a.exclude_season} held out)" if a.exclude_season else ""))
    table = pair_correlations(z, same_team=True)
    print(f"\n  {'pair':<8} {'n':>7} {'rho':>8} {'se':>7} {'shipped':>8}")
    for r in table.iter_rows(named=True):
        key = (r["pos_a"], r["pos_b"])
        ship = TEAMMATE_RHO.get(key, TEAMMATE_RHO.get(key[::-1]))
        shown = f"{ship:>+8.3f}" if ship is not None else f"{'--':>8}"
        print(f"  {r['pos_a']}-{r['pos_b']:<5} {r['n']:>7} {r['rho']:>+8.3f} {r['se']:>7.3f} "
              f"{shown}")
    rho = teammate_rho_from(table)
    if set(rho) != set(TEAMMATE_RHO):
        print(f"  the table holds {sorted(rho)} of the shipped edges {sorted(TEAMMATE_RHO)}; "
              f"nothing recorded", file=sys.stderr)
        return 1
    note("predict.TEAMMATE_RHO", rho)
    return 0


if __name__ == "__main__":
    sys.exit(main())
