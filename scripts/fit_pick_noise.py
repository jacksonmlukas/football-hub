"""Fit the pick-noise line on this league's drafts, and sweep the ceiling it is cut at (#287).

`hub.draft.availability.PICK_NOISE_INTERCEPT` / `PICK_NOISE_SLOPE` are `sigma(pick) = a + b *
pick`, fitted by `noise_from_picks` on the picks whose consensus rank sits under
`PICK_NOISE_FIT_CEILING`. Until #287 the only committed way to run that fit was the four-line
snippet in `docs/pick-noise.md`; this is the same fit as a script, and it prints two things
the constant's docstring now carries:

* the fitter's own sentence, which since #287 states how many distinct resamples four
  clusters can give the interval and what such an interval cannot resolve;
* the ceiling sweep -- one row per ceiling in `--ceilings`, default the
  {120, 144, 168, 192, 216} the ticket named -- so the cut is published as a sensitivity
  and not asserted.

    uv run python scripts/fit_pick_noise.py
    uv run python scripts/fit_pick_noise.py --ceilings 120,144,168,192,216

It needs an ESPN session: the drafts live nowhere but `historical_picks`, which reads the
league's draft history through `espn_api`. Nothing here writes a constant. If the sweep
shows the shipped ceiling on a cliff, that is a separate decision (#287's disposition), not
a number this script moves.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from hub import holdout
from hub.cli import unavailable
from hub.draft.availability import (
    PICK_NOISE_FIT_CEILING,
    PICK_NOISE_INTERCEPT,
    PICK_NOISE_SLOPE,
    PICK_NOISE_SWEEP_CEILINGS,
    historical_picks,
    noise_from_picks,
    sweep_ceilings,
)

DEFAULT_SEASONS = (2022, 2023, 2024, 2025)
SCRIPT = "scripts/fit_pick_noise.py"
KEYS = ("availability.PICK_NOISE_INTERCEPT", "availability.PICK_NOISE_SLOPE")


def sweep_lines(rows: Sequence[dict]) -> list[str]:
    """The sweep as the table the module docstring carries."""
    out = [f"  {'ceiling':>7} {'applied':>7} {'n':>5} {'a':>6} {'slope':>6} "
           f"{'95% CI':>16} {'sigma@100':>10}"]
    for r in rows:
        ci = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]" if r["ci"] else "fallback"
        mark = "  <- shipped" if r["ceiling"] == PICK_NOISE_FIT_CEILING else ""
        out.append(f"  {r['ceiling']:>7} {r['applied']:>7.0f} {r['n']:>5} {r['a']:>6.2f} "
                   f"{r['b']:>6.3f} {ci:>16} {r['sigma_100']:>10.1f}{mark}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="scripts/fit_pick_noise.py",
        description="Fit sigma(pick) = a + b * pick on this league's drafts and sweep the ceiling.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS),
                    help="draft seasons to fit on (default: the four the draft gate replays)")
    ap.add_argument("--ceilings", default=",".join(str(c) for c in PICK_NOISE_SWEEP_CEILINGS),
                    help="fit ceilings to sweep; each is min(pool, ceiling) when applied")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    holdout.add_arguments(ap)
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    if a.exclude_season is not None:
        seasons = [s for s in seasons if s != a.exclude_season]
    ceilings = [int(c) for c in a.ceilings.split(",") if c.strip()]
    note = holdout.recording(a, holdout.command_line(SCRIPT, a))

    from hub.fetch.espn import resolve_league_id

    try:
        df = historical_picks(resolve_league_id(), seasons)
        if df.height == 0:
            raise RuntimeError("no draft matched a rank")
    except Exception as e:
        why = (f"not refitted: this league's draft history was unavailable "
               f"on {datetime.now(UTC).date().isoformat()}; the fit "
               f"needs an ESPN session (availability.historical_picks)")
        for key in KEYS:
            note(key, None, why_not=why)
        return unavailable(SCRIPT, "this league's draft history", e)
    print(f"  {df.height} matched picks over {df['year'].n_unique()} drafts, seasons {seasons}"
          + (f" (season {a.exclude_season} held out)" if a.exclude_season else ""))

    (a_fit, b_fit), said = noise_from_picks(df, draws=a.draws, seed=a.seed)
    print(said)
    print(f"  shipped: sigma = {PICK_NOISE_INTERCEPT:.2f} + {PICK_NOISE_SLOPE:.3f} * pick "
          f"(ceiling {PICK_NOISE_FIT_CEILING}); this run: {a_fit:.2f} + {b_fit:.3f}")
    note("availability.PICK_NOISE_INTERCEPT", round(a_fit, 2))
    note("availability.PICK_NOISE_SLOPE", round(b_fit, 3))

    print("\n  ceiling sweep (#287): the cut published as a sensitivity, not asserted")
    print("\n".join(sweep_lines(sweep_ceilings(df, ceilings, draws=a.draws, seed=a.seed))))
    print("  a ceiling past the pool collapses to the pool; the interval on each row is over "
          "the same drafts as every other row, so rows cannot be read as resolvably different")
    return 0


if __name__ == "__main__":
    sys.exit(main())
