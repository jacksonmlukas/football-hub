"""Fit TALENT_CV and TALENT_CV_BY_POS on this league's drafts, with a season held out (#294).

`python -m hub.draft.calibrate` is the fit (docs/talent-cv.md); this is the same fit behind
the two flags every fitting script takes. It needs the league's ESPN draft history
(`calibrate.draft_outcomes` reads `espn.league_history`, which needs the private-league
cookies); without a session it exits with the sentence `hub.cli.unavailable` prints and,
under `--record`, writes that the key could not be refitted and why, so a hold-out replay
reads the shipped value and says so on its run line rather than silently.

    uv run python scripts/fit_talent_cv.py
    uv run python scripts/fit_talent_cv.py --exclude-season 2024 --record

The headline seasons are 2023-25 (`calibrate.main` excludes 2022 by default: a quarter of
its drafted players have retired out of ESPN's universe, and they are the busts). Holding
2022 out therefore changes nothing about the fit and is recorded as such.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from hub import holdout
from hub.cli import unavailable
from hub.config import DRAFTED_POSITIONS
from hub.draft.calibrate import DRAFTED_THROUGH, draft_outcomes, fit_talent_cv
from hub.models.predict import TALENT_CV, TALENT_CV_BY_POS

DEFAULT_SEASONS = (2023, 2024, 2025)
SCRIPT = "scripts/fit_talent_cv.py"
KEYS = ("predict.TALENT_CV", "predict.TALENT_CV_BY_POS")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog=SCRIPT, description="Fit TALENT_CV from past drafts.")
    ap.add_argument("--seasons", default=",".join(str(s) for s in DEFAULT_SEASONS))
    holdout.add_arguments(ap)
    a = ap.parse_args(argv)
    seasons = [int(s) for s in a.seasons.split(",") if s.strip()]
    if a.exclude_season is not None:
        seasons = [s for s in seasons if s != a.exclude_season]
    note = holdout.recording(a, holdout.command_line(SCRIPT, a))

    try:
        df = draft_outcomes(seasons)
    except Exception as e:
        why = (f"not refitted: this league's past drafts were unavailable "
               f"on {datetime.now(UTC).date().isoformat()}; the fit "
               f"needs an ESPN session (calibrate.draft_outcomes)")
        for key in KEYS:
            note(key, None, why_not=why)
        return unavailable(SCRIPT, "your league's past drafts", e)
    got = fit_talent_cv(df, debias=True)
    print(f"  fitted on {got['n']} drafted player-seasons over {got['clusters']} players, "
          f"seasons {got['seasons']}, picks 1-{DRAFTED_THROUGH}"
          + (f" (season {a.exclude_season} held out)" if a.exclude_season else ""))
    print(f"    nominal {got['nominal']:.3f} (shipped {TALENT_CV}); dispersion "
          f"{got['talent_cv']:.3f} 95% CI [{got['ci95'][0]:.3f}, {got['ci95'][1]:.3f}]")
    for pos in DRAFTED_POSITIONS:
        print(f"    {pos:>3} nominal {got['nominal_by_position'][pos]:.3f} "
              f"(shipped {TALENT_CV_BY_POS.get(pos, TALENT_CV):.2f})")
    note("predict.TALENT_CV", round(got["nominal"], 2))
    note("predict.TALENT_CV_BY_POS",
         {p: round(got["nominal_by_position"][p], 2) for p in DRAFTED_POSITIONS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
