"""Measure IMPUTE_CV on rookies -- the players the board actually imputes (#277).

The measurement is `hub.draft.impute_cv`; this is its committed entry point, the one the
constant's docstring names in place of the scratchpad file the shipped value came from.

    uv run python scripts/fit_impute_cv.py
    uv run python scripts/fit_impute_cv.py --exclude-season 2024

Reads the archive through the repo's own readers (`board_as_of`, `expected_points`,
`nflverse.load`); a cache miss fetches, so run it where the caches are. Writes no constant.
"""
from __future__ import annotations

import sys

from hub.draft.impute_cv import main

if __name__ == "__main__":
    sys.exit(main())
