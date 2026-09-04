"""Where things live on disk, as a leaf.

`ROOT` is declared eight times across the repo -- `store`, `publish`, `board`, `tune`, `state`,
`nflverse`, `cfbd`, `odds` -- at two different `parents[]` depths. All eight resolve to the same
path and each depth is correct for its own nesting, so that is tidiness rather than a bug, and
`docs/improvements.md` #17 says tidiness alone does not earn a change to eight files.

**The part that is not tidiness** is that `hub.draft.adp_history` imported `ROOT` from
`hub.draft.board` -- a 780-line module pulling in nflreadpy, numpy, polars, contracts,
durability, regression, report, state, availability, picks and playoff_sos -- purely to learn a
`Path`. That is a cycle, and it costs `board.main` a function-local import with a comment saying
why.

A leaf on purpose: it imports nothing but `pathlib`, so anything may take a filename from it
without dragging a board builder along. Same reason `hub.names` exists.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
PROCESSED = DATA / "processed"
BOARD_PARQUET = PROCESSED / "draft_board.parquet"
# Beside the board for the same reason. `hub.season.roster` writes it and `hub.season.lineup`
# reads it, and the reader had the path as a bare string literal in an argparse default -- so
# moving the file would have left the reader looking at the old location in silence.
ROSTER_PARQUET = PROCESSED / "roster.parquet"
