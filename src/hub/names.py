"""One comparable key for a player, across every source that names him differently.

FantasyPros, nflverse and ESPN disagree about punctuation and generational suffixes --
`A.J. Brown` against `AJ Brown`, `Marvin Harrison Jr.` against `Marvin Harrison` -- and an
exact join drops those players silently. `docs/decisions.md` records that as a real data-loss
bug and names this function as its fix.

It lived as `hub.draft.state.player_key`: a **private** helper in the module that tracks which
players have been taken, imported by ten modules across three packages, two of them reaching
for it inside a function body -- which is what a caller does when an import feels wrong. It is
not draft state. It is how this repo decides that two spellings are one player, which is a
domain concept, and it now has a name in `CONTEXT.md` and a module of its own.

A leaf on purpose: it imports nothing from `hub`, so anything may import it without dragging
a board builder along.

`practice_key` joined it on 2026-08-27 for the same reason, one source at a time rather than
ten: `hub.models.injury._PS_WIDTH` was private and `hub.models.weekly_screen` needed the same
normalisation, so the second caller made it a seam. The guard in
`tests/contracts/test_cli_surface.py` caught the private import on the way in.
"""
from __future__ import annotations

import re
import unicodedata

# Generational suffixes carry no identity: no league contains both a Marvin Harrison and a
# Marvin Harrison Jr. Dropping them is safe and fixes the most common mismatch.
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


# Whole-key aliases: the consensus page's spelling on the left, the stats source's on the
# right, both already in key form. By whole name and never by first name -- `Mike` is not
# `Michael` for everyone, and `Irv Smith` / `Ito Smith`, `Robbie Anderson` / `Ryan Anderson`
# are pairs a first-initial rule would merge. Each entry is a player found by #246's sweep
# of every drafted-position consensus name 2021-2025 against its season's realised rows:
# absent exactly, matching under `relaxed_key`, and the same person. Every published
# draft-gate figure scored these players' seasons as zero. Add a pair here only with the
# season and rank it was found at.
ALIASES: dict[str, str] = {
    "gabriel davis": "gabe davis",            # 2021 (ecr 191), 2022 (75)
    "ken walker": "kenneth walker",           # 2022 (102); `III` is stripped as a suffix
    "kenneth gainwell": "kenny gainwell",     # 2021 (199), 2022 (122), 2023 (148)
    "joshua palmer": "josh palmer",           # 2022-2025 (125-197)
    "chigoziem okonkwo": "chig okonkwo",      # 2023 (140)
    "cameron ward": "cam ward",               # 2025 (172)
}


def player_key(name: str) -> str:
    """Collapse a display name to a comparable key.

    Lower-cased, accents folded, punctuation and generational suffixes removed, whitespace
    collapsed, then `ALIASES` applied to the whole key. `Ja'Marr Chase` and `JaMarr Chase`
    land on the same key; `Justin Jefferson` and `Justin Herbert` do not; `Gabriel Davis`
    lands on `gabe davis` because one source calls him that and the other does not (#246).
    """
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)          # punctuation: Ja'Marr -> jamarr, D.J. -> dj
    s = _SUFFIX.sub(" ", s)
    key = " ".join(s.split())
    return ALIASES.get(key, key)


def relaxed_key(name: str) -> str:
    """The first initial and the surname of a `player_key` -- `j jefferson` for both
    `Justin Jefferson` and `J. Jefferson`.

    Looser than `player_key` on purpose, and used for exactly one thing: telling a name that
    *failed to join* from a player who never played. The draft gate scores a drafted player
    with no realised row as zero (`backtest.score_roster`), which is right for a player who
    was hurt, cut or never played and wrong for one whose name a source spelled differently
    -- and a differential rate of the second between two arms is a bias in the headline
    number (issue #46). A drafted name absent from the realised set that matches a realised
    name here is counted as a join failure; one matching nothing is never-played. Nothing
    joins on this key. It cannot be used to *repair* a join without inventing players, and
    `state.suggest_unmatched` already answers the "did you mean" question at the console.

    A one-word name relaxes to itself: there is no initial to take, so the comparison is the
    exact one, which is the honest reading of a name that cannot be matched more loosely.
    """
    parts = player_key(name).split()
    if len(parts) < 2:
        return " ".join(parts)
    return f"{parts[0][0]} {parts[-1]}"


# Practice status arrives as the club's own prose -- "Did Not Participate In Practice",
# "Limited Participation In Practice", "Full Participation In Practice" -- and the first seven
# characters separate the three cases with nothing else colliding.
PRACTICE_WIDTH = 7


def practice_key(col: str = "practice_status", *, missing: str = "None"):
    """The practice designation, normalised, as a polars expression.

    An expression rather than a function over strings, so a whole frame is normalised in one
    vectorised pass -- `player_key` needs `map_elements` because a name has to be parsed, and a
    practice status only has to be truncated.
    """
    import polars as pl
    return pl.col(col).fill_null(missing).str.slice(0, PRACTICE_WIDTH)
