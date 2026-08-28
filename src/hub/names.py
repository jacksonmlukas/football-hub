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
"""
from __future__ import annotations

import re
import unicodedata

# Generational suffixes carry no identity: no league contains both a Marvin Harrison and a
# Marvin Harrison Jr. Dropping them is safe and fixes the most common mismatch.
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def player_key(name: str) -> str:
    """Collapse a display name to a comparable key.

    Lower-cased, accents folded, punctuation and generational suffixes removed, whitespace
    collapsed. `Ja'Marr Chase` and `JaMarr Chase` land on the same key; `Justin Jefferson`
    and `Justin Herbert` do not.
    """
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)          # punctuation: Ja'Marr -> jamarr, D.J. -> dj
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())
