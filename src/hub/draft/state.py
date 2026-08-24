"""Who is off the board.

The draft board is worthless on draft night if it keeps recommending players who are
already gone, and the thing that silently breaks that is name matching: ESPN says
"Marvin Harrison Jr.", FantasyPros says "Marvin Harrison". So every comparison goes
through _norm(), and anything that fails to match is reported rather than dropped.

State is an ordered list of picks, which is all we need: your roster is derivable from
your slot and the snake order, so there is nothing to keep in sync.
"""
from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from pathlib import Path

import polars as pl

from hub.draft.picks import snake_picks

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "data" / "processed" / "draft_state.json"

# Generational suffixes carry no identity: no league contains both a Marvin Harrison
# and a Marvin Harrison Jr. Dropping them is safe and fixes the most common mismatch.
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")


def _norm(name: str) -> str:
    """Collapse a display name to a comparable key."""
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)          # punctuation: Ja'Marr -> jamarr, D.J. -> dj
    s = _SUFFIX.sub(" ", s)
    return " ".join(s.split())


@dataclass(frozen=True)
class DraftState:
    """Picks in overall order. Index i is overall pick i+1."""
    taken: list[str] = field(default_factory=list)

    @property
    def n_taken(self) -> int:
        return len(self.taken)


def load(path: Path = STATE) -> DraftState:
    """Missing or unreadable state is an empty draft, never an exception.

    Draft night is the worst possible moment to surface a traceback, and an empty
    board is a recoverable state the operator can see and correct in one command.
    """
    try:
        return DraftState(taken=list(json.loads(Path(path).read_text())["taken"]))
    except Exception:  # noqa: BLE001
        return DraftState()


def save(state: DraftState, path: Path = STATE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"taken": state.taken}, indent=2))


def take(state: DraftState, *names: str) -> DraftState:
    return replace(state, taken=[*state.taken, *names])


def undo(state: DraftState, n: int = 1) -> DraftState:
    return replace(state, taken=state.taken[:-n] if n <= state.n_taken else [])


def remaining(board: pl.DataFrame, state: DraftState) -> pl.DataFrame:
    """The board minus everyone already drafted."""
    if not state.taken:
        return board
    gone = {_norm(n) for n in state.taken}
    keep = [_norm(p) not in gone for p in board["player"].to_list()]
    return board.filter(pl.Series(keep))


def unmatched(board: pl.DataFrame, state: DraftState) -> list[str]:
    """Picks that matched nobody on the board.

    Surfaced rather than swallowed: an unmatched pick means a player you think is gone
    is still being recommended, which is exactly the failure this module exists to stop.
    """
    known = {_norm(p) for p in board["player"].to_list()}
    return [n for n in state.taken if _norm(n) not in known]


def suggest_unmatched(board: pl.DataFrame, names) -> dict[str, str | None]:
    """Names that are not on the board, each with the board name it probably meant.

    A mistyped pick is the sharp edge of typing picks by hand: `take` records whatever it
    is given, so the misspelt player stays on the board as available and the next
    recommendation can hand back someone already drafted.

    A close match gets a suggestion. A name matching nothing gets `None` rather than a wild
    guess -- kickers and defences are drafted every round or two and the board excludes them
    by design, so they are always unmatched, and guessing at them would train the reader to
    ignore the warning.
    """
    import difflib

    known = {_norm(p): p for p in board["player"].to_list()}
    out: dict[str, str | None] = {}
    for name in names:
        key = _norm(name)
        if key in known:
            continue
        close = difflib.get_close_matches(key, list(known), n=1, cutoff=0.85)
        out[name] = known[close[0]] if close else None
    return out


def my_roster(state: DraftState, slot: int, teams: int = 12, rounds: int = 16) -> list[str]:
    """Your players, derived from the snake order rather than tracked separately."""
    mine = snake_picks(slot, teams, rounds)
    return [state.taken[p - 1] for p in mine if p <= state.n_taken]


def sync_from_espn(year: int | None = None, quiet: bool = False,
                   league_id: int | None = None, _league_factory=None) -> DraftState:
    """Read the draft straight from ESPN instead of typing picks in.

    Preferred on draft night: it cannot drift from reality and needs no operator. Falls
    back to whatever is on disk if the league or the draft is unreachable, because a
    stale board still beats a traceback while you are on the clock.
    """
    try:
        from hub.fetch.espn import league_settings
        if league_id is not None or year is not None:
            factory = _league_factory or _league
            lg = factory(year or 2026, league_id)
        else:
            lg, _ = league_settings()
        picks = [p.playerName for p in (lg.draft or [])]
        if not picks:
            # `quiet` exists because the poller calls this every few seconds. Printing
            # "not started" on each pass buried the board under ~1000 identical lines
            # over a three-hour draft -- the one thing on screen you actually read.
            if not quiet:
                print("  ESPN draft is empty (not started?); keeping local state.")
            return load()
        return DraftState(taken=picks)
    except Exception as e:  # noqa: BLE001
        if not quiet:
            print(f"  ESPN draft unavailable ({type(e).__name__}); keeping local state.")
        return load()


def _league(year: int, league_id: int | None = None):
    """A league handle. `league_id` overrides the configured one.

    The override exists so a practice draft can be read: ESPN mock-draft rooms are separate
    leagues with their own ids, and a sync locked to ESPN_LEAGUE_ID can only ever test
    everything except the live path.
    """
    import os
    from dotenv import load_dotenv
    from espn_api.football import League
    load_dotenv()
    return League(league_id=int(league_id if league_id is not None
                                else os.environ["ESPN_LEAGUE_ID"]), year=year,
                  espn_s2=os.environ.get("ESPN_S2") or None,
                  swid=os.environ.get("ESPN_SWID") or None)
