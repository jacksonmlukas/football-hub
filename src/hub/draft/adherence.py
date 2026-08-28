"""Did you take THE PICK, and where did you not?

Objective 1 is judged on decisions rather than the result, because the result is one season of
n=1 and cannot separate a good system from a lucky one. The threshold, fixed before the draft
in `docs/decisions.md`: **THE PICK at 12 of 16 turns, every deviation written down at the
time.**

This scores the reconstructible half. Which players you took follows from your slot and the
snake order; what THE PICK *would have said* at each turn follows from the board plus the picks
that had been made by then. Neither needs to be recorded live.

**Why you deviated does not follow from anything**, which is why the runbook has you write it on
paper. This module counts and locates deviations; it cannot explain them, and reporting a
deviation count without the reasons beside it would be the outcome-versus-decision confusion
that objective 1 is worded to avoid.

Post-hoc analysis. It touches nothing used on draft night, which is why building it during the
code freeze is safe.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from hub.draft.board import BOARD_PARQUET, board_age_hours
from hub.draft.optimize import the_pick
from hub.draft.picks import MY_SLOT, TEAMS, my_picks
from hub.draft.state import DraftState
from hub.draft.state import load as load_state
from hub.names import player_key

# docs/decisions.md, fixed 2026-08-27 before the draft rather than chosen in January.
THRESHOLD = 12
ROUNDS = 16

# The board as it stood when the draft opened. Every `--pick` rebuilds from source, so the live
# file drifts with ADP through the evening; the runbook copies this one before the first pick.
AS_DRAFTED = BOARD_PARQUET.with_name("draft_board.AS-DRAFTED.parquet")


def replay(board: pl.DataFrame, state: DraftState, *, my_slot: int = MY_SLOT,
           teams: int = TEAMS, rounds: int = ROUNDS) -> pl.DataFrame:
    """One row per turn of yours: what THE PICK said, what you took, whether they agree.

    THE PICK at your turn `n` is computed against the state *before* that pick — `taken[:n-1]`
    — which is the same thing you were looking at. Recomputing it against the finished draft
    would ask a different question and flatter every answer.
    """
    rows = []
    for turn, overall in enumerate(my_picks(rounds), start=1):
        if overall > state.n_taken:
            break                                   # draft did not reach this turn
        before = DraftState(taken=list(state.taken[:overall - 1]))
        tp = the_pick(board, before, my_slot=my_slot, teams=teams)
        took = state.taken[overall - 1]
        rows.append({
            "round": turn,
            "overall": overall,
            "the_pick": tp.player if tp else None,
            "via": tp.via if tp else None,
            "took": took,
            # Loose match, for the same reason the board matches loosely: a suffix or an
            # apostrophe should not read as a deviation.
            "followed": bool(tp and player_key(tp.player) == player_key(took)),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"round": pl.Int64, "overall": pl.Int64, "the_pick": pl.Utf8,
                "via": pl.Utf8, "took": pl.Utf8, "followed": pl.Boolean})


def verdict(scored: pl.DataFrame, threshold: int = THRESHOLD) -> tuple[bool, str]:
    """Against the pre-registered threshold, and honest about what it cannot see."""
    if scored.is_empty():
        return False, "no turns to score -- was the draft state loaded?"
    n = int(scored["followed"].sum())
    total = scored.height
    ok = n >= threshold
    head = (f"{'MET' if ok else 'MISSED'}: took THE PICK at {n} of {total} turns "
            f"(threshold {threshold} of {ROUNDS})")
    if total < ROUNDS:
        head += f" -- only {total} turns in the state, so this is partial"
    miss = scored.filter(~pl.col("followed"))
    if miss.is_empty():
        return ok, head
    lines = [f"    round {r['round']:>2} (pick {r['overall']:>3}): "
             f"took {r['took']}, THE PICK said {r['the_pick']}"
             for r in miss.iter_rows(named=True)]
    return ok, (head + f"\n  {miss.height} deviation(s) -- pair each with the note you wrote "
                f"at the time:\n" + "\n".join(lines))


# A copy older than this is almost certainly left over from a rehearsal rather than made
# before the first pick. A day is generous: the runbook says to rebuild on the morning of the
# draft, so the honest copy is hours old, never days.
STALE_HOURS = 24.0


def age_note(path: Path, now: float | None = None) -> list[str]:
    """How old the as-drafted copy is, and a warning if it cannot be this draft's.

    Found by rehearsing the runbook on 2026-08-27: a *missing* copy is reported loudly, but a
    *stale* one is not, and the stale case is the silent-plausible-wrong one -- a copy left
    behind by a rehearsal grades you against a board you never saw, with the ADP drift the
    copy exists to eliminate, and says nothing.
    """
    import time
    age = board_age_hours(path, time.time() if now is None else now)
    line = f"  as-drafted copy: {age:.1f}h old"
    if age <= STALE_HOURS:
        return [line]
    return [line, f"  WARNING: that predates the draft by {age / 24:.1f} days, so it is a "
                  f"rehearsal leftover, not the board you opened with.\n"
                  f"  Re-copy it from the draft-morning build, or pass --board explicitly."]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.adherence",
        description="Score draft adherence against the pre-registered threshold.")
    ap.add_argument("--board", default=None,
                    help=f"board to replay against; defaults to {AS_DRAFTED.name} if it "
                         f"exists, else the live board")
    ap.add_argument("--state", default=None, help="draft state json")
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    a = ap.parse_args(list(argv) if argv is not None else None)

    path = Path(a.board) if a.board else (AS_DRAFTED if AS_DRAFTED.exists() else BOARD_PARQUET)
    if not path.exists():
        print(f"hub.draft.adherence: no board at {path}. Run `make draft` first.")
        return 1
    if path == BOARD_PARQUET:
        print(f"  WARNING: replaying against {path.name}, which every --pick rebuilds. The "
              f"board has drifted since the draft.\n  The runbook copies "
              f"{AS_DRAFTED.name} before the first pick; without it this is approximate.")
    else:
        print(*age_note(path), sep="\n")
    state = load_state(Path(a.state)) if a.state else load_state()
    scored = replay(pl.read_parquet(path), state)
    print(f"\n  {scored.height} of your turns, replayed against {path.name}\n")
    for r in scored.iter_rows(named=True):
        mark = "  " if r["followed"] else "->"
        print(f"  {mark} r{r['round']:>2} pick {r['overall']:>3}  took {str(r['took'])[:24]:<24}"
              f"  THE PICK {str(r['the_pick'])[:24]}")
    print(f"\n  {verdict(scored, a.threshold)[1]}")
    return 0


if __name__ == "__main__":                                      # pragma: no cover
    raise SystemExit(main())
