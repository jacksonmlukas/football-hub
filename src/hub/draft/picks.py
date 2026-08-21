"""Snake draft pick numbers from your slot."""
from __future__ import annotations


def snake_picks(slot: int, teams: int = 12, rounds: int = 16) -> list[int]:
    """Overall pick numbers for a given slot in a snake draft.

    Odd rounds run 1..teams, even rounds reverse. Slot 1 in a 12-team draft picks
    1, 24, 25, 48, 49 ... which is why the turn structure matters so much more at the
    ends than in the middle: you wait 22 picks between turns, then get two in a row.
    """
    if not 1 <= slot <= teams:
        raise ValueError(f"slot {slot} outside 1..{teams}")
    out = []
    for r in range(1, rounds + 1):
        pos = slot if r % 2 else teams - slot + 1
        out.append((r - 1) * teams + pos)
    return out


def next_two(picks: list[int], drafted: int) -> tuple[int, int]:
    """Your current pick and the one after, given how many picks have been made."""
    upcoming = [p for p in picks if p > drafted]
    if not upcoming:
        raise ValueError("draft is over")
    return upcoming[0], (upcoming[1] if len(upcoming) > 1 else upcoming[0] + 1)


# Your league. Slot 3 of 12 produces a strict 19-5-19-5 alternation:
#   3, 22, 27, 46, 51, 70, 75, 94, 99, ...
# Odd rounds are followed by a 19-pick wait; even rounds by a 5-pick wait. That single fact
# should drive most in-draft decisions -- see draft_mode().
MY_SLOT = 3
TEAMS = 12


def my_picks(rounds: int = 16) -> list[int]:
    return snake_picks(MY_SLOT, TEAMS, rounds)


def draft_mode(current_pick: int, rounds: int = 16) -> str:
    """Which decision rule applies at this pick.

    'scarcity' -- a 19-pick wait follows, so take the player who will not survive it.
                  cost_of_waiting dominates; passing on a tier break is expensive.
    'value'    -- your next turn is 5 picks away, so almost anyone you like will still be
                  there. Take the highest VOR and ignore availability.

    The whole point of drafting from slot 3 is that these two rules alternate and most
    people apply one of them all draft long.
    """
    picks = my_picks(rounds)
    if current_pick not in picks:
        raise ValueError(f"{current_pick} is not one of your picks: {picks}")
    i = picks.index(current_pick)
    if i == len(picks) - 1:
        return "scarcity"
    return "scarcity" if picks[i + 1] - current_pick > 10 else "value"
