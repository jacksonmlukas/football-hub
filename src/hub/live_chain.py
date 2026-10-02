"""Whether a `live` loop that has reached its cap should hand the window to another one (#391).

`live.yml` runs a fixed two-hour loop and stops, wherever in a window it started. A loop that
started late -- #390's began at 02:34Z for a 00:20Z kickoff -- is therefore cut off by its own
cap while the game it was started for is still being played, and the next delivery of the
cron that would replace it is exactly the thing GitHub drops under load.

So at its cap the loop asks one question, and this is the answer: **does ESPN report a game in
progress right now?** If it does, the window is not over and the loop starts another. If it
does not, the loop ends, and it ends for the reason a window ends rather than because a
clock said so.

**Game state, never window shape.** `live.yml`'s header is explicit that the crons are the
only definition of a window and that the loop knows nothing about it: a second copy in code is
the drift this repo keeps paying for (`hub.fetch.odds._game_date`, `hub.schedule._kickoff`).
This module has no hour, no weekday and no kickoff time in it. The loop was already asking ESPN
every five minutes; this is that same question put once more at the end.

**What it cannot do, stated here because the ticket's replay is where it shows.** It extends a
loop that is running. It cannot start one: on the Monday of 2026-09-28/29 no loop was running
when the window opened (the previous one ended at ~20:18Z with the game still `pre`), so there
was nothing at a cap to extend. That half is `hub.watchdog_gap`'s, which reports the window
afterwards.

**An ESPN that cannot be reached is `UNKNOWN`, and the loop treats it like `QUIET`.** The
alternative is a chain of two-hour loops kept alive by an outage, with nothing to end it. The
cost is that a transient failure on the very last request ends coverage a cycle early, which
the crons are still there to repair.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# The loop's shell step branches on 0 alone; the other two are told apart for the log.
CONTINUE = 0
QUIET = 1
UNKNOWN = 2

# ESPN's own word for a game being played, as `hub.fetch.espn._overlay_row` reports it
# (`pre`, `in`, `post`).
IN_PROGRESS = "in"


def games_in_progress(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows of `live_state` whose game is being played, in the order given."""
    return [r for r in rows if r.get("state") == IN_PROGRESS]


def decide(rows: Iterable[Mapping[str, Any]]) -> int:
    """`CONTINUE` when any game is in progress, `QUIET` otherwise."""
    return CONTINUE if games_in_progress(rows) else QUIET


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.live_chain",
        description="Exit 0 when ESPN reports a game in progress (the loop should continue), "
                    "1 when none is, 2 when ESPN could not be asked.")
    ap.add_argument("--league", default="nfl", choices=("nfl", "cfb"))
    args = ap.parse_args(argv)

    # Imported here for the reason `hub.publish.live` imports it inside the function: the
    # fetch layer is not something `--help` should reach.
    from hub.fetch.espn import live_state
    try:
        rows = live_state(args.league)
    except Exception as e:
        print(f"unknown: ESPN could not be asked ({type(e).__name__}); not continuing")
        return UNKNOWN
    live = games_in_progress(rows)
    if live:
        print(f"continue: {len(live)} {args.league} game(s) in progress")
        return CONTINUE
    print(f"quiet: no {args.league} game in progress ({len(rows)} on the board)")
    return QUIET


if __name__ == "__main__":
    sys.exit(main())
