"""The draft-day poller.

`draft-day-ops/SKILL.md` sets the constraint: roughly 90 seconds per pick, so every answer
has to be one command away. That makes latency a correctness property rather than a
nicety -- a board that is right but arrives after the clock expires is wrong. The refresh
is budgeted and the output is bounded, and both are tested.

The thing this does that a static board cannot is recompute **replacement level from what
is actually left**. VOR against a preseason baseline overvalues a position immediately
after a run on it, which is precisely the moment someone consults the number. Take thirty
running backs off the board and RB replacement falls; the remaining backs are worth less
than the morning's board says, and the poller says so.

If this dies mid-draft, `site/data/draft_board.json` is still correct. That is the
documented fallback and the reason nothing here writes to the store.

    uv run python -m hub.draft.live --poll 10
    uv run python -m hub.draft.live --replay 2025
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from typing import Any, Sequence

import polars as pl

from hub.draft.board import DRAFTED_POSITIONS, MIN_GAMES, SLOTS, replacement_levels
from hub.draft.picks import MY_SLOT, TEAMS, draft_mode, my_picks, next_two
from hub.draft.state import DraftState, remaining, take

# A run is three or more picks at one position inside the last five -- the skill's
# definition, not an invention.
RUN_WINDOW = 5
RUN_THRESHOLD = 3

# You have ~90 seconds. Two is the plan's ceiling and generous against that.
REFRESH_BUDGET_S = 2.0

# Short enough to read while on the clock.
MAX_LINES = 24
TOP_N = 8


def detect_run(recent: Sequence[str]) -> str | None:
    """The position being run, if any, over the last RUN_WINDOW picks."""
    window = [p for p in list(recent)[-RUN_WINDOW:] if p]
    if not window:
        return None
    pos, count = Counter(window).most_common(1)[0]
    return pos if count >= RUN_THRESHOLD else None


def unfilled(counts: dict[str, int]) -> list[str]:
    """Starting slots you cannot fill from what you hold.

    `season.py` established that a surplus player scores exactly zero -- you can only
    start three WRs -- so a board that offers a fourth while a starting slot sits empty is
    recommending a player worth nothing. This is what the 2025 dry run walked into: pick
    51, holding RB3 WR2, no QB and no TE, and the board offered another WR.
    """
    return [p for p in DRAFTED_POSITIONS if counts.get(p, 0) < SLOTS.get(p, 0)]


def rank(view: dict[str, Any], key: str) -> pl.DataFrame:
    """Candidates in order, with below-replacement players removed.

    A negative VOR means the player is worse than what will be sitting on waivers all
    season. No edge redeems that, and the dry run put Jordan Love at VOR -1.0 on top of a
    round-4 board purely because his edge was large.
    """
    return (view["available"]
            .filter(pl.col(key).is_not_null() & (pl.col("vor_live") > 0))
            .sort(key, descending=True))


def _live_replacement(available: pl.DataFrame, teams: int) -> dict[str, float]:
    """Replacement level over the players still on the board.

    `replacement_levels` speaks the ffopportunity column names, so the board is renamed
    into that shape rather than duplicating the logic -- the flex allocation and the
    minimum-games rule both live there and should have one home.
    """
    shaped = available.select(
        pl.col("pos").alias("position"),
        pl.col("xfp_per_game"),
        pl.col("games"),
    )
    return replacement_levels(shaped, teams=teams, min_games=MIN_GAMES)


def refresh(board: pl.DataFrame, state: DraftState, *, my_slot: int = MY_SLOT,
            teams: int = TEAMS, rounds: int = 16) -> dict[str, Any]:
    """One recomputation of everything that changes when a pick is made."""
    available = remaining(board, state)

    taken_pos: list[str] = []
    if state.taken:
        by_name = dict(zip(board["player"].to_list(), board["pos"].to_list()))
        taken_pos = [by_name[n] for n in state.taken if n in by_name]

    levels = _live_replacement(available, teams)
    available = available.with_columns(
        (pl.col("xfp_per_game") - pl.col("pos").replace_strict(levels, default=0.0))
        .alias("vor_live"))

    from hub.draft.state import my_roster, unmatched as unmatched_picks

    held = my_roster(state, slot=my_slot, teams=teams, rounds=rounds)
    by_name = dict(zip(board["player"].to_list(), board["pos"].to_list()))
    roster = Counter(by_name[n] for n in held if n in by_name)

    picks = my_picks(rounds)
    upcoming = [p for p in picks if p > state.n_taken]
    if upcoming:
        now, nxt = next_two(picks, state.n_taken)
        mode = draft_mode(now, rounds)
    else:
        now, nxt, mode = None, None, None

    return {
        "available": available,
        "replacement": levels,
        "run": detect_run(taken_pos),
        "next_pick": now,
        "pick_after": nxt,
        "mode": mode,
        "taken": state.n_taken,
        "teams": teams,
        "roster": dict(roster),
        "unmatched": unmatched_picks(board, state),
    }


# Rounds 1-3 rank on value over replacement, not edge. draft-day-ops/SKILL.md is explicit:
# consensus is tight at the top, so a large edge there usually means a name-matching
# failure rather than an opportunity. Sorting by edge at pick 3 surfaced Godwin, Downs and
# Jordan Love -- all with NEGATIVE VOR -- as the top three names on a first-round board.
EDGE_FROM_ROUND = 4


def _sort_key(pick: int, teams: int) -> tuple[str, str]:
    rnd = (pick - 1) // teams + 1
    if rnd < EDGE_FROM_ROUND:
        return "vor_live", f"round {rnd}: edge is unreliable this early"
    return "edge", f"round {rnd}: the room lets these fall"


def render(view: dict[str, Any]) -> list[str]:
    """Bounded, and the warning comes first.

    Ordering is the whole design here: on the clock you read the top of the screen. A run
    changes which position to avoid, so it cannot be below a table.
    """
    out: list[str] = []
    if view["unmatched"]:
        names = ", ".join(view["unmatched"][:4])
        out.append(f"  {len(view['unmatched'])} UNMATCHED pick(s): {names}"
                   f" -- these players are still on the board below. Check the spelling"
                   f" before trusting a recommendation.")
    if view["run"]:
        out.append(f"  RUN ON {view['run']} -- value there has already collapsed. "
                   f"Take the best player at a position nobody just drafted.")

    if view["next_pick"] is None:
        out.append(f"  draft complete: {view['taken']} picks")
        return out

    rule = ("take who will not survive the wait" if view["mode"] == "scarcity"
            else "take the highest VOR; almost anyone survives")
    out.append(f"  pick {view['next_pick']} ({view['mode'].upper()}) -> next at "
               f"{view['pick_after']} | {view['taken']} gone | {rule}")
    out.append("  replacement: " + ", ".join(
        f"{p}={view['replacement'].get(p, 0.0):.1f}" for p in DRAFTED_POSITIONS))

    held = view["roster"]
    need = unfilled(held)
    shape = " ".join(f"{p}{held[p]}" for p in DRAFTED_POSITIONS if held.get(p))
    out.append(f"  you hold: {shape or '(nothing yet)'}"
               + (f" | need {', '.join(need)}" if need else " | starters complete"))

    key, why = _sort_key(view["next_pick"], view.get("teams", TEAMS))
    top = rank(view, key).head(TOP_N)
    out.append(f"  top {TOP_N} by {key} ({why}):")
    for r in top.iter_rows(named=True):
        edge = r.get("edge")
        fills = "*" if r["pos"] in need else " "
        out.append(f"  {fills} {str(r['player'])[:22]:<22} {str(r['pos'] or ''):<3} "
                   f"vor {r['vor_live']:>5.1f}  "
                   f"edge {'-' if edge is None else format(edge, '>6.1f')}  "
                   f"sd {r.get('sd') or 0:>4.1f}")
    if need:
        out.append("  * fills an empty starting slot")
    return out[:MAX_LINES]


def replay(board: pl.DataFrame, picks: Sequence[str], *, my_slot: int = MY_SLOT,
           teams: int = TEAMS, quiet: bool = False,
           collect_runs: bool = False) -> list[str]:
    """Walk a completed draft one pick at a time.

    The plan asks for this to reproduce the pick sequence, which is a real check that the
    state machine tracks reality. It is also the only way to rehearse before the day, and
    the only way to time a refresh against a full board without a live draft.
    """
    state = DraftState()
    seen: list[str] = []
    runs: list[str] = []
    slowest = 0.0
    for name in picks:
        state = take(state, name)
        seen.append(name)
        start = time.perf_counter()
        view = refresh(board, state, my_slot=my_slot, teams=teams)
        slowest = max(slowest, time.perf_counter() - start)
        if view["run"]:
            runs.append(view["run"])
    if not quiet:
        print(f"  replayed {len(seen)} picks | slowest refresh {slowest * 1000:.0f}ms "
              f"(budget {REFRESH_BUDGET_S * 1000:.0f}ms)")
        if runs:
            counts = Counter(runs)
            print("  runs seen: " + ", ".join(f"{p} x{n}" for p, n in counts.most_common()))
    return runs if collect_runs else seen


def _load_board() -> pl.DataFrame:
    import polars as _pl

    from hub.draft.board import ROOT
    path = ROOT / "data" / "processed" / "draft_board.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no board at {path}. Run `make draft` first -- the poller reads the board, "
            f"it does not build one.")
    return _pl.read_parquet(path)


def poll(interval: int = 10, *, my_slot: int = MY_SLOT, teams: int = TEAMS,
         heartbeat_s: int = 120) -> int:
    """Re-read the ESPN draft every `interval` seconds and reprint on change.

    Three things here are draft-night ergonomics rather than logic, and all three came
    from running this against the live league rather than from a test:

    * every print flushes. Python block-buffers a pipe, so `--poll 10 | tee draft.log`
      produced twenty-two seconds of nothing.
    * the sync is quiet. It used to announce "draft not started" on every pass, which over
      three hours is a thousand lines scrolling the board out of view.
    * a heartbeat every couple of minutes, because a silent poller and a hung poller look
      identical at 2am and you have ninety seconds to decide.
    """
    import signal

    from hub.draft import state as state_mod

    board = _load_board()
    last, last_beat = -1, 0.0
    stopping = False

    def _stop(*_: Any) -> None:
        nonlocal stopping
        stopping = True

    # SIGTERM as well as Ctrl-C: a poller killed by a window closing should still say
    # where the fallback board is.
    signal.signal(signal.SIGTERM, _stop)

    print(f"  polling ESPN every {interval}s. Ctrl-C to stop.", flush=True)
    try:
        while not stopping:
            st = state_mod.sync_from_espn(quiet=True)
            now = time.monotonic()
            if st.n_taken != last:
                last, last_beat = st.n_taken, now
                print("\n" + "\n".join(render(refresh(board, st, my_slot=my_slot,
                                                      teams=teams))), flush=True)
            elif now - last_beat >= heartbeat_s:
                last_beat = now
                print(f"  watching... {st.n_taken} picks, no change", flush=True)
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    print("\n  stopped. site/data/draft_board.json is still correct.", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.draft.live",
        description="Live draft board: removes drafted players, reprices, flags runs.")
    ap.add_argument("--poll", type=int, nargs="?", const=10, default=None,
                    metavar="SECONDS", help="poll the live ESPN draft (default 10s)")
    ap.add_argument("--replay", type=int, default=None, metavar="YEAR",
                    help="dry run against a completed draft")
    ap.add_argument("--slot", type=int, default=MY_SLOT)
    a = ap.parse_args(argv)

    if a.replay is not None:
        from hub.draft import state as state_mod
        board = _load_board()
        st = state_mod.sync_from_espn(a.replay)
        if not st.taken:
            print(f"hub.draft.live: no {a.replay} draft to replay", file=sys.stderr)
            return 1
        print(f"  replaying the {a.replay} draft against today's board")
        replay(board, st.taken, my_slot=a.slot)
        return 0

    if a.poll is None:
        ap.print_help()
        return 0
    return poll(a.poll, my_slot=a.slot)


if __name__ == "__main__":
    sys.exit(main())
