"""The study's closing line: one moneyline pull per distinct kickoff time, just before it (#450).

`docs/prereg/llm-forecasters.md`, *Market benchmark* / *Closing line*: a separate job snapshots
each game within 30 minutes of its kickoff. Best-effort -- a game with no closing snapshot
drops out of the secondary comparison only -- and this job never touches the forecast job.

**One call per kickoff time, never per game.** `market.snapshot("closing")` (h2h only, one
credit) returns every game, so a window is *covered* once any attempt falls inside it, and a
slate of 1pm games is one call. Ten a week at most (`closing.CLOSE_CALLS_PER_WEEK`), which is
what `market.WORST_MONTH_CREDITS` budgets.

**Reused from the hub's close capture (#495), not copied.** The window-and-coverage rule
(`closing.due`, with the prereg's 30-minute window), the weekly cap (`closing.calls_in_week`)
and the attempt record (`closing.read_attempts` / `record_attempt`) are pure functions of a
path the caller names, so the study passes paths under its own `forecasts` checkout and the
hub's `state/` is never written. What is *not* reused is `closing.capture` itself: it fetches
through `odds.snapshot` (spreads, totals and h2h, three credits, archived to the hub's
`state/odds/`) and checks the hub's floor -- the study needs `market.snapshot`'s per-book
de-vig, its own job's markets, its own floor and its own balance file (#445 (a)).

**Order, which is the safety argument.** Nothing due -> return, nothing read but the schedule
and the attempt file (a rerun spends nothing). Then the weekly cap, then the study floor on
the *freshest* balance either file holds (the hub's `state/odds.json` is read, never written:
the hub polls more often, so its reading is usually the newer one), then the key -- each
refusing before an attempt is recorded. Then the attempt is recorded and, in the workflow,
committed to the `forecasts` branch **before** the request, so a capture lost afterwards
still marks the window spent. The spend lands in the study's balance file through
`odds._write_state`, the fetcher's own counter, measured off `x-requests-remaining`.

**GitHub's delays (#468).** Scheduled starts arrive 4-8.5h late, so a cron alone cannot land
in a 30-minute window. Every trigger is gated here on the clock *at run time*, so a late
start finds the kickoff behind it and spends nothing -- a missed window is never retried
after kickoff (`closing.due` is strictly ahead). The trigger that actually lands is the game
window loop (`live.yml`), awake through every window and asking every five minutes through
`.github/scripts/study-close-dispatch.sh`; the workflow's own 15-minute crons are the backup
the issue specified, harmless when late.

    uv run python -m hub.forecasters.close_line due --out DIR       # exit 0 due, 1 not, 2 unknown
    uv run python -m hub.forecasters.close_line capture --out DIR [--commit-attempt]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import atomic, closing
from hub.declare import not_an_input
from hub.fetch import odds
from hub.forecasters import days, market, pipeline, sources
from hub.forecasters.inputs import GameInfo
from hub.paths import ROOT

# The prereg's "within 30 minutes of its kickoff".
WINDOW_MIN = not_an_input(
    30, "decides when the study's closing snapshot is taken; it changes no forecast or model")

# Under the `forecasts` checkout, beside the study's balance file (`pipeline.ODDS_STATE`).
# `.json`, never `.jsonl`: the runner reads every `*.jsonl` there as a forecast record.
CLOSING_DIR = Path("closing")
ATTEMPTS = CLOSING_DIR / "attempts.json"
STUDY_BRANCH = "forecasts"

_EMPTY = pl.DataFrame()


@dataclass(frozen=True)
class Capture:
    """What one trigger did: `status` is "nothing_due" or "captured"; `calls` is 0 or 1."""
    status: str
    calls: int = 0
    due: tuple[str, ...] = ()
    snapshot_id: str | None = None
    closed: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    path: Path | None = None


def _naive(t: datetime) -> datetime:
    """UTC without a zone: the convention of `hub.closing`'s frames and attempt record."""
    return t.astimezone(UTC).replace(tzinfo=None) if t.tzinfo else t


def _frame(kickoffs: Mapping[str, GameInfo]) -> pl.DataFrame:
    return pl.DataFrame({"game_id": list(kickoffs),
                         "kickoff": [_naive(g.kickoff) for g in kickoffs.values()]},
                        schema={"game_id": pl.Utf8, "kickoff": pl.Datetime})


def pending(kickoffs: Mapping[str, GameInfo], now: datetime,
            attempts: Sequence[datetime]) -> list[str]:
    """Games kicking off in the next `WINDOW_MIN` minutes whose window no attempt covers."""
    got = closing.due(_EMPTY, _frame(kickoffs), _naive(now), attempts, window_min=WINDOW_MIN)
    return got["game_id"].to_list()


def _reading(path: Path, when: datetime) -> tuple[datetime, int] | None:
    have = odds.known_balance(path, when, floor=market.STUDY_FLOOR)
    if have is None:
        return None
    try:
        seen = _naive(datetime.fromisoformat(str(odds._read_state(path)["checked_at"])))
    except (KeyError, ValueError, TypeError):
        seen = datetime.min.replace(tzinfo=UTC).replace(tzinfo=None)   # undated: oldest
    return seen, have


def balance(paths: Sequence[Path], when: datetime) -> int | None:
    """The newest known balance across the files (None if none knows one). One key, one
    account: whichever job read `x-requests-remaining` last has the truer number."""
    readings = [r for p in paths if (r := _reading(p, when)) is not None]
    return max(readings)[1] if readings else None


def capture(out: Path, *, now: datetime, kickoffs: Mapping[str, GameInfo],
            names: Callable[[Mapping[str, GameInfo]], Mapping[str, str]],
            get: market.Get | None = None, key: str | None = None,
            hub_state: Path | None = None,
            publish_attempt: Callable[[], None] | None = None) -> Capture:
    """One gated closing snapshot into `out` (the `forecasts` checkout). 0 or 1 calls."""
    when = _naive(now)
    attempts_path = out / ATTEMPTS
    tried = closing.read_attempts(attempts_path)
    due = pending(kickoffs, now, tried)
    if not due:
        return Capture("nothing_due")
    used = closing.calls_in_week(_EMPTY, _frame(kickoffs), when, tried)
    if used >= closing.CLOSE_CALLS_PER_WEEK:
        raise closing.CloseBudgetExceeded(
            f"{used} study closing calls in the last 7 days, cap is "
            f"{closing.CLOSE_CALLS_PER_WEEK}. Refusing before a request is formed.")
    study_state = out / pipeline.ODDS_STATE
    # GUARD study-close-floor-first: below the study floor on the freshest balance either
    # file holds, nothing is recorded and nothing is spent
    have = balance([study_state, Path(hub_state or odds.STATE)], when)
    if have is not None and have < market.STUDY_FLOOR:
        raise odds.QuotaFloor(f"{have} credits left, study floor is {market.STUDY_FLOOR}. "
                              f"Refusing before recording an attempt or spending one.")
    # /GUARD
    key = key or odds._api_key()
    if not key:
        raise odds.QuotaFloor("no ODDS_API_KEY set; cannot fetch, and no attempt recorded")
    # GUARD study-close-attempt-before-request: the spend is on the record before it happens,
    # so a lost capture cannot be paid for twice
    closing.record_attempt(when, attempts_path)
    if publish_attempt is not None:
        try:
            publish_attempt()
        except Exception as e:
            raise closing.AttemptNotRecorded(
                f"the attempt marker could not be committed ({type(e).__name__}: {e}); "
                f"no request was made") from e
    # /GUARD
    # Every game inside the window, not only the uncovered ones: the one call prices them all.
    soon = {g: i for g, i in kickoffs.items() if 0 < _lead(i, now) <= WINDOW_MIN}
    season = kickoffs[due[0]].season
    snap = market.snapshot("closing", now=now, season=season, get=get, key=key,
                           state_path=study_state, match=sources.matcher(soon, names(soon)))
    path, closed = _write(out, snap, soon, season)
    return Capture("captured", 1, tuple(due), snap.snapshot_id, closed,
                   tuple(g for g in due if g not in closed), path)


def _lead(info: GameInfo, at: datetime) -> float:
    """Minutes from `at` to the game's kickoff."""
    return (info.kickoff.astimezone(UTC) - at.astimezone(UTC)).total_seconds() / 60.0


def _write(out: Path, snap: market.Snapshot, kickoffs: Mapping[str, GameInfo],
           season: int) -> tuple[Path, tuple[str, ...]]:
    """The snapshot's closes: every matched game kicking off within the window after it.
    The raw payload is kept only as its hash (the prereg's *Reporting*)."""
    rows: list[dict[str, Any]] = []
    for gid, q in sorted(snap.quotes.items()):
        info = kickoffs.get(gid)
        if info is None:
            continue
        lead = _lead(info, snap.captured_at)
        if 0 < lead <= WINDOW_MIN:
            rows.append({"game_id": gid, "kickoff": info.kickoff.astimezone(UTC).isoformat(),
                         "lead_minutes": lead, "home_win_prob": q.home_win_prob,
                         "books": q.books})
    doc = {"snapshot_id": snap.snapshot_id, "job": snap.job,
           "captured_at": snap.captured_at.isoformat(), "source": market.SOURCE,
           "raw_sha256": snap.raw_sha256, "remaining": snap.remaining,
           "unmatched": len(snap.unmatched), "closes": rows}
    path = out / CLOSING_DIR / f"{season}" / f"{snap.snapshot_id}.json"
    atomic.write_text(path, json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return path, tuple(r["game_id"] for r in rows)


# --- CLI ---------------------------------------------------------------------------------

@dataclass
class Deps:
    """Every reach outside the process, so the CLI runs on stubs in tests."""
    schedule_frame: Callable[[int, bool], pl.DataFrame]
    team_names: Callable[[Mapping[str, GameInfo]], Mapping[str, str]]
    odds_get: market.Get | None = None
    odds_key: str | None = None
    hub_state: Path | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))
    publish: Callable[[Path], None] | None = None


def _schedule(season: int, refresh: bool) -> pl.DataFrame:      # pragma: no cover - network
    from hub.fetch import nflverse
    return (nflverse.load("schedules", [season], refresh=refresh)
            .filter(pl.col("season") == season))


def _commit_attempt(out: Path) -> None:                           # pragma: no cover - git
    """Commit and push the attempt marker to the `forecasts` branch, never the running ref."""
    subprocess.run([str(ROOT / ".github" / "scripts" / "commit-state.sh"),
                    "forecasts: closing-line attempt"], check=True, timeout=300,
                   env={**os.environ, "STATE_REPO": str(out), "STATE_PATHS": str(CLOSING_DIR),
                        "STATE_REF": STUDY_BRANCH})


def production() -> Deps:                                         # pragma: no cover - network
    return Deps(schedule_frame=_schedule, team_names=sources.team_names,
                publish=_commit_attempt)


def main(argv: Sequence[str] | None = None, *, echo: Callable[[str], object] = print,
         deps: Deps | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hub.forecasters.close_line")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("due").add_argument("--out", type=Path, required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--commit-attempt", action="store_true",
                     help="commit the attempt marker to the forecasts branch before the request")
    args = ap.parse_args(argv)
    d = deps or production()
    now = d.clock().astimezone(UTC)
    season = sources.season_of(days.eastern(now).date())
    try:
        kickoffs = sources.game_infos(d.schedule_frame(season, args.cmd == "capture"))
    except Exception as exc:
        echo(f"close_line: cannot read the schedule ({type(exc).__name__}: {exc}); "
             f"nothing spent")
        return 2
    if args.cmd == "due":
        return 0 if pending(kickoffs, now, closing.read_attempts(args.out / ATTEMPTS)) else 1
    publish = d.publish
    try:
        got = capture(args.out, now=now, kickoffs=kickoffs, names=d.team_names, get=d.odds_get,
                      key=d.odds_key, hub_state=d.hub_state,
                      publish_attempt=(lambda: publish(args.out))
                      if args.commit_attempt and publish is not None else None)
    except (odds.QuotaFloor, closing.CloseBudgetExceeded, closing.AttemptNotRecorded,
            odds.MarketUnreachable) as exc:
        echo(f"close_line: {exc}")
        return 1
    echo(json.dumps({"status": got.status, "calls": got.calls, "due": len(got.due),
                     "closed": len(got.closed), "missing": len(got.missing),
                     "snapshot_id": got.snapshot_id}, sort_keys=True))
    return 0


if __name__ == "__main__":                                        # pragma: no cover
    sys.exit(main())
