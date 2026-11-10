"""Every team in the league, once a week: rosters, standings, remaining schedule (#545).

Scoring a Move (GLOSSARY.md) needs the whole league, and `state/` journalled only our roster.
One ESPN League load -- the cookie path `hub.fetch.espn.league_settings` and `my_team` already
use, never a request per team -- becomes `state/league/<season>/wk<NN>.json`, which the `slate`
workflow commits beside the other `state/` files.

Each capture holds the 12 teams (id, name, which is ours, W-L-T, regular-season points-for, the
roster as `espn.roster_rows` rows plus ESPN's projection for every remaining week through 17),
the remaining regular-season schedule as (week, team, opponent) pairs, the playoff weeks, the
trade deadline (ISO UTC; `null` when ESPN carries none), and the free-agent pool
`hub.fetch.free_agents` last wrote, so the file is self-contained.

**Offline or cookie-less, nothing is written.** The CLI says so and exits 0: the last-good file
for the week stays, and so do the files of earlier weeks. A capture the schema contract refuses is
treated the same way. A re-run in the same week overwrites that week's file.

    uv run python -m hub.fetch.league --write     # load once, validate, write the week's file
    uv run python -m hub.fetch.league             # load and summarise; write nothing
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hub import atomic
from hub.config import SEASON_AHEAD
from hub.contracts import ContractViolation, check_league_capture
from hub.paths import STATE_DIR

STATE_LEAGUE = STATE_DIR / "league"

# The last NFL week a fantasy matchup is played in this league's calendar (week 18 is never one).
LAST_PROJECTED_WEEK = 17
# The capture runs from the first week to the last the NFL plays, so the week is a filename
# component and a bound on it, not a model input.
LAST_NFL_WEEK = 18


def _iso(ms: Any) -> str | None:
    """espn-api's epoch milliseconds as ISO UTC; 0, absent or unparseable is no deadline."""
    if not isinstance(ms, (int, float)) or isinstance(ms, bool) or not ms or ms < 0:
        return None
    return datetime.fromtimestamp(ms / 1000, UTC).isoformat()


def _playoff_weeks(reg_weeks: int, playoff_teams: int) -> list[int]:
    """The weeks after the regular season: one per round of a bracket of `playoff_teams`."""
    rounds = math.ceil(math.log2(playoff_teams)) if playoff_teams > 1 else 0
    return list(range(reg_weeks + 1, reg_weeks + 1 + rounds))


def _team_block(team: Any, week: int, mine: Any) -> dict[str, Any]:
    from hub.fetch.espn import roster_rows, weekly_projection
    rows = roster_rows(team, week).to_dicts()
    for row, p in zip(rows, team.roster, strict=True):
        got = {str(w): weekly_projection(p, w) for w in range(week, LAST_PROJECTED_WEEK + 1)}
        row["week_proj"] = {w: v for w, v in got.items() if v is not None}
    return {
        "team_id": int(team.team_id), "name": str(team.team_name), "is_mine": team is mine,
        "wins": int(team.wins), "losses": int(team.losses), "ties": int(team.ties),
        "points_for": float(team.points_for), "roster": rows,
    }


def build_capture(league: Any, *, swid: str | None = None, now: datetime | None = None,
                  free_agents: dict[str, Any] | None = None) -> dict[str, Any]:
    """The capture dict for an espn-api-shaped League. Pure: it reads `league` and nothing else."""
    from hub.fetch.espn import my_team
    stamp = (now or datetime.now(UTC)).astimezone(UTC)
    week = int(getattr(league, "current_week", 0) or 0)
    s = league.settings
    reg = int(s.reg_season_count)
    playoff_teams = int(s.playoff_team_count)
    mine = my_team(league, swid)
    teams = [_team_block(t, week, mine) for t in league.teams]
    schedule = []
    for t in league.teams:
        for w in range(week, reg + 1):
            if w - 1 < len(t.schedule):
                schedule.append({"week": w, "team_id": int(t.team_id),
                                 "opponent_id": int(t.schedule[w - 1].team_id)})
    return {
        "season": int(getattr(league, "year", None) or SEASON_AHEAD),
        "week": week,
        "captured_at": stamp.isoformat(),
        "my_team_id": int(mine.team_id),
        "teams": teams,
        "schedule": schedule,
        "settings": {
            "reg_season_weeks": reg, "playoff_team_count": playoff_teams,
            "playoff_weeks": _playoff_weeks(reg, playoff_teams),
            "trade_deadline": _iso(getattr(s, "trade_deadline", 0)),
        },
        "free_agents": free_agents,
    }


def read_free_agents() -> dict[str, Any] | None:
    """The pool `hub.fetch.free_agents` last wrote, rows as written, with its at-cap flag."""
    from hub.fetch import free_agents
    df, meta = free_agents.read()
    if df is None:
        return None
    return {"fetched_at": meta.get("fetched_at"), "capped": bool(meta.get("capped", False)),
            "rows": df.to_dicts()}


def load_league() -> Any:  # pragma: no cover - network
    """One League load, the same cookie path as the roster."""
    from hub.fetch.espn import league_settings
    return league_settings(SEASON_AHEAD).league


def capture(load: Callable[[], Any], *, out_root: Path, now: datetime | None = None,
            swid: str | None = None,
            free_agents: Callable[[], dict[str, Any] | None] = read_free_agents,
            write: bool = True) -> tuple[str, bool]:
    """Load once, build, validate, write. Returns `(status line, ok)`; never raises on a bad
    source. `ok` is False whenever nothing could be captured."""
    try:
        league = load()
        cap = build_capture(league, swid=swid, now=now, free_agents=free_agents())
    except Exception as e:                      # graceful degradation, per CLAUDE.md
        print(f"hub.fetch.league: load failed: {type(e).__name__}: {e}", file=sys.stderr)
        return f"ESPN unreachable or no cookie ({type(e).__name__}); nothing written", False
    if not 1 <= cap["week"] <= LAST_NFL_WEEK:
        return f"league week is {cap['week']}, not a scoring week; nothing written", False
    try:
        check_league_capture(cap)
    except ContractViolation as e:
        print(f"hub.fetch.league: {e}", file=sys.stderr)
        return "the capture was refused by its contract; nothing written", False
    summary = f"{len(cap['teams'])} teams, {len(cap['schedule'])} schedule pairs"
    if not write:
        return f"week {cap['week']}: {summary} (not written; pass --write)", True
    path = out_root / str(cap["season"]) / f"wk{cap['week']:02d}.json"
    atomic.write_text(path, json.dumps(cap, indent=1, sort_keys=True) + "\n")
    return f"week {cap['week']}: {summary}, wrote {path.name}", True


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.league",
        description="Capture every team's roster, standings and schedule (one ESPN load).")
    ap.add_argument("--write", action="store_true", help="write state/league/<season>/wk<NN>.json")
    ap.add_argument("--out", default=None, help="override the state/league directory")
    a = ap.parse_args(list(argv) if argv is not None else None)
    root = Path(a.out) if a.out else STATE_LEAGUE
    line, ok = capture(load_league, out_root=root, write=a.write)
    print(f"  {line}")
    # `--write` is the slate's optional step: it never fails the slate, so it exits 0 whatever
    # happened and the last-good file stays. A look without `--write` that captured nothing
    # fails, so an operator who ran it by hand is told.
    return 0 if (ok or a.write) else 1


if __name__ == "__main__":                       # pragma: no cover
    sys.exit(main())
