"""The Odds API: one market, one region, one pull.

The billing model is the whole design constraint. A request costs **markets x regions**,
so asking for spreads and totals across us and uk is four credits for one call.
`docs/decisions.md` records that this multiplier is why props are roadmap-only: a single
props pull would cost more than a month of the free tier.

"Cannot silently burn quota" therefore means two separate things, and this module does
both. The multiplier cannot be triggered by accident -- more than one market or region is
refused before a request is formed. And the balance is never a mystery: every pull reads
`x-requests-remaining` from the response, prints it, and stores it, and a stored balance
below the floor refuses the *next* pull before spending anything.

What the snapshot is for matters as much as what it costs. `hub.store.AS_OF_LINES` joins
lines to predictions on nflverse `game_id`, and this is the only thing that will ever
write more than one line per game. Keyed on The Odds API's own event ids it would be an
island, so events are mapped back to nflverse games via team abbreviations and kickoff
date. Every snapshot appends, because a single closing line makes the as-of join
degenerate -- line *movement* is the thing it exists to resolve.

    uv run python -m hub.fetch.odds --credits
    uv run python -m hub.fetch.odds --snapshot
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub import store
from hub.config import SEASON_AHEAD

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "data" / "raw" / "odds" / "state.json"

BASE = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"

# Exactly one of each. These are singular by intent -- see the multiplier note above.
MARKET = "spreads"
REGION = "us"

# Refuse the next pull below this. Sized to leave room for a full week of Sunday-morning
# snapshots after the balance is noticed, rather than stopping dead at zero.
CREDIT_FLOOR = 50


class MultiplierRefused(Exception):
    """More than one market or region: the request would cost a multiple of one credit."""


class QuotaFloor(Exception):
    """Stored balance is below the floor. Refused before spending a credit."""


def _api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("ODDS_API_KEY") or None


def _read_state(path: Path | None) -> dict[str, Any]:
    try:
        return json.loads(Path(path or STATE).read_text())
    except Exception:
        return {}


def credits_remaining(path: Path | None = None) -> int | None:
    """Last known balance, or None if we have never seen one.

    None is meaningfully different from zero: an unknown balance must not refuse the pull
    that would tell us what it is.
    """
    v = _read_state(path).get("remaining")
    return int(v) if v is not None else None


def _write_state(path: Path | None, remaining: int | None, when: datetime) -> None:
    p = Path(path or STATE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"remaining": remaining, "checked_at": when.isoformat()},
                            indent=2))


def _http_get(params: Mapping[str, Any], key: str) -> tuple[Any, Mapping[str, str]]:
    import requests
    r = requests.get(f"{BASE}/sports/{SPORT}/odds", timeout=30,
                     params={**params, "apiKey": key})
    r.raise_for_status()
    return r.json(), r.headers


def _team_abbrs() -> dict[str, str]:
    """Full name to abbreviation, from nflverse rather than a hardcoded table.

    The Odds API says "Philadelphia Eagles"; nflverse says "PHI". A literal map would be
    32 lines that go stale the next time a team relocates.
    """
    import nflreadpy as nfl
    t = nfl.load_teams()
    return dict(zip(t["team_name"].to_list(), t["team_abbr"].to_list(), strict=True))


def _schedule(season: int) -> pl.DataFrame:
    import nflreadpy as nfl
    return nfl.load_schedules().filter(pl.col("season") == season)


def _median_home_spread(event: Mapping[str, Any], home: str) -> float | None:
    """Median home-side line across books, in nflverse's sign convention.

    Median rather than mean or first: one book hanging an outlier should not become the
    line of record, and books disagree by half points routinely.

    The negation is the part to be careful about. The Odds API reports a *handicap*, so a
    home favourite is -8.5 -- they must win by more than 8.5. nflverse `spread_line` is the
    opposite: positive means the home team is favoured (2025_01_DAL_PHI is home PHI at
    +8.5, and PHI won). `store.AS_OF_LINES` and `store.verify` both already speak nflverse,
    so this converts once, here, rather than leaving two conventions loose in one table.
    Get it wrong and every backtest is confidently backwards while still looking calibrated.
    """
    points = []
    for book in event.get("bookmakers") or []:
        for market in book.get("markets") or []:
            if market.get("key") != MARKET:
                continue
            for outcome in market.get("outcomes") or []:
                if outcome.get("name") == home and outcome.get("point") is not None:
                    points.append(float(outcome["point"]))
    return -statistics.median(points) if points else None


def snapshot(season: int = SEASON_AHEAD, *, markets: str = MARKET, regions: str = REGION,
             state_path: Path | None = None, base: Path | None = None,
             floor: int = CREDIT_FLOOR, now: datetime | None = None) -> pl.DataFrame:
    """One pull, one market, one region. Appends a dated line per matched game."""
    if "," in markets or markets.strip() != markets or not markets:
        raise MultiplierRefused(
            f"markets={markets!r}: cost is markets x regions, so this is "
            f"{len(markets.split(','))} credits per call instead of one.")
    if "," in regions or regions.strip() != regions or not regions:
        raise MultiplierRefused(
            f"regions={regions!r}: cost is markets x regions, so this is "
            f"{len(regions.split(','))} credits per call instead of one.")

    when = now or datetime.now(UTC).replace(tzinfo=None)
    have = credits_remaining(state_path)
    if have is not None and have < floor:
        raise QuotaFloor(
            f"{have} credits left, floor is {floor}. Refusing before spending one.")

    key = _api_key()
    if not key:
        raise QuotaFloor("no ODDS_API_KEY set; cannot fetch")

    payload, headers = _http_get(
        {"markets": markets, "regions": regions, "oddsFormat": "american"}, key)

    remaining_hdr = headers.get("x-requests-remaining")
    remaining = int(float(remaining_hdr)) if remaining_hdr is not None else None
    _write_state(state_path, remaining, when)
    print(f"  odds snapshot: {len(payload or [])} events, "
          f"{remaining if remaining is not None else '?'} credits remaining")
    if remaining is not None and remaining < floor:
        print(f"  WARNING: below the floor of {floor}; the next pull will refuse")

    abbrs = _team_abbrs()
    sched = _schedule(season)
    lookup = {
        (r["home_team"], r["away_team"], str(r["gameday"])): (r["game_id"], r["week"])
        for r in sched.iter_rows(named=True)
    }

    rows, unmatched = [], 0
    for ev in payload or []:
        home = abbrs.get(ev.get("home_team", ""))
        away = abbrs.get(ev.get("away_team", ""))
        day = str(ev.get("commence_time", ""))[:10]
        hit = lookup.get((home, away, day))
        spread = _median_home_spread(ev, ev.get("home_team", ""))
        if not hit or spread is None:
            unmatched += 1
            continue
        game_id, week = hit
        rows.append({"game_id": game_id, "close_spread": spread,
                     "captured_at": when, "week": int(week)})

    if unmatched:
        # Named rather than dropped: a mapping that quietly loses half the slate looks
        # exactly like a quiet week.
        print(f"  {unmatched} unmatched events (no nflverse game for the team/date pair)")

    df = pl.DataFrame(rows, schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "captured_at": pl.Datetime, "week": pl.Int64})
    for wk in sorted(set(df["week"].to_list())):
        part = df.filter(pl.col("week") == wk).drop("week")
        # Snapshots append. A fixed name would overwrite the morning's line with the
        # afternoon's and leave the as-of join nothing to resolve.
        store.write(part, "lines", "nfl", season, wk, base=base,
                    name=f"snap-{when:%Y%m%dT%H%M%S}")
    return df.drop("week")


def credits_report(path: Path | None = None) -> int:
    have = credits_remaining(path)
    state = _read_state(path)
    if have is None:
        print("  odds credits: unknown (no pull recorded yet)")
    else:
        print(f"  odds credits: {have:,} remaining, floor {CREDIT_FLOOR} "
              f"(as of {state.get('checked_at', '?')})")
    if not _api_key():
        print("  no ODDS_API_KEY set; fetching unavailable, accounting still works")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.odds",
        description="One market, one region, one pull. Cost is markets x regions.")
    ap.add_argument("--snapshot", action="store_true", help="take one dated line snapshot")
    ap.add_argument("--credits", action="store_true", help="report the stored balance")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--state-path", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    spath = Path(a.state_path) if a.state_path else None

    if a.credits or not a.snapshot:
        return credits_report(spath)

    if not _api_key():
        print("hub.fetch.odds: no ODDS_API_KEY set; add one to .env to fetch. "
              "Balance accounting still works via --credits.", file=sys.stderr)
        return 1
    try:
        snapshot(a.season, state_path=spath)
    except (QuotaFloor, MultiplierRefused) as e:
        print(f"hub.fetch.odds: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
