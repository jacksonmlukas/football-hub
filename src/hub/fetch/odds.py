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
from hub.contracts import ODDS_SNAPSHOT
from hub.paths import STATE_DIR

ROOT = Path(__file__).resolve().parents[3]
# Under `state/`, not `data/raw/`. The balance is what the floor below refuses on, and
# `.gitignore` excludes the whole of `data/` as redistributed third-party payloads -- so on
# an Actions runner `credits_remaining` always answered None, the floor never refused, and
# the header read back from each response was written to a file the next run could not see.
# A guard that cannot fire reads as a guard.
STATE = STATE_DIR / "odds.json"

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


def _game_date(commence_time: str) -> str:
    """The date the game is *played on*, which is not the date its kickoff falls on in UTC.

    The Odds API stamps `commence_time` in UTC; nflverse's `gameday` is the local date in
    Eastern. Any kickoff at or after 20:00 ET is past midnight UTC, so slicing the raw string
    put it on the following day and it matched nothing.

    That is not a rounding error, it is the primetime slate. Measured against the 2026
    schedule: **55 of 272 games kick off at or after 20:00 ET** -- 17 Sunday, 17 Monday, 17
    Thursday -- and the 2026-09-04 snapshot lost 65 of 272 events, leaving the store with a
    line for 207 games and none for every Sunday, Monday and Thursday night game of the season.

    Converted rather than offset by a constant because the season crosses out of daylight
    saving in November: -4 through week 9 and -5 after it, and a fixed offset would fix the
    first half and break the second.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    raw = str(commence_time or "")
    try:
        utc = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw[:10]                # unparseable: fall back to the old behaviour, not a crash
    return utc.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _team_abbrs(valid: set[str]) -> dict[str, str]:
    """Full name to abbreviation, from nflverse rather than a hardcoded table.

    The Odds API says "Philadelphia Eagles"; nflverse says "PHI". A literal map would be
    32 lines that go stale the next time a team relocates.

    **`valid` is not optional in practice, and the reason is a silent collapse.**
    `nfl.load_teams()` lists "Los Angeles Rams" twice, once as `LA` and once as `LAR`, so
    building the map with a plain `dict(zip(...))` keeps whichever came last and throws the
    other away. It kept `LAR`. The 2026 schedule uses `LA`. Every Rams game therefore matched
    nothing -- 17 of them, which was the whole of the residual after the timezone fix, and it
    looked like a rounding error rather than one team's entire season.

    The `strict=True` on that zip gave false comfort: it checks that the two columns are the
    same length, which they were, and says nothing about duplicate keys.

    `valid` is required rather than defaulted for that reason: there is no principled way to
    pick between two abbreviations for one name without knowing which the season uses, and a
    default would just be choosing a different arbitrary winner. The caller has the schedule
    in hand, so it can say. A name whose abbreviations are all outside `valid` -- the historical
    entries, Oakland and San Diego and the rest -- keeps the first, and never appears in a
    payload for a season it did not play in.
    """
    import nflreadpy as nfl
    t = nfl.load_teams()
    out: dict[str, str] = {}
    for name, abbr in zip(t["team_name"].to_list(), t["team_abbr"].to_list(), strict=True):
        if name not in out or (abbr in valid and out[name] not in valid):
            out[name] = abbr
    return out


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
    # GUARD credit-floor-refuses [unit/test_fetch_odds.py]: a low balance spends nothing
    if have is not None and have < floor:
        raise QuotaFloor(
            f"{have} credits left, floor is {floor}. Refusing before spending one.")
    # /GUARD

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

    sched = _schedule(season)
    # The season's own abbreviations resolve a team that nflverse lists under two of them.
    in_play = set(sched["home_team"].to_list()) | set(sched["away_team"].to_list())
    abbrs = _team_abbrs(in_play)
    lookup = {
        (r["home_team"], r["away_team"], str(r["gameday"])): (r["game_id"], r["week"])
        for r in sched.iter_rows(named=True)
    }

    rows, no_game, no_line = [], 0, 0
    for ev in payload or []:
        home = abbrs.get(ev.get("home_team", ""))
        away = abbrs.get(ev.get("away_team", ""))
        hit = lookup.get((home, away, _game_date(ev.get("commence_time", ""))))
        spread = _median_home_spread(ev, ev.get("home_team", ""))
        if not hit:
            no_game += 1
            continue
        if spread is None:
            no_line += 1
            continue
        game_id, week = hit
        rows.append({"game_id": game_id, "close_spread": spread,
                     "captured_at": when, "week": int(week)})

    # Counted apart. These used to share one tally reported as "no nflverse game for the
    # team/date pair", which asserted the first cause for both -- and the first cause was
    # the one that was actually broken, so the message was right by accident and would have
    # misdirected the next person the moment a bookmaker simply had not posted a line.
    if no_game:
        print(f"  {no_game} events with no nflverse game for the team/date pair")
    if no_line:
        print(f"  {no_line} events matched a game but had no posted spread")

    df = pl.DataFrame(rows, schema={"game_id": pl.Utf8, "close_spread": pl.Float64,
                                    "captured_at": pl.Datetime, "week": pl.Int64})
    for wk in sorted(set(df["week"].to_list())):
        part = df.filter(pl.col("week") == wk).drop("week")
        # Asserted on what is stored, which is where the boundary is. Validating the whole
        # pull instead would fail `min_rows` on a pull that matched nothing -- and a pull
        # matching nothing writes nothing, so there is no partition to be wrong about.
        ODDS_SNAPSHOT.validate(part)
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
