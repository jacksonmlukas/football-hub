"""CFBD fetch layer: bulk endpoints only, and looping made impossible.

The free tier is 1,000 calls a month and `docs/cfbd-quota.md` is precise about how it
dies. One call per team is 136 calls for a week that `/games?year&week` covers in one. The
quota does not erode gradually -- a single loop left running spends a season's budget in a
minute, and multiple keys or rate-limit circumvention are terms violations that get access
revoked rather than throttled.

So this module does not ask callers to remember the rule. Three independent guards:

  1. **No parameter can name a team.** `bulk()` takes an endpoint, a year and optionally a
     week. There is nowhere to put a team, so the mistake cannot be typed.
  2. **Team and game keys are rejected by name.** The `extra` escape hatch exists for real
     API parameters, and refuses the ones that would narrow a request below a week.
  3. **A per-run call ceiling.** Twelve calls covers the documented weekly budget of five
     to eight with room to spare, and stops a 136-iteration loop at call thirteen. This is
     the guard that survives someone editing the other two.

Every response is cached under `data/raw/cfbd/`. Caching here is a quota mechanism, not a
speed one: a completed season is never re-fetched.

    uv run python -m hub.fetch.cfbd --quota
    uv run python -m hub.fetch.cfbd --week 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from hub.config import SEASON_AHEAD
from hub.contracts import CFBD_GAMES, CFBD_LINES, Contract
from hub.paths import STATE_DIR

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "raw" / "cfbd"
# Beside the odds balance and away from the cache, for the same reason. The responses under
# `CACHE` are third-party payloads this repo cannot publish and `.gitignore` is right to
# exclude them; the count of calls spent against a 1,000-a-month free tier is ours, and
# keeping it in the excluded tree meant every scheduled run started the month over.
QUOTA = STATE_DIR / "cfbd-quota.json"

BASE = "https://api.collegefootballdata.com"
FREE_TIER_MONTHLY = 1_000

# Twelve covers the documented 5-8 call week with headroom, and stops a loop over 136 FBS
# teams at call thirteen. Deliberately per-run rather than per-month: the monthly budget
# would let a loop spend 900 calls before noticing.
MAX_CALLS_PER_RUN = 12

ENDPOINTS: dict[str, str] = {
    "games": "/games",
    "lines": "/lines",
    "box": "/games/teams",
    "sp_ratings": "/ratings/sp",
    "season_stats": "/stats/season",
}

# What a weekly refresh pulls, matching the budget table in docs/cfbd-quota.md.
WEEKLY: tuple[str, ...] = ("games", "lines", "box")

# Parameters that would narrow a request below a whole week. `conference` is here too:
# ten conference calls is a cheaper version of the same mistake, and the filtering it
# would do is free in polars once the bulk payload is local.
FORBIDDEN_PARAMS = frozenset({
    "team", "home", "away", "gameId", "game_id", "id", "conference", "player", "playerId",
})

_CALLS_THIS_RUN = 0


def reset_run_budget() -> None:
    """Start a new run.

    "Per run" needs a defined boundary or it means "per process", which would make a
    long-lived caller trip the ceiling on legitimate work hours later. The boundary is CLI
    entry: one invocation, one budget. It is deliberately not reset inside `week()`, since
    a loop over `bulk()` is exactly what the ceiling exists to stop.
    """
    global _CALLS_THIS_RUN
    _CALLS_THIS_RUN = 0


class LoopRefused(Exception):
    """A request that would fan out per team or per game, or name an unknown endpoint."""


class QuotaExceeded(Exception):
    """Refused rather than spend a call: either the run ceiling or the monthly budget."""


def _api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ.get("CFBD_API_KEY") or None


def _month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(UTC)).strftime("%Y-%m")


def quota_used(path: Path | None = None) -> int:
    """Calls spent this billing month. A corrupt counter reads as zero.

    Erring low is deliberate: an unreadable counter that blocked every fetch would take
    the pipeline down over bookkeeping, and the run ceiling still bounds the damage.
    """
    p = Path(path or QUOTA)
    try:
        return int(json.loads(p.read_text()).get(_month_key(), 0))
    except Exception:
        return 0


def _record_call(path: Path | None = None) -> None:
    p = Path(path or QUOTA)
    try:
        counts = json.loads(p.read_text())
    except Exception:
        counts = {}
    counts[_month_key()] = int(counts.get(_month_key(), 0)) + 1
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(counts, indent=2, sort_keys=True))


def quota_report(path: Path | None = None) -> int:
    used = quota_used(path)
    left = FREE_TIER_MONTHLY - used
    print(f"  CFBD quota, {_month_key()}: {used:,} used of {FREE_TIER_MONTHLY:,}, "
          f"{left:,} remaining")
    print(f"  run ceiling: {MAX_CALLS_PER_RUN} calls (a per-team loop stops here)")
    if not _api_key():
        print("  no CFBD_API_KEY set; fetching is unavailable, accounting still works")
    return 0


def _http_get(path: str, params: Mapping[str, Any], key: str) -> Any:
    import requests
    r = requests.get(f"{BASE}{path}", params=dict(params), timeout=30,
                     headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()


def _cache_path(endpoint: str, year: int, week: int | None, cache: Path | None) -> Path:
    root = Path(cache or CACHE)
    stem = f"{year}" if week is None else f"{year}-w{week:02d}"
    return root / endpoint / f"{stem}.parquet"


def bulk(endpoint: str, year: int, week: int | None = None, *,
         extra: Mapping[str, Any] | None = None,
         cache: Path | None = None, quota_path: Path | None = None) -> pl.DataFrame:
    """One bulk pull. There is no way to ask this for a single team.

    Note the signature: endpoint, year, week. A caller who wants Alabama pulls the week and
    filters in polars, because that is one call instead of one hundred and thirty-six.
    """
    global _CALLS_THIS_RUN

    if endpoint not in ENDPOINTS:
        raise LoopRefused(
            f"unknown endpoint {endpoint!r}. Known: {', '.join(sorted(ENDPOINTS))}")

    params: dict[str, Any] = {"year": year}
    if week is not None:
        params["week"] = week
    if extra:
        narrowing = sorted(set(extra) & FORBIDDEN_PARAMS)
        if narrowing:
            raise LoopRefused(
                f"{narrowing} would narrow this below a week, which is how the 1,000/month "
                f"quota dies. Pull the bulk payload and filter it in polars instead.")
        params.update(extra)

    path = _cache_path(endpoint, year, week, cache)
    if path.exists():
        return pl.read_parquet(path)

    if _CALLS_THIS_RUN >= MAX_CALLS_PER_RUN:
        raise QuotaExceeded(
            f"{MAX_CALLS_PER_RUN} calls in one run; a week costs 5-8. This is a loop.")
    if quota_used(quota_path) >= FREE_TIER_MONTHLY:
        raise QuotaExceeded(
            f"monthly budget of {FREE_TIER_MONTHLY:,} is spent. Waiting beats a second key: "
            f"multiple keys are a terms violation that gets access revoked.")

    key = _api_key()
    if not key:
        raise LoopRefused("no CFBD_API_KEY set; cannot fetch")

    # Counted in `finally`, because the quota is spent by the *request*, not by the reply.
    # These two lines used to sit after the call, so anything `_http_get` raised -- a 429, a
    # 500, a timeout, all of which `raise_for_status` turns into an exception -- spent a real
    # call that neither counter ever saw. A rate-limited endpoint retried a few times could
    # burn the monthly budget while `quota_used()` stayed flat and the run ceiling below
    # never fired, which is precisely the failure this module exists to make impossible.
    #
    # A connection error that never reached CFBD is over-counted by one. That is the safe
    # direction: erring high refuses a little early, erring low is how the budget disappears.
    try:
        payload = _http_get(ENDPOINTS[endpoint], params, key)
    finally:
        _CALLS_THIS_RUN += 1
        _record_call(quota_path)

    df = pl.DataFrame(payload, infer_schema_length=None) if payload else pl.DataFrame()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    return df


# Endpoint to contract. Neither of these has ever met a live response -- there was no CFBD
# key on the machine they were written on -- so both carry `verified_against_live=False` and
# a first failure should be read as "the declaration was a guess" before "the source broke".
CONTRACTS: dict[str, Contract] = {"games": CFBD_GAMES, "lines": CFBD_LINES}


def week(year: int, week_no: int, *, cache: Path | None = None,
         quota_path: Path | None = None) -> dict[str, pl.DataFrame]:
    """The documented weekly slate: games, lines, box scores. Three calls, not 408."""
    out: dict[str, pl.DataFrame] = {}
    print(f"  CFBD week {week_no}, {year}")
    for endpoint in WEEKLY:
        df = bulk(endpoint, year, week_no, cache=cache, quota_path=quota_path)
        # A registry rather than a branch, and the same shape `hub.fetch.nflverse` uses:
        # `box` has no declared contract and is not being given a weak one to fill the row.
        if (contract := CONTRACTS.get(endpoint)) is not None:
            contract.validate(df)
        out[endpoint] = df
        print(f"    {endpoint:<14} {df.height:>6,} rows | {len(df.columns):>3} cols")
    print(f"  quota: {quota_used(quota_path):,} of {FREE_TIER_MONTHLY:,} this month")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.cfbd",
        description="CFBD bulk fetch. Week-level endpoints only, by construction.")
    ap.add_argument("--quota", action="store_true", help="report calls used this month")
    ap.add_argument("--week", type=int, default=None, help="pull one week's bulk slate")
    ap.add_argument("--year", type=int, default=SEASON_AHEAD)
    ap.add_argument("--quota-path", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    reset_run_budget()
    qpath = Path(a.quota_path) if a.quota_path else None

    if a.quota:
        return quota_report(qpath)
    if a.week is None:
        # Not success. `make slate` runs this with no `--week` unless `WEEK` is set and the
        # scheduled run sets none, so this branch printed a healthy-looking quota report and
        # exited 0 -- every scheduled run reporting success for a fetch that did not happen.
        # The Makefile marks this source optional with a leading `-`, so a non-zero exit
        # still lets the slate continue; what changes is that it stops lying about it.
        quota_report(qpath)
        print("hub.fetch.cfbd: no --week given, so nothing was fetched. That was a quota "
              "report. Pass --week N, or `make slate WEEK=N`.", file=sys.stderr)
        return 1

    if not _api_key():
        print("hub.fetch.cfbd: no CFBD_API_KEY set; add one to .env to fetch. "
              "Quota accounting still works via --quota.", file=sys.stderr)
        return 1
    week(a.year, a.week, quota_path=qpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
