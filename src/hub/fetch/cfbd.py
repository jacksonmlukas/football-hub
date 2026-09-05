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

**Which week, and what a run that fetched none says.** `configured_week` owns the first
question and its docstring owns the argument. The second is `record_run`: every invocation
of this CLI leaves a record in `site/data/cfbd.json` saying whether it fetched, and a run
that fetched nothing is distinguishable there from one that fetched and found nothing. That
file rather than an exit code because `make slate` marks this source optional with a leading
`-` -- which is correct, an unconfigured extra must not take a Sunday down, and which means
the exit code reaches nobody.

    uv run python -m hub.fetch.cfbd --quota
    uv run python -m hub.fetch.cfbd --week 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import polars as pl

from hub import jsonio
from hub.config import SEASON_AHEAD
from hub.contracts import CFBD_GAMES, CFBD_LINES, Contract
from hub.paths import SITE, STATE_DIR

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "data" / "raw" / "cfbd"
# Beside the odds balance and away from the cache, for the same reason. The responses under
# `CACHE` are third-party payloads this repo cannot publish and `.gitignore` is right to
# exclude them; the count of calls spent against a 1,000-a-month free tier is ours, and
# keeping it in the excluded tree meant every scheduled run started the month over.
QUOTA = STATE_DIR / "cfbd-quota.json"

# Where a run says what it did, beside the artifacts the page reads. Not a log line: the
# slate marks this source optional with a leading `-`, so neither the exit code nor stderr
# reaches anything downstream, and the workflow commits `site/data` -- which makes this a
# committed statement rather than a scroll-back.
#
# **Counts, never rows.** `docs/cfbd-quota.md`: redistributing CFBD payloads is a terms
# violation that gets access revoked, `data/raw/` is gitignored for exactly that reason, and
# everything under `site/data` is committed to a repo intended to go public.
STATUS = SITE / "cfbd.json"

# The season's first game, as a date. The whole of this module's configuration; the argument
# for it being a date rather than a week number is in `configured_week`.
CFB_WEEK_ONE_ENV = "CFB_WEEK_ONE"

# The last college week that is a *week*. Weeks 1-15 are the regular season through the
# conference championships; past that is the postseason, which CFBD asks for as a
# `seasonType` and not as a week number. A run counting past this records that there is no
# week rather than fetching one that does not exist.
REGULAR_SEASON_WEEKS = 15

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


def _env() -> Mapping[str, str]:
    """The process environment with `.env` folded into it.

    One reader rather than two, because this module now takes two things from it -- the key
    and the season's start date -- and a test that has to defeat `load_dotenv` twice to say
    what the environment is will eventually only defeat it once.
    """
    from dotenv import load_dotenv
    load_dotenv()
    return os.environ


def _api_key() -> str | None:
    return _env().get("CFBD_API_KEY") or None


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
        # GUARD team-keys-refused [unit/test_fetch_cfbd.py]: narrowing below a week is refused
        if narrowing:
            raise LoopRefused(
                f"{narrowing} would narrow this below a week, which is how the 1,000/month "
                f"quota dies. Pull the bulk payload and filter it in polars instead.")
        # /GUARD
        params.update(extra)

    path = _cache_path(endpoint, year, week, cache)
    if path.exists():
        return pl.read_parquet(path)

    # GUARD run-ceiling-stops-a-loop [unit/test_fetch_cfbd.py]: a loop stops at call 13
    if _CALLS_THIS_RUN >= MAX_CALLS_PER_RUN:
        raise QuotaExceeded(
            f"{MAX_CALLS_PER_RUN} calls in one run; a week costs 5-8. This is a loop.")
    # /GUARD
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
        #
        # `df.height and` because both CFBD contracts declare `min_rows=1`, so validating a
        # frame of no rows reports "0 rows < min 1; missing columns [...]" -- which reads as
        # the source having changed shape when it has simply said "nothing here", and which
        # a week nobody has played yet answers with. Emptiness is a state the caller reports
        # (`record_run`, below); it is not a contract failure, and the contract still sees
        # every response that has rows, which is every response a renamed field could hide in.
        if df.height and (contract := CONTRACTS.get(endpoint)) is not None:
            contract.validate(df)
        out[endpoint] = df
        print(f"    {endpoint:<14} {df.height:>6,} rows | {len(df.columns):>3} cols")
    print(f"  quota: {quota_used(quota_path):,} of {FREE_TIER_MONTHLY:,} this month")
    return out


class WeekChoice(NamedTuple):
    """The college week to fetch, or None and the sentence saying why there is none.

    The reason travels with the decision, for the reason `publish.Kept` does: it belongs to
    the run that declined. A standing sentence kept somewhere else is one that eventually
    describes a different run than the one the reader is looking at.
    """
    week: int | None
    why: str


def configured_week(now: datetime | None = None) -> WeekChoice:
    """Which college week a scheduled run fetches, counted from the season's start date.

    **Issue #56 asks where the week comes from. Three other answers were available and each
    is worse.**

    *From the store*, the way `hub.publish.default_week` takes the latest week already
    predicted. That works for the NFL because the store is full of NFL weeks. The college
    calendar is a different one -- its own week-1 date, a Week 0 the NFL has no equivalent
    of, fifteen regular-season weeks against eighteen -- so the store's number would be
    confidently wrong here, and it cannot answer at all before anything has been written.

    *From the calendar alone.* College week 1 is not on a fixed date; it moves with Labor
    Day and with whether a season opens a week early. Code that works it out is guessing,
    and a wrong guess does not fail loudly -- it fetches the wrong week, caches it, and
    reports a successful refresh.

    *From CFBD's own `/calendar`.* Authoritative, and it puts a network call on the question
    of what week it is. `publish.default_week` already rejected that reasoning for the NFL
    side and the sentence holds here: a weekly refresh that needs a live API to decide what
    week it is has one more way to fail on a Sunday.

    So configuration -- and specifically **the date of the season's first game rather than a
    week number**. A pinned number (`CFB_WEEK=3`) is right for seven days and silently wrong
    for the rest of the season: in November it would still fetch week 3, cache it, and
    record a successful refresh. A start date stated once in August stays true through
    January, which is the difference between a system that needs somebody every Wednesday
    and one that does not -- and CLAUDE.md is blunt that the first kind dies in October.

    It lives in the environment beside `CFBD_API_KEY`: `.env` locally, a repository variable
    in Actions. A source that already needs configuring before it can fetch at all is the
    right place to put the one more line, rather than inventing a config surface for it.

    **The date is week 1's first game; the week boundary is the Tuesday before it.** Those
    are two different things and both are needed. The date is the fact a human states and it
    is naturally the opening Thursday or Saturday; a college week is the Tuesday-to-Monday
    block that game falls in, which is where CFBD's own week numbers change over. Counting
    plain seven-day blocks from an opening Thursday puts the Wednesday 11:00 UTC refresh a
    week behind from week 2 on -- fetching the week that just finished rather than the one
    it is refreshing for. So the stated date is snapped back to its Tuesday and the count
    runs from there.

    Unset, unparseable, before the first game, or past the regular season, this returns no
    week and the sentence for it. Nothing here ever invents one -- `--week N` is how a human
    asks for a specific week, including a backfill.
    """
    raw = (_env().get(CFB_WEEK_ONE_ENV) or "").strip()
    if not raw:
        return WeekChoice(None, (
            f"nothing was fetched: {CFB_WEEK_ONE_ENV} is not set, so nothing here knows "
            f"which college week it is. Set it to the date of the season's first game "
            f"(YYYY-MM-DD), or pass --week N"))
    try:
        first = date.fromisoformat(raw)
    except ValueError:
        return WeekChoice(None, (
            f"nothing was fetched: {CFB_WEEK_ONE_ENV}={raw!r} is not a YYYY-MM-DD date, "
            f"and a week is not being guessed from it"))
    # Back to the Tuesday that opens the stated date's week. `weekday()` is Monday 0, so
    # Tuesday is 1 and `(w - 1) % 7` is how many days back that Tuesday is -- zero when the
    # date given is already one.
    opens = first - timedelta(days=(first.weekday() - 1) % 7)
    today = (now or datetime.now(UTC)).date()
    if today < opens:
        return WeekChoice(None, (
            f"nothing was fetched: the season has not started -- its first game is "
            f"{first.isoformat()}, whose week opens {opens.isoformat()}"))
    # Whole weeks since week 1 opened, one-based. Counted in UTC while the games are played
    # in North America, which moves the Monday-to-Tuesday boundary by a few hours; the
    # scheduled runs are Wednesday 11:00 and Saturday 14:00 UTC, both far from it.
    n = (today - opens).days // 7 + 1
    if n > REGULAR_SEASON_WEEKS:
        return WeekChoice(None, (
            f"nothing was fetched: week {n} counted from {opens.isoformat()} is past the "
            f"{REGULAR_SEASON_WEEKS}-week regular season, and the postseason is a "
            f"seasonType rather than a week number"))
    return WeekChoice(n, "")


def record_run(season: int, week_no: int | None, *,
               rows: Mapping[str, int] | None = None, why: str | None = None,
               path: Path | None = None, quota_path: Path | None = None) -> dict[str, Any]:
    """Write what this run did, in the three states `publish.Artifact.record` writes.

    That mapping, because a reader who has learned to read the manifest should not have to
    learn a second vocabulary for this file:

    * **fetched, with rows** -- `stale: false`, `reason: null`. `Artifact.record`'s fresh
      payload.
    * **fetched, and nothing there** -- `stale: true` with a reason in the run's own words.
      `publish.Kept`: the source answered, and an empty answer is never reported fresh.
    * **not fetched at all** -- `stale: true`, `fetched: false`, and the sentence from
      whatever declined. The state issue #27 asked for and #56 found missing: without it
      "no week was configured" and "the week held no games" are the same silence.

    `fetched` as a field of its own rather than leaving the distinction to prose, because
    the slate workflow reads this with `jq` and a workflow cannot read a sentence.

    **Counts, never rows.** See `STATUS`: a CFBD payload committed under `site/data` would
    be redistribution, which is the terms violation `docs/cfbd-quota.md` warns costs access.
    """
    fetched = rows is not None
    counts = dict(rows or {})
    summary = ", ".join(f"{k} {v:,} rows" for k, v in counts.items())
    if not fetched:
        stale, reason = True, (why or "nothing was fetched")
    elif not sum(counts.values()):
        stale, reason = True, (f"week {week_no} of {season} was read and came back empty")
    else:
        stale, reason = False, None
    got: dict[str, Any] = {
        "name": "cfbd", "source": "hub.fetch.cfbd", "generated_at": jsonio.stamp(),
        "season": season, "week": week_no, "fetched": fetched,
        "stale": stale, "reason": reason,
        "rows_by_endpoint": counts,
        "quota": {"month": _month_key(), "used": quota_used(quota_path),
                  "limit": FREE_TIER_MONTHLY},
    }
    p = Path(path or STATUS)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(jsonio.dumps(got, indent=2))
    # The reason already opens with what happened -- "nothing was fetched: ...", "week 2 of
    # 2026 was read and came back empty" -- so prefixing it with a verdict only stutters.
    print(f"  cfbd: {reason or f'week {week_no} fetched, ' + summary}; recorded in {p}")
    return got


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.cfbd",
        description="CFBD bulk fetch. Week-level endpoints only, by construction.")
    ap.add_argument("--quota", action="store_true", help="report calls used this month")
    ap.add_argument("--week", type=int, default=None,
                    help=f"pull one week's bulk slate; defaults to the week counted from "
                         f"${CFB_WEEK_ONE_ENV}")
    ap.add_argument("--year", type=int, default=SEASON_AHEAD)
    ap.add_argument("--status-path", default=None,
                    help="where to record what this run did (default site/data/cfbd.json)")
    ap.add_argument("--quota-path", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    reset_run_budget()
    qpath = Path(a.quota_path) if a.quota_path else None
    spath = Path(a.status_path) if a.status_path else None

    if a.quota:
        # An accounting question, not a refresh, so it leaves no run record: `make check`
        # asking how much of the month is left must not overwrite what the last slate said
        # it did. This branch is also what the old no-week path fell into, which is how a
        # fetch that never happened came to print a healthy-looking report and exit 0.
        return quota_report(qpath)

    # `--week` is a human asking for a specific week -- a backfill, or a rerun. Everything
    # else is a scheduled run, which has to work out the week for itself or say that it
    # could not; `configured_week` holds the argument for how.
    week_no, why_not = (WeekChoice(a.week, "") if a.week is not None else configured_week())

    # GUARD no-week-is-recorded [unit/test_fetch_cfbd.py]: no week means a record, not silence
    if week_no is None:
        record_run(a.year, None, why=why_not, path=spath, quota_path=qpath)
        print(f"hub.fetch.cfbd: {why_not}", file=sys.stderr)
        return 1
    # /GUARD

    if not _api_key():
        why = ("nothing was fetched: no CFBD_API_KEY set; add one to .env. Quota "
               "accounting still works via --quota")
        record_run(a.year, week_no, why=why, path=spath, quota_path=qpath)
        print(f"hub.fetch.cfbd: {why}", file=sys.stderr)
        return 1

    try:
        got = week(a.year, week_no, quota_path=qpath)
    except Exception as e:
        # Caught rather than raised, because the traceback goes to the same place the exit
        # code does -- nowhere. An optional source that is down must not halt the slate
        # (CLAUDE.md's degradation rule, and the leading `-` in the Makefile), and it must
        # not be indistinguishable from one that succeeded either.
        why = f"nothing was fetched: {type(e).__name__}: {e}"[:400]
        record_run(a.year, week_no, why=why, path=spath, quota_path=qpath)
        print(f"hub.fetch.cfbd: {why}", file=sys.stderr)
        return 1

    # A week that answered, whether or not it held anything. `record_run` tells those two
    # apart; the exit code does not try to, because the fetch happened either way.
    record_run(a.year, week_no, rows={k: v.height for k, v in got.items()},
               path=spath, quota_path=qpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
