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

**And the cache can now be corrected** (#175). It used to be permanent by file existence --
if the path was there it was returned, with no maximum age, no refresh parameter and nothing
recording when the bytes arrived. That is the expensive half of a metered source rather than
the cheap one: it is both the reason to cache and the reason a stale price could never be
put right. So a written entry carries its capture time in a `.capture.json` beside it,
`refresh=True` re-fetches, and `max_age` states a bound past which an entry is re-fetched
instead of served. An entry written before this reads back as an *unknown* capture time and
never a guessed one -- the file's mtime dates the file, not the fetch. Both parameters
default off, so nothing that was not asked to refresh spends anything, and a refresh that
cannot be afforded serves the cached payload and says that it did.

Those three are about a loop. A fourth is about the suite: `_http_get` refuses to run under
pytest at all, outside the one directory that exists to hit live APIs. Tests patch the
transport and should, but a test suite that only fails to spend quota because nobody has set
`CFB_WEEK_ONE` on this laptop is not a guard, it is a coincidence (#68).

**Which week, and what a run that fetched none says.** `configured_week` owns the first
question and its docstring owns the argument. The second is `record_run`: every invocation
of this CLI leaves a record in `site/data/cfbd.json` saying whether it fetched, and a run
that fetched nothing is distinguishable there from one that fetched and found nothing. That
file rather than an exit code because `make slate` marks this source optional with a leading
`-` -- which is correct, an unconfigured extra must not take a Sunday down, and which means
the exit code reaches nobody.

    uv run python -m hub.fetch.cfbd --quota
    uv run python -m hub.fetch.cfbd --week 3
    uv run python -m hub.fetch.cfbd --week 3 --refresh
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
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

# The pytest node running right now, or nothing outside a test. pytest sets this for the
# duration of each test's setup, call and teardown, and nothing else in this repo writes it.
PYTEST_NODE_ENV = "PYTEST_CURRENT_TEST"

# The one suite allowed to reach CFBD. `tests/golden/` exists to diff a live response against
# the frozen fixture -- it is the only thing in the repo that knows whether the two CFBD
# contracts, both written from documentation, resemble reality -- and it is marked `golden`
# and deselected by default (`addopts = "-m 'not golden'"`, pyproject.toml). Node ids are
# relative to pytest's rootdir, so a run started from inside `tests/` does not match and is
# refused: erring toward refusal is the direction this module errs in everywhere.
LIVE_TEST_SUITE = "tests/golden/"

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


class LiveCallRefused(Exception):
    """A test reached the live transport. Refused before anything left the process."""


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
    # GUARD no-live-call-from-the-suite [unit/test_fetch_cfbd.py]: a test cannot spend quota
    #
    # #68, and the only place in this module where the answer does not depend on the machine
    # it runs on. `tests/contracts/test_quota_state_survives_a_runner.py` drove this CLI with
    # neither the environment reader nor the cache patched. That was harmless while the CLI
    # had no week to resolve and took its no-week branch; once it began counting one from
    # `CFB_WEEK_ONE` it became three live calls on every run of the suite for anyone holding
    # a key. Measured 2026-09-05: the `.env` on the machine this was written on carries
    # neither the anchor nor a key, so nothing was spent that day -- but `.env.example` now
    # instructs a developer to set the anchor, and the key is already a repository secret, so
    # the only thing between the suite and the account was which machine ran it.
    #
    # Patching the transport per test is the right habit and every test in the sibling module
    # does it. It is not a guarantee, because the guarantee has to hold for the test nobody
    # remembered to patch. This does, and it costs one environment read per request.
    node = os.environ.get(PYTEST_NODE_ENV, "")
    if node and not node.startswith(LIVE_TEST_SUITE):
        raise LiveCallRefused(
            f"{node.split(' ')[0]} would spend a live CFBD call. The free tier is "
            f"{FREE_TIER_MONTHLY:,} a month and docs/cfbd-quota.md records that "
            f"rate-limit circumvention gets access revoked rather than throttled, so no "
            f"test fetches: patch `_http_get`, the way tests/unit/test_fetch_cfbd.py does. "
            f"Only {LIVE_TEST_SUITE} may reach CFBD, and it is deselected by default.")
    # /GUARD
    import requests
    r = requests.get(f"{BASE}{path}", params=dict(params), timeout=30,
                     headers={"Authorization": f"Bearer {key}"})
    r.raise_for_status()
    return r.json()


def _cache_path(endpoint: str, year: int, week: int | None, cache: Path | None) -> Path:
    root = Path(cache or CACHE)
    stem = f"{year}" if week is None else f"{year}-w{week:02d}"
    return root / endpoint / f"{stem}.parquet"


def _capture_path(path: Path) -> Path:
    """The capture record sits beside its cache entry, the way `nflverse._pin_path` does.

    Beside rather than in a registry of its own, for the reason ADR-0006 keeps a fitted
    constant with its provenance: a record kept away from the thing it describes is one that
    stops matching it. Beside rather than *inside*, because the entry is a third-party
    payload this module re-validates against a declared contract, and a `captured_at` column
    would be a column CFBD never sent turning up in every frame the contract checks.
    """
    return path.with_suffix(".capture.json")


def _capture_beside(path: Path) -> datetime | None:
    """When the payload at `path` was fetched, or None where nothing on disk says.

    None is the honest answer three ways and they are all one answer: an entry written
    before this existed has no record beside it, an interrupted write leaves a file that is
    not JSON, and a hand-edited one can hold anything.

    The file's own mtime is deliberately not consulted, and that is #175's fourth criterion.
    A clone, a copy, a restore or a `touch` rewrites it, so it dates the *file* and not the
    fetch -- and a capture time that is confidently wrong is worse than one that is missing,
    because unknown means "ask again" and a fabricated one means "no need to". Every entry
    written before this change reads back unknown here, which is the truth about it.
    """
    try:
        raw = json.loads(_capture_path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("captured_at"), str):
        return None
    try:
        when = datetime.fromisoformat(raw["captured_at"])
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=UTC)


def _record_capture(path: Path) -> None:
    """Stamp a freshly written entry with the moment it was fetched.

    A failure here is swallowed on purpose: a payload that could not be stamped is still a
    payload, and taking a fetched week down over its sidecar would spend the call and throw
    the rows away. What it costs is an entry that reads back as unknown, which is the state
    the reader above is built for.
    """
    payload = {"captured_at": jsonio.stamp()}
    try:
        _capture_path(path).write_text(jsonio.dumps(payload, indent=2))
    except OSError:
        pass


def captured_at(endpoint: str, year: int, week: int | None = None, *,
                cache: Path | None = None) -> datetime | None:
    """When the cached payload for one key was fetched, or None where nothing says.

    Public because the age of the bytes is a question a caller has to be able to ask without
    fetching anything: `max_age` acts on it, and a report saying how old a served college
    price is needs the same number. Before #175 there was no way to ask at all -- the cache
    was permanent by file existence, so a price captured in September was indistinguishable
    in January from one captured that morning.
    """
    return _capture_beside(_cache_path(endpoint, year, week, cache))


def _past_its_age(captured: datetime | None, *, refresh: bool, max_age: timedelta | None,
                  now: datetime | None = None) -> bool:
    """Whether a cached entry has to be fetched again before it may be served.

    Three states rather than two. `refresh=True` is a caller saying so outright. `max_age`
    is a caller stating a bound the bytes have to be inside. Neither given, the entry stands
    however old it is -- the behaviour this module has always had, kept as the default on
    purpose: a completed season does not move, and a default bound would have the two
    scheduled runs re-fetching weeks they already hold against a budget of 1,000 a month.

    An entry of *unknown* age is past a stated bound. It cannot be shown to be inside one,
    and serving it would answer a question about age with a silence that reads as a yes.
    Nothing reaches this without a caller having asked for a bound first, and a refusal to
    spend serves the cache anyway -- so the worst case here is one call that was not needed,
    against a stale price that could never be corrected, which is what #175 was filed about.
    """
    if refresh:
        return True
    if max_age is None:
        return False
    if captured is None:
        return True
    return (now or datetime.now(UTC)) - captured > max_age


def _cannot_spend(key: str | None, quota_path: Path | None) -> Exception | None:
    """Why this call must not be made, or None if it may be.

    Returned rather than raised, and that is the whole of #175's third criterion. A refusal
    with a cached payload behind it and a refusal with nothing behind it are different
    events: the first is a degradation the caller should be told about and served through,
    the second has nothing to serve. Raising here made every refusal the second kind, which
    was harmless only while the cache could never be asked to refresh.
    """
    # GUARD run-ceiling-stops-a-loop [unit/test_fetch_cfbd.py]: a loop stops at call 13
    if _CALLS_THIS_RUN >= MAX_CALLS_PER_RUN:
        return QuotaExceeded(
            f"{MAX_CALLS_PER_RUN} calls in one run; a week costs 5-8. This is a loop.")
    # /GUARD
    if quota_used(quota_path) >= FREE_TIER_MONTHLY:
        return QuotaExceeded(
            f"monthly budget of {FREE_TIER_MONTHLY:,} is spent. Waiting beats a second key: "
            f"multiple keys are a terms violation that gets access revoked.")
    if not key:
        return LoopRefused("no CFBD_API_KEY set; cannot fetch")
    return None


def _describe(endpoint: str, year: int, week: int | None) -> str:
    """One cache key in words, for a line an operator reads."""
    return f"{endpoint} {year}" if week is None else f"{endpoint} {year} week {week}"


def bulk(endpoint: str, year: int, week: int | None = None, *,
         extra: Mapping[str, Any] | None = None,
         cache: Path | None = None, quota_path: Path | None = None,
         refresh: bool = False, max_age: timedelta | None = None) -> pl.DataFrame:
    """One bulk pull. There is no way to ask this for a single team.

    Note the signature: endpoint, year, week. A caller who wants Alabama pulls the week and
    filters in polars, because that is one call instead of one hundred and thirty-six.

    **The cache is no longer permanent by file existence** (#175). A written entry carries
    the moment it was captured, `refresh=True` re-fetches it outright, and `max_age` states
    a bound past which it is re-fetched rather than served. Neither given, the entry stands
    however old it is, which is what every existing caller gets and what the two scheduled
    runs still do.

    And a refusal serves rather than raises where there is something to serve: a refresh
    that would pass the run ceiling or the monthly budget, or one with no key to make it
    with, prints what it did and hands back the cached payload with its capture time -- or
    with the fact that its capture time is unknown, which is what an entry written before
    any of this reads back as.
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
    captured = _capture_beside(path)
    held = path.exists()
    if held and not _past_its_age(captured, refresh=refresh, max_age=max_age):
        return pl.read_parquet(path)

    key = _api_key()
    refused = _cannot_spend(key, quota_path)
    # GUARD a-refused-refresh-serves-the-cache [unit/test_fetch_cfbd.py]: refuse, and still
    # answer
    #
    # The refusal is the one this module always made; what changes is that there is now
    # something behind it. A refresh that cannot be afforded must not also lose the payload
    # already on disk -- CLAUDE.md's degradation rule, and the reason `make slate` marks
    # this source optional. And it says so out loud, because a served-stale payload that
    # reads like a fresh one is the whole of what #175 was filed about.
    if refused is not None and held:
        where = _describe(endpoint, year, week)
        age = ("captured " + captured.isoformat() if captured
               else "whose capture time is unknown")
        print(f"  cfbd: {where} was not refreshed: {refused} "
              f"Serving the cached payload, {age}.")
        return pl.read_parquet(path)
    # /GUARD
    if refused is not None:
        raise refused
    assert key is not None      # `_cannot_spend` refuses a missing key above

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
    _record_capture(path)
    return df


# Endpoint to contract. Neither of these has ever met a live response -- there was no CFBD
# key on the machine they were written on -- so both carry `verified_against_live=False` and
# a first failure should be read as "the declaration was a guess" before "the source broke".
CONTRACTS: dict[str, Contract] = {"games": CFBD_GAMES, "lines": CFBD_LINES}


def week(year: int, week_no: int, *, cache: Path | None = None,
         quota_path: Path | None = None, refresh: bool = False,
         max_age: timedelta | None = None) -> dict[str, pl.DataFrame]:
    """The documented weekly slate: games, lines, box scores. Three calls, not 408.

    `refresh` and `max_age` are handed straight to `bulk` and mean what they mean there.
    Three endpoints, so a refresh of a week costs three calls and not one -- which is why
    neither has a default that would make a scheduled run pay it.
    """
    out: dict[str, pl.DataFrame] = {}
    print(f"  CFBD week {week_no}, {year}")
    for endpoint in WEEKLY:
        df = bulk(endpoint, year, week_no, cache=cache, quota_path=quota_path,
                  refresh=refresh, max_age=max_age)
        # A registry rather than a branch, and the same shape `hub.fetch.nflverse` uses:
        # `box` has no declared contract and is not being given a weak one to fill the row.
        #
        # GUARD empty-week-is-still-checked [unit/test_fetch_cfbd.py]: an empty response is
        # validated with its row minimum relaxed, never skipped
        #
        # This read `if df.height and (contract := ...)` (#70). Both CFBD contracts declare
        # `min_rows=1`, so that made an empty college week not checked at all -- and those two
        # are the only contracts the provenance work constrains, so the skip took the check
        # off exactly the pair that motivated it.
        #
        # The reason for it was right. A week nobody has played yet answers `[]`, and
        # reporting that as "0 rows < min 1; missing columns [...]" reads as the source having
        # changed shape when it has said "nothing here" -- which is the fetched-and-empty
        # state `record_run` exists to distinguish, and it could not exist while emptiness
        # raised. What was wrong is where the fix went: skipping the contract also stops it
        # answering the question it is for.
        #
        # So "empty" is narrowed instead. A response with no rows *and no columns* is the
        # source saying nothing, and there is no shape in it to check -- `[]` parses to a
        # (0, 0) frame. A response with no rows and columns of its own is a different animal:
        # `{"data": []}` from a reshaped endpoint is also zero rows, and is a rename wearing
        # an empty week's clothes. That one is checked against everything the contract
        # declares except the row count, which is the only clause emptiness is allowed to
        # relax.
        if (contract := CONTRACTS.get(endpoint)) is not None:
            if not df.height:
                contract = replace(contract, min_rows=0)
            if df.width:
                df = contract.validate(df)  # the repaired frame is the one stored
        # /GUARD
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
               error: BaseException | None = None,
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
    That is why a failure arrives here as `error` -- the exception itself -- rather than as a
    sentence somebody wrote from it. See the guard below.
    """
    # GUARD status-carries-no-payload-text [unit/test_fetch_cfbd.py]: the type, never the
    # message
    #
    # #70: this used to be `why = f"...{type(e).__name__}: {e}"[:400]` at the call site,
    # and an exception's message is an open channel with nothing bounding what a third
    # party can put through it. A `ContractViolation` quotes the values that broke the
    # contract: measured 2026-09-05, a planted bad frame produced `week range [99, 99]
    # outside [1, 20]; homePoints range [131, 131] outside [0, 120]` -- four payload
    # values and a column name, into a file the slate commits to a repo intended to go
    # public. Truncating bounded how much escaped and left it rows all the same.
    #
    # So the exception arrives here whole and only its *type* is recorded. A type name
    # comes out of this repo or its dependencies; it can say `ContractViolation` rather
    # than `ConnectionError` without saying anything a source sent us. The message is not
    # lost -- `main` prints it to stderr, which is a log and not a commit.
    #
    # An HTTP status is the one exception to "type only", and it is one because it is a
    # count: three digits from the protocol, not a field of anybody's payload. It is also the
    # difference between "the source is down" and "the key is wrong" without opening a log,
    # which is most of what this file gets read for.
    if error is not None:
        status = getattr(getattr(error, "response", None), "status_code", None)
        code = f" (HTTP {status})" if isinstance(status, int) else ""
        why = (f"nothing was fetched: {type(error).__name__}{code}. Its message is on stderr "
               f"and deliberately not here: a contract violation quotes the payload values "
               f"that broke it, and this file is committed")
    # /GUARD
    fetched = rows is not None
    counts = dict(rows or {})
    summary = ", ".join(f"{k} {v:,} rows" for k, v in counts.items())
    if not fetched:
        stale, reason = True, (why or "nothing was fetched")
    elif not sum(counts.values()):
        stale, reason = True, (f"week {week_no} of {season} was read and came back empty")
    else:
        stale, reason = False, None
    # Through `jsonio.summary`, which stamps `shape: "summary"`. This envelope reports what
    # was fetched without carrying it -- `rows_by_endpoint` is counts, not rows -- and saying
    # so here is what #227 replaced `NOT_ROW_SHAPED` with. The first publish of this file, by
    # the scheduled slate on 2026-09-09, failed the row-count contract on the next human push
    # ten commits later, for an artifact that was correct and simply not on a list.
    got: dict[str, Any] = jsonio.summary(
        "cfbd", "hub.fetch.cfbd",
        season=season, week=week_no, fetched=fetched,
        stale=stale, reason=reason,
        rows_by_endpoint=counts,
        quota={"month": _month_key(), "used": quota_used(quota_path),
               "limit": FREE_TIER_MONTHLY},
    )
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
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch this week even if it is cached; spends quota")
    ap.add_argument("--max-age-hours", type=float, default=None,
                    help="re-fetch a cached week captured longer ago than this, and one "
                         "whose capture time is unknown; unset, a cached week is served "
                         "whatever its age")
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
        got = week(a.year, week_no, quota_path=qpath, refresh=a.refresh,
                   max_age=(timedelta(hours=a.max_age_hours)
                            if a.max_age_hours is not None else None))
    except Exception as e:
        # Caught rather than raised, because the traceback goes to the same place the exit
        # code does -- nowhere. An optional source that is down must not halt the slate
        # (CLAUDE.md's degradation rule, and the leading `-` in the Makefile), and it must
        # not be indistinguishable from one that succeeded either.
        #
        # The exception goes to `record_run` whole and is redacted there, rather than being
        # rendered into a sentence here: the file it lands in is committed, and `{e}` was how
        # a contract violation's quoted payload values got into it (#70). Stderr gets the
        # message in full, because a log is not a commit and the operator has to be able to
        # read what actually broke.
        record_run(a.year, week_no, error=e, path=spath, quota_path=qpath)
        print(f"hub.fetch.cfbd: nothing was fetched: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    # A week that answered, whether or not it held anything. `record_run` tells those two
    # apart; the exit code does not try to, because the fetch happened either way.
    record_run(a.year, week_no, rows={k: v.height for k, v in got.items()},
               path=spath, quota_path=qpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())
