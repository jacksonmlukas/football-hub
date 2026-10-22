"""The study's market benchmark: a de-vigged moneyline from the hub's own feed.

`docs/prereg/llm-forecasters.md`, *Market benchmark*: the same odds feed as the hub's market
prior -- The Odds API, `hub.fetch.odds` -- snapshotted in the forecast run itself, so the benchmark
and the models see one moment. Each book is de-vigged on its own (proportional: decimal odds,
implied probabilities normalised to sum to 1), *then* the median home probability is taken
across books; taking each side's median first can pair prices no single book offered.

**A separate pull, not a third hub market (maintainer, 2026-10-06, option (b)).** The hub's
`MARKETS` stay spreads and totals and its polls are untouched. This module declares its own
jobs and their markets, refuses anything else before a request is formed, and keeps its own
floor -- but writes the *same* balance file, because the credits are one key's, and the hub's
`CREDIT_FLOOR` is sized so the hub's polls stop before this module's do.

The forecast pull asks for spreads as well as the moneyline, because a line-aware packet
carries the spread from *this run's* snapshot (the as-of rule), and a spread from the hub's
last poll would be another moment. The closing pull needs the moneyline only.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hub.fetch import odds
from hub.forecasters.inputs import MarketSnapshot

SOURCE = "the-odds-api v4 /sports/americanfootball_nfl/odds"
H2H = "h2h"

# What each job may ask for, and so what one call costs (markets x one region).
JOBS: dict[str, tuple[str, ...]] = {"forecast": (H2H, odds.MARKET), "closing": (H2H,)}

# This module refuses below this; the hub's polls refuse below `odds.CREDIT_FLOOR`.
STUDY_FLOOR = 10

# A worst-case calendar month of this module's pulls, which the hub's floor has to leave room
# for above `STUDY_FLOOR`. Worst week: five forecast run days (Thursday, a Saturday slate, a
# Sunday with an international morning game moved to Saturday's run, Sunday, Monday -- or a
# Thanksgiving Wednesday) at 2 credits, plus ten distinct kickoff times at 1 credit for the
# closing line = 20; a 31-day month of such weeks is 20 x 31 / 7 = 88.6.
WORST_MONTH_CREDITS = 89


def devig(home: float, away: float) -> tuple[float, float]:
    """One book's two decimal prices, proportionally de-vigged: (p_home, p_away), summing to 1."""
    for price in (home, away):
        if not (math.isfinite(price) and price > 1.0):
            raise ValueError(f"{price!r} is not a decimal price (finite, above 1)")
    inv_home, inv_away = 1.0 / home, 1.0 / away
    total = inv_home + inv_away
    return inv_home / total, inv_away / total


@dataclass(frozen=True)
class Quote:
    """One game's market: the median de-vigged home probability across `books`, and the
    median home spread across the books quoting one (None if none does)."""
    home_win_prob: float
    home_spread: float | None
    books: int


def _market(book: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    return next((m for m in book.get("markets") or [] if m.get("key") == key), None)


def _book_home_prob(book: Mapping[str, Any], home: str, away: str) -> float | None:
    """A book's de-vigged home probability, or None if its moneyline is not exactly the two
    teams (a missing side or a third outcome is skipped, not guessed at)."""
    m = _market(book, H2H)
    outcomes = (m or {}).get("outcomes") or []
    prices = {o.get("name"): o.get("price") for o in outcomes}
    if len(outcomes) != 2 or set(prices) != {home, away}:
        return None
    try:
        return devig(float(prices[home]), float(prices[away]))[0]
    except (TypeError, ValueError):
        return None


def _book_home_spread(book: Mapping[str, Any], home: str) -> float | None:
    m = _market(book, odds.MARKET)
    for o in (m or {}).get("outcomes") or []:
        if o.get("name") == home and o.get("point") is not None:
            try:
                return float(o["point"])
            except (TypeError, ValueError):
                return None
    return None


def quote(event: Mapping[str, Any]) -> Quote | None:
    """The game's market, or None when no book quotes a usable moneyline (a missing line,
    which the prereg leaves out of every comparison against the benchmark)."""
    home, away = event.get("home_team", ""), event.get("away_team", "")
    books = event.get("bookmakers") or []
    probs = [p for b in books if (p := _book_home_prob(b, home, away)) is not None]
    if not probs:
        return None
    spreads = [s for b in books if (s := _book_home_spread(b, home)) is not None]
    return Quote(home_win_prob=statistics.median(probs),
                 home_spread=statistics.median(spreads) if spreads else None,
                 books=len(probs))


@dataclass(frozen=True)
class Snapshot:
    """One pull: its id, when it was taken, the quote per matched game, and the payload's hash
    (the raw payload is published only as this hash; see the prereg's *Reporting*)."""
    snapshot_id: str
    captured_at: datetime
    job: str
    raw_sha256: str
    quotes: Mapping[str, Quote]
    unmatched: tuple[str, ...] = ()
    remaining: int | None = None
    raw: bytes = field(default=b"", repr=False)

    def for_packet(self, game_id: str) -> MarketSnapshot | None:
        """The packet's market input for a game, or None if it has no moneyline or no spread
        (a line-aware packet needs both, from this snapshot)."""
        q = self.quotes.get(game_id)
        if q is None or q.home_spread is None:
            return None
        return MarketSnapshot(snapshot_id=self.snapshot_id, snapshot_time=self.captured_at,
                              source=SOURCE, fetched_at=self.captured_at,
                              home_win_prob=q.home_win_prob, home_spread=q.home_spread)


Get = Callable[[Mapping[str, Any], str], tuple[Any, Mapping[str, str]]]


def _default_match(season: int) -> Callable[[Mapping[str, Any]], str | None]:  # pragma: no cover
    index = odds._game_index(season)

    def match(ev: Mapping[str, Any]) -> str | None:
        hit = odds._match_game(ev, index)
        return hit[0] if hit else None
    return match


def snapshot(job: str, *, now: datetime | None = None, season: int | None = None,
             get: Get | None = None, key: str | None = None, state_path: Path | None = None,
             match: Callable[[Mapping[str, Any]], str | None] | None = None) -> Snapshot:
    """One call for every upcoming game, under `job`'s declared markets and this module's floor."""
    # GUARD study-jobs-only: an undeclared job (and so an undeclared market) spends nothing
    if job not in JOBS:
        raise ValueError(f"unknown job {job!r}; declared: {sorted(JOBS)}")
    # /GUARD
    markets = ",".join(JOBS[job])
    declared_cost = len(JOBS[job]) * len(odds.REGIONS)
    when = (now or datetime.now(UTC)).astimezone(UTC)
    naive = when.replace(tzinfo=None)                    # the shared state file's convention

    have = odds.known_balance(state_path, naive, floor=STUDY_FLOOR)
    # GUARD study-floor-refuses: below the study's own floor, nothing is spent
    if have is not None and have < STUDY_FLOOR:
        raise odds.QuotaFloor(f"{have} credits left, study floor is {STUDY_FLOOR}. "
                              f"Refusing before spending one.")
    # /GUARD
    key = key or odds._api_key()
    if not key:
        raise odds.QuotaFloor("no ODDS_API_KEY set; cannot fetch")

    try:
        payload, headers = (get or odds._http_get)(
            {"markets": markets, "regions": odds.REGION, "oddsFormat": "decimal"}, key)
    except Exception as exc:
        raise odds.MarketUnreachable(
            f"{type(exc).__name__}: {odds.redact_key(str(exc), key)}") from None

    hdr = headers.get("x-requests-remaining")
    remaining = int(float(hdr)) if hdr is not None else None
    cost = have - remaining if have is not None and remaining is not None and have >= remaining \
        else None
    odds._write_state(state_path, remaining, naive, cost=cost, declared=declared_cost,
                      asked=f"study:{job}:{markets}")

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    matcher = match or _default_match(season if season is not None else when.year)
    quotes: dict[str, Quote] = {}
    unmatched: list[str] = []
    for ev in payload or []:
        gid = matcher(ev)
        if gid is None:
            unmatched.append(str(ev.get("id")))
            continue
        q = quote(ev)
        if q is not None:
            quotes[gid] = q
    return Snapshot(snapshot_id=f"{when:%Y%m%dT%H%M%SZ}-{digest[:12]}", captured_at=when,
                    job=job, raw_sha256=digest, quotes=quotes, unmatched=tuple(unmatched),
                    remaining=remaining, raw=raw)
