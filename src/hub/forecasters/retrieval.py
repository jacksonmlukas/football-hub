"""The retrieval arm: one named LLM, its own line-aware packet, plus a news search per game (#511).

`docs/prereg/llm-forecasters.md`, *Contestants*, entry of 2026-10-22: the arm is **paired**.
It is the same model, with the same settings, as one named bare arm (`RetrievalSpec.paired_with`)
and it sees that arm's line-aware packet byte for byte, plus the news items this module
retrieved. So the paired difference -- retrieval minus bare, game by game -- isolates
retrieval, and the record says which arm it differences against (`paired_with`) and carries
the bare packet's hash (`base_packet_hash`) so scoring can check the pair saw the same packet.

**The queries are fixed**, two per team, in `QUERY_TEMPLATES`, run in order home then away:
an injury report and the inactives (#511's agent half). Beat-writer notes arrive through the
same news results; there is no free-text query a run could vary.

**The search cap is `SEARCHES_PER_GAME = 4`**: the two fixed queries for each of two teams, the
middle of #511's 3-5 planning figure. At one credit a basic search that is ~64 credits for a
16-game week, ~650 for weeks 9-18 (burn-in included) -- under a tenth of the ~9k promo
balance, so a second paired model could be added without a new budget. A query list longer
than the cap is refused before a search is made.

**The point-in-time rule** (the injury report's, `sources.py`, applied to news). Every search
runs during the run, so fetch time certifies nothing; each item is judged by its *own*
`published_date`: kept only if it strictly precedes `run_start` and is no older than
`LOOKBACK_DAYS` before it. An item with no usable date (absent, unparseable, no timezone) is
dropped, never dated by fetch time. Every dropped item is logged with its URL, title, raw
date and reason, and counted; the rule is written into the block's `basis`, as the injury
report's is into `published_at_basis`. `validate_retrieved` re-applies it to a finished packet.

**The token budget.** At most `MAX_ITEMS` items, newest first, each snippet cut to
`SNIPPET_CHARS` characters; items cut by the budget are logged as dropped (`over_budget`).
`PROMPT_EXTRA_TOKENS` is the spending guard's upper bound on what that adds to a prompt.

**Credits are counted apart from dollars.** The promo's credits are not dollars, so the
season's Tavily spend is a credit count in the study's own state file (`CREDITS_STATE` under
the `forecasts` checkout, written after every search, so a crash cannot lose one), checked
against `RetrievalSpec.season_credit_cap` before each game's searches. Credits are also
priced at `RetrievalSpec.usd_per_credit` into each record's `cost_usd`, so the dollar season
cap sees retrieval once the promo is gone. Running out of credits stops the arm, not the run:
the bare arms are the confirmatory ones.

**The hash.** A `RetrievedPacket` is the bare packet and the retrieval block; its
`canonical_hash` is the record's `packet_hash`, so every snippet the model saw is hashed with
the other inputs, and the snippets themselves are committed with the record.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from hub import atomic
from hub.declare import not_an_input
from hub.fetch import tavily
from hub.forecasters.inputs import SourceStamp
from hub.forecasters.packet import (
    Condition,
    Failure,
    Packet,
    PacketRefused,
    canonical_hash,
    plain,
    validate_packet,
)

ARM = "retrieval"
SOURCE = "tavily:search(topic=news)"
CREDITS_STATE = Path("state") / "tavily.json"

QUERY_TEMPLATES: tuple[str, ...] = ("{team} injury report", "{team} inactives")

# Declared `not_an_input`, not `chosen`: `chosen` is covered by the hub's `config_digest`, so
# these six would stamp every published hub prediction with a new model version (`839ceda9` ->
# `c4405e1e`, measured 2026-10-22) for numbers no hub prediction reads. They are the study's
# settings, fixed by the prereg's 2026-10-22 entry and recorded in every retrieval record.
SEARCHES_PER_GAME = not_an_input(4, "the LLM study's search cap a game, fixed by the prereg; "
                                     "no hub prediction reads it")
RESULTS_PER_SEARCH = not_an_input(5, "the LLM study's results asked of one search, fixed by the "
                                     "prereg; no hub prediction reads it")
LOOKBACK_DAYS = not_an_input(7, "the LLM study's oldest admissible news item, in days before "
                                "run start; no hub prediction reads it")
MAX_ITEMS = not_an_input(10, "the LLM study's news items shown to the retrieval arm, fixed by "
                             "the prereg; no hub prediction reads it")
SNIPPET_CHARS = not_an_input(600, "the LLM study's characters kept of one news snippet, fixed "
                                  "by the prereg; no hub prediction reads it")
# The guard's upper bound on the news block's tokens: 10 items x (600 snippet chars + title,
# URL and date) is under 9,000 characters, ~2,300 tokens at four characters a token.
PROMPT_EXTRA_TOKENS = not_an_input(2500, "the LLM study's spending-guard ceiling; no hub "
                                         "prediction reads it")

TIME_BASIS = (f"tavily: each result's own `published_date`, kept only if strictly before run "
              f"start and no more than {LOOKBACK_DAYS} days before it; a result with no "
              f"parseable, zoned date, or dated at or after run start, is dropped, logged and "
              f"counted (fetch time never stands in)")


class CreditsExhausted(RuntimeError):
    """The season's credit cap cannot cover one more game's searches. Nothing was searched."""


class RetrievalFailed(RuntimeError):
    """This game's retrieval could not be completed; the arm has no forecast for it. `credits`
    is what was spent before the failure."""

    def __init__(self, reason: str, credits: int = 0) -> None:
        super().__init__(reason)
        self.credits = credits


@dataclass(frozen=True)
class Item:
    query: str
    url: str
    title: str
    published_at: datetime
    snippet: str


@dataclass(frozen=True)
class Dropped:
    query: str
    url: str
    title: str
    published_raw: str | None
    reason: str          # "undated" | "at_or_after_run_start" | "older_than_lookback" | ...


@dataclass(frozen=True)
class RetrievalBlock:
    published_before: datetime          # the run start: every item strictly precedes it
    basis: str
    queries: tuple[str, ...]
    searches: int
    credits: int
    items: tuple[Item, ...]
    dropped: tuple[Dropped, ...]
    stamp: SourceStamp


@dataclass(frozen=True)
class RetrievedPacket:
    base: Packet                        # the paired bare arm's line-aware packet, unchanged
    retrieval: RetrievalBlock


def retrieved_hash(rp: RetrievedPacket) -> str:
    return canonical_hash(rp)


# --- the queries and the point-in-time rule ------------------------------------------------

def queries_for(packet: Packet, full_names: Mapping[str, str]) -> list[str]:
    """The fixed queries for one game, home team first. `full_names` maps an abbreviation to
    the name a news search finds (an abbreviation missing from it is searched as is)."""
    teams = (packet.game.home_team, packet.game.away_team)
    return [t.format(team=full_names.get(team, team)) for team in teams for t in QUERY_TEMPLATES]


def _snippet(text: str) -> str:
    return " ".join(text.split())[:SNIPPET_CHARS]


def admit(searches: Sequence[tavily.Search],
          run_start: datetime) -> tuple[tuple[Item, ...], tuple[Dropped, ...]]:
    """Every hit judged by its own publication time (module doc); then deduplicated by URL
    (first query wins) and cut to the budget, newest first."""
    oldest = run_start - timedelta(days=LOOKBACK_DAYS)
    kept: dict[str, Item] = {}
    dropped: list[Dropped] = []
    for s in searches:
        for h in s.hits:
            when = tavily.published_at(h.published_raw)
            if when is None:
                reason: str | None = "undated"
            elif when >= run_start:
                reason = "at_or_after_run_start"
            elif when < oldest:
                reason = "older_than_lookback"
            else:
                reason = "duplicate_url" if h.url in kept else None
            if reason is not None or when is None:
                dropped.append(Dropped(s.query, h.url, h.title, h.published_raw,
                                       reason or "undated"))
                continue
            kept[h.url] = Item(s.query, h.url, h.title, when.astimezone(UTC),
                               _snippet(h.content))
    ranked = sorted(kept.values(), key=lambda i: (-i.published_at.timestamp(), i.url))
    for i in ranked[MAX_ITEMS:]:
        dropped.append(Dropped(i.query, i.url, i.title, i.published_at.isoformat(),
                               "over_budget"))
    return tuple(ranked[:MAX_ITEMS]), tuple(dropped)


def validate_retrieved(rp: RetrievedPacket, *, snapshot_id: str | None) -> None:
    """The second door: the bare packet's own checks, then every retrieved item against the
    run start. Raises `PacketRefused` with every failure at once."""
    out: list[Failure] = []
    try:
        validate_packet(rp.base, snapshot_id=snapshot_id)
    except PacketRefused as exc:
        out.extend(exc.failures)
    b, rs = rp.retrieval, rp.base.run_start
    if rp.base.condition != Condition.LINE_AWARE:
        out.append(Failure("retrieval_condition", "the retrieval arm extends a line_aware "
                                                  f"packet, not {rp.base.condition!r}"))
    if b.published_before != rs:
        out.append(Failure("retrieval_cutoff", f"published_before={b.published_before} is not "
                                               f"the packet's run_start {rs.isoformat()}"))
    if len(b.queries) > SEARCHES_PER_GAME or b.searches > SEARCHES_PER_GAME:
        out.append(Failure("retrieval_cap", f"{max(len(b.queries), b.searches)} searches "
                                            f"exceed the cap of {SEARCHES_PER_GAME}"))
    oldest = rs - timedelta(days=LOOKBACK_DAYS)
    for i in b.items:
        t = i.published_at
        if t.tzinfo is None or not oldest <= t < rs:
            out.append(Failure("retrieval_published", f"item {i.url} published_at={t} is not "
                                                      f"in [{oldest.isoformat()}, "
                                                      f"{rs.isoformat()})"))
    if out:
        raise PacketRefused(out)


# --- credits -------------------------------------------------------------------------------

def credits_used(path: Path) -> int:
    """Credits spent so far this season, from the study's own state file (0 with none)."""
    try:
        return int(json.loads(path.read_text()).get("credits_used", 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def _spend(path: Path, credits: int, *, now: datetime) -> None:
    try:
        old = json.loads(path.read_text())
        old = old if isinstance(old, dict) else {}
    except (OSError, ValueError):
        old = {}
    state = {"credits_used": int(old.get("credits_used", 0)) + credits,
             "searches": int(old.get("searches", 0)) + 1, "updated_at": now.isoformat()}
    atomic.write_text(path, json.dumps(state, sort_keys=True, indent=1) + "\n")


class Searcher(Protocol):
    def search(self, query: str, *, days: int, max_results: int) -> tavily.Search: ...


class Retriever:
    """Searches one game's fixed queries as of `run_start`, under the season's credit cap."""

    def __init__(self, client: Searcher, *, full_names: Mapping[str, str], state_path: Path,
                 season_credit_cap: int, clock: Callable[[], datetime]) -> None:
        self._client, self._names, self._state = client, full_names, state_path
        self._cap, self._clock = season_credit_cap, clock
        self.exhausted = False

    def retrieve(self, base: Packet, *, snapshot_id: str | None) -> RetrievedPacket:
        queries = queries_for(base, self._names)
        # GUARD retrieval-search-cap [unit/test_forecasters_retrieval_cap.py]: a game never gets
        # more searches than the declared cap
        if len(queries) > SEARCHES_PER_GAME:
            raise RetrievalFailed(f"{len(queries)} queries exceed the cap of "
                                  f"{SEARCHES_PER_GAME} searches a game; none was run")
        # /GUARD
        worst = len(queries) * tavily.CREDITS_PER_SEARCH
        if credits_used(self._state) + worst > self._cap:
            self.exhausted = True
            raise CreditsExhausted(f"{credits_used(self._state)} of {self._cap} Tavily credits "
                                   f"used; one more game needs {worst}")
        searches: list[tavily.Search] = []
        spent = 0
        for q in queries:
            try:
                s = self._client.search(q, days=LOOKBACK_DAYS, max_results=RESULTS_PER_SEARCH)
            except tavily.TavilyError as exc:
                raise RetrievalFailed(f"search {len(searches) + 1} of {len(queries)} failed: "
                                      f"{exc}", spent) from None
            _spend(self._state, s.credits, now=self._clock())
            spent += s.credits
            searches.append(s)
        items, dropped = admit(searches, base.run_start)
        rp = RetrievedPacket(base, RetrievalBlock(
            published_before=base.run_start, basis=TIME_BASIS, queries=tuple(queries),
            searches=len(searches), credits=spent, items=items, dropped=dropped,
            stamp=SourceStamp(SOURCE, self._clock().astimezone(UTC))))
        try:
            validate_retrieved(rp, snapshot_id=snapshot_id)
        except PacketRefused as exc:
            raise RetrievalFailed(f"retrieved packet refused: {exc}", spent) from None
        return rp


def plain_block(b: RetrievalBlock) -> dict[str, Any]:
    """The block as the record commits it: every query, item, drop and count."""
    return plain(b)
