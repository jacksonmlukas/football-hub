"""Tavily's news search: the one adapter the LLM study's retrieval arm reads (#511).

`docs/prereg/llm-forecasters.md`, *Contestants* (2026-10-22 entry): one LLM is paired with
itself plus a news search per game at forecast time. This module turns one query into one
POST to `https://api.tavily.com/search` and the reply into `Hit`s -- the URL, title, snippet
and the raw `published_date` -- and nothing else. Which queries run, how many, and which hits
a forecast may see are the study's rules and live in `hub.forecasters.retrieval`.

**The key.** `KEY_ENV` (`TAVILY_API_KEY`) is read once, when the client is built, and an
absent or empty value refuses in one sentence (`MissingKey`) before anything is sent. It is a
repository secret mapped into the forecast step only (`.github/workflows/forecast.yml`), never
`.env`, so an agent lane cannot spend. It travels in the `Authorization` header Tavily's API
takes, never in a URL or a log line; a failed call becomes a `TavilyError` whose text this
module writes itself -- the exception's type name or the HTTP status -- never the transport's
own message, which can quote the request it failed on.

**One transport seam.** `Transport` is `(url, headers, body) -> decoded JSON`, the same shape
as `hub.forecasters.providers`; the default calls `cached.refuse_live_call` first, so a test
that forgot to stub it fails before a credit is spent rather than spending one.

**Credits.** A basic-depth search is one Tavily credit. The reply's own `usage.credits` is
taken where Tavily reports it (`include_usage`), and the declared `CREDITS_PER_SEARCH`
otherwise, so the study's credit count is a measurement where one exists.

**The payload shape is hand-built**, as the Odds API's was: no key exists on any machine an
agent runs on, so the parser is held against the shape Tavily documents, not a capture.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Any

from hub.declare import not_an_input
from hub.fetch import cached

KEY_ENV = "TAVILY_API_KEY"
URL = "https://api.tavily.com/search"

# "basic" is one credit a search ("advanced" is two); the planning figure in #511 assumed it.
SEARCH_DEPTH = "basic"
CREDITS_PER_SEARCH = not_an_input(1, "Tavily's price of one basic search, counted against the "
                                      "LLM study's credit cap; no hub prediction reads it")

# Seconds one search may take: the run's 30-minute budget is shared with every model call.
REQUEST_TIMEOUT_S = 30

Transport = Callable[[str, Mapping[str, str], Mapping[str, Any]], Mapping[str, Any]]


class MissingKey(RuntimeError):
    """`TAVILY_API_KEY` is not in the environment. Nothing was sent."""


class TavilyError(RuntimeError):
    """One search failed. The text is this module's own, never the transport's."""


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    content: str
    published_raw: str | None          # exactly as Tavily sent it; parsed by `published_at`


@dataclass(frozen=True)
class Search:
    query: str
    hits: tuple[Hit, ...]
    credits: int


def published_at(raw: str | None) -> datetime | None:
    """A hit's publication instant, or None when there is none to trust: absent, unparseable,
    or without a timezone (a naive time could be any of 24 hours). Tavily's news topic sends
    RFC 2822 (`Sun, 19 Oct 2026 14:00:00 GMT`); ISO 8601 is accepted too."""
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    try:
        t = parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        try:
            t = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return t if t.tzinfo is not None else None


def _post(url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> Mapping[str, Any]:
    cached.refuse_live_call("would search Tavily (one credit a search)", patch="transport=",
                            tests="tests/unit/test_forecasters_retrieval.py")
    return _send(url, headers, body)


def _send(url: str, headers: Mapping[str, str],
          body: Mapping[str, Any]) -> Mapping[str, Any]:            # pragma: no cover - network
    import requests
    r = requests.post(url, headers=dict(headers), json=dict(body), timeout=REQUEST_TIMEOUT_S)
    if r.status_code >= 400:
        raise TavilyError(f"tavily: HTTP {r.status_code}")
    return r.json()


def _int(v: Any, default: int) -> int:
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return n if n >= 0 else default


class Client:
    """One search per `search` call. Built -- and so keyed -- before anything is fetched."""

    def __init__(self, *, env: Mapping[str, str] | None = None,
                 transport: Transport | None = None) -> None:
        value = (os.environ if env is None else env).get(KEY_ENV, "")
        if not value.strip():
            raise MissingKey(f"{KEY_ENV} is not set, so the retrieval arm cannot search; add "
                             f"it as a repository secret (never to .env).")
        self._key = value
        self._transport = transport or _post

    def search(self, query: str, *, days: int, max_results: int) -> Search:
        body = {"query": query, "topic": "news", "search_depth": SEARCH_DEPTH, "days": days,
                "max_results": max_results, "include_answer": False,
                "include_raw_content": False, "include_images": False, "include_usage": True}
        headers = {"authorization": f"Bearer {self._key}", "content-type": "application/json"}
        try:
            data = self._transport(URL, headers, body)
        except TavilyError:
            raise
        except Exception as exc:
            # Dropped on purpose: a transport's message can quote the request, header included.
            raise TavilyError(f"tavily: {type(exc).__name__}") from None
        try:
            hits = tuple(Hit(url=str(r["url"]), title=str(r.get("title") or ""),
                             content=str(r.get("content") or ""),
                             published_raw=None if r.get("published_date") is None
                             else str(r["published_date"]))
                         for r in data["results"])
        except (AttributeError, KeyError, TypeError) as exc:
            raise TavilyError(f"tavily: unreadable reply ({type(exc).__name__})") from None
        usage = data.get("usage") if isinstance(data.get("usage"), Mapping) else {}
        return Search(query, hits, _int((usage or {}).get("credits"), CREDITS_PER_SEARCH))
