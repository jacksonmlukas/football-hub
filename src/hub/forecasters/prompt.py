"""The fixed prompt, one per condition, and the hash `prereg-llm-v0` pins.

The prompt hash covers the *template* -- the system text and the frame around the packet -- not
a rendered game, so it is one number per condition for the season and changes only when the
wording does. `PINNED` holds those numbers as literals: editing the wording fails
`test_the_prompt_hash_is_stable_and_pinned` until the pin is updated on purpose, and after v0
that update is a deviations-log row.

The packet is shown as JSON with its `stamp` block, `run_start` and schema bookkeeping removed:
fetch times are the record's evidence for the as-of rule, not information about the game, and
the odds snapshot's id and time are the record's too. What remains is exactly the prereg's
field table for the condition.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from hub.forecasters.packet import Condition, Packet, plain

_COMMON = (
    "You are forecasting the outcome of one NFL regular-season game. You have no tools, no web "
    "access and no information beyond what is given here and what you already know. Give the "
    "probability that the HOME team wins. Ties are excluded from scoring, so treat the outcome "
    "as home win or away win.\n\n"
    "Answer with a single JSON object and nothing else, in exactly this form:\n"
    '{"p_home_win": <a number between 0 and 1>, "rationale": "<at most 60 words>"}'
)

SYSTEM: dict[Condition, str] = {
    Condition.BLIND: _COMMON,
    Condition.LINE_AWARE: _COMMON + (
        "\n\nThe packet includes the betting market's de-vigged home win probability and the "
        "home point spread, taken this morning. Use them however you judge best."),
}

USER_TEMPLATE = ("Game information (all of it known before this forecast was requested):"
                 "\n\n{packet}\n")

_HIDDEN = ("stamp", "run_start", "schema_version", "condition")
_HIDDEN_MARKET = ("snapshot_id", "snapshot_time")


def _shown(p: Packet) -> dict[str, Any]:
    d = plain(p)
    for k in _HIDDEN:
        d.pop(k, None)
    if d.get("market") is None:
        d.pop("market", None)
    else:
        d["market"] = {k: v for k, v in d["market"].items() if k not in _HIDDEN_MARKET}
    return d


def render(p: Packet) -> str:
    """The user message for one packet: deterministic for a given packet."""
    return USER_TEMPLATE.format(packet=json.dumps(_shown(p), sort_keys=True, indent=2,
                                                  default=_iso))


def _iso(o: Any) -> str:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"{type(o).__name__} in a packet is not shown to a model")


def prompt_hash(condition: Condition) -> str:
    raw = f"{SYSTEM[Condition(condition)]}\x00{USER_TEMPLATE}"
    return hashlib.sha256(raw.encode()).hexdigest()


PINNED: dict[Condition, str] = {
    Condition.BLIND: "358a2c0e625fea211f9ef39f8efcacbf75318e269e4a96110bc405b3509adbc6",
    Condition.LINE_AWARE: "d8f7a09df3141f9c0d60d2c610aa23dab3bfc920cd98332bba7d3cf14c57010d",
}


# --- the retrieval arm (#511) ---------------------------------------------------------------
#
# The paired arm sees its bare twin's line-aware packet rendered exactly as `render` renders it,
# followed by the news items. Only the opening sentence of the system text differs -- the bare
# arms are told they have nothing beyond the packet, which is not true here -- and the answer
# form and the betting-market sentence are the line-aware text's, character for character.

_BARE_OPENING = ("You have no tools, no web access and no information beyond what is given "
                 "here and what you already know.")
_RETRIEVAL_OPENING = (
    "You have no tools and no web access of your own. Besides what you already know, you have "
    "what is given here: the game information and a set of news search results retrieved for "
    "this game before this forecast was requested, each with its title, address and "
    "publication time. Treat them as reports, which can be wrong or out of date.")
if _BARE_OPENING not in SYSTEM[Condition.LINE_AWARE]:      # a reworded bare text, not a no-op
    raise RuntimeError(  # pragma: no cover - import time; the pin test reads the result
        "the retrieval prompt's opening no longer matches the bare text's")
RETRIEVAL_SYSTEM = SYSTEM[Condition.LINE_AWARE].replace(_BARE_OPENING, _RETRIEVAL_OPENING)

RETRIEVAL_USER_TEMPLATE = (USER_TEMPLATE + "\nNews search results (each published before this "
                           "forecast was requested):\n\n{news}\n")

_NEWS_FIELDS = ("published_at", "title", "url", "snippet")


def render_retrieved(base: Packet, items: Sequence[Any]) -> str:
    """The retrieval arm's user message: the bare packet as `render` shows it, then the items
    (newest first, as retrieved), each with only what a reader of the story would see."""
    news = [{k: getattr(i, k) for k in _NEWS_FIELDS} for i in items]
    return RETRIEVAL_USER_TEMPLATE.format(
        packet=json.dumps(_shown(base), sort_keys=True, indent=2, default=_iso),
        news=json.dumps(news, sort_keys=True, indent=2, default=_iso))


def retrieval_prompt_hash() -> str:
    return hashlib.sha256(f"{RETRIEVAL_SYSTEM}\x00{RETRIEVAL_USER_TEMPLATE}".encode()).hexdigest()


PINNED_RETRIEVAL = "ad1701c295e19f7fe4ef4ceefaa0147220b53c4178a2f3500f50cb2941265ac5"
