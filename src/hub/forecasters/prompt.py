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
