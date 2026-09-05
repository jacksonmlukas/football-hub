"""JSON that a browser can actually parse.

Python's `json` emits bare `NaN` and `Infinity` by default and reads them back happily.
**JSON has no such literals and neither does JavaScript**, so `JSON.parse` throws on the whole
document -- not on the offending field, on the document.

Found 2026-08-29 by opening `site/index.html`: `draft_board.json` carried 135 `NaN`s and the
page rendered "No draft board. Run `make draft`" against a present, fresh, 199KB artifact. That
artifact is what `docs/draft-night.md` names as the **last-resort fallback** for draft night --
"serve `site/data/draft_board.json`, the top 300 by ECR, which covers all 192 picks" -- so the
safety net was unreadable by the only thing that reads it.

A leaf: it imports `json` and `math` and nothing from `hub`, so the two writers that need it --
`hub.publish` and `hub.draft.board` -- can share one implementation without either importing
the other.

`artifact` is here for the same reason and arrived the same way. `site/data/live.json` had two
writers with two documents: the site writer produced the envelope every panel reads, the poller
produced `{ts, games, detail}`, and the page reads the first. So the poller -- the thing a
Sunday exists to run -- would have overwritten the file with something the page cannot read,
and the dashboard worked only because nothing was polling. Neither writer can import the other
without a cycle, which is precisely the situation this module was created for.
"""
from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from typing import Any


def finite(value: Any) -> Any:
    """`NaN` and `Infinity` become `null`, at any depth, leaving everything else alone."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return value


def dumps(payload: Any, *, indent: int | None = None) -> str:
    """`json.dumps` that cannot emit a token a browser will reject.

    `allow_nan=False` as well as the scrub, so a future non-finite value raises here rather
    than shipping a document that fails to parse in the field.
    """
    return json.dumps(finite(payload), indent=indent, default=str, allow_nan=False)


def stamp() -> str:
    """Now, in UTC, to the second. The freshness field every artifact carries.

    UTC and not local: the whole claim of a dated artifact is that December can read August,
    and a naive local timestamp is ambiguous the moment the machine or the season changes.
    """
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def artifact(name: str, source: str, rows: list[Any], **extra: Any) -> dict[str, Any]:
    """The shape every published artifact has, so two writers of one file cannot disagree.

    `generated_at` is the freshness stamp the page ages a panel by and the watchdog reads as
    a heartbeat. `n` is carried rather than left to the reader because the manifest reports it
    without opening `rows`.

    `extra` is for what one artifact has and another does not -- the league a scoreboard is
    for, the season a slate belongs to. It cannot displace the envelope: a caller passing
    `rows` or `generated_at` here would be redefining the thing this exists to fix.
    """
    # GUARD envelope-cannot-be-overridden: extra never displaces the shape every reader depends on
    clash = {"name", "source", "generated_at", "n", "rows"} & set(extra)
    if clash:
        raise ValueError(f"{sorted(clash)} belong to the envelope and cannot be overridden")
    # /GUARD
    return {"name": name, "source": source, "generated_at": stamp(),
            "n": len(rows), "rows": rows, **extra}
