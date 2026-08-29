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
"""
from __future__ import annotations

import json
import math
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
