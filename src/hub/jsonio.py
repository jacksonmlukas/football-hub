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
from pathlib import Path
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


def file_stamp(path: Path) -> str:
    """When a file was last written, in the shape `stamp` gives, for artifacts read off disk.

    An artifact derived from a file is as fresh as that file, not as fresh as the run that
    opened it. Stamping the run's own clock on it is how a producer that read week-old data
    publishes a panel dated today -- see `artifact`'s `as_of`.
    """
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(
        microsecond=0).isoformat()


# The two shapes a published artifact can be, and the vocabulary `shape` is written in.
#
# A **rows** artifact carries `rows` and the `n` that counts them. A **summary** carries
# neither and reports in fields of its own -- `n_scored` and a calibration curve for the track
# record, `rows_by_endpoint` and `quota` for the CFBD envelope, whose whole point is to say
# what was fetched without carrying it.
#
# The key exists because until #227 that difference lived in `NOT_ROW_SHAPED`, a set of
# filenames in `tests/contracts/test_published_envelopes.py`, while the writers that decide it
# live here and in `hub.fetch.cfbd`. Nothing connected the two, so they could only agree by
# someone remembering -- and on 2026-09-09 nobody did: the scheduled slate published
# `site/data/cfbd.json` for the first time and the contract failed on the next human push, ten
# commits later, for a file that was correct. An artifact that says what it is can be checked
# for saying it; a list of names elsewhere can only be checked for being current, by hand.
ROWS = "rows"
SUMMARY = "summary"
SHAPES = (ROWS, SUMMARY)


def artifact(name: str, source: str, rows: list[Any],
             as_of: str | None = None, **extra: Any) -> dict[str, Any]:
    """The shape every published artifact has, so two writers of one file cannot disagree.

    `generated_at` is the freshness stamp the page ages a panel by and the watchdog reads as
    a heartbeat. `n` is carried rather than left to the reader because the manifest reports it
    without opening `rows`.

    `extra` is for what one artifact has and another does not -- the league a scoreboard is
    for, the season a slate belongs to. It cannot displace the envelope: a caller passing
    `rows` or `generated_at` here would be redefining the thing this exists to fix.

    `as_of` is how a producer says its rows are older than its run, and it is deliberately
    *not* spelled `generated_at`. A producer that re-derives from a live source is as fresh as
    its run and takes the default; one that reads a file is only as fresh as the file and
    passes `file_stamp(path)`. Leaving the default in place there is what let a served roster
    publish as this week's (issue #122).

    The spelling matters because `generated_at` in `extra` still has to raise. That is the
    accidental override the guard below was written for, and giving the deliberate path the
    same name would have retired the guard while appearing to keep it -- the keyword would
    bind to the parameter and never reach the check. The existing envelope test caught this.
    """
    # GUARD envelope-cannot-be-overridden: extra never displaces the shape every reader depends on
    clash = {"name", "source", "generated_at", "shape", "n", "rows"} & set(extra)
    if clash:
        raise ValueError(f"{sorted(clash)} belong to the envelope and cannot be overridden")
    # /GUARD
    return {"name": name, "source": source,
            "generated_at": as_of if as_of is not None else stamp(),
            "shape": ROWS, "n": len(rows), "rows": rows, **extra}


def summary(name: str, source: str, as_of: str | None = None,
            **extra: Any) -> dict[str, Any]:
    """The envelope for an artifact that reports rather than carries -- the second shape.

    Same `name`, `source` and `generated_at` as `artifact`, and `shape: "summary"` in place of
    `n`/`rows`. Everything the artifact actually says goes in `extra`, because what a summary
    says is the part that differs between one summary and the next: the track record's
    `n_scored` and calibration curve, the CFBD envelope's `rows_by_endpoint` and `quota`.

    **Why this exists rather than a flag on `artifact`.** Both writers already built these
    envelopes by hand -- three literal `"name"`/`"source"`/`"generated_at"` keys each -- which
    is what let one of them be published without any contract having read it. Passing through
    a constructor is what makes the declaration something the writer emits rather than
    something a person remembers to add.

    `rows` and `n` are refused here for the same reason `extra` cannot displace the envelope
    in `artifact`: an artifact carrying rows is row-shaped, and one that carries them while
    declaring `summary` would put the declaration and the document in exactly the
    disagreement the envelope exists to prevent.
    """
    # GUARD summary-carries-no-rows: a summary restates no envelope key and carries no rows
    clash = {"name", "source", "generated_at", "shape", "n", "rows"} & set(extra)
    if clash:
        raise ValueError(
            f"{sorted(clash)} belong to the envelope and cannot be overridden; `n` and `rows` "
            f"belong to a row-shaped artifact, which this is not")
    # /GUARD
    return {"name": name, "source": source,
            "generated_at": as_of if as_of is not None else stamp(),
            "shape": SUMMARY, **extra}
