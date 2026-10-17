"""`site/data/ledger.json` -- one row per thing that faced a gate or a screen (#483).

The README's "what this repo has actually concluded" table and the ADR index hold these
verdicts in prose that has been restated many times. A third hand-written copy in the page
would drift from both, so this one is *generated*: a verdict is a small front-matter block at
the top of the document that records it, and the next publish picks up a restatement by
reading that block. Nothing else is edited.

**Sources** (the README table is deliberately not parsed -- it is prose):

* a ``---`` front-matter block carrying ``verdict:`` at the top of any ``docs/adr/*.md`` or
  ``docs/*.md``: ``name``, ``verdict`` (one of `VERDICTS`), ``figure`` (the one number the
  verdict rests on, as a string with its unit), ``as_of`` (the date of the last restatement)
  and optionally ``issue``;
* the table in ``docs/signal-screens.md``, for the signals screened before any implementation
  and found null. Those have no ADR, so ``adr`` is null for them.

**Parsing is defensive.** A block that is malformed, names a verdict outside the vocabulary or
carries a non-date ``as_of`` yields no row and a *problem* -- it never raises, so a bad edit to
one document cannot blank the Ledger (CLAUDE.md's degradation rule). `problems` is what the
contract test asserts is empty, which is how a dropped verdict becomes a red test rather than a
missing row nobody notices.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from hub import jsonio

ROOT = Path(__file__).resolve().parents[2]

VERDICTS = ("Removed", "Demoted", "Shown never ranked", "Inert", "Null",
            "Inconclusive", "Not established", "Adopted")

SCREENS = "docs/signal-screens.md"
INDEX = "docs/architecture.md"

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Every form the docs use to date a revision of a verdict, found by grepping docs/adr/ and
# method.md: "Restated 2026-..", "Re-scored/Re-scoring ..", "Amendment, 2026-..", "Amended ..",
# "Superseded figures, 2026-..", "Attributed ..", "Resolved ..", "Corrected ..", "Settled ..".
# A form missing from this list is a revision the as_of check cannot see (ADR-0016's
# "Amendment, 2026-10-07" and ADR-0009's "Superseded figures, 2026-10-09" both were).
_REVISION_WORDS = (r"restated|re-scored|re-scoring|amendment|amended|superseded figures|"
                   r"attributed|resolved|corrected|settled|dated note")
_RESTATED = re.compile(rf"(?:{_REVISION_WORDS})\D{{0,16}}?(\d{{4}}-\d{{2}}-\d{{2}})",
                       re.IGNORECASE)
_ANY_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

# What an ADR's index row says when the ADR records a verdict: its decision cell or its status
# cell. Deliberately narrow -- an ADR about a mechanism ("A Gate requires every season") is
# not a verdict and must not be forced to carry one. Widening this is the way to hold a new
# kind of verdict to the same rule.
_CLAIMS_A_VERDICT = re.compile(
    r"never ranked on|does not pick|waits for real|resolved:? (?:SHOW|REMOVE)", re.IGNORECASE)


def front_matter(text: str) -> dict[str, str] | None:
    """The ``key: value`` pairs of a leading ``---`` block, or None if there is not one.

    Not YAML: a flat block of single-line pairs, with an optional pair of double quotes around
    a value. A line that is not a pair is skipped rather than fatal.
    """
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    out: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        key, sep, value = line.partition(":")
        if not sep or not key.strip() or " " in key.strip():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        out[key.strip()] = value
    return None  # never closed


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _doc_row(path: Path, root: Path, block: dict[str, str]) -> tuple[dict[str, Any] | None,
                                                                    list[str]]:
    rel = _rel(path, root)
    bad = []
    for key in ("name", "verdict", "figure", "as_of"):
        if not block.get(key):
            bad.append(f"{rel}: front matter has a verdict but no `{key}`")
    if block.get("verdict") and block["verdict"] not in VERDICTS:
        bad.append(f"{rel}: verdict {block['verdict']!r} is not one of {list(VERDICTS)}")
    if block.get("as_of") and not _DATE.match(block["as_of"]):
        bad.append(f"{rel}: as_of {block['as_of']!r} is not a YYYY-MM-DD date")
    issue = block.get("issue")
    if issue and not issue.isdigit():
        bad.append(f"{rel}: issue {issue!r} is not a number")
    if bad:
        return None, bad
    in_adr = path.parent.name == "adr"
    row: dict[str, Any] = {
        "name": block["name"], "verdict": block["verdict"], "figure": block["figure"],
        "adr": rel if in_adr else None, "issue": int(issue) if issue else None,
        "as_of": block["as_of"], "source": rel,
    }
    if block.get("detail"):
        row["detail"] = block["detail"]  # free text here; screen rows carry a doc path
    return row, []


def latest_revision(text: str) -> str | None:
    """The latest dated revision block in a document's body (its front matter excluded)."""
    body = text.split("\n---\n", 1)[-1] if text.startswith("---") else text
    dates = _RESTATED.findall(body)
    return max(dates) if dates else None


def _later_than_as_of(text: str, as_of: str, rel: str) -> list[str]:
    """A restatement dated after `as_of` means the block was not updated with it."""
    body = text.split("\n---\n", 1)[-1] if text.startswith("---") else text
    late = sorted({d for d in _RESTATED.findall(body) if d > as_of})
    return [f"{rel}: restated {d}, after the front matter's as_of {as_of}" for d in late]


def _docs(root: Path) -> list[Path]:
    docs = root / "docs"
    return sorted([*docs.glob("*.md"), *docs.glob("adr/*.md")])


def _screen_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


_MINUS = "\N{MINUS SIGN}"
_NUM = rf"[+{_MINUS}-]?\d*\.?\d+"
_INTERVAL = re.compile(rf"\[\s*({_NUM})\s*,\s*({_NUM})\s*\]")
# A screen's figure: the partial r, with its 95% interval when the row carries one (#505).
_FIGURE = re.compile(rf"(?:partial )?r\s*=\s*{_NUM}(?:,\s*95% interval\s*\[[^\]]*\])?")


def _screen_rows(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The signals screened before the season and found null, from the screens index.

    A preseason screen is one whose horizon is not a week (the next-week rows are a different
    programme and carry their own table). A result cell opening ``**inconclusive`` is the
    verdict ``Inconclusive``, with its 95% interval read into ``interval`` when it carries one
    (#505); one still opening ``**null`` is the old ``Null``. A struck-through cell (``~~``)
    is withdrawn and is not a verdict.
    """
    path = root / SCREENS
    try:
        text = path.read_text()
    except OSError:
        return [], [f"{SCREENS}: cannot be read"]
    dates = _RESTATED.findall(text)
    if not dates:
        return [], [f"{SCREENS}: carries no restatement date to date its rows by"]
    as_of = max(dates)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = _screen_cells(line)
        if len(cells) < 4 or not cells[2].lower().startswith(("**null", "**inconclusive")):
            continue
        signal, horizon, result, detail = cells[0], cells[1], cells[2], cells[3]
        if "week" in horizon.lower():
            continue
        fig = _FIGURE.search(result)
        verdict = "Inconclusive" if result.lower().startswith("**inconclusive") else "Null"
        bounds = _INTERVAL.search(fig.group(0).replace(_MINUS, "-")) if fig else None
        link = re.search(r"\]\(([^)#]+)", detail)
        rows.append({
            "name": f"{signal} ({horizon})", "verdict": verdict,
            "figure": fig.group(0).replace(_MINUS, "-") if fig else None,
            "adr": None, "issue": None, "as_of": as_of, "source": SCREENS,
            "detail": f"docs/{link.group(1)}" if link else None,
            **({"interval": [float(bounds.group(1)), float(bounds.group(2))]} if bounds else {}),
        })
    return rows, []


def build(root: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    """Every row, and every reason a source that claimed a verdict yielded none."""
    root = root or ROOT
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    for path in _docs(root):
        try:
            text = path.read_text()
        except OSError as e:
            notes.append(f"{_rel(path, root)}: unreadable ({e.__class__.__name__})")
            continue
        block = front_matter(text)
        if block is None or "verdict" not in block:
            continue
        row, bad = _doc_row(path, root, block)
        notes += bad
        if row:
            rows.append(row)
            notes += _later_than_as_of(text, row["as_of"], row["source"])
    more, bad = _screen_rows(root)
    rows += more
    notes += bad
    return rows, notes


def _index_rows(root: Path) -> list[tuple[str, str]]:
    """(adr path, decision + status) for each accepted row of the ADR index."""
    try:
        text = (root / INDEX).read_text()
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        m = re.match(r"\|\s*\[\d{4}\]\((adr/[^)]+\.md)\)\s*\|(.*)$", line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(2).split("|")]
        if len(cells) >= 2 and cells[1].lower().startswith("accepted"):
            out.append((f"docs/{m.group(1)}", f"{cells[0]} {cells[1]}"))
    return out


def anchor_problems(rows: list[dict[str, Any]], root: Path) -> list[str]:
    """Every row must cite a document that exists."""
    bad = []
    for r in rows:
        for key in ("source", "adr", "detail"):
            ref = r.get(key)
            if key == "detail" and ref and not ref.startswith("docs/"):
                continue  # a front-matter note, not a pointer
            if ref and not (root / ref).is_file():
                bad.append(f"{r['name']}: `{key}` cites {ref}, which does not exist")
        if not r.get("source"):
            bad.append(f"{r['name']}: cites no source document")
    return bad


def problems(root: Path | None = None) -> list[str]:
    """What the contract test asserts is empty."""
    root = root or ROOT
    rows, bad = build(root)
    bad += anchor_problems(rows, root)
    have = {r["adr"] for r in rows if r["adr"]}
    for adr, claim in _index_rows(root):
        if _CLAIMS_A_VERDICT.search(claim) and adr not in have:
            bad.append(f"{adr}: the index says accepted and its row names a verdict "
                       f"({claim[:60]!r}), but it yields no ledger row")
    names = [r["name"] for r in rows]
    bad += [f"two rows are named {n!r}" for n in sorted({n for n in names if names.count(n) > 1})]
    return bad


def ledger(out: Path | None = None, root: Path | None = None) -> Any:
    """Write `ledger.json`, or keep last-good. Never raises over a bad source document."""
    from hub import publish  # lazy: publish wires this module in, so the import is circular

    rows, bad = build(root)
    for line in bad:
        print(f"  ledger: skipped -- {line}", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: (VERDICTS.index(r["verdict"]), r["name"]))
    payload = jsonio.artifact(
        "ledger", "docs/adr + docs front matter + docs/signal-screens.md", rows,
        skipped=len(bad))
    return publish._publish(out or publish.SITE, "ledger", payload)
