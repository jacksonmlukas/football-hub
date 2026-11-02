"""The This week and Games views point at the Record with the same counts its statement states (#486).

The statement moved into the Record view; a reader on another view still has to see how many
model numbers are pre-registered and where they are scored. Run through the page's real `render()`
against a stub DOM (`dashboard_render.mjs`):

* both views carry the pointer, in the markup and in what `render()` writes into it;
* its counts are read off the track record artifact and equal the ones the Record statement
  states, for every shape of count: some pre-registered, none, all, unknown;
* "Record" links to `#/record`, and nothing off the feed reaches the markup unescaped.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import UTC, datetime
from html import unescape
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
RENDER = Path(__file__).resolve().parent / "dashboard_render.mjs"
HOSTILE = "<script>alert(1)</script>"


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _render(node: str, ov: dict) -> dict:
    got = subprocess.run([node, str(RENDER)], input=json.dumps({"site": str(SITE), "ov": ov}),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def _tr(**kw) -> dict:
    tr = json.loads((SITE / "data" / "track_record.json").read_text())
    tr["generated_at"] = datetime.now(UTC).isoformat()
    tr.update(kw)
    return tr


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html))).strip()


def _statement_counts(body: str) -> tuple[int, int]:
    """(pre-registered, backtest) as the Record statement words them."""
    t = _text(body)
    m = re.search(r"(\d+) pre-registered, (\d+) scored", t)
    if m:
        return int(m[1]), int(m[2]) - int(m[1])
    m = re.search(r"(\d+) predictions scored retrospectively, (\d+) pre-registered", t)
    assert m, f"the statement states no counts: {t!r}"
    return int(m[2]), int(m[1]) - int(m[2])


def _note_counts(html: str) -> tuple[int, int]:
    m = re.search(r"Model numbers: (\d+) pre-registered, (\d+) backtest\.", _text(html))
    assert m, f"the pointer states no counts: {_text(html)!r}"
    return int(m[1]), int(m[2])


def _view(page: str, view: str, nxt: str) -> str:
    return page[page.index(f'id="v-{view}"'):page.index(f'id="v-{nxt}"')]


def test_both_views_carry_the_pointer_in_the_markup():
    page = (SITE / "index.html").read_text()
    assert 'id="mn-week"' in _view(page, "week", "games")
    assert 'id="mn-games"' in _view(page, "games", "record")
    assert 'id="mn-' not in _view(page, "record", "study"), "the Record is the target, not a pointer"


@pytest.mark.parametrize("pre, scored", [(64, 80), (0, 95), (9, 9), (1, 120)])
def test_the_pointers_counts_equal_the_statements(node, pre, scored):
    got = _render(node, {"track_record": _tr(n_preregistered=pre, n_scored=scored)})
    want = _statement_counts(got["panels"]["st-backtest"]["body"])
    assert want == (pre, scored - pre)
    for view in ("mn-week", "mn-games"):
        assert _note_counts(got["notes"][view]) == want, view


def test_the_pointer_reads_the_committed_artifact_not_a_typed_figure(node):
    tr = _tr()
    got = _render(node, {"track_record": tr})
    for view in ("mn-week", "mn-games"):
        assert _note_counts(got["notes"][view]) == (tr["n_preregistered"],
                                                    tr["n_scored"] - tr["n_preregistered"])


def test_the_pointer_links_to_the_record_and_says_where_scoring_is(node):
    note = _render(node, {"track_record": _tr()})["notes"]["mn-week"]
    assert 'How they score: <a href="#/record">Record</a>.' in note


def test_an_unknown_count_or_a_missing_record_is_not_shown_as_a_number(node):
    unknown = _render(node, {"track_record": _tr(n_preregistered=None)})["notes"]
    missing = _render(node, {"track_record": None})["notes"]
    for notes, said in ((unknown, "could not be read"), (missing, "no track record")):
        for html in notes.values():
            assert said in html and "0 pre-registered" not in html and 'href="#/record"' in html


def test_a_hostile_count_is_escaped(node):
    notes = _render(node, {"track_record": _tr(n_preregistered=HOSTILE)})["notes"]
    for html in notes.values():
        assert "<script>" not in html and "&lt;script&gt;" in html
