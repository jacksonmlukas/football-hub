"""The Ledger view lists what was measured and cut, one row per verdict (#488).

Each property is a function of `panels/ledger.js` run in node, or of the page's real `render()`
against a stub DOM (`dashboard_render.mjs`), on the frozen `ledger.json` fixture:

* a row is drawn for every published entry, sorted by verdict class (Removed first) and then by
  `as_of`, newest first;
* the page holds no verdict text of its own: the word on a row is the published word, and an
  empty ledger names no verdict at all;
* every ADR link in a rendered row resolves to a committed file under `docs/adr/`, like the
  archive-name test for the weeks;
* the verdict sits above the name at 390px (held here against the rule that makes it so, and in
  the screenshot test against a real layout);
* the colours come from the tokens #478 defined.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from hub.publish_ledger import VERDICTS

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
JS = SITE / "js"
FIXTURE = ROOT / "tests" / "fixtures" / "site" / "data" / "ledger.json"
RENDER = Path(__file__).resolve().parent / "dashboard_render.mjs"
HOSTILE = "<script>alert(1)</script>"


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _mod(rel: str) -> str:
    return json.dumps((JS / rel).as_uri())


def _run(node: str, body: str) -> str:
    got = subprocess.run([node, "--input-type=module", "-e", body],
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def _body(node: str, art: dict | None) -> str:
    return _run(node, f"import {{ ledgerBody }} from {_mod('panels/ledger.js')};\n"
                      f"console.log(ledgerBody({json.dumps(art)}));")


def _rows(html: str) -> list[str]:
    return re.findall(r'<li class="lg-row[^"]*".*?</li>', html, flags=re.S)


def _text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _first(pattern: str, text: str) -> str:
    m = re.search(pattern, text, re.S)
    assert m, f"{pattern!r} not found in {text[:200]!r}"
    return m.group(1)


def _verdict_of(row: str) -> str:
    return _text(_first(r'<span class="lg-verdict[^"]*">(.*?)</span>', row))


def _name_of(row: str) -> str:
    return _text(_first(r'<span class="lg-name">(.*?)</span>', row))


def _entry(name, verdict, as_of, adr=None, issue=None, figure="r = 0.10"):
    return {"name": name, "verdict": verdict, "figure": figure, "adr": adr, "issue": issue,
            "as_of": as_of, "source": adr or "docs/signal-screens.md"}


def _art(rows):
    return {"name": "ledger", "shape": "rows", "n": len(rows), "rows": rows, "skipped": 0,
            "generated_at": "2026-10-17T16:58:15+00:00"}


FIXTURE_ART = json.loads(FIXTURE.read_text())

# The order the issue fixes: Removed first. Nothing else in the issue ranks the classes, so the
# rest follows the vocabulary's own order, which `hub.publish_ledger.VERDICTS` states.
CLASS_ORDER = ("Removed", "Demoted", "Shown never ranked", "Adopted", "Inert", "Null",
               "Inconclusive", "Not established")


# --- one row per entry, in order ---------------------------------------------------------------

def test_one_row_per_published_entry_with_name_and_figure(node):
    html = _body(node, FIXTURE_ART)
    rows = _rows(html)
    assert len(rows) == len(FIXTURE_ART["rows"])
    assert sorted(_name_of(r) for r in rows) == sorted(e["name"] for e in FIXTURE_ART["rows"])
    figures = [_text(_first(r'<span class="lg-figure">(.*?)</span>', r)) for r in rows]
    assert sorted(figures) == sorted(e["figure"] or "" for e in FIXTURE_ART["rows"])


def test_every_verdict_class_the_publisher_can_emit_has_a_place_in_the_order():
    assert set(CLASS_ORDER) == set(VERDICTS), "a verdict the publisher adds needs a place here"


def test_rows_sort_by_verdict_class_then_as_of_newest_first(node):
    rows = [
        _entry("null-old", "Null", "2026-09-01"),
        _entry("removed-old", "Removed", "2026-08-01"),
        _entry("unconfirmed", "Not established", "2026-10-30"),
        _entry("null-new", "Null", "2026-10-01"),
        _entry("removed-new", "Removed", "2026-10-05"),
        _entry("demoted", "Demoted", "2026-07-01"),
        _entry("inert", "Inert", "2026-10-29"),
    ]
    got = [_name_of(r) for r in _rows(_body(node, _art(rows)))]
    assert got == ["removed-new", "removed-old", "demoted", "inert", "null-new", "null-old",
                   "unconfirmed"]


def test_equal_class_and_date_keep_the_published_order(node):
    rows = [_entry("b", "Null", "2026-10-01"), _entry("a", "Null", "2026-10-01")]
    assert [_name_of(r) for r in _rows(_body(node, _art(rows)))] == ["b", "a"]


def test_a_verdict_outside_the_order_sorts_last_and_is_still_shown(node):
    rows = [_entry("odd", "Zorp", "2026-10-30"), _entry("known", "Not established", "2026-01-01")]
    html = _body(node, _art(rows))
    got = [(_name_of(r), _verdict_of(r)) for r in _rows(html)]
    assert got == [("known", "Not established"), ("odd", "Zorp")]


# --- the page holds no verdict text of its own -------------------------------------------------

def test_the_verdict_word_on_a_row_is_the_published_word(node):
    rows = [_entry("x", "Made-up verdict", "2026-10-01")]
    assert [_verdict_of(r) for r in _rows(_body(node, _art(rows)))] == ["Made-up verdict"]
    html = _body(node, FIXTURE_ART)
    assert [_verdict_of(r) for r in _rows(html)] == [
        e["verdict"] for e in sorted(
            FIXTURE_ART["rows"], key=lambda e: (CLASS_ORDER.index(e["verdict"]),
                                                "".join(chr(0x10FFFF - ord(c)) for c in e["as_of"])))]


@pytest.mark.parametrize("art", [None, {}, _art([]), {"rows": "nope"}])
def test_an_empty_or_unreadable_ledger_names_no_verdict(node, art):
    text = _text(_body(node, art))
    assert text, "an empty ledger still says why it is empty"
    for word in VERDICTS:
        assert word.lower() not in text.lower(), f"{word!r} is verdict text the page invented"


def test_the_intro_and_method_text_name_no_verdict_either(node):
    text = _text(_body(node, _art([])))
    assert "pre-registered" in text.lower() and "docs/track-record.md" in _body(node, _art([]))
    for word in VERDICTS:
        assert word.lower() not in text.lower()


# --- links -------------------------------------------------------------------------------------

_BLOB = re.compile(r'href="https://github\.com/jacksonmlukas/football-hub/blob/main/([^"]+)"')
_ISSUE = re.compile(r'href="https://github\.com/jacksonmlukas/football-hub/issues/(\d+)"')


def _real_ledger() -> dict:
    return json.loads((SITE / "data" / "ledger.json").read_text())


@pytest.mark.parametrize("art", [FIXTURE_ART, "site"])
def test_every_adr_link_in_a_rendered_row_is_a_committed_file(node, art):
    art = _real_ledger() if art == "site" else art
    rows = _rows(_body(node, art))
    paths = [p for r in rows for p in _BLOB.findall(r)]
    assert paths, "no ADR link was drawn at all, so nothing was checked"
    assert len(paths) == sum(1 for e in art["rows"] if e["adr"])
    for p in paths:
        assert p.startswith("docs/adr/") and p.endswith(".md"), p
        assert (ROOT / p).is_file(), f"{p} is linked from the Ledger and is not committed"


def test_a_row_with_no_adr_draws_no_adr_link_and_an_issue_links_to_its_number(node):
    rows = [_entry("screen", "Null", "2026-10-01"),
            _entry("adr", "Removed", "2026-10-02", adr="docs/adr/0016-x.md", issue=123)]
    got = {_name_of(r): r for r in _rows(_body(node, _art(rows)))}
    assert _BLOB.findall(got["screen"]) == [] and _ISSUE.findall(got["screen"]) == []
    assert _BLOB.findall(got["adr"]) == ["docs/adr/0016-x.md"]
    assert _ISSUE.findall(got["adr"]) == ["123"]


def test_the_track_record_and_method_links_resolve(node):
    html = _body(node, _art([]))
    linked = _BLOB.findall(html)
    assert "docs/track-record.md" in linked and "docs/method.md" in linked
    for p in linked:
        assert (ROOT / p).is_file(), p


def test_a_hostile_feed_value_is_escaped(node):
    rows = [_entry(HOSTILE, HOSTILE, HOSTILE, adr=f'docs/adr/{HOSTILE}.md', issue=None,
                   figure=HOSTILE)]
    html = _body(node, _art(rows))
    assert "<script>" not in html and "&lt;script&gt;" in html


# --- the stylesheet ----------------------------------------------------------------------------

def _css() -> str:
    page = (SITE / "index.html").read_text()
    return _first(r"<style>(.*?)</style>", page)


def _rule(selector: str) -> str:
    m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", _css())
    assert m, f"no rule for {selector}"
    return m.group(1)


def test_the_verdict_is_a_block_above_the_name_and_uses_the_display_face():
    row = _rule(".lg-row")
    assert "display: grid" in row and "grid-template-columns" not in row, \
        "a single-track grid stacks the verdict over the name; columns or a flex row do not"
    verdict = _rule(".lg-verdict")
    assert "var(--font-display)" in verdict
    assert "display: block" in verdict


@pytest.mark.parametrize("selector,token", [
    (".lg-loss", "var(--bad)"), (".lg-kept", "var(--turf)"), (".lg-null", "var(--muted)"),
    (".lg-unconfirmed", "var(--st-unc-ink)")])
def test_each_verdict_class_takes_its_colour_from_a_token(selector, token):
    assert token in _rule(selector)


def test_the_ledger_rules_name_no_colour_literal():
    css = _css()
    rules = re.findall(r"\.lg-[^{]*\{[^}]*\}", css)
    assert rules
    for r in rules:
        assert not re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", r), r


@pytest.mark.parametrize("verdict,cls", [
    ("Removed", "lg-loss"), ("Demoted", "lg-kept"), ("Shown never ranked", "lg-kept"),
    ("Inert", "lg-null"), ("Null", "lg-null"), ("Not established", "lg-unconfirmed")])
def test_a_verdict_gets_the_colour_class_the_issue_names(node, verdict, cls):
    row = _rows(_body(node, _art([_entry("x", verdict, "2026-10-01")])))[0]
    assert f'lg-verdict {cls}"' in row


# --- the page --------------------------------------------------------------------------------

def _render(node: str, ov: dict) -> dict:
    got = subprocess.run([node, str(RENDER)], input=json.dumps({"site": str(SITE), "ov": ov}),
                         capture_output=True, text=True, timeout=60)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def test_render_draws_the_ledger_panel_from_the_artifact(node):
    out = _render(node, {"ledger": FIXTURE_ART})
    panel = out["panels"]["p-ledger"]
    assert len(_rows(panel["body"])) == len(FIXTURE_ART["rows"])


def test_a_missing_ledger_says_so_and_names_no_verdict(node):
    out = _render(node, {"ledger": None})
    body = out["panels"]["p-ledger"]["body"]
    assert 'class="warn"' in body
    for word in VERDICTS:
        assert word.lower() not in _text(body).lower()


def test_the_view_has_a_panel_and_the_router_no_longer_holds_a_placeholder_for_it():
    page = (SITE / "index.html").read_text()
    assert re.search(r'<section id="p-ledger"[^>]*><h3>What got measured, and what got cut', page)
    assert "ledger:" not in (JS / "router.js").read_text().split("const RENDERERS")[1].split("};")[0]
