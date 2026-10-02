"""What `site/index.html` reads off each artifact, extracted from the page, so a producer cannot
drop it.

`test_live_artifact_shape.py` records this for the live overlay alone, and it exists because
two processes wrote that file in different shapes and nothing could notice. Six other panels
had no such record: a producer that stopped writing `gain`, `withheld` or `set_total` renamed
nothing and broke no test -- the panel just rendered a blank, an `undefined`, or silently took
a fallback branch.

The near-miss that prompted this was the mirror image. Issue #122 made the roster producer
carry a truthful `generated_at`, and the panel spent its age slot on a row count and took
staleness from a manifest boolean that stays false for any payload that parses -- so a correct
field would have published into a page that never displayed it, with every test green. It was
caught by reading `index.html`, not by anything automatic (issue #127).

**The page is the source of truth (#395).** This file used to keep `READS`, a hand-written
copy of what the page reads, tied back to the page by two further tests. The copy is gone: the
keys are extracted from the page's own `load(...)` sites by `page_reads`, and each producer is
checked against what the page reads, so there is no second list to fall behind the first.
No page knowledge lives in `hub.publish`; producers keep emitting what they emit.

**Asserted against the committed artifacts**, which is what the page will actually be served.
The honest limit of that, stated rather than left for a reader to discover: a producer change
is caught when the site is next published, not at the moment the producer changes. Running
every producer here instead would need each one's inputs stubbed, and a stub that pairs a
producer with a payload it could not have produced is the fixture shape this repo has an
incident about. The committed file is the real thing.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"
SRC = ROOT / "src" / "hub"
PAGE = ROOT / "site" / "index.html"

# Panels whose artifact this file does not pin, each with the reason. An absence is how a
# record stops covering something without anyone deciding to; a named entry is a decision.
# (`preds` is loaded as `load(key)` with a computed name, so the extraction below never sees
# it as a literal; the manifest keys it is found through are read off `man` like any other.)
NOT_PINNED_HERE: dict[str, str] = {
    "live": "`test_live_artifact_shape.py` already records it, and pins both of its writers "
            "against each other -- which is more than this file does. Duplicating the list "
            "here would give two records free to disagree.",
}

# Keys the page reads off an artifact that the producer writes only in a state the committed
# file is not in, so the committed file cannot show them. The survivor panel reads the
# elimination fields behind `surv.status === "eliminated"`, and `publish.py` writes them as
# keyword arguments on that branch only. Each is checked against the producer's *source*
# instead -- see `test_a_conditional_key_is_still_named_by_a_producer` -- so the exemption
# cannot outlive the field. A named entry is a decision; widening this to "any key the file
# lacks" would turn the contract back into the thing it replaces.
CONDITIONAL: dict[str, dict[str, str]] = {
    "survivor": {
        "eliminated_week": "written only once the entry is out",
        "eliminated_team": "written only once the entry is out",
        "buyback_open": "written only once the entry is out",
        "buyback_cutoff_week": "written only once the entry is out",
    },
}


def page_reads(text: str) -> dict[str, set[str]]:
    """`artifact -> {key the page reads off it}`, extracted from the page's own JavaScript.

    Every `<name> = ... await load("<artifact>")` binds a JS name to an artifact, and every
    `<name>.<key>` / `<name>?.<key>` is a read of that key.

    Stated limits, not hidden ones. It sees the first property off the bound name and no
    deeper (`ros.rows` yes, a row's own fields no); a read through a destructure, a bracket or
    an alias is not seen. A name rebound to something else in another scope would be read as
    this artifact's -- widening, never narrowing, what is checked. `load(key)` with a computed
    name binds nothing here.
    """
    reads: dict[str, set[str]] = {}
    for name, artifact in re.findall(r'(\w+)\s*=\s*[^;\n]*?await load\("([a-z_]+)"\)', text):
        keys = re.findall(rf"(?<![\w.$]){re.escape(name)}\??\.([A-Za-z_]\w*)", text)
        reads.setdefault(artifact, set()).update(keys)
    return reads


def _page_reads() -> dict[str, set[str]]:
    return {a: ks for a, ks in page_reads(PAGE.read_text()).items() if a not in NOT_PINNED_HERE}


def unwritten(keys: set[str], payload: dict, conditional: frozenset[str] = frozenset()
              ) -> list[str]:
    """The keys the page reads that `payload` does not carry, minus the conditional ones."""
    return sorted(k for k in keys if k not in payload and k not in conditional)


def _published(name: str) -> dict:
    matches = sorted(SITE.glob(f"{name}.json"))
    if not matches:
        pytest.skip(f"{name}.json is not published in this checkout")
    return json.loads(matches[0].read_text())


@pytest.mark.parametrize("artifact", sorted(_page_reads()))
def test_the_published_artifact_carries_what_the_page_reads(artifact):
    """The property `live` has had since two producers disagreed about its shape: the page
    reads a key, so some producer must write it."""
    missing = unwritten(_page_reads()[artifact], _published(artifact),
                        frozenset(CONDITIONAL.get(artifact, {})))
    assert not missing, (
        f"the page reads {missing} off `{artifact}` and the published artifact does not "
        f"carry them, so that panel renders blank or takes a fallback branch")


def test_the_extraction_is_not_blind():
    """A scan that finds nothing is indistinguishable from a page that reads nothing, which is
    the failure a hand list could not have: every artifact the page loads by name has to yield
    at least one key, or the extraction has stopped seeing it."""
    loaded = set(re.findall(r'await load\("([a-z_]+)"\)', PAGE.read_text()))
    loaded -= set(NOT_PINNED_HERE)
    reads = _page_reads()
    assert loaded, "found no `await load(...)` in the page; the extraction is blind"
    blind = sorted(loaded - {a for a, ks in reads.items() if ks})
    assert not blind, f"loaded by the page but no key read off them: {blind}"


def _keyword_names() -> set[str]:
    """Every keyword-argument name passed anywhere in `src/hub` -- how a producer spells a
    field it writes (`dict(eliminated_week=week, ...)`)."""
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.keyword) and node.arg:
                names.add(node.arg)
    return names


def test_a_conditional_key_is_still_named_by_a_producer():
    """The exemption's expiry: a CONDITIONAL key the producer no longer names, or the page no
    longer reads, is a stale exemption and fails."""
    names, reads = _keyword_names(), _page_reads()
    for artifact, keys in CONDITIONAL.items():
        for key in keys:
            assert key in reads.get(artifact, set()), (
                f"CONDITIONAL says the page reads `{artifact}.{key}`; it no longer does")
            assert key in names, (
                f"CONDITIONAL says a producer writes `{key}` on some branch; no keyword "
                f"argument in src/hub names it")


# -- the extraction, proven (rule 18) -----------------------------------------------------

def test_a_planted_read_in_a_copy_of_the_page_is_found_and_a_planted_unread_key_is_not(
        tmp_path):
    page = tmp_path / "index.html"
    page.write_text(PAGE.read_text() + "\n<script>\nasync function planted() {\n"
                    '  const p = await load("roster");\n  return p.zz_planted_read;\n}\n'
                    "</script>\n", encoding="utf-8")
    found = page_reads(page.read_text())
    assert "zz_planted_read" in found["roster"]
    assert "zz_planted_unread" not in found["roster"], (
        "a key nothing in the page reads must not be extracted")
    assert "zz_planted_read" not in page_reads(PAGE.read_text())["roster"], (
        "the plant leaked into the real page")
    # ...and the contract's own check, run on the planted extraction, fails on exactly it.
    assert "zz_planted_read" in unwritten(found["roster"], _published("roster"))


def test_a_key_the_page_reads_and_no_producer_writes_fails_the_contract():
    """The consumer of the extraction, planted: a read the payload does not carry is reported,
    one it does carry or one declared conditional is not."""
    assert unwritten({"a", "zz_planted_read"}, {"a": 1}) == ["zz_planted_read"]
    assert unwritten({"a", "b"}, {"a": 1}, frozenset({"b"})) == []
