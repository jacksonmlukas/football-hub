"""The page escapes what it renders.

`site/index.html` builds its tables by concatenating field values into HTML strings and
assigning them to `innerHTML`. Player names, team names and status text arrive from ESPN and
nflverse and went in unescaped: an apostrophe or an ampersand in a name is enough to produce
malformed markup, and anything angle-bracketed in a third-party feed is injected into the
page. The site is public, unattended, and refreshed twice a week by a workflow with a push
token -- nobody is watching it render.

The exposure is small today, which is exactly why it was worth fixing while it was cheap.

Two tests, and the second is the one that lasts. Running `esc` and `table` for real proves
they escape; the static guard proves nobody added a thirteenth interpolation and forgot.
`table` escaping by default is what makes the guard's job small: a column cannot forget.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[2] / "site" / "index.html"

# Names carrying every character that matters, in the shapes a feed actually produces.
HOSTILE = "<script>alert(1)</script>"
REAL = "Ja'Marr Chase & D/ST <TEN>"


def _lift(pattern: str) -> str:
    """Pull one function out of the page so node can run the real thing.

    Lifted rather than reimplemented. A copy of the escaper in this file would pass while
    the page's own copy was wrong, which is the failure mode of testing a transcription.
    """
    got = re.search(pattern, PAGE.read_text())
    assert got, f"{pattern!r} no longer matches the page; the test is looking at nothing"
    return got.group(0)


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _run(node: str, body: str) -> str:
    got = subprocess.run([node, "-e", body], capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout.strip()


def test_a_name_carrying_markup_renders_as_text(node):
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    out = _run(node, f"{esc}\nconsole.log(esc({json.dumps(HOSTILE)}));")
    assert "<script>" not in out
    assert out == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_an_ordinary_name_survives_intact(node):
    """A gate that mangles `Ja'Marr Chase` is one that gets turned off. The entities have to
    be entities, and they have to be the right ones."""
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    out = _run(node, f"{esc}\nconsole.log(esc({json.dumps(REAL)}));")
    assert out == "Ja&#39;Marr Chase &amp; D/ST &lt;TEN&gt;"
    assert "Chase" in out and "D/ST" in out


def test_the_table_escapes_every_cell_without_being_asked(node):
    """The structural half. Escaping at each call site is discipline; escaping in `table` is
    a property, and it is why a new column cannot reintroduce this."""
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    tbl = _lift(r"function table\(cols, rows\) \{[\s\S]*?\n\}")
    body = (f"{esc}\n{tbl}\n"
            f"console.log(table([{{label: 'player', get: r => r.player}}], "
            f"[{{player: {json.dumps(HOSTILE)}}}]));")
    out = _run(node, body)
    assert "<script>" not in out, "a raw script tag reached the page's markup"
    assert "&lt;script&gt;" in out


def test_a_missing_value_renders_as_nothing_not_as_undefined(node):
    esc = _lift(r"const esc = v => [\s\S]*?\}\[c\]\)\);")
    assert _run(node, f"{esc}\nconsole.log('[' + esc(null) + esc(undefined) + ']');") == "[]"


# --- nobody added a thirteenth interpolation and forgot ----------------------

# Values that arrive from a third party and end up inside markup: fields read off a row,
# and the locals the prose around the tables builds from them. `r.live.*` is ESPN's
# scoreboard; the rest are names off nflverse and the ESPN league.
FROM_A_FEED = ("player", "pos", "team", "teams", "injury_status", "nfl_team",
               "detail", "state", "away_score", "home_score", "withheld",
               "held", "gone", "start", "sit")


def test_every_feed_value_in_the_markup_goes_through_esc():
    """Read off the page rather than remembered. Each `${...}` that names a third-party
    value has to escape it, and the ones that legitimately build markup -- `html`, a nested
    template, a call to `table` -- name no such value and are not caught here.

    Verified against the page as it stood before the fix: this reports the three hand-built
    slate cells and the roster and survivor prose. The `table` getters are not `${...}` at
    all, which is why `table` escaping on its own carries them."""
    text = PAGE.read_text()
    raw = []
    for expr in re.findall(r"\$\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text):
        names = re.findall(r"[\w.]+", expr)
        names = [n for part in names for n in part.split(".")]
        if not any(n in FROM_A_FEED for n in names):
            continue
        if "esc(" in expr or "table(" in expr or "fmt(" in expr or "pct(" in expr:
            continue
        raw.append(expr.strip())
    assert not raw, (
        f"these interpolate a feed value straight into markup: {raw}. Wrap each in `esc`, "
        f"or move the column into `table`, which escapes on its own.")


def test_the_scan_would_notice_an_unescaped_value():
    """The guard's own premise. A regex that matched no interpolations at all would make the
    assertion above vacuously true, which is how a guard like this dies quietly."""
    text = PAGE.read_text()
    exprs = re.findall(r"\$\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", text)
    assert len(exprs) > 20, f"only {len(exprs)} interpolations found; the page uses many more"
    assert any("esc(" in e for e in exprs), "no escaped interpolation found at all"


def test_the_getters_return_text_rather_than_markup():
    """`table` escapes unconditionally, so a getter returning `&mdash;` would render the
    entity literally. The em dash is the character now, and nothing may go back."""
    text = PAGE.read_text()
    getters = re.findall(r"get: r => ([^\n]*)", text)
    entities = [g for g in getters if "&" in g and ";" in g]
    assert not entities, f"these getters return HTML entities: {entities}"
