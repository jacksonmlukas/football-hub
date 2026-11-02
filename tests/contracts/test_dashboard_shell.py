"""The page's shell: tokens, the four state chips and the hash router (#478).

Three properties, each a function of text or of a module run in node, each with a planted control
that has to come back red:

* the four chips are built in one place (`site/js/state.js`), differ in glyph and outline, and a
  stale chip cannot be built without an age;
* the router (`site/js/router.js`) maps each of the five hashes to exactly one visible view, sends
  anything else to This week, and stops animating under `prefers-reduced-motion`;
* every colour on the page is a token on bare `:root`, redefined for dark, and no rule names a
  colour literal anywhere else (the stylesheet, or a script that draws markup).

Run against the page the same way `test_dashboard_escapes.py` does: modules are imported from
`site/js/`, never lifted out of a file. The escape scan in that file covers this shell's modules
too, since it reads every file under `js/`.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
JS = SITE / "js"
PAGE = SITE / "index.html"
VIEW_IDS = ("week", "games", "record", "study", "ledger")


@pytest.fixture(scope="module")
def node():
    exe = shutil.which("node")
    if not exe:                                              # pragma: no cover
        pytest.skip("node is not installed; the runner has it")
    return exe


def _mod(rel: str) -> str:
    assert (JS / rel).exists(), f"no module {rel} under {JS}"
    return json.dumps((JS / rel).as_uri())


def _run(node: str, body: str):
    got = subprocess.run([node, "--input-type=module", "-e", body],
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return json.loads(got.stdout)


def _style(text: str | None = None) -> str:
    text = PAGE.read_text() if text is None else text
    return text[text.index("<style>"):text.index("</style>")]


# --- the four chips ----------------------------------------------------------

CHIPS = {  # state -> (builder call, glyph)
    "frozen": ('chipFrozen("locked")', "lock"),
    "live": ('chipLive("live")', "dot"),
    "stale": ('chipStale("3h ago")', "clock"),
    "unconfirmed": ('chipUnconfirmed("assumed")', "question"),
}


def _chips(node: str) -> dict:
    calls = ", ".join(f"{k}: {call}" for k, (call, _) in CHIPS.items())
    return _run(node, f"import * as s from {_mod('state.js')};\n"
                      f"const {{ chipFrozen, chipLive, chipStale, chipUnconfirmed }} = s;\n"
                      f"console.log(JSON.stringify({{ {calls} }}));")


def test_each_chip_carries_its_own_class_state_and_glyph(node):
    got = _chips(node)
    for state, (_, glyph) in CHIPS.items():
        html = got[state]
        assert f'class="chip chip-{state}"' in html, state
        assert f'data-state="{state}"' in html, state
        assert f'data-glyph="{glyph}"' in html, state
    glyphs = [g for s in CHIPS for g in re.findall(r'data-glyph="(\w+)"', got[s])]
    assert len(set(glyphs)) == 4, f"two states share a glyph: {glyphs}"


def test_the_stale_chip_shows_its_age_and_no_other_chip_does(node):
    got = _chips(node)
    assert 'class="chip-age">3h ago<' in got["stale"]
    assert all("chip-age" not in got[s] for s in ("frozen", "live", "unconfirmed"))


def test_a_stale_chip_cannot_be_built_without_an_age(node):
    out = _run(node, f"import {{ chipStale }} from {_mod('state.js')};\n"
                     "const r = [];\n"
                     "for (const a of [undefined, null, '', '   ', 5]) {\n"
                     "  try { chipStale(a); r.push('built'); } catch (e) { r.push(e.name); }\n"
                     "}\n"
                     "try { chipStale(); r.push('built'); } catch (e) { r.push(e.name); }\n"
                     "console.log(JSON.stringify(r));")
    assert out == ["TypeError"] * 6


def test_a_chip_escapes_the_text_it_is_given(node):
    out = _run(node, f"import {{ chipStale, chipFrozen }} from {_mod('state.js')};\n"
                     "const h = '<script>alert(1)</script>';\n"
                     "console.log(JSON.stringify(chipStale(h, h) + chipFrozen(h)));")
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_the_four_chips_differ_in_outline_and_the_live_one_alone_is_filled():
    style = _style()

    def rule(state: str) -> str:
        got = re.search(rf"(?m)^\s*\.chip-{state}\s*\{{([^}}]*)\}}", style)
        assert got, f"no rule for .chip-{state}"
        return got.group(1)

    assert "solid" in rule("frozen")
    assert "dotted" in rule("stale")
    assert "dashed" in rule("unconfirmed") and "repeating-linear-gradient" in rule("unconfirmed")
    assert "background: var(--pylon)" in rule("live")
    for state in ("frozen", "stale", "unconfirmed"):
        assert "background: var(--pylon)" not in rule(state)


def test_no_panel_still_draws_the_old_tag_classes():
    for f in sorted(JS.rglob("*.js")):
        assert 'class="tag' not in f.read_text(), f"{f.name} bypasses the chip builders"
    assert ".tag" not in _style(), "the old .tag rules are still in the stylesheet"


# --- the router --------------------------------------------------------------

_STUB = """
const body = { innerHTML: "" };
const els = {};
const mk = id => ({ id, hidden: false, attrs: {}, setAttribute(k, v) { this.attrs[k] = v; },
                    removeAttribute(k) { delete this.attrs[k]; },
                    querySelector: () => null });
for (const v of ["week", "games", "record", "study", "ledger"]) {
  els["v-" + v] = mk("v-" + v); els["nav-" + v] = mk("nav-" + v);
}
const doc = { getElementById: id => els[id] ?? null };
const state = () => ({
  visible: Object.keys(els).filter(k => k.startsWith("v-") && !els[k].hidden),
  current: Object.keys(els).filter(k => k.startsWith("nav-") && els[k].attrs["aria-current"]),
  body: body.innerHTML,
});
"""


def _router(node: str, script: str):
    return _run(node, f"import * as r from {_mod('router.js')};\n{_STUB}\n{script}")


@pytest.mark.parametrize("view", VIEW_IDS)
@pytest.mark.parametrize("form", ["#/{}", "#{}", "#/{}/"])
def test_each_hash_shows_exactly_its_own_view(node, view, form):
    out = _router(node, f"const id = r.show(doc, {json.dumps(form.format(view))});\n"
                        "console.log(JSON.stringify({ id, ...state() }));")
    assert out["id"] == view
    assert out["visible"] == [f"v-{view}"], out
    assert out["current"] == [f"nav-{view}"], out


@pytest.mark.parametrize("hash_", ["", "#", "#/", "#/nope", "#nope", "#/WEEK", "#/games/x", None])
def test_an_unknown_or_empty_hash_shows_this_week(node, hash_):
    out = _router(node, f"const id = r.show(doc, {json.dumps(hash_)});\n"
                        "console.log(JSON.stringify({ id, ...state() }));")
    assert out["id"] == "week" and out["visible"] == ["v-week"], out


def test_switching_views_leaves_only_the_last_one_visible(node):
    out = _router(node, "for (const h of ['#/games', '#/record', '#/study', '#/week'])"
                        " r.show(doc, h);\nconsole.log(JSON.stringify(state()));")
    assert out["visible"] == ["v-week"] and out["current"] == ["nav-week"]


def test_no_view_shows_a_placeholder_any_more(node):
    """Study (#487) and the Ledger (#488) were the last two; render() draws every view's body."""
    for view in VIEW_IDS:
        out = _router(node, f"r.show(doc, '#/{view}');\nconsole.log(JSON.stringify(state()));")
        assert out["body"] == "", out


def test_the_routers_views_are_the_pages_views_in_the_pages_order(node):
    out = _router(node, "console.log(JSON.stringify({ ids: r.VIEWS.map(v => v.id),"
                        " hrefs: r.VIEWS.map(v => r.hrefFor(v.id)), dflt: r.DEFAULT_VIEW }));")
    assert tuple(out["ids"]) == VIEW_IDS and out["dflt"] == "week"
    text = PAGE.read_text()
    assert re.findall(r'<div class="view" id="v-(\w+)"', text) == list(VIEW_IDS)
    assert re.findall(r'<a id="nav-(\w+)" href="([^"]+)"', text) == [
        (i, h) for i, h in zip(VIEW_IDS, out["hrefs"], strict=True)]


def test_the_bar_is_sticky_and_the_column_is_capped_near_1080px():
    style = _style()
    assert re.search(r"\.bar\s*\{[^}]*position:\s*sticky;[^}]*top:\s*0", style)
    assert re.search(r"\.page\s*\{[^}]*max-width:\s*1080px", style)
    assert re.search(r"\.bar-in\s*\{[^}]*max-width:\s*1080px", style)


_SWAP = """
const calls = [];
const vt = { startViewTransition: f => { calls.push("vt"); f(); } };
const win = reduced => ({ matchMedia: q => ({ matches: reduced && q.includes("reduce") }) });
let ran = 0;
r.swap(win(false), vt, () => ran++);
r.swap(win(true), vt, () => ran++);
r.swap(win(false), {}, () => ran++);
console.log(JSON.stringify({ calls, ran }));
"""


def test_the_view_transition_runs_only_where_supported_and_not_under_reduced_motion(node):
    out = _router(node, _SWAP)
    # Three swaps all change the view; only the first, supported and not reduced, animates.
    assert out == {"calls": ["vt"], "ran": 3}


_MOUNT = """
const handlers = {};
const win = { location: { hash: "#/record" }, addEventListener: (e, f) => { handlers[e] = f; },
              scrollTo() {}, matchMedia: () => ({ matches: true }) };
r.mount(win, doc);
const first = state().visible;
win.location.hash = "#/ledger"; handlers.hashchange();
const second = state().visible;
win.location.hash = "#/record"; handlers.hashchange();     // what the back button does
console.log(JSON.stringify({ first, second, back: state().visible, wired: Object.keys(handlers) }));
"""


def test_mount_shows_the_hash_it_loads_on_and_follows_the_back_button(node):
    out = _router(node, _MOUNT)
    assert out == {"first": ["v-record"], "second": ["v-ledger"], "back": ["v-record"],
                   "wired": ["hashchange"]}


def test_the_view_transition_stops_under_reduced_motion_in_the_stylesheet():
    style = _style()
    blocks = re.findall(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\n  \}", style, re.S)
    joined = "\n".join(blocks) + "".join(
        re.findall(r"@media \(prefers-reduced-motion: reduce\) \{([^}]*)\}", style))
    assert "::view-transition-group(*)" in joined and "animation: none" in joined
    assert ".chip-live .glyph" in joined


# --- tokens: no colour literal outside the two :root blocks -------------------

NAMED = ("white", "black", "red", "green", "blue", "gray", "grey", "orange", "yellow", "purple",
         "pink", "silver", "navy", "teal", "maroon", "olive", "lime", "aqua", "fuchsia", "brown")
_COLOUR_FN = re.compile(r"\b(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\(", re.I)
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_ROOT = re.compile(r":root\s*\{[^}]*\}")
_COLOUR_PROPS = ("color", "background", "border", "outline", "fill", "stroke", "box-shadow",
                 "text-decoration", "caret-color", "accent-color")


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def colour_literals(css: str) -> list[str]:
    """Colour literals in `css` once every bare `:root {...}` block is set aside."""
    rest = _ROOT.sub("", _strip_comments(css))
    found = _HEX.findall(rest) + _COLOUR_FN.findall(rest)
    for prop, value in re.findall(r"([\w-]+)\s*:\s*([^;{}]+)", rest):
        if any(prop == p or prop.startswith(p + "-") for p in _COLOUR_PROPS):
            found += [w for w in re.findall(r"[a-z]+", value.lower()) if w in NAMED]
    return found


def script_colour_literals(text: str) -> list[str]:
    """Hex or functional colours in a script's strings. Line comments are dropped first, since
    they name issues (`#352`) that read as three-digit hex."""
    code = "\n".join(re.sub(r"(?<![:\"'])//.*$", "", ln) for ln in text.splitlines())
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.S)
    return _HEX.findall(code) + _COLOUR_FN.findall(code)


def test_no_colour_literal_appears_outside_the_token_blocks():
    assert colour_literals(_style()) == []


def test_no_script_draws_a_colour_literal():
    for f in sorted(JS.rglob("*.js")):
        assert script_colour_literals(f.read_text()) == [], f.name


def test_the_check_sees_a_colour_literal_in_a_component_rule():
    """Planted: each way of writing a colour in a rule must be found."""
    base = _style()
    for plant in ("#fff", "#1d4ed8", "rgba(0, 0, 0, 0.5)", "hsl(10 20% 30%)", "white"):
        css = base.replace(".view[hidden] { display: none; }",
                           f".view[hidden] {{ display: none; color: {plant}; }}")
        assert css != base
        assert colour_literals(css), f"{plant} was not found"
    commented = base.replace(".view[hidden] { display: none; }",
                             ".view[hidden] { display: none; } /* was #fff */")
    assert colour_literals(commented) == [], "a colour named in a comment is not a rule"
    assert script_colour_literals("el.style.color = '#ff0000'; // #352")
    assert script_colour_literals("// #352 only a comment\nconst a = 1;") == []


def _root_blocks(css: str) -> tuple[str, str]:
    light = _ROOT.search(css[css.index("--bg:") - 40:])
    dark_at = css.index("prefers-color-scheme: dark")
    dark = _ROOT.search(css[dark_at:])
    assert light and dark
    return light.group(0), dark.group(0)


def _literal_tokens(block: str) -> set[str]:
    return {name for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", block)
            if _HEX.search(value) or _COLOUR_FN.search(value)}


def tokens_missing_from_dark(css: str) -> set[str]:
    light, dark = _root_blocks(css)
    return {t for t in _literal_tokens(light) if not re.search(rf"{re.escape(t)}\s*:", dark)}


def test_every_colour_token_is_redefined_for_dark():
    css = _style()
    light, _ = _root_blocks(css)
    assert len(_literal_tokens(light)) >= 20, "the token block lost its colours"
    assert tokens_missing_from_dark(css) == set()


def test_the_palette_has_the_named_roles():
    light, _ = _root_blocks(_style())
    for token in ("--bg", "--fg", "--turf", "--pylon", "--st-frozen-rule", "--st-stale-ink",
                  "--st-unc-ink"):
        assert re.search(rf"{token}\s*:", light), f"no {token} token"
    assert "--accent: var(--pylon)" in light


def test_the_check_sees_a_token_missing_from_dark():
    css = _style()
    planted = css.replace("--pylon: #fb923c;", "")
    assert planted != css
    assert tokens_missing_from_dark(planted) == {"--pylon"}


# --- type --------------------------------------------------------------------

def test_the_faces_are_applied_and_mono_is_only_for_digests():
    style = _strip_comments(_style())
    assert re.search(r"body\s*\{[^}]*var\(--font-text\)", style)
    assert re.search(r"body\s*\{[^}]*font-variant-numeric:\s*tabular-nums", style)
    assert re.search(r"\.wordmark\s*\{[^}]*var\(--font-display\)", style)
    assert re.search(r"\.zone-title\s*\{[^}]*var\(--font-display\)", style)
    uses = re.findall(r"(?m)^\s*([^\n{}]*)\{[^}]*font(?:-family)?:[^;}]*var\(--font-mono\)", style)
    assert [u.strip() for u in uses] == ["code.digest"], f"mono is used by {uses}"
