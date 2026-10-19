"""Self-hosted fonts stay same-origin, licensed, bounded and non-blocking (ADR-0026, #477).

K1 said no webfont and no CDN. ADR-0026 amends it: WOFF2 files may live in `site/fonts/` and be
served from the same origin. The amendment holds only while four things stay true, and each is
a check here rather than a sentence:

* every `@font-face` source is a relative same-origin path, and no page or script names a font
  host (the property K1's no-CDN clause existed for);
* every font file has its SIL OFL text beside it (`<Family>-OFL.txt`);
* the fonts weigh at most 120 KB together (the ADR chose about 80 KB);
* every `@font-face` says `font-display: swap`, so a failed load never blanks text.

Each check is a function of text or a directory, so each has a planted control: a throwaway
tree with an external `@font-face` URL, a font with no licence, an oversized font, and a face
without `swap` must all come back red.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
BUDGET_BYTES = 120 * 1024
FONT_HOSTS = ("fonts.googleapis.com", "fonts.gstatic.com", "use.typekit.net", "fonts.bunny.net",
              "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "use.fontawesome.com")
_FACE = re.compile(r"@font-face\s*\{(.*?)\}", re.S)
_URL = re.compile(r"url\(\s*['\"]?([^'\")\s]+)")


def _text_files(site: Path):
    for p in sorted(site.rglob("*")):
        if p.suffix in {".html", ".css", ".js", ".mjs"} and "data" not in p.relative_to(site).parts:
            yield p


def external_font_problems(site: Path) -> list[str]:
    bad = []
    for p in _text_files(site):
        text = p.read_text()
        for body in _FACE.findall(text):
            for u in _URL.findall(body):
                if re.match(r"(?:[a-z][a-z0-9+.-]*:|//)", u, re.I) or u.startswith("/"):
                    bad.append(f"{p.name}: @font-face src {u} is not a relative same-origin path")
        for host in FONT_HOSTS:
            if host in text:
                bad.append(f"{p.name}: names the font host {host}")
    return bad


def licence_problems(site: Path) -> list[str]:
    fonts = site / "fonts"
    bad = []
    for f in sorted(fonts.glob("*.woff2")):
        lic = fonts / f"{f.name.split('-')[0]}-OFL.txt"
        if not lic.is_file():
            bad.append(f"{f.name}: no {lic.name} beside it")
        elif "SIL OPEN FONT LICENSE VERSION 1.1" not in lic.read_text().upper():
            bad.append(f"{lic.name}: is not the SIL OFL 1.1 text")
    return bad


def weight_problems(site: Path) -> list[str]:
    total = sum(f.stat().st_size for f in (site / "fonts").glob("*.woff2"))
    return [f"fonts weigh {total} bytes, over the {BUDGET_BYTES} budget"] if total > BUDGET_BYTES else []


def swap_problems(site: Path) -> list[str]:
    bad = []
    faces = [(p.name, b) for p in _text_files(site) for b in _FACE.findall(p.read_text())]
    if not faces:
        bad.append("no @font-face declared in site/")
    for name, body in faces:
        if not re.search(r"font-display\s*:\s*swap\b", body):
            bad.append(f"{name}: an @font-face without font-display: swap")
    return bad


def _tree(tmp_path, *, src="fonts/A-Bold.woff2", swap=True, licence=True, size=10):
    site = tmp_path / "site"
    (site / "fonts").mkdir(parents=True)
    d = "font-display: swap;" if swap else ""
    (site / "index.html").write_text(
        f'<style>@font-face {{ font-family: A; {d} src: url("{src}") format("woff2"); }}</style>')
    (site / "fonts" / "A-Bold.woff2").write_bytes(b"x" * size)
    if licence:
        (site / "fonts" / "A-OFL.txt").write_text("SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007")
    return site


# --- the real site -------------------------------------------------------------------------

def test_the_site_ships_fonts():
    assert list((SITE / "fonts").glob("*.woff2")), "ADR-0026 commits WOFF2 files to site/fonts/"


def test_fonts_are_same_origin():
    assert external_font_problems(SITE) == []


def test_every_font_has_its_ofl_licence_beside_it():
    assert licence_problems(SITE) == []


def test_total_font_weight_is_bounded():
    assert weight_problems(SITE) == []


def test_every_font_face_swaps():
    assert swap_problems(SITE) == []


# --- planted controls: each check must go red on the defect it names ------------------------

def test_control_clean_tree_is_green(tmp_path):
    site = _tree(tmp_path)
    assert external_font_problems(site) == licence_problems(site) == []
    assert weight_problems(site) == swap_problems(site) == []


def test_control_an_external_font_face_url_is_red(tmp_path):
    for i, src in enumerate(("https://example.com/a.woff2", "//cdn.example.com/a.woff2", "/abs/a.woff2")):
        assert external_font_problems(_tree(tmp_path / str(i), src=src)) != [], src


def test_control_a_google_fonts_stylesheet_link_is_red(tmp_path):
    site = _tree(tmp_path)
    (site / "index.html").write_text(
        '<link href="https://fonts.googleapis.com/css2?family=Geist" rel="stylesheet">')
    assert external_font_problems(site) != []


def test_control_a_font_without_a_licence_is_red(tmp_path):
    assert licence_problems(_tree(tmp_path, licence=False)) != []


def test_control_a_licence_that_is_not_the_ofl_is_red(tmp_path):
    site = _tree(tmp_path)
    (site / "fonts" / "A-OFL.txt").write_text("all rights reserved")
    assert licence_problems(site) != []


def test_control_an_oversized_font_is_red(tmp_path):
    assert weight_problems(_tree(tmp_path, size=BUDGET_BYTES + 1)) != []


def test_control_a_face_without_swap_is_red(tmp_path):
    assert swap_problems(_tree(tmp_path, swap=False)) != []
