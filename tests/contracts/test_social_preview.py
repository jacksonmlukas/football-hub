"""The link-preview card and the two inbound links (#489, option A).

* `site/index.html` carries the Open Graph and Twitter tags, `og:image` is absolute, and the image
  it names resolves to a committed file that is exactly 1200x630 (what `summary_large_image`
  expects; a different size is cropped or dropped by the scrapers);
* the README's opening line links to the site, and `docs/method.md` links to the Ledger view.

Each check has a planted control run on text it should reject, so a regex that matches anything
cannot pass: see `test_the_checks_reject_a_broken_page`.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
PAGES_ROOT = "https://jacksonmlukas.github.io/football-hub/"
REQUIRED = (
    ("property", "og:title"), ("property", "og:description"), ("property", "og:image"),
    ("property", "og:type"), ("property", "og:url"),
    ("name", "twitter:card"), ("name", "twitter:title"), ("name", "twitter:description"),
    ("name", "twitter:image"),
)


def meta_tags(html: str) -> dict[str, str]:
    """`{property-or-name: content}` for every <meta> in `html` that has both."""
    out: dict[str, str] = {}
    for tag in re.findall(r"<meta\b[^>]*>", html):
        key = re.search(r'\b(?:property|name)="([^"]+)"', tag)
        val = re.search(r'\bcontent="([^"]*)"', tag)
        if key and val:
            out[key.group(1)] = val.group(1)
    return out


def png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", head[16:24])


def problems(html: str) -> list[str]:
    tags = meta_tags(html)
    found = [f"missing {k}" for _, k in REQUIRED if not tags.get(k)]
    if tags.get("twitter:card") != "summary_large_image":
        found.append("twitter:card is not summary_large_image")
    for k in ("og:image", "twitter:image"):
        v = tags.get(k, "")
        if v and not v.startswith(PAGES_ROOT):
            found.append(f"{k} is not an absolute URL under {PAGES_ROOT}")
    if tags.get("og:image") != tags.get("twitter:image"):
        found.append("og:image and twitter:image differ")
    return found


def test_the_page_carries_the_preview_tags():
    assert problems((SITE / "index.html").read_text()) == []


def test_the_image_the_tags_name_is_committed_and_1200_by_630():
    image = meta_tags((SITE / "index.html").read_text())["og:image"]
    path = SITE / image[len(PAGES_ROOT):]
    assert path.is_file(), f"og:image names {image}, which is not a file under site/"
    assert png_size(path) == (1200, 630)


def test_the_readme_opens_with_a_link_to_the_site():
    first = next(ln for ln in (ROOT / "README.md").read_text().splitlines()[1:] if ln.strip())
    assert f"]({PAGES_ROOT})" in first, f"README opening line has no link to the site: {first!r}"


def test_method_links_to_the_ledger_view():
    assert f"]({PAGES_ROOT}#/ledger)" in (ROOT / "docs" / "method.md").read_text()


def test_the_checks_reject_a_broken_page():
    good = (SITE / "index.html").read_text()
    drop = lambda s, key: re.sub(rf'<meta [^>]*"{re.escape(key)}"[^>]*>', "", s)  # noqa: E731
    for _, key in REQUIRED:
        assert problems(drop(good, key)), f"dropping {key} was not caught"
    assert problems(good.replace(f'content="{PAGES_ROOT}og-image.png"', 'content="og-image.png"'))
    assert problems(good.replace("summary_large_image", "summary"))
    assert problems(good.replace(f'"og:image" content="{PAGES_ROOT}og-image.png"',
                                 f'"og:image" content="{PAGES_ROOT}other.png"'))
    assert meta_tags("<meta property=x>") == {}
