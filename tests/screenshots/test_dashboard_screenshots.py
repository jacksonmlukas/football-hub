"""Pixel baselines for the dashboard: two widths x two colour schemes (#475).

Lands before the module split so that split can prove it changed no pixel. Run with
`uv run pytest -m screenshots tests/screenshots`; the default runs deselect the marker, because
this needs a browser (`uv run playwright install chromium`, once).

What is held still, so a baseline moves only when the page does:
  * the data: a frozen copy of week 6's artifacts under `tests/fixtures/site/data`, never
    `site/data` (which a nightly commit rewrites) and never `data/`;
  * the clock: Playwright's `page.clock.set_fixed_time`, which pins `Date.now()` and
    `new Date()` in the page before its scripts run. The page needs no test hook, so
    production behaviour is untouched;
  * the network: the page is served from 127.0.0.1 by a fixture thread, and any request to
    another host is aborted and fails the test;
  * animation: the screenshot is taken with CSS animations/transitions finished.

Baselines live in a directory per OS (`tests/golden/screenshots/<darwin|linux>/`). Chromium's
text rasterisation differs between operating systems by more than any sane tolerance, and the
page's system font stack differs with it, so one set of PNGs cannot serve a laptop and the CI
runner.

Updating is explicit: `UPDATE_SCREENSHOTS=1 uv run pytest -m screenshots tests/screenshots`
rewrites the baselines for this OS. Nothing writes them otherwise; a missing baseline fails.
"""
from __future__ import annotations

import http.server
import os
import socketserver
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image, ImageChops
from playwright.sync_api import Browser, sync_playwright

pytestmark = pytest.mark.screenshots

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
FIXTURE_DATA = ROOT / "tests" / "fixtures" / "site" / "data"
OS_KEY = "darwin" if sys.platform == "darwin" else "linux"
BASELINES = ROOT / "tests" / "golden" / "screenshots" / OS_KEY
OUT = ROOT / "test-results" / "screenshots"

# After the manifest's last artifact (2026-10-17T16:58Z), so every age renders as a fixed
# number of hours rather than a day count that would roll over.
NOW = "2026-10-17T18:00:00Z"

VIEWPORTS = {"1280": (1280, 900), "390": (390, 844)}
Scheme = Literal["light", "dark"]
SCHEMES: tuple[Scheme, ...] = ("light", "dark")

# A pixel counts as changed when any channel moves by more than CHANNEL_TOLERANCE (of 255);
# the image fails when more than MAX_CHANGED_FRACTION of its pixels changed.
#   * Same machine, same Chromium, same fixtures: renders are bit-identical, so these only
#     absorb what a Chromium patch release does to glyph anti-aliasing: a few edge pixels
#     of text, each off by a few levels.
#   * 8/255 is below the step of a one-hex-digit change to a colour token (16/255), so a
#     token edit is never inside the tolerance.
#   * 0.05% of a 1280-wide page is a few thousand pixels: more than glyph-edge noise, far
#     less than one moved or recoloured element.
CHANNEL_TOLERANCE = 8
MAX_CHANGED_FRACTION = 0.0005


class _Handler(http.server.SimpleHTTPRequestHandler):
    """`site/` as the web root, except `/data/` which is the frozen fixture set."""

    def translate_path(self, path: str) -> str:
        rel = path.split("?", 1)[0].split("#", 1)[0].lstrip("/")
        if rel.startswith("data/"):
            return str(FIXTURE_DATA / rel[len("data/"):])
        return str(SITE / rel)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    with sync_playwright() as p:
        b = p.chromium.launch()
        try:
            yield b
        finally:
            b.close()


def capture(browser: Browser, base_url: str, width: str, scheme: Scheme, now: str = NOW) -> Path:
    """Render the dashboard and return the path of the PNG, under `test-results/`."""
    w, h = VIEWPORTS[width]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{OS_KEY}-{width}-{scheme}-{now.replace(':', '')}.png"
    context = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=1)
    try:
        page = context.new_page()
        page.emulate_media(color_scheme=scheme, reduced_motion="reduce")
        page.clock.set_fixed_time(now)
        foreign: list[str] = []

        def gate(route) -> None:  # anything that is not this test's own server is a failure
            url = route.request.url
            if url.startswith(base_url) or url.startswith("data:"):
                route.continue_()
            else:
                foreign.append(url)
                route.abort()

        page.route("**/*", gate)
        page.goto(base_url + "/index.html")
        # `render()` writes the subtitle once the manifest has loaded; the panels follow.
        page.wait_for_function(
            "document.getElementById('subtitle').textContent.includes('week')")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(out), full_page=True, animations="disabled", caret="initial")
        assert not foreign, f"the page reached off-host: {foreign}"
    finally:
        context.close()
    return out


def changed_fraction(a: Path, b: Path, diff_to: Path | None = None) -> float:
    """Share of pixels whose largest channel difference exceeds CHANNEL_TOLERANCE."""
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    if ia.size != ib.size:
        return 1.0
    channels = [
        c.point(lambda v: 255 if v > CHANNEL_TOLERANCE else 0)
        for c in ImageChops.difference(ia, ib).split()
    ]
    mask = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    changed = mask.histogram()[255]
    if diff_to is not None and changed:
        # Differing pixels in red over a dimmed copy of the actual image.
        shown = ib.point(lambda v: v // 3)
        shown.paste(Image.new("RGB", ib.size, (255, 0, 0)), mask=mask)
        shown.save(diff_to)
    return changed / (ia.size[0] * ia.size[1])


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("width", VIEWPORTS)
def test_dashboard_matches_baseline(browser, base_url, width, scheme):
    actual = capture(browser, base_url, width, scheme)
    baseline = BASELINES / f"{width}-{scheme}.png"
    if os.environ.get("UPDATE_SCREENSHOTS") == "1":
        BASELINES.mkdir(parents=True, exist_ok=True)
        baseline.write_bytes(actual.read_bytes())
        return
    assert baseline.exists(), (
        f"no baseline for {OS_KEY}/{baseline.name}. Baselines are per OS; create them "
        "deliberately with UPDATE_SCREENSHOTS=1 and commit them."
    )
    diff = OUT / f"{OS_KEY}-{width}-{scheme}-diff.png"
    frac = changed_fraction(baseline, actual, diff)
    assert frac <= MAX_CHANGED_FRACTION, (
        f"{width}px {scheme}: {frac:.4%} of pixels changed (limit {MAX_CHANGED_FRACTION:.2%}). "
        f"actual: {actual}; diff: {diff}. If the change is intended, regenerate with "
        "UPDATE_SCREENSHOTS=1 and commit the new baselines in the same commit."
    )


def test_the_pinned_clock_is_what_fixes_the_ages(browser, base_url):
    """Control for the clock: the same pinned time twice is byte-identical, and a later pinned
    time renders differently, so the stability above is the pin's doing and not because the
    page never shows an age."""
    a = capture(browser, base_url, "390", "light")
    b = capture(browser, base_url, "390", "light")
    later = capture(browser, base_url, "390", "light", now="2026-10-20T18:00:00Z")
    assert a.read_bytes() == b.read_bytes()
    assert changed_fraction(a, later) > 0, "ages did not respond to the clock"
