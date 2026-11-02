"""#489: render the site's 1200x630 Open Graph / Twitter preview, `site/og-image.png`.

The picture is the This week view, light scheme, drawn from the same frozen fixture data
(`tests/fixtures/site/data`), the same pinned clock and the same off-host gate as the dashboard
pixel baselines (`tests/screenshots/test_dashboard_screenshots.py`), whose server and clock it
imports rather than copies. Not `site/data`, so the image changes only when this script or the
page does, never because a nightly commit moved a number.

    uv run python scripts/render_og_image.py            # rewrites site/og-image.png
    uv run python scripts/render_og_image.py --out PATH # somewhere else, e.g. to compare

Needs the Chromium Playwright already uses (`uv run playwright install chromium`, once).
Re-run it and commit the PNG when the This week view changes enough that the card misleads.
Text rasterisation is per OS, so the committed file is whichever machine ran this last; the
contract (`tests/contracts/test_social_preview.py`) holds its size, not its pixels.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "screenshots"))

from playwright.sync_api import sync_playwright  # noqa: E402
from test_dashboard_screenshots import NOW, _Handler, _Server  # noqa: E402

WIDTH, HEIGHT = 1200, 630
DEFAULT_OUT = ROOT / "site" / "og-image.png"


def render(out: Path) -> None:
    server = _Server(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    foreign: list[str] = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(
                    viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
                page = context.new_page()
                page.emulate_media(color_scheme="light", reduced_motion="reduce")
                page.clock.set_fixed_time(NOW)

                def keep_on_host(route) -> None:
                    url = route.request.url
                    if url.startswith(base_url) or url.startswith("data:"):
                        route.continue_()
                    else:
                        foreign.append(url)
                        route.abort()

                page.route("**/*", keep_on_host)
                page.goto(f"{base_url}/index.html#/week")
                page.wait_for_function(
                    "document.getElementById('subtitle').textContent.includes('week')")
                page.wait_for_load_state("networkidle")
                out.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(out), animations="disabled", caret="initial",
                                clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT})
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    if foreign:
        out.unlink(missing_ok=True)
        raise SystemExit(f"the page reached off-host: {foreign}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    render(args.out)
    print(f"wrote {args.out} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
