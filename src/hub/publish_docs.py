"""Publish the documents the Ledger links to, beside the site (#541).

The Ledger links each verdict to its ADR, and the footer to `docs/method.md` and
`docs/track-record.md`. The repository is private, so a link into it is dead for a reader
arriving from a résumé. The Pages build therefore runs this module: it writes each of those
documents under `site/docs/` as a small self-contained HTML page, and the Ledger links to that
copy, relative to the site.

**Why HTML and not the `.md` served as-is.** GitHub Pages sends Markdown as `text/markdown`,
which browsers download rather than show. The copy is the document's own text in a wrapped
`<pre>`: no renderer to depend on, nothing to drift, and readable at 390px. A link from one
document to another unpublished one (`weekly-screen.md`) is left as text for the same reason
issue references are: it points at something outside the published set.

**What is published is a closed set**, `published()`: every `docs/adr/*.md`, the method and the
track record. A new ADR is published by existing. The sweep that cleared the set for
publication (secrets, addresses, pool and league member names, personal data) was done once by
hand for #541; a new document is covered by it only if its author re-runs the same sweep, which
is why the set is named here rather than globbing all of `docs/`.

Nothing under `site/docs/` is committed (see `.gitignore`): it is built at deploy time from the
committed sources, so there is no second copy to edit.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

from hub import atomic

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"

# Published documents outside docs/adr/. Adding one is the decision to publish it.
EXTRA = ("docs/method.md", "docs/track-record.md")

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root {{ --bg: #f6f5ef; --fg: #1b1f1c; --muted: #596058; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #0f1310; --fg: #e8ebe6; --muted: #9aa49c; }} }}
body {{ margin: 0 auto; max-width: 52rem; padding: 16px; background: var(--bg); color: var(--fg);
  font: 15px/1.5 system-ui, sans-serif; }}
a {{ color: inherit; }}
p.back {{ color: var(--muted); }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: 13.5px/1.55 ui-monospace, Menlo,
  monospace; }}
</style></head><body>
<p class="back"><a href="{back}">&larr; Back to the Ledger</a></p>
<pre>{text}</pre>
</body></html>
"""


def published(root: Path | None = None) -> list[str]:
    """Repo-relative paths of the documents the site publishes, sorted."""
    root = root or ROOT
    adrs = sorted(p.relative_to(root).as_posix() for p in (root / "docs" / "adr").glob("*.md"))
    return adrs + [p for p in EXTRA if (root / p).is_file()]


def target(source: str) -> str:
    """Where a source document lands, relative to the site root: `docs/x.md` -> `docs/x.html`."""
    return source.removesuffix(".md") + ".html"


def build(site: Path | None = None, root: Path | None = None) -> list[str]:
    """Write every published document under `site`; return the site-relative paths written."""
    site, root = site or SITE, root or ROOT
    written = []
    for source in published(root):
        text = (root / source).read_text(encoding="utf-8")
        title = next((ln.lstrip("# ").strip() for ln in text.splitlines()
                      if ln.startswith("#")), Path(source).stem)
        out = site / target(source)
        depth = len(Path(target(source)).parts) - 1
        atomic.write_text(out, _PAGE.format(title=html.escape(title), text=html.escape(text),
                                            back="../" * depth + "#/ledger"))
        written.append(target(source))
    return written


if __name__ == "__main__":
    # Inline rather than a `main()`: this reads committed files and takes no input that can be
    # absent, so it has no refusal for `test_cli_surface`'s absent-input drive to exercise.
    _written = build()
    print(f"  published {len(_written)} documents under site/docs/")
    sys.exit(0 if _written else 1)
