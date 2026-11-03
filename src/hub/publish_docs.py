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

**What is published is a closed set**, `SWEPT`: each document cleared for publication, with the
digest of the text that was cleared. The sweep (secrets, addresses, pool and league member
names, personal data) was done by hand for #541. A document is published only while its text
still has that digest, so neither a new ADR nor an edit to a published one reaches the site
until someone re-runs the sweep and records the new digest here (`python -m hub.publish_docs
--digests` prints them). Every ADR is in `SWEPT` or `NOT_PUBLISHED`; the contract suite fails
on one in neither, so holding a document back is a decision too, not an omission.

**A held document is a page saying so, not a 404 and not a failed deploy.** A listed document
that is absent, or whose text no longer has its digest, is written as a short notice in place
of its text. The Ledger's link still lands, nothing unswept is served, and the deploy goes on
refreshing scores (the graceful-degradation rule in CLAUDE.md). The run prints a warning
naming it, and the contract suite is red until the digest is recorded.

Nothing under `site/docs/` is committed (see `.gitignore`): it is built at deploy time from the
committed sources, so there is no second copy to edit.
"""

from __future__ import annotations

import hashlib
import html
import sys
from collections.abc import Collection
from pathlib import Path

from hub import atomic
from hub.paths import ROOT

SITE = ROOT / "site"

# Each published document and the sha256 (first 12 hex) of the text the sweep cleared. Adding
# a row, or changing a digest, is the statement that the sweep was re-run on that text.
SWEPT: dict[str, str] = {
    "docs/adr/0001-duckdb-query-layer.md": "c8a8b0e85eca",
    "docs/adr/0002-one-forecaster-protocol.md": "bd5a5cf222e3",
    "docs/adr/0003-make-now-dagster-in-october.md": "ac80bb267e08",
    "docs/adr/0004-hydra-config-digest.md": "d4532525eef4",
    "docs/adr/0005-tiered-sync-polling.md": "f9bd66cd03ec",
    "docs/adr/0006-fitted-constants-live-with-their-provenance.md": "b03c2df5212c",
    "docs/adr/0007-measurements-that-steer-the-product-are-committed-code.md": "a06ae050c1a0",
    "docs/adr/0008-the-simulator-indexes-the-board.md": "641bb0bc603c",
    "docs/adr/0009-championship-equity-does-not-pick.md": "167351d75d40",
    "docs/adr/0010-edge-is-displayed-but-never-ranked-on.md": "9dbb96998904",
    "docs/adr/0011-the-pick-ranks-on-corrected-adp.md": "d6da4b88a768",
    "docs/adr/0012-the-lineup-optimiser-waits-for-real-variance.md": "a1f1069ddcfc",
    "docs/adr/0013-the-snap-trend-is-shown-and-never-ranked-on.md": "d8e27b9459dc",
    "docs/adr/0014-a-provisional-rule-may-act-where-no-gate-can-run.md": "ba913375eb27",
    "docs/adr/0015-the-weekly-gate-is-a-decision-not-an-accuracy-test.md": "58da340f14dd",
    "docs/adr/0016-the-weekly-projection-is-shown-and-never-ranked-on.md": "4231549ced29",
    "docs/adr/0017-the-market-usage-blend-is-a-model-not-a-shrinkage.md": "6a50f7c932a2",
    "docs/adr/0018-live-scores-are-not-part-of-the-record.md": "bff7c5c3e6a2",
    "docs/adr/0019-a-gate-requires-every-season.md": "69b48b1c3fc0",
    "docs/adr/0020-the-board-is-not-split-into-a-reader-and-a-builder.md": "cbe2376c2a77",
    "docs/adr/0021-a-correction-is-a-shape-not-a-module.md": "2810f99971e6",
    "docs/adr/0022-the-boards-xfp-stays-the-published-total.md": "aee58a974a3d",
    "docs/adr/0023-a-refused-injury-report-nulls-three-columns.md": "42f82eb91255",
    "docs/adr/0024-a-modelling-decision-becomes-agent-work-when-its-alternatives-can-be-measured.md":
        "ada7eba5837c",
    "docs/adr/0025-systematic-bias-bounds-the-bottom-up-plan-and-seven-angles-are-declined.md":
        "211da01b2f51",
    "docs/adr/0026-self-hosted-fonts.md": "191321c0bf7e",
    "docs/adr/0027-a-consensus-capture-is-in-time-per-player.md": "9b70b0831b82",
    "docs/method.md": "41b7375639a9",
    "docs/track-record.md": "9667ed99d568",
}

# ADRs deliberately kept off the site. An ADR in neither this nor SWEPT fails the contract suite.
NOT_PUBLISHED: tuple[str, ...] = ()

_HELD = ("This document has changed since it was last cleared for publication, or has moved. "
         "It will be back here once it has been reviewed again.")

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


def digest(text: str) -> str:
    """The first 12 hex of the sha256 of `text`, as `SWEPT` records it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def published(swept: dict[str, str] | None = None) -> list[str]:
    """Repo-relative paths of the documents the site publishes, sorted."""
    return sorted(SWEPT if swept is None else swept)


def held(root: Path | None = None, swept: dict[str, str] | None = None) -> list[str]:
    """The listed documents that are absent or whose text is not the text that was swept."""
    root, swept = root or ROOT, SWEPT if swept is None else swept
    return [source for source in published(swept)
            if not (root / source).is_file()
            or digest((root / source).read_text(encoding="utf-8")) != swept[source]]


def target(source: str) -> str:
    """Where a source document lands, relative to the site root: `docs/x.md` -> `docs/x.html`."""
    return source.removesuffix(".md") + ".html"


def build(site: Path | None = None, root: Path | None = None,
          swept: dict[str, str] | None = None,
          holding: Collection[str] | None = None) -> list[str]:
    """Write every published document under `site`; return the site-relative paths written.
    A held document (`held`) is written as a notice in place of its text. `holding` is the
    result of `held(root, swept)` when the caller already has it, so a deploy reads and hashes
    each document once; left out, it is computed here."""
    site, root = site or SITE, root or ROOT
    holding = set(held(root, swept) if holding is None else holding)
    written = []
    for source in published(swept):
        if source in holding:
            title, text = Path(source).stem, _HELD
        else:
            text = (root / source).read_text(encoding="utf-8")
            title = next((ln.lstrip("# ").strip() for ln in text.splitlines()
                          if ln.startswith("#")), Path(source).stem)
        out = site / target(source)
        depth = len(Path(target(source)).parts) - 1
        atomic.write_text(out, _PAGE.format(title=html.escape(title), text=html.escape(text),
                                            back="../" * depth + "#/ledger"))
        written.append(target(source))
    return written


def run(site: Path | None = None, root: Path | None = None,
        swept: dict[str, str] | None = None) -> int:
    """The Pages step: build, warn on each held document, and fail only when nothing was written."""
    holding = held(root, swept)
    for source in holding:
        print(f"::warning::{source} is held: absent, or changed since it was swept "
              f"(record its digest in hub.publish_docs.SWEPT after re-running the sweep)")
    written = build(site, root, swept, holding)
    print(f"  published {len(written)} documents under site/docs/")
    return 0 if written else 1


def digests(root: Path | None = None, swept: dict[str, str] | None = None) -> int:
    """`--digests`: print each listed document's current digest as a `SWEPT` row, to record
    once the sweep has been re-run on that text."""
    root = root or ROOT
    for source in published(swept):
        path = root / source
        sha = digest(path.read_text(encoding="utf-8")) if path.is_file() else "absent"
        print(f'    "{source}": "{sha}",')
    return 0


if __name__ == "__main__":
    # Inline rather than a `main()`: this reads committed files and takes no input that can be
    # absent, so it has no refusal for `test_cli_surface`'s absent-input drive to exercise.
    sys.exit(digests() if sys.argv[1:] == ["--digests"] else run())
