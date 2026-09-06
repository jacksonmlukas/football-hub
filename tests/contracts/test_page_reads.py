"""What `site/index.html` reads off each artifact, recorded, so a producer cannot drop it.

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

**Asserted against the committed artifacts**, which is what the page will actually be served.
The honest limit of that, stated rather than left for a reader to discover: a producer change
is caught when the site is next published, not at the moment the producer changes. Running
every producer here instead would need each one's inputs stubbed, and a stub that pairs a
producer with a payload it could not have produced is the fixture shape this repo has an
incident about. The committed file is the real thing.

**The list is tied to the page in both directions.** A key recorded here that the page does
not read is dead weight that will outlive its panel; an artifact the page loads and this file
does not record is the gap being closed. Both are asserted below, so the record cannot quietly
cover less than it did yesterday.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site" / "data"
PAGE = ROOT / "site" / "index.html"

# artifact -> the JS name the page binds it to, and the keys it reads off that name.
READS: dict[str, tuple[str, tuple[str, ...]]] = {
    "roster": ("ros", ("generated_at", "n", "rows", "gain", "set_total", "optimal_total",
                       "withheld", "start", "sit")),
    "draft_board": ("boardArt", ("generated_at", "rows")),
    "track_record": ("tr", ("generated_at", "n_scored", "n_preregistered", "brier",
                            "log_loss", "bins", "seasons", "note")),
    "survivor": ("surv", ("rows", "season", "spent", "survival", "weeks_played",
                          "weeks_remaining", "unpriced_weeks", "snapshot_only_weeks",
                          "unconfirmed")),
    "manifest": ("man", ("artifacts", "generated_at", "season", "week")),
}

# Panels whose artifact this file does not pin, each with the reason. An absence is how a
# record stops covering something without anyone deciding to; a named entry is a decision.
NOT_PINNED_HERE: dict[str, str] = {
    "live": "`test_live_artifact_shape.py` already records it, and pins both of its writers "
            "against each other -- which is more than this file does. Duplicating the list "
            "here would give two records free to disagree.",
    "preds": "the weekly slate is one file per week (`preds_2026_wk01.json`), so there is no "
             "fixed name to read here. The page finds it through the manifest, whose own keys "
             "are recorded above.",
}


def _published(name: str) -> dict:
    matches = sorted(SITE.glob(f"{name}.json"))
    if not matches:
        pytest.skip(f"{name}.json is not published in this checkout")
    return json.loads(matches[0].read_text())


@pytest.mark.parametrize("artifact", sorted(READS))
def test_the_published_artifact_carries_what_the_page_reads(artifact):
    """The property `live` has had since two producers disagreed about its shape."""
    _js, keys = READS[artifact]
    got = _published(artifact)
    missing = [k for k in keys if k not in got]
    assert not missing, (
        f"the page reads {missing} off `{artifact}` and the published artifact does not "
        f"carry them, so that panel renders blank or takes a fallback branch")


@pytest.mark.parametrize("artifact", sorted(READS))
def test_the_page_actually_reads_what_is_recorded(artifact):
    """A key recorded and not read is dead weight that outlives its panel, and it makes the
    record look more complete than it is."""
    js, keys = READS[artifact]
    page = PAGE.read_text()
    read = set(re.findall(rf"\b{re.escape(js)}\??\.([A-Za-z_]+)", page))
    unread = sorted(set(keys) - read)
    assert not unread, (
        f"recorded as read off `{artifact}` but `{js}.<key>` never appears in the page: "
        f"{unread}")


def test_every_artifact_the_page_loads_is_recorded_or_explained():
    """The gap this file closes, held open. An artifact the page loads and nothing records is
    exactly the state six panels were in."""
    loaded = set(re.findall(r'await load\("([a-z_]+)"\)', PAGE.read_text()))
    accounted = set(READS) | set(NOT_PINNED_HERE)
    assert loaded <= accounted, (
        f"the page loads artifacts nothing records: {sorted(loaded - accounted)}. Add the "
        f"keys it reads to READS, or say in NOT_PINNED_HERE why they are pinned elsewhere.")


def test_nothing_is_recorded_for_an_artifact_that_is_gone():
    stale = sorted(set(READS) - {p.stem for p in SITE.glob("*.json")})
    assert not stale, f"recorded reads for artifacts that are not published: {stale}"
