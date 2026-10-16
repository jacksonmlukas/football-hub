"""The two page shapes `hub.fetch.bigten.parse_article` reads, held to frozen pages (#499).

The conference's page changed under the parser on 2026-10-07: the Next.js `__NEXT_DATA__` CMS
record became a server-rendered Sidearm article whose content is an iframe onto the report
vendor. Both pages are frozen here -- the Sidearm one trimmed from a real capture
(`archive/bigten/availability/2026/page/4814b76d3a0e98c2.html`, the article region whole), the
Next.js one hand-built because the archive holds none (every 2026 page is the new shape).
A third shape must raise, or the next redesign is archived green.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hub.fetch import bigten

FIXTURES = Path(__file__).resolve().parents[1] / "golden" / "fixtures"
SIDEARM = (FIXTURES / "bigten_page_sidearm.html").read_bytes()
NEXTDATA = (FIXTURES / "bigten_page_nextdata.synthetic.html").read_bytes()


def test_the_sidearm_page_parses_to_its_title_and_the_vendor_embeds():
    art = bigten.parse_article(SIDEARM)
    assert art.title == "2026 Big Ten Football Availability Reports"
    assert art.updated_at is None
    # The safelinks wrapper is unwrapped and the duplicate vendor roots both survive.
    assert art.embeds == (
        "https://app.hdintelligence.com/?source=B10&sport=Football&conf=B10&type=report",
        "https://app.hdintelligence.com/?source=B10&sport=Football&conf=B10&type=archive",
    )
    assert art.links() == []
    assert art.record["shape"] == "sidearm"


def test_the_nextdata_page_still_parses():
    art = bigten.parse_article(NEXTDATA)
    assert art.title == "2026 FB Availability Reports"
    assert art.updated_at == "2026-09-17T00:12:44.000Z"
    assert art.links() == [("https://bigten.org/api/media/file/abc-Week_4.pdf",
                            "Week #4 (Sept. 18-19)")]
    assert art.embeds == ()


@pytest.mark.parametrize("mangle", [
    # planted: the headline is renamed, the template is otherwise intact
    lambda b: b.replace(b'class="article-headline"', b'class="headline"'),
    # planted: the same template carrying a different article
    lambda b: b.replace(b"Availability Reports", b"Spring Game Recap"),
    # planted: the content div is gone
    lambda b: b.replace(b'class="article-content"', b'class="body"'),
    # planted: the article wrapper is gone, and so is the Next.js record
    lambda b: b.replace(b"article-wrapper", b"wrapper"),
])
def test_a_page_of_neither_shape_raises(mangle):
    mangled = mangle(SIDEARM)
    assert mangled != SIDEARM, "the mangling changed nothing, so the control proves nothing"
    with pytest.raises(bigten.PageShapeChanged):
        bigten.parse_article(mangled)
