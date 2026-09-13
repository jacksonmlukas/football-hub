"""The Big Ten availability archive (#215).

Everything here runs against a page built by hand from the shape `bigten.org` served on
2026-09-11 -- a Next.js `__NEXT_DATA__` blob holding one CMS article record -- and against a
transport that never leaves the process. The 2026 page held no report on that day, so what
the four-a-week regime does to the body is a guess until it starts; these tests are about
what the module *keeps* and what it *records* under every outcome, not about parsing a
report nobody has seen.

Nothing here spends a CFBD call: the transport in `hub.fetch.cfbd` is patched where a test
wants a lines snapshot, and refuses on its own where a test forgets.
"""
import json
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from hub.contracts import ContractViolation
from hub.fetch import bigten, cfbd

# --- the world ------------------------------------------------------------------

# Thursday 17 September 2026, 01:41 UTC: eleven minutes after the evening capture instant that
# follows the regime's first report deadline (8pm ET Wednesday), which is the shape of a run
# GitHub delivered on time.
FIRST_RUN = datetime(2026, 9, 17, 1, 41, tzinfo=UTC)

PDF = b"%PDF-1.4 fake week four"


def page_with(body: str, updated: str = "2026-09-17T00:12:44.000Z",
              title: str = "2026 FB Availability Reports") -> bytes:
    record = {"_content_type_uid": "article", "title": title, "updatedAt": updated,
              "body": body, "id": 60323}
    data = {"props": {"pageProps": {"id": 60323, "fallback": {"60323": record}}}}
    return (b"<html><head></head><body><nav>Big Ten</nav>"
            b'<script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(data).encode()
            + b"</script></body></html>")


ONE_LINK = ('<div class="payload-richtext"><p><a href="/api/media/file/abc-Week_4.pdf">'
            "Week&nbsp;#4 <strong>(Sept. 18-19)</strong></a></p></div>")


@pytest.fixture(autouse=True)
def nothing_real_is_written(tmp_path, monkeypatch):
    """Every default write path, pointed at the test's own directory -- including CFBD's,
    because a lines snapshot goes through its cache and its counter."""
    monkeypatch.setattr(bigten, "ARCHIVE", tmp_path / "archive")
    monkeypatch.setattr(bigten, "INDEX", tmp_path / "archive" / "captures.json")
    monkeypatch.setattr(bigten, "LINES", tmp_path / "lines")
    monkeypatch.setattr(bigten, "STATUS", tmp_path / "site" / "bigten.json")
    monkeypatch.setattr(cfbd, "CACHE", tmp_path / "cfbd-cache")
    monkeypatch.setattr(cfbd, "QUOTA", tmp_path / "state" / "cfbd-quota.json")
    monkeypatch.setattr(cfbd, "STATUS", tmp_path / "site" / "cfbd.json")
    cfbd.reset_run_budget()


@pytest.fixture
def clock(monkeypatch):
    """The module's one clock, set to a moment rather than read off the machine."""
    def _set(when: datetime) -> datetime:
        monkeypatch.setattr(bigten, "_now", lambda: when)
        return when
    return _set


@pytest.fixture
def season(monkeypatch):
    """A configured season: the college week counts from the 5th of September."""
    values = {"CFB_WEEK_ONE": "2026-09-05"}
    monkeypatch.setattr(cfbd, "_env", lambda: values)
    return values


@pytest.fixture
def web(monkeypatch):
    """The conference's site, as a dict of url -> bytes; a url not in it is unreachable."""
    pages: dict[str, bytes] = {}
    calls: list[str] = []

    def _get(url: str, cap: int | None = None) -> bytes:
        calls.append(url)
        if url not in pages:
            raise OSError(f"unreachable: {url}")
        got = pages[url]
        if isinstance(got, BaseException):
            raise got
        return got
    monkeypatch.setattr(bigten, "_get", _get)
    pages["calls"] = calls  # type: ignore[assignment]
    return pages


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.setattr(cfbd, "_api_key", lambda: None)


@pytest.fixture
def cfbd_lines(monkeypatch):
    """A CFBD that answers `/lines` with one priced game and records what it was asked."""
    calls: list[tuple[str, dict]] = []

    def _fake(path, params, key):
        calls.append((path, dict(params)))
        return [{"id": 1, "season": 2026, "week": 3, "homeTeam": "Rutgers",
                 "awayTeam": "USC", "lines": [{"provider": "x", "spread": -7.5}]}]
    monkeypatch.setattr(cfbd, "_http_get", _fake)
    monkeypatch.setattr(cfbd, "_api_key", lambda: "test-key")
    return calls


# --- deadlines ------------------------------------------------------------------------

def test_a_run_delivered_late_belongs_to_the_deadline_it_was_scheduled_after():
    """GitHub delivers a scheduled run anywhere up to two hours late. The capture is for
    the deadline, not for the wall-clock."""
    deadline, at = bigten.deadline_at(FIRST_RUN + timedelta(hours=2))
    assert (deadline.name, bigten.deadline_id(at)) == ("evening", "2026-09-17T0130Z")


def test_the_gameday_deadlines_are_saturday_and_the_evening_one_runs_into_sunday():
    sat = datetime(2026, 9, 19, tzinfo=UTC)
    early = bigten.deadline_at(sat.replace(hour=15, minute=30))
    late = bigten.deadline_at(sat.replace(hour=23))
    night = bigten.deadline_at(sat + timedelta(days=1, hours=2))
    assert [s.name for s, _ in (early, late, night)] == ["gameday-early", "gameday-late",
                                                         "evening"]
    assert bigten.deadline_id(night[1]) == "2026-09-20T0130Z"


def test_a_moment_before_the_deadline_belongs_to_the_previous_one():
    deadline, at = bigten.deadline_at(datetime(2026, 9, 17, 1, 29, tzinfo=UTC))
    assert bigten.deadline_id(at) == "2026-09-16T0130Z", "Wednesday's evening deadline, a day back"
    deadline, at = bigten.deadline_at(datetime(2026, 9, 15, 12, tzinfo=UTC))
    assert bigten.deadline_id(at) == "2026-09-13T0130Z", "Monday: Sunday's evening deadline, two back"
    assert deadline.name == "evening"


def test_a_run_at_the_deadline_instant_is_that_deadline_not_the_previous_one():
    """Review finding: the boundary was inclusive and nothing held it there. A run that
    fires at exactly 01:30:00 is the deadline it was scheduled for."""
    deadline, at = bigten.deadline_at(datetime(2026, 9, 17, 1, 30, tzinfo=UTC))
    assert (deadline.name, bigten.deadline_id(at)) == ("evening", "2026-09-17T0130Z")


def test_the_week_is_the_deadlines_week_not_the_wall_clocks(web, season):
    """Review finding: a run delivered after the Tuesday week boundary was attributed to
    the deadline before it -- correctly -- and then filed and priced under the week the
    clock said. The lines snapshot is the deadline's week, like the reports it sits beside."""
    # Week 1 opens Saturday 5 September, so the boundary into week 3 is Tuesday the 15th.
    # A run at noon that Tuesday belongs to Sunday's evening deadline, which is week 2.
    cap = bigten.capture(now=datetime(2026, 9, 15, 12, tzinfo=UTC), skip_lines=True)
    assert bigten.deadline_id(cap.at) == "2026-09-13T0130Z"
    assert cap.week == 2, "the capture is filed under the week the clock said, not its own"


def test_a_week_holds_seven_deadlines():
    got = bigten.expected_deadlines(bigten.REPORTS_BEGIN, bigten.REPORTS_BEGIN + timedelta(days=7))
    assert len(got) == 7
    assert bigten.deadline_id(got[0][1]) == "2026-09-17T0130Z", "the first capture follows the first deadline"
    assert [s.name for s, _ in got].count("evening") == 5


def test_every_deadline_has_a_cron_and_no_two_collide():
    crons = [s.cron() for s in bigten.DEADLINES]
    assert len(set(crons)) == len(crons)
    assert all(len(c.split()) == 5 for c in crons)


# --- the page ---------------------------------------------------------------------

def test_the_article_record_is_read_with_its_own_timestamp():
    art = bigten.parse_article(page_with(ONE_LINK))
    assert art.updated_at == "2026-09-17T00:12:44.000Z"
    assert art.title == "2026 FB Availability Reports"


def test_links_are_absolute_and_their_labels_are_text():
    art = bigten.parse_article(page_with(ONE_LINK))
    assert art.links() == [("https://bigten.org/api/media/file/abc-Week_4.pdf",
                            "Week #4 (Sept. 18-19)")]


@pytest.mark.parametrize("page,missing", [
    (b"<html>redesigned</html>", "__NEXT_DATA__"),
    (b'<script id="__NEXT_DATA__" type="application/json">{nope</script>', "not JSON"),
    (b'<script id="__NEXT_DATA__" type="application/json">{"props":{"pageProps":{}}}</script>',
     "fallback"),
    (b'<script id="__NEXT_DATA__" type="application/json">'
     b'{"props":{"pageProps":{"fallback":{"1":{"_content_type_uid":"image"}}}}}</script>',
     "no article record"),
    (b'<script id="__NEXT_DATA__" type="application/json">'
     b'{"props":{"pageProps":{"fallback":{"1":{"_content_type_uid":"article","title":"FB Availability Reports"}}}}}'
     b"</script>", "lacks"),
])
def test_a_page_of_another_shape_says_what_it_lacks(page, missing):
    with pytest.raises(bigten.PageShapeChanged, match=missing):
        bigten.parse_article(page)


# --- the archive ------------------------------------------------------------------

def test_a_capture_keeps_the_article_and_every_linked_document(web, season, tmp_path):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert cap.fetched and cap.parsed and cap.error is None
    kinds = [r["kind"] for r in cap.rows]
    assert kinds == ["article", "file"]
    kept = tmp_path / "archive" / cap.rows[1]["path"]
    assert kept.read_bytes() == PDF
    assert cap.rows[1]["label"] == "Week #4 (Sept. 18-19)"
    assert cap.rows[1]["report_updated_at"] == "2026-09-17T00:12:44.000Z"
    assert all(r["deadline"] == "2026-09-17T0130Z" for r in cap.rows)


def test_an_unchanged_document_is_indexed_again_and_not_stored_again(web, season, tmp_path):
    """The row is the evidence the deadline was checked; the bytes are the same bytes."""
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.capture(now=FIRST_RUN, skip_lines=True)
    second = bigten.capture(now=FIRST_RUN + timedelta(days=1), skip_lines=True)
    assert [r["new_content"] for r in second.rows] == [False, False]
    index = bigten.read_index()
    assert index.height == 4
    assert index["deadline"].n_unique() == 2
    stored = list((tmp_path / "archive" / "availability").rglob("*.*"))
    assert len(stored) == 2, f"one article record and one PDF, not {stored}"


def test_a_changed_document_is_a_new_file_beside_the_old_one(web, season, tmp_path):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.capture(now=FIRST_RUN, skip_lines=True)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF + b" revised"
    web[bigten.PAGE] = page_with(ONE_LINK, updated="2026-09-18T00:05:00.000Z")
    second = bigten.capture(now=FIRST_RUN + timedelta(days=1), skip_lines=True)
    assert [r["new_content"] for r in second.rows] == [True, True]
    pdfs = sorted((tmp_path / "archive" / "availability").rglob("*.pdf"))
    assert len(pdfs) == 2 and pdfs[0].read_bytes() != pdfs[1].read_bytes()


def test_a_page_the_parser_cannot_read_is_archived_whole(web, season, tmp_path):
    """The bytes survive the parser being wrong, which is the state the first live run
    may well find it in."""
    web[bigten.PAGE] = b"<html>the conference redesigned its site</html>"
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert cap.fetched and not cap.parsed
    assert isinstance(cap.error, bigten.PageShapeChanged)
    assert [r["kind"] for r in cap.rows] == ["page"]
    assert (tmp_path / "archive" / cap.rows[0]["path"]).read_bytes() == web[bigten.PAGE]


def test_one_unreachable_document_does_not_lose_the_others(web, season, capsys):
    two = ONE_LINK.replace("</p>", '<a href="/api/media/file/gone.pdf">gone</a></p>')
    web[bigten.PAGE] = page_with(two)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert [r["kind"] for r in cap.rows] == ["article", "file"]
    assert isinstance(cap.error, OSError)
    assert "gone.pdf" in capsys.readouterr().err


def test_the_index_is_refused_when_a_row_breaks_the_contract(tmp_path):
    bad = [{"deadline": "2026-09-17T0130Z", "deadline_name": "evening", "captured_at": "x",
            "season": 2019, "kind": "file", "url": "u", "label": None,
            "report_updated_at": None, "sha256": "s", "bytes": 1, "path": "p",
            "new_content": True}]
    (tmp_path / "archive").mkdir()
    bigten.INDEX.write_text(json.dumps(bad))
    with pytest.raises(ContractViolation, match="season"):
        bigten.read_index()


def test_an_absent_index_is_an_empty_checked_frame():
    got = bigten.read_index()
    assert got.height == 0 and list(got.columns) == list(bigten.INDEX_SCHEMA)


# --- the odds snapshot --------------------------------------------------------------

def test_the_snapshot_is_one_bulk_call_for_the_week(web, season, cfbd_lines, tmp_path):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN)
    assert cfbd_lines == [("/lines", {"year": 2026, "week": 3})], (
        "one /lines call, year and week scoped, and never a team")
    assert cap.lines_rows == 1 and cap.lines_why is None
    lines = [r for r in cap.rows if r["kind"] == "lines"]
    assert len(lines) == 1
    assert (tmp_path / "lines" / lines[0]["path"]).exists()
    assert lines[0]["path"] == "2026/w03/2026-09-17T0130Z.parquet"


def test_a_snapshot_is_taken_afresh_at_every_deadline(web, season, cfbd_lines):
    """A cached week is the price at an earlier deadline. `refresh=True` is the point."""
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.capture(now=FIRST_RUN)
    bigten.capture(now=FIRST_RUN + timedelta(days=1))
    assert len(cfbd_lines) == 2


def test_a_refused_refresh_that_served_the_cache_is_not_this_deadline_s_price(
        web, season, cfbd_lines, monkeypatch):
    """`cfbd.bulk` serves the cached week when it may not spend. That is the right thing
    for a price and the wrong thing to *call* a snapshot: the row is kept and the stamp says
    it is an earlier capture."""
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    first = bigten.capture(now=FIRST_RUN)
    assert first.lines_why is None
    monkeypatch.setattr(cfbd, "MAX_CALLS_PER_RUN", 1)       # the next refresh is refused
    second = bigten.capture(now=FIRST_RUN + timedelta(days=1))
    assert len(cfbd_lines) == 1
    assert second.lines_rows == 1
    assert second.lines_why is not None and "earlier capture" in second.lines_why
    got = bigten.record_run(second, now=FIRST_RUN + timedelta(days=1))
    assert got["stale"] and "earlier capture" in got["reason"]


def test_no_key_keeps_the_reports_and_says_the_snapshot_is_missing(web, season, no_key):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN)
    assert [r["kind"] for r in cap.rows] == ["article", "file"]
    assert cap.lines_rows is None
    assert cap.lines_why == "not captured: LoopRefused"


def test_no_season_keeps_the_reports_and_names_the_anchor(web, monkeypatch):
    monkeypatch.setattr(cfbd, "_env", lambda: {})
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN)
    assert cap.week is None and "CFB_WEEK_ONE" in (cap.lines_why or "")
    assert len(cap.rows) == 2


def test_skip_lines_spends_nothing(web, season, cfbd_lines):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert cfbd_lines == []


# --- the stamp --------------------------------------------------------------------

def test_a_full_capture_is_recorded_fresh(web, season, cfbd_lines):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    cap = bigten.capture(now=FIRST_RUN)
    got = bigten.record_run(cap, now=FIRST_RUN)
    assert got["shape"] == "summary" and got["name"] == "bigten"
    assert (got["fetched"], got["stale"], got["reason"]) == (True, False, None)
    assert got["deadline"] == {"id": "2026-09-17T0130Z", "name": "evening"}
    assert got["documents"] == {"seen": 2, "new": 2, "reports": 1}
    assert got["lines"]["rows"] == 1 and got["lines"]["why"] is None
    assert got["missed"] == {"count": 0, "deadlines": []}
    assert got["archive_rows"] == 3
    assert json.loads(bigten.STATUS.read_text()) == got


def test_a_missing_snapshot_makes_the_capture_stale_and_says_why(web, season, no_key):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    got = bigten.record_run(bigten.capture(now=FIRST_RUN), now=FIRST_RUN)
    assert got["fetched"] and got["stale"]
    assert "odds snapshot not captured: LoopRefused" in got["reason"]


def test_a_shape_change_is_recorded_as_degraded_not_as_nothing(web, season):
    web[bigten.PAGE] = b"<html>redesigned</html>"
    got = bigten.record_run(bigten.capture(now=FIRST_RUN, skip_lines=True), now=FIRST_RUN)
    assert got["fetched"] and got["stale"]
    assert "PageShapeChanged" in got["reason"] and got["documents"]["seen"] == 1


def test_one_lost_document_is_recorded_as_degraded_with_the_rest_kept(web, season):
    two = ONE_LINK.replace("</p>", '<a href="/api/media/file/gone.pdf">gone</a></p>')
    web[bigten.PAGE] = page_with(two)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    got = bigten.record_run(bigten.capture(now=FIRST_RUN, skip_lines=True), now=FIRST_RUN)
    assert got["fetched"] and got["stale"]
    assert "linked document" in got["reason"] and "OSError" in got["reason"]
    assert got["documents"] == {"seen": 2, "new": 2, "reports": 1}


# --- what the capture will and will not follow (#265, #278) -----------------------------

def test_only_the_conferences_own_documents_are_archived_and_the_rest_are_stamped(
        web, season, tmp_path, capsys):
    """The body is the conference's CMS and the CMS carries what the conference's CMS
    carries: an ad, a partner site, a video. Following every absolute href committed a
    third party's bytes into this repository. An off-host link and a link that is not a
    document are skipped and *named* in the stamp -- a legitimate report behind a new
    host has to be visible rather than silently dropped -- and an oversized one is
    refused mid-stream and recorded as a document this capture failed to keep."""
    body = ONE_LINK.replace("</p>", (
        '<a href="https://ads.example.net/promo.pdf">partner</a>'
        '<a href="/fb/schedule/">the schedule page</a>'
        '<a href="/api/media/file/huge-Week_4.mp4">a video</a>'
        '<a href="/api/media/file/big-Week_4.pdf">too big</a></p>'))
    web[bigten.PAGE] = page_with(body)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    web["https://bigten.org/api/media/file/big-Week_4.pdf"] = bigten.DocumentTooLarge(
        "over the cap")
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert [r["kind"] for r in cap.rows] == ["article", "file"]
    assert cap.rows[1]["url"].endswith("abc-Week_4.pdf")
    assert "ads.example.net" not in web["calls"] and not any(
        "promo.pdf" in u or "schedule" in u or ".mp4" in u for u in web["calls"]), (
        "a skipped link is never fetched, which is the point of an allowlist")
    assert [(k["host"], k["why"].split(":")[0]) for k in cap.skipped] == [
        ("ads.example.net", "off-host"), ("bigten.org", "not a document"),
        ("bigten.org", "not a document"), ("bigten.org", "too large")]
    assert isinstance(cap.error, bigten.DocumentTooLarge), (
        "a report the capture refused to keep is a report not kept")
    got = bigten.record_run(cap, now=FIRST_RUN)
    assert got["stale"] and "DocumentTooLarge" in got["reason"]
    assert got["documents"] == {"seen": 2, "new": 2, "reports": 1}
    assert got["skipped"] == {"count": 4, "by_reason": {"off-host": 1, "not a document": 2,
                                                          "too large": 1},
                              "hosts": ["ads.example.net", "bigten.org"]}
    err = capsys.readouterr().err
    assert "promo.pdf" in err and "off-host" in err
    stored = sorted(str(x.name) for x in (tmp_path / "archive" / "availability").rglob("*.*"))
    assert len(stored) == 2, stored


def test_off_host_and_non_document_skips_alone_do_not_degrade_the_capture(web, season):
    """An ad link beside the report is the CMS's normal state; a run that is stale for
    it is stale forever, which trains the reader to ignore stale."""
    body = ONE_LINK.replace("</p>", '<a href="https://ads.example.net/x">ad</a></p>')
    web[bigten.PAGE] = page_with(body)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    got = bigten.record_run(bigten.capture(now=FIRST_RUN, skip_lines=True), now=FIRST_RUN)
    assert not got["stale"] and got["skipped"]["count"] == 1


def test_a_subdomain_of_the_conference_is_the_conference(web, season):
    """The document CDN, when the conference moves the files onto one, is under its own
    domain; a host that merely ends in the letters is not."""
    assert bigten.admit("https://assets.bigten.org/reports/Week_4.pdf") is None
    assert bigten.admit("https://bigten.org/api/media/file/Week_4.xlsx") is None
    assert bigten.admit("https://bigten.org/api/media/file/Week_4.pdf#page=2") is None
    assert bigten.admit("https://bigten.org/api/media/file/Week_4.pdf?v=3") is None
    assert (bigten.admit("https://notbigten.org/Week_4.pdf") or "").startswith("off-host")
    assert (bigten.admit("https://bigten.org.example.com/Week_4.pdf") or "").startswith(
        "off-host")
    assert (bigten.admit("https://bigten.org/api/media/file/Week_4") or "").startswith(
        "not a document")


def test_the_streamed_transport_refuses_a_document_over_the_cap(monkeypatch):
    """`_get` is the door and the cap is on the door: the bytes past it are never held."""
    class _Resp:
        def __init__(self, chunks): self._chunks = chunks
        def raise_for_status(self): pass
        def iter_content(self, chunk_size): yield from self._chunks
        def __enter__(self): return self
        def __exit__(self, *a): return False
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp([b"x" * 10, b"y" * 10]))
    monkeypatch.delenv(bigten.PYTEST_NODE_ENV, raising=False)
    assert bigten._get("https://bigten.org/f.pdf", cap=25) == b"x" * 10 + b"y" * 10
    assert bigten._get("https://bigten.org/f.pdf", cap=20) == b"x" * 10 + b"y" * 10
    with pytest.raises(bigten.DocumentTooLarge, match="15"):
        bigten._get("https://bigten.org/f.pdf", cap=15)


def test_a_parsed_page_with_no_report_documents_is_stale_inside_the_regime(web, season):
    """#278: the link loop ran zero times, no error was set, and the run was recorded as
    fetched and not stale. Forty tests and none covered it. Inside the reporting regime a
    capture that archives no report is degraded, with the reason."""
    web[bigten.PAGE] = page_with('<div class="payload-richtext"><p>Reports post here.</p></div>')
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    assert cap.fetched and cap.parsed and cap.error is None and cap.reports == 0
    got = bigten.record_run(cap, now=FIRST_RUN)
    assert got["fetched"] and got["stale"]
    assert "no report documents" in got["reason"]
    assert got["documents"] == {"seen": 1, "new": 1, "reports": 0}


def test_the_watchdog_calls_a_fresh_stamp_with_no_reports_empty(season, tmp_path):
    """A stamp written on time that archived nothing is the stall wearing a fresh date."""
    now = GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10)
    p = _stamp(tmp_path, GAMEDAY + timedelta(minutes=95),
               documents={"seen": 1, "new": 1, "reports": 0})
    got = bigten.watch(p, now=now)
    assert got.startswith("inside 2026-10-03T1500Z empty "), got
    kept = _stamp(tmp_path, GAMEDAY + timedelta(minutes=95),
                  documents={"seen": 2, "new": 2, "reports": 1})
    assert bigten.watch(kept, now=now).startswith("inside 2026-10-03T1500Z ok ")
    # A stamp from before `reports` was recorded is read as it always was.
    assert bigten.watch(_stamp(tmp_path, GAMEDAY + timedelta(minutes=95)),
                        now=now).startswith("inside 2026-10-03T1500Z ok ")


def test_the_availability_article_is_chosen_by_title_not_by_dict_order():
    """#278: the record was taken by dict order. A related-content block that lands first
    in `fallback` would have been archived as the reports article, still green."""
    other = {"_content_type_uid": "article", "title": "Week 4 Game Notes",
             "updatedAt": "2026-09-17T00:00:00.000Z", "body": "<p>notes</p>", "id": 1}
    reports = {"_content_type_uid": "article", "title": "2026 FB Availability Reports",
               "updatedAt": "2026-09-17T00:12:44.000Z", "body": ONE_LINK, "id": 60323}
    data = {"props": {"pageProps": {"fallback": {"1": other, "60323": reports}}}}
    page = (b'<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(data).encode() + b"</script></html>")
    assert bigten.parse_article(page).title == "2026 FB Availability Reports"
    data["props"]["pageProps"]["fallback"] = {"1": other}
    page = (b'<html><script id="__NEXT_DATA__" type="application/json">'
            + json.dumps(data).encode() + b"</script></html>")
    with pytest.raises(bigten.PageShapeChanged, match="availability"):
        bigten.parse_article(page)


def test_status_reports_the_last_runs_skipped_links(web, season, clock, capsys, tmp_path):
    clock(FIRST_RUN)
    body = ONE_LINK.replace("</p>", '<a href="https://ads.example.net/x">ad</a></p>')
    web[bigten.PAGE] = page_with(body)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.main(["--capture", "--skip-lines", "--status-path", str(tmp_path / "s.json")])
    assert bigten.main(["--status", "--status-path", str(tmp_path / "s.json")]) == 0
    out = capsys.readouterr().out
    assert "skipped 1 link" in out and "off-host" in out and "ads.example.net" in out
    # `--status` reads the stamp it is pointed at, not the default (review finding).
    assert bigten.main(["--status"]) == 0
    assert "skipped" not in capsys.readouterr().out


def test_an_unreachable_page_is_recorded_as_not_fetched(web, season):
    got = bigten.record_run(bigten.capture(now=FIRST_RUN, skip_lines=True), now=FIRST_RUN)
    assert (got["fetched"], got["stale"]) == (False, True)
    assert "OSError" in got["reason"]


def test_the_stamp_never_carries_a_message_from_the_source(web, season):
    """A failure arrives as the exception and only its type is written; `cfbd.record_run`
    argues why, and the argument holds for a page a third party controls."""
    web[bigten.PAGE] = b"<html>redesigned: SECRET-SHAPED-TEXT</html>"
    cap = bigten.capture(now=FIRST_RUN, skip_lines=True)
    bigten.record_run(cap, now=FIRST_RUN)
    text = bigten.STATUS.read_text()
    assert "SECRET-SHAPED-TEXT" not in text
    assert "no __NEXT_DATA__" not in text


def test_missed_deadlines_are_the_deadlines_with_no_row_behind_them(web, season):
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.capture(now=FIRST_RUN, skip_lines=True)
    # The Thursday and Friday evening deadlines pass with no run; Saturday morning's runs.
    saturday = datetime(2026, 9, 19, 15, 20, tzinfo=UTC)
    cap = bigten.capture(now=saturday, skip_lines=True)
    got = bigten.record_run(cap, now=saturday)
    assert got["missed"] == {"count": 2, "deadlines": ["2026-09-18T0130Z", "2026-09-19T0130Z"]}


def test_missed_deadlines_start_at_the_season_where_that_is_later(monkeypatch):
    """A stale REPORTS_BEGIN next August must not report a summer of missed deadlines."""
    monkeypatch.setattr(cfbd, "_env", lambda: {"CFB_WEEK_ONE": "2027-09-04"})
    empty = pl.DataFrame(schema=bigten.INDEX_SCHEMA)
    now = datetime(2027, 9, 8, 12, tzinfo=UTC)
    got = bigten.missed_deadlines(empty, now=now, opens=bigten.cfbd.week_one_opens()[1])
    assert got == ["2027-09-01T0130Z", "2027-09-02T0130Z", "2027-09-03T0130Z",
                   "2027-09-04T0130Z", "2027-09-04T1500Z", "2027-09-04T2100Z",
                   "2027-09-05T0130Z", "2027-09-08T0130Z"]
    assert all(s >= "2027-08-31" for s in got)


# --- the CLI ----------------------------------------------------------------------

def test_before_the_first_report_the_run_records_that_and_spends_nothing(
        web, season, cfbd_lines, clock, capsys):
    """GUARD nothing-to-capture-is-recorded: the transport is never touched."""
    clock(datetime(2026, 9, 11, 1, 41, tzinfo=UTC))
    code = bigten.main(["--capture"])
    assert code == 1
    got = json.loads(bigten.STATUS.read_text())
    assert got["fetched"] is False and "2026-09-17T00:00" in got["reason"]
    assert web["calls"] == [] and cfbd_lines == []
    assert "Traceback" not in capsys.readouterr().err


def test_after_the_regular_season_the_run_records_that_and_spends_nothing(
        web, season, cfbd_lines, clock):
    clock(datetime(2027, 1, 20, 1, 41, tzinfo=UTC))
    assert bigten.main(["--capture"]) == 1
    got = json.loads(bigten.STATUS.read_text())
    assert got["fetched"] is False and "regular season" in got["reason"]
    assert web["calls"] == [] and cfbd_lines == []


def test_the_cli_captures_and_exits_zero(web, season, cfbd_lines, clock, capsys):
    clock(FIRST_RUN)
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    assert bigten.main(["--capture"]) == 0
    got = json.loads(bigten.STATUS.read_text())
    assert got["stale"] is False and got["archive_rows"] == 3
    assert "captured" in capsys.readouterr().out


def test_an_unreachable_page_is_a_sentence_and_a_non_zero_exit(web, season, clock, capsys):
    clock(FIRST_RUN)
    assert bigten.main(["--capture", "--skip-lines"]) == 1
    out = capsys.readouterr()
    assert "Traceback" not in out.err and "OSError" in out.err


def test_a_refused_index_is_a_sentence_and_a_non_zero_exit(web, season, clock, tmp_path,
                                                            capsys):
    clock(FIRST_RUN)
    (tmp_path / "archive").mkdir()
    bigten.INDEX.write_text("this is not an index")
    assert bigten.main(["--capture", "--skip-lines"]) == 1
    out = capsys.readouterr()
    assert "Traceback" not in out.err and "JSONDecodeError" in out.err
    assert json.loads(bigten.STATUS.read_text())["fetched"] is False


def test_status_reports_the_archive_and_the_missed_deadlines(web, season, clock, capsys):
    clock(FIRST_RUN)
    web[bigten.PAGE] = page_with(ONE_LINK)
    web["https://bigten.org/api/media/file/abc-Week_4.pdf"] = PDF
    bigten.main(["--capture", "--skip-lines"])
    clock(datetime(2026, 9, 19, 15, 20, tzinfo=UTC))
    assert bigten.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "2 rows over 1 captured deadlines" in out and "missed deadlines" in out
    assert "2026-09-18T0130Z" in out


# --- no test reaches the conference -------------------------------------------------

def test_the_transport_refuses_a_live_call_from_the_default_suite():
    """GUARD no-live-call-from-the-suite. `PYTEST_CURRENT_TEST` is set by pytest itself
    while this runs, so nothing is patched: this is the real door being tried."""
    with pytest.raises(bigten.LiveCallRefused, match=r"tests/unit/test_fetch_bigten\.py"):
        bigten._get(bigten.PAGE)


def test_the_network_has_exactly_one_door_in_this_module():
    import inspect
    src = inspect.getsource(bigten)
    users = [ln.strip() for ln in src.splitlines()
             if "requests" in ln and not ln.strip().startswith("#")]
    assert users == ["import requests",
                     'with requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT},',
                     ], (
        f"something other than `_get` reaches the network now: {users}")


# --- the watchdog's reading of the stamp (#240) ------------------------------------------
#
# `watchdog.yml` measures the committed `site/data/bigten.json` against the deadline the
# capture was scheduled for, inside a window that opens `WATCH_DELAY` after each deadline --
# the allowance for GitHub delivering a scheduled start late -- and closes when the next
# deadline fires. Outside every window the check says so rather than measuring.

# Saturday 3 October 2026. The gameday-early deadline is 15:00 UTC, the gameday-late 21:00.
GAMEDAY = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)


def _stamp(tmp_path, generated_at: datetime | str | None, **extra):
    p = tmp_path / "site" / "bigten.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    body: dict = {"fetched": True, "missed": {"count": 0, "deadlines": []}, **extra}
    if generated_at is not None:
        body["generated_at"] = (generated_at.isoformat() if isinstance(generated_at, datetime)
                                else generated_at)
    p.write_text(json.dumps(body))
    return p


def test_a_stamp_written_since_the_deadline_is_healthy_inside_its_window(season, tmp_path):
    now = GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10)
    p = _stamp(tmp_path, GAMEDAY + timedelta(minutes=95))     # the capture, delivered late
    got = bigten.watch(p, now=now)
    assert got.startswith("inside 2026-10-03T1500Z ok "), got
    assert got.split()[3].isdigit()
    # A stamp with no offset on its clock is read as UTC, which is the only clock any
    # envelope here is written on, rather than refused as unreadable.
    naive = _stamp(tmp_path, "2026-10-03T16:35:00")
    assert bigten.watch(naive, now=now) == got


def test_a_stamp_older_than_the_deadline_is_stale_once_the_delay_has_passed(season, tmp_path):
    """The stall this exists to file: the deadline fired, the delay a late start is allowed
    has passed, and nothing has written the stamp since the deadline."""
    now = GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10)
    p = _stamp(tmp_path, GAMEDAY - timedelta(hours=13))       # last night's capture
    got = bigten.watch(p, now=now)
    assert got.startswith("inside 2026-10-03T1500Z stale "), got
    behind = int(got.split()[3])
    assert behind == int((now - (GAMEDAY - timedelta(hours=13))).total_seconds())


def test_inside_the_delay_after_a_deadline_the_check_is_outside_a_window(season, tmp_path):
    """A capture scheduled at the deadline may still be queued -- `live.yml` records starts
    delivered 97-126 minutes late -- so a stale stamp here is not evidence of a stall, and
    the check reports that it is outside a window rather than measuring."""
    p = _stamp(tmp_path, GAMEDAY - timedelta(hours=13))
    for minutes in (0, 1, 60, int(bigten.WATCH_DELAY.total_seconds() // 60) - 1):
        got = bigten.watch(p, now=GAMEDAY + timedelta(minutes=minutes))
        assert got.startswith("outside 2026-10-03T1500Z "), (minutes, got)
        until = int(got.split()[2])
        assert until == int(bigten.WATCH_DELAY.total_seconds()) - minutes * 60
    # And the moment the delay has passed, it measures.
    assert bigten.watch(p, now=GAMEDAY + bigten.WATCH_DELAY).startswith("inside ")


def test_the_window_closes_when_the_next_deadline_fires(season, tmp_path):
    """Saturday 21:00 is the next deadline after 15:00: at 21:00 the check is inside the
    delay for that one, whatever the 15:00 capture did."""
    p = _stamp(tmp_path, GAMEDAY + timedelta(minutes=95))
    late = GAMEDAY + timedelta(hours=6)
    assert bigten.watch(p, now=late - timedelta(minutes=1)).startswith(
        "inside 2026-10-03T1500Z ok ")
    assert bigten.watch(p, now=late).startswith("outside 2026-10-03T2100Z ")


def test_before_the_first_report_there_is_no_window_to_measure_in(season, tmp_path):
    p = _stamp(tmp_path, bigten.REPORTS_BEGIN - timedelta(days=30))
    got = bigten.watch(p, now=bigten.REPORTS_BEGIN - timedelta(hours=1))
    assert got.startswith("preseason "), got
    assert bigten.REPORTS_BEGIN.isoformat() in got
    # `REPORTS_BEGIN` is midnight and the first deadline fires at 01:30, so for ninety
    # minutes the season has begun and the most recent deadline is still last week's.
    got = bigten.watch(p, now=bigten.REPORTS_BEGIN + timedelta(minutes=30))
    assert got.startswith("preseason ") and "no deadline has fired" in got, got
    got = bigten.watch(p, now=bigten.REPORTS_BEGIN + timedelta(hours=1, minutes=30))
    assert got.startswith("outside 2026-09-17T0130Z "), got


def test_with_no_season_configured_nothing_captures_so_nothing_is_measured(monkeypatch,
                                                                            tmp_path):
    monkeypatch.setattr(cfbd, "_env", lambda: {})
    p = _stamp(tmp_path, GAMEDAY - timedelta(hours=13))
    got = bigten.watch(p, now=GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10))
    assert got.startswith("preseason "), got
    assert "CFB_WEEK_ONE" in got


def test_an_absent_or_unreadable_stamp_is_named_and_not_reported_as_an_age(season, tmp_path):
    """Three failures with three fixes, apart -- `heartbeat.sh`'s rule, and its own history
    is a sentinel that turned an absent field into 56.7 years of staleness."""
    now = GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10)
    assert bigten.watch(tmp_path / "nowhere.json", now=now) == "unreachable"
    assert bigten.watch(_stamp(tmp_path, None), now=now) == "unreadable"
    assert bigten.watch(_stamp(tmp_path, "last tuesday"), now=now) == "unreadable"
    (tmp_path / "site" / "bigten.json").write_text("not json {")
    assert bigten.watch(tmp_path / "site" / "bigten.json", now=now) == "unreadable"


def test_the_cli_answers_watch_with_the_one_line_the_workflow_reads(season, tmp_path, capsys,
                                                                    clock):
    clock(GAMEDAY + bigten.WATCH_DELAY + timedelta(minutes=10))
    p = _stamp(tmp_path, GAMEDAY + timedelta(minutes=95))
    assert bigten.main(["--watch", "--status-path", str(p)]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("inside 2026-10-03T1500Z ok "), out
    assert bigten.main(["--watch", "--status-path", str(tmp_path / "missing.json")]) == 0
    assert capsys.readouterr().out.strip() == "unreachable"
