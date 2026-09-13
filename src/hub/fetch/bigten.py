"""The Big Ten availability archive, captured on the conference's own deadlines (#215).

**What this is.** From 19 September 2026 every Big Ten conference game gets four availability
reports: three days, two days and one day before kickoff, each by 8pm ET, and a gameday
report two hours before kickoff. No archive of them exists anywhere, no model is trained on
one, and a report the conference has replaced cannot be fetched afterwards. So this module
is an archive first and a fetcher second: it keeps the bytes the conference served, says
which deadline it kept them for, and pairs each capture with an odds snapshot taken in the
same run, so that the pairing is a fact about the run rather than something reconstructed
later from two clocks.

**Where the reports come from, and what is metered.** `bigten.org/fb/availability-reports/`
is a Next.js page whose content is a CMS article record in its `__NEXT_DATA__` blob: a title,
an `updatedAt`, and a body of links to documents under `/api/media/file/`. In 2024 that was
one PDF per week; the 2026 page existed and held nothing on the day this was written, so
what the four-a-week regime does to that shape is unknown until it starts. That is the
reason for the shape of the capture below -- every linked document the conference serves
from its own domain is archived by content hash (a link to anyone else's host, to a page
rather than a document, or past a size cap is skipped and named in the stamp, #265), and the
article record itself is archived too, so a report published as a table in the body rather
than as a linked file is still kept. Inside the reporting regime a capture that parses the
page and archives no report is recorded degraded rather than green (#278). None of that
costs quota: the conference's page is public and unmetered.

The odds snapshot is the metered half. CFBD has no availability endpoint (checked against
its OpenAPI spec on 2026-09-11, not from memory), and `/lines?year&week` is the one bulk
call that prices every Big Ten game at once -- **one call per deadline**, and never one per
team; `hub.fetch.cfbd` is where that is impossible rather than merely discouraged. The
per-week cost is stated in `docs/cfbd-quota.md`, which the disposition requires before the
first live run.

**A capture is *for* a deadline, and a deadline is a scheduled instant.** The conference's
deadlines are 8pm ET and two hours before kickoff; this module's `DEADLINES` are the fixed
UTC instants after them that a capture is attributed to. The workflow's crons fire on those;
GitHub delivers a scheduled run anywhere up to two hours late; and a capture that names the
wall-clock it happened at cannot be compared to the one before it.
So every capture is attributed to the most recent deadline at or before the moment it ran, and
the set of deadlines between the regime's first day and now, minus those the index holds a row
for, is the list of deadlines that were **missed** -- recorded in the stamp rather than left as a
gap someone has to notice.

**Degradation, which is the whole design.** Every failure leaves a record and never a
traceback: the page unreachable, the page's shape changed (the raw page is archived so the
bytes are not lost while the parser is fixed), the odds call refused or failed (the reports
are still archived and the stamp says the pairing is missing), no key, no anchor. The stamp
at `site/data/bigten.json` carries `generated_at` like every other envelope, so the
watchdog can read a silent stall as a stall.

    uv run python -m hub.fetch.bigten --capture
    uv run python -m hub.fetch.bigten --capture --skip-lines      # archive only, spend nothing
    uv run python -m hub.fetch.bigten --status
    uv run python -m hub.fetch.bigten --watch                     # the watchdog's one line

**And the watchdog reads it here** (#240). `watchdog.yml` is what files an incident when the
capture silently stops -- the cron disabled, the page's shape changed at every deadline, the
job failing before it commits -- and its window logic was the live overlay's, which has no
notion of a deadline. `watch` is the Big Ten half: the stamp measured against the deadline
it should have been written for, inside a window that opens `WATCH_DELAY` after each
deadline and closes when the next one fires. Outside every window it says so rather than
measuring, which is the rule `.github/scripts/window_state.sh` learned from issue #193.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import polars as pl

from hub import atomic, jsonio
from hub.config import SEASON_AHEAD
from hub.contracts import BIGTEN_CAPTURES, CFBD_LINES
from hub.fetch import cfbd
from hub.paths import ROOT, SITE

ORIGIN = "https://bigten.org"
PAGE = f"{ORIGIN}/fb/availability-reports/"

# Committed, deliberately, and the one archive in this repo that is. `data/raw/` is
# gitignored because CFBD and nflverse payloads cannot be redistributed; the conference's
# availability reports are published for exactly that purpose, and a runner that fetched them
# into a gitignored tree would lose them when the job ended -- which is the whole failure
# this ticket exists to prevent.
ARCHIVE = ROOT / "archive" / "bigten"
INDEX = ARCHIVE / "availability" / "captures.json"

# The odds snapshots are CFBD payloads and stay under the gitignored tree, per
# `docs/cfbd-quota.md`. The workflow encrypts and commits them under `archive/bigten/lines/`
# when it has a key to do so with; locally they are simply here.
LINES = ROOT / "data" / "raw" / "bigten" / "lines"

STATUS = SITE / "bigten.json"

# The first deadline, as an instant: the regime's first games are Saturday 19 September 2026
# and the three-day report for them is due 8pm ET on Wednesday the 16th, which is 00:00 UTC
# on the 17th. An instant rather than a date because every deadline is one, and a date would
# have to say which clock it was on. A value in the past suppresses nothing, so a stale one
# next August costs a few empty captures of a page before the season and never silence
# inside one. The *end* is not a second constant: captures stop when
# `cfbd.configured_week` says the regular season is over, which is the same anchor the
# college week is counted from and is bumped once a year for that reason.
REPORTS_BEGIN = datetime(2026, 9, 17, 0, 0, tzinfo=UTC)

# The pytest node running right now, or nothing outside a test. Same guard as
# `hub.fetch.cfbd._http_get`, for the same reason: the transport refuses under the default
# suite so that the test nobody remembered to patch cannot reach the network.
PYTEST_NODE_ENV = "PYTEST_CURRENT_TEST"
LIVE_TEST_SUITE = "tests/golden/"

USER_AGENT = "football-hub/0.1 (+https://github.com/jacksonmlukas/football-hub)"

# What the capture will follow out of the article's body (#265). The body is the
# conference's CMS, and a CMS carries whatever it carries -- an ad, a partner site, a video --
# and every one of those was an absolute href this loop fetched to any origin, with no size
# cap, and committed into the tracked archive. Three rules, each admitting everything the
# regime is expected to publish: the host is the conference's own domain or a subdomain of it
# (the document CDN, if the files move onto one, is under it); the link is a document by
# extension; and the bytes stop at a cap that clears any report by two orders of magnitude.
# A link the rules refuse is recorded in the stamp with the reason -- a legitimate report
# behind a host these rules do not know has to be visible rather than silently dropped.
ALLOWED_HOST = "bigten.org"
DOCUMENT_EXTENSIONS = frozenset({"pdf", "xlsx", "xls", "csv", "docx", "doc", "txt"})
DOCUMENT_CAP = 16 * 1024 * 1024

# The words the availability-reports article's title carries. The page's `fallback` map
# holds every CMS record the page was rendered from, and taking the first `article` by dict
# order archived whichever related-content block landed first (#278).
AVAILABILITY_TITLE = re.compile(r"availability\s+reports?", re.I)

# How many missed deadlines the stamp lists in full. The count is always exact; the list is
# capped so the stamp stays a stamp.
MISSED_LISTED = 20


@dataclass(frozen=True)
class Deadline:
    """One scheduled capture: a name, and when it fires each week, in UTC.

    `weekday` is cron's numbering -- Sunday 0 -- because the workflow's crons are what this
    is compared against, and the comparison should not have to translate.
    """
    name: str
    weekday: int
    hour: int
    minute: int

    def cron(self) -> str:
        return f"{self.minute} {self.hour} * * {self.weekday}"


# The schedule, and the argument for each line. `tests/contracts/test_bigten_schedule.py`
# holds the workflow's crons to exactly these.
#
# *Evening, 01:30 UTC Wednesday through Sunday.* The midweek reports are due 8pm ET, three,
# two and one day before kickoff. For a Saturday game that is Wednesday, Thursday and Friday
# evening; for a Friday game, Tuesday through Thursday. 01:30 UTC is 9:30pm EDT and 8:30pm
# EST -- after the deadline in both daylight-saving states, so one cron serves the whole
# season without the two-state widening `live.yml` has to do. Sunday's 01:30 UTC is Saturday
# night ET and catches the gameday reports for the late kickoffs, and Friday night's
# gameday report for a Friday game.
#
# *Gameday, 15:00 and 21:00 UTC Saturday.* Gameday reports are due two hours before kickoff:
# 10am ET for a noon game, 1:30pm for 3:30, 5-6pm for the evening windows. 15:00 UTC (11am
# EDT) is after the noon-kickoff reports; 21:00 UTC (5pm EDT) is after the afternoon ones;
# the Sunday-01:30 evening deadline above takes the rest. Three gameday captures rather than one
# per kickoff, because every capture is one CFBD call and the kickoff times are not known
# to this module without another one.
DEADLINES: tuple[Deadline, ...] = (
    Deadline("evening", 3, 1, 30),
    Deadline("evening", 4, 1, 30),
    Deadline("evening", 5, 1, 30),
    Deadline("evening", 6, 1, 30),
    Deadline("evening", 0, 1, 30),
    Deadline("gameday-early", 6, 15, 0),
    Deadline("gameday-late", 6, 21, 0),
)

# The index's declared column order, so a frame built from a fresh file and a frame built
# from an old one have the same dtypes whatever the rows happen to hold. An all-null column
# infers as `Null`, which the contract rightly refuses.
INDEX_SCHEMA: dict[str, type[pl.DataType]] = {
    "deadline": pl.Utf8, "deadline_name": pl.Utf8, "captured_at": pl.Utf8, "season": pl.Int64,
    "kind": pl.Utf8, "url": pl.Utf8, "label": pl.Utf8, "report_updated_at": pl.Utf8,
    "sha256": pl.Utf8, "bytes": pl.Int64, "path": pl.Utf8, "new_content": pl.Boolean,
}


class LiveCallRefused(Exception):
    """A test reached the transport. Refused before anything left the process."""


class PageShapeChanged(Exception):
    """The page no longer carries the CMS record this module reads. The raw page is kept."""


def _now() -> datetime:
    """One clock, so a test can set it. Everything here that asks the time asks this."""
    return datetime.now(UTC)


class DocumentTooLarge(Exception):
    """A linked document ran past `DOCUMENT_CAP` mid-stream; the bytes were not kept."""


def _get(url: str, cap: int | None = None) -> bytes:
    """The one door to the network in this module.

    `cap` is enforced on the stream: the response is read in chunks and abandoned the
    moment it would pass the cap, so the bytes past it are never appended, let alone
    written.
    """
    # GUARD no-live-call-from-the-suite [unit/test_fetch_bigten.py]: a test cannot reach
    # the conference
    node = os.environ.get(PYTEST_NODE_ENV, "")
    if node and not node.startswith(LIVE_TEST_SUITE):
        raise LiveCallRefused(
            f"{node.split(' ')[0]} would fetch {url}. No test reaches the network: patch "
            f"`_get`, the way tests/unit/test_fetch_bigten.py does.")
    # /GUARD
    import requests
    with requests.get(url, timeout=30, headers={"User-Agent": USER_AGENT},
                      stream=True) as r:
        r.raise_for_status()
        held = bytearray()
        for chunk in r.iter_content(chunk_size=1 << 16):
            # Checked before the chunk is appended, so nothing past the cap is ever held
            # here. One chunk can still be larger than asked for -- `requests` decodes a
            # gzipped body as it streams -- and that chunk is the residual this cannot see.
            if cap is not None and len(held) + len(chunk) > cap:
                raise DocumentTooLarge(f"{url} ran past {cap} bytes; not kept")
            held.extend(chunk)
    return bytes(held)


def admit(url: str) -> str | None:
    """Why the capture will not follow this link, or None if it will (#265).

    The reason starts with one of three words, which is what the stamp counts by:
    `off-host: <host>`, `not a document: <ext>`. `too large` is the third and is decided by
    the stream rather than here.
    """
    host = urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    if host != ALLOWED_HOST and not host.endswith("." + ALLOWED_HOST):
        return f"off-host: {host}"
    ext = _ext(url, "")
    if ext not in DOCUMENT_EXTENSIONS:
        return f"not a document: {ext or 'no extension'}"
    return None


# --- deadlines ----------------------------------------------------------------------

def _cron_weekday(day: date) -> int:
    """Python's Monday-0 weekday as cron's Sunday-0."""
    return (day.weekday() + 1) % 7


def deadline_at(now: datetime) -> tuple[Deadline, datetime]:
    """The most recent deadline at or before `now`, and the instant it fired.

    A run delivered two hours late belongs to the deadline it was scheduled after, not to
    the wall-clock it happened at. Walked back day by day rather than computed, because a
    week has seven of these and the arithmetic for "the previous Saturday 21:00" is where
    an off-by-one hides.
    """
    for back in range(8):
        day = (now - timedelta(days=back)).date()
        candidates = [
            (s, datetime(day.year, day.month, day.day, s.hour, s.minute, tzinfo=UTC))
            for s in DEADLINES if s.weekday == _cron_weekday(day)
        ]
        fired = [(s, at) for s, at in candidates if at <= now]
        if fired:
            return max(fired, key=lambda p: p[1])
    raise AssertionError("DEADLINES covers a week; something fired in the last seven days")


def deadline_id(at: datetime) -> str:
    return at.strftime("%Y-%m-%dT%H%MZ")


def expected_deadlines(since: datetime, until: datetime) -> list[tuple[Deadline, datetime]]:
    """Every deadline that should have been captured between `since` and `until`."""
    out: list[tuple[Deadline, datetime]] = []
    day = since.date()
    while day <= until.date():
        wd = _cron_weekday(day)
        for s in DEADLINES:
            if s.weekday != wd:
                continue
            at = datetime(day.year, day.month, day.day, s.hour, s.minute, tzinfo=UTC)
            if since <= at <= until:
                out.append((s, at))
        day += timedelta(days=1)
    return sorted(out, key=lambda p: p[1])


# --- the page -------------------------------------------------------------------

_NEXT_DATA = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)
_HREF = re.compile(r'<a\b[^>]*\bhref="([^"]+)"[^>]*>(.*?)</a>', re.S | re.I)
_TAGS = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Article:
    """The CMS record behind the page: what the conference last said, and when."""
    title: str
    updated_at: str
    body: str
    record: dict[str, Any]

    def links(self) -> list[tuple[str, str]]:
        """(absolute url, label) for every anchor in the body, in page order."""
        out = []
        for href, inner in _HREF.findall(self.body):
            label = html.unescape(_TAGS.sub(" ", inner))
            label = " ".join(label.split())
            out.append((urljoin(ORIGIN, html.unescape(href)), label))
        return out


def parse_article(page: bytes) -> Article:
    """The article record out of the page, or `PageShapeChanged` naming what was missing.

    Deliberately narrow. The page is 1.6MB of navigation around a few hundred bytes of
    content, and the content is a single CMS record with a stable shape -- `title`,
    `updatedAt`, `body` -- that also carries the conference's own timestamp, which an HTML
    scrape would not. A rename of any of those raises, and the caller archives the raw page
    so the bytes survive the parser being wrong.
    """
    m = _NEXT_DATA.search(page.decode("utf-8", errors="replace"))
    if not m:
        raise PageShapeChanged("no __NEXT_DATA__ script on the page")
    try:
        data = json.loads(m.group(1))
    except ValueError as e:
        raise PageShapeChanged("__NEXT_DATA__ is not JSON") from e
    props = data.get("props", {}).get("pageProps", {})
    fallback = props.get("fallback")
    if not isinstance(fallback, dict):
        raise PageShapeChanged("pageProps.fallback is not a record map")
    articles = [v for v in fallback.values()
                if isinstance(v, dict) and v.get("_content_type_uid") == "article"]
    if not articles:
        raise PageShapeChanged("no article record in pageProps.fallback")
    # By title, not by dict order: a related-content block that lands first in the map is
    # an article record too, and archiving it as the reports article would be green (#278).
    named = [v for v in articles
             if isinstance(v.get("title"), str) and AVAILABILITY_TITLE.search(v["title"])]
    if not named:
        raise PageShapeChanged(
            f"none of the {len(articles)} article records is the availability-reports "
            f"article by title")
    rec = named[0]
    missing = [k for k in ("title", "updatedAt", "body") if not isinstance(rec.get(k), str)]
    if missing:
        raise PageShapeChanged(f"article record lacks {missing}")
    return Article(title=rec["title"], updated_at=rec["updatedAt"], body=rec["body"],
                   record=rec)


# --- the archive ----------------------------------------------------------------

def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _ext(url: str, default: str) -> str:
    tail = url.rsplit("/", 1)[-1].split("?", 1)[0].split("#", 1)[0]
    return tail.rsplit(".", 1)[-1].lower() if "." in tail else default


def read_index(path: Path | None = None) -> pl.DataFrame:
    """The capture index as a frame, checked. Absent, empty; corrupt, a refusal."""
    p = Path(path or INDEX)
    rows: list[dict[str, Any]] = []
    if p.exists():
        rows = json.loads(p.read_text())
    df = pl.DataFrame(rows, schema=INDEX_SCHEMA)
    # GUARD index-is-checked [unit/test_fetch_bigten.py]: the index is validated on every
    # read and every write, with the row floor relaxed only for an archive that is empty
    contract = replace(BIGTEN_CAPTURES, min_rows=0) if not df.height else BIGTEN_CAPTURES
    return contract.validate(df)
    # /GUARD


def _write_index(df: pl.DataFrame, path: Path) -> None:
    df = BIGTEN_CAPTURES.validate(df)
    atomic.write_text(path, jsonio.dumps(df.to_dicts(), indent=1) + "\n")


@dataclass
class Capture:
    """What one run kept, and what it could not."""
    deadline: Deadline
    at: datetime
    season: int
    week: int | None
    rows: list[dict[str, Any]]
    fetched: bool = False
    parsed: bool = False
    error: BaseException | None = None
    lines_rows: int | None = None
    lines_captured_at: str | None = None
    lines_why: str | None = None
    # Links the rules refused, as {url, host, why}; the stamp carries the counts and hosts.
    skipped: list[dict[str, str]] = field(default_factory=list)

    @property
    def documents(self) -> int:
        return sum(1 for r in self.rows if r["kind"] != "lines")

    @property
    def reports(self) -> int:
        """Linked documents kept -- the reports themselves, not the article or the page."""
        return sum(1 for r in self.rows if r["kind"] == "file")

    @property
    def new_documents(self) -> int:
        return sum(1 for r in self.rows if r["kind"] != "lines" and r["new_content"])


def _keep(cap: Capture, *, kind: str, url: str, label: str | None,
          updated_at: str | None, data: bytes, ext: str, archive: Path,
          seen: set[str], captured_at: str) -> None:
    """Archive one document by content hash and index it, new or not."""
    digest = _sha(data)
    rel = Path("availability") / str(cap.season) / kind / f"{digest[:16]}.{ext}"
    target = archive / rel
    new = digest not in seen
    if new:
        atomic.write_bytes(target, data)
        seen.add(digest)
    cap.rows.append({
        "deadline": deadline_id(cap.at), "deadline_name": cap.deadline.name,
        "captured_at": captured_at,
        "season": cap.season, "kind": kind, "url": url, "label": label,
        "report_updated_at": updated_at, "sha256": digest, "bytes": len(data),
        "path": rel.as_posix(), "new_content": new,
    })


def capture(*, now: datetime | None = None, season: int = SEASON_AHEAD,
            archive: Path | None = None, index: Path | None = None,
            lines_dir: Path | None = None, skip_lines: bool = False,
            quota_path: Path | None = None) -> Capture:
    """One capture: the page, every document it links that `admit` allows, and the odds
    snapshot beside them.

    Nothing here raises for a failed fetch. The page unreachable is a `Capture` with
    `fetched=False` and the error; the page's shape changed is one with the raw page archived
    and `parsed=False`; the odds call refused or failed leaves `lines_why` and the report rows
    intact. `record_run` turns each into the stamp, and `main` into an exit code.
    """
    at_now = now or _now()
    deadline, at = deadline_at(at_now)
    archive = Path(archive or ARCHIVE)
    index = Path(index or INDEX)
    held = read_index(index)
    seen = set(held["sha256"].to_list()) if held.height else set()
    # The deadline's week, not the wall-clock's: a run delivered across the Tuesday boundary
    # is attributed to the deadline before it, and its snapshot has to be priced and filed
    # under that deadline's week or the file names one week and prices another.
    week = cfbd.configured_week(at).week
    cap = Capture(deadline=deadline, at=at, season=season, week=week, rows=[])
    stamp = jsonio.stamp()

    try:
        page = _get(PAGE)
    except Exception as e:
        cap.error = e
        return cap
    cap.fetched = True

    article: Article | None
    try:
        article = parse_article(page)
    except PageShapeChanged as e:
        article, cap.error = None, e
    # GUARD shape-change-keeps-the-bytes [unit/test_fetch_bigten.py]: a page the parser
    # cannot read is archived whole rather than dropped
    if article is None:
        _keep(cap, kind="page", url=PAGE, label=None, updated_at=None, data=page,
              ext="html", archive=archive, seen=seen, captured_at=stamp)
    # /GUARD
    if article is not None:
        cap.parsed = True
        record = jsonio.dumps(article.record, indent=1).encode()
        _keep(cap, kind="article", url=PAGE, label=article.title,
              updated_at=article.updated_at, data=record, ext="json",
              archive=archive, seen=seen, captured_at=stamp)
        for url, label in article.links():
            if (why := admit(url)) is not None:
                cap.skipped.append({"url": url, "host": urlparse(url).netloc, "why": why})
                print(f"  bigten: {url} was not followed: {why}", file=sys.stderr)
                continue
            try:
                data = _get(url, cap=DOCUMENT_CAP)
            except DocumentTooLarge as e:
                # A document on the conference's host that the cap refused is a report this
                # capture did not keep: recorded as skipped *and* as the run's error.
                cap.skipped.append({"url": url, "host": urlparse(url).netloc,
                                    "why": f"too large: over {DOCUMENT_CAP} bytes"})
                cap.error = cap.error or e
                print(f"  bigten: {url} was not kept: {e}", file=sys.stderr)
                continue
            except Exception as e:
                # One document unreachable is not the capture failing: the rest are kept
                # and the stamp names the kind of failure.
                cap.error = cap.error or e
                print(f"  bigten: {url} was not fetched: {type(e).__name__}",
                      file=sys.stderr)
                continue
            _keep(cap, kind="file", url=url, label=label or None,
                  updated_at=article.updated_at, data=data, ext=_ext(url, "bin"),
                  archive=archive, seen=seen, captured_at=stamp)

    if skip_lines:
        cap.lines_why = "skipped: --skip-lines"
    elif week is None:
        cap.lines_why = "no college week to price: " + cfbd.configured_week(at_now).why
    else:
        _snapshot_lines(cap, lines_dir=Path(lines_dir or LINES), quota_path=quota_path,
                        seen=seen, captured_at=stamp)

    if cap.rows:
        merged = pl.concat([held, pl.DataFrame(cap.rows, schema=INDEX_SCHEMA)])
        _write_index(merged, index)
    return cap


def _snapshot_lines(cap: Capture, *, lines_dir: Path, quota_path: Path | None,
                    seen: set[str], captured_at: str) -> None:
    """The odds snapshot for this capture's week: one `/lines` call, kept beside the reports.

    `refresh=True`, because the whole point is the price *at this deadline* and a cached
    week is the price at some earlier one. `bulk` refuses to spend where it cannot -- the
    run ceiling, the month, no key -- and serves the cache when it holds one. Whether that
    happened is read off the call counter, before and after: a counter that did not move is
    a week that was served rather than fetched. Not off a clock -- the cache stamp is to
    the second and the run's "now" is whatever the caller said it was -- and the counter is
    the one thing that moves on every request, answered or not.
    """
    assert cap.week is not None
    before = cfbd.quota_used(quota_path)
    try:
        df = cfbd.bulk("lines", cap.season, cap.week, quota_path=quota_path, refresh=True)
        if not df.height:
            df = replace(CFBD_LINES, min_rows=0).validate(df) if df.width else df
        else:
            df = CFBD_LINES.validate(df)
    except Exception as e:
        cap.lines_why = f"not captured: {type(e).__name__}"
        return
    when = cfbd.captured_at("lines", cap.season, cap.week)
    cap.lines_rows = df.height
    cap.lines_captured_at = when.isoformat() if when else None
    # GUARD a-served-week-is-not-a-snapshot [unit/test_fetch_bigten.py]: a refused refresh
    # that served the cached week is recorded as an earlier price, never as this deadline's
    if cfbd.quota_used(quota_path) == before:
        cap.lines_why = ("served from an earlier capture: the call was refused and the "
                         "cached week was served instead")
    # /GUARD
    target = lines_dir / str(cap.season) / f"w{cap.week:02d}" / f"{deadline_id(cap.at)}.parquet"
    atomic.write_parquet(df, target)
    data = target.read_bytes()
    digest = _sha(data)
    cap.rows.append({
        "deadline": deadline_id(cap.at), "deadline_name": cap.deadline.name,
        "captured_at": captured_at,
        "season": cap.season, "kind": "lines",
        "url": f"cfbd:/lines?year={cap.season}&week={cap.week}", "label": None,
        "report_updated_at": None, "sha256": digest, "bytes": len(data),
        "path": target.relative_to(lines_dir).as_posix(), "new_content": digest not in seen,
    })


# --- the stamp ------------------------------------------------------------------

def missed_deadlines(index: pl.DataFrame, *, now: datetime, since: datetime = REPORTS_BEGIN,
                 opens: date | None = None) -> list[str]:
    """Every deadline between the regime's first day and now with no capture behind it.

    Bounded below by the season's own opening where that is later than `since`, so a stale
    `REPORTS_BEGIN` next August does not report a summer of missed deadlines for a season
    that had not started. Today's deadline counts as missed only once its capture has had a
    chance to run: a run that *is* the capture for the current deadline writes its row before
    this is asked.
    """
    start = since
    if opens is not None:
        start = max(since, datetime(opens.year, opens.month, opens.day, tzinfo=UTC))
    have = set(index["deadline"].to_list()) if index.height else set()
    return [deadline_id(at) for _, at in expected_deadlines(start, now)
            if deadline_id(at) not in have]


def _skipped_summary(skipped: list[dict[str, str]]) -> dict[str, Any]:
    """Counts by reason and the hosts involved -- never the URLs, for the reason the stamp
    records no payload text: a link is a string a third party wrote."""
    by_reason: dict[str, int] = {}
    for k in skipped:
        word = k["why"].split(":", 1)[0]
        by_reason[word] = by_reason.get(word, 0) + 1
    return {"count": len(skipped), "by_reason": by_reason,
            "hosts": sorted({k["host"] for k in skipped})}


def record_run(cap: Capture | None, *, why: str | None = None, season: int = SEASON_AHEAD,
               index: Path | None = None, path: Path | None = None,
               quota_path: Path | None = None, now: datetime | None = None) -> dict[str, Any]:
    """What this run did, in the three states `hub.fetch.cfbd.record_run` writes.

    * **captured** -- `fetched: true`, `stale: false`. The page was read and parsed and the
      odds snapshot is the one this deadline asked for (or was deliberately skipped).
    * **captured, degraded** -- `fetched: true`, `stale: true`, with the reason: the page's
      shape changed and only the raw bytes were kept, a linked document could not be
      fetched or was over the cap, the page parsed but linked no report inside the regime
      (#278), or the odds snapshot is an earlier one or missing. Links the rules declined
      (#265) are counted under `skipped` and do not on their own degrade the run: an ad
      beside the report is the CMS's normal state.
    * **not captured** -- `fetched: false`, `stale: true`, and the sentence from whatever
      declined: before the first report, no season, the page unreachable.

    **Counts, hashes and deadline ids, never payload text.** The one exception is the
    conference's own `updatedAt`, which is a timestamp and the fact this archive is built
    around. A failure arrives here as the exception and only its type is recorded, for the
    reason `cfbd.record_run` gives: a message is an open channel from a third party into a
    committed file.
    """
    at_now = now or _now()
    _first, opens, _why = cfbd.week_one_opens()
    # The index refusing is one of the things this may be recording, so it is read for the
    # count and the missed list and read as empty when it cannot be.
    try:
        held = read_index(index)
    except Exception:
        held = pl.DataFrame(schema=INDEX_SCHEMA)
    missed = missed_deadlines(held, now=at_now, opens=opens)
    if cap is None:
        fetched, stale, reason = False, True, (why or "nothing was captured")
        deadline: dict[str, Any] = {"id": None, "name": None}
        week: int | None = None
        documents = {"seen": 0, "new": 0, "reports": 0}
        skipped = _skipped_summary([])
        lines: dict[str, Any] = {"rows": None, "captured_at": None, "why": reason}
    else:
        deadline = {"id": deadline_id(cap.at), "name": cap.deadline.name}
        week = cap.week
        documents = {"seen": cap.documents, "new": cap.new_documents, "reports": cap.reports}
        skipped = _skipped_summary(cap.skipped)
        lines = {"rows": cap.lines_rows, "captured_at": cap.lines_captured_at,
                 "why": cap.lines_why}
        kind = type(cap.error).__name__ if cap.error is not None else None
        if not cap.fetched:
            fetched, stale = False, True
            reason = (f"nothing was captured: the page could not be fetched ({kind}). Its "
                      f"message is on stderr and deliberately not here")
        elif not cap.parsed:
            fetched, stale = True, True
            reason = (f"the page's shape changed ({kind}); the raw page is archived and "
                      f"nothing was parsed out of it")
        elif cap.error is not None:
            fetched, stale = True, True
            reason = f"a linked document could not be fetched ({kind}); the rest were kept"
        elif cap.reports == 0 and cap.at >= REPORTS_BEGIN:
            # The loop ran zero times and nothing was wrong, which inside the regime is the
            # page having changed under the parser -- a table, a reorganised CMS -- and was
            # recorded green (#278).
            fetched, stale = True, True
            reason = ("the page parsed but linked no report documents; nothing was archived "
                      "from it" + (f" ({skipped['count']} links skipped: "
                                   f"{', '.join(skipped['by_reason'])})"
                                   if skipped["count"] else ""))
        elif cap.lines_why and not cap.lines_why.startswith("skipped"):
            fetched, stale = True, True
            reason = f"reports captured; odds snapshot {cap.lines_why}"
        else:
            fetched, stale, reason = True, False, None
    got: dict[str, Any] = jsonio.summary(
        "bigten", "hub.fetch.bigten",
        season=season, week=week, deadline=deadline, fetched=fetched, stale=stale, reason=reason,
        documents=documents, skipped=skipped, lines=lines,
        missed={"count": len(missed), "deadlines": missed[-MISSED_LISTED:]},
        archive_rows=held.height,
        quota={"month": cfbd._month_key(), "used": cfbd.quota_used(quota_path),
               "limit": cfbd.FREE_TIER_MONTHLY},
    )
    p = Path(path or STATUS)
    atomic.write_text(p, jsonio.dumps(got, indent=2))
    said = reason or (f"deadline {deadline['id']} captured: {documents['seen']} documents, "
                      f"{documents['new']} new; lines {lines['rows']} rows")
    print(f"  bigten: {said}; {len(missed)} deadlines missed so far; recorded in {p}")
    return got


def not_yet(now: datetime | None = None) -> str | None:
    """Why no capture should be attempted now, or None if one should.

    Before the first report there is nothing to keep and a snapshot would spend a call on a
    page that says nothing. After the regular season the same, and that end is the season
    anchor's rather than a second date here.
    """
    at_now = now or _now()
    if at_now < REPORTS_BEGIN:
        # No "now" in the sentence: the stamp's `generated_at` says when, and a reason that
        # moved with the clock would make every pre-season run a commit.
        return (f"nothing was captured: the first availability report is due "
                f"{REPORTS_BEGIN.isoformat()}, and this run is before it")
    choice = cfbd.configured_week(at_now)
    if choice.week is None:
        return "nothing was captured: " + choice.why.removeprefix("nothing was fetched: ")
    return None


# How long after a deadline the watchdog waits before it will call the capture stalled
# (#240). A cron is a request for a start: `live.yml` records GitHub delivering scheduled
# starts 97-126 minutes late, and the capture itself takes under a minute once it runs. Three
# hours clears the worst delivery observed by half again, so a check inside the window is
# looking at a capture that has had every chance to run -- and the cost of the margin is that
# a stall goes unreported for three hours, against a page that serves last-good throughout.
WATCH_DELAY = timedelta(hours=3)


def watch(path: Path | None = None, *, now: datetime | None = None) -> str:
    """One line for the watchdog: what the stamp says about the deadline it should follow.

    Prints one of:

      preseason <reason>              nothing captures yet -- before the first report, or no
                                      season anchor -- so there is nothing to measure
      outside <deadline> <seconds>    the deadline fired less than `WATCH_DELAY` ago; a late
                                      start may still be queued, so a stale stamp is not yet
                                      evidence. The window opens in <seconds>
      inside <deadline> ok <age>      the stamp was written since the deadline; <age> is how
                                      long ago
      inside <deadline> stale <age>   the deadline fired, the delay has passed, and the stamp
                                      predates it by <age> seconds of now -- the stall
      inside <deadline> empty <age>   the stamp is fresh and says the page was fetched and
                                      no report document was archived (#278)
      unreachable                     no stamp at all
      unreadable                      a stamp with no `generated_at` this can read

    The window is the span from `WATCH_DELAY` after a deadline until the next deadline fires,
    because the stall is the same fact for the whole of it: nothing has written since the
    deadline. A capture stalled across the two midweek deadlines no watchdog cron sits after
    is still stale at the next one it does, since the stamp has not moved past that either.

    Read off the committed file rather than the published one, which is the opposite of the
    live overlay's check: `bigten.yml` commits the stamp and the commit is the record, while a
    push made with `GITHUB_TOKEN` triggers no deploy, so the published copy lags the commit
    until something else deploys. The watchdog's checkout has the commit.

    The three failures that are not a stall are named apart, for `heartbeat.sh`'s reason:
    the fix for each is different and one sentinel standing for all three is how the live
    check reported 56.7 years of staleness on every run.
    """
    at_now = now or _now()
    if (why := not_yet(at_now)) is not None:
        return f"preseason {why.removeprefix('nothing was captured: ')}"
    _deadline, at = deadline_at(at_now)
    if at < REPORTS_BEGIN:
        return (f"preseason the first availability report is due {REPORTS_BEGIN.isoformat()}, "
                f"and no deadline has fired since")
    ident = deadline_id(at)
    opens = at + WATCH_DELAY
    if at_now < opens:
        return f"outside {ident} {int((opens - at_now).total_seconds())}"
    p = Path(path or STATUS)
    if not p.exists():
        return "unreachable"
    try:
        stamp = json.loads(p.read_text())
        written = datetime.fromisoformat(stamp["generated_at"])
    except (ValueError, KeyError, TypeError, OSError):
        return "unreadable"
    if written.tzinfo is None:
        written = written.replace(tzinfo=UTC)
    age = int((at_now - written).total_seconds())
    if written < at:
        return f"inside {ident} stale {age}"
    # Written on time and archived no report: the stall wearing a fresh date (#278). A
    # stamp from before `reports` was recorded has nothing to say about it and reads as ok.
    docs = stamp.get("documents")
    if stamp.get("fetched") and isinstance(docs, dict) and docs.get("reports") == 0:
        return f"inside {ident} empty {age}"
    return f"inside {ident} ok {age}"


def status_report(index: Path | None = None, now: datetime | None = None,
                  status: Path | None = None) -> int:
    at_now = now or _now()
    held = read_index(index)
    _first, opens, _why = cfbd.week_one_opens()
    missed = missed_deadlines(held, now=at_now, opens=opens)
    deadlines = sorted(set(held["deadline"].to_list())) if held.height else []
    print(f"  bigten archive: {held.height:,} rows over {len(deadlines)} captured deadlines")
    if deadlines:
        print(f"  first {deadlines[0]}, latest {deadlines[-1]}")
    print(f"  missed deadlines since {REPORTS_BEGIN.date().isoformat()}: {len(missed)}")
    for s in missed[-MISSED_LISTED:]:
        print(f"    {s}")
    # What the last run declined to follow, off its stamp (#265): the one place a report
    # behind a host the rules do not know becomes visible.
    try:
        stamp = json.loads(Path(status or STATUS).read_text())
        skipped = stamp["skipped"]
        n = int(skipped["count"])
    except (OSError, ValueError, KeyError, TypeError):
        n = 0
        skipped = {}
    if n:
        reasons = ", ".join(f"{k} {v}" for k, v in skipped["by_reason"].items())
        print(f"  last run skipped {n} link{'s' if n != 1 else ''}: {reasons}; "
              f"hosts {', '.join(skipped['hosts'])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="hub.fetch.bigten",
        description="Archive the Big Ten availability reports for the current deadline, "
                    "with an odds snapshot beside them. One CFBD call per capture.")
    ap.add_argument("--capture", action="store_true",
                    help="fetch the page, archive the documents it links from the "
                         "conference's own host, snapshot lines")
    ap.add_argument("--status", action="store_true",
                    help="what the archive holds and which deadlines were missed")
    ap.add_argument("--watch", action="store_true",
                    help="one line for the watchdog: the stamp against the deadline it "
                         "should follow, or that this moment is outside every window")
    ap.add_argument("--skip-lines", action="store_true",
                    help="archive the reports and spend no CFBD call")
    ap.add_argument("--season", type=int, default=SEASON_AHEAD)
    ap.add_argument("--archive", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--index", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--lines-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--status-path", default=None,
                    help="where to record what this run did (default site/data/bigten.json)")
    ap.add_argument("--quota-path", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    cfbd.reset_run_budget()
    index = Path(a.index) if a.index else None
    spath = Path(a.status_path) if a.status_path else None
    qpath = Path(a.quota_path) if a.quota_path else None

    if a.watch:
        # Exit 0 whatever the line says: the watchdog reads the word, and a non-zero here
        # would fail the step before the verdict that turns the word into an incident.
        print(watch(spath))
        return 0
    if a.status or not a.capture:
        return status_report(index, status=spath)

    # GUARD nothing-to-capture-is-recorded [unit/test_fetch_bigten.py]: before the first
    # report, or with no season, the run leaves a record and spends nothing
    if (why := not_yet()) is not None:
        record_run(None, why=why, season=a.season, index=index, path=spath, quota_path=qpath)
        print(f"hub.fetch.bigten: {why}", file=sys.stderr)
        return 1
    # /GUARD

    try:
        cap = capture(season=a.season, archive=Path(a.archive) if a.archive else None,
                      index=index, lines_dir=Path(a.lines_dir) if a.lines_dir else None,
                      skip_lines=a.skip_lines, quota_path=qpath)
    except Exception as e:
        # The index refusing, or a disk that will not take a write. Caught for the reason
        # `cfbd.main` catches: the exit code of a scheduled job reaches nobody, and the
        # record has to be written by whatever survived.
        record_run(None, why=f"nothing was captured: {type(e).__name__}", season=a.season,
                   index=index, path=spath, quota_path=qpath)
        print(f"hub.fetch.bigten: nothing was captured: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1

    record_run(cap, season=a.season, index=index, path=spath, quota_path=qpath)
    if cap.error is not None:
        print(f"hub.fetch.bigten: {type(cap.error).__name__}: {cap.error}", file=sys.stderr)
    return 0 if cap.fetched else 1


if __name__ == "__main__":
    sys.exit(main())
