"""`qb_source_from_nfeloqb`: the nfeloqb per-team state as a packet source.

The state carries the starter per team and no timestamp at all, so the update time is handed
in -- the committer date of the commit that last touched `qb_elos.csv`, which only the caller
can look up -- and an unknown time stays `None`, which the packet then refuses.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest
from forecasters_fixtures import RUN_START, _build, _prov

from hub.fetch import nfeloqb
from hub.forecasters.packet import PacketRefused
from hub.forecasters.qb import QB_TIME_BASIS, qb_source_from_nfeloqb


def _state() -> pl.DataFrame:
    return pl.DataFrame(
        [{"team": "KC", "qb": "Patrick Mahomes", "qb_value": 120.0, "qb_adj": 1.5, "tenure": 80,
          "as_of": "2026-10-04"},
         {"team": "BUF", "qb": "Josh Allen", "qb_value": 118.0, "qb_adj": 0.0, "tenure": 70,
          "as_of": "2026-10-04"}],
        schema=nfeloqb.STATE_SCHEMA)


def test_starters_come_from_the_state_and_the_time_from_the_caller():
    when = datetime(2026, 10, 6, 16, 12, 55, tzinfo=UTC)
    src = qb_source_from_nfeloqb(_state(), commit_time=when, stamp=_prov("nfeloqb"))
    assert src.starters == {"KC": "Patrick Mahomes", "BUF": "Josh Allen"}
    assert src.updated_at == when and src.updated_at_basis == QB_TIME_BASIS
    assert _build(qb_source=src).starting_qbs.home == "Patrick Mahomes"


def test_an_unknown_commit_time_is_kept_unknown_and_the_packet_refuses_it():
    src = qb_source_from_nfeloqb(_state(), commit_time=None, stamp=_prov("nfeloqb"))
    assert src.updated_at is None
    with pytest.raises(PacketRefused) as exc:
        _build(qb_source=src)
    assert {f.check for f in exc.value.failures} == {"qb_source_updated"}


def test_a_commit_made_after_run_start_is_refused():
    src = qb_source_from_nfeloqb(_state(), commit_time=RUN_START + timedelta(seconds=1),
                                 stamp=_prov("nfeloqb"))
    with pytest.raises(PacketRefused):
        _build(qb_source=src)


# --- the latest commit before the run, not the pin (maintainer, 2026-10-06) -------------
#
# The hub pins `nfeloqb.COMMIT` for reproducibility, and on 2026-10-06 that pin was 3.5 weeks
# behind the source. The study reads the newest commit touching `qb_elos.csv` whose committer
# date precedes run start, and the file *at that commit* -- so the update time and the bytes
# are the same fact.

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from hub.forecasters.qb import QbSourceUnavailable, latest_commit_before, qb_source_at  # noqa: E402

_FIXTURES = Path(__file__).resolve().parents[1] / "golden" / "fixtures"
_SHA = "3880b8ef" + "0" * 32
_COMMITTED = "2026-10-06T16:12:55Z"


def _commits(*, committer: str = _COMMITTED, author: str = "2026-10-01T00:00:00Z",
             sha: str = _SHA) -> bytes:
    return json.dumps([{"sha": sha, "commit": {"author": {"date": author},
                                               "committer": {"date": committer}}}]).encode()


def _csv() -> bytes:
    got = json.loads((_FIXTURES / "nfeloqb_qb_elos.synthetic.json").read_text())
    cols = list(got[0])
    def cell(v):
        return "" if v is None else str(v)
    return ("\n".join([",".join(cols)] + [",".join(cell(r[c]) for c in cols) for r in got])
            + "\n").encode()


class _Get:
    """A fake network: the GitHub commits API answers `commits`, raw content answers the CSV."""
    def __init__(self, commits: bytes):
        self.commits, self.urls = commits, []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        return self.commits if url.startswith("https://api.github.com/") else _csv()


def test_the_commit_lookup_asks_for_the_file_and_stops_at_run_start():
    get = _Get(_commits())
    latest_commit_before(RUN_START, get=get)
    (url,) = get.urls
    assert url.startswith(f"https://api.github.com/repos/{nfeloqb.REPO}/commits?")
    assert f"path={nfeloqb.FILE}" in url and "per_page=1" in url
    assert f"until={RUN_START.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}" in url


def test_the_update_time_is_the_committer_date_not_the_author_date():
    sha, when = latest_commit_before(RUN_START, get=_Get(_commits()))
    assert sha == _SHA
    assert when == datetime(2026, 10, 6, 16, 12, 55, tzinfo=UTC)


def test_a_commit_not_strictly_before_run_start_is_refused_even_if_the_api_returns_it():
    at = f"{RUN_START.astimezone(UTC):%Y-%m-%dT%H:%M:%SZ}"
    with pytest.raises(QbSourceUnavailable):
        latest_commit_before(RUN_START, get=_Get(_commits(committer=at)))


def test_no_commit_before_run_start_is_unavailable_not_unknown():
    with pytest.raises(QbSourceUnavailable):
        latest_commit_before(RUN_START, get=_Get(b"[]"))


def test_the_file_is_read_at_that_commit_not_at_the_pin_or_the_branch():
    get = _Get(_commits())
    src = qb_source_at(RUN_START, get=get, stamp=_prov("nfeloqb"))
    raw = [u for u in get.urls if u.startswith("https://raw.githubusercontent.com/")]
    assert raw == [nfeloqb.url(_SHA)]
    assert nfeloqb.COMMIT is None or nfeloqb.COMMIT not in raw[0]
    assert src.updated_at == datetime(2026, 10, 6, 16, 12, 55, tzinfo=UTC)
    assert src.updated_at_basis == QB_TIME_BASIS
    assert src.starters                       # parsed from the file at that commit
