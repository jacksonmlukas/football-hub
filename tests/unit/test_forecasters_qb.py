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
