"""The free-agent fetch keeps last-good and says why (#493)."""
import json
from datetime import UTC, datetime

import polars as pl
import pytest

from hub.fetch import free_agents as F
from hub.fetch.espn import FREE_AGENT_SCHEMA

NOW = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


def _pool(*ids):
    rows = [{"player": f"P {i}", "key": f"p {i}", "pos": "RB", "espn_id": i,
             "nfl_team": "GB", "injury_status": "ACTIVE", "espn_avg": 5.0, "espn_total": 70.0,
             "espn_week": 7, "espn_week_proj": 5.5, "percent_owned": 10.0} for i in ids]
    return pl.DataFrame(rows, schema=FREE_AGENT_SCHEMA)


def _boom():
    raise ConnectionError("ESPN is down")


def test_a_good_fetch_writes_the_pool_and_a_sidecar(tmp_path):
    path = tmp_path / "fa.parquet"
    line, df = F.refresh(lambda: (_pool(1, 2), False), path, NOW)
    assert df is not None and df.height == 2 and "fetched 2 free agents" in line
    assert pl.read_parquet(path).height == 2
    meta = json.loads(F.meta_path(path).read_text())
    assert meta == {"fetched_at": NOW.isoformat(), "n": 2, "capped": False, "last_error": None}


def test_a_failed_fetch_serves_last_good_untouched_and_records_the_reason(tmp_path):
    path = tmp_path / "fa.parquet"
    F.refresh(lambda: (_pool(1, 2, 3), False), path, NOW)
    before = path.read_bytes()
    line, df = F.refresh(_boom, path, NOW)
    assert df is not None and df.height == 3
    assert "serving last-good" in line and "ESPN is down" in line
    assert path.read_bytes() == before                    # the pool keeps its own date
    meta = F.read_meta(path)
    assert "ESPN is down" in meta["last_error"] and meta["n"] == 3   # earlier facts survive


def test_a_failed_fetch_with_nothing_on_disk_is_a_sentence_not_a_frame(tmp_path):
    line, df = F.refresh(_boom, tmp_path / "fa.parquet", NOW)
    assert df is None and "none are on disk" in line and "ESPN is down" in line


def test_a_pool_at_the_cap_says_so(tmp_path):
    line, _ = F.refresh(lambda: (_pool(1), True), tmp_path / "fa.parquet", NOW)
    assert "cap" in line
    assert F.read_meta(tmp_path / "fa.parquet")["capped"] is True


def test_read_is_none_before_any_fetch_and_the_pool_after(tmp_path):
    path = tmp_path / "fa.parquet"
    assert F.read(path) == (None, {})
    F.refresh(lambda: (_pool(1), False), path, NOW)
    df, meta = F.read(path)
    assert df is not None and df.height == 1 and meta["n"] == 1


def test_an_unreadable_sidecar_is_no_note_not_an_error(tmp_path):
    path = tmp_path / "fa.parquet"
    F.meta_path(path).write_text("not json")
    assert F.read_meta(path) == {}
    F.meta_path(path).write_text("[1]")
    assert F.read_meta(path) == {}


def test_the_summary_counts_by_position_and_prints_no_rows():
    assert F.summary(_pool(1, 2)) == "RB 2"


def test_main_with_no_league_and_nothing_on_disk_exits_non_zero(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    assert F.main(["--out", str(tmp_path / "nope.parquet")]) == 1
    assert "hub.fetch.free_agents" in capsys.readouterr().err


def test_main_writes_and_summarises_without_printing_rows(tmp_path, capsys, monkeypatch):
    import hub.fetch.espn as espn
    monkeypatch.setattr(espn, "fetch_free_agents", lambda: (_pool(1, 2), False))
    assert F.main(["--out", str(tmp_path / "fa.parquet")]) == 0
    out = capsys.readouterr().out
    assert "fetched 2 free agents" in out and "by position: RB 2" in out
    assert "P 1" not in out


@pytest.mark.parametrize("flag", ["--help"])
def test_help_exits_zero(flag):
    with pytest.raises(SystemExit) as e:
        F.main([flag])
    assert e.value.code == 0
