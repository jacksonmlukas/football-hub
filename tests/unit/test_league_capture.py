"""The weekly league capture (#545): every team's roster, standings and remaining schedule.

Synthetic objects shaped like `espn_api`'s -- no League is ever loaded here.
"""
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from league_fakes import make_league

from hub.fetch import league as L

NOW = datetime(2026, 11, 11, 15, 0, tzinfo=UTC)


def build(league=None, **kw):
    return L.build_capture(league or make_league(), swid="swid-2", now=NOW,
                           free_agents=None, **kw)


def test_league_capture_holds_every_team_and_marks_ours():
    cap = build()
    assert [t["team_id"] for t in cap["teams"]] == [1, 2, 3, 4]
    assert [t["is_mine"] for t in cap["teams"]] == [False, True, False, False]
    assert cap["my_team_id"] == 2
    assert (cap["season"], cap["week"], cap["captured_at"]) == (2026, 10, NOW.isoformat())


def test_league_capture_standings_and_roster_projections():
    t = build()["teams"][1]
    assert (t["wins"], t["losses"], t["ties"], t["points_for"]) == (1, 8, 0, 1001.0)
    [row] = t["roster"]
    assert row["player"] == "Player 2" and row["espn_week"] == 10
    assert row["espn_week_proj"] == 11.5
    # each remaining week through 17 where ESPN gives one; a 0.0 is no projection
    assert row["week_proj"] == {"10": 11.5, "11": 9.0}


def test_league_capture_schedule_runs_from_the_capture_week_to_the_end_of_the_regular_season():
    cap = build()
    mine = [s for s in cap["schedule"] if s["team_id"] == 2]
    assert [s["week"] for s in mine] == [10, 11, 12, 13, 14]
    assert mine[0] == {"week": 10, "team_id": 2, "opponent_id": 3}
    assert len(cap["schedule"]) == 4 * 5
    assert set(mine[0]) == {"week", "team_id", "opponent_id"}      # no home/away


def test_league_capture_playoff_weeks_follow_the_regular_season():
    s = build()["settings"]
    assert (s["reg_season_weeks"], s["playoff_team_count"]) == (14, 4)
    assert s["playoff_weeks"] == [15, 16]
    assert build(make_league(playoff=6))["settings"]["playoff_weeks"] == [15, 16, 17]


def test_league_capture_trade_deadline_is_iso_utc_and_zero_is_null():
    assert build()["settings"]["trade_deadline"] == "2024-12-02T00:40:00+00:00"
    assert build(make_league(deadline_ms=0))["settings"]["trade_deadline"] is None


def test_league_capture_copies_the_free_agent_pool_as_written():
    pool = {"fetched_at": "2026-11-11T14:00:00+00:00", "capped": True,
            "rows": [{"player": "Waiver Guy", "espn_id": 9}]}
    cap = L.build_capture(make_league(), swid="swid-2", now=NOW, free_agents=pool)
    assert cap["free_agents"] == pool
    assert build()["free_agents"] is None


# --- the CLI: one load, a file per week, nothing written when ESPN is out of reach ----------

def run(tmp_path, loader, **kw):
    return L.capture(loader, out_root=tmp_path, now=NOW, swid="swid-2",
                     free_agents=lambda: None, **kw)[0]


def test_league_capture_writes_one_file_per_week_and_a_rerun_overwrites(tmp_path):
    loads = []

    def loader():
        loads.append(1)
        return make_league()

    run(tmp_path, loader)
    run(tmp_path, loader)
    [f] = list(tmp_path.rglob("*.json"))
    assert f == tmp_path / "2026" / "wk10.json"
    assert len(loads) == 2                                    # one load per run, not per team
    assert json.loads(f.read_text())["week"] == 10


def test_league_capture_offline_writes_nothing_and_leaves_last_good(tmp_path):
    run(tmp_path, make_league)
    good = (tmp_path / "2026" / "wk10.json").read_bytes()

    def down():
        raise ConnectionError("ESPN unreachable")

    msg = run(tmp_path, down)
    assert "nothing written" in msg
    assert (tmp_path / "2026" / "wk10.json").read_bytes() == good
    assert sorted(p.name for p in tmp_path.rglob("*")) == ["2026", "wk10.json"]


def test_league_capture_offline_with_no_last_good_creates_nothing(tmp_path):
    def down():
        raise OSError("no cookie")

    run(tmp_path, down)
    assert list(tmp_path.rglob("*")) == []


def test_league_capture_refuses_a_payload_the_contract_rejects_and_keeps_last_good(tmp_path):
    run(tmp_path, make_league)
    good = (tmp_path / "2026" / "wk10.json").read_bytes()
    bad = make_league()
    bad.teams[0].team_id = 2                                  # two teams with one id
    msg = run(tmp_path, lambda: bad)
    assert "refused" in msg
    assert (tmp_path / "2026" / "wk10.json").read_bytes() == good


def test_league_capture_main_exits_zero_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "load_league", lambda: (_ for _ in ()).throw(OSError("offline")))
    monkeypatch.setattr(L, "STATE_LEAGUE", Path(tmp_path))
    assert L.main(["--write"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert list(tmp_path.rglob("*")) == []
    assert L.main([]) == 1                    # a look by hand that captured nothing says so
    assert L.main(["--write", "--out", str(tmp_path / "elsewhere")]) == 0
    assert list(tmp_path.rglob("*")) == []


def test_league_capture_a_build_bug_is_not_called_a_network_failure(tmp_path):
    broken = make_league()
    del broken.settings.reg_season_count        # build_capture raises AttributeError
    line, ok = L.capture(lambda: broken, out_root=tmp_path, now=NOW, swid="swid-2",
                         free_agents=lambda: None)
    assert not ok and "the capture could not be built (AttributeError:" in line
    assert "unreachable" not in line and "nothing written" in line
    assert list(tmp_path.rglob("*")) == []


def test_league_capture_file_is_compact_json_with_a_trailing_newline(tmp_path):
    run(tmp_path, make_league)
    text = (tmp_path / "2026" / "wk10.json").read_text()
    assert text.endswith("}\n") and "\n" not in text[:-1] and ": " not in text[:200]
    assert json.loads(text)["week"] == 10


def test_league_capture_without_write_summarises_and_creates_nothing(tmp_path):
    line, ok = L.capture(make_league, out_root=tmp_path, now=NOW, swid="swid-2",
                         free_agents=lambda: None, write=False)
    assert ok and "not written" in line and list(tmp_path.rglob("*")) == []


def test_league_capture_reads_the_pool_last_good_with_its_cap_flag(tmp_path, monkeypatch):
    from hub.fetch import free_agents as FA
    parquet = tmp_path / "free_agents.parquet"
    monkeypatch.setattr(FA, "FREE_AGENTS_PARQUET", parquet)
    assert L.read_free_agents() is None                # nothing fetched yet
    pool = FIXTURE_POOL["rows"]
    import polars as pl

    from hub.fetch.espn import FREE_AGENT_SCHEMA
    FA.refresh(lambda: (pl.DataFrame(pool, schema=FREE_AGENT_SCHEMA), True), parquet, now=NOW)
    got = L.read_free_agents()
    assert got is not None and got["capped"] is True and got["rows"] == pool
    assert got["fetched_at"] == NOW.isoformat()


def test_league_capture_preseason_week_is_not_captured(tmp_path):
    msg = run(tmp_path, lambda: make_league(week=0))
    assert "nothing written" in msg and list(tmp_path.rglob("*")) == []


FIXTURE = (Path(__file__).resolve().parents[1] / "golden" / "fixtures"
           / "league_capture.synthetic.json")
FIXTURE_POOL = {"fetched_at": "2026-11-11T14:00:00+00:00", "capped": False, "rows": [
    {"player": "Sample Runner", "key": "sample runner", "pos": "RB", "espn_id": 900001,
     "nfl_team": "DAL", "injury_status": "ACTIVE", "espn_avg": 9.5, "espn_total": 133.0,
     "espn_week": 10, "espn_week_proj": 10.1, "percent_owned": 41.2}]}


def test_league_capture_frozen_fixture_is_what_the_builder_produces():
    """The fixture pins the shape: a change to `build_capture`'s output is a change to
    `league_capture.synthetic.json` in the same commit."""
    built = L.build_capture(make_league(), swid="swid-2", now=NOW, free_agents=FIXTURE_POOL)
    assert json.loads(FIXTURE.read_text()) == json.loads(json.dumps(built))


@pytest.mark.parametrize("week", [1, 18])
def test_league_capture_names_the_file_by_zero_padded_week(tmp_path, week):
    run(tmp_path, lambda: make_league(week=week))
    assert (tmp_path / "2026" / f"wk{week:02d}.json").exists()
