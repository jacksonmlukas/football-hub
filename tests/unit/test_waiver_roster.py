"""The waiver ranking rides on roster.json, labelled provisional (#493)."""
import json

import polars as pl
import pytest

from hub import publish
from hub.contracts import ContractViolation, check_roster_artifact
from hub.fetch import free_agents as fa_mod
from hub.fetch.espn import FREE_AGENT_SCHEMA
from hub.fetch.replay import serve
from hub.season import stream, waiver


def _roster():
    return pl.DataFrame(
        {"player": ["Chase", "Weak Back"], "pos": ["WR", "RB"], "nfl_team": ["CIN", "GB"],
         "slot": ["WR", "BE"], "espn_total": [150.0, 80.0], "espn_week_proj": [11.0, 5.0],
         "mu": [19.7, 6.0], "sd": [6.0, 4.0], "projected": [True, True],
         "starting": [True, False], "injury_status": ["ACTIVE", "ACTIVE"],
         "available": [True, True], "can_start": [True, True], "missing_games": [0, 0]})


@pytest.fixture(autouse=True)
def offline():
    serve(schedules=pl.DataFrame(
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "result": pl.Float64}))


def test_the_roster_artifact_carries_the_waiver_block_and_refuses_an_unlabelled_one(
        tmp_path, monkeypatch):
    src, site = tmp_path / "roster.parquet", tmp_path / "site"
    site.mkdir()
    _roster().write_parquet(src)
    row = {"player": "Rb Alpha", "value": 50.0, "replaces": "Weak Back", "blocked": None,
           "recommended": True}
    got = {"label": "provisional", "rows": [row], "recommended": ["Rb Alpha"]}
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: got)
    monkeypatch.setattr(publish, "_roster_stream", lambda df: stream.unavailable("stubbed"))
    payload = publish.roster(out=site, path=src)
    assert isinstance(payload, dict) and payload["waiver"] == got
    assert json.loads((site / "roster.json").read_text())["waiver"]["label"] == "provisional"
    # a ranking presented as anything stronger is refused
    with pytest.raises(ContractViolation, match="not 'provisional'"):
        check_roster_artifact({**payload, "waiver": {**got, "label": "edge"}})


@pytest.mark.parametrize("bad", [
    {"value": -1.0},                # worse than the replacement
    {"value": 0.0},                 # no gain
    {"value": None},                # no projection
    {"value": 12.0, "replaces": None},      # no named player to replace
    {"value": 12.0, "blocked": "OUT"},      # cannot play
    {"value": float("nan")},
])
def test_a_recommendation_without_a_positive_gain_is_refused(bad):
    row = {"player": "X", "value": 5.0, "replaces": "Y", "blocked": None,
           "recommended": True, **bad}
    with pytest.raises(ContractViolation, match="recommended without a positive value"):
        check_roster_artifact({"rows": [], "waiver": {"label": "provisional", "rows": [row]}})


def test_an_unrecommended_row_may_carry_a_negative_value():
    row = {"player": "X", "value": -4.0, "replaces": "Y", "blocked": None, "recommended": False}
    check_roster_artifact({"rows": [], "waiver": {"label": "provisional", "rows": [row]}})


def test_a_payload_without_the_block_still_passes():
    check_roster_artifact({"rows": []})


def test_the_real_seam_degrades_labelled_with_no_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(fa_mod, "FREE_AGENTS_PARQUET", tmp_path / "nope.parquet")
    got = publish._roster_waiver(_roster())
    assert got["label"] == "provisional" and got["rows"] == [] and got["missing"]


def test_the_real_seam_never_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("bad pool")
    monkeypatch.setattr(waiver, "for_roster", boom)
    got = publish._roster_waiver(_roster())
    assert got["label"] == "provisional" and "RuntimeError" in got["missing"]


def test_the_stream_seam_reads_the_last_good_pool_for_availability(tmp_path, monkeypatch):
    path = tmp_path / "fa.parquet"
    monkeypatch.setattr(fa_mod, "FREE_AGENTS_PARQUET", path)
    pool = pl.DataFrame([{"player": "Our D", "key": "our d", "pos": "D/ST", "espn_id": 1,
                          "nfl_team": "NYJ", "injury_status": "ACTIVE", "espn_avg": 5.0,
                          "espn_total": 70.0, "espn_week": 5, "espn_week_proj": 5.0,
                          "percent_owned": 3.0}], schema=FREE_AGENT_SCHEMA)
    fa_mod.refresh(lambda: (pool, False), path)
    seen: list = []

    def spy(roster, season, free_agents=None, pool_meta=None, **k):
        seen.append((free_agents, pool_meta))
        return stream.unavailable("spied")
    monkeypatch.setattr(stream, "for_roster", spy)
    publish._roster_stream(_roster())
    assert seen[0][0].height == 1 and seen[0][1]["n"] == 1   # type: ignore[union-attr]


def test_an_unreadable_pool_does_not_stop_the_stream(monkeypatch):
    def boom(*a, **k):
        raise OSError("disk")
    monkeypatch.setattr(fa_mod, "read", boom)
    got = publish._roster_stream(_roster())
    assert got["label"] == "heuristic"
