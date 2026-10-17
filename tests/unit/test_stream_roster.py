"""The K / D/ST streaming heuristic rides on roster.json, labelled (#494)."""
import json

import polars as pl
import pytest

from hub import publish
from hub.contracts import ContractViolation, check_roster_artifact
from hub.fetch.replay import serve
from hub.season import stream


def _roster():
    return pl.DataFrame(
        {"player": ["Chase"], "pos": ["WR"], "nfl_team": ["CIN"], "mu": [19.7],
         "sd": [6.0], "projected": [True], "starting": [True], "injury_status": ["ACTIVE"],
         "available": [True], "can_start": [True], "missing_games": [0]})


@pytest.fixture(autouse=True)
def offline():
    serve(schedules=pl.DataFrame(
        schema={"game_id": pl.Utf8, "season": pl.Int32, "week": pl.Int32,
                "home_team": pl.Utf8, "away_team": pl.Utf8, "result": pl.Float64}))


def test_the_roster_artifact_carries_the_stream_and_refuses_an_unlabelled_one(
        tmp_path, monkeypatch):
    src, site = tmp_path / "roster.parquet", tmp_path / "site"
    site.mkdir()
    _roster().write_parquet(src)
    got = {"label": "heuristic", "dst": [], "k": []}
    monkeypatch.setattr(publish, "_roster_stream", lambda df: got)
    payload = publish.roster(out=site, path=src)
    assert isinstance(payload, dict) and payload["stream"] == got
    assert json.loads((site / "roster.json").read_text())["stream"]["label"] == "heuristic"
    # an unlabelled ordering would be a claim of edge nobody has measured
    with pytest.raises(ContractViolation, match="not 'heuristic'"):
        check_roster_artifact({**payload, "stream": {**got, "label": "edge"}})
    row = {"team": "AAA", "rank": 1, "opp_implied": None, "missing": None}
    with pytest.raises(ContractViolation, match="ranked with no opp_implied"):
        check_roster_artifact({**payload, "stream": {**got, "dst": [row]}})
    gone = {"team": "AAA", "rank": None, "own_implied": None, "missing": None}
    with pytest.raises(ContractViolation, match="unranked with no reason"):
        check_roster_artifact({**payload, "stream": {**got, "k": [gone]}})


def test_the_real_seam_degrades_labelled_when_the_schedule_has_no_open_week():
    got = publish._roster_stream(_roster())
    assert got["label"] == "heuristic" and got["dst"] == [] and got["missing"]
    assert got == stream.for_roster(_roster(), publish.SEASON_AHEAD)
