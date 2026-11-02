"""`hub.forecasters.sources`: the schedule, injury, odds-matching and hub adapters (#447).

Pure functions of payloads already in hand; nothing here reaches the network.
"""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from forecasters_slate import DUE, NAMES, RUN_START, injuries_payload, odds_payload, schedule_frame

from hub.forecasters import sources
from hub.forecasters.inputs import SourceStamp
from hub.forecasters.packet import MarketBlock
from hub.models.market import MARGIN_SD, normal_cdf

STAMP = SourceStamp("test", RUN_START)


def test_kickoff_is_eastern_wall_clock_through_the_daylight_change():
    assert sources.kickoff("2026-10-25", "13:00") == datetime(2026, 10, 25, 17, tzinfo=UTC)
    assert sources.kickoff("2026-11-01", "13:00") == datetime(2026, 11, 1, 18, tzinfo=UTC)
    assert sources.kickoff("2026-11-01", None) == datetime(2026, 11, 1, 18, tzinfo=UTC)


def test_the_schedule_marks_a_game_final_only_with_both_scores():
    sched = sources.schedule_from_nflverse(schedule_frame(), STAMP)
    by = {g.game_id: g for g in sched.games}
    assert by["2026_06_KC_BUF"].final and by["2026_06_KC_BUF"].home_score == 27
    assert not by["2026_07_BUF_KC"].final and by["2026_07_BUF_KC"].home_score is None
    assert all(g.completed_at is None for g in sched.games)      # no source says when it ended


def test_game_infos_flag_a_neutral_site_and_skip_the_postseason():
    frame = schedule_frame()
    infos = sources.game_infos(frame)
    assert infos["2026_07_JAX_NYJ"].neutral_site and not infos["2026_07_BUF_KC"].neutral_site
    post = frame.head(1).with_columns(game_id=frame["game_id"].head(1) + "_POST",
                                      game_type=frame["game_type"].head(1).str.replace("REG", "WC"))
    assert "2026_06_KC_BUF_POST" not in sources.game_infos(post)


def test_injuries_keep_only_items_dated_before_run_start_and_say_so():
    payload = injuries_payload()
    payload["injuries"][0]["injuries"].append(
        {"status": "Out", "date": "2026-10-18T13:45Z",                  # after run start
         "athlete": {"displayName": "Late News", "team": {"abbreviation": "KC"}}})
    rep = sources.injuries_from_espn(payload, run_start=RUN_START, stamp=STAMP)
    players = {e.player for e in rep.entries}
    assert "Late News" not in players and "Travis Kelce" in players
    assert rep.published_at is not None and rep.published_at < RUN_START
    assert rep.published_at == RUN_START.replace(second=0) - timedelta(hours=14)
    assert "strictly before run start" in rep.published_at_basis


def test_injuries_map_espn_abbreviations_to_nflverse():
    rep = sources.injuries_from_espn(injuries_payload(), run_start=RUN_START, stamp=STAMP)
    assert {e.team for e in rep.entries} == {"KC", "WAS", "LA"}


def test_an_injury_report_with_nothing_before_run_start_has_no_time():
    rep = sources.injuries_from_espn(injuries_payload(dated=RUN_START + timedelta(hours=31)),
                                     run_start=RUN_START, stamp=STAMP)
    assert rep.published_at is None and rep.entries == ()


def test_unreadable_injury_items_are_skipped():
    payload = {"injuries": [{"injuries": [{"status": "Out", "date": "not a date",
                                           "athlete": {"displayName": "X",
                                                       "team": {"abbreviation": "KC"}}},
                                          {"status": "Out", "date": "2026-10-17T10:00",
                                           "athlete": {"displayName": "Naive"}}]}]}
    rep = sources.injuries_from_espn(payload, run_start=RUN_START, stamp=STAMP)
    assert rep.entries == ()


def test_the_matcher_finds_each_due_game_by_teams_and_eastern_date():
    infos = sources.game_infos(schedule_frame())
    match = sources.matcher(infos, NAMES)
    assert [match(ev) for ev in odds_payload()] == list(DUE)
    late = odds_payload(("2026_07_WAS_LA",))[0]                   # 20:20 ET is 00:20Z next day
    assert late["commence_time"].startswith("2026-10-19") and match(late) == "2026_07_WAS_LA"
    assert match({"home_team": "Nobody", "away_team": "Nobody",
                  "commence_time": "2026-10-18T17:00:00Z"}) is None
    assert match({"commence_time": "garbage"}) is None


def test_the_hub_is_the_ratings_model_on_this_runs_spread():
    hub = sources.RatingsHub(2026, 7, version="abc123")
    block = MarketBlock("snap", RUN_START, 0.6, -3.5)
    assert hub.version == "abc123"
    assert hub.predict("g", block) == pytest.approx(normal_cdf(3.5 / MARGIN_SD))
    away_favoured = hub.predict("g", MarketBlock("snap", RUN_START, 0.4, 3.5))
    assert away_favoured is not None and away_favoured < 0.5


def test_the_commit_comes_from_the_workflow_first(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "f00d")
    assert sources.commit() == "f00d"
    monkeypatch.delenv("GITHUB_SHA")
    assert sources.commit()                                       # the checkout's HEAD


@pytest.mark.parametrize(("day", "season"), [(date(2026, 10, 18), 2026),
                                             (date(2027, 1, 3), 2026)])
def test_season_of(day, season):
    assert sources.season_of(day) == season
