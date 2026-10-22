"""The information packet the LLM forecasters see (docs/prereg/llm-forecasters.md, "Information
packet").

Each as-of rule in the pre-registration gets its own tests, and each of those has a *planted
failure*: a packet whose field is tampered so the rule is broken, which must be refused. A check
that has only ever been shown passing proves nothing (memory: plant the failure before trusting
a check), so every rule is exercised at two doors -- the builder refusing bad inputs, and
`validate_packet` refusing a bad packet however it came to exist (a stored one, a hand-edited
one).

Nothing here reads the clock: `run_start` is always passed. Nothing reaches the network or
`data/`.
"""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from forecasters_fixtures import (
    FETCHED,
    KICKOFF,
    RUN_START,
    SNAPSHOT,
    _build,
    _day,
    _game,
    _info,
    _injuries,
    _market,
    _prov,
    _qb,
    _schedule,
    _season_games,
)

from hub.contracts import ContractViolation
from hub.forecasters.inputs import Schedule
from hub.forecasters.packet import (
    PACKET_CONTRACT,
    Condition,
    PacketRefused,
    canonical_json,
    packet_hash,
    validate_packet,
)

ROOT = Path(__file__).resolve().parents[2]


def _refusal_checks(exc: pytest.ExceptionInfo[PacketRefused]) -> set[str]:
    return {f.check for f in exc.value.failures}


# --- what the packet says ------------------------------------------------------------------


def test_blind_packet_carries_no_market_and_line_aware_does():
    blind, aware = _build(), _build(Condition.LINE_AWARE)
    assert blind.market is None and blind.condition == "blind"
    assert aware.market is not None and aware.condition == "line_aware"
    assert aware.market.home_win_prob == 0.6125 and aware.market.home_spread == -3.5
    assert aware.market.snapshot_id == SNAPSHOT
    assert "market" not in canonical_json(blind).replace('"market":null', "")


def test_a_blind_packet_handed_a_market_is_refused_rather_than_trusted_to_ignore_it():
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.BLIND, market=_market(), snapshot_id=SNAPSHOT)
    assert "condition" in _refusal_checks(exc)


def test_a_line_aware_packet_without_a_market_is_refused():
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.LINE_AWARE, market=None)
    assert "condition" in _refusal_checks(exc)


def test_game_block_has_teams_sides_neutral_flag_and_kickoff():
    p = _build(game=_info(neutral_site=True))
    assert (p.game.home_team, p.game.away_team) == ("KC", "BUF")
    assert p.game.neutral_site is True and p.game.kickoff == KICKOFF
    assert p.game.game_id == "2026_05_BUF_KC" and (p.game.season, p.game.week) == (2026, 5)


def test_probable_starting_qb_for_each_team_and_none_when_the_source_names_nobody():
    p = _build()
    assert (p.starting_qbs.home, p.starting_qbs.away) == ("Patrick Mahomes", "Josh Allen")
    assert _build(qb_source=_qb(starters={"KC": "Patrick Mahomes"})).starting_qbs.away is None


def test_season_record_and_point_differential_through_the_last_completed_game():
    p = _build()
    kc, buf = p.records.home, p.records.away
    assert (kc.wins, kc.losses, kc.ties, kc.games) == (2, 1, 1, 4)
    assert (kc.points_for, kc.points_against, kc.point_diff) == (92, 75, 17)
    assert (buf.wins, buf.losses, buf.ties, buf.games) == (2, 1, 1, 4)
    assert (buf.points_for, buf.points_against, buf.point_diff) == (95, 64, 31)  # 30+24+21+20


def test_last_three_results_newest_first_with_scores_sides_and_outcome():
    r = _build().recent_results.home
    assert [x.game_id for x in r] == ["w4_kc_mia", "w3_kc_den", "w2_buf_kc"]
    assert (r[0].points_for, r[0].points_against, r[0].outcome, r[0].venue) == (31, 14, "W", "home")
    assert (r[1].points_for, r[1].points_against, r[1].outcome) == (10, 17, "L")
    assert (r[2].points_for, r[2].points_against, r[2].outcome, r[2].venue) == (24, 24, "T", "away")
    assert r[2].opponent == "BUF"


def test_rest_days_count_eastern_calendar_days_from_the_last_completed_game():
    p = _build()                       # last games Sun Oct 4 for both; this one Sun Oct 11
    assert (p.rest_days.home, p.rest_days.away) == (7, 7)
    # A Thursday-night last game: Thu Oct 8 8:15pm ET is 00:15Z Friday and must count as Thursday.
    thu = datetime(2026, 10, 9, 0, 15, tzinfo=UTC)
    games = [*_season_games(), _game("w5_thu", "BUF", "NE", 17, 9, thu,
                                     completed_at=thu + timedelta(hours=3))]
    p = _build(schedule=_schedule(games))
    assert p.rest_days.away == 3 and p.rest_days.home == 7


def test_season_opener_has_no_rest_days_and_an_empty_record():
    p = _build(schedule=_schedule([]))
    assert p.rest_days.home is None and p.records.home.games == 0
    assert p.records.home.last_completed_at is None and p.recent_results.home == ()


def test_only_regular_season_games_of_this_season_count():
    old = _game("old", "KC", "LV", 50, 0, _day(1) - timedelta(days=300),
                completed_at=_day(1) - timedelta(days=300, hours=-3), season=2025)
    post = _game("post", "KC", "LV", 40, 0, _day(2), gtype="POST",
                 completed_at=_day(2) + timedelta(hours=3))
    p = _build(schedule=_schedule([*_season_games(), old, post]))
    assert p.records.home.points_for == 92 and p.records.home.games == 4


def test_injury_report_keeps_only_out_doubtful_questionable_in_a_stable_order():
    inj = _build().injuries
    assert [(e.player, e.status) for e in inj.home] == [("Chris Jones", "Out"),
                                                        ("Travis Kelce", "Questionable")]
    assert [(e.player, e.status) for e in inj.away] == [("Von Miller", "Doubtful")]
    assert inj.published_at == RUN_START - timedelta(hours=10)


def test_every_field_group_records_its_source_and_fetch_time():
    p = _build(Condition.LINE_AWARE)
    assert set(p.stamp) == {"game", "starting_qbs", "rest_days", "records",
                                 "recent_results", "injuries", "market"}
    for name, prov in p.stamp.items():
        assert prov.source and prov.fetched_at.tzinfo is not None, name
    assert p.stamp["injuries"].source == "espn:injuries"
    assert p.stamp["starting_qbs"].fetched_at == FETCHED
    assert "market" not in _build().stamp


def test_degraded_sources_are_recorded_as_such():
    p = _build(qb_source=replace(_qb(), stamp=_prov("nfeloqb", "last_good")))
    assert p.stamp["starting_qbs"].served_from == "last_good"


# --- as-of check 1: records, differentials, recent results ----------------------------------


def _with_extra(**kw: Any) -> Schedule:
    # An extra KC game played the day before this one's run, +100 points so inclusion is loud.
    g = _game("w5_extra", "KC", "NYG", 100, 0, RUN_START - timedelta(hours=30), **kw)
    return _schedule([*_season_games(), g])


def test_a_game_kicked_off_but_unfinished_is_excluded_from_record_and_results():
    started = _game("w5_live", "KC", "NYG", 14, 7, RUN_START - timedelta(hours=1), final=False)
    p = _build(schedule=_schedule([*_season_games(), started]))
    assert p.records.home.points_for == 92 and p.records.home.games == 4
    assert "w5_live" not in [r.game_id for r in p.recent_results.home]
    assert p.rest_days.home == 7                      # rest ignores the unfinished game too


def test_a_game_the_feed_does_not_call_final_is_excluded_however_old_its_kickoff():
    # A postponed or suspended game with a stale partial score, kicked off long before the
    # run: the completion bound alone would let it in, the final flag keeps it out.
    stale = _game("w5_susp", "KC", "NYG", 100, 0, RUN_START - timedelta(hours=30), final=False)
    assert _build(schedule=_schedule([*_season_games(), stale])).records.home.games == 4


def test_a_game_finished_after_run_start_is_excluded_even_though_the_data_says_final():
    # The feed is read during the run, so it can already say final for a game that ended after
    # run_start. That score is information from the future of the run.
    late = _game("w5_late", "KC", "NYG", 100, 0, RUN_START - timedelta(hours=1),
                 completed_at=RUN_START + timedelta(minutes=1))
    p = _build(schedule=_schedule([*_season_games(), late]))
    assert p.records.home.points_for == 92 and p.records.home.games == 4
    assert "w5_late" not in [r.game_id for r in p.recent_results.home]


def test_a_final_game_with_no_completion_time_counts_only_once_the_bound_has_passed():
    # Neither nflverse nor ESPN says when a game ended, so absent a time the packet assumes it
    # ended no earlier than kickoff + the longest game it will credit. Inside that window: out.
    recent = _game("w5_nots", "KC", "NYG", 100, 0, RUN_START - timedelta(hours=2))
    assert _build(schedule=_schedule([*_season_games(), recent])).records.home.games == 4
    old = _game("w5_nots2", "KC", "NYG", 100, 0, RUN_START - timedelta(hours=30))
    p = _build(schedule=_schedule([*_season_games(), old]))
    assert p.records.home.games == 5 and p.records.home.points_for == 192
    assert p.recent_results.home[0].completed_at_basis == "kickoff_plus_bound"


def test_a_game_that_finished_before_run_start_is_included():
    p = _build(schedule=_with_extra(completed_at=RUN_START - timedelta(hours=27)))
    assert p.records.home.games == 5 and p.recent_results.home[0].game_id == "w5_extra"
    assert p.records.home.last_completed_at == RUN_START - timedelta(hours=27)
    assert p.recent_results.home[0].completed_at_basis == "completed_at"


def test_a_game_finishing_exactly_at_run_start_is_excluded():
    p = _build(schedule=_with_extra(completed_at=RUN_START))
    assert p.records.home.games == 4


def test_planted_leak_in_a_record_is_refused_by_validation():
    p = _build()
    leaked = replace(p, records=replace(
        p.records, home=replace(p.records.home, last_completed_at=RUN_START + timedelta(hours=1))))
    with pytest.raises(PacketRefused) as exc:
        validate_packet(leaked, snapshot_id=None)
    assert "records_as_of" in _refusal_checks(exc)


def test_planted_leak_in_a_recent_result_is_refused_by_validation():
    p = _build()
    row = replace(p.recent_results.away[0], completed_at=RUN_START + timedelta(seconds=1))
    leaked = replace(p, recent_results=replace(p.recent_results, away=(row, *p.recent_results.away[1:])))
    with pytest.raises(PacketRefused) as exc:
        validate_packet(leaked, snapshot_id=None)
    assert "recent_results_as_of" in _refusal_checks(exc)


def test_a_clean_packet_validates():
    validate_packet(_build(), snapshot_id=None)
    validate_packet(_build(Condition.LINE_AWARE), snapshot_id=SNAPSHOT)


# --- as-of check 2: the injury report ------------------------------------------------------


def test_an_injury_report_published_after_run_start_is_refused():
    with pytest.raises(PacketRefused) as exc:
        _build(injuries=_injuries(RUN_START + timedelta(minutes=1)))
    assert "injury_published" in _refusal_checks(exc)


def test_an_injury_report_published_exactly_at_run_start_is_refused():
    with pytest.raises(PacketRefused):
        _build(injuries=_injuries(RUN_START))


def test_an_injury_report_with_no_publication_time_is_refused_not_waved_through():
    with pytest.raises(PacketRefused) as exc:
        _build(injuries=_injuries(None))
    assert "injury_published" in _refusal_checks(exc)


def test_a_stale_but_earlier_injury_report_is_accepted():
    assert _build(injuries=_injuries(RUN_START - timedelta(days=9))).injuries.published_at


def test_planted_injury_leak_is_refused_by_validation_of_a_stored_packet():
    p = _build()
    leaked = replace(p, injuries=replace(p.injuries, published_at=RUN_START + timedelta(hours=2)))
    with pytest.raises(PacketRefused) as exc:
        validate_packet(leaked, snapshot_id=None)
    assert "injury_published" in _refusal_checks(exc)


# --- as-of check 3: the starting-QB source -------------------------------------------------


def test_a_qb_source_updated_after_run_start_is_refused():
    with pytest.raises(PacketRefused) as exc:
        _build(qb_source=_qb(RUN_START + timedelta(minutes=1)))
    assert "qb_source_updated" in _refusal_checks(exc)


def test_a_qb_source_updated_exactly_at_run_start_is_refused():
    with pytest.raises(PacketRefused):
        _build(qb_source=_qb(RUN_START))


def test_a_qb_source_with_no_update_time_is_refused_not_waved_through():
    with pytest.raises(PacketRefused) as exc:
        _build(qb_source=_qb(None))
    assert "qb_source_updated" in _refusal_checks(exc)


def test_a_stale_but_earlier_qb_source_is_accepted():
    p = _build(qb_source=_qb(RUN_START - timedelta(days=24)))
    assert p.starting_qbs.updated_at == RUN_START - timedelta(days=24)


def test_planted_qb_leak_is_refused_by_validation_of_a_stored_packet():
    p = _build()
    leaked = replace(p, starting_qbs=replace(p.starting_qbs, updated_at=RUN_START + timedelta(hours=1)))
    with pytest.raises(PacketRefused) as exc:
        validate_packet(leaked, snapshot_id=None)
    assert "qb_source_updated" in _refusal_checks(exc)


# --- as-of check 4: the market snapshot ----------------------------------------------------


def test_market_from_another_snapshot_is_refused():
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.LINE_AWARE, market=_market("snap-yesterday"))
    assert "market_snapshot" in _refusal_checks(exc)


def test_a_market_that_matches_this_runs_snapshot_is_accepted():
    block = _build(Condition.LINE_AWARE, market=_market(SNAPSHOT)).market
    assert block is not None and block.snapshot_id == SNAPSHOT


def test_planted_market_from_another_snapshot_is_refused_by_validation():
    p = _build(Condition.LINE_AWARE)
    assert p.market is not None
    foreign = replace(p, market=replace(p.market, snapshot_id="snap-yesterday"))
    with pytest.raises(PacketRefused) as exc:
        validate_packet(foreign, snapshot_id=SNAPSHOT)
    assert "market_snapshot" in _refusal_checks(exc)


def test_a_run_with_no_snapshot_id_cannot_pass_a_market():
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.LINE_AWARE, snapshot_id=None)
    assert "market_snapshot" in _refusal_checks(exc)


def test_every_failed_check_is_reported_together():
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.LINE_AWARE, qb_source=_qb(RUN_START), injuries=_injuries(RUN_START),
               market=_market("snap-yesterday"))
    assert _refusal_checks(exc) == {"qb_source_updated", "injury_published", "market_snapshot"}
    assert "snap-yesterday" in str(exc.value)


def test_naive_run_start_and_naive_source_times_are_refused():
    with pytest.raises(PacketRefused) as exc:
        _build(run_start=RUN_START.replace(tzinfo=None))
    assert "run_start" in _refusal_checks(exc)
    with pytest.raises(PacketRefused) as exc:
        _build(qb_source=_qb(RUN_START.replace(tzinfo=None) - timedelta(days=1)))
    assert "qb_source_updated" in _refusal_checks(exc)


# --- the contract --------------------------------------------------------------------------


def test_a_market_probability_in_percent_not_proportion_is_a_contract_violation():
    bad = replace(_market(), home_win_prob=61.25)
    with pytest.raises(PacketRefused) as exc:
        _build(Condition.LINE_AWARE, market=bad)
    assert "contract" in _refusal_checks(exc)
    assert "market_home_win_prob" in str(exc.value)


def test_the_contract_itself_refuses_a_frame_with_a_missing_field():
    from hub.forecasters.packet import flat_frame
    frame = flat_frame(_build(Condition.LINE_AWARE))
    PACKET_CONTRACT.validate(frame)
    with pytest.raises(ContractViolation, match="market_home_win_prob"):
        PACKET_CONTRACT.validate(frame.drop("market_home_win_prob"))
    with pytest.raises(ContractViolation, match="home_rest_days"):
        PACKET_CONTRACT.validate(frame.with_columns(pl.lit(99).alias("home_rest_days")))


# --- canonical serialisation and the hash --------------------------------------------------

GOLDEN_BLIND = "c417cce83c7d83fd6274030422af709f695579d37a30ae8588c7f496b7fa341c"
GOLDEN_AWARE = "9f67636d882f5794f334463cbdf890268fc55e5c3c52510e315f9a76b5c72861"


def test_canonical_json_sorts_keys_and_fixes_float_format():
    assert canonical_json({"b": 1, "a": 0.5}) == canonical_json({"a": 0.5, "b": 1})
    assert canonical_json({"a": 0.5, "b": 1}) == '{"a":0.500000,"b":1}'
    assert canonical_json({"x": -0.0}) == '{"x":0.000000}'
    assert canonical_json({"x": -1e-9}) == '{"x":0.000000}'
    assert canonical_json({"x": 1 / 3}) == '{"x":0.333333}'
    assert canonical_json({"t": datetime(2026, 10, 10, 10, 0, tzinfo=UTC)}) == \
        '{"t":"2026-10-10T10:00:00.000000Z"}'


def test_canonical_json_refuses_nan_infinity_and_naive_datetimes():
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError):
            canonical_json({"x": bad})
    with pytest.raises(ValueError):
        canonical_json({"t": datetime(2026, 10, 10, 10, 0)})


def test_canonical_json_normalises_any_offset_to_utc():
    from zoneinfo import ZoneInfo
    eastern = datetime(2026, 10, 10, 10, 0, tzinfo=ZoneInfo("America/New_York"))
    assert canonical_json({"t": eastern}) == '{"t":"2026-10-10T14:00:00.000000Z"}'


def test_hash_is_stable_across_rebuilds_and_across_interpreters():
    a, b = _build(Condition.LINE_AWARE), _build(Condition.LINE_AWARE)
    assert a is not b and canonical_json(a) == canonical_json(b)
    assert packet_hash(a) == packet_hash(b) and len(packet_hash(a)) == 64
    # A fresh interpreter with a different hash seed: nothing may depend on dict or set order.
    code = ("import sys; sys.path.insert(0, 'tests'); "
            "from forecasters_fixtures import _build; "
            "from hub.forecasters.packet import Condition, packet_hash; "
            "print(packet_hash(_build(Condition.LINE_AWARE)))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, "PYTHONHASHSEED": "12345"}, check=True)
    assert out.stdout.strip() == packet_hash(a)


def test_hash_is_pinned_to_a_literal_so_format_drift_is_loud():
    assert packet_hash(_build()) == GOLDEN_BLIND
    assert packet_hash(_build(Condition.LINE_AWARE)) == GOLDEN_AWARE


def test_blind_and_line_aware_packets_hash_differently():
    assert packet_hash(_build()) != packet_hash(_build(Condition.LINE_AWARE))


def _leaves(obj: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, (*path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _leaves(v, (*path, i))
    else:
        yield path, obj


def _perturb(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v + 0.01
    if isinstance(v, datetime):
        return v + timedelta(seconds=1)
    if v is None:
        return "x"
    return str(v) + "x"


def test_hash_changes_when_any_single_field_changes():
    from hub.forecasters.packet import canonical_hash, plain
    d = plain(_build(Condition.LINE_AWARE))
    base = canonical_hash(d)
    leaves = list(_leaves(d))
    assert len(leaves) > 60                         # the walk really covers the packet
    for path, value in leaves:
        e = copy.deepcopy(d)
        node = e
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = _perturb(value)
        assert canonical_hash(e) != base, f"changing {path} did not change the hash"


def test_hash_changes_when_an_input_changes():
    base = packet_hash(_build())
    assert packet_hash(_build(qb_source=_qb(starters={"KC": "Carson Wentz", "BUF": "Josh Allen"}))) != base
    assert packet_hash(_build(game=_info(neutral_site=True))) != base
    assert packet_hash(_build(run_start=RUN_START + timedelta(minutes=1))) != base
