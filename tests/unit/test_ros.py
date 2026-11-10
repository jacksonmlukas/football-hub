"""Scoring a Move over the rest of the season (#501): `hub.season.ros`.

The capture is hand-built (no League is loaded and nothing reaches a network): twelve teams on
the real 14-week round-robin shape, so the bracket `hub.draft.season.champion` plays is the one
the league plays.
"""
from __future__ import annotations

import itertools
import json
import time
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from hub.draft import season as draft_season
from hub.season import ros

NOW = datetime(2026, 11, 11, 15, 0, tzinfo=UTC)
SLOTS = [("QB", 1), ("RB", 3), ("WR", 3), ("TE", 1)]
BYES = {"DAL": 7, "PHI": 9}
_IDS = itertools.count(1)
SCALE = {"QB": 1.6, "RB": 1.2, "WR": 1.1, "TE": 0.8}


def _player(name, pos, proj, team="DAL", slot="BE", wp=None, avg=None):
    return {"player": name, "key": name.lower(), "pos": pos, "espn_id": next(_IDS),
            "nfl_team": team, "slot": slot, "injury_status": "ACTIVE",
            "espn_avg": proj if avg is None else avg, "espn_total": 100.0, "espn_week": 5,
            "espn_week_proj": proj, "week_proj": wp or {}}


def _roster(tid, base):
    rows = [_player(f"{pos}{k} t{tid}", pos, base * SCALE[pos], team="PHI")
            for pos, n in SLOTS for k in range(n)]
    rows.append(_player(f"RB bench t{tid}", "RB", base * 0.7, team="PHI"))   # the flex
    rows.append(_player(f"WR bench t{tid}", "WR", base * 0.4, team="PHI"))   # the second flex
    rows.append(_player(f"WR deep t{tid}", "WR", base * 0.2, team="PHI"))    # starts nowhere
    return rows


def make_capture(week=5, deadline="2027-01-01T00:00:00+00:00", captured=NOW):
    base = [8.0 + 0.5 * i for i in range(12)]
    teams = [{"team_id": i + 1, "name": f"Team {i + 1}", "is_mine": i == 0,
              "wins": 2 + (i % 3), "losses": 2, "ties": 0, "points_for": 400.0 + 3 * i,
              "roster": _roster(i + 1, base[i])} for i in range(12)]
    sched = []
    for w, pairs in enumerate(draft_season._round_robin(12, 14), start=1):
        if w >= week:
            for a, b in pairs:
                sched += [{"week": w, "team_id": a + 1, "opponent_id": b + 1},
                          {"week": w, "team_id": b + 1, "opponent_id": a + 1}]
    pool = [_player("Waiver Star", "RB", 30.0, team="PHI"), _player("Waiver Dud", "RB", 1.0, team="PHI"),
            _player("Waiver Twin", "RB", 8.4, team="PHI")]
    for p in pool:
        p.pop("slot")
        p["percent_owned"] = 5.0
        p.pop("week_proj")
    return {"season": 2026, "week": week, "captured_at": captured.isoformat(), "my_team_id": 1,
            "teams": teams, "schedule": sched,
            "settings": {"reg_season_weeks": 14, "playoff_team_count": 6,
                         "playoff_weeks": [15, 16, 17], "trade_deadline": deadline},
            "free_agents": {"fetched_at": captured.isoformat(), "capped": False, "rows": pool}}


def score(cap, move, n=2000, seed=1, now=NOW, byes=BYES):
    return ros.score_move(cap, move, byes=byes, n_draws=n, seed=seed, now=now)


def move_add(add="Waiver Star", drop="WR deep t1"):
    return ros.Move(give=(drop,), get=(add,))


# --- seeding on a real schedule ------------------------------------------------------------

def _pairs(reg=14):
    return [(w, a, b) for w, ps in enumerate(draft_season._round_robin(12, reg), start=1)
            for a, b in ps]


def test_seeding_orders_the_table_exactly_as_the_draft_simulators_seed_table():
    pts = np.random.default_rng(3).normal(100, 20, (400, 17, 12))
    want_w, want_s = draft_season.seed_table(pts)
    got_w, got_s = ros.seed_standings(pts, _pairs(), wins0=np.zeros(12), points0=np.zeros(12),
                                      reg_weeks=14)
    assert np.array_equal(got_w, want_w)
    assert np.array_equal(got_s, want_s)


def test_seeding_starts_from_the_record_a_tie_is_half_a_win_and_points_break_ties():
    pts = np.full((1, 3, 4), 100.0)               # every remaining game is a tie
    wins0 = np.array([1.0, 1.5, 1.0, 0.0])
    pts0 = np.array([50.0, 10.0, 70.0, 0.0])
    sched = [(1, 0, 1), (1, 2, 3), (1, 1, 0), (1, 3, 2)]   # each game listed from both sides
    wins, seeds = ros.seed_standings(pts, sched, wins0=wins0, points0=pts0, reg_weeks=1,
                                     playoff_teams=4)
    assert wins[0].tolist() == [1.5, 2.0, 1.5, 0.5]
    # team 1 leads on wins; 0 and 2 tie on 1.5, and 2's points-for (70 + 100) beats 0's (50 + 100)
    assert seeds[0].tolist() == [1, 2, 0, 3]


# --- the weekly mean -----------------------------------------------------------------------

def test_a_weeks_mean_is_the_captured_projection_else_the_flat_one_zeroed_on_the_bye():
    p = _player("A", "WR", 10.0, team="DAL", wp={"6": 13.0}, avg=9.0)
    mu, flat = ros.weekly_means(p, range(5, 10), BYES)
    assert mu.tolist() == [10.0, 13.0, 0.0, 10.0, 10.0]      # DAL's bye is week 7
    assert flat == 3                                          # weeks 5, 8, 9


def test_with_no_current_week_projection_the_seasons_per_game_average_is_the_flat_mean():
    p = _player("A", "WR", 0.0, team="PHI", avg=9.0)
    p["espn_week_proj"] = None
    mu, _ = ros.weekly_means(p, range(5, 8), BYES)
    assert mu.tolist() == [9.0, 9.0, 9.0]


def test_a_team_with_no_known_bye_is_not_zeroed():
    mu, _ = ros.weekly_means(_player("A", "WR", 10.0, team="XXX"), range(5, 8), BYES)
    assert mu.tolist() == [10.0] * 3


# --- scoring a Move ------------------------------------------------------------------------

def test_planted_gap_a_thirty_point_back_for_a_bench_back_is_seen_and_a_twin_is_not():
    cap = make_capture()
    big = score(cap, move_add(), n=4000)
    assert big.playoffs.delta > 2 * big.playoffs.se and big.playoffs.delta > 0
    assert big.title.delta > 0 and not big.indistinguishable["playoffs"]
    twin = score(cap, move_add("Waiver Twin"), n=4000)
    assert twin.indistinguishable["playoffs"]


def test_an_add_that_changes_nothing_in_the_lineup_is_exactly_zero_with_no_error():
    got = score(make_capture(), move_add("Waiver Dud"))
    assert (got.playoffs.delta, got.playoffs.se) == (0.0, 0.0)
    assert got.indistinguishable == {"playoffs": True, "bye": True, "title": True}


def test_a_trade_reports_the_counterpartys_change_and_an_add_has_none():
    cap = make_capture()
    got = score(cap, ros.Move(give=("RB0 t1",), get=("QB0 t12",)), n=3000, seed=2)
    assert got.counterparty is not None
    assert got.counterparty.name == "Team 12"
    assert got.counterparty.playoffs.se > 0
    assert score(cap, move_add(), n=500).counterparty is None


def test_a_trade_for_players_on_two_teams_is_refused():
    with pytest.raises(ros.MoveError, match="one team"):
        score(make_capture(), ros.Move(give=("RB0 t1",), get=("QB0 t12", "QB0 t11")), n=100)


def test_an_unknown_player_is_named():
    with pytest.raises(ros.UnknownPlayer) as e:
        score(make_capture(), ros.Move(give=("RB bench t1", "Nobody Here"),
                                       get=("Waiver Star",)), n=100)
    assert "Nobody Here" in str(e.value)


def test_the_same_seed_gives_the_same_answer():
    assert score(make_capture(), move_add(), n=800, seed=5) == \
        score(make_capture(), move_add(), n=800, seed=5)


def test_twenty_thousand_draws_run_in_under_ten_seconds():
    cap = make_capture(week=1)                      # the longest remaining season
    t = time.perf_counter()
    got = score(cap, move_add(), n=20_000)
    assert time.perf_counter() - t < 10
    assert got.n_draws == 20_000


# --- warnings ------------------------------------------------------------------------------

def test_a_week_old_capture_is_stale_and_a_passed_trade_deadline_is_flagged_for_trades_only():
    cap = make_capture(deadline="2026-11-01T00:00:00+00:00")
    later = NOW + timedelta(days=8)
    trade = score(cap, ros.Move(give=("RB0 t1",), get=("QB0 t12",)), n=200, now=later)
    text = " ".join(trade.warnings)
    assert "stale" in text and "trade deadline" in text
    assert "trade deadline" not in " ".join(score(cap, move_add(), n=200, now=later).warnings)
    fresh = score(make_capture(), move_add(), n=200)
    assert not [w for w in fresh.warnings if "stale" in w or "deadline" in w]


def test_without_bye_weeks_the_result_says_the_flat_projection_is_not_zeroed():
    got = score(make_capture(), move_add(), n=200, byes=None)
    assert any("bye" in w for w in got.warnings)


# --- the command line ----------------------------------------------------------------------

def write(tmp_path, week=5):
    d = tmp_path / "league" / "2026"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"wk{week:02d}.json"
    p.write_text(json.dumps(make_capture(week=week)))
    return p


def test_the_cli_prints_the_verdict_and_the_assumptions(tmp_path, capsys):
    p = write(tmp_path)
    code = ros.main(["--capture", str(p), "--add", "Waiver Star", "--drop", "WR deep t1",
                     "--draws", "1500", "--no-byes"])
    out = capsys.readouterr().out
    assert code == 0
    assert "flatters weak managers" in out and "flat projection" in out
    assert "P(playoffs)" in out and "P(bye)" in out and "P(title)" in out
    twin = ros.main(["--capture", str(p), "--add", "Waiver Dud", "--drop", "WR deep t1",
                     "--draws", "500", "--no-byes"])
    assert twin == 0 and "can't tell apart" in capsys.readouterr().out


def test_the_default_capture_is_the_newest_week_of_the_newest_season(tmp_path):
    write(tmp_path, week=5)
    newest = write(tmp_path, week=10)
    (tmp_path / "league" / "2025").mkdir()
    (tmp_path / "league" / "2025" / "wk17.json").write_text("{}")
    assert ros.newest_capture(tmp_path / "league") == newest


def test_a_missing_default_capture_is_a_sentence_and_exit_zero(tmp_path, capsys):
    code = ros.main(["--league-dir", str(tmp_path / "none"), "--add", "A", "--drop", "B"])
    assert code == 0 and "no league capture" in capsys.readouterr().out


def test_an_unknown_player_on_the_command_line_is_named_and_fails(tmp_path, capsys):
    p = write(tmp_path)
    code = ros.main(["--capture", str(p), "--add", "Ghost Back", "--drop", "WR deep t1",
                     "--no-byes"])
    assert code == 1 and "Ghost Back" in capsys.readouterr().err


def test_a_named_capture_that_is_not_there_fails(tmp_path, capsys):
    code = ros.main(["--capture", str(tmp_path / "nope.json"), "--add", "A", "--drop", "B"])
    assert code == 1 and "nope.json" in capsys.readouterr().err


def test_a_trade_is_given_as_two_lists(tmp_path, capsys):
    p = write(tmp_path)
    code = ros.main(["--capture", str(p), "--trade", "RB0 t1,WR0 t1", "QB0 t12",
                     "--draws", "800", "--no-byes"])
    assert code == 0 and "Team 12" in capsys.readouterr().out


# --- what is refused -----------------------------------------------------------------------

def test_a_bracket_the_draft_simulator_does_not_play_is_refused():
    cap = make_capture()
    cap["settings"]["playoff_team_count"] = 4
    with pytest.raises(ros.MoveError, match="six-team"):
        score(cap, move_add(), n=10)


def test_a_capture_past_the_regular_season_is_refused():
    cap = make_capture()
    cap["week"] = 15
    with pytest.raises(ros.MoveError, match="past the regular season"):
        score(cap, move_add(), n=10)


def test_a_stray_file_under_the_league_directory_is_skipped(tmp_path):
    write(tmp_path, week=5)
    (tmp_path / "league" / "notes").mkdir()
    (tmp_path / "league" / "notes" / "wkfoo.json").write_text("{}")
    got = ros.newest_capture(tmp_path / "league")
    assert got is not None and got.name == "wk05.json"


def test_a_move_must_name_players_on_both_sides(tmp_path):
    p = write(tmp_path)
    with pytest.raises(SystemExit):
        ros.main(["--capture", str(p), "--trade", "A", ","])
    with pytest.raises(SystemExit):
        ros.main(["--capture", str(p), "--add", "Waiver Star"])
    with pytest.raises(SystemExit):
        ros.main(["--capture", str(p), "--trade", "A", "B", "--add", "C"])


def test_a_capture_the_contract_refuses_is_a_sentence_and_exit_one(tmp_path, capsys):
    p = write(tmp_path)
    cap = json.loads(p.read_text())
    cap["teams"] = []
    p.write_text(json.dumps(cap))
    assert ros.main(["--capture", str(p), "--add", "A", "--drop", "B"]) == 1
    assert "not a usable capture" in capsys.readouterr().err


def test_an_unreadable_schedule_degrades_to_unzeroed_byes_and_says_so(tmp_path, capsys,
                                                                       monkeypatch):
    def refuse(season):
        raise OSError("no schedule cached")
    monkeypatch.setattr(ros.draft_season, "bye_weeks", refuse)
    p = write(tmp_path)
    code = ros.main(["--capture", str(p), "--add", "Waiver Star", "--drop", "WR deep t1",
                     "--draws", "300"])
    out = capsys.readouterr()
    assert code == 0 and "bye weeks unavailable" in out.err and "bye weeks could not" in out.out


def test_a_move_the_players_cannot_make_is_a_sentence_and_exit_one(tmp_path, capsys):
    p = write(tmp_path)
    code = ros.main(["--capture", str(p), "--trade", "RB0 t1", "QB0 t12,QB0 t11", "--no-byes"])
    assert code == 1 and "one team" in capsys.readouterr().err


def test_with_no_capture_named_the_newest_one_is_scored(tmp_path, capsys):
    write(tmp_path, week=5)
    write(tmp_path, week=6)
    code = ros.main(["--league-dir", str(tmp_path / "league"), "--add", "Waiver Star",
                     "--drop", "WR deep t1", "--draws", "300", "--no-byes"])
    assert code == 0 and "wk06.json" in capsys.readouterr().out


# --- the capture's edges -------------------------------------------------------------------

def test_a_schedule_pair_outside_the_games_left_changes_nothing():
    cap = make_capture(week=5)
    base = score(cap, move_add(), n=500)
    cap["schedule"] += [{"week": 3, "team_id": 1, "opponent_id": 2},
                        {"week": 3, "team_id": 2, "opponent_id": 1},
                        {"week": 15, "team_id": 3, "opponent_id": 4}]
    assert score(cap, move_add(), n=500) == base


def test_playoff_weeks_that_do_not_follow_the_regular_season_are_refused():
    cap = make_capture()
    cap["settings"]["playoff_weeks"] = [16, 17, 18]
    with pytest.raises(ros.MoveError, match="playoff weeks"):
        score(cap, move_add(), n=10)


def test_naive_timestamps_are_read_as_utc():
    cap = make_capture(deadline="2026-11-01T00:00:00")
    cap["captured_at"] = "2026-11-01T00:00:00"
    got = score(cap, ros.Move(give=("RB0 t1",), get=("QB0 t12",)), n=100,
                now=NOW + timedelta(days=8))
    text = " ".join(got.warnings)
    assert "stale" in text and "trade deadline" in text
