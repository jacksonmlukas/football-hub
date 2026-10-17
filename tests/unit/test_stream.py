"""K / D/ST streaming by implied totals (#494): a heuristic, shown with its basis."""
import json
from datetime import datetime

import polars as pl

from hub.season import stream

AT = datetime(2026, 10, 18, 10, 0)           # Sunday morning, naive UTC
EARLY = datetime(2026, 10, 18, 17, 0)
LATE = datetime(2026, 10, 18, 20, 0)
CAP = datetime(2026, 10, 17, 12, 0)
IN_PLAY = datetime(2026, 10, 18, 17, 30)


def _games():
    rows = [
        # game_id, home, away, kickoff, price_source, priced_at
        ("g1", "AAA", "BBB", LATE, "live", CAP),
        ("g2", "CCC", "DDD", LATE, "live", CAP),
        ("g3", "EEE", "FFF", LATE, "live", CAP),
        ("g4", "GGG", "HHH", LATE, "schedule", None),            # no capture priced it
        ("g5", "III", "JJJ", datetime(2026, 10, 18, 9, 0), "live", CAP),   # already started
        ("g6", "KKK", "LLL", LATE, "live", CAP),                 # capture carries no total
        ("g7", "MMM", "NNN", EARLY, "live", IN_PLAY),            # only capture is in play
    ]
    return pl.DataFrame(
        {"game_id": [r[0] for r in rows], "week": [5] * len(rows),
         "home_team": [r[1] for r in rows], "away_team": [r[2] for r in rows],
         "kickoff": [r[3] for r in rows], "result": [None] * len(rows),
         "price_source": [r[4] for r in rows], "priced_at": [r[5] for r in rows]},
        schema={"game_id": pl.Utf8, "week": pl.Int32, "home_team": pl.Utf8,
                "away_team": pl.Utf8, "kickoff": pl.Datetime, "result": pl.Float64,
                "price_source": pl.Utf8, "priced_at": pl.Datetime})


SEEN = {
    ("g1", CAP): {"close_spread": 3.0, "close_total": 44.0},
    ("g2", CAP): {"close_spread": -7.0, "close_total": 50.0},
    ("g3", CAP): {"close_spread": 0.0, "close_total": 40.0},
    ("g5", CAP): {"close_spread": 1.0, "close_total": 41.0},
    ("g6", CAP): {"close_spread": 1.0, "close_total": None},
    ("g7", IN_PLAY): {"close_spread": 9.0, "close_total": 60.0},
}


def _roster():
    return pl.DataFrame(
        {"player": ["Bills D/ST", "Kicker Kay", "Bye Defence", "Mystery K"],
         "pos": ["D/ST", "K", "D/ST", "K"],
         "nfl_team": ["BBB", "CCC", "ZZZ", "FA"]})


def _built():
    return stream.build(_games(), _roster(), 5, frozenset({"ZZZ"}), SEEN, AT)


def _order(rows):
    return [(r["team"], r["rank"]) for r in rows if r["rank"] is not None]


def test_implied_totals_split_the_total_by_the_home_spread():
    assert stream.implied(3.0, 44.0) == (23.5, 20.5)
    assert stream.implied(-7.0, 50.0) == (21.5, 28.5)


def test_planted_slate_gives_the_expected_dst_order():
    """Lowest opponent implied total first; a tie shares a rank and the next is skipped."""
    assert _order(_built()["dst"]) == [
        ("EEE", 1), ("FFF", 1), ("AAA", 3), ("DDD", 4), ("BBB", 5), ("CCC", 6)]


def test_planted_slate_gives_the_expected_k_order():
    """Highest own implied total first -- the reverse sense of the D/ST rule."""
    assert _order(_built()["k"]) == [
        ("DDD", 1), ("AAA", 2), ("CCC", 3), ("BBB", 4), ("EEE", 5), ("FFF", 5)]


def test_each_row_carries_the_inputs_so_the_page_can_show_the_working():
    row = {r["team"]: r for r in _built()["dst"]}["AAA"]
    assert (row["opp"], row["home"], row["spread"], row["total"]) == ("BBB", True, 3.0, 44.0)
    assert (row["own_implied"], row["opp_implied"]) == (23.5, 20.5)
    assert row["priced_at"] == CAP.isoformat()


def test_a_team_on_bye_is_excluded_and_says_so():
    rows = {r["team"]: r for r in _built()["dst"]}
    assert rows["ZZZ"]["rank"] is None and rows["ZZZ"]["missing"] == "bye week"
    assert rows["ZZZ"]["rostered"] == ["Bye Defence"]
    assert all(r["rank"] is None for r in _built()["k"] if r["team"] == "ZZZ")


def test_a_game_with_no_pre_kickoff_price_is_null_with_a_reason():
    rows = {r["team"]: r for r in _built()["dst"]}
    for t in ("GGG", "HHH"):
        assert rows[t]["rank"] is None and rows[t]["opp_implied"] is None
        assert "no capture priced this game" in rows[t]["missing"]
    # the only capture is after kickoff: never a market price
    assert "after kickoff" in rows["MMM"]["missing"] and rows["MMM"]["rank"] is None


def test_started_games_and_missing_totals_are_not_ranked():
    rows = {r["team"]: r for r in _built()["k"]}
    assert rows["III"]["missing"] == "the game has already kicked off"
    assert rows["KKK"]["missing"] == "the capture carried no total"
    assert rows["KKK"]["rank"] is None and rows["III"]["rank"] is None


def test_unranked_rows_come_last():
    ranks = [r["rank"] for r in _built()["dst"]]
    first_null = ranks.index(None)
    assert all(r is None for r in ranks[first_null:])


def test_rostered_units_are_flagged_beside_the_rest_and_unplaceable_ones_named():
    s = _built()
    assert {r["team"]: r["rostered"] for r in s["dst"]}["BBB"] == ["Bills D/ST"]
    assert {r["team"]: r["rostered"] for r in s["k"]}["CCC"] == ["Kicker Kay"]
    assert s["unplaced"] == [{"player": "Mystery K", "pos": "K",
                              "reason": "his team is not in this week's schedule"}]


def test_it_is_labelled_a_heuristic_with_formula_and_unknown_availability():
    s = _built()
    assert s["label"] == "heuristic" and "not a measured edge" in s["caveat"]
    assert s["basis"]["dst"] == "lowest opponent implied total first"
    assert s["basis"]["k"] == "highest own implied total first"
    assert "(total + spread) / 2" in s["formula"]
    assert s["available_known"] is False and "#493" in s["availability_reason"]
    assert "dome and weather are not used" in s["not_used"]


def test_unavailable_is_labelled_and_empty():
    s = stream.unavailable("no schedule")
    assert s["label"] == "heuristic" and s["dst"] == [] and s["k"] == []
    assert s["missing"] == "no schedule"


def test_for_roster_degrades_when_the_schedule_cannot_be_read(monkeypatch):
    from hub import schedule

    def boom(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(schedule, "priced_games", boom)
    s = stream.for_roster(_roster(), season=2026, at=AT)
    assert "OSError" in s["missing"] and s["dst"] == []


def test_for_roster_ranks_the_open_week(monkeypatch):
    from hub import market_row, schedule
    monkeypatch.setattr(schedule, "priced_games", lambda *a, **k: _games())
    monkeypatch.setattr(market_row, "captures", lambda season, base: SEEN)
    s = stream.for_roster(_roster(), season=2026, at=AT)
    assert s["week"] == 5 and _order(s["dst"])[0] == ("EEE", 1)


def test_for_roster_with_no_open_week_says_so(monkeypatch):
    from hub import schedule
    monkeypatch.setattr(schedule, "priced_games", lambda *a, **k: _games().head(0))
    assert stream.for_roster(_roster(), at=AT)["missing"] == "no week is open in the schedule"


def test_record_writes_once_first_write_wins(tmp_path):
    s = _built()
    path, note = stream.record(s, 2026, base=tmp_path, now=AT)
    assert path is not None and "journalled" in note
    first = json.loads(path.read_text())
    assert first["week"] == 5 and first["label"] == "heuristic" and first["scored"] is False
    assert first["dst"][0]["team"] == "EEE"
    again, note2 = stream.record({**s, "dst": []}, 2026, base=tmp_path, now=AT)
    assert again == path and "already journalled" in note2
    assert json.loads(path.read_text())["dst"][0]["team"] == "EEE"
    assert [d["week"] for d in stream.read(2026, tmp_path)] == [5]


def test_record_refuses_a_week_with_no_ranking(tmp_path):
    path, note = stream.record(stream.unavailable("offline"), 2026, base=tmp_path)
    assert path is None and "offline" in note and stream.read(2026, tmp_path) == []


def test_render_leads_with_the_label_and_shows_rostered_units():
    text = stream.render(_built(), top=2)
    assert text.splitlines()[0].startswith("  [heuristic]")
    assert "<- Bills D/ST" in text and "<- Bye Defence" in text and "bye week" in text
    assert "no ranking" in stream.render(stream.unavailable("offline"))
