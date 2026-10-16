"""The lineup each source would set, and what it scored (#492).

The fixture is built so the three sources agree on everyone except the flex slot, and each
picks a different flex: `mu` the third back, ECR the fourth receiver, ESPN's weekly the
second tight end. A journal that cannot tell the sources apart records one lineup three times.
"""
import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from hub.fetch import consensus
from hub.names import player_key
from hub.season import lineup_journal as J

SEASON, WEEK = 2026, 7
FIRST = date(2026, 10, 25)
EARLY = datetime(2026, 10, 20, 12, tzinfo=UTC)      # before the first game day, Eastern
LATE = datetime(2026, 10, 25, 18, tzinfo=UTC)       # after the deadline

# name, pos, mu, ecr, espn weekly
PLAYERS = [
    ("Qa One", "QB", 20.0, 5.0, 20.0), ("Qb Two", "QB", 8.0, 150.0, 5.0),
    ("Ra One", "RB", 18.0, 3.0, 18.0), ("Rb Two", "RB", 17.0, 4.0, 17.0),
    ("Rc Three", "RB", 12.0, 80.0, 6.0), ("Rd Four", "RB", 2.0, 200.0, 1.0),
    ("Wa One", "WR", 16.0, 6.0, 16.0), ("Wb Two", "WR", 15.0, 7.0, 15.0),
    ("Wc Three", "WR", 14.0, 8.0, 14.0), ("Wd Four", "WR", 10.0, 40.0, 7.0),
    ("We Five", "WR", 1.0, 210.0, 1.0),
    ("Ta One", "TE", 13.0, 9.0, 13.0), ("Tb Two", "TE", 5.0, 90.0, 14.5),
]


def _roster(espn: bool = True, week: int | None = WEEK) -> pl.DataFrame:
    df = pl.DataFrame({
        "player": [p[0] for p in PLAYERS], "key": [player_key(p[0]) for p in PLAYERS],
        "pos": [p[1] for p in PLAYERS], "nfl_team": ["XXX"] * len(PLAYERS),
        "mu": [p[2] for p in PLAYERS], "sd": [5.0] * len(PLAYERS),
        "projected": [True] * len(PLAYERS), "can_start": [True] * len(PLAYERS),
        "available": [True] * len(PLAYERS), "starting": [False] * len(PLAYERS),
        "espn_week": [week] * len(PLAYERS),
        "espn_week_proj": [p[4] if espn else None for p in PLAYERS]},
        schema_overrides={"espn_week": pl.Int64, "espn_week_proj": pl.Float64})
    return df


def _capture(base, when=EARLY, week=WEEK, drop=(), by_mu=False):
    rows = pl.DataFrame({"player": [p[0] for p in PLAYERS if p[0] not in drop],
                         "pos": [p[1] for p in PLAYERS if p[0] not in drop],
                         "team": ["XXX"] * (len(PLAYERS) - len(drop)),
                         "ecr": [(100.0 - p[2]) if by_mu else p[3]
                                 for p in PLAYERS if p[0] not in drop]})
    doc = consensus.build(SEASON, week, rows, date(2026, 10, 24), FIRST, when, "d")
    # Bypass the write-side refusal for the late one: the journal must exclude it at the READ.
    from hub import atomic
    path = consensus.capture_path(SEASON, week, when, base)
    atomic.write_text(path, json.dumps(doc, sort_keys=True) + "\n")


def _doc(path):
    assert path is not None, "nothing was journalled"
    return json.loads(path.read_text())


def test_the_three_sources_set_three_different_lineups_when_they_disagree(tmp_path):
    _capture(tmp_path / "cons")
    path, _ = J.record(_roster(), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "cons", now=EARLY)
    doc = _doc(path)
    flex = {s: next(n for n in doc["sources"][s]["lineup"]
                    if n in ("Rc Three", "Wd Four", "Tb Two")) for s in J.SOURCES}
    assert flex == {"mu": "Rc Three", "ecr": "Wd Four", "espn": "Tb Two"}
    lineups = [tuple(doc["sources"][s]["lineup"]) for s in J.SOURCES]
    assert len(set(lineups)) == 3
    assert doc["disagree"] is True and doc["scored"] is False


def test_sources_that_agree_are_not_marked_as_disagreeing(tmp_path):
    # One ordering for all three: ESPN's weekly equals mu, and the capture ranks by mu.
    df = _roster().with_columns(pl.col("mu").alias("espn_week_proj"))
    _capture(tmp_path / "cons", by_mu=True)
    path, _ = J.record(df, season=SEASON, base=tmp_path / "j", captures=tmp_path / "cons")
    doc = _doc(path)
    assert doc["sources"]["mu"]["lineup"] == doc["sources"]["espn"]["lineup"]
    assert doc["sources"]["mu"]["lineup"] == doc["sources"]["ecr"]["lineup"]
    assert doc["disagree"] is False


def test_missing_ecr_is_null_with_a_reason_and_the_others_are_still_journalled(tmp_path):
    path, _ = J.record(_roster(), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "no-captures")
    src = _doc(path)["sources"]
    assert src["ecr"]["lineup"] is None
    assert "no pre-kickoff consensus capture for week 7" in src["ecr"]["reason"]
    assert src["mu"]["lineup"] and src["espn"]["lineup"]


def test_a_capture_taken_after_the_deadline_is_not_read(tmp_path):
    _capture(tmp_path / "cons", when=LATE)
    ranks, reason = J.ecr_ranks(SEASON, WEEK, tmp_path / "cons")
    assert ranks == {} and reason is not None and "all at or after the deadline" in reason


def test_missing_espn_weekly_is_null_with_a_reason(tmp_path):
    _capture(tmp_path / "cons")
    path, _ = J.record(_roster(espn=False), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "cons")
    src = _doc(path)["sources"]
    assert src["espn"]["lineup"] is None and "no weekly projection" in src["espn"]["reason"]
    assert src["mu"]["lineup"] and src["ecr"]["lineup"]


def test_a_roster_frame_from_before_the_weekly_columns_still_journals(tmp_path):
    old = _roster().drop("espn_week", "espn_week_proj")
    path, _ = J.record(old, season=SEASON, week=WEEK, base=tmp_path / "j",
                          captures=tmp_path / "none")
    src = _doc(path)["sources"]
    assert "predates" in src["espn"]["reason"] and src["mu"]["lineup"]


def test_too_few_ranked_players_to_fill_a_lineup_is_null_with_a_reason(tmp_path):
    _capture(tmp_path / "cons", drop=("Qa One", "Qb Two"))      # no quarterback ranked
    path, _ = J.record(_roster(), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "cons")
    ecr = _doc(path)["sources"]["ecr"]
    assert ecr["lineup"] is None and "too few" in ecr["reason"]


def test_no_scoring_period_journals_nothing(tmp_path):
    got, note = J.record(_roster(week=None), season=SEASON, base=tmp_path / "j")
    assert got is None and "no scoring period" in note
    assert not (tmp_path / "j").exists()


def test_recording_a_week_twice_changes_nothing(tmp_path):
    _capture(tmp_path / "cons")
    kw = {"season": SEASON, "base": tmp_path / "j", "captures": tmp_path / "cons"}
    J.record(_roster(), now=EARLY, **kw)
    first = J.journal_path(SEASON, WEEK, tmp_path / "j").read_text()
    changed = _roster().with_columns((pl.col("mu") * -1).alias("mu"))
    _, note = J.record(changed, **kw)
    assert J.journal_path(SEASON, WEEK, tmp_path / "j").read_text() == first and "already journalled" in note


def test_settle_writes_each_lineups_realised_points_once(tmp_path):
    _capture(tmp_path / "cons")
    J.record(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "cons")
    pts = {player_key(p[0]): p[2] for p in PLAYERS}               # everyone scores their mu
    pts[player_key("Tb Two")] = 30.0
    assert J.settle(SEASON, WEEK, pts, tmp_path / "j").endswith("scored")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    base = 20 + 18 + 17 + 16 + 15 + 14 + 13
    assert doc["scored"] is True
    assert doc["realised"]["mu"]["points"] == base + 12
    assert doc["realised"]["ecr"]["points"] == base + 10
    assert doc["realised"]["espn"]["points"] == base + 30
    again = J.settle(SEASON, WEEK, dict.fromkeys(pts, 0.0), tmp_path / "j")
    assert "already scored" in again
    assert json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text()) == doc


def test_settle_with_no_points_leaves_the_week_unscored(tmp_path):
    J.record(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    assert "no realised points" in J.settle(SEASON, WEEK, {}, tmp_path / "j")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    assert doc["scored"] is False and doc["realised"]["mu"] is None


def test_a_null_lineup_has_null_realised_points(tmp_path):
    J.record(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    J.settle(SEASON, WEEK, {player_key("Qa One"): 1.0}, tmp_path / "j")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    assert doc["realised"]["ecr"] is None and doc["realised"]["mu"]["n_no_row"] == 7


def test_read_returns_the_weeks_in_order(tmp_path):
    for wk in (8, 7):
        J.record(_roster(week=wk), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    assert [d["week"] for d in J.read(SEASON, tmp_path / "j")] == [7, 8]


@pytest.mark.parametrize("bad", ["", "{not json"])
def test_an_unreadable_capture_degrades_to_a_reason(tmp_path, bad):
    p = consensus.capture_path(SEASON, WEEK, EARLY, tmp_path / "cons")
    p.parent.mkdir(parents=True)
    p.write_text(bad)
    ranks, reason = J.ecr_ranks(SEASON, WEEK, tmp_path / "cons")
    assert ranks == {} and reason is not None and "unreadable" in reason
