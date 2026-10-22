"""The lineup each source would set, and what it scored (#492).

The fixture is built so the three sources agree on everyone except the flex slot, and each
picks a different flex: `mu` the third back, ECR the fourth receiver, ESPN's weekly the
second tight end. A journal that cannot tell the sources apart records one lineup three times.
"""
import json
from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from hub import atomic, publish
from hub.fetch import consensus
from hub.names import player_key
from hub.season import lineup_journal as J
from hub.season import lineup_rule
from hub.season.roster import lock

_REAL_NOW = J._now          # the autouse fixture below replaces it for every test
SEASON, WEEK = 2026, 7
FIRST = date(2026, 10, 25)
EARLY = datetime(2026, 10, 10, 12, tzinfo=UTC)      # before every week used here starts
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


def _schedule(weeks=(6, 7, 8), final=True, open_games=0) -> pl.DataFrame:
    """Two regular-season games a week, the first on `FIRST` for week 7 (a week earlier for 6).
    `open_games` of week 7's games have no final score yet."""
    rows = []
    for wk in weeks:
        day = date(2026, 10, 25) + timedelta(days=7 * (wk - 7))
        for i in range(2):
            done = final and not (wk == 7 and i < open_games)
            rows.append({"game_id": f"{wk}-{i}", "season": SEASON, "week": wk,
                         "game_type": "REG", "home_team": "A", "away_team": "B",
                         "gameday": day.isoformat(),
                         "home_score": 20 if done else None,
                         "away_score": 17 if done else None})
    return pl.DataFrame(rows, schema_overrides={"home_score": pl.Int64, "away_score": pl.Int64})


@pytest.fixture(autouse=True)
def _pinned_world(monkeypatch):
    """The CLI's two seams, pinned: a complete schedule and a clock before every week here."""
    monkeypatch.setattr(J, "_schedule", lambda season: _schedule())
    monkeypatch.setattr(J, "_now", lambda: EARLY)


def _rec(df, **kw):
    """`record` with the clock pinned before week 6's first game day, unless a test says."""
    kw.setdefault("now", EARLY)
    kw.setdefault("schedule", _schedule())
    return J.record(df, **kw)


def _settle(season, week, points, base, **kw):
    kw.setdefault("schedule", _schedule())
    return J.settle(season, week, points, base, **kw)


def _doc(path):
    assert path is not None, "nothing was journalled"
    return json.loads(path.read_text())


def test_the_three_sources_set_three_different_lineups_when_they_disagree(tmp_path):
    _capture(tmp_path / "cons")
    path, _ = _rec(_roster(), season=SEASON, base=tmp_path / "j",
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
    path, _ = _rec(df, season=SEASON, base=tmp_path / "j", captures=tmp_path / "cons")
    doc = _doc(path)
    assert doc["sources"]["mu"]["lineup"] == doc["sources"]["espn"]["lineup"]
    assert doc["sources"]["mu"]["lineup"] == doc["sources"]["ecr"]["lineup"]
    assert doc["disagree"] is False


def test_missing_ecr_is_null_with_a_reason_and_the_others_are_still_journalled(tmp_path):
    path, _ = _rec(_roster(), season=SEASON, base=tmp_path / "j",
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
    path, _ = _rec(_roster(espn=False), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "cons")
    src = _doc(path)["sources"]
    assert src["espn"]["lineup"] is None and "no weekly projection" in src["espn"]["reason"]
    assert src["mu"]["lineup"] and src["ecr"]["lineup"]


def test_a_roster_frame_from_before_the_weekly_columns_still_journals(tmp_path):
    old = _roster().drop("espn_week", "espn_week_proj")
    path, _ = _rec(old, season=SEASON, week=WEEK, base=tmp_path / "j",
                          captures=tmp_path / "none")
    src = _doc(path)["sources"]
    assert "predates" in src["espn"]["reason"] and src["mu"]["lineup"]


def test_too_few_ranked_players_to_fill_a_lineup_is_null_with_a_reason(tmp_path):
    _capture(tmp_path / "cons", drop=("Qa One", "Qb Two"))      # no quarterback ranked
    path, _ = _rec(_roster(), season=SEASON, base=tmp_path / "j",
                       captures=tmp_path / "cons")
    ecr = _doc(path)["sources"]["ecr"]
    assert ecr["lineup"] is None and "too few" in ecr["reason"]


def test_no_scoring_period_journals_nothing(tmp_path):
    got, note = _rec(_roster(week=None), season=SEASON, base=tmp_path / "j")
    assert got is None and "no scoring period" in note
    assert not (tmp_path / "j").exists()


def test_recording_a_week_twice_changes_nothing(tmp_path):
    _capture(tmp_path / "cons")
    kw = {"season": SEASON, "base": tmp_path / "j", "captures": tmp_path / "cons"}
    _rec(_roster(), now=EARLY, **kw)
    first = J.journal_path(SEASON, WEEK, tmp_path / "j").read_text()
    changed = _roster().with_columns((pl.col("mu") * -1).alias("mu"))
    _, note = _rec(changed, **kw)
    assert J.journal_path(SEASON, WEEK, tmp_path / "j").read_text() == first and "already journalled" in note


def test_settle_writes_each_lineups_realised_points_once(tmp_path):
    _capture(tmp_path / "cons")
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "cons")
    pts = {player_key(p[0]): p[2] for p in PLAYERS}               # everyone scores their mu
    pts[player_key("Tb Two")] = 30.0
    assert _settle(SEASON, WEEK, pts, tmp_path / "j").startswith(f"week {WEEK}: scored")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    base = 20 + 18 + 17 + 16 + 15 + 14 + 13
    assert doc["scored"] is True
    assert doc["realised"]["mu"]["points"] == base + 12
    assert doc["realised"]["ecr"]["points"] == base + 10
    assert doc["realised"]["espn"]["points"] == base + 30
    again = _settle(SEASON, WEEK, dict.fromkeys(pts, 0.0), tmp_path / "j")
    assert "already scored" in again
    assert json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text()) == doc


def test_settle_with_no_points_leaves_the_week_unscored(tmp_path):
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    assert "no realised points" in _settle(SEASON, WEEK, {}, tmp_path / "j")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    assert doc["scored"] is False and doc["realised"]["mu"] is None


def test_a_null_lineup_has_null_realised_points(tmp_path):
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    _settle(SEASON, WEEK, {player_key("Qa One"): 1.0}, tmp_path / "j")
    doc = json.loads(J.journal_path(SEASON, WEEK, tmp_path / "j").read_text())
    assert doc["realised"]["ecr"] is None and doc["realised"]["mu"]["n_no_row"] == 7


def test_read_returns_the_weeks_in_order(tmp_path):
    for wk in (8, 7):
        _rec(_roster(week=wk), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    assert [d["week"] for d in J.read(SEASON, tmp_path / "j")] == [7, 8]


@pytest.mark.parametrize("bad", ["", "{not json"])
def test_an_unreadable_capture_degrades_to_a_reason(tmp_path, bad):
    p = consensus.capture_path(SEASON, WEEK, EARLY, tmp_path / "cons")
    p.parent.mkdir(parents=True)
    p.write_text(bad)
    ranks, reason = J.ecr_ranks(SEASON, WEEK, tmp_path / "cons")
    assert ranks == {} and reason is not None and "unreadable" in reason


# --- the CLI -------------------------------------------------------------------

def test_the_cli_records_a_week_and_a_second_run_leaves_it(tmp_path, capsys):
    roster = tmp_path / "roster.parquet"
    _roster().write_parquet(roster)
    args = ["--record", "--roster", str(roster), "--out", str(tmp_path / "j")]
    assert J.main(args) == 0
    assert J.main(args) == 0
    out = capsys.readouterr().out
    assert "journalled" in out and "already journalled" in out
    assert len(J.read(J.SEASON_AHEAD, tmp_path / "j")) == 1


def test_a_record_after_kickoff_is_refused_with_a_reason_and_exit_zero(tmp_path, monkeypatch,
                                                                       capsys):
    """Rule 18: the week's first game day has started, so the roster's `mu` has seen results
    the pre-kickoff ranking has not. Nothing is written; the slate is not hurt."""
    monkeypatch.setattr(J, "_now", lambda: datetime(2026, 10, 25, 18, tzinfo=UTC))
    roster = tmp_path / "roster.parquet"
    _roster().write_parquet(roster)
    assert J.main(["--record", "--roster", str(roster), "--out", str(tmp_path / "j")]) == 0
    assert "has started" in capsys.readouterr().out
    assert J.read(J.SEASON_AHEAD, tmp_path / "j") == []


def test_the_kickoff_boundary_is_midnight_eastern_on_the_first_game_day(tmp_path):
    just_before = datetime(2026, 10, 25, 3, 59, tzinfo=UTC)       # 23:59 EDT the day before
    just_after = datetime(2026, 10, 25, 4, 0, tzinfo=UTC)
    kw = {"season": SEASON, "base": tmp_path / "j", "captures": tmp_path / "x"}
    assert _rec(_roster(), now=just_before, **kw)[0] is not None
    other = tmp_path / "k"
    got, why = _rec(_roster(), now=just_after, **{**kw, "base": other})
    assert got is None and "has started" in why


def test_an_unreadable_schedule_refuses_the_record_rather_than_guessing(tmp_path):
    got, why = J.record(_roster(), season=SEASON, base=tmp_path / "j", now=EARLY,
                        schedule=None, captures=tmp_path / "x")
    assert got is None and "no schedule" in why


def test_an_existing_entry_is_left_even_after_kickoff(tmp_path):
    kw = {"season": SEASON, "base": tmp_path / "j", "captures": tmp_path / "x"}
    _rec(_roster(), **kw)
    got, why = _rec(_roster(), now=datetime(2026, 10, 26, tzinfo=UTC), **kw)
    assert got is not None and "already journalled" in why


def test_a_partially_loaded_week_stays_unscored_with_a_reason(tmp_path):
    """Rule 18: nflverse has points for some of the week but one game is not final. Scoring
    now would freeze the week short, so it is left alone and a later run completes it."""
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    pts = {player_key(p[0]): p[2] for p in PLAYERS}
    why = _settle(SEASON, WEEK, pts, tmp_path / "j", schedule=_schedule(open_games=1))
    assert "1 of 2 games not final" in why
    doc = _doc(J.journal_path(SEASON, WEEK, tmp_path / "j"))
    assert doc["scored"] is False and doc["realised"]["mu"] is None
    assert _settle(SEASON, WEEK, pts, tmp_path / "j").startswith(f"week {WEEK}: scored")


def test_settle_without_a_schedule_leaves_the_week_unscored(tmp_path):
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    why = J.settle(SEASON, WEEK, {"a": 1.0}, tmp_path / "j", schedule=None)
    assert "no schedule" in why
    assert _doc(J.journal_path(SEASON, WEEK, tmp_path / "j"))["scored"] is False


def test_a_week_missing_from_the_schedule_is_refused_and_left_unscored(tmp_path):
    got, why = J.record(_roster(), season=SEASON, week=WEEK, base=tmp_path / "j", now=EARLY,
                        schedule=_schedule(weeks=(6,)), captures=tmp_path / "x")
    assert got is None and "not in the schedule" in why
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    assert "not in the schedule" in J.settle(SEASON, WEEK, {"a": 1.0}, tmp_path / "j",
                                             schedule=_schedule(weeks=(6,)))


def test_settling_a_week_with_no_entry_says_so(tmp_path):
    assert "no journal entry" in _settle(SEASON, WEEK, {"a": 1.0}, tmp_path / "j")


def test_the_real_clock_is_timezone_aware():
    assert _REAL_NOW().tzinfo is not None


def test_the_cli_needs_a_verb():
    with pytest.raises(SystemExit):
        J.main([])


def test_an_ecr_that_ranks_nobody_on_the_roster_is_null_with_a_reason(tmp_path):
    _capture(tmp_path / "cons", drop=tuple(p[0] for p in PLAYERS[1:]))   # one ranked, not a lineup
    got, _ = _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "cons")
    assert _doc(got)["sources"]["ecr"]["lineup"] is None
    only_unmatched = J._best(_roster(), {"zzz": 1.0}, "ECR")
    assert only_unmatched["lineup"] is None and "ranks none" in only_unmatched["reason"]


def test_the_cli_without_a_roster_says_so_and_fails(tmp_path, capsys):
    assert J.main(["--record", "--roster", str(tmp_path / "nope.parquet")]) == 1
    assert "no roster" in capsys.readouterr().err


def test_the_cli_settles_only_weeks_the_roster_has_moved_past(tmp_path, monkeypatch):
    season = J.SEASON_AHEAD
    base = tmp_path / "j"
    for wk in (6, 7):
        _rec(_roster(week=wk), season=season, week=wk, base=base, captures=tmp_path / "x")
    roster = tmp_path / "roster.parquet"
    _roster(week=7).write_parquet(roster)
    monkeypatch.setattr(J, "_realised_points", lambda s, w: {player_key("Qa One"): 20.0})
    assert J.main(["--settle", "--roster", str(roster), "--out", str(base)]) == 0
    scored = {d["week"]: d["scored"] for d in J.read(season, base)}
    assert scored == {6: True, 7: False}, "week 7 is the current week: not final yet"


def test_a_failed_realised_read_leaves_the_week_unscored(tmp_path, monkeypatch, capsys):
    base = tmp_path / "j"
    _rec(_roster(week=6), season=J.SEASON_AHEAD, week=6, base=base, captures=tmp_path / "x")
    roster = tmp_path / "roster.parquet"
    _roster(week=7).write_parquet(roster)

    def boom(s, w):
        raise OSError("offline")
    monkeypatch.setattr(J, "_realised_points", boom)
    assert J.main(["--settle", "--roster", str(roster), "--out", str(base)]) == 0
    assert J.read(J.SEASON_AHEAD, base)[0]["scored"] is False
    assert "not settled" in capsys.readouterr().err


# --- what was actually published (#517) ------------------------------------------------------
#
# The week is the season's switch week, read from the rule rather than written here, so it stays
# a consensus week wherever the boundary sits. Wednesday has no capture at all, so the rule falls
# back to `mu` (flex Rc Three); Friday's lineup capture ranks by ECR (flex Wd Four); Saturday's
# publish sets that. A journal frozen on Wednesday scores Rc Three, whom nobody started.

SW = lineup_rule.ECR_FROM_WEEK[SEASON]
THU = date(2026, 10, 22) + timedelta(days=7 * (SW - 7))      # week 7: Thu 10-22, Sun 10-25
SUN = THU + timedelta(days=3)
WED_AM = datetime.combine(THU - timedelta(days=1), datetime.min.time(), UTC) + timedelta(hours=11)
FRI_AM = WED_AM + timedelta(days=2, hours=2)
SAT_AM = WED_AM + timedelta(days=3, hours=3)                 # 14:00 UTC, the roster's run
SUN_START = consensus.lineup_deadline_for(SUN)               # 00:00 Eastern, Sunday
MU_SET = {"Qa One", "Ra One", "Rb Two", "Wa One", "Wb Two", "Wc Three", "Ta One", "Rc Three"}


def _week_schedule(final: bool = False) -> pl.DataFrame:
    """The switch week: a Thursday game and a Sunday game."""
    return pl.DataFrame(
        [{"game_id": f"{SW}-{i}", "season": SEASON, "week": SW, "game_type": "REG",
          "home_team": "A", "away_team": "B", "gameday": d.isoformat(),
          "home_score": 20 if final else None, "away_score": 17 if final else None}
         for i, d in enumerate((THU, SUN))],
        schema_overrides={"home_score": pl.Int64, "away_score": pl.Int64})


def _pub_roster(**cols) -> pl.DataFrame:
    df = _roster(week=SW).with_columns(
        pl.Series("starting", [p[0] in MU_SET for p in PLAYERS]),
        pl.lit(False).alias("on_bye"), pl.lit("ACTIVE").alias("injury_status"),
        pl.lit("BE").alias("slot"))
    return df.with_columns(*[pl.Series(k, v) for k, v in cols.items()])


def _ecr_rows(players=PLAYERS) -> pl.DataFrame:
    return pl.DataFrame({"player": [p[0] for p in players], "pos": [p[1] for p in players],
                         "team": ["XXX"] * len(players), "ecr": [p[3] for p in players]})


def _friday_lineup_capture(base, rows=None) -> None:
    consensus.write_lineup_capture(consensus.build_lineup(
        SEASON, SW, _ecr_rows() if rows is None else rows, THU + timedelta(days=1),
        THU - timedelta(days=2), THU, SUN, FRI_AM, "d"), base)


@pytest.fixture
def site(monkeypatch, tmp_path):
    """The publisher pointed at a tmp site, journal and capture tree, on the switch week."""
    from hub.season import stream, waiver
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "cons")
    monkeypatch.setattr(J, "ROOT", tmp_path / "j")
    monkeypatch.setattr(publish, "SITE", tmp_path / "site")
    monkeypatch.setattr(J, "_schedule", lambda season: _week_schedule())
    monkeypatch.setattr(publish, "_roster_stream", lambda df: stream.unavailable("stubbed"))
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: waiver.unavailable("stubbed"))
    return tmp_path


def _publish_at(tmp, when, monkeypatch) -> dict:
    monkeypatch.setattr(J, "_now", lambda: when)
    src = tmp / "roster.parquet"
    _pub_roster().write_parquet(src)
    publish.roster(path=src)
    return json.loads((tmp / "site" / "roster.json").read_text())


def _entry() -> dict:
    return _doc(J.journal_path(SEASON, SW))


def test_the_saturday_publish_is_journalled_and_settle_scores_it(site, monkeypatch):
    """The acceptance case: the lineup capture lands after Wednesday, and the journal's
    published lineup is Saturday's published roster -- not the Wednesday rule's."""
    J.record(_pub_roster(), season=SEASON, now=WED_AM, schedule=_week_schedule())
    wednesday = {k: v for k, v in _entry().items() if k != "published"}
    _publish_at(site, WED_AM, monkeypatch)
    assert _entry()["published"]["source"] == "mu"
    _friday_lineup_capture(site / "cons")
    art = _publish_at(site, SAT_AM, monkeypatch)
    doc = _entry()
    pub = doc["published"]
    shown = sorted(r["player"] for r in art["rows"] if r["best_start"])
    assert pub["lineup"] == shown and "Wd Four" in shown and "Rc Three" not in shown
    assert pub["source"] == art["lineup_rule"]["source"] == "ecr"
    assert pub["ecr_week"] == SW and pub["ecr_capture"]["purpose"] == consensus.LINEUP
    assert pub["decided_at"] == SAT_AM.isoformat() and "locked" not in pub
    # The Wednesday comparison is exactly as it was written: the publish touched none of it.
    assert {k: v for k, v in doc.items() if k != "published"} == wednesday
    assert doc["rule"]["source"] == "mu" and "Rc Three" in doc["rule"]["lineup"]
    pts = {player_key(p[0]): p[2] for p in PLAYERS} | {player_key("Wd Four"): 3.0}
    note = J.settle(SEASON, SW, pts, schedule=_week_schedule(final=True))
    assert note == f"week {SW}: scored"
    doc = _entry()
    assert doc["set_scored_from"] == "published"
    assert doc["realised"]["set"]["points"] == round(sum(pts[player_key(n)] for n in shown), 2)
    assert doc["realised"]["set"]["points"] != doc["realised"]["mu"]["points"]


def test_no_published_write_from_the_first_sunday_on(site, monkeypatch):
    """Rule 18: the publish at the start of Sunday, Eastern, finds the decision frozen."""
    _publish_at(site, SAT_AM, monkeypatch)
    before = J.journal_path(SEASON, SW).read_text()
    _friday_lineup_capture(site / "cons")                     # would change the decision
    _publish_at(site, SUN_START, monkeypatch)
    assert J.journal_path(SEASON, SW).read_text() == before
    _publish_at(site, SUN_START - timedelta(minutes=1), monkeypatch)
    assert J.journal_path(SEASON, SW).read_text() != before, "a minute earlier is in time"


def test_record_published_refusals_say_why(tmp_path):
    lk, rule = lineup_rule.decide(_pub_roster(), SEASON, captures=tmp_path / "none")
    kw = {"base": tmp_path / "j", "now": SAT_AM}
    assert "no scoring period" in J.record_published(SEASON, None, lk, rule, **kw)[1]
    assert "no schedule" in J.record_published(SEASON, SW, lk, rule, schedule=None, **kw)[1]
    thursday_only = _week_schedule().filter(pl.col("gameday") == THU.isoformat())
    assert "no Sunday game" in J.record_published(SEASON, SW, lk, rule,
                                                  schedule=thursday_only, **kw)[1]
    got, why = J.record_published(SEASON, SW, lk, rule, schedule=_week_schedule(),
                                  base=tmp_path / "j", now=SUN_START)
    assert got is None and "has started" in why
    assert not (tmp_path / "j").exists()


def test_a_scored_week_keeps_its_published_decision(tmp_path):
    lk, rule = lineup_rule.decide(_pub_roster(), SEASON, captures=tmp_path / "none")
    kw = {"base": tmp_path / "j", "schedule": _week_schedule(), "now": SAT_AM}
    path, _ = J.record_published(SEASON, SW, lk, rule, **kw)
    assert path is not None
    J.settle(SEASON, SW, {player_key("Qa One"): 1.0}, tmp_path / "j",
             schedule=_week_schedule(final=True))
    before = path.read_text()
    _, why = J.record_published(SEASON, SW, lk, rule, **kw)
    assert "already scored" in why and path.read_text() == before


def test_a_publish_before_any_record_is_kept_when_record_adds_the_comparison(tmp_path):
    """A week whose Wednesday `record` did not run: the publish writes its block alone, the
    record then adds the comparison beside it, and settle scores an entry of either shape."""
    lk, rule = lineup_rule.decide(_pub_roster(), SEASON, captures=tmp_path / "none")
    kw = {"schedule": _week_schedule(), "now": WED_AM}
    path, _ = J.record_published(SEASON, SW, lk, rule, base=tmp_path / "j", **kw)
    assert path is not None and "sources" not in _doc(path)
    alone = tmp_path / "alone"
    J.record_published(SEASON, SW, lk, rule, base=alone, **kw)
    J.settle(SEASON, SW, {player_key("Qa One"): 1.0}, alone, schedule=_week_schedule(True))
    scored = _doc(J.journal_path(SEASON, SW, alone))
    assert scored["realised"]["mu"] is None and scored["set_scored_from"] == "published"
    published = _doc(path)["published"]
    J.record(_pub_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "none", **kw)
    doc = _doc(path)
    assert doc["published"] == published and doc["sources"]["mu"]["lineup"]


def test_the_locked_players_travel_with_the_published_decision(tmp_path):
    df = _pub_roster(kicked_off=[p[0] == "Qa One" for p in PLAYERS])
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path / "none")
    assert lk.locked == ("Qa One",)
    path, _ = J.record_published(SEASON, SW, lk, rule, base=tmp_path / "j",
                                 schedule=_week_schedule(), now=SAT_AM)
    assert path is not None and _doc(path)["published"]["locked"] == ["Qa One"]


def test_an_old_format_week_falls_back_to_the_rule_block_and_says_so(tmp_path):
    _rec(_roster(), season=SEASON, base=tmp_path / "j", captures=tmp_path / "x")
    pts = {player_key(p[0]): p[2] for p in PLAYERS}
    note = _settle(SEASON, WEEK, pts, tmp_path / "j")
    assert "scored from the Wednesday `rule` block" in note and "#517" in note
    doc = _doc(J.journal_path(SEASON, WEEK, tmp_path / "j"))
    assert doc["set_scored_from"] == "rule"
    assert doc["realised"]["set"]["points"] == round(
        sum(pts[player_key(n)] for n in doc["rule"]["lineup"]), 2)


def test_a_publish_into_a_scratch_directory_is_not_journalled(site, monkeypatch):
    src = site / "roster.parquet"
    _pub_roster().write_parquet(src)
    monkeypatch.setattr(J, "_now", lambda: SAT_AM)
    publish.roster(out=site / "scratch", path=src)
    assert (site / "scratch" / "roster.json").exists()
    assert not J.journal_path(SEASON, SW).exists()


def test_a_journal_that_cannot_be_written_never_fails_the_publish(site, monkeypatch):
    def boom(*a, **k):
        raise OSError("read-only")
    monkeypatch.setattr(J, "record_published", boom)
    lk, rule = lineup_rule.decide(_pub_roster(), SEASON)
    assert "not journalled (OSError)" in publish._roster_journal(SW, lk, rule)


def test_players_on_bye_are_in_no_sources_candidate_pool(tmp_path):
    """Rule 18: Wd Four is ECR's flex; on bye he is no source's, as he is not `lock`'s."""
    _capture(tmp_path / "cons")
    ecr = J.ecr_ranks(SEASON, WEEK, tmp_path / "cons")
    open_ = J.lineups(_roster(), *ecr, *J.espn_weekly(_roster()))
    assert "Wd Four" in open_["ecr"]["lineup"], "off bye he is ECR's flex"
    df = _roster().with_columns(pl.Series("on_bye", [p[0] == "Wd Four" for p in PLAYERS]))
    got = J.lineups(df, *ecr, *J.espn_weekly(df))
    assert all(v["lineup"] and "Wd Four" not in v["lineup"] for v in got.values())
    assert "Wd Four" not in lock(df, ranks=ecr[0]).best_lineup


def test_a_full_entry_stays_well_under_the_state_cap(tmp_path):
    """`tests/contracts/test_quota_state_survives_a_runner.py` holds a journal entry to a
    counter's 4,096 bytes; the `published` block must leave room under its headroom: long
    names, all three sources answering, a lineup capture's label and a locked player."""
    long = [(f"{p[0]} Longsurname-Doubleteam", *p[1:]) for p in PLAYERS]
    df = pl.DataFrame({
        "player": [p[0] for p in long], "key": [player_key(p[0]) for p in long],
        "pos": [p[1] for p in long], "nfl_team": ["XXX"] * len(long),
        "mu": [p[2] for p in long], "sd": [5.0] * len(long),
        "projected": [True] * len(long), "can_start": [True] * len(long),
        "available": [True] * len(long),
        "starting": [p[0].rsplit(" ", 1)[0] in MU_SET for p in long],
        "espn_week": [SW] * len(long), "espn_week_proj": [p[4] for p in long],
        "kicked_off": [p[0].startswith("Qa") for p in long]},
        schema_overrides={"espn_week": pl.Int64})
    cons = tmp_path / "cons"
    cap = consensus.build(SEASON, SW, _ecr_rows(long), THU - timedelta(days=2), THU, WED_AM, "d")
    atomic.write_text(consensus.capture_path(SEASON, SW, WED_AM, cons),
                      json.dumps(cap, sort_keys=True) + "\n")
    _friday_lineup_capture(cons, _ecr_rows(long))
    path, _ = J.record(df, season=SEASON, base=tmp_path / "j", now=WED_AM,
                       schedule=_week_schedule(), captures=cons)
    lk, rule = lineup_rule.decide(df, SEASON, SW, cons, ecr={}, ecr_reason="stale")
    J.record_published(SEASON, SW, lk, rule, base=tmp_path / "j", schedule=_week_schedule(),
                       now=SAT_AM)
    assert path is not None
    doc = _doc(path)
    assert all(doc["sources"][s]["lineup"] for s in J.SOURCES)
    assert doc["published"]["ecr_capture"] and doc["published"]["locked"]
    assert len(json.dumps(doc)) < 0.8 * 4096, len(json.dumps(doc))
