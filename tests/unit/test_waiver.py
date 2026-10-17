"""Waiver ranking against our replacement level (#493): a provisional rule, shown with its basis.

The controls (rule 18): a planted free-agent list with known projections gives a known order;
a free agent worse than the player he would replace has value <= 0 and is not recommended; the
subtraction of the replacement is what makes the order, so removing it turns these red.
"""
import json
from datetime import UTC, datetime

import polars as pl
import pytest

from hub import publish
from hub.fetch import free_agents as fa_mod
from hub.fetch.espn import FREE_AGENT_SCHEMA
from hub.season import waiver

AT = datetime(2026, 10, 18, 10, 0)           # Sunday morning of week 5, naive UTC


def _roster():
    rows = [
        ("Starter Back", "RB", "GB", "RB", 200.0, 14.0),
        ("Weak Back", "RB", "GB", "BE", 80.0, 5.0),
        ("Hurt Back", "RB", "GB", "IR", 0.0, None),        # on IR: never "replaced"
        ("Wr One", "WR", "DAL", "WR", 150.0, 11.0),
        ("Wr Two", "WR", "DAL", "BE", 120.0, 8.0),
        ("Our K", "K", "BUF", "K", 100.0, 8.0),
        ("Our D", "D/ST", "NYJ", "D/ST", 90.0, 7.0),
    ]
    return pl.DataFrame(
        {"player": [r[0] for r in rows], "pos": [r[1] for r in rows],
         "nfl_team": [r[2] for r in rows], "slot": [r[3] for r in rows],
         "espn_total": [r[4] for r in rows], "espn_week_proj": [r[5] for r in rows]},
        schema={"player": pl.Utf8, "pos": pl.Utf8, "nfl_team": pl.Utf8, "slot": pl.Utf8,
                "espn_total": pl.Float64, "espn_week_proj": pl.Float64})


def _fa(rows):
    """rows: (name, pos, team, rest-of-season, status, owned)"""
    out = [{"player": n, "key": n.lower(), "pos": p, "espn_id": i, "nfl_team": t,
            "injury_status": s, "espn_avg": a / 14 if a else 0.0, "espn_total": a,
            "espn_week": 5, "espn_week_proj": a / 14 if a else None, "percent_owned": o}
           for i, (n, p, t, a, s, o) in enumerate(rows, start=1)]
    return pl.DataFrame(out, schema=FREE_AGENT_SCHEMA)


PLANTED = [
    ("Rb Alpha", "RB", "GB", 130.0, "ACTIVE", 41.0),       # +50 over Weak Back (80)
    ("Rb Beta", "RB", "GB", 95.0, "ACTIVE", 30.0),         # +15
    ("Rb Gamma", "RB", "GB", 79.0, "ACTIVE", 20.0),        # -1: worse than the replacement
    ("Rb Out", "RB", "GB", 140.0, "OUT", 50.0),            # +60 but cannot play
    ("Rb Ghost", "RB", "GB", 0.0, "ACTIVE", 1.0),          # ESPN carries nothing
    ("Wr Xray", "WR", "DAL", 125.0, "ACTIVE", 12.0),       # +5 over Wr Two (120)
    ("Wr Yankee", "WR", "DAL", 110.0, "ACTIVE", 9.0),      # -10
    ("K Zulu", "K", "BUF", 112.0, "ACTIVE", 33.0),         # +12 over Our K (100)
    ("Te Tango", "TE", "KC", 100.0, "ACTIVE", 5.0),        # we roster no tight end
]


def _built(**kw):
    return waiver.build(_fa(PLANTED), _roster(), week=5, **kw)


def test_a_planted_pool_gives_the_expected_order_by_value():
    got = _built()
    assert [(r["player"], r["value"]) for r in got["rows"]] == [
        ("Rb Out", 60.0), ("Rb Alpha", 50.0), ("Rb Beta", 15.0), ("K Zulu", 12.0),
        ("Wr Xray", 5.0), ("Rb Gamma", -1.0), ("Wr Yankee", -10.0)]
    assert [r["rank"] for r in got["rows"]] == [1, 2, 3, 4, 5, 6, 7]


def test_the_value_is_net_of_the_player_replaced_not_the_raw_projection():
    """The subtraction: Rb Gamma projects 79, a fine number, and is a loss against the 80 he
    would replace. A ranking on the raw projection would put Wr Xray (125) above Rb Beta (95)."""
    by = {r["player"]: r for r in _built()["rows"]}
    assert by["Rb Beta"]["replaces"] == "Weak Back" and by["Rb Beta"]["replaces_proj"] == 80.0
    assert by["Wr Xray"]["replaces"] == "Wr Two"
    assert by["Rb Beta"]["value"] > by["Wr Xray"]["value"]
    assert by["Rb Beta"]["proj"] < by["Wr Xray"]["proj"]


def test_a_free_agent_worse_than_our_replacement_is_not_recommended():
    by = {r["player"]: r for r in _built()["rows"]}
    for name in ("Rb Gamma", "Wr Yankee"):
        assert by[name]["value"] <= 0 and by[name]["recommended"] is False
    assert not {"Rb Gamma", "Wr Yankee"} & set(_built()["recommended"])


def test_a_player_who_cannot_play_is_shown_and_never_recommended():
    by = {r["player"]: r for r in _built()["rows"]}
    assert by["Rb Out"]["blocked"] == "OUT" and by["Rb Out"]["recommended"] is False
    assert _built()["recommended"] == ["Rb Alpha", "Rb Beta", "K Zulu", "Wr Xray"]


def test_the_player_on_injured_reserve_is_never_the_one_replaced():
    """Hurt Back projects 0 and sits lowest; replacing him would make every back look +130."""
    assert waiver.replacements(_roster())["RB"]["player"] == "Weak Back"


def test_an_unprojected_free_agent_has_no_value_and_a_position_with_no_spot_is_counted():
    got = _built()
    assert "Rb Ghost" not in [r["player"] for r in got["rows"]]
    assert got["unprojected"] == 1
    assert got["no_roster_spot"] == 1 and "Te Tango" not in [r["player"] for r in got["rows"]]


def test_a_rostered_player_with_no_projection_is_the_replacement_and_says_so():
    ros = _roster().with_columns(
        pl.when(pl.col("player") == "Weak Back").then(0.0).otherwise(pl.col("espn_total"))
        .alias("espn_total"))
    got = waiver.build(_fa(PLANTED), ros, week=5)
    row = next(r for r in got["rows"] if r["player"] == "Rb Alpha")
    assert row["value"] == 130.0 and row["replaces_no_projection"] is True


def test_the_replacement_tie_goes_to_the_first_name_not_the_row_order():
    ros = pl.DataFrame({"player": ["Zed", "Abe"], "pos": ["WR", "WR"], "slot": ["BE", "BE"],
                        "espn_total": [50.0, 50.0], "espn_week_proj": [None, None]})
    assert waiver.replacements(ros)["WR"]["player"] == "Abe"


def test_only_the_top_few_per_position_are_published():
    many = [(f"Wr {i}", "WR", "DAL", 130.0 + i, "ACTIVE", 1.0) for i in range(9)]
    got = waiver.build(_fa(many), _roster(), week=5, top=3)
    assert [r["player"] for r in got["rows"]] == ["Wr 8", "Wr 7", "Wr 6"]


def test_what_is_shown_beside_a_row_never_moves_the_order():
    """ADR-0013 / ADR-0016: snap trend, ECR, weekly projection and ownership are shown, not
    sorted on. Inverting every one of them leaves the order exactly where it was."""
    base = [r["player"] for r in _built()["rows"]]
    snaps = {r["key"]: {"delta": 0.3 - i * 0.1, "recent": 0.5, "baseline": 0.2}
             for i, r in enumerate(_fa(PLANTED).iter_rows(named=True))}
    ecr = {r["key"]: float(100 - i) for i, r in enumerate(_fa(PLANTED).iter_rows(named=True))}
    flipped = _built(snaps=snaps, ecr=ecr)
    assert [r["player"] for r in flipped["rows"]] == base
    row = next(r for r in flipped["rows"] if r["player"] == "Rb Alpha")
    assert row["snap_delta"] == pytest.approx(0.3) and row["ecr"] == 100.0


def test_the_block_is_labelled_provisional_with_its_basis_and_makes_no_bid():
    got = _built()
    assert got["label"] == "provisional" and "ADR-0014" in got["caveat"]
    assert "unshrunk" in got["shrinkage"] and "#497" in got["shrinkage"]
    assert "never enter the value or the order" in got["shown_not_sorted"]
    assert "no bid" in got["faab"]
    assert not any("bid" in k for r in got["rows"] for k in r)


def test_week_value_is_shown_when_both_sides_have_a_weekly_projection():
    by = {r["player"]: r for r in _built()["rows"]}
    assert by["Rb Alpha"]["week_value"] == pytest.approx(130 / 14 - 5.0)
    assert by["Rb Out"]["week_value"] == pytest.approx(140 / 14 - 5.0)


# --- bye and playoff weeks ------------------------------------------------------------------

def _games():
    rows = []
    for w in range(1, 19):
        ko = datetime(2026, 10, 18, 17) + (w - 5) * (datetime(2026, 10, 25) - datetime(2026, 10, 18))
        pairs = [("CCC", "DDD")] if w == 6 else [("AAA", "BBB"), ("CCC", "DDD")]
        rows += [{"game_id": f"{w}{h}", "week": w, "home_team": h, "away_team": a,
                  "kickoff": ko, "result": None} for h, a in pairs]
    return pl.DataFrame(rows, schema={"game_id": pl.Utf8, "week": pl.Int32,
                                      "home_team": pl.Utf8, "away_team": pl.Utf8,
                                      "kickoff": pl.Datetime, "result": pl.Float64})


def test_the_calendar_names_each_teams_bye_and_its_playoff_opponents():
    cal = waiver.calendar(_games())
    assert cal["AAA"]["bye_week"] == 6 and cal["CCC"]["bye_week"] is None
    assert cal["AAA"]["playoff"] == {"15": "BBB", "16": "BBB", "17": "BBB"}


def test_a_bye_inside_the_playoff_weeks_reads_as_none():
    g = _games().filter(~((pl.col("week") == 16) & (pl.col("home_team") == "AAA")))
    assert waiver.calendar(g)["AAA"]["playoff"]["16"] is None
    assert waiver.calendar(g)["AAA"]["bye_week"] == 6


def test_a_row_carries_the_bye_week_and_whether_the_team_is_on_a_bye_now():
    pool = _fa([("Rb Alpha", "RB", "AAA", 130.0, "ACTIVE", 1.0)])
    got = waiver.build(pool, _roster(), week=6, bye_now=frozenset({"AAA"}),
                       cal=waiver.calendar(_games()))
    r = got["rows"][0]
    assert r["bye_week"] == 6 and r["on_bye_now"] is True
    assert r["playoff"]["15"] == "BBB"


# --- the snap-share trend -------------------------------------------------------------------

def _snaps():
    rows = []
    for w in range(1, 9):
        rows.append(("Riser", "p1", w, "REG", 0.2 if w <= 6 else 0.6))
        rows.append(("Faller", "p2", w, "REG", 0.7 if w <= 6 else 0.3))
        rows.append(("Twin", "p3", w, "REG", 0.5))
        rows.append(("Twin", "p4", w, "REG", 0.1))                 # one key, two PFR ids
    rows.append(("Playoff Only", "p5", 8, "POST", 0.9))
    return pl.DataFrame(rows, orient="row", schema={
        "player": pl.Utf8, "pfr_player_id": pl.Utf8, "week": pl.Int32,
        "game_type": pl.Utf8, "offense_pct": pl.Float64})


def test_snap_trend_is_recent_weeks_against_the_earlier_baseline():
    got = waiver.snap_trend(_snaps(), week=9)
    assert got["riser"]["delta"] == pytest.approx(0.4)
    assert got["riser"]["recent"] == pytest.approx(0.6) and got["riser"]["baseline"] == pytest.approx(0.2)
    assert got["faller"]["delta"] == pytest.approx(-0.4)


def test_an_ambiguous_key_and_a_playoff_only_row_get_no_trend():
    got = waiver.snap_trend(_snaps(), week=9)
    assert "twin" not in got and "playoff only" not in got


def test_too_few_weeks_leave_a_player_out_and_the_reason_is_stated():
    assert waiver.snap_trend(_snaps(), week=3) == {}
    assert "weeks of snaps" in str(waiver._snap_reason(6, True))
    assert waiver._snap_reason(10, True) is None
    assert "unavailable" in str(waiver._snap_reason(10, False))
    assert "no week" in str(waiver._snap_reason(None, False))


# --- reading last-good and degrading --------------------------------------------------------

@pytest.fixture
def world(monkeypatch, tmp_path):
    from hub import schedule
    from hub.season import lineup_journal
    monkeypatch.setattr(schedule, "priced_games", lambda *a, **k: _games())
    monkeypatch.setattr(waiver, "_load_snaps", lambda season: _snaps())
    monkeypatch.setattr(lineup_journal, "ecr_ranks",
                        lambda season, week, base=None: ({"rb alpha": 12.0}, None))
    path = tmp_path / "fa.parquet"
    fa_mod.refresh(lambda: (_fa(PLANTED), False), path, datetime(2026, 10, 17, 12, tzinfo=UTC))
    return path


def test_for_roster_reads_last_good_and_assembles_the_block(world):
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    assert got["week"] == 5 and got["pool_size"] == len(PLANTED) and got["capped"] is False
    assert got["recommended"][0] == "Rb Alpha" and got["last_error"] is None
    assert got["stale"] is None and got["as_of"].startswith("2026-10-17")
    row = next(r for r in got["rows"] if r["player"] == "Rb Alpha")
    assert row["ecr"] == 12.0
    assert "weeks of snaps" in got["snap_reason"]        # week 5 has four weeks behind it


def test_a_failed_fetch_is_served_last_good_with_the_reason_on_the_block(world):
    fa_mod.refresh(lambda: (_ for _ in ()).throw(ConnectionError("ESPN is down")), world,
                   datetime(2026, 10, 18, tzinfo=UTC))
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    assert "ESPN is down" in got["last_error"]
    assert got["recommended"][0] == "Rb Alpha"            # the pool is still the last good one


def test_an_old_pool_is_flagged_stale(world):
    got = waiver.for_roster(_roster(), at=datetime(2026, 10, 30, 10), fa_path=world)
    assert "days old" in got["stale"]


def test_no_pool_yet_is_unavailable_with_the_command_to_run(tmp_path):
    got = waiver.for_roster(_roster(), at=AT, fa_path=tmp_path / "nope.parquet")
    assert got["label"] == "provisional" and got["rows"] == []
    assert "hub.fetch.free_agents" in got["missing"]


def test_an_empty_roster_is_unavailable_not_a_ranking_against_nothing(world):
    got = waiver.for_roster(_roster().head(0), at=AT, fa_path=world)
    assert got["rows"] == [] and "no roster" in got["missing"]


def test_an_unreadable_schedule_still_ranks_and_says_the_calendar_is_missing(world, monkeypatch):
    from hub import schedule

    def boom(*a, **k):
        raise OSError("schedule host down")
    monkeypatch.setattr(schedule, "priced_games", boom)
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    assert got["week"] is None and got["recommended"][0] == "Rb Alpha"
    assert "OSError" in got["calendar_reason"]
    assert got["rows"][0]["bye_week"] is None


def test_unreadable_snap_counts_do_not_stop_the_ranking(world, monkeypatch):
    def boom(season):
        raise OSError("no network")
    monkeypatch.setattr(waiver, "_load_snaps", boom)
    got = waiver.for_roster(_roster(), at=datetime(2026, 10, 18, 10), fa_path=world)
    assert got["recommended"][0] == "Rb Alpha" and got["snap_reason"]


def test_a_garbled_fetch_time_is_reported_not_raised(world):
    meta = fa_mod.meta_path(world)
    meta.write_text(json.dumps({"fetched_at": "last tuesday", "capped": True}))
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    assert "unreadable" in got["stale"] and got["capped"] is True


def test_a_game_with_a_team_the_canon_cannot_place_is_skipped_by_the_calendar():
    g = pl.concat([_games(), pl.DataFrame(
        [{"game_id": "x", "week": 3, "home_team": "", "away_team": "AAA",
          "kickoff": None, "result": None}], schema=_games().schema)])
    assert waiver.calendar(g)["AAA"]["bye_week"] == 6


def test_a_season_with_no_open_week_still_ranks_and_says_so(world, monkeypatch):
    from hub import schedule
    done = _games().with_columns(pl.lit(1.0).alias("result"),
                                 pl.lit(datetime(2026, 1, 1)).alias("kickoff"))
    monkeypatch.setattr(schedule, "priced_games", lambda *a, **k: done)
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    assert got["week"] is None and "no week is open" in got["calendar_reason"]
    assert got["recommended"][0] == "Rb Alpha"


def test_render_prints_the_failed_fetch_note(world):
    fa_mod.refresh(lambda: (_ for _ in ()).throw(ConnectionError("ESPN is down")), world,
                   datetime(2026, 10, 18, tzinfo=UTC))
    out = waiver.render(waiver.for_roster(_roster(), at=AT, fa_path=world))
    assert "NOTE (last_error)" in out and "ESPN is down" in out


# --- the journal ----------------------------------------------------------------------------

def test_record_writes_the_recommendations_once_and_first_write_wins(tmp_path, world):
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    path, note = waiver.record(got, 2026, tmp_path, now=datetime(2026, 10, 18, tzinfo=UTC))
    assert path is not None and "4 recommended" in note
    doc = json.loads(path.read_text())
    assert [p["player"] for p in doc["picks"]] == ["Rb Alpha", "Rb Beta", "K Zulu", "Wr Xray"]
    assert doc["claimed"] is None and doc["why_different"] is None and doc["scored"] is False
    assert doc["label"] == "provisional"
    before = path.read_bytes()
    again, note2 = waiver.record({**got, "rows": []}, 2026, tmp_path)
    assert again == path and "already journalled" in note2 and path.read_bytes() == before


def test_no_open_week_journals_nothing():
    path, note = waiver.record(waiver.unavailable("nothing yet"), 2026, None)
    assert path is None and "nothing yet" in note


def _journalled(tmp_path, world):
    got = waiver.for_roster(_roster(), at=AT, fa_path=world)
    waiver.record(got, 2026, tmp_path)


def test_settle_scores_three_finished_weeks_against_the_player_replaced(tmp_path, world):
    _journalled(tmp_path, world)
    pts = {5: {"rb alpha": 20.0, "weak back": 4.0}, 6: {"rb alpha": 10.0},
           7: {"rb alpha": 12.0, "weak back": 6.0}}
    assert waiver.settle(2026, 5, pts, tmp_path, last_final=7) == "week 5: scored"
    doc = waiver.read(2026, tmp_path)[0]
    alpha = next(p for p in doc["outcome"]["picks"] if p["player"] == "Rb Alpha")
    assert (alpha["in_points"], alpha["out_points"], alpha["net"]) == (42.0, 10.0, 32.0)
    assert alpha["n_no_row"] == 1                          # Weak Back had no row in week 6
    assert doc["scored"] is True and doc["outcome"]["horizon_weeks"] == [5, 6, 7]
    assert waiver.settle(2026, 5, pts, tmp_path, last_final=7).endswith("left as it was")


def test_settle_leaves_an_unfinished_horizon_unscored(tmp_path, world):
    _journalled(tmp_path, world)
    assert "not all final" in waiver.settle(2026, 5, {5: {}}, tmp_path, last_final=6)
    assert "not all final" in waiver.settle(2026, 5, {5: {}}, tmp_path, last_final=None)
    assert waiver.read(2026, tmp_path)[0]["scored"] is False
    assert "no journal entry" in waiver.settle(2026, 9, {}, tmp_path, last_final=18)


def test_the_last_final_week_stops_at_the_first_open_game():
    def sched(done):
        rows = [{"game_type": "REG", "week": w, "home_score": 20 if w in done else None,
                 "away_score": 10 if w in done else None} for w in range(1, 19)]
        return pl.DataFrame(rows, schema={"game_type": pl.Utf8, "week": pl.Int32,
                                          "home_score": pl.Int32, "away_score": pl.Int32})
    assert waiver.last_final_week(sched({1, 2, 3, 5})) == 3
    assert waiver.last_final_week(sched(set())) is None
    assert waiver.last_final_week(None) is None


def test_render_names_the_label_and_the_recommended_rows(world):
    out = waiver.render(waiver.for_roster(_roster(), at=AT, fa_path=world))
    assert out.startswith("  [provisional]") and "* Rb Alpha" in out and "Rb Out" in out
    assert "[OUT]" in out and "no ranking" not in out
    assert "no ranking" in waiver.render(waiver.unavailable("no pool"))


# --- it rides on roster.json ----------------------------------------------------------------

def test_the_publisher_seam_hands_the_roster_and_degrades_labelled(tmp_path, monkeypatch):
    monkeypatch.setattr(fa_mod, "FREE_AGENTS_PARQUET", tmp_path / "nope.parquet")
    got = publish._roster_waiver(_roster())
    assert got["label"] == "provisional" and got["rows"] == []
