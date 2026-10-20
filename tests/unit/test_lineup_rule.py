"""Which source sets the published lineup: `mu` through week 6, weekly ECR from week 7 (#492).

`FIRST` below is read off `ECR_FROM_WEEK`, so these tests move with the constant.

The roster is built so the two sources differ in exactly one slot, the flex: `mu` starts the third
back (Rc Three), consensus the fourth receiver (Wd Four). A switch that does not switch publishes
the same lineup either side of the boundary, and every test below can see it.
"""
import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from hub import atomic, publish
from hub.contracts import ContractViolation, check_roster_artifact
from hub.fetch import consensus
from hub.names import player_key
from hub.season import lineup_journal, lineup_rule
from hub.season.roster import lock

SEASON = 2026
EARLY = datetime(2026, 10, 10, 12, tzinfo=UTC)          # before every week used here starts
FIRST = lineup_rule.ECR_FROM_WEEK[2026]    # the first ECR week of 2026; the week before is the old lock
OLD = FIRST - 1

# name, pos, mu, ecr
PLAYERS = [
    ("Qa One", "QB", 20.0, 5.0), ("Qb Two", "QB", 8.0, 150.0),
    ("Ra One", "RB", 18.0, 3.0), ("Rb Two", "RB", 17.0, 4.0),
    ("Rc Three", "RB", 12.0, 80.0), ("Rd Four", "RB", 2.0, 200.0),
    ("Wa One", "WR", 16.0, 6.0), ("Wb Two", "WR", 15.0, 7.0),
    ("Wc Three", "WR", 14.0, 8.0), ("Wd Four", "WR", 10.0, 40.0),
    ("We Five", "WR", 1.0, 210.0),
    ("Ta One", "TE", 13.0, 9.0), ("Tb Two", "TE", 5.0, 90.0),
]
MU_FLEX, ECR_FLEX = "Rc Three", "Wd Four"
CORE = ["Qa One", "Ra One", "Rb Two", "Wa One", "Wb Two", "Wc Three", "Ta One"]


def _roster(week: int | None, on_bye: tuple[str, ...] = ()) -> pl.DataFrame:
    n = len(PLAYERS)
    return pl.DataFrame({
        "player": [p[0] for p in PLAYERS], "key": [player_key(p[0]) for p in PLAYERS],
        "pos": [p[1] for p in PLAYERS], "nfl_team": ["XXX"] * n,
        "mu": [p[2] for p in PLAYERS], "sd": [5.0] * n,
        "projected": [True] * n, "can_start": [True] * n, "available": [True] * n,
        # The set lineup is the model's, so a switch to consensus shows as one swap.
        "starting": [p[0] in CORE or p[0] == MU_FLEX for p in PLAYERS],
        "on_bye": [p[0] in on_bye for p in PLAYERS],
        "injury_status": ["ACTIVE"] * n, "slot": ["BE"] * n,
        "espn_week": [week] * n, "espn_week_proj": [None] * n},
        schema_overrides={"espn_week": pl.Int64, "espn_week_proj": pl.Float64})


def _capture(base, week: int, drop=()) -> None:
    keep = [p for p in PLAYERS if p[0] not in drop]
    rows = pl.DataFrame({"player": [p[0] for p in keep], "pos": [p[1] for p in keep],
                         "team": ["XXX"] * len(keep), "ecr": [p[3] for p in keep]})
    first = date(2026, 10, 25 + 7 * (week - FIRST))
    doc = consensus.build(SEASON, week, rows, first, first, EARLY, "d")
    atomic.write_text(consensus.capture_path(SEASON, week, EARLY, base),
                      json.dumps(doc, sort_keys=True) + "\n")


def _flex(lineup) -> str:
    return next(n for n in lineup if n in (MU_FLEX, ECR_FLEX, "Tb Two"))


# --- the boundary -----------------------------------------------------------------------------

def test_the_decided_boundary_is_week_7_of_2026_by_the_schedule():
    """The maintainer's 2026-10-20 restatement (#492), pinned as literals so a drift in the constant
    is red here and not silently followed by every test that reads it: 10-15..10-19 is week 6, so
    Sunday 10-25 (week 7) is the first ECR lineup."""
    assert lineup_rule.ECR_FROM_WEEK == {2026: 7}
    assert [lineup_rule.rule_for(2026, w) for w in (5, 6, 7, 8)] == ["mu", "mu", "ecr", "ecr"]


def test_the_rule_is_mu_before_the_decided_week_and_ecr_from_it():
    assert [lineup_rule.rule_for(SEASON, w) for w in (1, OLD, FIRST, FIRST + 1, 17)] == \
        ["mu", "mu", "ecr", "ecr", "ecr"]
    assert lineup_rule.rule_for(SEASON, None) == "mu"
    assert lineup_rule.rule_for(2025, 12) == "mu", "the decision was for 2026 only"


def test_the_old_week_publishes_the_models_lineup_even_with_a_capture_in_hand(tmp_path):
    """Replaying the old lock week gives the lineup it gave: the capture is there and is not used."""
    _capture(tmp_path, OLD)
    df = _roster(OLD)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path)
    assert rule.source == "mu" and rule.rule == "mu" and rule.reason is None
    assert lk == lock(df), "the week before the switch is the old lock, field for field"
    assert _flex(lk.best_lineup) == MU_FLEX


def test_the_first_ecr_week_publishes_the_consensus_lineup(tmp_path):
    _capture(tmp_path, FIRST)
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == FIRST and rule.reason is None
    assert rule.label == "provisional"
    assert _flex(lk.best_lineup) == ECR_FLEX
    assert sorted(lk.best_lineup) == sorted([*CORE, ECR_FLEX])
    assert lk.start == [ECR_FLEX] and lk.bench == [MU_FLEX]
    # Totals stay in `mu`: the model rates the consensus flex two points lower.
    assert lk.gain == pytest.approx(-2.0)


def test_consensus_never_starts_a_player_on_bye(tmp_path):
    _capture(tmp_path, FIRST)
    lk, rule = lineup_rule.decide(_roster(FIRST, on_bye=("Wa One",)), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and "Wa One" not in lk.best_lineup
    assert "Wa One" in lk.withheld


# --- graceful degradation ---------------------------------------------------------------------

def test_a_missing_first_week_capture_serves_the_previous_weeks_and_says_so(tmp_path):
    _capture(tmp_path, OLD)
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == OLD
    assert f"week {FIRST}" in (rule.reason or "") and "last-good" in (rule.reason or "")
    assert _flex(lk.best_lineup) == ECR_FLEX


def test_no_capture_at_all_serves_the_models_lineup_and_says_why(tmp_path):
    df = _roster(FIRST)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path / "none")
    assert rule.source == "mu" and rule.rule == "ecr" and rule.label == "model"
    assert "no earlier capture" in (rule.reason or "")
    assert "model's lineup (mu)" in (rule.reason or "")
    assert lk == lock(df)


def test_an_unreadable_capture_degrades_rather_than_raising(tmp_path):
    bad = consensus.capture_path(SEASON, FIRST, EARLY, tmp_path)
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "mu" and rule.reason and "unreadable" in rule.reason
    assert lk.best_lineup


def test_consensus_that_ranks_too_few_to_fill_a_lineup_falls_back_to_mu(tmp_path):
    # The first ECR week ranks no quarterback; the week before (last-good) ranks no tight end.
    _capture(tmp_path, FIRST, drop=("Qa One", "Qb Two"))
    _capture(tmp_path, OLD, drop=("Ta One", "Tb Two"))
    df = _roster(FIRST)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path)
    assert rule.source == "mu"
    assert "too few" in (rule.reason or "") and f"week {OLD}" in (rule.reason or "")
    assert lk == lock(df)


def test_a_roster_with_no_scoring_period_keeps_mu_and_says_the_switch_cannot_be_placed(tmp_path):
    _capture(tmp_path, FIRST)
    _, rule = lineup_rule.decide(_roster(None), SEASON, captures=tmp_path)
    assert rule.source == "mu" and "cannot be placed" in (rule.reason or "")


# --- the label --------------------------------------------------------------------------------

def test_the_ecr_rule_is_labelled_provisional_and_never_an_edge(tmp_path):
    _capture(tmp_path, FIRST)
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    d = rule.as_dict()
    assert d["label"] == "provisional" and d["source"] == "ecr" and d["ecr_from_week"] == FIRST
    assert "provisional rule (ADR-0014)" in d["caveat"] and "not a measured edge" in d["caveat"]
    _, mu = lineup_rule.decide(_roster(OLD), SEASON, captures=tmp_path)
    assert mu.as_dict()["label"] == "model" and mu.as_dict()["caveat"] is None


def test_the_contract_refuses_an_unlabelled_or_unexplained_rule():
    ok = {"source": "ecr", "rule": "ecr", "label": "provisional", "week": FIRST, "ecr_week": FIRST,
          "reason": None}
    check_roster_artifact({"rows": [], "lineup_rule": ok})
    # Each bad rule breaks exactly one check, so each check is the only thing that refuses it.
    for bad, said in (({**ok, "label": "model"}, "label"),
                      ({**ok, "source": "espn", "rule": "espn"}, "source"),
                      ({**ok, "ecr_week": OLD}, "no reason"),
                      ({**ok, "source": "mu", "label": "model"}, "no reason")):
        with pytest.raises(ContractViolation, match=said):
            check_roster_artifact({"rows": [], "lineup_rule": bad})


# --- through the publisher and the journal ----------------------------------------------------

@pytest.fixture
def _publishable(monkeypatch, tmp_path):
    from hub.season import stream, waiver
    monkeypatch.setattr(consensus, "ROOT", tmp_path / "cons")
    monkeypatch.setattr(publish, "_roster_stream", lambda df: stream.unavailable("stubbed"))
    monkeypatch.setattr(publish, "_roster_waiver", lambda df: waiver.unavailable("stubbed"))
    return tmp_path


def _published(tmp_path, df) -> dict:
    src = tmp_path / "roster.parquet"
    df.write_parquet(src)
    publish.roster(out=tmp_path / "site", path=src)
    return json.loads((tmp_path / "site" / "roster.json").read_text())


def test_the_published_roster_carries_the_rule_and_its_lineup(_publishable):
    tmp = _publishable
    _capture(tmp / "cons", OLD)
    _capture(tmp / "cons", FIRST)
    for week, src, flex in ((OLD, "mu", MU_FLEX), (FIRST, "ecr", ECR_FLEX)):
        art = _published(tmp, _roster(week))
        assert art["lineup_rule"]["source"] == src and art["lineup_rule"]["week"] == week
        best = sorted(r["player"] for r in art["rows"] if r["best_start"])
        assert best == sorted([*CORE, flex]), week
    assert art["lineup_rule"]["label"] == "provisional"
    assert art["start"] == [ECR_FLEX] and art["sit"] == [MU_FLEX]


def test_the_journal_records_what_was_published_and_scores_it(tmp_path):
    _capture(tmp_path / "cons", FIRST)
    sched = pl.DataFrame({"game_id": [f"{FIRST}-0"], "season": [SEASON], "week": [FIRST],
                          "game_type": ["REG"], "home_team": ["A"], "away_team": ["B"],
                          "gameday": ["2026-10-25"], "home_score": [20], "away_score": [17]})
    path, _ = lineup_journal.record(_roster(FIRST), season=SEASON, base=tmp_path / "j",
                                    captures=tmp_path / "cons", now=EARLY, schedule=sched)
    assert path is not None
    doc = json.loads(path.read_text())
    assert doc["rule"]["source"] == "ecr" and doc["rule"]["label"] == "provisional"
    assert doc["rule"]["lineup"] == sorted([*CORE, ECR_FLEX])
    points = {player_key(n): 10.0 for n in CORE} | {player_key(ECR_FLEX): 3.0}
    lineup_journal.settle(SEASON, FIRST, points, tmp_path / "j", schedule=sched)
    doc = json.loads(path.read_text())
    assert doc["realised"]["set"] == {"points": 73.0, "n_no_row": 0}


# --- the week's lineup capture: after the strict one, before last-good ------------------------
#
# Week 7 of 2026 for real (10-15..10-19 is week 6): Thursday 10-22, Sunday 10-25, week 6's last
# game Monday 10-19. The
# archive publishes on Fridays, so the lineup capture is Friday 10-23's page. Its ranks swap the
# two flex candidates, so a lineup set on it is visibly not one set on the strict capture.

LINEUP_FLIP = {MU_FLEX: 39.0, ECR_FLEX: 81.0}         # the lineup capture prefers Rc Three
FRIDAY = datetime(2026, 10, 23, 15, tzinfo=UTC)


def _lineup_capture(base, week: int = FIRST, captured_at: datetime = FRIDAY,
                    scrape: date = date(2026, 10, 23)) -> None:
    rows = pl.DataFrame({"player": [p[0] for p in PLAYERS], "pos": [p[1] for p in PLAYERS],
                         "team": ["XXX"] * len(PLAYERS),
                         "ecr": [LINEUP_FLIP.get(p[0], p[3]) for p in PLAYERS]})
    first, sunday = date(2026, 10, 22), date(2026, 10, 25)
    doc = consensus.build_lineup(SEASON, week, rows, scrape, date(2026, 10, 20), first, sunday,
                                 captured_at, "d")
    atomic.write_text(consensus.lineup_path(SEASON, week, captured_at, base),
                      json.dumps(doc, sort_keys=True) + "\n")


def test_the_weeks_strict_capture_is_preferred_to_its_lineup_capture(tmp_path):
    _capture(tmp_path, FIRST)
    _lineup_capture(tmp_path)
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_capture is None and rule.reason is None
    assert _flex(lk.best_lineup) == ECR_FLEX
    assert "ecr_capture" not in rule.as_dict()


def test_with_no_strict_capture_the_lineup_capture_sets_it_ahead_of_last_good(tmp_path):
    _capture(tmp_path, OLD)                               # last-good, which must not be used
    _lineup_capture(tmp_path)
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == FIRST and rule.reason is None
    assert _flex(lk.best_lineup) == MU_FLEX, "set on the lineup capture's ranks"
    assert rule.ecr_capture is not None and rule.ecr_capture["purpose"] == "lineup"
    assert rule.ecr_capture["scrape_date"] == "2026-10-23"


def test_a_lineup_capture_that_cannot_fill_a_lineup_falls_through_to_last_good(tmp_path):
    _capture(tmp_path, OLD)
    thin = consensus.build_lineup(
        SEASON, FIRST, pl.DataFrame({"player": ["Ra One"], "pos": ["RB"], "team": ["XXX"],
                                 "ecr": [1.0]}),
        date(2026, 10, 23), date(2026, 10, 20), date(2026, 10, 22), date(2026, 10, 25),
        FRIDAY, "d")
    consensus.write_lineup_capture(thin, tmp_path)
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == OLD and rule.ecr_capture is None
    assert "last-good" in (rule.reason or "")


def test_a_strict_capture_too_thin_to_fill_a_lineup_hands_over_to_the_lineup_capture(tmp_path):
    _capture(tmp_path, FIRST, drop=("Qa One", "Qb Two"))
    _lineup_capture(tmp_path)
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_capture is not None
    assert "too few" in (rule.reason or "") and "lineup capture" in (rule.reason or "")


def test_a_lineup_capture_taken_after_sunday_starts_is_never_served(tmp_path):
    """The plant at the read: a file that reached the tree with a capture time at Sunday's
    deadline (00:00 Eastern, 04:00 UTC) is dropped, and the week falls to last-good."""
    _capture(tmp_path, OLD)
    _lineup_capture(tmp_path, captured_at=datetime(2026, 10, 25, 4, tzinfo=UTC))
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.ecr_capture is None and rule.ecr_week == OLD
    _lineup_capture(tmp_path, captured_at=datetime(2026, 10, 25, 3, 59, tzinfo=UTC))
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.ecr_capture is not None, "a minute earlier is in time"


def test_an_unreadable_lineup_capture_degrades_to_last_good_rather_than_raising(tmp_path):
    _capture(tmp_path, OLD)
    bad = consensus.lineup_path(SEASON, FIRST, FRIDAY, tmp_path)
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    assert lineup_rule.lineup_ecr(SEASON, FIRST, tmp_path) == ({}, None)
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == OLD and rule.ecr_capture is None


def test_replaying_the_old_lock_week_is_unchanged_by_a_lineup_capture(tmp_path):
    _capture(tmp_path, OLD)
    df = _roster(OLD)
    before = lineup_rule.decide(df, SEASON, captures=tmp_path)
    _lineup_capture(tmp_path, week=OLD)
    after = lineup_rule.decide(df, SEASON, captures=tmp_path)
    assert after == before and after[0] == lock(df)
    assert after[1].as_dict() == before[1].as_dict() and "ecr_capture" not in after[1].as_dict()


def test_a_lineup_capture_is_labelled_as_one_and_keeps_the_provisional_caveat(tmp_path):
    _lineup_capture(tmp_path)
    _, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    d = rule.as_dict()
    note = d["ecr_capture"]["note"]
    assert note == "consensus ranks as of Fri 2026-10-23, taken after Thursday's games"
    assert d["basis"] == lineup_rule.ECR_LINEUP_BASIS and "not a pre-kickoff" in d["basis"]
    assert "provisional rule (ADR-0014)" in d["caveat"] and "not a measured edge" in d["caveat"]
    assert d["label"] == "provisional"
    check_roster_artifact({"rows": [], "lineup_rule": d})


def test_the_contract_refuses_a_lineup_capture_that_does_not_say_what_it_is():
    ok = {"source": "ecr", "rule": "ecr", "label": "provisional", "week": FIRST, "ecr_week": FIRST,
          "reason": None, "ecr_capture": {"purpose": "lineup", "scrape_date": "2026-10-23",
                                          "note": "consensus ranks as of Fri 2026-10-23"}}
    check_roster_artifact({"rows": [], "lineup_rule": ok})
    for bad in ({**ok, "ecr_capture": {**ok["ecr_capture"], "note": ""}},
                {**ok, "ecr_capture": {**ok["ecr_capture"], "purpose": "strict"}},
                {**ok, "ecr_capture": "Friday"}):
        with pytest.raises(ContractViolation, match="labelled lineup capture"):
            check_roster_artifact({"rows": [], "lineup_rule": bad})


def test_the_journals_strict_read_never_sees_a_lineup_capture(tmp_path):
    _lineup_capture(tmp_path)
    ranks, why = lineup_journal.ecr_ranks(SEASON, FIRST, tmp_path)
    assert ranks == {} and f"no pre-kickoff consensus capture for week {FIRST}" in (why or "")
    assert consensus.read_captures(SEASON, tmp_path) == []


def test_the_published_roster_shows_the_lineup_capture_and_its_ranks(_publishable):
    tmp = _publishable
    _lineup_capture(tmp / "cons")
    art = _published(tmp, _roster(FIRST))
    rule = art["lineup_rule"]
    assert rule["source"] == "ecr" and rule["ecr_capture"]["purpose"] == "lineup"
    assert "taken after Thursday's games" in rule["ecr_capture"]["note"]
    by_name = {r["player"]: r["ecr"] for r in art["rows"]}
    assert by_name[MU_FLEX] == 39.0, "the ecr column shows the ranks that set the lineup"
    assert art["ecr_reason"] is None
    best = sorted(r["player"] for r in art["rows"] if r["best_start"])
    assert best == sorted([*CORE, MU_FLEX])



# --- #516: a kicked-off game closes the call on the consensus path too ------------------------
#
# The lineup capture is taken after Thursday's games, so a Thursday player can be missing from
# it. Ra One (a set starter) played Thursday and is not on the page; Wd Four (on the bench) also
# played Thursday. An open search reads Ra One as unranked, SITs him, and STARTs Wd Four.

THURSDAY = ("Ra One", "Wd Four")


def _lineup_capture_without(base, drop: tuple[str, ...]) -> None:
    keep = [p for p in PLAYERS if p[0] not in drop]
    rows = pl.DataFrame({"player": [p[0] for p in keep], "pos": [p[1] for p in keep],
                         "team": ["XXX"] * len(keep),
                         "ecr": [LINEUP_FLIP.get(p[0], p[3]) for p in keep]})
    doc = consensus.build_lineup(SEASON, FIRST, rows, date(2026, 10, 23), date(2026, 10, 20),
                                 date(2026, 10, 22), date(2026, 10, 25), FRIDAY, "d")
    atomic.write_text(consensus.lineup_path(SEASON, FIRST, FRIDAY, base),
                      json.dumps(doc, sort_keys=True) + "\n")


def _kicked_off(df: pl.DataFrame, who: tuple[str, ...] = THURSDAY) -> pl.DataFrame:
    return df.with_columns(pl.col("player").is_in(list(who)).alias("kicked_off"))


def test_a_started_player_missing_from_the_lineup_capture_is_not_treated_as_unranked(tmp_path):
    _lineup_capture_without(tmp_path, ("Ra One",))
    lk, rule = lineup_rule.decide(_kicked_off(_roster(FIRST)), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_capture is not None
    assert "Ra One" in lk.best_lineup, "kept as set, not dropped as unranked"
    assert "Wd Four" not in lk.best_lineup, "started on the bench: never started"
    assert lk.locked == THURSDAY
    assert lk.start == [] and lk.bench == [] and lk.gain == pytest.approx(0.0)


def test_the_same_capture_without_the_lock_makes_both_moves(tmp_path):
    """The control: the same capture and roster, nothing kicked off -- Ra One is unranked and
    sat, Wd Four is started. These are the moves the test above refuses."""
    _lineup_capture_without(tmp_path, ("Ra One",))
    lk, rule = lineup_rule.decide(_roster(FIRST), SEASON, captures=tmp_path)
    assert rule.source == "ecr"
    assert "Ra One" in lk.bench and "Wd Four" in lk.start


def test_the_published_roster_says_who_is_locked_and_swaps_neither(_publishable):
    tmp = _publishable
    _lineup_capture_without(tmp / "cons", ("Ra One",))
    art = _published(tmp, _kicked_off(_roster(FIRST)))
    assert art["locked"] == list(THURSDAY)
    assert art["locked_reasons"] == dict.fromkeys(THURSDAY, "locked: game started")
    assert {r["player"] for r in art["rows"] if r["locked"]} == set(THURSDAY)
    named = {s[k] for s in art["swaps"] for k in ("out", "in")}
    assert named.isdisjoint(THURSDAY) and not set(art["start"] + art["sit"]) & set(THURSDAY)
    check_roster_artifact(art)


def test_a_week_with_no_kickoff_publishes_byte_identical_json(_publishable):
    """The column all false is the parquet before #516 (no column): the same bytes."""
    tmp = _publishable
    _capture(tmp / "cons", FIRST)
    texts = []
    for df in (_roster(FIRST), _kicked_off(_roster(FIRST), ())):
        src = tmp / "roster.parquet"
        df.write_parquet(src)
        site = tmp / f"site{len(texts)}"
        publish.roster(out=site, path=src)
        texts.append((site / "roster.json").read_text())
    old, new = (json.loads(t) for t in texts)
    # `generated_at` is the parquet's mtime; everything the lock decided is what is compared.
    old.pop("generated_at"), new.pop("generated_at")
    assert json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
    assert "locked" not in new and all("locked" not in r for r in new["rows"])
