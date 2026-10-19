"""Which source sets the published lineup: `mu` through week 7, weekly ECR from week 8 (#492).

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
    first = date(2026, 10, 18 + 7 * (week - 7))
    doc = consensus.build(SEASON, week, rows, first, first, EARLY, "d")
    atomic.write_text(consensus.capture_path(SEASON, week, EARLY, base),
                      json.dumps(doc, sort_keys=True) + "\n")


def _flex(lineup) -> str:
    return next(n for n in lineup if n in (MU_FLEX, ECR_FLEX, "Tb Two"))


# --- the boundary -----------------------------------------------------------------------------

def test_the_rule_is_mu_through_week_7_and_ecr_from_week_8():
    assert [lineup_rule.rule_for(SEASON, w) for w in (1, 7, 8, 9, 17)] == \
        ["mu", "mu", "ecr", "ecr", "ecr"]
    assert lineup_rule.rule_for(SEASON, None) == "mu"
    assert lineup_rule.rule_for(2025, 12) == "mu", "the decision was for 2026 only"


def test_week_7_publishes_the_models_lineup_even_with_a_capture_in_hand(tmp_path):
    """Replaying week 7 gives the lineup week 7 gave: the capture is there and is not used."""
    _capture(tmp_path, 7)
    df = _roster(7)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path)
    assert rule.source == "mu" and rule.rule == "mu" and rule.reason is None
    assert lk == lock(df), "week 7 is the old lock, field for field"
    assert _flex(lk.best_lineup) == MU_FLEX


def test_week_8_publishes_the_consensus_lineup(tmp_path):
    _capture(tmp_path, 8)
    lk, rule = lineup_rule.decide(_roster(8), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == 8 and rule.reason is None
    assert rule.label == "provisional"
    assert _flex(lk.best_lineup) == ECR_FLEX
    assert sorted(lk.best_lineup) == sorted([*CORE, ECR_FLEX])
    assert lk.start == [ECR_FLEX] and lk.bench == [MU_FLEX]
    # Totals stay in `mu`: the model rates the consensus flex two points lower.
    assert lk.gain == pytest.approx(-2.0)


def test_consensus_never_starts_a_player_on_bye(tmp_path):
    _capture(tmp_path, 8)
    lk, rule = lineup_rule.decide(_roster(8, on_bye=("Wa One",)), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and "Wa One" not in lk.best_lineup
    assert "Wa One" in lk.withheld


# --- graceful degradation ---------------------------------------------------------------------

def test_a_missing_week_8_capture_serves_week_7s_and_says_so(tmp_path):
    _capture(tmp_path, 7)
    lk, rule = lineup_rule.decide(_roster(8), SEASON, captures=tmp_path)
    assert rule.source == "ecr" and rule.ecr_week == 7
    assert "week 8" in (rule.reason or "") and "last-good" in (rule.reason or "")
    assert _flex(lk.best_lineup) == ECR_FLEX


def test_no_capture_at_all_serves_the_models_lineup_and_says_why(tmp_path):
    df = _roster(8)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path / "none")
    assert rule.source == "mu" and rule.rule == "ecr" and rule.label == "model"
    assert "no earlier capture" in (rule.reason or "")
    assert "model's lineup (mu)" in (rule.reason or "")
    assert lk == lock(df)


def test_an_unreadable_capture_degrades_rather_than_raising(tmp_path):
    bad = consensus.capture_path(SEASON, 8, EARLY, tmp_path)
    bad.parent.mkdir(parents=True)
    bad.write_text("{not json")
    lk, rule = lineup_rule.decide(_roster(8), SEASON, captures=tmp_path)
    assert rule.source == "mu" and rule.reason and "unreadable" in rule.reason
    assert lk.best_lineup


def test_consensus_that_ranks_too_few_to_fill_a_lineup_falls_back_to_mu(tmp_path):
    # Week 8 ranks no quarterback; week 7 (last-good) ranks no tight end.
    _capture(tmp_path, 8, drop=("Qa One", "Qb Two"))
    _capture(tmp_path, 7, drop=("Ta One", "Tb Two"))
    df = _roster(8)
    lk, rule = lineup_rule.decide(df, SEASON, captures=tmp_path)
    assert rule.source == "mu"
    assert "too few" in (rule.reason or "") and "week 7" in (rule.reason or "")
    assert lk == lock(df)


def test_a_roster_with_no_scoring_period_keeps_mu_and_says_the_switch_cannot_be_placed(tmp_path):
    _capture(tmp_path, 8)
    _, rule = lineup_rule.decide(_roster(None), SEASON, captures=tmp_path)
    assert rule.source == "mu" and "cannot be placed" in (rule.reason or "")


# --- the label --------------------------------------------------------------------------------

def test_the_ecr_rule_is_labelled_provisional_and_never_an_edge(tmp_path):
    _capture(tmp_path, 8)
    _, rule = lineup_rule.decide(_roster(8), SEASON, captures=tmp_path)
    d = rule.as_dict()
    assert d["label"] == "provisional" and d["source"] == "ecr" and d["ecr_from_week"] == 8
    assert "provisional rule (ADR-0014)" in d["caveat"] and "not a measured edge" in d["caveat"]
    _, mu = lineup_rule.decide(_roster(7), SEASON, captures=tmp_path)
    assert mu.as_dict()["label"] == "model" and mu.as_dict()["caveat"] is None


def test_the_contract_refuses_an_unlabelled_or_unexplained_rule():
    ok = {"source": "ecr", "rule": "ecr", "label": "provisional", "week": 8, "ecr_week": 8,
          "reason": None}
    check_roster_artifact({"rows": [], "lineup_rule": ok})
    # Each bad rule breaks exactly one check, so each check is the only thing that refuses it.
    for bad, said in (({**ok, "label": "model"}, "label"),
                      ({**ok, "source": "espn", "rule": "espn"}, "source"),
                      ({**ok, "ecr_week": 7}, "no reason"),
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
    _capture(tmp / "cons", 7)
    _capture(tmp / "cons", 8)
    for week, src, flex in ((7, "mu", MU_FLEX), (8, "ecr", ECR_FLEX)):
        art = _published(tmp, _roster(week))
        assert art["lineup_rule"]["source"] == src and art["lineup_rule"]["week"] == week
        best = sorted(r["player"] for r in art["rows"] if r["best_start"])
        assert best == sorted([*CORE, flex]), week
    assert art["lineup_rule"]["label"] == "provisional"
    assert art["start"] == [ECR_FLEX] and art["sit"] == [MU_FLEX]


def test_the_journal_records_what_was_published_and_scores_it(tmp_path):
    _capture(tmp_path / "cons", 8)
    sched = pl.DataFrame({"game_id": ["8-0"], "season": [SEASON], "week": [8],
                          "game_type": ["REG"], "home_team": ["A"], "away_team": ["B"],
                          "gameday": ["2026-10-25"], "home_score": [20], "away_score": [17]})
    path, _ = lineup_journal.record(_roster(8), season=SEASON, base=tmp_path / "j",
                                    captures=tmp_path / "cons", now=EARLY, schedule=sched)
    assert path is not None
    doc = json.loads(path.read_text())
    assert doc["rule"]["source"] == "ecr" and doc["rule"]["label"] == "provisional"
    assert doc["rule"]["lineup"] == sorted([*CORE, ECR_FLEX])
    points = {player_key(n): 10.0 for n in CORE} | {player_key(ECR_FLEX): 3.0}
    lineup_journal.settle(SEASON, 8, points, tmp_path / "j", schedule=sched)
    doc = json.loads(path.read_text())
    assert doc["realised"]["set"] == {"points": 73.0, "n_no_row": 0}
