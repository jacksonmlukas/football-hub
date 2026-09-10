"""The decision journal. Entries in the order they happened, never rewritten.

The property under test throughout is that a decision is a claim made when nobody knew the
answer, and a record that can be revised once the answer arrives is not a record of a decision.
"""
import datetime as dt
from pathlib import Path
from typing import TypedDict, Unpack

import numpy as np
import polars as pl
import pytest

from hub.season import journal, pool

AT = dt.datetime(2026, 9, 7, 12, 0)


class _Decision(TypedDict, total=False):
    """`journal.record`'s keyword surface, so the helper below can be typed.

    Merging two plain dicts and splatting the result widens every value to a union of
    everything either dict can hold, and the checker then reports one error per parameter --
    thirteen of them, for calls that are correct. Declaring the shape keeps the helper's
    ergonomics (twenty-one call sites pass only what they vary) while letting the checker see
    what is actually being passed.
    """

    season: int
    week: int
    kind: str
    chose: str
    fallback: str | None
    fallback_note: str | None
    market_price: float | None
    price_note: str | None
    expected_dollars: float | None
    chose_survives: float | None
    fallback_survives: float | None
    survival_given_up: float | None
    credits_before: float | None
    credits_after: float | None
    at: dt.datetime | None
    base: Path


def _decide(tmp_path: Path, **kw: Unpack[_Decision]) -> str:
    call: _Decision = {"season": 2026, "week": 1, "kind": "pick", "chose": "LAC", "at": AT,
                       "base": tmp_path}
    call.update(kw)
    # A pick owes the free alternative it was taken over, so a caller that says nothing about
    # the fallback gets the case where we took it -- which is the one that owes nothing else.
    if call["kind"] in journal.FALLBACK_KINDS and "fallback" not in kw:
        call["fallback"] = call["chose"]
    return journal.record(**call)


def _departure(tmp_path: Path, *, ours: float = 0.40, free: float = 0.42,
               **kw: Unpack[_Decision]) -> str:
    """A pick that left the free one, with the two survival figures its cost is made of."""
    call: _Decision = {"chose": "SF", "fallback": "KC", "chose_survives": ours,
                       "fallback_survives": free, "survival_given_up": free - ours}
    call.update(kw)
    return _decide(tmp_path, **call)


def test_a_decision_and_its_outcome_are_two_rows_joined_on_a_key(tmp_path):
    """Not one row rewritten. `store.write` refuses to overwrite a partition whose contents
    differ, so a decision rewritten with its outcome attached is exactly what it raises on --
    which is why they are two writes joined afterwards."""
    k = _decide(tmp_path)
    journal.settle(k, season=2026, week=1, survived=False, dollars=-20.0, base=tmp_path)
    got = journal.read(2026, base=tmp_path)
    assert got.height == 1
    assert got["key"][0] == k and got["survived"][0] is False and got["dollars"][0] == -20.0


def test_the_outcome_attaches_without_touching_the_decision(tmp_path):
    k = _decide(tmp_path, market_price=-320.0, expected_dollars=1.5)
    before = journal.read(2026, base=tmp_path)
    journal.settle(k, season=2026, week=1, survived=True, dollars=0.0, base=tmp_path)
    after = journal.read(2026, base=tmp_path)
    assert after.height == 1
    for col in ("chose", "market_price", "expected_dollars", "at"):
        assert before[col][0] == after[col][0], col
    assert after["survived"][0] is True


def test_a_decision_without_an_outcome_still_reads_as_a_decision(tmp_path):
    """A left join. Dropping it would make a week that has not been played read as a week
    nobody decided anything in."""
    _decide(tmp_path)
    got = journal.read(2026, base=tmp_path)
    assert got.height == 1 and got["survived"][0] is None


def test_two_decisions_in_one_week_do_not_collide(tmp_path):
    """A week holds a pick and, when that pick loses, a buyback. Sharing a partition name
    would make them one row and lose the first."""
    _decide(tmp_path, kind="pick", chose="LAC", at=AT)
    _decide(tmp_path, kind="buyback", chose="re-enter",
            at=AT + dt.timedelta(hours=6))
    got = journal.read(2026, base=tmp_path)
    assert got.height == 2
    assert sorted(got["kind"].to_list()) == ["buyback", "pick"]
    assert got["key"].n_unique() == 2


def test_rewriting_a_decision_under_its_own_name_is_refused(tmp_path):
    """The append-only guarantee, at the level `store` enforces it."""
    _decide(tmp_path, expected_dollars=1.0)
    with pytest.raises(FileExistsError):
        _decide(tmp_path, expected_dollars=99.0)


def test_an_unposted_market_records_its_absence_rather_than_a_zero(tmp_path):
    """A null price read as 0.0 is a claim nobody made -- and 0.0 is a real, meaningful price
    on a spread, which is what makes the confusion silent."""
    _decide(tmp_path, market_price=None, price_note="no line posted at decision time")
    got = journal.read(2026, base=tmp_path)
    assert got["market_price"][0] is None
    assert "no line posted" in got["price_note"][0]


def test_an_unknown_credit_balance_is_not_a_free_decision(tmp_path):
    """`credits_remaining` reports nothing at all before the first pull, so a missing reading
    on either side is unknown cost, not zero cost."""
    _decide(tmp_path, credits_before=500.0, credits_after=None)
    got = journal.read(2026, base=tmp_path)
    assert got["cost_credits"][0] is None
    assert "unknown" in got["cost_note"][0]


def test_a_known_balance_records_the_difference(tmp_path):
    _decide(tmp_path, credits_before=500.0, credits_after=497.0)
    got = journal.read(2026, base=tmp_path)
    assert got["cost_credits"][0] == pytest.approx(3.0)
    assert got["cost_note"][0] is None


def test_a_pick_that_matched_the_free_one_is_recorded_as_such(tmp_path):
    """`docs/method.md` rule 5: gate against the simplest thing that already works. A season
    of weeks where we took auto-pick's team anyway is a season the model earned nothing."""
    _decide(tmp_path, chose="LAC", fallback="LAC")
    _departure(tmp_path, week=2, at=AT + dt.timedelta(days=7))
    got = journal.read(2026, base=tmp_path).sort("week")
    assert got["matched_fallback"].to_list() == [True, False]


def test_unmatched_weeks_is_the_weeks_that_did_not_match(tmp_path):
    """It returned the complement of its own name -- the weeks that *did* match -- with a
    docstring and a test that both described the matched case, so the name, the prose and the
    assertion agreed with each other and disagreed with the code. Nothing called it, which is
    why this was a landmine rather than a figure that had been published upside down.

    Week 1 took the free pick, week 2 left it. Only week 2 is a week the model was asked to
    earn anything in."""
    _decide(tmp_path, week=1, chose="LAC", fallback="LAC")
    _departure(tmp_path, week=2, at=AT + dt.timedelta(days=7))
    assert journal.unmatched_weeks(2026, base=tmp_path) == [2]


def test_a_week_with_no_fallback_recorded_is_in_neither_list(tmp_path):
    """`matched_fallback` is null there, and a null is not a departure. A week nobody wrote
    the free pick down for is not a week the model beat it, and not one it matched."""
    _decide(tmp_path, week=1, chose="LAC", fallback="LAC")
    _decide(tmp_path, week=2, fallback=None,
            fallback_note="every team is spent; auto-pick had nothing to assign",
            at=AT + dt.timedelta(days=7))
    assert journal.unmatched_weeks(2026, base=tmp_path) == []


def test_a_decision_with_no_fallback_recorded_claims_neither(tmp_path):
    """Absent is not False. A week where nobody wrote down the free alternative is not a week
    the model beat it -- and, for a kind that has a free alternative every week, it has to say
    why there was none rather than simply not mentioning it."""
    _decide(tmp_path, fallback=None, fallback_note="every team is spent")
    got = journal.read(2026, base=tmp_path)
    assert got["matched_fallback"][0] is None
    assert got["fallback_note"][0] == "every team is spent"


def test_the_journal_is_queryable_by_season_and_by_week(tmp_path):
    _decide(tmp_path, week=1)
    _decide(tmp_path, week=2, at=AT + dt.timedelta(days=7))
    assert journal.read(2026, base=tmp_path).height == 2
    assert journal.read(2026, week=2, base=tmp_path)["week"].to_list() == [2]
    assert journal.read(2025, base=tmp_path).is_empty()


def test_an_empty_store_reads_as_an_empty_journal(tmp_path):
    """A fresh clone has no journal, which is not an error -- it is a season nobody has
    decided anything in yet."""
    got = journal.read(2026, base=tmp_path)
    assert got.is_empty()
    assert isinstance(got, pl.DataFrame)


# --- what a decision cost, which one rule makes mandatory ---------------------


def test_a_departure_from_the_free_pick_must_say_what_it_cost(tmp_path):
    """The column was optional and the obligation is not. `docs/decisions.md` registers the
    survivor contrarian threshold under ADR-0014, whose own logging duty is the week, the
    chalk pick, ours, and the probability cost accepted -- so a journal that accepts the entry
    anyway records a decision taken in breach of its rule, and records it as complete."""
    with pytest.raises(ValueError, match="ADR-0014"):
        _decide(tmp_path, chose="SF", fallback="KC", survival_given_up=None)


def test_zero_is_an_answer_and_not_an_omission(tmp_path):
    """Two plans that survive alike cost nothing to choose between. That is a finding; None
    is a blank. It is now a finding that has to be arrived at: the two figures it is the
    difference of are on the row beside it."""
    _departure(tmp_path, ours=0.41, free=0.41)
    got = journal.read(2026, base=tmp_path)
    assert got["survival_given_up"][0] == 0.0
    assert got["chose_survives"][0] == pytest.approx(0.41)


def test_taking_the_free_pick_needs_no_cost_recorded(tmp_path):
    """Nothing was given up, so there is nothing to record. The obligation binds departures."""
    _decide(tmp_path, chose="LAC", fallback="LAC")
    got = journal.read(2026, base=tmp_path)
    assert got["matched_fallback"][0] is True and got["survival_given_up"][0] is None


def test_what_a_departure_gave_up_is_read_back(tmp_path):
    _departure(tmp_path, ours=0.400, free=0.431)
    got = journal.read(2026, base=tmp_path)
    assert got["survival_given_up"][0] == pytest.approx(0.031)
    assert got["matched_fallback"][0] is False


def test_a_decision_that_cost_no_credits_records_a_zero(tmp_path):
    """Distinct from unknown. The balance was read on both sides and had not moved, which
    says the decision was made and cost nothing -- not that nobody looked."""
    _decide(tmp_path, credits_before=500.0, credits_after=500.0)
    got = journal.read(2026, base=tmp_path)
    assert got["cost_credits"][0] == 0.0
    assert got["cost_note"][0] is None


# --- the three ways past ADR-0014's check, each closed ------------------------
#
# The check tested that a float was present. It now tests that a survival cost is consistent:
# a cost is the difference between two survival figures, so both are named and the difference
# is checked against them.


def test_a_zero_nobody_computed_is_refused(tmp_path):
    """The escape the error message itself used to suggest. `0.0` alone says the two plans
    survive alike, which is a measurement -- and passing it with nothing behind it satisfied
    the check while recording no measurement at all."""
    with pytest.raises(ValueError, match="ADR-0014"):
        _decide(tmp_path, chose="SF", fallback="KC", survival_given_up=0.0)


def test_a_cost_that_disagrees_with_its_two_figures_is_refused(tmp_path):
    """Consistency, not presence. A cost that does not follow from the figures it is the
    difference of was not computed from them, whatever it is."""
    with pytest.raises(ValueError, match="not what the two survival figures say"):
        _decide(tmp_path, chose="SF", fallback="KC", chose_survives=0.40,
                fallback_survives=0.42, survival_given_up=0.31)


def test_a_pick_that_names_no_free_alternative_at_all_is_refused(tmp_path):
    """Evading the duty by silence rather than by a wrong number. Auto-pick assigns a team
    every week, so a pick has a free alternative and has to name it -- or say why it did
    not."""
    with pytest.raises(ValueError, match="free alternative"):
        _decide(tmp_path, chose="SF", fallback=None)


def test_a_kind_no_rule_has_heard_of_is_refused(tmp_path):
    """`kind` decides which obligations a row carries, so free text is a row that owes
    nothing: file the departure under a name nothing validates and the check never runs."""
    with pytest.raises(ValueError, match="not a journal kind"):
        _decide(tmp_path, kind="note", chose="SF", fallback="KC")
    assert set(journal.KINDS) == {"pick", "buyback"}


def test_a_survival_figure_outside_zero_and_one_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a probability"):
        _decide(tmp_path, chose="SF", fallback="KC", chose_survives=1.4,
                fallback_survives=0.42, survival_given_up=-0.98)


# --- settling against the key, which already says season and week -------------


def test_settling_a_key_nothing_wrote_is_refused(tmp_path):
    """An outcome against a key no decision carries joins to nothing. Written anyway, it sits
    in the store where no read will find it and the decision it meant reads unsettled."""
    k = _decide(tmp_path)
    with pytest.raises(KeyError, match="no decision named"):
        journal.settle(k[:-1] + "9", survived=True, base=tmp_path)


def test_settling_with_a_season_that_disagrees_with_the_key_is_refused(tmp_path):
    """They were free parameters beside a key that already encodes them, so an outcome filed
    under the wrong season went to a partition the right season never joins -- silently, and
    the decision stayed unsettled forever."""
    k = _decide(tmp_path)
    with pytest.raises(ValueError, match="disagrees with the key"):
        journal.settle(k, season=2025, survived=True, base=tmp_path)
    with pytest.raises(ValueError, match="disagrees with the key"):
        journal.settle(k, week=9, survived=True, base=tmp_path)


def test_an_outcome_needs_only_the_key_it_is_settling(tmp_path):
    """The season and week come off the key, which is the one place they can be wrong in
    exactly one way."""
    k = _decide(tmp_path)
    journal.settle(k, survived=True, dollars=0.0, base=tmp_path)
    got = journal.read(2026, base=tmp_path)
    assert got.height == 1 and got["survived"][0] is True
    assert journal.parse_key(k) == (2026, 1, "pick")


# --- the two smaller columns --------------------------------------------------


def test_a_credit_balance_that_rose_is_unknown_cost_rather_than_a_negative_one(tmp_path):
    """A quota top-up between the two readings. `credits_remaining` is a last-known balance,
    not a meter, so the difference goes negative -- in the one column whose whole purpose is
    that absent is not zero."""
    with pytest.raises(ValueError, match="top-up"):
        _decide(tmp_path, credits_before=100.0, credits_after=600.0)


def test_a_price_note_survives_alongside_a_price(tmp_path):
    """It was dropped the moment there was a price, so the note saying which source a price
    came from, or what was odd about it, could not be written at all."""
    _decide(tmp_path, market_price=-320.0, price_note="closing price, snapshot 18:04")
    got = journal.read(2026, base=tmp_path)
    assert got["market_price"][0] == pytest.approx(-320.0)
    assert got["price_note"][0] == "closing price, snapshot 18:04"


def test_a_string_that_is_not_a_key_is_refused_rather_than_parsed_loosely(tmp_path):
    """The key is what carries the season and week now, so a string that is not one cannot be
    read as though it were."""
    with pytest.raises(ValueError, match="not a journal key"):
        journal.parse_key("last tuesday's pick")


def test_a_season_nobody_decided_anything_in_has_no_unmatched_weeks(tmp_path):
    """A fresh clone, not an error. Nothing was chosen, so nothing departed from the free
    pick."""
    assert journal.unmatched_weeks(2026, base=tmp_path) == []


# --- ADR-0014's checks against a real `Weekly`, rather than against fixtures written to
# --- satisfy them. Every survival figure below comes out of the simulator.

def _hoard() -> pl.DataFrame:
    """KC is barely the best pick in week 1 and a near-lock in week 2, so the recommendation
    departs from the free pick -- which is the only case ADR-0014's duty binds."""
    rows = []
    for wk, games in {1: [("KC", "LV", 0.70), ("SF", "SEA", 0.68), ("BUF", "NYJ", 0.66)],
                      2: [("KC", "LV", 0.97), ("SF", "SEA", 0.55), ("BUF", "NYJ", 0.54)]
                      }.items():
        for i, (a, b, p) in enumerate(games):
            rows += [(wk, a, p, f"{wk}-{i}"), (wk, b, 1 - p, f"{wk}-{i}")]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows]})


def _weekly():
    return pool.weekly(_hoard(), [1, 2], week=1, entries=12, pot=420.0, trials=400,
                       rng=np.random.default_rng(0))


def test_the_duty_is_discharged_against_a_real_weekly_and_the_signs_agree(tmp_path):
    """The check no caller was exercising, run on figures the simulator produced.

    `Weekly.given_up` is `fallback.survives - recommend.survives` and `_check_adr_0014`
    recomputes exactly that from the two figures on the row: the sign conventions agree, and
    a departure that *gains* survival is written as a negative cost rather than clamped."""
    w = _weekly()
    assert not w.matched and w.given_up < 0      # the departure survives better here
    k = journal.record_weekly(w, season=2026, at=AT, base=tmp_path)

    got = journal.read(2026, base=tmp_path)
    row = got.filter(pl.col("key") == k).to_dicts()[0]
    assert row["chose"] == w.recommend and row["fallback"] == w.fallback
    assert row["survival_given_up"] == pytest.approx(w.given_up)
    assert (row["fallback_survives"] - row["chose_survives"]
            == pytest.approx(row["survival_given_up"]))
    assert journal.unmatched_weeks(2026, base=tmp_path) == [1]


def test_the_logged_cost_is_not_the_quantity_the_rule_fires_on(tmp_path):
    """The finding of #204, pinned so it cannot be quietly reconciled later.

    ADR-0014 adopts the contrarian threshold as "the win-probability cost is under ~8pp" and
    `docs/decisions.md` logs "the probability cost accepted". `survival_given_up` is a
    season-survival difference, and on this grid the two disagree in sign as well as scale --
    the chalk is 2.0pp better on the week and 27.8pp worse over the season. That is the thesis
    of the survivor plan working as intended, not a defect in either number, which is exactly
    why writing one under the other's name would be the error.

    **The season figure moved under #151** and this is the record of it: it was -9.9pp while
    our own entry played out the rest of the season under the *field's* sampling rule, which
    took a near-random team in week 2 whichever team it had spent in week 1. Playing the plan
    instead, week 2 takes KC at 0.97 behind an SF pick and SF at 0.55 behind a KC one, so the
    gap widens to the 0.70x0.55 - 0.68x0.97 = -0.275 the grid was built to produce, plus the
    sampling error of 400 trials. The disagreement this test is about is unchanged in sign and
    is now larger; what it cost to leave in place is the difference between the two numbers."""
    w = _weekly()
    chose = next(c for c in w.candidates if c.team == w.recommend)
    fb = next(c for c in w.candidates if c.is_fallback)

    week_cost = fb.win_prob - chose.win_prob                # what the ADR thresholds on
    assert week_cost == pytest.approx(0.02, abs=1e-9)       # 2.0pp, and under ~8pp
    assert w.given_up == pytest.approx(-0.2779, abs=5e-4)   # -27.8pp, the other direction

    journal.record_weekly(w, season=2026, at=AT, base=tmp_path)
    row = journal.read(2026, base=tmp_path).to_dicts()[0]
    # The taken team's own price is on the row; the chalk's is not, so the cost the rule is
    # stated in cannot be reconstructed from the journal.
    assert row["market_price"] == pytest.approx(chose.win_prob)
    assert "fallback_price" not in row and "win_prob_given_up" not in row


def test_an_overridden_pick_is_costed_against_what_was_entered(tmp_path):
    """`Weekly.given_up` is a cost against the *recommendation*. Forwarded blindly for an
    operator who took something else, it files a comparison nobody made -- and the invariant
    catches it, which is the check doing real work rather than restating its inputs."""
    w = _weekly()
    other = next(c for c in w.candidates
                 if c.team not in (w.recommend, w.fallback))
    k = journal.record_weekly(w, season=2026, chose=other.team, at=AT, base=tmp_path)
    row = journal.read(2026, base=tmp_path).filter(pl.col("key") == k).to_dicts()[0]
    assert row["chose"] == other.team
    assert row["chose_survives"] == pytest.approx(other.survives)
    assert row["survival_given_up"] != pytest.approx(w.given_up)


def test_a_team_the_week_never_priced_is_refused_rather_than_costed(tmp_path):
    """A row whose survival figures came from a candidate nobody valued would satisfy the
    ADR-0014 check and mean nothing -- the fourth way past a duty that is about content."""
    with pytest.raises(ValueError, match="not one of the teams this week priced"):
        journal.record_weekly(_weekly(), season=2026, chose="MIA", base=tmp_path)
