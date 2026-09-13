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
    fallback_price: float | None
    week_cost: float | None
    price_note: str | None
    expected_dollars: float | None
    chose_survives: float | None
    fallback_survives: float | None
    survival_given_up: float | None
    credits_before: float | None
    credits_after: float | None
    pool_digest: str | None
    grid_digest: str | None
    seed: int | None
    trials: int | None
    entries: int | None
    pot: float | None
    outlay: float | None
    plan_source: str | None
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
               price: float = 0.68, free_price: float = 0.70,
               **kw: Unpack[_Decision]) -> str:
    """A pick that left the free one, with the two survival figures its season cost is made
    of and the two prices its week cost is made of (#209)."""
    call: _Decision = {"chose": "SF", "fallback": "KC", "chose_survives": ours,
                       "fallback_survives": free, "survival_given_up": free - ours,
                       "market_price": price, "fallback_price": free_price,
                       "week_cost": free_price - price}
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
    k = _decide(tmp_path, market_price=0.76, expected_dollars=1.5)
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
        _departure(tmp_path, ours=0.40, free=0.42, survival_given_up=0.31)


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
        _departure(tmp_path, ours=1.4, free=0.42)


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
    _decide(tmp_path, market_price=0.76, price_note="closing price, snapshot 18:04")
    got = journal.read(2026, base=tmp_path)
    assert got["market_price"][0] == pytest.approx(0.76)
    assert got["price_note"][0] == "closing price, snapshot 18:04"


def test_a_market_price_outside_the_unit_interval_is_refused_on_every_row(tmp_path):
    """One unit per column (#239). `market_price` is a win probability, because `week_cost`
    is a difference of it and `fallback_price` and is only a cost when both are one. The
    range check fired only once a `fallback_price` stood beside it, so a matched pick could
    carry 5.0 there and a moneyline row and a probability row were indistinguishable in a
    query. It fires whether or not the free pick's price is on the row."""
    with pytest.raises(ValueError, match=r"market_price=5\.0 is not a probability"):
        _decide(tmp_path, market_price=5.0)
    with pytest.raises(ValueError, match=r"market_price=5\.0 is not a probability"):
        _departure(tmp_path, price=5.0, free_price=0.70, week_cost=0.70 - 5.0)
    assert journal.read(2026, base=tmp_path).is_empty()


def test_a_moneyline_is_converted_before_it_is_written_and_says_so(tmp_path):
    """The other unit, and where it lands (#239). An American price is not a probability
    and is refused as one; the caller that has a moneyline converts it through
    `hub.models.props.implied` and records the source in `price_note`, so the number on
    the row is the same unit `record_weekly` writes and the note says how it got there."""
    from hub.models.props import implied

    with pytest.raises(ValueError, match=r"market_price=-320\.0 is not a probability"):
        _decide(tmp_path, market_price=-320.0)
    p = implied(-320.0)
    assert p is not None and p == pytest.approx(320 / 420)
    _decide(tmp_path, market_price=p, price_note="implied from moneyline -320, vig in")
    got = journal.read(2026, base=tmp_path)
    assert got["market_price"][0] == pytest.approx(320 / 420)
    assert 0.0 <= got["market_price"][0] <= 1.0
    assert got["price_note"][0] == "implied from moneyline -320, vig in"


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


def test_both_costs_are_logged_and_the_threshold_quantity_is_the_weeks(tmp_path):
    """The finding of #204, settled by #209: the journal logs both quantities under distinct
    names, and the ADR's threshold is on the week.

    ADR-0014 adopts the contrarian threshold as "the win-probability cost is under ~8pp".
    `survival_given_up` is a season-survival difference, and on this grid the two disagree
    in sign as well as scale -- the chalk is 2.0pp better on the week and 27.8pp worse over
    the season. That is the thesis of the survivor plan working as intended, not a defect in
    either number, which is exactly why writing one under the other's name was the error:
    `week_cost` is now the column the rule's duty is discharged by, and the season figure
    stays beside it as the argument for the plan.

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
    # Both prices are on the row and the week cost is their difference, so the ~8pp the
    # ADR conditions adoption on is recoverable from the journal alone.
    assert row["market_price"] == pytest.approx(chose.win_prob)
    assert row["fallback_price"] == pytest.approx(fb.win_prob)
    assert row["week_cost"] == pytest.approx(week_cost, abs=1e-9)
    assert row["week_cost"] == pytest.approx(row["fallback_price"] - row["market_price"])
    # And the two disagree in sign on this row, which is the plan's thesis, kept.
    assert row["week_cost"] > 0 > row["survival_given_up"]


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


# --- a row carries what it takes to reproduce its own figure (#162) ------------
#
# The schema carried the dollar figure and the survival given up and none of the provenance,
# so a row could not be re-derived or told apart from one produced under different rules.
# After #160 the pool digest exists and our plan is an input, so a row can name its rules,
# its board, its seed and its trials -- and a re-run from those has to land on the figure.


def test_a_row_carries_what_it_takes_to_run_its_figure_again(tmp_path):
    """The first two criteria at once: the provenance is on the row, and re-running from it
    reproduces the figure *exactly* -- `==`, not approximately, because the seed and the
    trial count are what make two runs the same run."""
    from hub.config import PoolConfig, pool_digest
    w = _weekly()
    k = journal.record_weekly(w, season=2026, at=AT, base=tmp_path)
    row = journal.read(2026, base=tmp_path).filter(pl.col("key") == k).to_dicts()[0]

    assert row["pool_digest"] == pool_digest(PoolConfig())
    assert row["grid_digest"] == pool.grid_digest(_hoard())
    assert row["seed"] == w.seed and row["trials"] == 400
    assert row["entries"] == 12 and row["pot"] == 420.0 and row["outlay"] == 0.0
    assert row["plan_source"] and "optimiser" in row["plan_source"]
    # Every re-run column off `Weekly` is filled; the field's digest is the one the caller
    # that read the pool state supplies, and this week's field was stated by hand (#280).
    assert all(row[c] is not None for c in journal.RERUN_COLUMNS if c != "pool_state_digest")
    assert row["pool_state_digest"] is None

    # A different generator on purpose: `_weekly` drew its seed from `default_rng(0)`, and a
    # re-run that happened to draw the same one would reproduce the figure with `seed`
    # ignored. Only the row's own seed may carry the trials across.
    again = pool.weekly(_hoard(), [1, 2], week=row["week"], entries=row["entries"],
                        pot=row["pot"], outlay=row["outlay"], trials=row["trials"],
                        rng=np.random.default_rng(999), seed=row["seed"])
    got = next(c for c in again.candidates if c.team == row["chose"])
    assert got.expected_dollars == row["expected_dollars"]
    assert got.survives == row["chose_survives"]
    assert again.seed == row["seed"] and again.pool_digest == row["pool_digest"]


def test_a_double_pick_week_is_recorded_with_both_teams_and_priced_as_their_product(tmp_path):
    """#256's journal half. HOARD moved onto weeks 13 and 14, which the default rules make
    double: the row's `chose` spells both teams, its `market_price` is the pair's win
    probability on the week -- the product, the unit `week_cost` is stated in -- and an
    operator's `SF+KC` is the same pick as the priced `KC+SF`."""
    grid = _hoard().with_columns(pl.col("week") + 12)
    w = pool.weekly(grid, [13, 14], week=13, entries=12, pot=420.0, trials=50, seed=7)
    assert len(pool.pick_teams(w.recommend)) == 2 and w.fallback == "KC+SF"
    k = journal.record_weekly(w, season=2026, chose="SF+KC", at=AT, base=tmp_path)
    row = journal.read(2026, base=tmp_path).filter(pl.col("key") == k).to_dicts()[0]
    assert pool.pick_teams(row["chose"]) == ("KC", "SF")
    assert row["fallback"] == "KC+SF" and row["matched_fallback"] is True
    assert row["market_price"] == pytest.approx(0.70 * 0.68)
    assert row["fallback_price"] == pytest.approx(0.70 * 0.68) and row["week_cost"] == 0.0
    # A pair nobody priced -- KC and its own opponent -- is refused like any other stranger.
    with pytest.raises(ValueError, match="not one of the teams this week priced"):
        journal.record_weekly(w, season=2026, chose="KC+LV", at=AT, base=tmp_path)


def test_a_row_names_the_field_it_was_priced_against_and_re_runs_against_the_archive(tmp_path):
    """#280. The re-run columns carried the rules, the board, the seed and the trials, and
    not the field -- the live count, the pot, our Ledger -- so a week could be re-derived
    against whatever the pool host said *now*. The row carries `pool_state_digest`; the
    archive resolves it to the field as it stood; and the figure re-run against that field
    lands exactly, where the current field gives another figure."""
    import datetime as dt

    from hub.fetch import pool as fetch_pool
    then = fetch_pool.PoolState(season=2026, week=1, field_size=12, pot=420.0, entries=tuple(
        fetch_pool.Entry(index=i, alive=True, used=()) for i in range(12)))
    fetch_pool.write_state(then, tmp_path, when=dt.datetime(2026, 9, 9, 9, 0, tzinfo=dt.UTC))
    digest = fetch_pool.state_digest(then)
    w = pool.weekly(_hoard(), [1, 2], week=1, entries=then.alive, pot=then.pot,
                    ledger=then.entries[0].used, trials=50, seed=11)
    k = journal.record_weekly(w, season=2026, at=AT, base=tmp_path, pool_state_digest=digest)
    row = journal.read(2026, base=tmp_path).filter(pl.col("key") == k).to_dicts()[0]
    assert row["pool_state_digest"] == digest
    assert "pool_state_digest" in journal.RERUN_COLUMNS

    # The field moves on: three out, the pot grown, KC spent -- the current state.
    now = fetch_pool.PoolState(season=2026, week=2, field_size=12, pot=480.0, entries=(
        fetch_pool.Entry(index=0, alive=True, used=("KC",)),
        *(fetch_pool.Entry(index=i, alive=i < 9, used=("SF",)) for i in range(1, 12))))
    fetch_pool.write_state(now, tmp_path, when=dt.datetime(2026, 9, 16, 9, 0, tzinfo=dt.UTC))
    assert fetch_pool.read_state(tmp_path) == now
    field = fetch_pool.archived_state(row["pool_state_digest"], season=2026, base=tmp_path)
    assert field is not None and field == then and field != now

    again = pool.weekly(_hoard(), [1, 2], week=row["week"], entries=field.alive, pot=field.pot,
                        ledger=field.entries[0].used, outlay=row["outlay"],
                        trials=row["trials"], seed=row["seed"], rng=np.random.default_rng(3))
    got = next(c for c in again.candidates if c.team == row["chose"])
    assert got.expected_dollars == row["expected_dollars"]
    assert (again.entries, again.pot) == (row["entries"], row["pot"])
    current = pool.weekly(_hoard(), [1, 2], week=row["week"], entries=now.alive, pot=now.pot,
                          ledger=now.entries[0].used, trials=row["trials"], seed=row["seed"])
    assert row["chose"] not in {c.team for c in current.candidates} or next(
        c for c in current.candidates if c.team == row["chose"]
    ).expected_dollars != row["expected_dollars"]

    # A row priced against a field stated by hand carries none, and says so by the null.
    bare = journal.record_weekly(w, season=2026, at=AT + dt.timedelta(hours=1), base=tmp_path)
    assert journal.read(2026, base=tmp_path).filter(
        pl.col("key") == bare)["pool_state_digest"][0] is None


def test_two_rows_under_different_pool_rules_are_told_apart_by_the_journal_alone(tmp_path):
    """Nothing outside the journal is read. The two rows carry the same week, the same pick
    and the same trials, and differ only in the digest of the rules they were priced under."""
    from hub.config import PoolConfig
    split = pool.weekly(_hoard(), [1, 2], week=1, entries=12, pot=420.0, trials=50,
                        pool=PoolConfig(co_survivor_rule="split"), seed=7)
    roll = pool.weekly(_hoard(), [1, 2], week=1, entries=12, pot=420.0, trials=50,
                       pool=PoolConfig(co_survivor_rule="rollover"), seed=7)
    journal.record_weekly(split, season=2026, at=AT, base=tmp_path)
    journal.record_weekly(roll, season=2026, at=AT + dt.timedelta(hours=1), base=tmp_path)
    got = journal.read(2026, base=tmp_path)
    assert got.height == 2
    assert got["pool_digest"].n_unique() == 2
    assert got["grid_digest"].n_unique() == 1 and got["seed"].n_unique() == 1


def test_a_row_written_before_provenance_existed_reads_as_one_that_cannot_be_rerun(tmp_path):
    """The fourth criterion. A partition written under the old schema has none of the
    provenance columns; it still reads, and every one of them is null on it -- not a
    default, not a value nobody measured. A store holding *only* such rows has no column to
    union, which is the case `read` fills rather than fails."""
    from hub import store
    old = {c: t for c, t in journal.SCHEMA.items() if c not in journal.RERUN_COLUMNS}
    k = journal.key(2026, 1, "pick", AT)
    row = pl.DataFrame({c: [None] for c in old}, schema=old).with_columns(
        pl.lit(k).alias("key"), pl.lit("pick").alias("kind"), pl.lit(2026).alias("season"),
        pl.lit(1).alias("week"), pl.lit(AT).alias("at"), pl.lit("LAC").alias("chose"),
        pl.lit("LAC").alias("fallback"), pl.lit(True).alias("matched_fallback"))
    store.write(row, journal.TABLE, journal.LEAGUE, 2026, 1, name=k, base=tmp_path)

    got = journal.read(2026, base=tmp_path)
    assert got.height == 1 and got["chose"][0] == "LAC"
    assert list(got.columns[:len(journal.SCHEMA)]) == list(journal.SCHEMA)
    assert all(got[c][0] is None for c in journal.RERUN_COLUMNS)


def test_provenance_comes_whole_or_not_at_all(tmp_path):
    """A row naming the rules and not the seed reads as reproducible to a query on
    `pool_digest` and is not. Either both are present or the row says it cannot be
    re-derived."""
    with pytest.raises(ValueError, match="provenance has to come whole"):
        _decide(tmp_path, pool_digest="deadbeef", seed=None)
    with pytest.raises(ValueError, match="provenance has to come whole"):
        _decide(tmp_path, pool_digest=None, seed=3)


def test_provenance_means_every_rerun_column_not_just_the_two(tmp_path):
    """Review finding on #162: the digest and the seed were the only pair checked, so a row
    carrying both and none of `grid_digest`, `trials`, `entries`, `pot` or `outlay` was
    written, and `report` then printed it as re-derivable. It cannot be re-run without the
    board, the trial count or the stakes, so it is refused like any other partial row."""
    def whole(**over):
        kw: dict = {"pool_digest": "deadbeef", "grid_digest": "cafe", "seed": 3,
                    "trials": 100, "entries": 21, "pot": 420.0, "outlay": 20.0}
        kw.update(over)
        return kw
    _decide(tmp_path, **whole())
    for missing in whole():
        with pytest.raises(ValueError, match="provenance has to come whole"):
            _decide(tmp_path, **whole(**{missing: None}))
    # A buyback has no plan, so `plan_source` is the one rerun column that may be absent
    # beside the other seven.
    _decide(tmp_path, week=2, **whole(plan_source=None))


# --- ADR-0014's threshold quantity, logged under its own name (#209) -----------------------


def test_a_departure_without_the_weeks_cost_is_refused(tmp_path):
    """The threshold is stated in win probability on the week, so a departure that records
    only the season figure has not discharged the duty -- however consistent that figure is
    with its own two survival numbers."""
    with pytest.raises(ValueError, match="without recording the week's cost"):
        _decide(tmp_path, chose="SF", fallback="KC", chose_survives=0.40,
                fallback_survives=0.42, survival_given_up=0.02)


def test_a_week_cost_that_disagrees_with_its_two_prices_is_refused(tmp_path):
    """The same shape as the survival trio: a cost is a difference, both prices are on the
    row, and the cost is checked against them."""
    with pytest.raises(ValueError, match="not what the two prices say"):
        _departure(tmp_path, price=0.68, free_price=0.70, week_cost=0.08)
    with pytest.raises(ValueError, match=r"fallback_price=1\.7 is not a probability"):
        _departure(tmp_path, price=0.68, free_price=1.7, week_cost=1.02)


def test_the_column_the_adr_names_as_its_threshold_is_the_one_the_journal_writes(tmp_path):
    """The drift test the ticket asks for. ADR-0014's dated line names the column its
    threshold is stated in; that column has to exist in the schema, be the difference of
    the two prices `record_weekly` writes, and not be the season figure. If the ADR is
    re-pointed at another column, or the column is renamed, or its arithmetic changes,
    this fails -- and the two quantities cannot silently drift back into one name."""
    import pathlib
    import re
    adr = next((pathlib.Path(__file__).resolve().parents[2] / "docs" / "adr").glob(
        "0014-*.md")).read_text()
    # Prose wraps, and the amendment is a blockquote, so the phrase may straddle a `> `.
    adr = re.sub(r"\s*\n>?\s*", " ", adr)
    named = re.search(r"threshold quantity is `journal\.(\w+)`", adr)
    assert named, "ADR-0014 names no journal column as its threshold quantity"
    column = named.group(1)
    assert column == "week_cost" and column in journal.SCHEMA
    assert column != "survival_given_up"
    assert "the fallback's win probability minus the chosen team's" in adr

    w = _weekly()
    journal.record_weekly(w, season=2026, at=AT, base=tmp_path)
    row = journal.read(2026, base=tmp_path).to_dicts()[0]
    chose = next(c for c in w.candidates if c.team == w.recommend)
    fb = next(c for c in w.candidates if c.is_fallback)
    assert row[column] == pytest.approx(fb.win_prob - chose.win_prob, abs=1e-9)
    assert row[column] != pytest.approx(row["survival_given_up"])


def test_the_journal_report_prints_both_costs_under_their_own_names(tmp_path):
    journal.record_weekly(_weekly(), season=2026, at=AT, base=tmp_path)
    lines = journal.report(journal.read(2026, base=tmp_path))
    assert "week cost" in lines[0] and "given up" in lines[0]
    assert "+2.0pp" in lines[1] and "-27.8pp" in lines[1]
