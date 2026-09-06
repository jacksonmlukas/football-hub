"""The decision journal. Entries in the order they happened, never rewritten.

The property under test throughout is that a decision is a claim made when nobody knew the
answer, and a record that can be revised once the answer arrives is not a record of a decision.
"""
import datetime as dt

import polars as pl
import pytest

from hub.season import journal

AT = dt.datetime(2026, 9, 7, 12, 0)


def _decide(tmp_path, **kw):
    base = {"season": 2026, "week": 1, "kind": "pick", "chose": "LAC", "at": AT,
            "base": tmp_path}
    return journal.record(**{**base, **kw})


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
    _decide(tmp_path, week=2, chose="SF", fallback="KC", at=AT + dt.timedelta(days=7))
    got = journal.read(2026, base=tmp_path).sort("week")
    assert got["matched_fallback"].to_list() == [True, False]
    assert journal.unmatched_weeks(2026, base=tmp_path) == [1]


def test_a_decision_with_no_fallback_recorded_claims_neither(tmp_path):
    """Absent is not False. A week where nobody wrote down the free alternative is not a week
    the model beat it."""
    _decide(tmp_path, fallback=None)
    got = journal.read(2026, base=tmp_path)
    assert got["matched_fallback"][0] is None


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
