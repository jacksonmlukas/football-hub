"""The pool as a whole. These pin the correlation, because that is what decides the answer.

A field playing chalk dies together, and how together it dies is what sets the week the pool
ends -- which is what a buyback is priced against. Draw the games per entry instead of per
game and eliminations become independent, the field thins smoothly, and the contest stretches
past anything real. Most of what follows is that one property, tested from angles that would
each fail differently if it broke.
"""
import math

import numpy as np
import polars as pl
import pytest

from hub.config import PoolConfig
from hub.season import pool


def _grid(rows):
    """rows: (week, team_a, team_b, p_a) -- one row per fixture, expanded to two grid rows."""
    out = []
    for i, (w, a, b, p) in enumerate(rows):
        gid = f"{w}-{i}"
        out += [(w, a, float(p), gid), (w, b, 1.0 - float(p), gid)]
    return pl.DataFrame({"week": [r[0] for r in out], "team": [r[1] for r in out],
                         "win_prob": [r[2] for r in out], "game_id": [r[3] for r in out]})


def test_a_team_and_its_opponent_cannot_both_win():
    """The sharpest form of "one draw per game". Two entries forced onto opposite sides of one
    fixture: exactly one survives, every trial, so the pool never ends in week 1. Drawn per
    team instead, both would lose a quarter of the time and the pool would end immediately."""
    g = _grid([(1, "KC", "LV", 0.5)])
    out = pool.simulate(g, [1], entries=2, ledgers=[{"LV"}, {"KC"}], trials=400,
                        rng=np.random.default_rng(0))
    assert out.ending_week.get(1, 0.0) == 0.0
    assert out.alive_by_week[1] == 1.0


def test_two_entries_on_one_team_share_its_outcome():
    """The other side of the same property. Both entries forced onto KC: they survive together
    or die together, so the pool ends in week 1 exactly when KC loses -- about half the time.
    Independent draws would end it only when *both* lost, about a quarter."""
    g = _grid([(1, "KC", "LV", 0.5)])
    out = pool.simulate(g, [1], entries=2, ledgers=[{"LV"}, {"LV"}], trials=800,
                        rng=np.random.default_rng(0))
    assert out.ending_week.get(1, 0.0) == pytest.approx(0.5, abs=0.06)


def test_the_field_thins_rather_than_dying_as_one_block():
    """Why rivals are sampled and not deterministic. Under one deterministic rule every entry
    picks the same team every week, so the count goes 21 to 0 in a single step and there is no
    partially-thinned field to price a buyback against."""
    g = _grid([(w, a, b, p) for w in (1, 2, 3)
               for a, b, p in (("KC", "LV", 0.8), ("SF", "SEA", 0.7),
                               ("BUF", "NYJ", 0.75), ("DAL", "NYG", 0.6))])
    out = pool.simulate(g, [1, 2, 3], entries=21, trials=300, rng=np.random.default_rng(1))
    assert 0 < out.alive_by_week[1] < 21
    assert out.alive_by_week[3] < out.alive_by_week[1]


def test_an_entry_never_picks_a_team_it_has_already_used():
    """The no-repeat rule, at the level it is enforced."""
    wk = pool.weeks_from_grid(_grid([(1, "KC", "LV", 0.8), (1, "SF", "SEA", 0.7)]), [1])[0]
    rng = np.random.default_rng(0)
    for _ in range(50):
        got = pool._pick(rng, wk, {"KC", "SF"}, 1)
        assert got is not None and got[0] in {"LV", "SEA"}


def test_an_entry_that_cannot_cover_the_week_is_eliminated():
    """Elimination by the ledger rather than by losing. With every team in this week already
    spent there is no legal pick, and that is the 24-distinct-team constraint biting."""
    wk = pool.weeks_from_grid(_grid([(1, "KC", "LV", 0.8)]), [1])[0]
    assert pool._pick(np.random.default_rng(0), wk, {"KC", "LV"}, 1) is None


def test_co_survivors_is_a_distribution_not_a_single_bucket():
    """How many people are left decides how a pot splits, so "someone survived" is not an
    answer. Reported per count."""
    g = _grid([(w, a, b, p) for w in (1, 2)
               for a, b, p in (("KC", "LV", 0.9), ("SF", "SEA", 0.85),
                               ("BUF", "NYJ", 0.8), ("DAL", "NYG", 0.8))])
    out = pool.simulate(g, [1, 2], entries=12, trials=300, rng=np.random.default_rng(2))
    assert len(out.co_survivors) > 1
    assert sum(out.co_survivors.values()) + sum(out.ending_week.values()) == pytest.approx(1.0)


def test_a_bigger_field_lasts_longer():
    """More entries is more chances somebody is still alive, so the contest runs later. A
    directional check: the point value is not the claim."""
    g = _grid([(w, a, b, p) for w in (1, 2, 3)
               for a, b, p in (("KC", "LV", 0.7), ("SF", "SEA", 0.65),
                               ("BUF", "NYJ", 0.6), ("DAL", "NYG", 0.55))])
    small = pool.simulate(g, [1, 2, 3], entries=2, trials=400, rng=np.random.default_rng(3))
    big = pool.simulate(g, [1, 2, 3], entries=25, trials=400, rng=np.random.default_rng(3))
    assert sum(big.ending_week.values()) < sum(small.ending_week.values())


def test_a_pool_of_one_ends_when_that_entry_does():
    g = _grid([(1, "KC", "LV", 0.0)])          # KC cannot win, so the only entry must lose
    out = pool.simulate(g, [1], entries=1, ledgers=[{"LV"}], trials=50,
                        rng=np.random.default_rng(0))
    assert out.ending_week[1] == 1.0
    assert out.co_survivors == {}


def test_the_same_seed_reproduces_the_same_outcome():
    """Sorted iteration everywhere a set could decide it. `docs/weekly-blend-gate.md` records
    an interval that moved between identical runs because `.unique()` order fed a bootstrap."""
    g = _grid([(w, a, b, p) for w in (1, 2)
               for a, b, p in (("KC", "LV", 0.8), ("SF", "SEA", 0.7))])
    kw = {"entries": 8, "trials": 200}
    a = pool.simulate(g, [1, 2], rng=np.random.default_rng(7), **kw)
    b = pool.simulate(g, [1, 2], rng=np.random.default_rng(7), **kw)
    assert a == b


def test_a_double_pick_week_takes_two_teams_from_two_fixtures():
    """Weeks 13-18 need two, both of which must win. Read from the pool config, so a rule
    correction is a re-run.

    **Asserted on the game identities and not on the team names**, which is #156. Two distinct
    teams is satisfied by KC and LV -- the two sides of one fixture -- so the old form of this
    test passed against a sampler that handed an entry a week it had already lost. The names
    differ in exactly the case the rule exists to forbid, which is why they cannot be the
    thing checked.
    """
    g = _grid([(13, "KC", "LV", 0.8), (13, "SF", "SEA", 0.7)])
    wk = pool.weeks_from_grid(g, [13], PoolConfig(double_pick_weeks=(13,)))[0]
    assert wk.picks == 2
    rng = np.random.default_rng(0)
    for _ in range(200):
        got = pool._pick(rng, wk, set(), 2)
        assert got is not None and len(set(got)) == 2
        assert len({wk.fixture[t] for t in got}) == 2


def test_an_entry_that_cannot_field_two_fixtures_is_eliminated_rather_than_given_both_sides():
    """The other half of the same rule, from the ledger rather than from the draw.

    Two teams are left and they are the two sides of one game, so this week cannot be covered:
    one of the pair loses whatever is picked. `None` is elimination by the no-repeat rule --
    the entry spent too much to cover the week -- and it is a different answer from picking and
    losing, which the count above it distinguishes: there are two legal *teams* here and one
    legal *fixture*, and only the second is what a week can be covered from.
    """
    g = _grid([(13, "KC", "LV", 0.8), (13, "SF", "SEA", 0.7)])
    wk = pool.weeks_from_grid(g, [13], PoolConfig(double_pick_weeks=(13,)))[0]
    avail = {t for t in wk.pickable if t not in {"SF", "SEA"}}
    assert avail == {"KC", "LV"}
    assert pool._pick(np.random.default_rng(0), wk, {"SF", "SEA"}, 2) is None
    # And one pick from that same fixture is still perfectly legal, so this is a statement
    # about covering two picks and not about the fixture being unusable.
    assert pool._pick(np.random.default_rng(0), wk, {"SF", "SEA"}, 1) is not None


def test_a_grid_without_a_game_key_is_refused():
    """Rather than falling back to per-team draws, which is the silent version of the bug this
    module exists to avoid."""
    g = _grid([(1, "KC", "LV", 0.8)]).drop("game_id")
    with pytest.raises(ValueError, match="game_id"):
        pool.weeks_from_grid(g, [1])


# --- a week the grid cannot price, which is not a week nobody survived --------


def _half(week, team, p, gid):
    """One side of a fixture with no opponent beside it -- what the grid holds when only one
    team in a game has a posted price. `weeks_from_grid` drops the fixture, and a week made
    entirely of these is a week with no games and no teams."""
    return pl.DataFrame({"week": [week], "team": [team], "win_prob": [float(p)],
                         "game_id": [gid]})


def test_a_week_nothing_prices_is_refused_rather_than_read_as_the_pool_ending():
    """The distinction this guard exists for, asserted on both sides of it.

    Two grids, and every entry is left with nothing in each. They are not the same fact. The
    first is a pool that ended in week 1 -- the only entry had to hold a team that cannot
    win -- and `ending_week[1] == 1.0` is the true answer. The second is a week where each
    fixture is priced on one side only, so there is no team to pick at all; the field is
    eliminated for want of a grid, and reporting that as an ending week names a week nobody
    played as the week the contest finished. Asserting only that something raises would not
    separate these, because before the guard *neither* raised and both returned 1.0."""
    ended = _grid([(1, "KC", "LV", 0.0)])       # KC cannot win, and our entry must hold KC
    out = pool.simulate(ended, [1], entries=1, ledgers=[{"LV"}], trials=50,
                        rng=np.random.default_rng(0))
    assert out.ending_week[1] == 1.0
    assert out.co_survivors == {}

    unpriced = pl.concat([_half(1, "KC", 0.6, "1-a"), _half(1, "SF", 0.7, "1-b")])
    with pytest.raises(pool.UnpricedWeek, match="no completely priced fixture") as e:
        pool.simulate(unpriced, [1], entries=1, trials=50, rng=np.random.default_rng(0))
    assert "2 fixtures priced on one side only" in str(e.value)


def test_a_double_pick_week_with_one_fixture_is_refused_rather_than_killing_the_field():
    """The same refusal as the week above, counted against the picks the week takes instead
    of against zero -- which is where the guard stopped and not where its reasoning did.

    One priced fixture in a double-pick week gives `_pick` two teams and forces it to take
    both sides of one game. One of them loses, so every entry in the field dies with
    certainty, and `ending_week` then names week 1 as the week the contest finished -- a week
    `survivor.coverage` calls missing and `survivor.solve` calls infeasible. Asserting only
    that something raises would not separate that from a pool that really ended, so the same
    grid is asserted on both sides of the rule: it is a legal week at one pick and a refusal
    at two."""
    g = _grid([(1, "KC", "LV", 0.85)])
    assert pool.weeks_from_grid(g, [1])[0].picks == 1     # fine when the week takes one

    dbl = PoolConfig(double_pick_weeks=(1,))
    with pytest.raises(pool.UnpricedWeek, match="takes 2 picks") as e:
        pool.simulate(g, [1], entries=4, pool=dbl, trials=50,
                      rng=np.random.default_rng(0))
    assert "1 completely priced fixture" in str(e.value)


def test_a_team_below_the_floor_is_drawn_but_never_handed_to_an_entry():
    """One rule for what may be taken, read by the simulator as well as by `auto_pick`.

    LV is priced at 1e-5. Its fixture still has to be *drawn* -- KC's win depends on it, and
    dropping the fixture would refuse to simulate a week the board has fully priced -- so LV
    stays in `teams` and in `prob`. It is not in `pickable`, because `auto_pick` and `weekly`
    both refuse it, and our own entry was being valued over seasons in which it took teams
    the pick side would never have allowed.

    Asserted at `_pick` rather than through `simulate`, because the two answers are
    indistinguishable downstream: an entry handed LV loses with probability 1 - 1e-5, so
    "eliminated for having no legal pick" and "eliminated holding a team it should never have
    been offered" both read out as the same ending week. The difference is only visible where
    the choice is made."""
    g = _grid([(1, "KC", "LV", 0.99999), (1, "SF", "SEA", 0.6)])
    wk = pool.weeks_from_grid(g, [1])[0]
    assert "LV" in wk.teams and "LV" in wk.prob
    assert wk.pickable == frozenset({"KC", "SF", "SEA"})
    assert pool.auto_pick(g, 1) == "KC"

    rng = np.random.default_rng(0)
    # An entry holding everything but LV has no legal pick, rather than one it may not take.
    assert pool._pick(rng, wk, {"KC", "SF", "SEA"}, 1) is None
    # And LV is never among the picks offered while other teams remain.
    assert all(pool._pick(rng, wk, {"KC"}, 1) != ["LV"] for _ in range(50))


def test_a_partially_priced_week_says_how_many_fixtures_it_dropped():
    """The week is still usable and is not what was asked for, and both have to be readable."""
    g = pl.concat([_grid([(1, "KC", "LV", 0.8)]), _half(1, "SF", 0.7, "1-solo")])
    wk = pool.weeks_from_grid(g, [1])[0]
    assert wk.dropped == 1
    assert wk.teams == ("KC", "LV")


def test_the_prices_and_the_team_list_come_from_the_same_fixtures():
    """They disagreed by construction: the team list counted completely priced fixtures while
    the prices were read off every row in the week, so a team no draw could ever return a
    result for still carried a weight."""
    g = pl.concat([_grid([(1, "KC", "LV", 0.8)]), _half(1, "SF", 0.99, "1-solo")])
    wk = pool.weeks_from_grid(g, [1])[0]
    assert set(wk.prob) == set(wk.teams) == {"KC", "LV"}
    assert {t for a, b, _ in wk.games for t in (a, b)} == set(wk.teams)


def test_fewer_starting_ledgers_than_entries_is_refused():
    """Matched to entries by position, so a short list is either an entry reading somebody
    else's spent teams or an index error thrown partway through a trial. Neither is a pool
    outcome, so it is refused before any trial runs."""
    g = _grid([(1, "KC", "LV", 0.8), (1, "SF", "SEA", 0.7)])
    with pytest.raises(ValueError, match="every one of them"):
        pool.simulate(g, [1], entries=3, ledgers=[{"KC"}], trials=10,
                      rng=np.random.default_rng(0))


# --- our own entry, carrying what it has already spent ------------------------
#
# The quantity a buyback is priced against. The failure to avoid is one-over-the-field, which
# is blind to the ledger: identical in week 2 and week 6, so a buyback figure built on it never
# moves with the teams already gone.


def test_a_fuller_ledger_is_worth_less():
    """The whole point of taking a ledger at all. Same field, same weeks, same fixtures -- the
    only difference is that this entry has already spent the teams worth having."""
    g = _grid([(w, a, b, p) for w in (1, 2, 3)
               for a, b, p in (("KC", "LV", 0.85), ("SF", "SEA", 0.8), ("BUF", "NYJ", 0.75))])
    kw = {"entries": 6, "trials": 600}
    fresh = pool.entry_outcome(g, [1, 2, 3], rng=np.random.default_rng(4), **kw)
    spent = pool.entry_outcome(g, [1, 2, 3], ledger=("KC", "SF", "BUF"),
                               rng=np.random.default_rng(4), **kw)
    assert spent.survives < fresh.survives
    assert spent.sole < fresh.sole


def test_the_entry_shares_outcomes_with_rivals_on_its_team():
    """Pinned to a closed form, because the alternative is arithmetically distinguishable.

    One fixture, KC at 0.9. Our ledger holds KC so we must take LV; the rival samples by win
    probability and takes KC nine times in ten. Exactly one side wins, so we survive when LV
    wins (0.1) and are alone when the rival was on KC (0.9): 0.09. Drawn per entry instead,
    the rival could lose while on the winning side and sole would be nearer 0.018."""
    g = _grid([(1, "KC", "LV", 0.9)])
    out = pool.entry_outcome(g, [1], entries=2, ledger=("KC",), trials=4000,
                             rng=np.random.default_rng(5))
    assert out.survives == pytest.approx(0.10, abs=0.02)
    assert out.sole == pytest.approx(0.09, abs=0.02)
    # And the pot splits when the rival came along: 0.1 * (0.9 * 1 + 0.1 * 1/2).
    assert out.share == pytest.approx(0.095, abs=0.02)


def test_share_sits_between_sole_and_survival():
    """Finishing level with two others is worth a third, not nothing and not everything."""
    g = _grid([(w, a, b, p) for w in (1, 2)
               for a, b, p in (("KC", "LV", 0.9), ("SF", "SEA", 0.85), ("BUF", "NYJ", 0.8))])
    out = pool.entry_outcome(g, [1, 2], entries=9, trials=600, rng=np.random.default_rng(6))
    assert 0 < out.sole <= out.share <= out.survives <= 1


def test_an_entry_that_cannot_cover_the_week_is_worth_nothing():
    """Elimination by the no-repeat ledger rather than by losing -- the 24-team constraint."""
    g = _grid([(1, "KC", "LV", 0.9)])
    out = pool.entry_outcome(g, [1], entries=3, ledger=("KC", "LV"), trials=100,
                             rng=np.random.default_rng(0))
    assert out.survives == 0.0 and out.sole == 0.0 and out.share == 0.0


def test_the_same_seed_and_ledger_reproduce_the_same_figure():
    g = _grid([(w, a, b, p) for w in (1, 2)
               for a, b, p in (("KC", "LV", 0.8), ("SF", "SEA", 0.7))])
    kw = {"entries": 5, "ledger": ("KC",), "trials": 300}
    a = pool.entry_outcome(g, [1, 2], rng=np.random.default_rng(11), **kw)
    b = pool.entry_outcome(g, [1, 2], rng=np.random.default_rng(11), **kw)
    assert a == b


# --- the buyback, which is an investment decision and not a survival one ------


def _season(weeks=(1, 2, 3)):
    return _grid([(w, a, b, p) for w in weeks
                  for a, b, p in (("KC", "LV", 0.85), ("SF", "SEA", 0.8),
                                  ("BUF", "NYJ", 0.75), ("DAL", "NYG", 0.7))])


def test_a_buyback_inherits_the_ledger_rather_than_starting_fresh():
    """Written before the arithmetic, because this is the part that gets implemented as a
    fresh entry by accident. The commissioner confirmed a re-entry keeps its used teams, so
    buying back in week 6 with six teams spent is a weaker entry than the same $20 in week 2 --
    and a figure that does not move with the ledger is the tell that it was priced as fresh."""
    g = _season()
    kw = {"live_entries": 8, "pot": 420.0, "trials": 400}
    early = pool.buyback(g, [1, 2, 3], week=1, ledger=(), rng=np.random.default_rng(3), **kw)
    late = pool.buyback(g, [1, 2, 3], week=1, ledger=("KC", "SF", "BUF"),
                        rng=np.random.default_rng(3), **kw)
    assert late.spent == 3 and early.spent == 0
    assert late.equity < early.equity


def test_equity_above_the_fee_recommends_buying_back():
    """A big pot and a thin field: the share is worth more than the $20 it costs."""
    b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=(), live_entries=2, pot=420.0,
                     trials=400, rng=np.random.default_rng(8))
    assert b.available and b.recommend
    assert b.net > 0 and b.equity > b.fee
    assert b.breakeven == pytest.approx(b.equity)


def test_equity_below_the_fee_recommends_against():
    """A small pot and a crowded field. The breakeven is still reported, because a decision
    without its margin is not a decision -- a net of -$0.40 and -$40 read the same otherwise."""
    b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=("KC", "SF"), live_entries=60,
                     pot=40.0, trials=400, rng=np.random.default_rng(8))
    assert b.available and not b.recommend
    assert b.net < 0 and b.breakeven == pytest.approx(b.equity)


def test_the_sign_of_net_is_the_recommendation():
    """Pinned so a negative figure can never be read as a positive one."""
    for pot, live in ((420.0, 2), (40.0, 60)):
        b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=(), live_entries=live, pot=pot,
                         trials=300, rng=np.random.default_rng(9))
        assert b.recommend == (b.net > 0)
        assert b.net == pytest.approx(b.equity - b.fee)


def test_a_buyback_past_the_deadline_is_unavailable_not_priced():
    """Reported rather than returned as a zero, which would read as a live decision that came
    out badly instead of an option that does not exist."""
    b = pool.buyback(_season(), [1, 2, 3], week=7, ledger=(), live_entries=8, pot=420.0,
                     trials=50, rng=np.random.default_rng(0))
    assert not b.available and not b.recommend
    assert "the last one that allows it" in b.reason


def test_rival_buybacks_move_the_pot_and_the_field_together():
    """Both sides, which is the point. Modelling only the fees would make every buyback look
    better than it is: the money they add comes with the people who added it."""
    kw = {"ledger": (), "live_entries": 8, "pot": 420.0, "trials": 300}
    alone = pool.buyback(_season(), [1, 2, 3], week=1, rival_buybacks=0,
                         rng=np.random.default_rng(10), **kw)
    crowd = pool.buyback(_season(), [1, 2, 3], week=1, rival_buybacks=3,
                         rng=np.random.default_rng(10), **kw)
    assert crowd.pot > alone.pot
    assert crowd.field > alone.field
    assert crowd.pot - alone.pot == pytest.approx(3 * alone.fee)
    assert crowd.field - alone.field == 3


def test_a_cap_of_zero_means_no_buyback_and_no_growth():
    """Zero is a rule, not an absence. Nobody re-enters, so the pot is the pot."""
    b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=(), live_entries=8, pot=420.0,
                     rival_buybacks=5, pool=PoolConfig(buyback_cap=0), trials=50,
                     rng=np.random.default_rng(0))
    assert not b.available
    assert b.pot == 420.0
    assert "cap is zero" in b.reason


def test_rival_buybacks_cannot_exceed_the_cap():
    b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=(), live_entries=8, pot=420.0,
                     rival_buybacks=99, pool=PoolConfig(buyback_cap=2), trials=50,
                     rng=np.random.default_rng(0))
    assert b.field == 8 + 1 + 2


def test_the_decision_reports_as_lines_with_the_breakeven_on_the_page():
    """Lines rather than prints, the reason `paired_report` is shaped that way: a block that
    prints cannot be composed, capped, or asserted on."""
    b = pool.buyback(_season(), [1, 2, 3], week=1, ledger=("KC",), live_entries=8, pot=420.0,
                     trials=200, rng=np.random.default_rng(2))
    lines = pool.report(b)
    assert any("breakeven" in ln for ln in lines)
    assert any("$" in ln for ln in lines)
    assert any("already spent" in ln for ln in lines)


# --- our own entry plays our plan, not the field's rule -----------------------
#
# Issue #151. Our entry is index 0 of the same arrays, and until this there was no branch for
# `i == 0`: it sampled under the rival rule, so every published figure answered "what is an
# entry worth to somebody who does not use this repo". The error runs one way and grows with
# the horizon, which is why it decided the buyback verdict on its own.
#
# Every test below runs on a 32-team board over ten or more weeks, because the failure is only
# visible where the no-repeat ledger binds. On a three-team toy an entry spends its options in
# two weeks and the optimiser has nothing to be right about.


_TEAMS = tuple(f"T{i:02d}" for i in range(32))
# A strength ladder, T00 weakest to T31 strongest, priced through a logistic on the gap.
#
# The wobble on top of the ladder is load-bearing rather than realism. An evenly spaced ladder
# gives many fixtures the identical strength gap and therefore the identical price, so the
# season has many optima and which one comes back is the solver's tie-break. That is fine for
# a survival figure and fatal for the test below, which asserts that solving the remainder
# mid-trial reproduces the plan solved at the start *exactly* -- a claim about a unique
# answer. With the wobble every gap is distinct, so the optimum is unique and the assertion
# is about the code rather than about CBC.
_STRENGTH = [-1.2 + 2.4 * i / 31 + 0.03 * math.sin(7.3 * i) for i in range(32)]


def _board(weeks, *, flat=()):
    """A 32-team board: sixteen fixtures a week, every team playing every week.

    Matchups move by the circle rotation, so the biggest favourite is a different team most
    weeks and a plan has something to solve. Ten weeks of it needs ten distinct teams and
    eighteen needs twenty-four, which is the ledger binding rather than a toy running out.

    A `flat` week is one where nothing is a favourite except the strongest team, at 0.97
    against a board of coin flips. That is the hoarding case in `hub.season.survivor`'s
    docstring made concrete on a real-sized grid: spending T31 in week 1 costs a coin flip in
    the flat week, so the greedy rule and the optimiser give different answers and it is
    possible to say which one our entry played. The coin flips are spread by a thousandth
    apiece for the same reason the ladder wobbles.
    """
    order = list(range(32))
    rows = []
    for w in range(1, max(weeks) + 1):
        pairs = [(order[i], order[31 - i]) for i in range(16)]
        if w in weeks:
            for i, (a, b) in enumerate(pairs):
                if w in flat:
                    p = 0.97 if a == 31 else 0.03 if b == 31 else 0.50 + 0.001 * i
                else:
                    p = 1.0 / (1.0 + math.exp(-(_STRENGTH[a] - _STRENGTH[b])))
                rows += [(w, _TEAMS[a], p, f"{w}-{i}"), (w, _TEAMS[b], 1.0 - p, f"{w}-{i}")]
        order = [order[0], order[-1], *order[1:-1]]
    return pl.DataFrame({"week": [r[0] for r in rows], "team": [r[1] for r in rows],
                         "win_prob": [r[2] for r in rows], "game_id": [r[3] for r in rows]},
                        schema={"week": pl.Int64, "team": pl.Utf8, "win_prob": pl.Float64,
                                "game_id": pl.Utf8})


def _free_chain(grid, weeks):
    """The free pick, taken every week and carrying what it spends: `auto_pick` as a plan.

    The thing a plan has to beat, per `docs/method.md` rule 5, and also what
    `pool._best_available` falls back to when no solver can run -- so it is both arms of two
    different tests below.
    """
    led, out = set(), {}
    for w in weeks:
        got = pool.auto_pick(grid, w, sorted(led))
        out[w] = (got,)
        led.add(got)
    return pool.Plan(out, "the free pick every week")


TEN = list(range(1, 11))


def test_our_entry_survives_materially_more_often_than_the_field_rule_gave_it():
    """The headline of #151, asserted rather than eyeballed.

    Two arms on the same board, the same field and the same seed: our entry playing the plan
    it would actually follow, and our entry under the rival sampling rule this module valued
    it with until now. `pool._entry_trials` with no plan is that old rule exactly -- it is the
    same trial loop and the same `_play`, with the one branch on index 0 not taken.

    The arms are **not** paired trial by trial, and that is stated because it matters which
    way it cuts: replaying a plan consumes no draws from the generator while sampling consumes
    one a week, so the two streams diverge after week 1 and the difference carries the
    variance of two independent means rather than of one paired one. The bound below is the
    conservative one -- four standard errors of an unpaired difference in proportions -- so
    the pairing this cannot have only makes the test harder to pass. `docs/method.md` rule 3:
    the trials are the unit here, and there are `trials` of them, not `trials` x `weeks`.
    """
    import math

    g = _board(TEN, flat=(6,))
    kw = {"entries": 12, "trials": 400}
    mine = pool.entry_outcome(g, TEN, rng=np.random.default_rng(0), **kw)
    wks = pool.weeks_from_grid(g, TEN)
    field = pool._entry_trials(np.random.default_rng(0), wks, TEN, ledger=set(), ours=None,
                               **kw)

    se = math.sqrt(mine.survives * (1 - mine.survives) / mine.trials
                   + field.survives * (1 - field.survives) / field.trials)
    assert mine.survives - field.survives > 4 * se
    # And it is a difference worth having in dollars, not merely a detectable one: the old
    # rule left our entry among the also-rans for ten weeks of chalk-weighted noise.
    assert mine.survives > 0.2 > field.survives
    assert field.plan is None and mine.plan is not None


def test_the_picks_our_entry_plays_are_the_optimisers_and_not_the_free_ones():
    """Which rule produced them, pinned on a board where the two rules disagree.

    Week 6 is flat: nothing is a favourite except T31. The free pick spends T31 in week 1,
    where it is merely the best of sixteen real favourites, and arrives in week 6 holding a
    3% dog. The optimiser hoards it. That is the whole argument for solving a season as one
    assignment problem, and it is the argument our entry was not getting the benefit of.
    """
    g = _board(TEN, flat=(6,))
    out = pool.entry_outcome(g, TEN, entries=12, trials=50, rng=np.random.default_rng(0))
    plan, free = out.plan, _free_chain(g, TEN)
    assert plan is not None
    assert "optimiser" in plan.source
    assert plan.picks != free.picks
    assert plan.picks[6] == ("T31",) and free.picks[6] != ("T31",)
    # A team a week, none of them twice: the no-repeat ledger, which is what makes this a
    # season-long problem rather than ten weekly ones.
    assert sorted(plan.picks) == TEN
    assert len({t for ts in plan.picks.values() for t in ts}) == len(TEN)
    # Nothing was invalidated, so the figure is about the picks it names.
    assert out.replans == 0.0


def test_two_candidate_plans_meet_the_same_season_and_the_better_one_wins():
    """The seam #159 needs, and the reason a plan is solved outside the trial loop.

    Both arms are handed a plan rather than solving one, both get the same seed, and our
    entry consumes no draws either way -- so the field they meet is the identical field, the
    games fall the identical way, and the only difference between the two figures is the
    picks. `docs/method.md` rule 7. The optimiser's plan beats the free chain here by the
    hoarded week, and the comparison says so on shared draws rather than on two seasons.
    """
    g = _board(TEN, flat=(6,))
    best = pool.solve_plan(g, TEN)
    free = _free_chain(g, TEN)
    kw = {"entries": 12, "trials": 400}
    a = pool.entry_outcome(g, TEN, plan=best, rng=np.random.default_rng(2), **kw)
    b = pool.entry_outcome(g, TEN, plan=free, rng=np.random.default_rng(2), **kw)
    # Played as given, both of them: a plan handed in is not quietly re-solved.
    assert a.plan == best and b.plan == free
    assert a.replans == 0.0 and b.replans == 0.0
    assert a.survives > b.survives


def test_two_candidate_plans_meet_the_identical_field(monkeypatch):
    """The pairing itself, asserted on the field rather than on the two figures (#159).

    One seed was not one season. `_play` drew each week's games as it reached it and stopped
    when the last live entry died, ours included -- so the first trial our entry outlasted the
    field under one plan and not the other consumed a different number of draws, the stream
    offset, and every later trial met a different season. That is not a subtle loss: the
    trials it corrupts are exactly the ones where the two plans differ, which are the only
    trials carrying the comparison.

    Recorded on the rivals' picks because that sequence is downstream of everything the two
    arms are supposed to share -- a rival appears in it only while it is alive, and it is
    alive only because of games it won. A different draw anywhere shows up here.
    """
    g = _board(TEN, flat=(6,))
    best, free = pool.solve_plan(g, TEN), _free_chain(g, TEN)
    kw = {"entries": 12, "trials": 300}

    def _rivals(plan):
        seen = []
        real = pool._pick
        monkeypatch.setattr(
            pool, "_pick",
            lambda rng, week, ledger, k: seen.append(
                (tuple(sorted(ledger)), tuple(got := real(rng, week, ledger, k) or ()))) or got)
        out = pool.entry_outcome(g, TEN, plan=plan, rng=np.random.default_rng(2), **kw)
        monkeypatch.undo()
        return out, seen

    a, mine = _rivals(best)
    b, theirs = _rivals(free)
    assert a.survives != b.survives, "the two plans have to differ, or this proves nothing"
    assert mine == theirs
    assert len(mine) > 1000


def test_two_plans_that_name_the_same_picks_give_exactly_the_same_figure():
    """The pairing's sharpest form: identical picks, identical seed, identical figure.

    Not `approx`. Two candidates that are the same pick differ by zero in every trial, so
    every figure the comparison rests on -- the means and the per-trial vector they are means
    of -- has to agree to the bit. A difference here would be the simulator reading something
    other than the picks, which is the failure the pairing exists to rule out.

    The plans are distinct objects carrying different `source` prose, so what is shared is the
    picks and nothing else.
    """
    g = _board(TEN, flat=(6,))
    best = pool.solve_plan(g, TEN)
    twin = pool.Plan(dict(best.picks), "the same picks, arrived at by another road")
    kw = {"entries": 12, "trials": 300}
    a = pool.entry_outcome(g, TEN, plan=best, rng=np.random.default_rng(2), **kw)
    b = pool.entry_outcome(g, TEN, plan=twin, rng=np.random.default_rng(2), **kw)
    assert a.plan is not None and b.plan is not None
    assert a.plan != b.plan and a.plan.picks == b.plan.picks
    assert (a.survives, a.sole, a.share, a.share_sd) == (b.survives, b.sole, b.share,
                                                         b.share_sd)
    assert a.share_each == b.share_each


def test_the_per_trial_share_is_positive_exactly_where_the_entry_survived():
    """What `share_each` is, pinned so a paired difference taken from it means what it says.

    The three means are means of this vector, and survival is its support rather than a second
    vector -- so a caller taking a paired difference of survival and a paired difference of
    money is reading one record of one set of trials, not two records that could disagree.
    """
    g = _board(TEN, flat=(6,))
    out = pool.entry_outcome(g, TEN, entries=12, trials=200, rng=np.random.default_rng(0))
    xs = out.share_each
    assert len(xs) == out.trials == 200
    assert sum(x > 0 for x in xs) / len(xs) == out.survives
    assert sum(xs) / len(xs) == pytest.approx(out.share)
    assert sum(x == 1.0 for x in xs) / len(xs) == out.sole
    assert out.shared == pytest.approx(out.survives - out.sole)


def test_a_plan_that_will_not_play_is_re_solved_into_the_one_we_would_have_solved():
    """The invalidation rule, and the only thing that makes replaying a plan safe.

    Both reasons a week can be invalid are in the plan handed in, because they are different
    failures and either one alone would leave the other untested. Odd weeks name T31, which is
    already spent -- the no-repeat ledger. Even weeks name a team that is not on this board at
    all, which stands for a pick the week as drawn cannot offer: below `MIN_PROB`, or in a
    fixture the grid priced on one side only and `weeks_from_grid` dropped. Without the second
    check that team would be played, lose by default, and read out as an entry that picked
    badly rather than one handed a pick that does not exist.

    Every week is therefore solved again from the ledger the trial actually holds. Solving the
    *remainder* rather than the week is what makes that answer identical to having solved once
    at the start -- so the two figures are not merely close, they are the same figure, on the
    same seed.

    Both halves are asserted. Checking the survival alone would pass if invalidation silently
    did nothing and both arms played one solved plan; checking `replans` alone would pass if
    the re-solve returned something legal and bad.
    """
    spent = ("T31",)
    g = _board(TEN, flat=(6,))
    stale = pool.Plan({w: ("T31",) if w % 2 else ("NOT_ON_THIS_BOARD",) for w in TEN},
                      "a team already spent, or one this week cannot offer")
    kw = {"entries": 12, "ledger": spent, "trials": 300}
    fresh = pool.entry_outcome(g, TEN, rng=np.random.default_rng(3), **kw)
    again = pool.entry_outcome(g, TEN, plan=stale, rng=np.random.default_rng(3), **kw)
    assert fresh.replans == 0.0 and again.replans >= 1.0
    assert again.survives == fresh.survives
    # What was handed in comes back unchanged, so a caller can still see what it asked for
    # rather than what the trials made of it.
    assert again.plan == stale


def test_our_branch_does_not_reach_a_rival():
    """Rivals are unaffected, asserted where a leak would be unmissable.

    Two entries, ours and one rival, on the same board. If `_play`'s branch had been written
    without the index test the rival would hold our teams every week, survive exactly when we
    do, and this ratio would be 1.0. It is a rival sampling near-chalk over six weeks instead.

    `test_the_entry_shares_outcomes_with_rivals_on_its_team` pins the same thing from the
    other side and to a closed form: the rival there still takes the favourite nine times in
    ten, which is the sampling rule and not a plan.
    """
    six = list(range(1, 7))
    out = pool.entry_outcome(_board(six), six, entries=2, trials=400,
                             rng=np.random.default_rng(1))
    assert out.survives > 0.4
    assert (out.survives - out.sole) / out.survives < 0.25


def test_a_double_pick_week_is_planned_as_two_teams_from_two_fixtures():
    """Weeks 13-18 take a pair and both have to win, so two sides of one game is fatal.

    Run at the pool's real shape -- eighteen weeks, six of them double -- because that is
    where the ledger binds hardest: twenty-four picks against thirty-two teams, which is the
    constraint `hub.season.survivor.solve` is one assignment problem for.
    """
    weeks = list(range(1, 19))
    g = _board(weeks)
    plan = pool.solve_plan(g, weeks)
    assert "optimiser" in plan.source
    assert sorted(plan.picks) == weeks
    for w in range(13, 19):
        assert len(plan.picks[w]) == 2
    assert len({t for ts in plan.picks.values() for t in ts}) == 24
    # Two picks, two games: the sides of one fixture never appear together.
    wk = pool.weeks_from_grid(g, weeks, PoolConfig())[12]
    assert len({wk.fixture[t] for t in plan.picks[13]}) == 2


def test_a_ledger_with_nothing_left_ends_the_entry_rather_than_repeating_a_team():
    """The no-repeat rule where it bites hardest, on our side of the branch.

    Every one of the thirty-two teams is spent, so no assignment covers the weeks ahead. The
    optimiser says so, the fallback finds nothing either, and the entry is eliminated for want
    of a legal pick -- which is what `_pick` returning None does to a rival, reached by a
    different road. A plan with no week in it is the honest answer here; filling the gap with
    a repeat would be worth a great deal and is against the rules.
    """
    out = pool.entry_outcome(_board([1, 2]), [1, 2], entries=4, ledger=_TEAMS, trials=20,
                             rng=np.random.default_rng(0))
    assert out.survives == 0.0 and out.share == 0.0
    assert out.plan is not None and out.plan.picks == {}
    assert "no assignment covers these weeks" in out.plan.source


def test_a_figure_computed_without_the_optimiser_says_so(monkeypatch):
    """Graceful degradation, and the acceptance criterion that the docstring name which rule
    ran: `CLAUDE.md` says a module with a failed dependency serves the best answer it has
    rather than erroring, and `Plan.source` is where that answer admits what it is.

    With no solver the fallback is the free pick taken every week -- the greedy rule
    `hub.season.survivor` exists to beat -- so the figure is still a figure and is a weaker
    claim than the same number off the optimiser. Asserting the source alone would not say
    that; the picks are asserted against `_free_chain` so the fallback is pinned to a rule
    somebody can name.
    """
    def _no_solver(*a, **k):
        raise RuntimeError("no CBC on this machine")

    monkeypatch.setattr(pool, "solve", _no_solver)
    g = _board(TEN, flat=(6,))
    out = pool.entry_outcome(g, TEN, entries=12, trials=300, rng=np.random.default_rng(0))
    assert out.plan is not None
    assert "the optimiser could not run" in out.plan.source
    assert "no CBC on this machine" in out.plan.source
    assert out.plan.picks == _free_chain(g, TEN).picks
    assert out.survives > 0.0


def test_a_plan_naming_one_team_in_a_double_pick_week_is_not_a_plan_for_that_week():
    """Arity is part of what makes a plan playable, and a week that takes two is where it
    shows. Handed one team for week 13, our entry does not field half a pick and does not
    quietly take one -- the week is invalid, the remainder is solved again, and the answer is
    the one solving from the start would have given, to the trial.

    Reached by halving a real plan rather than by inventing one, so the teams named are legal
    in every other respect and arity is the only thing wrong with them.
    """
    weeks = [13, 14]
    g = _board(weeks)
    best = pool.solve_plan(g, weeks)
    half = pool.Plan({w: ts[:1] for w, ts in best.picks.items()}, "half of a double-pick plan")
    kw = {"entries": 12, "trials": 200}
    fresh = pool.entry_outcome(g, weeks, rng=np.random.default_rng(5), **kw)
    given = pool.entry_outcome(g, weeks, plan=half, rng=np.random.default_rng(5), **kw)
    assert all(len(ts) == 2 for ts in best.picks.values())
    assert given.replans >= 1.0 and fresh.replans == 0.0
    assert given.survives == fresh.survives
    assert given.plan == half


def test_a_double_week_with_one_fixture_left_eliminates_rather_than_taking_both_sides():
    """The trade `hub.season.survivor.solve` calls guaranteed fatal and attractive, reached
    from the ledger instead of from the grid.

    Thirty of the thirty-two teams are spent, so the only legal pair left in this double-pick
    week is the two sides of one fixture. One of them loses, so taking both is not a way to
    cover the week -- it is a way to lose it while conserving two favourites, which is exactly
    why the optimiser has a constraint against it and why the fallback must not undo that. The
    optimiser refuses the week and says why; the fallback finds one team and no legal partner;
    the entry is eliminated for want of a pair.

    `test_a_double_pick_week_with_one_fixture_is_refused_rather_than_killing_the_field` is the
    same rule where the *grid* is short a fixture, which is a refusal rather than a result. A
    ledger this deep is a result: the week really is uncoverable, for this entry alone.
    """
    g = _board([13])
    wk = pool.weeks_from_grid(g, [13], PoolConfig())[0]
    intact = {wk.games[0][0], wk.games[0][1]}
    out = pool.entry_outcome(g, [13], entries=6, ledger=tuple(t for t in _TEAMS
                                                              if t not in intact),
                             trials=20, rng=np.random.default_rng(0))
    assert out.survives == 0.0
    assert out.plan is not None and out.plan.picks == {}
    assert "both sides of one fixture" in out.plan.source


# --- no entry takes both sides of one fixture (#156) -----------------------------------------
#
# The rule the week builder's own docstring stated and nothing enforced on the pick side. The
# game identity was read for the *draw* -- so a team and its opponent could not both win -- and
# never for the *pick*, where the only exclusion applied was the entry's own ledger. In a
# double-pick week that hands an entry a week it has already lost, and can eliminate it
# outright. `solve` forbids it as `one_side_wk` and `_best_available` takes one team per
# fixture, so the defect lived exactly where the money is priced: the rival sampler.
#
# It was unreachable before #151 and #152 -- under the old field rule almost nothing survived
# to week 13 -- which is why a test written against it then would have passed vacuously.


def _watch_picks(monkeypatch) -> list[tuple[int, int]]:
    """Every set of picks any entry is handed, as (teams asked for, fixtures they span).

    Both sides of `_play`'s branch, because #156 applies to our own entry as well as to
    rivals: a rival reaches `pool._pick` and our entry reaches `pool._Ours.picks`, and the
    two are the only ways a pick enters a trial.

    Watched rather than reconstructed afterwards, because a self-inflicted elimination leaves
    no trace in `EntryOutcome`. The entry is simply gone, and being handed both sides of one
    game reads out identically to having picked two favourites and lost one.
    """
    seen: list[tuple[int, int]] = []
    real_pick, real_ours = pool._pick, pool._Ours.picks

    def watched_pick(rng, week, ledger, k):
        got = real_pick(rng, week, ledger, k)
        if got is not None:
            seen.append((len(got), len({week.fixture[t] for t in got})))
        return got

    def watched_ours(self, at, ledger):
        got = real_ours(self, at, ledger)
        if got is not None:
            seen.append((len(got), len({self.wks[at].fixture[t] for t in got})))
        return got

    monkeypatch.setattr(pool, "_pick", watched_pick)
    monkeypatch.setattr(pool._Ours, "picks", watched_ours)
    return seen


def test_the_self_inflicted_elimination_rate_in_a_double_pick_season_is_zero(monkeypatch):
    """The acceptance criterion, counted over a season rather than argued from one draw.

    Every week here takes two picks, so every pick handed out is a chance to break the rule --
    on a real board that would be six weeks in eighteen, and only for the entries that got
    there. A self-inflicted elimination is a pick set spanning fewer fixtures than it names
    teams: one of that pair loses with certainty, so the week is lost at the moment it is
    entered and no draw can save it.

    The rate is zero and not merely small. It is a constraint, not a tendency: there is no
    trial count at which one of these is acceptable.
    """
    weeks = list(range(1, 7))
    g = _board(weeks)
    cfg = PoolConfig(double_pick_weeks=tuple(weeks))
    seen = _watch_picks(monkeypatch)
    out = pool.entry_outcome(g, weeks, entries=12, pool=cfg, trials=200,
                             rng=np.random.default_rng(0))
    # The season has to have actually been played, or a rate of zero is a rate over nothing.
    assert out.survives > 0.0
    assert len(seen) > 4000
    assert all(n == 2 for n, _ in seen), "every week here takes two picks"
    self_inflicted = [s for s in seen if s[1] < s[0]]
    assert self_inflicted == []
    assert len(self_inflicted) / len(seen) == 0.0


def test_the_rule_reaches_our_own_entry_and_not_only_the_field(monkeypatch):
    """The same count with the field removed, so our arm cannot hide behind twenty rivals.

    One entry, which is ours: every pick recorded came through `_Ours` and the plan it
    replays. Our side was already correct -- the optimiser's `one_side_wk` and
    `_best_available`'s one-per-fixture walk see to that -- and it is pinned here because
    "the rule applies to our own entry as well as rivals" is a claim about both arms, and a
    claim held by only one of them is a claim nothing re-checks when the other moves.
    """
    weeks = list(range(1, 7))
    g = _board(weeks)
    cfg = PoolConfig(double_pick_weeks=tuple(weeks))
    seen = _watch_picks(monkeypatch)
    out = pool.entry_outcome(g, weeks, entries=1, pool=cfg, trials=50,
                             rng=np.random.default_rng(0))
    assert out.plan is not None and "optimiser" in out.plan.source
    assert seen and all(n == 2 and f == 2 for n, f in seen)


# --- Field concentration (#152) -------------------------------------------------------------
#
# The knob is stated and not fitted, so what these pin is the *mechanism*: that it reaches the
# rival sampler, that it moves ownership in the direction it claims to, that it reaches nothing
# else, and that its default changes nothing. What value it should hold is not a question a
# test can answer -- `pool.sensitivity` reports the range instead, and the last test here is
# that the range is reported rather than a point.


def test_the_default_concentration_samples_exactly_on_win_probability():
    """The no-op claim, at the one place it has to hold.

    A knob defaulting to "today's behaviour" is worth nothing if the default is only *close*
    to today's behaviour: every figure in the repo would move by a hair the day it landed, and
    nobody could tell that drift from a real change afterwards. `prob ** 1.0` is exactly
    `prob` for a finite positive float, so the weight the sampler reads is the identical
    object-valued float and not a rounding of it. Asserted with `==` on purpose.
    """
    wk = pool.weeks_from_grid(_board([1]), [1], PoolConfig())[0]
    assert wk.weight == {t: wk.prob[t] for t in sorted(wk.pickable)}
    assert all(wk.weight[t] == wk.prob[t] for t in wk.pickable)


def test_raising_the_concentration_raises_the_best_teams_ownership():
    """The acceptance criterion, on a full-sized grid: sixteen fixtures, thirty-two teams.

    On a toy week of two fixtures almost any weighting concentrates, so the claim has to be
    made where it is actually in doubt -- a real slate has fifteen other favourites and the
    exponent lifts all of them too. The share still rises at every step, and it rises *slowly*:
    a sixteenth of the field at 1.0 is about a quarter of it at 16.0, not at 2.0. That is why
    `DEFAULT_CONCENTRATIONS` is geometric and runs as far as it does.

    **And the knob changes only the weights, never the legal set.** `pickable` is `MIN_PROB`
    against the *price*, and applying the exponent before that test instead would let a high
    concentration quietly delete the week's underdogs from what anybody -- rivals and our own
    optimiser alike -- is allowed to take. That is a different rule wearing this one's name,
    it passes every other assertion in this file, and the set is pinned here because this is
    the test already holding a week at each point on the axis.
    """
    g = _board([1])
    shares, legal = [], []
    for k in pool.DEFAULT_CONCENTRATIONS:
        wk = pool.weeks_from_grid(g, [1], PoolConfig(field_concentration=k))[0]
        chalk, own = pool._chalk_share(wk)
        assert chalk == "T31", "the strongest team on the ladder is the week's chalk"
        shares.append(own)
        legal.append(wk.pickable)
    assert shares == sorted(shares) and len(set(shares)) == len(shares)
    assert shares[0] == pytest.approx(1 / 16, abs=0.01), "1.0 is the sixteenth #152 names"
    assert shares[-1] > 4 * (shares[0] - 1 / 32), "and the axis reaches a real field's crowding"
    assert len(set(legal)) == 1 and legal[0] == frozenset(_TEAMS)


def test_the_concentration_reaches_the_sampler_and_not_just_the_weight():
    """That the knob is wired to `_pick` rather than only to the arithmetic beside it.

    `_chalk_share` reads `_Week.weight` too, so the test above would go on passing if `_pick`
    quietly went back to reading `_Week.prob` -- the ownership it reported would be a number
    nobody drew with. This counts real draws instead, which is the only assertion here that
    can tell the sampler's weight from a weight computed near it.
    """
    g = _board([1])
    drawn = {}
    for k in (1.0, 16.0):
        wk = pool.weeks_from_grid(g, [1], PoolConfig(field_concentration=k))[0]
        rng = np.random.default_rng(0)
        drawn[k] = sum(pool._pick(rng, wk, set(), 1) == ["T31"] for _ in range(3000)) / 3000
    assert drawn[1.0] == pytest.approx(0.057, abs=0.02)
    assert drawn[16.0] == pytest.approx(0.270, abs=0.03)
    assert drawn[16.0] > 3 * drawn[1.0]


def test_a_concentration_that_inverts_the_rule_is_refused():
    """Negative is not "less concentrated", it is a field chasing the worst team on the board.

    Refused rather than clamped, because clamping would put a point on the sweep's axis that
    reads as a concentration and is not one. A NaN is refused in the same breath: it would
    propagate through the weights into `rng.choice`'s probability vector and raise several
    frames from the setting that caused it.
    """
    g = _board([1])
    for bad in (-1.0, -0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="field_concentration"):
            pool.weeks_from_grid(g, [1], PoolConfig(field_concentration=bad))
    # Zero is the axis's floor and a legal point on it: a field picking uniformly.
    wk = pool.weeks_from_grid(g, [1], PoolConfig(field_concentration=0.0))[0]
    assert pool._chalk_share(wk)[1] == pytest.approx(1 / len(wk.pickable))


def test_the_concentration_changes_the_field_our_plan_meets_and_not_our_picks():
    """What #151 bought, held: the knob is the rivals' rule, so it must reach the field our
    entry is scored against and not the picks it plays.

    **The two halves are not equally hard to break, and saying which is which is the point.**
    That our own figures *move* is the falsifiable half: it fails the moment the exponent stops
    reaching `_pick`, which is how a sweep would come to report a knob that does nothing.

    That the *plan* does not move is a weaker claim than it looks, and it was worth finding out
    by trying to break it. An exponent is a monotone transform of every price, so feeding
    `_Week.weight` to the optimiser instead of `_Week.prob` rescales its objective and leaves
    the argmax alone -- the plan comes back identical. So this half does not catch a misrouted
    weight; it is a guard against the knob acquiring a *shape* that could reach our picks at
    all, and it is cheap enough to be worth keeping on those terms rather than on stronger ones
    it does not meet.
    """
    weeks = [1, 2, 3, 4]
    g = _board(weeks)
    lo, hi = (pool.entry_outcome(g, weeks, entries=21, trials=150,
                                 pool=PoolConfig(field_concentration=k),
                                 rng=np.random.default_rng(3))
              for k in (1.0, 16.0))
    assert lo.plan is not None and hi.plan is not None
    assert lo.plan.picks == hi.plan.picks and lo.plan.source == hi.plan.source
    assert lo.share != hi.share and lo.survives != hi.survives


def test_the_sweep_reports_a_range_over_the_knob_rather_than_a_point():
    """The deliverable: a row per concentration, and a dollar figure that is an interval.

    Every field in the row has to move with the axis or the sweep is reporting a knob that
    does nothing -- so ownership, the pool's lifetime and our own equity are each checked for
    a spread. The report's final line is the range itself, which is the form ADR-0024 asks a
    stated parameter's figures to be published in.
    """
    g = _board([1, 2, 3, 4])
    rows = pool.sensitivity(g, [1, 2, 3, 4], entries=21, at=(1.0, 4.0, 16.0), pot=420.0,
                            trials=150, rng=np.random.default_rng(5))
    assert [r.concentration for r in rows] == [1.0, 4.0, 16.0]
    own = [r.chalk_share for r in rows]
    assert own == sorted(own) and own[0] < own[-1]
    assert len({r.wiped_out for r in rows}) > 1, "the pool's lifetime moves with the knob"
    assert len({round(r.equity, 6) for r in rows}) > 1, "so does the dollar figure"
    assert all(r.pot == 420.0 and r.equity == pytest.approx(r.entry.share * 420.0)
               for r in rows)
    lines = pool.sensitivity_report(rows)
    assert "stated, never fitted" in lines[0]
    assert lines[-1].startswith("  equity $") and " to $" in lines[-1]
    assert pool.sensitivity_report([]) == ["\n  no concentrations swept"]


def test_the_sweep_anchors_on_the_behaviour_already_in_the_tree():
    """The row at 1.0 is not a new number, it is `simulate` at the default reseeded.

    This is what makes the table readable as a *sensitivity* rather than as five unrelated
    runs: one of the rows is the figure the module publishes today, and the reader can see
    which. Reseeding per row is what buys it, so this fails the moment the sweep starts
    sharing one generator across the axis.
    """
    g = _board([1, 2, 3])
    rows = pool.sensitivity(g, [1, 2, 3], entries=21, at=(1.0,), trials=120,
                            rng=np.random.default_rng(5))
    seed = int(np.random.default_rng(5).integers(2 ** 32))
    direct = pool.simulate(g, [1, 2, 3], entries=21, trials=120,
                           rng=np.random.default_rng(seed))
    assert rows[0].field == direct
