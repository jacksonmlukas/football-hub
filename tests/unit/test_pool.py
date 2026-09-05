"""The pool as a whole. These pin the correlation, because that is what decides the answer.

A field playing chalk dies together, and how together it dies is what sets the week the pool
ends -- which is what a buyback is priced against. Draw the games per entry instead of per
game and eliminations become independent, the field thins smoothly, and the contest stretches
past anything real. Most of what follows is that one property, tested from angles that would
each fail differently if it broke.
"""
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


def test_a_double_pick_week_takes_two_distinct_teams():
    """Weeks 13-18 need two, both of which must win. Read from the pool config, so a rule
    correction is a re-run."""
    g = _grid([(13, "KC", "LV", 0.8), (13, "SF", "SEA", 0.7)])
    wk = pool.weeks_from_grid(g, [13], PoolConfig(double_pick_weeks=(13,)))[0]
    assert wk.picks == 2
    got = pool._pick(np.random.default_rng(0), wk, set(), 2)
    assert got is not None and len(set(got)) == 2


def test_a_grid_without_a_game_key_is_refused():
    """Rather than falling back to per-team draws, which is the silent version of the bug this
    module exists to avoid."""
    g = _grid([(1, "KC", "LV", 0.8)]).drop("game_id")
    with pytest.raises(ValueError, match="game_id"):
        pool.weeks_from_grid(g, [1])


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
