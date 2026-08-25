"""What 6-of-12 with two byes actually rewards.

`docs/championship-leverage.md` builds its draft-time strategy on "8 make playoffs, no
byes" and concludes the regular season is nearly a formality. The live league is 6 of 12
with byes for seeds 1-2, and this module measures what that structure pays for instead of
re-arguing it.

The tests are mostly symmetry properties, which is what makes a Monte Carlo harness like
this checkable at all: twelve identical rosters must produce 1/12 titles, 2/12 byes, 6/12
playoff berths and 7 wins apiece. If any of those drift, the bracket or the seeding is
wired wrong and every number the doc quotes is wrong with it.

One test pins a trap rather than a property. The first version of this measurement scaled
player spread to ask "does more variance help?" and found that it helped enormously at
every roster strength -- which was wrong. Best-lineup selection is a max over the roster,
so raising player spread raises the expected maximum: the sweep was quietly adding points,
not variance. `calibrate` exists to remove that, and `test_raising_spread_alone_also_raises_the_mean`
keeps the confound visible so nobody re-runs the uncalibrated version.
"""
import numpy as np
import pytest

from hub.draft import leverage

FAST = 4000


# --- symmetry: twelve identical teams --------------------------------------

def test_an_identical_league_gives_everyone_the_same_title_odds():
    p = leverage.simulate(n_sims=FAST)
    assert p["title"] == pytest.approx(1 / 12, abs=0.015)


def test_an_identical_league_gives_everyone_the_bye_rate_the_structure_implies():
    """Two byes among twelve teams. A bracket that seeded four or none would still look
    plausible in aggregate, so this is the load-bearing structural check."""
    assert leverage.simulate(n_sims=FAST)["bye"] == pytest.approx(2 / 12, abs=0.02)


def test_an_identical_league_makes_the_playoffs_half_the_time():
    assert leverage.simulate(n_sims=FAST)["playoff"] == pytest.approx(0.5, abs=0.03)


def test_an_identical_league_splits_the_wins_evenly():
    assert leverage.simulate(n_sims=FAST)["wins"] == pytest.approx(7.0, abs=0.15)


def test_title_probabilities_over_all_seeds_account_for_every_season():
    """Somebody wins every year, and only a seeded team can."""
    assert leverage.seed_value(n_sims=FAST).sum() == pytest.approx(1.0, abs=1e-9)


# --- the bye ---------------------------------------------------------------

def test_seeding_pays_monotonically():
    sv = leverage.seed_value(n_sims=FAST)
    assert list(sv) == sorted(sv, reverse=True)


def test_there_is_a_discontinuity_between_the_bye_seeds_and_the_rest():
    """The whole reason the doc's "seeding buys marginally easier matchups and nothing
    else" is false. Seeds 1-2 play two games to a title; 3-6 play three."""
    sv = leverage.seed_value(n_sims=FAST)
    assert sv[1] > 1.8 * sv[2]


# --- strength --------------------------------------------------------------

def test_a_stronger_roster_wins_more_titles():
    weak = leverage.simulate(k=0.90, n_sims=FAST)["title"]
    strong = leverage.simulate(k=1.10, n_sims=FAST)["title"]
    assert strong > 4 * weak


def test_regular_season_wins_carry_real_title_equity():
    """Directly contradicts "dP(champ)/d(regular-season win) is close to zero". If this
    ever comes back near zero the doc's original argument is back in play."""
    lo, hi = leverage.simulate(k=0.95, n_sims=FAST), leverage.simulate(k=1.05, n_sims=FAST)
    per_win = (hi["title"] - lo["title"]) / (hi["wins"] - lo["wins"])
    assert per_win > 0.02


# --- the confound that invalidated the first attempt -----------------------

def test_raising_spread_alone_also_raises_the_mean():
    """The trap, pinned. The starting lineup is the best legal subset of the roster, so
    player spread feeds the expected maximum. Any "same mean, more variance" sweep that
    skips `calibrate` is measuring extra points and will conclude that variance is free."""
    assert leverage.team_mean(vol=1.8) > leverage.team_mean(vol=1.0) * 1.10


def test_calibrate_puts_the_mean_back():
    target = leverage.team_mean(vol=1.0)
    k = leverage.calibrate(target, vol=1.8)
    assert leverage.team_mean(k=k, vol=1.8) == pytest.approx(target, rel=0.005)
    assert k < 1.0, "holding the mean fixed at higher spread must cost projected points"


# --- the two variances point different ways --------------------------------

def test_weekly_spread_does_not_buy_titles_for_a_strong_roster():
    """At a genuinely fixed mean, weekly boom-bust is not an edge -- head-to-head wastes
    the surplus. This is the half of the doc's ceiling advice that does not survive."""
    tgt = leverage.team_mean(k=1.10)
    flat = leverage.simulate(k=leverage.calibrate(tgt, vol=0.7), vol=0.7,
                             cv_mult=1.10 / leverage.calibrate(tgt, vol=0.7), n_sims=FAST)
    spiky = leverage.simulate(k=leverage.calibrate(tgt, vol=1.8), vol=1.8,
                              cv_mult=1.10 / leverage.calibrate(tgt, vol=1.8), n_sims=FAST)
    assert spiky["title"] < flat["title"]


def test_season_long_upside_does_buy_titles():
    """The half that survives, and for a reason the doc never gives: the payoff in seeding
    is convex, so a wider spread of *season outcomes* is worth more at the top than it
    costs at the bottom."""
    tgt = leverage.team_mean(k=1.0)
    narrow = leverage.simulate(k=leverage.calibrate(tgt, cv_mult=0.5), cv_mult=0.5, n_sims=FAST)
    wide = leverage.simulate(k=leverage.calibrate(tgt, cv_mult=2.0), cv_mult=2.0, n_sims=FAST)
    assert wide["title"] > 1.5 * narrow["title"]


# --- the CLI --------------------------------------------------------------

def test_the_cli_reports_the_gradient_the_old_doc_denied(capsys):
    """`--sims` is deliberately tiny here. The point is that the table renders and the
    per-win gradient is printed at all -- "close to zero" was the claim, so a number has to
    appear next to it."""
    assert leverage.main(["--sims", "300"]) == 0
    out = capsys.readouterr().out
    assert "pp/win" in out
    assert "seed 1 (bye)" in out


def test_the_cli_labels_the_two_variances_separately(capsys):
    """They point opposite ways, so output that did not distinguish them would reproduce
    exactly the conflation this measurement exists to undo."""
    leverage.main(["--sims", "300"])
    out = capsys.readouterr().out
    assert "weekly" in out and "season" in out


def test_the_headline_direction_survives_a_tiny_sample(capsys):
    """Robustness, not precision. If the season-long result only appears at 20k sims it is
    probably a seed artifact; it should already be visible in a few hundred."""
    tgt = leverage.team_mean(k=1.10)
    narrow = leverage.simulate(k=leverage.calibrate(tgt, cv_mult=0.5), cv_mult=0.5, n_sims=1500)
    wide = leverage.simulate(k=leverage.calibrate(tgt, cv_mult=2.0), cv_mult=2.0, n_sims=1500)
    assert wide["title"] > narrow["title"]


# --- the harness must not drift from the model it measures ----------------

def test_it_draws_through_the_real_simulator():
    """`_season` used to re-implement `simulate_weeks`' internals, and quietly kept the old
    weekly model when that one moved: proportional spread, normal draws, spread keyed to
    the projection rather than realised talent. Every number in docs/six-of-twelve.md was
    then computed against a model the repo had stopped using.

    Pinning it by behaviour rather than by inspection: the harness and the simulator, given
    the same seed and inputs, must produce the same weekly points.
    """
    from hub.draft.season import simulate_weeks
    pts, _, _ = leverage._season(1.0, 1.0, 1.0, 64, 11)
    direct = simulate_weeks(
        leverage.ROSTERS, np.tile(leverage.MU, leverage.TEAMS),
        np.tile(leverage.SD, leverage.TEAMS), leverage.POOL_POS,
        n_sims=64, weeks=leverage.SIM_WEEKS, rng=np.random.default_rng(11),
        talent_cv=leverage._talent_cv(1.0))
    assert np.allclose(pts, direct)


def test_the_harness_uses_the_square_root_spread_law():
    """It hardcoded SD = MU * 0.55, which is the constant docs/weekly-spread.md replaced."""
    from hub.draft.season import WEEKLY_K
    for i, p in enumerate(leverage.POS):
        assert leverage.SD[i] == pytest.approx(
            WEEKLY_K[str(p)] * np.sqrt(leverage.MU[i]), rel=1e-6)
