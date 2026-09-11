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

from hub.exhibits import leverage

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


# --- calibrating at the settings the row is evaluated under (#173) ----------
#
# Against an analytic stand-in for `team_mean` rather than the simulator, because the claim
# is about *which settings the solver is handed* and that has an exact answer. Driving it
# through 60,000 Monte Carlo draws would replace a checkable equality with a tolerance, and
# a tolerance is precisely what the bug hid inside: the uncalibrated row missed its target
# by half a per cent, which reads as noise unless you know what to compare it to.

K0 = 0.90                      # the roster strength the weak sweep rows hold the mean at
_TARGET = 90.0                 # `_mean(K0, 1.0, 1.0)`, by construction below


def _mean(k=1.0, vol=1.0, cv_mult=1.0, **_kw) -> float:
    """A stand-in with `team_mean`'s two real properties and no simulator.

    Rising in `k`, and rising in *both* variance knobs -- which is the whole reason
    `calibrate` exists. The clip at zero inside `simulate_weeks` is what makes the real
    function rise with `cv_mult`; the coefficients here are arbitrary, the dependence is not.
    """
    return 100.0 * k * (1.0 + 0.2 * (cv_mult - 1.0)) * (1.0 + 0.1 * (vol - 1.0))


def test_the_stand_in_has_the_dependence_the_bug_turns_on():
    """If `team_mean` did not move with `cv_mult`, calibrating at the wrong one would cost
    nothing and this ticket would be about tidiness. It does, and so does the stand-in."""
    assert _mean(K0) == pytest.approx(_TARGET)
    assert _mean(0.9, 1.0, 1.2) > _mean(0.9, 1.0, 1.0)
    assert leverage.team_mean(cv_mult=2.0) > leverage.team_mean(cv_mult=0.5)


def test_the_volatility_row_is_calibrated_under_its_own_talent_multiplier(monkeypatch):
    """#173's first criterion, on the row that had it wrong.

    The weekly-spread row holds season-long spread fixed in absolute points, so its talent
    multiplier is `k0 / k` -- a function of the multiplier being solved for. Solved that way,
    `100k * (1 + 0.2(0.9/k - 1)) * 1.08 = 90` gives `k = 49/60`, and the row's achieved mean
    is its target exactly rather than nearly.
    """
    monkeypatch.setattr(leverage, "team_mean", _mean)
    k, at, got = leverage.calibrated(_TARGET, vol=1.8, cv_mult=lambda m: K0 / m)
    assert k == pytest.approx(49 / 60, abs=1e-3)
    assert at == pytest.approx(K0 / k, rel=1e-9)
    assert got == pytest.approx(_TARGET, rel=1e-4)


def test_a_sweep_row_simulates_at_the_settings_it_calibrated_under(monkeypatch):
    """#173's first criterion asserted where it can actually fail: across both calls.

    Reporting the miss is not enough on its own, and finding that out is what this test is.
    A row that solved at `cv_mult = 1.0` and *also* measured its mean at 1.0 reports a
    perfect calibration and still simulates at something else -- the mistake is invisible to
    any check that only looks at the calibration half. So what is asserted here is the pair:
    the settings handed to `simulate` are the settings `team_mean` was asked about, and the
    mean at those settings is the target.

    `sweep_row` is written so that they come from one variable rather than two spellings,
    which is why this passes for a structural reason rather than a vigilant one.
    """
    seen = {}
    monkeypatch.setattr(leverage, "team_mean", _mean)
    monkeypatch.setattr(leverage, "simulate",
                        lambda **kw: seen.update(kw) or {"playoff": 0.5, "bye": 0.1,
                                                         "title": 0.1, "wins": 7.0})
    _, err = leverage.sweep_row(_TARGET, vol=1.8, cv_mult=lambda m: K0 / m, n_sims=10)
    assert seen["vol"] == 1.8
    assert seen["cv_mult"] == pytest.approx(K0 / seen["k"], rel=1e-9)
    assert _mean(seen["k"], seen["vol"], seen["cv_mult"]) == pytest.approx(_TARGET, rel=1e-4)
    assert err < leverage.CALIBRATION_TOL


def test_calibrating_at_the_default_multiplier_misses_the_target(monkeypatch):
    """The defect itself, as the number it was worth. Solving at `cv_mult = 1.0` and then
    simulating at `k0 / k` is what the loop did; the mean it reports as fixed is 1.6% away
    from the one it was calibrated to, on a stand-in where the right answer is exact.

    Larger than `CALIBRATION_TOL`, which is the point of having stated one.
    """
    monkeypatch.setattr(leverage, "team_mean", _mean)
    k = leverage.calibrate(_TARGET, vol=1.8)            # the old call, cv_mult defaulted
    got = _mean(k, 1.8, K0 / k)                         # the settings it was then run at
    assert k == pytest.approx(5 / 6, abs=1e-3)
    assert leverage.missed_target(got, _TARGET) > leverage.CALIBRATION_TOL


def test_a_constant_multiplier_still_means_a_constant(monkeypatch):
    """The callable is an addition, not a replacement. A float has to solve exactly as it
    did, or the multiplier loop -- which #173 requires be left alone -- moves underneath."""
    monkeypatch.setattr(leverage, "team_mean", _mean)
    by_value = leverage.calibrate(_TARGET, vol=1.8, cv_mult=1.3)
    by_function = leverage.calibrate(_TARGET, vol=1.8, cv_mult=lambda _k: 1.3)
    assert by_value == pytest.approx(by_function, abs=1e-9)


def test_the_multiplier_loop_calibration_is_unchanged(monkeypatch):
    """#173's third criterion. `calibrated` adds a measurement beside `calibrate`; it must
    not be a second calibration that could drift from it."""
    monkeypatch.setattr(leverage, "team_mean", _mean)
    for cvm in (0.5, 2.0):
        k, at, _got = leverage.calibrated(_TARGET, cv_mult=cvm)
        assert k == pytest.approx(leverage.calibrate(_TARGET, cv_mult=cvm), abs=1e-9)
        assert at == cvm, "a constant multiplier resolves to itself, whatever k came back"


def test_a_row_that_hit_its_target_is_not_marked_and_one_that_missed_is():
    """The tolerance has to be visible in the output or it is a number nobody can check a
    row against."""
    assert leverage.missed_target(101.0, 100.0) == pytest.approx(0.01)
    assert "OFF" not in leverage._miss(0.0)
    assert "OFF" not in leverage._miss(leverage.CALIBRATION_TOL / 2)
    assert "OFF" in leverage._miss(2 * leverage.CALIBRATION_TOL)


def test_every_row_of_the_real_sweep_is_run_at_the_settings_it_was_solved_for(monkeypatch):
    """#173's first two criteria over the CLI's own eight rows rather than over a call.

    Two things are asserted per row, and they are different claims. **That the row is
    self-consistent** -- the mean at the settings `simulate` was given is the row's target --
    is #173. **That the weekly rows hold season-long spread fixed in absolute points** --
    `cv_mult * k == k0` -- is the second confound `docs/six-of-twelve.md` records, and a row
    that quietly dropped it would still be internally consistent and would no longer be the
    comparison the table's label makes. Only asserting the first would let the second go.

    `team_mean` is the analytic stand-in so the equality is exact and the run is quick;
    `simulate` records rather than draws, since no outcome column is under test here.
    """
    seen = []

    def recorded(**kw):
        if "cv_mult" in kw:                 # a sweep row; the strength table passes neither
            seen.append(kw)
        k = kw.get("k", 1.0)
        # Varying with `k` so the strength table's pp/win gradient has a denominator.
        return {"playoff": 0.5 * k, "bye": 0.1 * k, "title": 0.1 * k, "wins": 7.0 * k}

    monkeypatch.setattr(leverage, "team_mean", _mean)
    monkeypatch.setattr(leverage, "simulate", recorded)
    assert leverage.main(["--sims", "10"]) == 0

    assert len(seen) == 8, seen
    for k0, rows in ((0.90, seen[:4]), (1.10, seen[4:])):
        target = _mean(k=k0)
        weekly, season = rows[:2], rows[2:]
        for row in rows:
            assert _mean(row["k"], row["vol"], row["cv_mult"]) == pytest.approx(
                target, rel=leverage.CALIBRATION_TOL), row
        for row in weekly:
            assert row["cv_mult"] * row["k"] == pytest.approx(k0, rel=1e-9), (
                "the weekly row must hold absolute season-long spread fixed while it "
                f"rescales the mean, so cv_mult is k0/k: {row}")
        assert sorted(r["cv_mult"] for r in season) == [0.5, 2.0]
        assert sorted(r["vol"] for r in season) == [1.0, 1.0], "the season rows sweep cv only"


def test_every_sweep_row_reports_its_calibration_error_and_lands_inside_it(capsys):
    """#173's second criterion, end to end. Eight rows, each carrying the miss between the
    mean it was calibrated to and the mean it was simulated at -- and none of them OFF.

    `--sims` is tiny because the sweep's *outcome* columns are not what is under test here;
    `calibrate` and `team_mean` do not read it, so the calibration is the same one the full
    run does.
    """
    assert leverage.main(["--sims", "300"]) == 0
    out = capsys.readouterr().out
    assert "mean err" in out
    rows = [ln for ln in out.splitlines()
            if ln.split()[:2] in (["weak", "weekly"], ["weak", "season"],
                                  ["strong", "weekly"], ["strong", "season"])]
    assert len(rows) == 8, rows
    assert not [ln for ln in rows if "OFF" in ln], (
        f"a sweep row is not the fixed-mean comparison it claims to be: {rows}")


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
