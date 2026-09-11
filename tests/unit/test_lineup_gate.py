"""Does the lineup optimiser beat sorting a column?

`hub.season.lineup` is 97% covered by unit tests, which is a claim about its internals and not
about whether it is right. The draft board was also well covered while recommending a fourth
quarterback. So it gets a gate: a model is tested against the simplest thing that already
works, which here is "start your highest projections".

Everything below runs offline. The statistics have to be exercisable without nflverse, or the
gate is one nobody re-runs -- which is how P0 ended up unreproducible.
"""
import polars as pl
import pytest

from hub.names import player_key
from hub.season import lineup_gate as lg


def _realised(rows):
    return pl.DataFrame({"player": [player_key(r[0]) for r in rows],
                         "week": [r[1] for r in rows],
                         "points": [r[2] for r in rows]},
                        schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})


# --- the pre-registered actions, and that they are wired to the shared rule ------
#
# The three branches themselves -- including the boundary where an interval endpoint is
# exactly zero -- are tested once in `test_experiment.py`, because there is one rule now
# (ADR-0019). What is this gate's own is which sentence each branch produces. Since #135
# there is no per-gate `verdict` wrapper either: the rule and the sentences meet inside
# `experiment.run_gate`, which `main` calls with this gate's `ACTIONS`.

def _verdict(summary, seasons):
    from hub.models.experiment import gate
    return gate(summary, seasons, lg.ACTIONS)


def _gate_run(paired, *, ceiling_arm=lg.DECLARED_CEILING_ARM, **kw):
    """This gate's call, as `main` spells it, with the width history pointed nowhere."""
    from hub.models.experiment import SEASON_CLUSTER, run_gate
    return run_gate(paired, cluster=SEASON_CLUSTER, actions=lg.ACTIONS, name="lineup",
                    arm_a="optimiser", arm_b="projections", unit=lg.UNIT, bootstrap=200,
                    ceiling=lg.declared_ceiling(paired, ceiling_arm=ceiling_arm),
                    record_width=False, **kw)


def _yrs(gains):
    return pl.DataFrame({"season": list(range(2022, 2022 + len(gains))),
                         "gain": [float(g) for g in gains], "n": [10] * len(gains)})


def _sum(lo, hi):
    return {"n": 80.0, "clusters": 80.0, "mean": (lo + hi) / 2, "lo": lo, "hi": hi,
            "p_better": 0.5}


def test_an_interval_above_zero_in_every_season_trusts_the_optimiser():
    status, said = _verdict(_sum(0.5, 3.0), _yrs([0.4, 0.6, 0.9]))
    assert status == "ADOPT" and said.startswith("TRUST")


def test_an_interval_containing_zero_says_start_your_projections():
    """The likely branch, and it has an action rather than being a disappointment."""
    status, said = _verdict(_sum(-1.0, 2.0), _yrs([0.4, -0.6, 0.9]))
    assert status == "SHOW" and said.startswith("START YOUR PROJECTIONS")


def test_an_interval_below_zero_in_every_season_removes_it():
    """Evidence demotes as well as promotes -- the asymmetry P0's rule originally lacked."""
    status, said = _verdict(_sum(-3.0, -0.5), _yrs([-0.4, -0.6, -0.9]))
    assert status == "REMOVE" and said.startswith("REMOVE")


def test_adr_0012s_own_numbers_still_read_as_start_your_projections():
    """Regression on the recorded result: +0.00, CI [-0.00, +0.00] over four seasons.
    Unifying the rule tightened this gate, and it must not have moved what it published."""
    status, said = _verdict(_sum(-0.00, 0.00), _yrs([0.0, 0.0, 0.0, 0.0]))
    assert status == "SHOW" and said.startswith("START YOUR PROJECTIONS")


# --- the two arms ---------------------------------------------------------
#
# Both see ONLY projections. An earlier version of this file scored the optimiser arm on
# realised weekly scores, which gave it information the baseline did not have: it measured
# the value of perfect foresight (+31 points a game) and could not fail. The optimiser's sole
# advantage over sorting is that it reads `sd` as well as `mu`.

def _roster(sd=None):
    """QB1 RB2 WR3 TE1 + flex candidates. (player, pos, mu, sd)."""
    base = [("QB1", "QB", 20.0), ("RB1", "RB", 18.0), ("RB2", "RB", 14.0),
            ("WR1", "WR", 16.0), ("WR2", "WR", 13.0), ("WR3", "WR", 11.0),
            ("TE1", "TE", 9.0), ("WR4", "WR", 8.0), ("RB3", "RB", 7.0)]
    sd = sd or {}
    return [(n, p, m, sd.get(n, 2.0)) for n, p, m in base]


def _flat(r, pts=10.0, extra=()):
    rows = [(n, w, pts) for n, _, _, _ in r for w in range(1, 15)]
    return _realised(rows + list(extra))


def test_the_baseline_starts_the_highest_projections():
    """Eight slots at ten points each. RB3 is projected lowest, so he sits."""
    r = _roster()
    grid = lg.weekly_grid([n for n, _, _, _ in r], _flat(r))
    got = lg.projection_lineup_points(grid, [p for _, p, _, _ in r], [m for _, _, m, _ in r])
    assert got == pytest.approx(80.0)


def test_the_baseline_lineup_does_not_change_week_to_week():
    """A projection does not change, so neither does the lineup it implies. Giving the
    baseline a weekly choice would be scoring it as an optimiser."""
    r = _roster()
    grid = lg.weekly_grid([n for n, _, _, _ in r], _flat(r, extra=[("RB3", 5, 500.0)]))
    got = lg.projection_lineup_points(grid, [p for _, p, _, _ in r], [m for _, _, m, _ in r])
    assert got == pytest.approx(80.0), "a benched player's spike must not be collected"


def test_the_optimiser_cannot_see_realised_scores():
    """The defect this file was rewritten to remove. Moving a huge week onto a player the
    optimiser did not start must not change what it starts -- it never saw it."""
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    quiet = lg.weekly_grid(names, _flat(r))
    spike = lg.weekly_grid(names, _flat(r, extra=[("RB3", 5, 500.0)]))
    a = lg.optimiser_lineup_points(quiet, names, pos, mu, sd)
    b = lg.optimiser_lineup_points(spike, names, pos, mu, sd)
    # RB3 is the lowest projection and stays benched in both, so the spike is never collected.
    assert a == pytest.approx(b)


def test_the_optimiser_can_prefer_upside_over_projection():
    """The hypothesis under test, isolated. Against a strong opponent, a lower-mu player with
    much larger sd can carry a higher chance of winning -- and sorting on mu cannot see it."""
    from hub.season.lineup import optimize
    players = pl.DataFrame({
        "player": ["QB1", "RB1", "RB2", "WR1", "WR2", "WR3", "TE1", "STEADY", "SWINGY"],
        "pos": ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "WR", "WR"],
        "mu": [20.0, 18.0, 14.0, 16.0, 13.0, 11.0, 9.0, 9.0, 8.0],
        "sd": [3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 30.0],
    })
    # a hopeless matchup: you need variance, not expectation
    started = optimize(players, opp_mu=250.0, opp_sd=25.0)["starters"]["player"].to_list()
    assert "SWINGY" in started and "STEADY" not in started


def test_the_two_arms_agree_when_variance_is_uniform():
    """With identical sd there is nothing for variance-awareness to exploit, so the optimiser
    must not manufacture a difference out of tie-breaking."""
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    grid = lg.weekly_grid(names, _flat(r))
    assert lg.optimiser_lineup_points(grid, names, pos, mu, sd) == pytest.approx(
        lg.projection_lineup_points(grid, pos, mu))


def test_a_roster_that_cannot_field_a_lineup_scores_zero():
    """Nine quarterbacks fills no RB slot. The gate must report, not raise."""
    r = [(f"QB{i}", "QB", 10.0, 2.0) for i in range(9)]
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    grid = lg.weekly_grid(names, _flat(r))
    assert lg.optimiser_lineup_points(grid, names, pos,
                                      [m for _, _, m, _ in r], [v for _, _, _, v in r]) == 0.0


def test_a_player_who_never_played_scores_zero_not_null():
    r = _roster()
    rows = [(n, w, 10.0) for n, _, _, _ in r[:-1] for w in range(1, 15)]
    grid = lg.weekly_grid([n for n, _, _, _ in r], _realised(rows))
    assert grid[-1].sum() == 0.0


def test_points_are_per_game():
    r = _roster()
    names, pos = [n for n, _, _, _ in r], [p for _, p, _, _ in r]
    mu, sd = [m for _, _, m, _ in r], [v for _, _, _, v in r]
    g7 = lg.weekly_grid(names, _flat(r), weeks=7)
    g14 = lg.weekly_grid(names, _flat(r), weeks=14)
    assert lg.optimiser_lineup_points(g7, names, pos, mu, sd) == pytest.approx(
        lg.optimiser_lineup_points(g14, names, pos, mu, sd))


# --- the paired comparison ------------------------------------------------

def test_compare_pairs_one_row_per_roster():
    r = _roster()
    real = _flat(r)
    got = lg.compare({2024: [r, r], 2025: [r]}, {2024: real, 2025: real})
    assert got.height == 3
    assert set(got["season"].to_list()) == {2024, 2025}
    assert (got["diff"].abs() < 1e-9).all()


def test_an_empty_roster_does_not_crash_the_gate():
    """A season where the draft produced nothing must report, not raise."""
    got = lg.compare({2024: [[]]}, {2024: _realised([])})
    assert got.height == 1
    assert got["projection"][0] == 0.0


# --- the gate reads its inputs the way the live tool does -----------------

def test_the_gate_uses_the_same_moments_object_the_simulator_does():
    """mu and sd come from `predict.moments`, not a private copy -- so the optimiser is fed
    exactly what it is fed live, and the gate cannot pass by being handed nicer numbers."""
    import inspect

    from hub.season import lineup_gate
    src = inspect.getsource(lineup_gate.main)
    assert "from hub.models.predict import moments" in src


def test_a_roster_of_one_position_still_reports():
    """Nine quarterbacks fills no RB slot, so the optimiser has no legal lineup. The paired
    frame must still have a row -- a gate that drops its failures overstates its arm."""
    r = [(f"QB{i}", "QB", 10.0, 2.0) for i in range(9)]
    got = lg.compare({2024: [r]}, {2024: _flat(r)})
    assert got.height == 1
    assert got["optimiser"][0] == 0.0


# --- the ceiling is a variance oracle, not foresight (issue #43) -----------

def _volatile_roster():
    """A roster where the projection is right on average and wrong about spread.

    Every player projects the same mu, so sorting on projection cannot tell them apart and
    the only information available is how much they swing. That is the quantity this gate's
    ceiling is meant to bound, isolated.
    """
    names = [("QB1", "QB"), ("RB1", "RB"), ("RB2", "RB"), ("WR1", "WR"), ("WR2", "WR"),
             ("WR3", "WR"), ("TE1", "TE"), ("WR4", "WR"), ("RB3", "RB")]
    return [(n, p, 12.0, 2.0) for n, p in names]


def _swingy(roster, weeks=14):
    """Realised weeks where half the roster is steady and half alternates hard."""
    rows = []
    for i, (n, _p, _m, _s) in enumerate(roster):
        for w in range(1, weeks + 1):
            pts = 12.0 if i % 2 == 0 else (24.0 if w % 2 == 0 else 0.0)
            rows.append((n, w, pts))
    return _realised(rows)


def test_the_oracle_changes_only_the_spread():
    """Criterion two. Both arms already see the same `mu`; the optimiser's only advantage is
    that it reads `sd`. A ceiling that also knew `mu` would bound a different question --
    the one this module records an earlier arm accidentally measuring at +31 a game."""
    import inspect

    from hub.season import lineup_gate as lg
    roster = _volatile_roster()
    grid = lg.weekly_grid([n for n, _, _, _ in roster], _swingy(roster), 14)
    names = [n for n, _, _, _ in roster]
    pos = [p for _, p, _, _ in roster]
    mu = [m for _, _, m, _ in roster]

    # The projection reaches the oracle untouched: same object, not a recomputed one.
    src = inspect.getsource(lg.variance_oracle_points)
    assert "grid.mean" not in src, "the oracle is reading realised means, so it is foresight"
    got = lg.variance_oracle_points(grid, names, pos, mu)
    assert got > 0


def _separating_roster():
    """A roster and a season on which the two candidate ceiling arms give different numbers.

    A contested flex and a real bench, which is what makes the two arms able to differ at all:
    with a roster the size of the lineup everybody starts and every arm scores the same. `S`
    is steady and better; `V` swings to a lower mean. Both project the same `mu`, so only the
    realised numbers separate them -- and they separate them differently depending on which
    realised number an arm is allowed to see.

    Shared by the two tests that hold the arms apart, because a fixture on which they
    coincided would let either of them pass while proving nothing -- which is the failure
    `_contested_flex` has for this particular question: there both arms take 85.0.
    """
    base = [("QB1", "QB"), ("RB1", "RB"), ("RB2", "RB"), ("WR1", "WR"), ("WR2", "WR"),
            ("TE1", "TE"), *[(f"F{i}", "WR") for i in range(4)]]
    roster = [(n, p, 22.0, 2.0) for n, p in base] + [("S", "WR", 22.0, 2.0),
                                                     ("V", "WR", 22.0, 2.0)]
    rows = []
    for n, _p, _m, _s in roster:
        for w in range(1, 15):
            pts = (25.0 if n == "S" else 40.0 if (n == "V" and w % 2 == 0)
                   else 0.0 if n == "V" else 1.0 if n.startswith("F") else 10.0)
            rows.append((n, w, pts))
    return roster, _realised(rows)


def test_full_foresight_is_strictly_larger_than_the_variance_oracle():
    """Criterion three. The two must not be quietly interchangeable: a ceiling that drifted
    into full foresight would make this gate look powered when it was not."""
    from hub.season import lineup_gate as lg
    roster, real = _separating_roster()
    names = [n for n, _, _, _ in roster]
    pos = [p for _, p, _, _ in roster]
    mu = [m for _, _, m, _ in roster]
    grid = lg.weekly_grid(names, real, 14)

    oracle = lg.variance_oracle_points(grid, names, pos, mu)
    foresight = lg.foresight_lineup_points(grid, pos)
    assert foresight > oracle, (
        f"full foresight ({foresight:.2f}) did not beat the variance oracle ({oracle:.2f}), "
        f"so the two bound the same thing and one of them is mislabelled")


def _contested_flex():
    """A roster whose last starting slot only a true spread gets right, and the season for it.

    Every player projects the same `mu` and carries the same `sd`, so neither sorting on
    projection nor an optimiser reading a flat spread can tell the three flex candidates
    apart: **the two arms tie by construction and this gate's own effect is exactly zero.**
    The realised weeks then separate them. `SWINGY` alternates 0 and 30 for a mean of 15
    where both alternatives sit flat at 5, and eight starters at a projected ten apiece is
    below `OPP_MU`, so a rule that could read the real spread starts him and collects it.

    The zero effect is the fixture's whole point. `_volatile_roster` above scores 96.0 on the
    baseline, the optimiser *and* the oracle, so `ceiling >= effect` holds there whatever the
    ceiling arm reads -- a passing assertion over three copies of one number. Here the two
    are +0.00 and +10.00 and the comparison has something to fail on.
    """
    base = [("QB1", "QB"), ("RB1", "RB"), ("RB2", "RB"), ("WR1", "WR"), ("WR2", "WR"),
            ("WR3", "WR"), ("TE1", "TE"), ("STEADY", "WR"), ("SWINGY", "WR"),
            ("SPARE", "WR")]
    roster = [(n, p, 10.0, 2.0) for n, p in base]
    rows = []
    for n, _p in base:
        for w in range(1, 15):
            pts = (30.0 if w % 2 == 0 else 0.0) if n == "SWINGY" else \
                  5.0 if n in ("STEADY", "SPARE") else 10.0
            rows.append((n, w, pts))
    return roster, _realised(rows)


def test_the_gate_reports_its_ceiling_beside_its_effect():
    """Criterion one, in this gate's own units -- points per team game, which are not the
    draft backtest's and are never compared against them.

    The numbers are the assertion: the baseline and the optimiser both take 75.0 a game and
    the variance oracle takes 85.0, so the effect is +0.00 against a ceiling of +10.00. A
    fixture on which those two coincide would let a ceiling arm that had quietly become the
    control arm pass this.
    """
    roster, real = _contested_flex()
    got = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True)
    assert {"oracle", "ceiling_diff"} <= set(got.columns)
    assert got["projection"][0] == pytest.approx(75.0)
    assert got["optimiser"][0] == pytest.approx(75.0)
    assert got["oracle"][0] == pytest.approx(85.0)
    assert got["diff"][0] == pytest.approx(0.0)
    assert got["ceiling_diff"][0] == pytest.approx(10.0)
    assert got["ceiling_diff"][0] > got["diff"][0]
    plain = lg.compare({2024: [roster]}, {2024: real}, weeks=14)
    assert "oracle" not in plain.columns, "the ceiling must be opt-in, not a shape change"


# --- which arm the ceiling is, which is #138 and not this code's to decide -----
#
# `docs/gate-power.md` pre-registers stage 2 against a *foresight* ceiling; #43 deliberately
# built a variance oracle and argued for it. The two disagree about whether this gate is
# runnable, because the ceiling is the denominator stage 2 divides by. #45 builds the
# mechanism and takes the arm as a parameter; #138 answers which arm is declared.
#
# So what is asserted here is that the choice is *reachable and honest* -- both arms run, they
# genuinely differ, the printed line names which one ran, and the default is the status quo.
# Nothing here asserts which arm is right.


def test_both_ceiling_arms_are_reachable_and_give_different_ceilings():
    """The parameter is real only if the two arms disagree, so this runs on the fixture built
    to separate them -- not on `_contested_flex`, where both arms take 85.0 and this test
    would pass against a parameter that selected nothing at all."""
    roster, real = _separating_roster()
    oracle = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True,
                        ceiling_arm="variance-oracle")
    sight = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True,
                       ceiling_arm="foresight")
    assert sight["oracle"][0] > oracle["oracle"][0], (
        "the two arms produced the same ceiling, so the parameter selects nothing")
    # The arm under test and the incumbent are untouched by the choice: only the bound moves.
    assert sight["diff"][0] == pytest.approx(oracle["diff"][0])


def test_the_declared_arm_is_the_default_and_is_the_variance_oracle_today():
    """The default is a statement of the status quo -- what #43 shipped and what this gate has
    been running -- and not of #138's answer. If #138 lands and this changes, it changes
    here and nowhere else."""
    assert lg.DECLARED_CEILING_ARM == "variance-oracle"
    roster, real = _contested_flex()
    default = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True)
    named = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True,
                       ceiling_arm=lg.DECLARED_CEILING_ARM)
    assert default["oracle"][0] == pytest.approx(named["oracle"][0])


def test_one_line_switches_every_path_that_names_an_arm():
    """#138's answer has to be a one-line change, which is only true if nothing else in the
    tree names an arm. `DECLARED_CEILING_ARM` is the default of `compare`, of
    `declared_ceiling` and of the CLI flag, so all three follow it."""
    import inspect

    for fn in (lg.compare, lg.declared_ceiling):
        default = inspect.signature(fn).parameters["ceiling_arm"].default
        assert default == lg.DECLARED_CEILING_ARM, fn.__name__
    assert set(lg.CEILING_ARMS) == set(lg.CEILING_ARM_NAMES)
    assert lg.DECLARED_CEILING_ARM in lg.CEILING_ARMS


def test_the_printed_line_names_the_arm_that_actually_ran():
    """A run under a non-default arm must not be readable as a run under the default one --
    the ceiling's whole use is as a bound, and two arms bound different questions."""
    roster, real = _contested_flex()
    for arm, expected in [("variance-oracle", "not foresight"),
                          ("foresight", "full foresight")]:
        paired = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True,
                            ceiling_arm=arm)
        said = "\n".join(_gate_run(paired, ceiling_arm=arm).lines)
        assert expected in said, arm


# --- and an operator can ask for it (issue #134) --------------------------
#
# #43 built the arm and left it unreachable: `compare` took the option and nothing passed it
# one, so `docs/gate-power.md` stage 2 -- which compares each gate's MDE against its own
# ceiling -- could not be run for this gate at all. What that stage needs is the number said
# beside the effect it bounds, in the unit this harness measures in.

def test_asking_for_the_ceiling_does_not_move_the_effect_the_gate_reports():
    """Asserted rather than assumed, and on the frame where the ceiling is a real number: an
    all-flat roster holds this no matter what the extra arm does to the columns beside it."""
    roster, real = _contested_flex()
    plain = lg.compare({2024: [roster]}, {2024: real}, weeks=14)
    withc = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True)
    assert plain["projection"].to_list() == withc["projection"].to_list()
    assert plain["optimiser"].to_list() == withc["optimiser"].to_list()
    assert plain["diff"].to_list() == withc["diff"].to_list(), (
        "asking for the ceiling changed the effect the gate reports")


def test_the_ceiling_line_names_its_arm_its_unit_and_that_it_travels_nowhere():
    """Stage 2 reads three gates' ceilings and holds each against its own MDE. Three numbers
    in three units under one word is the confusion the line is written to prevent, so it
    carries the arm and the unit rather than a bare figure -- and this gate's arm is a
    perfect *spread*, which `foresight_lineup_points` exists to stop it being mistaken for.
    """
    roster, real = _contested_flex()
    paired = lg.compare({2024: [roster]}, {2024: real}, weeks=14, ceiling=True)
    said = [ln for ln in _gate_run(paired).lines if "ceiling" in ln.lower()]
    assert len(said) == 1, "a ceiling that bounds the effect says one thing and no more"
    assert "+10.00" in said[0]
    assert lg.UNIT in said[0]
    assert "not foresight" in said[0]
    assert "not comparable" in said[0]


def test_a_ceiling_that_does_not_bound_the_effect_says_so_loudly():
    """A bound that does not bound reads exactly like a tight one, and a tight one is what
    would license a verdict nothing supports. The draft gate has said this since #42; until
    #134 neither season gate could say it at all, and since #135 all three say it through
    one function."""
    paired = pl.DataFrame({"season": [2024, 2024], "roster": [0, 1], "diff": [4.0, 6.0],
                           "ceiling_diff": [1.0, 1.0]})
    said = [ln for ln in _gate_run(paired).lines if "CEILING BELOW THE EFFECT" in ln]
    assert len(said) == 1
    assert "+1.00" in said[0] and "+5.00" in said[0]


def test_a_run_that_asked_for_no_ceiling_prints_no_ceiling_line():
    """Not a blank and not a `nan` set against a unit -- the shape `paired_report` already
    uses for a field nothing computed, because `nan` beside a unit reads as a measurement."""
    roster, real = _contested_flex()
    plain = lg.compare({2024: [roster]}, {2024: real}, weeks=14)
    assert lg.declared_ceiling(plain) is None
    assert not [ln for ln in _gate_run(plain).lines if "ceiling" in ln.lower()]


# --- what produced these rows (issue #133) --------------------------------
#
# The gate wrote a bare parquet: no configuration digest, no data digest. Two runs over
# different archives were therefore indistinguishable after the fact, which is the silent
# case the pinning layer exists to remove -- and this is the gate ADR-0012 defers to.
#
# Since #135 the stamp is the run's, so "every gate stamps the data it scored against" is a
# property of `experiment.run_gate` held in `tests/unit/test_gate_run.py`. What stays here is
# that *this* gate's frame comes back stamped and that the file it writes is that frame.

def test_the_published_frame_names_the_config_and_the_data_that_made_it(tmp_path,
                                                                        monkeypatch):
    """The file a reader is left with has to say what it was scored against.

    Read back off disk rather than off the returned frame, because what a later reader opens
    is the parquet, and a stamp that lived only in memory would satisfy every assertion about
    the return value while publishing the same bare rows as before.
    """
    from hub.config import UNPINNED
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    paired = pl.DataFrame({"season": [2024, 2025], "roster": [0, 0], "diff": [1.0, -2.0]})
    out = tmp_path / "lineup_paired.parquet"

    run = _gate_run(paired)
    run.stamped.write_parquet(out)
    back = pl.read_parquet(out)

    assert back.height == paired.height
    assert {"cfg_digest", "data_digest"} <= set(back.columns)
    assert back["data_digest"].unique().to_list() == [UNPINNED], (
        "a run that pinned nothing must say so rather than publishing eight characters that "
        "look like a digest of data")
    assert "nothing was loaded through the pinning layer" in "\n".join(run.lines)


def test_a_pinned_load_moves_this_gates_data_digest_and_leaves_its_config_alone(monkeypatch):
    """The property a reader acts on: two runs over different archives do not carry the same
    stamp, and the archive moving does not pretend the model moved with it."""
    from hub.config import UNPINNED
    from hub.fetch import nflverse as nv

    paired = pl.DataFrame({"season": [2024], "roster": [0], "diff": [1.0]})
    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    unpinned = _gate_run(paired).stamped

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {
        "entry": nv.Pin(source="player_stats", as_of="2026-09-04", digest="abcd1234",
                        rows=10, pinned_at=None)})
    run = _gate_run(paired)
    pinned = run.stamped

    assert unpinned["data_digest"][0] == UNPINNED
    assert pinned["data_digest"][0] != UNPINNED
    assert "over 1 pinned source(s)" in "\n".join(run.lines)
    assert pinned["cfg_digest"][0] == unpinned["cfg_digest"][0]


def test_this_gate_stamps_by_the_one_rule_rather_than_by_a_second_copy_of_it():
    """Two gates that agree because somebody keeps them agreeing is the arrangement this
    module already records losing: `cohort` is imported rather than restated because the
    recipe had been written out twice. The same argument decides the stamp -- and since #135
    the stamp is not addressed through a draft module at all. Nothing in this module names a
    digest column; the run does."""
    import inspect

    src = inspect.getsource(lg)
    assert "stamped_for_publication" not in src
    assert "cfg_digest" not in src and "data_digest" not in src
    assert "run_gate(" in src


# --- the gate names its own reads, however it is invoked (issue #247) ---------------------
#
# #192 scoped `backtest.main` and `weekly_screen.main` with `reads_of_one_run` and left the two
# season-side gates unwired. Called in-process by anything that has already read something --
# the one gate run of #135, a notebook -- this gate inherited the enclosing run's reads and
# published a digest over bytes it never touched, a failure that looks exactly like a clean
# digest.

def _in_process_gate(monkeypatch, tmp_path, *, inner):
    """Drive `main` with the network replaced.

    The walk-forward loader records `inner` as the one read it made and hands back a board;
    the moments, the Cohort and `compare` are replaced with the smallest thing `main` will
    accept, since each is exercised above and the seam here is what the stamp names. The
    width history is pointed nowhere.
    """
    from functools import partial

    from hub.draft import cohort as cohort_mod
    from hub.fetch import nflverse as nv
    from hub.models import predict
    from hub.models.experiment import run_gate

    board = pl.DataFrame({"player": ["A", "B"], "pos": ["QB", "RB"],
                          "proj_ppg": [20.0, 15.0], "games": [16, 16]})

    def loads(seasons, load, *, on_season=None):
        nv._remember(tmp_path / "the-gates-own-entry.parquet", inner)
        # `load` is `main`'s own `_as_of_keeping_the_report`; calling it is what fills the
        # report dict the Cohort is drafted with.
        return ({yr: load(yr)[0] for yr in seasons},
                {yr: _flat([("A", "QB", 20.0, 2.0), ("B", "RB", 15.0, 2.0)]) for yr in seasons})

    moments = pl.DataFrame({"player": ["A", "B"], "pos": ["QB", "RB"], "mu": [20.0, 15.0],
                            "sd": [2.0, 2.0], "games": [16, 16]})
    paired = pl.DataFrame({"season": [2024] * 4, "roster": [0, 1, 2, 3],
                           "projections": [10.0, 11.0, 9.0, 10.5],
                           "optimiser": [9.0, 10.0, 8.5, 9.0]})
    paired = paired.with_columns((pl.col("optimiser") - pl.col("projections")).alias("diff"))
    monkeypatch.setattr(lg, "board_as_of", lambda yr: (board, None))
    monkeypatch.setattr(lg, "walk_forward_inputs", loads)
    monkeypatch.setattr(predict, "moments", lambda board: moments)
    monkeypatch.setattr(cohort_mod, "cohort",
                        lambda *a, **k: cohort_mod.Cohort([[0, 1]], [[]], ["QB", "RB"]))
    monkeypatch.setattr(lg, "compare", lambda *a, **k: paired)
    monkeypatch.setattr(lg, "run_gate", partial(run_gate, record_width=False, bootstrap=100))
    out = tmp_path / "paired.parquet"
    assert lg.main(["--seasons", "2024", "--drafts", "1", "--out", str(out)]) == 0
    return pl.read_parquet(out)


def test_the_gate_called_in_process_names_only_its_own_reads(monkeypatch, tmp_path, capsys):
    """Called from inside a run that has already read something, the gate's published digest
    covers the gate's reads and not the enclosing run's -- and the enclosing run still ends
    up holding both, because a scope narrows what a component reports and must never be a
    way for a run to lose a read."""
    from hub.config import data_digest
    from hub.fetch import nflverse as nv

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    outer = nv.Pin(source="player_stats", as_of=None, digest="0ut51de0", rows=1,
                   pinned_at=None)
    inner = nv.Pin(source="ff_opportunity", as_of="2024-09-01", digest="1n51de01", rows=1,
                   pinned_at=None)
    # The enclosing run: something else in this process has read a source already.
    nv._remember(tmp_path / "the-enclosing-runs-entry.parquet", outer)

    stamped = _in_process_gate(monkeypatch, tmp_path, inner=inner)

    assert stamped["data_digest"].unique().to_list() == [data_digest([inner])], (
        "the gate's digest is not a digest over the gate's own read")
    assert stamped["data_digest"][0] != data_digest([outer, inner]), (
        "the gate published a digest over the enclosing run's reads as well as its own")
    assert "over 1 pinned source(s)" in capsys.readouterr().out
    assert sorted(p.source for p in nv.pins_this_run()) == ["ff_opportunity",
                                                            "player_stats"], (
        "the gate's read did not reach the run around it: scoping lost a read")
