"""One prediction object: what a player does in a week.

`docs/next.md`. The pieces existed but as three implementations of one idea --
`weekly_moments` handed back two moments, the skewed correlated draw lived inside
`simulate_weeks`, and `lineup.py` computed its own group spread. Three copies is how they
drift apart, and drift is silent.

The split is by subject rather than by convenience: **this module is what a player does, and
`hub.draft.season` is how a league works.** Dispersion laws, skew, talent and correlation
live here; rosters, schedules, brackets and lineups live there.

A constraint the unification has to respect. Component-derived spread was measured *worse*
than the fitted square-root law at predicting a player's actual weekly sd -- mean error 1.365
against 1.140, P(better) 0.0% (`docs/component-projection.md`). So the object keeps the
fitted laws for its moments and exposes components alongside them. Unifying must not quietly
swap a validated number for a tidier one.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import predict


def _frame():
    return pl.DataFrame({"proj_ppg": [18.0, 12.0, 5.0],
                         "position": ["QB", "RB", "WR"]})


# --- the object owns the fitted laws --------------------------------------

def test_moments_carry_mean_spread_and_skew():
    got = predict.moments(_frame())
    assert {"mu", "sd", "skew"} <= set(got.columns)


def test_spread_still_follows_the_square_root_law():
    """`sd = 0.55 * mu` assumed spread is proportional to the mean. Fitted against 1,174
    player-seasons of nflverse weekly scoring, the exponent is 0.498 +/- 0.012 -- flat on
    the Poisson value and about 42 standard errors from 1.

    That is not a curve-fitting accident, it is what aggregating component stats produces:
    weekly points are a sum of count-driven pieces (receptions, carries, touchdowns) whose
    variance grows with their mean, so the spread of the total grows with its square root.
    See docs/weekly-spread.md.
    """
    got = predict.moments(pl.DataFrame({"proj_ppg": [4.0, 16.0],
                                        "position": ["WR", "WR"]}))
    # four times the mean is twice the spread, not four times
    assert got["sd"][1] / got["sd"][0] == pytest.approx(2.0, rel=0.01)


def test_a_boom_bust_profile_is_a_property_of_the_level_not_a_setting():
    """The consequence that matters for the draft board: relative volatility falls as
    projection rises. A 5-point-a-game flier really is nearly twice the lottery, per point,
    that a 20-point-a-game starter is -- and the old constant said they were identical."""
    got = predict.moments(pl.DataFrame({"proj_ppg": [5.0, 20.0],
                                        "position": ["RB", "RB"]}))
    cv = (got["sd"] / got["mu"]).to_list()
    assert cv[0] > 1.8 * cv[1]


def test_positions_keep_their_own_coefficient():
    got = predict.moments(pl.DataFrame({"proj_ppg": [10.0, 10.0],
                                        "position": ["QB", "WR"]}))
    assert got["sd"][0] < got["sd"][1]
    assert predict.WEEKLY_K["WR"] > predict.WEEKLY_K["QB"]


def test_a_missing_position_falls_back_to_the_pooled_coefficient():
    xp = pl.DataFrame({"proj_ppg": [9.0]},
                      schema={"proj_ppg": pl.Float64}).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("position"))
    got = predict.moments(xp)
    assert got["sd"][0] == pytest.approx(predict.WEEKLY_K_POOLED * 3.0, rel=1e-6)


def test_a_player_projected_at_nothing_has_no_spread():
    """Under the old floor a zero-projection player still carried sd = 2.0, which the
    best-lineup rule turned into free points. sqrt(0) is 0."""
    got = predict.moments(pl.DataFrame({"proj_ppg": [0.0], "position": ["WR"]}))
    assert got["sd"][0] == 0.0


def test_skew_is_carried_per_position():
    got = predict.moments(_frame())
    by = dict(zip(got["position"].to_list(), got["skew"].to_list(), strict=True))
    assert by["QB"] < by["RB"]


def test_the_numbers_are_pinned():
    """A refactor that changes a number is not a refactor.

    This assertion used to compare `predict.moments` against `season.weekly_moments`, which
    was an *alias for the same object* -- it compared a function with itself and could not
    fail. Pin the values instead, so a change to the law has to be deliberate.
    """
    got = predict.moments(_frame())
    assert got["mu"].to_list() == [18.0, 12.0, 5.0]
    assert got["sd"].to_list() == pytest.approx(
        [7.976164491784255, 7.170690343335151, 4.762824792074552], rel=1e-12)
    assert got["skew"].to_list() == [0.15, 0.67, 0.66]


def test_there_is_only_one_implementation_of_the_weekly_moments():
    """`predict.weekly_moments` was a dead twin of `predict.moments` -- same law, no skew,
    zero callers. Two functions for one quantity is how they drift, and the drift is silent
    because whichever one a caller picked still returned plausible numbers."""
    assert not hasattr(predict, "weekly_moments")


def test_season_still_re_exports_for_back_compat():
    """Callers in calibrate, leverage and optimize import these from season today. Moving
    the definition must not break them."""
    from hub.draft import season
    assert season.TALENT_CV == predict.TALENT_CV
    assert season.WEEKLY_K == predict.WEEKLY_K
    assert season.talent_cv_for(np.array(["RB"]))[0] == predict.TALENT_CV_BY_POS["RB"]


# --- the draw ---------------------------------------------------------------

def test_a_draw_has_the_mean_spread_and_skew_it_was_asked_for():
    rng = np.random.default_rng(0)
    z = predict.correlated_normal(rng, (60000,), np.array(["WR"]), None)
    got = predict.skewed(12.0, 7.0, 0.66, z)
    assert got.mean() == pytest.approx(12.0, rel=0.03)
    assert got.std() == pytest.approx(7.0, rel=0.06)


def test_teammates_correlate_and_others_do_not():
    rng = np.random.default_rng(1)
    pos = np.array(["QB", "WR"])
    same = predict.correlated_normal(rng, (40000, 2), pos, np.array(["KC", "KC"]))
    apart = predict.correlated_normal(rng, (40000, 2), pos, np.array(["KC", "DEN"]))
    assert np.corrcoef(same[:, 0], same[:, 1])[0, 1] > 0.15
    assert abs(np.corrcoef(apart[:, 0], apart[:, 1])[0, 1]) < 0.05


# --- a block that will not factor ------------------------------------------
#
# Issue #171. A block that is not positive semi-definite used to be caught and skipped, so
# that team's players were drawn independently -- the exact model the correlation structure
# exists to replace -- with no counter, no warning and nothing recorded. The draw it produces
# has the shape and dtype of a correlated one, so the count is the only evidence there is.


def _star(pass_catchers: int) -> tuple[np.ndarray, np.ndarray]:
    """One team: a quarterback and `pass_catchers` receivers, as (pos, nfl_team).

    Only the quarterback's edges are non-zero, so the block is a star and is PSD exactly
    while the receivers' squared correlations sum below one. At +0.232 that turns over
    between 18 and 19 receivers -- an absurd roster, and the point: this is a numerical
    property of `TEAMMATE_RHO` rather than a mock, so a refit that makes real blocks fail
    is caught by the same code path these tests drive.
    """
    pos = np.array(["QB"] + ["WR"] * pass_catchers)
    return pos, np.array(["ONE"] * (pass_catchers + 1))


def _block(pos) -> np.ndarray:
    """The correlation block `correlated_normal` builds for one team's positions.

    Built from the shipped `TEAMMATE_RHO` rather than written out, so a refit that changes
    what a real board's blocks look like changes what these tests are handed.
    """
    pos = list(pos)
    r = np.eye(len(pos))
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            r[i, j] = r[j, i] = predict.teammate_rho(pos[i], pos[j])
    return r


def _bipartite(quarterbacks: int, catchers: int) -> np.ndarray:
    """The real failure mode from #171: a team carrying a second quarterback.

    One quarterback makes the block a star, which is PSD until its catchers' squared
    correlations sum past one -- around nineteen receivers, so never. Two make it bipartite:
    each quarterback correlates with every catcher and with the other at zero, and the bound
    halves. All four blocks that failed on the live board of 2026-09-07 are this shape.
    """
    return _block(["QB"] * quarterbacks + ["WR"] * catchers)


def _teams(good: int, bad_pass_catchers: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """`good` two-man teams that factor, plus optionally one that does not."""
    pos: list[str] = []
    team: list[str] = []
    for i in range(good):
        pos += ["QB", "WR"]
        team += [f"T{i}", f"T{i}"]
    if bad_pass_catchers:
        p, t = _star(bad_pass_catchers)
        pos += p.tolist()
        team += t.tolist()
    return np.array(pos), np.array(team)


def test_a_block_that_will_not_factor_is_repaired_rather_than_dropped():
    """Issue #187, which changes #171's answer without touching #171's question.

    The block still will not factor. What used to happen next was a fall back to
    independence; what happens now is a repair to the nearest valid correlation matrix, so
    the team is drawn correlated under a matrix a *stated* distance from the fitted one.
    `independent` goes to zero by repair rather than by silence.
    """
    pos, team = _teams(good=20, bad_pass_catchers=19)
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, pos.size), pos, team,
                              report=report)
    assert report.blocks == 21, "every block carrying a correlation should be counted"
    assert report.independent == 0, "a repairable block must not fall back to independence"
    assert report.repaired == 1
    assert report.repaired_share == pytest.approx(1 / 21)


def test_the_repair_is_recorded_per_team_with_the_size_of_the_change():
    """Pre-registered in #187: if repair changes a block materially, the size of the change
    is published per team. Recorded whether or not it looks material -- the judgement of
    whether 0.013 matters is the reader's, and it cannot be made against a figure that was
    withheld for being small."""
    pos, team = _teams(good=20, bad_pass_catchers=19)
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, pos.size), pos, team,
                              report=report)
    assert set(report.repairs) == {"ONE"}, "the repair is filed under the team that needed it"
    got = report.repairs["ONE"]
    assert got.min_eig_before < 0.0, "the block being repaired is the one that will not factor"
    assert got.min_eig_after >= 0.0, "and the repaired one is a valid correlation matrix"
    assert got.moved > 0.0, "a repair that moved nothing is a repair that did not happen"
    assert got.size == 20
    lines = "\n".join(report.repair_lines())
    assert "ONE" in lines and f"{got.moved:.4f}" in lines


def test_a_repaired_run_is_distinguishable_from_one_that_needed_no_repair():
    """#171's defect, carried forward: both runs return an array of the same shape and
    dtype, so the report is the only thing that can tell them apart. Repair does not put
    that back -- it changes what the run has to say, not whether it says anything."""
    clean_pos, clean_team = _teams(good=21)
    fixed_pos, fixed_team = _teams(good=20, bad_pass_catchers=19)
    clean = predict.CorrelationReport()
    fixed = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, clean_pos.size),
                              clean_pos, clean_team, report=clean)
    predict.correlated_normal(np.random.default_rng(0), (4, fixed_pos.size),
                              fixed_pos, fixed_team, report=fixed)
    assert not clean.repairs and clean.repair_lines() == []
    assert fixed.repairs
    assert clean.note() != fixed.note()
    assert "repair" in fixed.note()


def test_a_block_that_cannot_be_repaired_still_falls_back_and_still_voids(monkeypatch):
    """**#171's counter has to stay able to fire.** Repair is attempted and checked, not
    assumed: a block that will not factor *after* repair is still drawn independently, still
    counted, and still refuses past the floor.

    A NaN correlation is the case that gets there -- a refit that writes one into
    `TEAMMATE_RHO` produces a block no projection can mend, as against the merely non-PSD
    block above, which the repair handles. Without a path like this the void would be a
    check that cannot fire, which is what #187's decision says to avoid.
    """
    monkeypatch.setitem(predict.TEAMMATE_RHO, ("QB", "WR"), float("nan"))
    pos, team = _star(3)
    with pytest.raises(predict.CorrelationVoid, match="would not factor"):
        predict.correlated_normal(np.random.default_rng(0), (4, pos.size), pos, team)


def test_the_repaired_block_is_a_valid_correlation_matrix():
    """Not merely factorable: unit diagonal, symmetric, and PSD. A repair that returned a
    covariance matrix would factor and would silently rescale every player's spread."""
    r = _bipartite(quarterbacks=2, catchers=10)
    assert np.linalg.eigvalsh(r).min() < 0.0, "the fixture has to be broken to be repaired"
    fixed = predict.nearest_correlation(r)
    assert np.allclose(np.diag(fixed), 1.0)
    assert np.allclose(fixed, fixed.T)
    assert np.linalg.eigvalsh(fixed).min() >= 0.0
    np.linalg.cholesky(fixed)


def test_the_repair_is_nearer_than_a_bare_eigenvalue_clip():
    """"Nearest" is the claim, so it is the thing to test. Clipping the eigenvalues at zero
    is the obvious one-liner and is *not* nearest: rescaling its diagonal back to one moves
    it again, by an amount that one-liner never measures. Higham's alternating projections
    converge on the intersection of both constraints instead."""
    r = _bipartite(quarterbacks=2, catchers=10)
    w, v = np.linalg.eigh(r)
    clipped = (v * np.maximum(w, 0.0)) @ v.T
    d = np.sqrt(np.diag(clipped))
    clipped = clipped / np.outer(d, d)
    near = predict.nearest_correlation(r)
    once = predict.nearest_correlation(r, iterations=1)
    assert (np.linalg.norm(near - r, "fro")
            < np.linalg.norm(clipped - r, "fro")), "the repair should be the nearer of the two"
    # And the iteration has to be doing something. Without this the test passes on a single
    # projection, which already beats the clip above -- so it would hold "nearest" against
    # the one-liner while saying nothing about whether the algorithm converges. The margin
    # is small (about 8e-6 on this block, against correlations quoted to three decimals);
    # it is asserted because the docstring claims *nearest*, not because 8e-6 moves a roster.
    assert (np.linalg.norm(near - r, "fro")
            < np.linalg.norm(once - r, "fro")), "converging should beat one projection"
    assert np.linalg.eigvalsh(clipped).min() < np.linalg.eigvalsh(near).min(), (
        "and the clip lands on the boundary of the PSD cone, where a Cholesky is a coin "
        "toss -- which is what the eigenvalue floor in the repair exists to clear")


def test_two_teams_with_the_same_block_are_each_recorded_under_their_own_name():
    """The blocks are cached, because one run factors the same handful of them thousands of
    times. Two teams with the same positions build the *same* block and share a cache entry,
    so a cached team name would file both repairs under whichever team was seen first --
    and the per-team record exists precisely to say which team was repaired."""
    pos, _ = _star(19)
    both_pos = np.concatenate([pos, pos])
    both_team = np.array(["AAA"] * pos.size + ["ZZZ"] * pos.size)
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, both_pos.size),
                              both_pos, both_team, report=report)
    assert set(report.repairs) == {"AAA", "ZZZ"}
    assert report.repairs["AAA"].team == "AAA"
    assert report.repairs["ZZZ"].team == "ZZZ"


def test_the_block_cache_does_not_survive_a_refit(monkeypatch):
    """The cache is keyed on the block and not on the positions that built it. Keyed on
    positions it would serve a factor of the old numbers the moment `TEAMMATE_RHO` moved --
    so a refit, or a test that monkeypatches the table, would go on measuring the model
    nobody is running."""
    pos = np.array(["QB", "WR"])
    team = np.array(["ONE", "ONE"])
    kw = {"pos": pos, "nfl_team": team}
    before = predict.correlated_normal(np.random.default_rng(5), (20000, 2), **kw)
    monkeypatch.setitem(predict.TEAMMATE_RHO, ("QB", "WR"), 0.80)
    after = predict.correlated_normal(np.random.default_rng(5), (20000, 2), **kw)
    assert np.corrcoef(before[:, 0], before[:, 1])[0, 1] < 0.35
    assert np.corrcoef(after[:, 0], after[:, 1])[0, 1] > 0.70


def test_free_agents_are_not_teammates():
    """"FA" is a label in the team column, not a team. Two free agents share no quarterback,
    so correlating them is correlating strangers -- and on the 2024 board they are a single
    43-player block, the worst-conditioned on it by an order of magnitude.

    This is load-bearing *because* of the repair. Before it, that block failed to factor and
    fell back to independence, which is the right answer for a free agent and was arrived at
    by accident; repairing it instead would turn an accidentally-correct independent draw
    into a confidently-wrong correlated one.
    """
    pos = np.array(["QB", "WR", "QB", "WR"])
    report = predict.CorrelationReport()
    z = predict.correlated_normal(np.random.default_rng(2), (40000, 4), pos,
                                  np.array(["FA", "FA", "KC", "KC"]), report=report)
    assert report.blocks == 1, "only the real team carries a block"
    assert abs(np.corrcoef(z[:, 0], z[:, 1])[0, 1]) < 0.03, "two free agents stay independent"
    assert np.corrcoef(z[:, 2], z[:, 3])[0, 1] > 0.15, "the real team still correlates"


# --- a committee: a within-team pairing that carries a negative correlation ---
#
# Issue #187's first criterion. The structure could express only "these two rise together";
# a handcuff, a committee backfield and a target split all need "one rises when the other
# falls", and until the repair above existed a negative entry could not be carried at all --
# it is what makes a block non-PSD in the first place.
#
# **These fixtures set the correlation rather than reading a fitted one, and deliberately.**
# `docs/correlation.md` measures RB1-RB2 at +0.013 and puts every non-quarterback pairing
# inside +/-0.03 of zero, so there is no fitted negative within-team number to ship. What is
# tested here is that the machinery carries whichever sign a fit produces, and what the
# *direction* of the resulting combined distribution then is.


def _committee(rho: float, monkeypatch) -> None:
    """Two backs on one team who take from each other at `rho`."""
    monkeypatch.setitem(predict.TEAMMATE_RHO, ("RB", "RB"), rho)


def test_a_within_team_pairing_can_carry_a_negative_correlation(monkeypatch):
    """The criterion itself. The draw has to come back negatively correlated -- a structure
    that clamped the fit at zero, or repaired the sign away, would pass every PSD check and
    price a committee as two independent backs."""
    _committee(-0.30, monkeypatch)
    z = predict.correlated_normal(np.random.default_rng(7), (60000, 2),
                                  np.array(["RB", "RB"]), np.array(["ONE", "ONE"]))
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] == pytest.approx(-0.30, abs=0.02)


def test_a_committee_is_narrower_than_two_independent_backs_not_wider(monkeypatch):
    """**This falsifies #187's second pre-registered outcome.**

    The ticket pre-registers that *two players in a committee produce a wider combined
    distribution than two independent players with the same marginals*. Holding the marginals
    fixed, the combined variance is `s1^2 + s2^2 + 2*rho*s1*s2`, so the sign of the change is
    the sign of rho and nothing else. A committee is negatively correlated by the ticket's own
    definition -- "one rises when the other falls" -- so its combined distribution is
    strictly **narrower**. The two halves of the pre-registration contradict each other, and
    no implementation can satisfy both.

    Asserted in both directions so the test is not measuring its own fixture: the same code
    gives wider for a positive pairing and narrower for a negative one, which is what makes
    the negative case a finding rather than an artefact.
    """
    sd = 6.0
    pair = [(sd, "RB", "ONE"), (sd, "RB", "ONE")]
    apart = [(sd, "RB", "ONE"), (sd, "RB", "TWO")]

    _committee(-0.30, monkeypatch)
    committee = predict.group_sd(pair)
    independent = predict.group_sd(apart)
    assert independent == pytest.approx(sd * np.sqrt(2.0))
    assert committee < independent, "a negative pairing narrows the combined distribution"
    assert committee == pytest.approx(sd * np.sqrt(2.0 - 0.60))

    _committee(+0.30, monkeypatch)
    assert predict.group_sd(pair) > independent, (
        "and a positive one widens it -- so the direction tracks the fitted sign, and the "
        "assertion above is not an artefact of the fixture")


def test_the_committee_direction_holds_in_the_draw_and_not_only_in_the_formula(monkeypatch):
    """`group_sd` is a closed form, so on its own it re-derives the algebra above rather than
    testing the simulator. This measures the same direction on the drawn weekly points that a
    roster is actually priced on, transform and clip included."""
    _committee(-0.30, monkeypatch)
    pos = np.array(["RB", "RB"])
    mu, sd = np.array([12.0, 12.0]), np.array([6.0, 6.0])
    skew = predict.weekly_skew_for(pos)
    rng = np.random.default_rng(11)
    z_pair = predict.correlated_normal(rng, (200000, 2), pos, np.array(["ONE", "ONE"]))
    z_apart = predict.correlated_normal(rng, (200000, 2), pos, np.array(["ONE", "TWO"]))
    together = predict.skewed(mu, sd, skew, z_pair).sum(axis=1)
    apart = predict.skewed(mu, sd, skew, z_apart).sum(axis=1)
    assert together.std() < apart.std() * 0.95, (
        "the committee's combined distribution is narrower in the draw too")


def test_a_committee_block_that_will_not_factor_is_repaired_with_its_sign_intact(monkeypatch):
    """The negative pairing is what makes a block non-PSD, so the two halves of #187 meet
    here: the repair has to keep a committee a committee. A repair that dragged the pairing
    up through zero would satisfy every PSD check by deleting the thing being modelled."""
    _committee(-0.95, monkeypatch)
    monkeypatch.setitem(predict.TEAMMATE_RHO, ("RB", "WR"), 0.80)
    pos = np.array(["RB", "RB", "WR"])
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, 3), pos,
                              np.array(["ONE"] * 3), report=report)
    assert report.repaired == 1, "the fixture has to be broken to be repaired"
    assert report.independent == 0
    z = predict.correlated_normal(np.random.default_rng(13), (60000, 3), pos,
                                  np.array(["ONE"] * 3))
    assert np.corrcoef(z[:, 0], z[:, 1])[0, 1] < -0.30, (
        "the repaired committee is still a committee")


def test_the_floor_is_a_share_and_not_a_count():
    """One lost block in twenty-one is inside the floor and reported; two in twenty-two is
    outside it and refused. A guard keyed to the count alone would treat these the same."""
    inside_pos, inside_team = _teams(good=20, bad_pass_catchers=19)
    predict.correlated_normal(np.random.default_rng(0), (4, inside_pos.size),
                              inside_pos, inside_team)
    report = predict.CorrelationReport(blocks=22, independent=2)
    assert report.share > report.floor
    with pytest.raises(predict.CorrelationVoid):
        report.check()


def test_the_accounting_does_not_move_a_run_with_nothing_to_report():
    """A clean run is unchanged. The counters read the block the Cholesky already built,
    so nothing about the draw depends on whether a report was passed."""
    pos, team = _teams(good=8)
    kw = {"pos": pos, "nfl_team": team}
    with_report = predict.correlated_normal(np.random.default_rng(3), (200, pos.size),
                                            report=predict.CorrelationReport(), **kw)
    without = predict.correlated_normal(np.random.default_rng(3), (200, pos.size), **kw)
    assert np.array_equal(with_report, without)


def test_a_team_with_nothing_to_correlate_is_not_counted_as_a_block():
    """Two receivers correlate at zero, so their block is the identity and independence
    costs it nothing. Counting it would make the share read low exactly when the teams that
    did lose something were few."""
    report = predict.CorrelationReport()
    predict.correlated_normal(np.random.default_rng(0), (4, 2), np.array(["WR", "WR"]),
                              np.array(["ONE", "ONE"]), report=report)
    assert report.blocks == 0
    assert report.note() == "correlation: no team block carried a correlation to apply."


# --- components live on the same object -----------------------------------

def test_the_component_line_is_reachable_from_the_same_object():
    """The reason to unify: a consumer that wants stats rather than points -- the props
    audit, a future volume model -- reads them from here instead of reaching into
    hub.models.volume separately."""
    got = predict.components(pick=12, position="WR", proj_ppg=15.5)
    assert got["receiving_yards"] > 0
    from hub.models.components import points
    assert points(got) == pytest.approx(15.5, rel=1e-6)


def test_components_are_empty_for_a_position_with_no_fitted_curve():
    assert predict.components(pick=50, position="K", proj_ppg=8.0) == {}


# --- correlation is owned here, not by the scoring module -----------------

def test_teammate_correlation_is_owned_by_the_prediction_module():
    """It was defined in `hub.models.components` and used two ways -- analytically by
    `lineup.py` for a closed-form spread, and by sampling in the league simulator. Two uses
    of one table is fine; the table living in the *scoring* module is not. Scoring is how
    stats become points, and who moves together is a prediction."""
    assert predict.teammate_rho("QB", "WR") > 0.2
    assert predict.teammate_rho("WR", "WR") == 0.0


def test_group_spread_counts_teammate_covariance():
    lone = predict.group_sd([(7.0, "QB", "KC"), (6.0, "WR", "DEN")])
    stack = predict.group_sd([(7.0, "QB", "KC"), (6.0, "WR", "KC")])
    assert stack > lone


def test_the_scoring_module_does_not_re_export_correlation():
    """It briefly did, for one caller in `lineup.py`. That re-export made `predict` import
    its own symbol back out of `components` -- a cycle that only stayed legal because the
    import sat inside a function body. One owner, no round trip."""
    from hub.models import components as C
    assert not hasattr(C, "teammate_rho")
    assert not hasattr(C, "group_sd")
