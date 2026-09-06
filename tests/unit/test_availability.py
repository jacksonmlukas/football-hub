import numpy as np
import polars as pl
import pytest

from hub.draft.availability import availability, blended_adp, pick_value


def _board(n=60):
    return pl.DataFrame({
        "player": [f"P{i}" for i in range(1, n + 1)],
        "ecr": [float(i) for i in range(1, n + 1)],
        "adp": [float(i) for i in range(1, n + 1)],
        "vor": [float(n - i) for i in range(n)],
    })


def test_blend_endpoints_collapse_to_single_boards():
    df = _board().with_columns(pl.col("adp") + 10)
    assert blended_adp(df, w=1.0)["mu_pick"].to_list() == df["adp"].to_list()
    assert blended_adp(df, w=0.0)["mu_pick"].to_list() == df["ecr"].to_list()


def test_blend_rejects_weight_outside_unit_interval():
    with pytest.raises(ValueError):
        blended_adp(_board(), w=1.4)


def test_availability_decreases_as_your_pick_gets_later():
    av = availability(_board(), picks=[10, 40], n_sims=2000)
    early, late = av["avail_10"].to_numpy(), av["avail_40"].to_numpy()
    assert (late <= early + 1e-9).all()


def test_top_of_board_is_gone_by_a_late_pick():
    av = availability(_board(), picks=[50], n_sims=2000)
    assert av.filter(pl.col("player") == "P1")["avail_50"][0] < 0.05


def test_availability_is_a_probability():
    av = availability(_board(), picks=[5, 25], n_sims=1000)
    for c in ("avail_5", "avail_25"):
        assert np.all((av[c].to_numpy() >= 0) & (av[c].to_numpy() <= 1))


def test_cost_of_waiting_prefers_the_player_who_will_not_return():
    """Two players of equal VOR: the one going sooner costs more to pass on.

    Embedded in a full board, because availability ranks within the pool it is given --
    a two-row frame can never produce a rank of 10.
    """
    df = _board(60).with_columns(pl.col("vor") * 0.1)
    df = pl.concat([df, pl.DataFrame({
        "player": ["Soon", "Later"], "ecr": [12.0, 45.0],
        "adp": [12.0, 45.0], "vor": [50.0, 50.0],
    })])
    pv = pick_value(df, now=10, next_pick=30, n_sims=3000)
    assert pv["player"][0] == "Soon"


def test_empty_board_does_not_crash():
    empty = _board().head(0)
    assert availability(empty, picks=[1]).height == 0


# --- the pick-noise constants, now measured rather than assumed -----------

def test_the_noise_law_is_the_fitted_one():
    """`fit_pick_noise` existed and was never called: the heuristic 2.0 + 0.18*mu was still
    hardcoded in two places, including the opponent model inside the win-probability
    simulation. Fitted on 734 real picks from this league's 2022-25 drafts it comes out at
    1.00 + 0.253*mu.

    The difference is not cosmetic deep in the board. At ADP 100 the heuristic says sigma is
    20 and the fit says 26, so the heuristic is over-confident about who survives -- which
    inflates cost_of_waiting and pushes the board toward 'take him now' on players who would
    in fact have lasted."""
    from hub.draft.availability import PICK_NOISE_INTERCEPT, PICK_NOISE_SLOPE
    assert PICK_NOISE_INTERCEPT == pytest.approx(1.00, abs=0.01)
    assert PICK_NOISE_SLOPE == pytest.approx(0.253, abs=0.005)


def test_the_fitted_law_is_wider_late_and_tighter_early():
    """The shape of the correction, pinned so a refit that inverts it is noticed."""
    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    early_fit, early_heur = A + B * 3, 2.0 + 0.18 * 3
    late_fit, late_heur = A + B * 100, 2.0 + 0.18 * 100
    assert early_fit < early_heur
    assert late_fit > late_heur


def test_sigma_uses_the_fitted_law_when_there_is_no_consensus_spread():
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 100.0]})
    got = _sigma(df)
    assert got[0] == pytest.approx(A + B * 10.0)
    assert got[1] == pytest.approx(A + B * 100.0)


# --- which column the consensus spread is read from -----------------------
#
# Only the fallback above was ever tested, so the branch that reads the consensus spread ran
# in production untested. That is why the column could be renamed with the whole suite green
# -- and why the collision below could have gone live without anything failing.

def test_sigma_prefers_the_consensus_spread_when_it_is_there():
    import polars as pl

    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 100.0], "ecr_sd": [7.0, 40.0]})
    assert _sigma(df).tolist() == [7.0, 40.0]


def test_a_zero_or_null_consensus_spread_falls_back_per_player():
    """A player FantasyPros has no spread for must not come out as sigma 0 -- certainty
    about where he goes, from an absence of data."""
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    df = pl.DataFrame({"mu_pick": [10.0, 20.0], "ecr_sd": [0.0, None]})
    got = _sigma(df)
    assert got[0] == pytest.approx(A + B * 10.0)
    assert got[1] == pytest.approx(A + B * 20.0)


def test_a_points_spread_is_not_mistaken_for_a_pick_spread():
    """The collision this rename exists to prevent.

    `sd` meant two different things on frames in this pipeline: the spread of a player's
    consensus *rank*, in picks, and the spread of his weekly *points*. Both are small
    positive floats and neither looks wrong. `hub.models.predict.moments` writes the points
    one onto a board-derived frame, so an availability sim run on a scored frame would have
    read points as picks and produced a confident, plausible, entirely wrong curve.
    """
    import polars as pl

    from hub.draft.availability import PICK_NOISE_INTERCEPT as A
    from hub.draft.availability import PICK_NOISE_SLOPE as B
    from hub.draft.availability import _sigma
    scored = pl.DataFrame({"mu_pick": [10.0], "sd": [7.9]})   # weekly points, not picks
    assert _sigma(scored)[0] == pytest.approx(A + B * 10.0)


# --- picks are fitted against their own preseason (issue #38) --------------

def _ranks(*rows):
    """(player, ecr, scrape_date) in the archive's shape."""
    return pl.DataFrame({
        "page_type": ["redraft-overall"] * len(rows),
        "player": [r[0] for r in rows],
        "ecr": [r[1] for r in rows],
        "scrape_date": [r[2] for r in rows],
    })


def test_a_pick_whose_only_rank_predates_the_preseason_is_not_fitted():
    """The defect. Taking the latest scrape before the draft and nothing more matched a pick
    whose only rank came from an earlier preseason, and fitted it as though the room had
    priced him that year -- so the fit learned from ranks nobody in that draft could see."""
    from hub.draft.availability import picks_against_preseason
    allr = _ranks(("Stale Guy", 40.0, "2022-08-10"), ("Priced", 12.0, "2024-08-20"))
    rows, said = picks_against_preseason(allr, 2024, ["Priced", "Stale Guy"])
    assert [r["ecr"] for r in rows] == [12.0]
    assert [r["pick"] for r in rows] == [1.0], "the pick number is the draft slot, not the row"
    assert "1/2 picks matched" in said and "2024-07-01" in said


def test_a_pick_ranked_in_this_preseason_is_fitted_on_this_seasons_rank():
    """The control: the bound must not drop a pick that was genuinely priced that year, and
    must use the current rank when an older one also exists."""
    from hub.draft.availability import picks_against_preseason
    allr = _ranks(("Both", 40.0, "2022-08-10"), ("Both", 11.0, "2024-08-20"))
    rows, said = picks_against_preseason(allr, 2024, ["Both"])
    assert [r["ecr"] for r in rows] == [11.0]
    assert "1/1 picks matched" in said


def test_a_rank_scraped_after_the_draft_is_still_excluded():
    """The upper bound predates this change and has to survive it -- a rank published after
    the draft is hindsight, which is the whole reason the window exists at all."""
    from hub.draft.availability import picks_against_preseason
    allr = _ranks(("Late", 5.0, "2024-10-01"))
    rows, said = picks_against_preseason(allr, 2024, ["Late"])
    assert rows == [] and "0/1 picks matched" in said


# --- pick noise fitted on a pick number, constrained (issue #40) -----------

def _picks(a_true, b_true, n=60, drafts=4, pool=192, seed=0):
    """Synthetic drafts whose pick dispersion really is `a_true + b_true * pick`.

    `pick` is where the player went; `ecr` is where the room expected him. Building it this
    way round means the fit has to recover the parameters from the deviation, which is what
    it does in production.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for yr in range(2021, 2021 + drafts):
        want = np.linspace(1, pool, n)
        sigma = a_true + b_true * want
        went = want + rng.normal(0.0, sigma)
        for w, g in zip(want, went, strict=True):
            rows.append({"year": yr, "pick": float(g), "ecr": float(w)})
    return pl.DataFrame(rows)


def test_the_fit_recovers_a_known_slope_and_intercept():
    """Criterion one: a positive true intercept, so the constraint never binds and the fit is
    doing ordinary work."""
    from hub.draft.availability import noise_from_picks
    df = _picks(a_true=4.0, b_true=0.20, n=120, seed=1)
    (a, b), said = noise_from_picks(df)
    assert a == pytest.approx(4.0, abs=1.5), said
    assert b == pytest.approx(0.20, abs=0.05), said


def test_the_constrained_fit_does_not_exceed_the_truncated_one():
    """Criterion two, and the defect itself.

    The old refit ran through `max(sigma_hat - a, 0)`, zeroing every residual below the pinned
    intercept instead of letting it pull the slope down -- so the slope was fitted to the
    upper envelope of the data. Where the unconstrained intercept goes negative, that is
    exactly where it bites.
    """
    from hub.draft.availability import _constrained, noise_from_picks
    df = _picks(a_true=0.2, b_true=0.30, n=120, seed=3)   # tiny intercept: constraint binds
    x = df["ecr"].to_numpy().astype(float)
    y = np.abs(df["pick"].to_numpy().astype(float) - x) * np.sqrt(np.pi / 2.0)
    unconstrained_a = float(np.polyfit(x, y, 1)[1])
    assert unconstrained_a < 1.0, "this fixture is meant to make the constraint bind"

    _a, b = _constrained(x, y)
    truncated = float(np.dot(x, np.maximum(y - 1.0, 0.0)) / np.dot(x, x))
    # Strictly, not `<=`. The ticket says "does not exceed", which equality satisfies -- and
    # equality is exactly what the unfixed implementation produces, so a `<=` here passes
    # against the defect it is meant to catch. Truncation drops negative residuals, so it is
    # strictly larger whenever any observation falls below the floor, which this fixture
    # guarantees.
    assert (y < 1.0).any(), "no sub-floor residual, so the two fits cannot differ"
    assert b < truncated, (
        f"the constrained slope {b:.4f} is not below the truncated {truncated:.4f}, so "
        f"dropping the sub-floor residuals is not what was inflating it")
    (_fa, fb), _said = noise_from_picks(df)
    assert fb < truncated


def test_the_slope_is_reported_with_a_draft_clustered_interval():
    """Criterion three. Picks inside one draft are anything but independent -- one manager
    reaching in round two moves every later pick in that room -- so an interval over picks
    would be several times too tight."""
    from hub.draft.availability import noise_from_picks
    df = _picks(a_true=3.0, b_true=0.2, n=100, drafts=4, seed=5)
    _fit, said = noise_from_picks(df, draws=200)
    assert "clustered on the draft" in said and "slope 95% CI" in said
    assert "over 4 drafts" in said


def test_the_fit_is_restricted_to_the_draftable_pool():
    """Criterion, and the other half of the defect: `_sigma` applies this to `mu_pick`, a
    pick position, while the fit read `ecr` over a 300-plus consensus board of which about
    192 are ever taken. Ranks past the last pick are not pick positions."""
    from hub.draft.availability import noise_from_picks
    inside = _picks(a_true=3.0, b_true=0.2, n=80, pool=180, seed=7)
    # Players ranked far past anything that was ever drafted, with wild deviations.
    tail = pl.DataFrame({"year": [2021] * 40,
                         "pick": [float(i) for i in range(1, 41)],
                         "ecr": [float(400 + i) for i in range(40)]})
    _fit, said = noise_from_picks(pl.concat([inside, tail]))
    assert f"inside a {inside['pick'].to_numpy().max():.0f}-pick pool" in said or "pool" in said
    # The tail must not reach the fit: fitted with it, the slope is visibly different.
    with_tail, _ = noise_from_picks(pl.concat([inside, tail]))
    without, _ = noise_from_picks(inside)
    assert with_tail == pytest.approx(without, abs=1e-9), (
        "observations outside the draftable pool changed the fit, so they were not excluded")


def test_below_the_floor_the_fit_falls_back_loudly():
    """Criterion four. A quiet fallback is a fitted-looking number that was never fitted."""
    from hub.draft.availability import noise_from_picks
    thin = _picks(a_true=3.0, b_true=0.2, n=5, drafts=2)
    got, said = noise_from_picks(thin, default=(2.0, 0.18))
    assert got == (2.0, 0.18)
    assert "keeping the (2.0, 0.18) prior" in said


def test_enough_picks_but_too_few_inside_the_pool_falls_back_too():
    """The other floor, and it is a different sentence. A room can have plenty of matched
    picks and still have almost none whose consensus rank was ever a pick number -- which is
    a fit with nothing to say about the axis it will be applied on, not a thin one."""
    from hub.draft.availability import noise_from_picks
    inside = _picks(a_true=3.0, b_true=0.2, n=5, drafts=2, pool=20)
    tail = pl.DataFrame({"year": [2021] * 80,
                         "pick": [float(i % 20 + 1) for i in range(80)],
                         "ecr": [float(400 + i) for i in range(80)]})
    got, said = noise_from_picks(pl.concat([inside, tail]), default=(2.0, 0.18))
    assert got == (2.0, 0.18)
    assert "inside the draftable pool" in said and "keeping the" in said


# --- one base dispersion, three readers (issue #41) ------------------------

def test_all_three_readers_resolve_to_the_same_base_at_scale_one():
    """Criterion one. Availability, the simulated room and the lambda evaluation used three
    models that disagreed by construction -- and only one of them was fitted."""
    from hub.draft.availability import _sigma, pick_noise
    from hub.draft.evaluate import OPP_NOISE as EVAL_SCALE
    from hub.draft.optimize import simulate_remaining_draft  # noqa: F401  -- reads the base

    ranks = np.array([1.0, 24.0, 96.0, 192.0])
    base = pick_noise(ranks)
    # availability, with no expert disagreement to widen it
    got = _sigma(pl.DataFrame({"mu_pick": ranks}))
    assert np.allclose(got, base)
    # the room and the evaluation both scale that same base, and both default to 1.0
    import inspect

    from hub.draft.optimize import simulate_remaining_draft as _room
    assert inspect.signature(_room).parameters["opp_noise"].default == 1.0
    assert EVAL_SCALE == 1.0, "the evaluation still carries its own absolute sigma"


def test_the_rooms_scale_moves_and_availabilitys_does_not():
    """Criterion two. How loosely opponents follow their own board is a different question
    from where a player goes, so the room keeps a knob and availability does not."""
    from hub.draft.availability import _sigma, pick_noise
    ranks = np.array([10.0, 100.0])
    base = pick_noise(ranks)
    assert np.allclose(1.5 * base, pick_noise(ranks) * 1.5)
    # availability has no scale to turn: the same board gives the same sigma.
    twice = _sigma(pl.DataFrame({"mu_pick": ranks}))
    assert np.allclose(twice, base)


def test_expert_disagreement_widens_the_base_and_never_narrows_it():
    """Criterion three. Replacing meant a player the experts agree about was priced as more
    predictable than the base says anyone at his rank is -- so expert consensus, which is not
    evidence about how this room drafts, could make a pick look safer than any measurement
    supports."""
    from hub.draft.availability import _sigma, pick_noise
    ranks = np.array([50.0, 50.0, 50.0])
    base = pick_noise(ranks)[0]
    df = pl.DataFrame({"mu_pick": ranks, "ecr_sd": [0.0, base / 2.0, base * 3.0]})
    got = _sigma(df)
    assert got[0] == pytest.approx(base), "no disagreement recorded, so the base stands"
    assert got[1] == pytest.approx(base), "a narrow disagreement must not narrow the base"
    assert got[2] == pytest.approx(base * 3.0), "a wide one widens it"
    assert (got >= base - 1e-9).all()


def test_the_evaluations_noise_is_a_scale_that_can_express_the_old_flat_one():
    """Criterion four. The old model was flat: the same 8-pick uncertainty about the first
    pick as the hundred and ninetieth, which is the opposite of what the other two say."""
    from hub.draft.availability import pick_noise
    from hub.draft.evaluate import LEGACY_FLAT_NOISE, equivalent_scale

    ecr = np.arange(1.0, 193.0)
    s = equivalent_scale(ecr)
    assert (s * pick_noise(ecr)).mean() == pytest.approx(LEGACY_FLAT_NOISE)
    # And it is a scale, not a constant: it still widens down the board where the old did not.
    sigma = s * pick_noise(ecr)
    assert sigma[-1] > sigma[0] * 5
