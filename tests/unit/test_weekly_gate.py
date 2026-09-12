"""Gate B: does a lineup set off the Weekly projection beat one set off consensus rank?

All offline. The first live run returned +11.1 points per team-week at P(better) 100% and was
VOID -- 6.2% of roster-weeks were a join failure against a pre-registered 2% floor. These tests
pin the machinery that said so.
"""
import numpy as np
import polars as pl
import pytest

from hub.league import STARTERS
from hub.season import weekly_gate as G


def _pos(n_qb=2, n_rb=4, n_wr=5, n_te=2):
    return ["QB"] * n_qb + ["RB"] * n_rb + ["WR"] * n_wr + ["TE"] * n_te


# --- the lineup rule --------------------------------------------------------

def test_the_lineup_is_chosen_again_every_week():
    """The entire subject. A static projection sets one lineup all season; a weekly one does
    not, and `lineup_gate` names that as the gap it could not measure."""
    pos = ["RB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "WR", "QB", "TE", "TE"]
    n = len(pos)
    realised = np.zeros((n, 2))
    realised[0, 0] = 50.0          # player 0 explodes in week 1
    realised[1, 1] = 50.0          # player 1 in week 2
    # Distinct scores, so no tie is broken by index: each week the man who explodes is top
    # and the other of the pair is bottom.
    mid = np.linspace(10.0, 20.0, n)
    score = np.column_stack([mid, mid])
    score[0, 0], score[1, 0] = 99.0, 0.1
    score[1, 1], score[0, 1] = 99.0, 0.1
    roster = list(range(n))
    got = G.season_points(realised, pos, score, roster, [], [1, 2])
    assert [got[1], got[2]] == [50.0, 50.0]
    static = np.column_stack([score[:, 0], score[:, 0]])
    assert G.season_points(realised, pos, static, roster, [], [1, 2])[2] == 0.0, \
        "the static lineup keeps starting week 1's man and benches week 2's"


# --- coverage, and the difference between an absence and a defect ----------

def _inputs(**over):
    """A `GateInputs` with everything aligned, so a test names only what it cares about."""
    import numpy as np
    pos = _pos()
    n = len(pos)
    base = {
        "rosters": {2024: [list(range(n))]}, "pos": {2024: pos},
        "realised": {2024: np.zeros((n, 18))}, "consensus": {2024: np.zeros((n, 18))},
        "weekly": {2024: np.zeros((n, 18))}, "pool": {2024: [[]]},
        "addable": {2024: np.ones((n, 18), dtype=bool)}, "se": {2024: np.zeros((n, 18))},
        "covered": {(2024, 5)},
        # Every cell the model's own number unless a test says otherwise, so a fixture that
        # cares about the fallback has to build one rather than inherit it.
        "projected": {2024: np.ones((n, 18), dtype=bool)}}
    base.update(over)
    return G.GateInputs(**base)


def _cov_fixture():
    """A join failure and a correct omission, in the same week."""
    import numpy as np
    n = len(_pos())
    cons = np.zeros((n, 18))
    realised = np.zeros((n, 18))
    cons[0, 4] = G.UNRANKED          # unranked and scored -> a join failure
    realised[0, 4] = 12.0
    cons[1, 4] = G.UNRANKED          # unranked and scored nothing -> correctly benched
    return _inputs(consensus={2024: cons}, realised={2024: realised})


def test_coverage_separates_a_correct_omission_from_a_join_failure():
    """A player who is out scores zero and being unranked is the incumbent's *answer*. A
    player who scored and was unranked is a name that did not match, and only that one biases
    the comparison."""
    c = G.coverage(_cov_fixture(), weeks=[5])
    assert c["unranked"] == pytest.approx(2 / 13)
    assert c["join_failure"] == pytest.approx(1 / 13)


def test_coverage_ignores_weeks_the_incumbent_does_not_cover():
    g = _cov_fixture()._replace(covered=set())
    assert G.coverage(g, weeks=[5])["cells"] == 0


def test_the_inputs_are_one_thing_rather_than_nine():
    """They were a nine-value positional tuple threaded through twelve parameters, and the
    ordering was knowledge duplicated across the return, the unpack and two call sites --
    checked nowhere. Swapping `consensus` and `weekly` inverts the entire result.

    Ten since #207, and the count is asserted rather than dropped for exactly the reason the
    nine were: a positional constructor is still what `assemble_universe` calls, so a field
    appended in the wrong place is a silent re-labelling of two columns.
    """
    g = _inputs()
    assert len(G.GateInputs._fields) == 10
    assert G.GateInputs._fields[3:5] == ("consensus", "weekly")
    assert G.GateInputs._fields[-1] == "projected"
    assert g._replace(covered={(2024, 9)}).covered == {(2024, 9)}, "and it is replaceable"


# --- the verdict, every branch ---------------------------------------------
#
# The branches are `experiment.gate` and the sentences are `ACTIONS`; what is this gate's own
# is `void_condition`, the one precondition only it has. Since #135 the two meet inside
# `experiment.run_gate`, which `main` calls, so the wrapper `verdict` used to be is spelled
# here as what it was.

def _verdict(summary, seasons, cover):
    from hub.models.experiment import gate
    return gate(summary, seasons, G.ACTIONS, void=G.void_condition(cover))


def _gate_run(paired, **kw):
    """This gate's call, as `main` spells it, with the width history pointed nowhere."""
    from hub.models.experiment import SEASON_CLUSTER, run_gate
    return run_gate(paired, cluster=SEASON_CLUSTER, actions=G.ACTIONS, name="weekly",
                    arm_a="weekly", arm_b="consensus", unit=G.UNIT, places=G.PLACES,
                    show_n=False, bootstrap=200, ceiling=G.declared_ceiling(paired),
                    record_width=False, **kw)


def _summary(mean, lo, hi, clusters=60):
    return {"n": 800.0, "clusters": float(clusters), "mean": mean, "lo": lo, "hi": hi,
            "p_better": 1.0 if lo > 0 else 0.0}


def _seasons(gains):
    return pl.DataFrame({"season": list(range(2022, 2022 + len(gains))),
                         "gain": gains, "n": [200] * len(gains)})


def test_a_join_failure_voids_the_run_however_large_the_result():
    """The branch that fired on the first live run: +11.1 points per team-week at P 100%,
    which is the repo's own rule that a result too large to believe is a bug."""
    status, note = _verdict(_summary(11.1, 10.1, 12.3), _seasons([10.0, 10.2, 13.2]),
                             {"cells": 680.0, "unranked": 0.157, "join_failure": 0.062})
    assert status == "VOID"
    assert "6.2%" in note and "2%" in note


def test_a_clean_join_lets_the_result_through():
    status, _ = _verdict(_summary(0.9, 0.3, 1.5), _seasons([0.8, 1.0, 0.9]),
                          {"cells": 680.0, "unranked": 0.1, "join_failure": 0.005})
    assert status == "ADOPT"


def test_the_floor_itself_is_not_a_void():
    """`VOID_FLOOR` is the share *above* which a run is void, as `verdict` spelled it before
    #135 moved the condition into `void_condition`. A run exactly at the floor still reports.
    Held because the mutant `<` for `<=` survived every other test in this file."""
    at = {"cells": 100.0, "unranked": 0.1, "join_failure": G.VOID_FLOOR}
    assert G.void_condition(at) is None
    above = dict(at, join_failure=G.VOID_FLOOR + 1e-9)
    assert (G.void_condition(above) or "").startswith("VOID")
    assert G.void_condition(None) is None
    assert G.void_condition({"cells": 0, "unranked": float("nan"),
                             "join_failure": float("nan")}) is None


def test_adopt_needs_every_season_as_well_as_the_interval():
    status, note = _verdict(_summary(0.9, 0.3, 1.5), _seasons([-0.2, 1.4, 1.5]), None)
    assert status == "SHOW" and "2/3" in note


def test_losing_in_every_season_removes_the_module():
    status, note = _verdict(_summary(-1.2, -1.8, -0.6), _seasons([-1.0, -1.3, -1.3]), None)
    assert status == "REMOVE" and "Delete" in note


def test_an_interval_containing_zero_is_shown_never_ranked_on():
    """The expected branch, and it carries an action rather than a disappointment."""
    status, note = _verdict(_summary(0.2, -0.4, 0.8), _seasons([0.1, 0.4, 0.1]), None)
    assert status == "SHOW"
    assert "NEVER RANK ON" in note and "absence of evidence" in note


def test_nothing_measured_does_not_adopt():
    assert _verdict(_summary(0.0, 0.0, 0.0, clusters=0), _seasons([]), None)[0] == "SHOW"


# --- pairing and the cluster bootstrap -------------------------------------

def _paired(seasons=4, rosters=6, weeks=10, gain=1.0, seed=0):
    """Roster-weeks with an effect at *both* levels a cluster could be drawn at.

    The season carries the larger one (sd 5.0) and the roster a smaller one (sd 1.5) nested
    inside it, which is the shape the real gate has and the shape that lets the three
    candidate units be told apart: row, roster, season each give a different interval here.
    A fixture flat at either level would let two of the three agree and prove nothing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2022, 2022 + seasons):
        year = rng.normal(0, 5.0)        # a season-level effect, shared by all its rosters
        for k in range(rosters):
            offset = rng.normal(0, 1.5)  # a roster-level effect, shared by all its weeks
            for w in range(1, weeks + 1):
                d = gain + year + offset + rng.normal(0, 1.0)
                rows.append({"season": season, "roster": k, "week": w,
                             "consensus": 100.0, "weekly": 100.0 + d, "diff": d})
    return pl.DataFrame(rows)


def test_the_bootstrap_resamples_seasons_not_rosters_or_rows():
    """Issue #45 moved this gate's unit from the roster to the season, and the argument that
    moved it is the one that had already moved it from the row to the roster.

    A roster's ten weeks share its players, its bye and its draft. But its season's six
    rosters share a board, a player pool, a schedule and one realisation of the year -- so
    they are six readings of a season, not six observations, and resampling them reports an
    interval too narrow for the same reason resampling rows did. Protocol item 3, one level
    further up.

    All three units are asserted here, in order, because the failure this guards against is
    stopping one level short -- which is exactly what the previous version of this gate did.
    """
    from hub.models.experiment import summarise
    paired = _paired()
    season = summarise(paired, cluster=G.CLUSTER, bootstrap=2000, seed=1)
    roster = summarise(paired, cluster=("season", "roster"), bootstrap=2000, seed=1)
    row = summarise(paired, bootstrap=2000, seed=1)

    widths = [row["hi"] - row["lo"], roster["hi"] - roster["lo"],
              season["hi"] - season["lo"]]
    assert widths[0] < widths[1] < widths[2], \
        f"each coarser unit must widen the interval: {widths}"
    assert season["clusters"] == 4 and roster["clusters"] == 24 and row["clusters"] == 240
    assert season["n"] == roster["n"] == row["n"] == 240
    # The mean is untouched by any of it: these seasons are balanced, so what moves is the
    # claim about precision and never the estimate.
    assert season["mean"] == pytest.approx(row["mean"])


def test_an_empty_frame_reports_rather_than_crashing():
    from hub.models.experiment import summarise
    s = summarise(pl.DataFrame(), cluster=G.CLUSTER)
    assert s["clusters"] == 0 and np.isnan(s["mean"])


def test_compare_emits_one_row_per_roster_week_and_skips_uncovered_weeks():
    import numpy as np
    n = len(_pos())
    scores = {2024: np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))}
    g = _inputs(realised={2024: np.ones((n, 18))}, consensus=scores, weekly=scores,
                covered={(2024, 3), (2024, 4)})
    out = G.compare(g, weeks=[3, 4, 5])
    assert out.height == 2 and sorted(out["week"].to_list()) == [3, 4]
    assert (out["diff"] == 0.0).all(), "identical arms differ by nothing"


def test_compare_takes_one_argument_and_a_named_set_of_options():
    """It took twelve parameters to run thirty-six lines -- the interface was the larger half.

    The set is pinned rather than counted, so adding one is a decision someone makes on
    purpose. `ceiling` was added for #43 and is the shape this test should allow: opt-in,
    defaulting off, and adding a column rather than changing what the gate measures. A
    parameter that moved the effect would deserve to fail here.

    **`restrict` is one that moved the effect, and it failed here first.** #206 changes what
    the gate measures on purpose -- both arms score only the roster-weeks both can price --
    so it is the one parameter in this set that defaults *on*, and unlike `ceiling` it is not
    opt-in. Both defaults are asserted below, in opposite directions, because both of them
    are the decision and not the convenience.
    """
    import inspect
    sig = inspect.signature(G.compare)
    positional = [p for p in sig.parameters.values()
                  if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD]
    assert len(positional) == 1 and positional[0].name == "g"
    assert set(sig.parameters) - {"g"} == {"weeks", "churn", "z", "mask_pool", "ceiling",
                                           "restrict"}
    assert sig.parameters["ceiling"].default is False, (
        "the ceiling must be opt-in; on by default it would change every existing run")
    assert sig.parameters["restrict"].default is True, (
        "#206 is the gate, not a sensitivity: a run that has to ask for it is a run that "
        "reports the fallback by default")


# --- the waiver rule, and the artifact it was written with ------------------

def _uni():
    """A universe: a full roster plus a deep free-agent pool."""
    pos = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE", "TE",
           "QB", "WR", "RB", "TE"]          # 10 rostered, then the pool
    roster = list(range(10))
    pool = list(range(10, 14))
    return pos, roster, pool


def test_a_backup_quarterback_is_not_added_to_replace_a_starting_receiver():
    """The artifact the first churn run was built with. Absolute weekly points are much larger
    at quarterback, so 'add the highest-scoring free agent' picked up a backup QB every week --
    Russell Wilson at a projected 24.4 who scored 5.1. You start one quarterback.

    Consensus rank does not make that mistake, because a ranking already prices scarcity, so
    the naive rule handed the incumbent a free win owing nothing to either arm's forecasting.
    """
    pos, roster, pool = _uni()
    # The rostered QB is better (26), so the free agent at 24 can never start -- but 24 still
    # towers over the worst bench player at 7, which is all the naive rule looked at.
    score = np.array([26.0, 12, 11, 10, 14, 13, 12, 9, 8, 7,     # rostered
                      24.0, 2.0, 2.0, 2.0])                       # pool: a 24-point QB
    naive_add = max(pool, key=lambda i: score[i])
    assert pos[naive_add] == "QB", "the naive rule would take him"
    assert G.waiver_swap(roster, pool, pos, score, starters=8) is None, \
        "and the lineup rule takes nobody, because none of them would start"


def test_a_free_agent_who_would_start_is_added():
    pos, roster, pool = _uni()
    score = np.array([22.0, 12, 11, 10, 14, 13, 12, 9, 8, 7,
                      1.0, 30.0, 1.0, 1.0])                       # a 30-point WR
    got = G.waiver_swap(roster, pool, pos, score, starters=8)
    assert got is not None
    add, drop = got
    assert pos[add] == "WR" and score[add] == 30.0
    assert drop in roster and score[drop] < score[add]


def test_nothing_is_added_when_nothing_improves_the_lineup():
    pos, roster, pool = _uni()
    score = np.array([22.0, 12, 11, 10, 14, 13, 12, 9, 8, 7, 1.0, 1.0, 1.0, 1.0])
    assert G.waiver_swap(roster, pool, pos, score, starters=8) is None


def test_the_roster_stays_legal():
    """Dropping the only quarterback to add a fourth receiver wins a week and forfeits the
    rest, so a player needed to fill a required slot is never droppable."""
    pos = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "RB", "WR"]
    roster, pool = list(range(8)), [8]
    score = np.array([1.0, 9, 8, 7, 6, 5, 4, 3, 99.0])
    got = G.waiver_swap(roster, pool, pos, score, starters=8)
    assert got is None or pos[got[1]] != "QB", "the only QB is not droppable"


def test_an_empty_pool_or_a_bare_roster_does_nothing():
    pos, roster, pool = _uni()
    score = np.ones(len(pos))
    assert G.waiver_swap(roster, [], pos, score, starters=8) is None
    assert G.waiver_swap(roster[:8], pool, pos, score, starters=8) is None


# --- churn off must reproduce the frozen gate ------------------------------

def test_churn_off_leaves_the_roster_alone():
    pos, roster, pool = _uni()
    realised = np.tile(np.arange(len(pos), dtype=float).reshape(-1, 1), (1, 18))
    score = np.tile(np.arange(len(pos), dtype=float).reshape(-1, 1), (1, 18))
    frozen = G.season_points(realised, pos, score, roster, pool, [1, 2, 3], churn=False)
    churned = G.season_points(realised, pos, score, roster, pool, [1, 2, 3], churn=True)
    assert len(set(frozen.values())) == 1, "a static roster scores the same every week"
    assert churned[3] > frozen[3], "and the churning one has improved by week three"


def test_the_pool_is_masked_to_players_both_arms_can_score():
    """The decision that keeps this able to fail. Consensus ranks 35.8% of a 935-player pool,
    so an unmasked pool would hand the arm under test six hundred players the incumbent cannot
    score at all."""
    pos, roster, pool = _uni()
    realised = np.zeros((len(pos), 18))
    realised[11, :] = 40.0
    score = np.tile(np.array([22.0, 12, 11, 10, 14, 13, 12, 9, 8, 7,
                              1.0, 30.0, 1.0, 1.0]).reshape(-1, 1), (1, 18))
    addable = np.ones((len(pos), 18), dtype=bool)
    addable[11, :] = False                       # the 30-point WR is unscorable by one arm
    blocked = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=True,
                              addable=addable)
    allowed = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=True)
    assert blocked[2] == 0.0 and allowed[2] == 40.0


def test_a_dropped_player_returns_to_the_pool():
    pos, roster, pool = _uni()
    score = np.zeros((len(pos), 18))
    score[:, :] = np.array([22.0, 12, 11, 10, 14, 13, 12, 9, 8, 7,
                            1.0, 30.0, 1.0, 1.0]).reshape(-1, 1)
    realised = np.zeros((len(pos), 18))
    # week 2 flips: the dropped man is now the best free agent again
    score[11, 1] = 1.0
    score[9, 1] = 0.0
    out = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=True)
    assert out is not None, "the swap and its reversal both run without error"


# --- the waiver decision is separable from the lineup decision --------------

def test_the_waiver_decision_can_read_a_different_score_from_the_lineup():
    """A waiver pick is the maximum over hundreds of candidates and so is biased upward; a
    lineup is a choice among players you already hold and has no such selection. So the add
    may be ranked on a lower confidence bound while the lineup stays on the mean."""
    pos, roster, pool = _uni()
    n = len(pos)
    score = np.tile(np.array([22.0, 12, 11, 10, 14, 13, 12, 9, 8, 7,
                              1.0, 30.0, 1.0, 1.0]).reshape(-1, 1), (1, 18))
    realised = np.zeros((n, 18))
    realised[11, :] = 40.0
    optimistic = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=True)
    # the same lineup scores, but the 30-point free agent is penalised out of contention
    cautious_add = score.copy()
    cautious_add[11, :] = 0.0
    careful = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=True,
                              add_score=cautious_add)
    assert optimistic[2] == 40.0, "on the mean he is added and pays off"
    assert careful[2] == 0.0, "penalised, he is never added"


def test_add_score_defaults_to_the_lineup_score():
    """Every run before the lower confidence bound existed had one score for both, and the
    default must reproduce it exactly."""
    pos, roster, pool = _uni()
    n = len(pos)
    score = np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))
    realised = np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))
    a = G.season_points(realised, pos, score, roster, pool, [1, 2, 3], churn=True)
    b = G.season_points(realised, pos, score, roster, pool, [1, 2, 3], churn=True,
                        add_score=score)
    assert a == b


def test_a_frozen_roster_cannot_see_the_waiver_score_at_all():
    """The pre-registered tripwire: the lower confidence bound is unreachable without churn,
    so a frozen gate that moves means something is wired wrong. It did not move."""
    pos, roster, pool = _uni()
    n = len(pos)
    score = np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))
    realised = np.ones((n, 18))
    plain = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=False)
    with_lcb = G.season_points(realised, pos, score, roster, pool, [1, 2], churn=False,
                               add_score=np.zeros_like(score))
    assert plain == with_lcb


# --- the weekly gate's ceiling is full foresight (issue #43) ---------------

def test_the_weekly_gates_ceiling_is_a_perfect_projection():
    """Full foresight is right *here*, unlike the lineup gate.

    What separates these two arms is the projection itself -- consensus against the weekly
    model -- so the largest effect any weekly projection could show is what a perfect one
    would. In the lineup gate both arms already share a projection and only spread differs,
    which is why its ceiling is a variance oracle and the two are not interchangeable.
    """
    import numpy as np

    # Real numbers, and deliberately so: the default fixture is all zeros, on which every
    # comparison below holds no matter what the ceiling reads. A test that passes when the
    # foresight arm is handed consensus instead of the realised frame is asserting the
    # outcome rather than that the arm produced it -- which is the shape this repo keeps
    # finding, so it is worth not shipping another one.
    pos = _pos()
    n = len(pos)
    rng = np.random.default_rng(0)
    realised = rng.uniform(0, 30, (n, 18))
    # Consensus and the weekly model are both wrong, in different directions.
    g = _inputs(realised={2024: realised},
                consensus={2024: realised[::-1].copy()},
                weekly={2024: rng.uniform(0, 30, (n, 18))},
                rosters={2024: [list(range(n))]})

    got = G.compare(g, weeks=[5], ceiling=True)
    assert {"foresight", "ceiling_diff"} <= set(got.columns)
    # A perfect projection cannot be beaten by either arm on the frame it is scored on, and
    # here it strictly beats both -- so the assertion has something to fail on.
    assert (got["foresight"] > got["consensus"]).all()
    assert (got["foresight"] > got["weekly"]).all()
    assert float(got["ceiling_diff"].to_numpy().mean()) > float(got["diff"].to_numpy().mean())


def _three_separate_numbers():
    """A week where the incumbent is wrong, the arm under test is better, and neither is
    perfect -- so the effect and the ceiling are two different numbers.

    Player `i` scores `i`. Consensus ranks on `-i`, which is the worst available ordering;
    the weekly model has the top of the board right and the bottom reversed, so it is better
    than consensus and short of perfect. The three arms come out at 41.0, 55.0 and 60.0 for
    an effect of +14.000 against a ceiling of +19.000.

    The separation is the fixture's point. `_inputs()` is all zeros, on which every arm
    scores nothing and every comparison between a ceiling and an effect holds no matter what
    the foresight arm reads -- a passing assertion over three copies of one number.
    """
    n = len(_pos())
    points = np.arange(n, dtype=float)
    half = points.copy()
    half[:7] = half[:7][::-1]
    return _inputs(realised={2024: np.tile(points.reshape(-1, 1), (1, 18))},
                   consensus={2024: np.tile((-points).reshape(-1, 1), (1, 18))},
                   weekly={2024: np.tile(half.reshape(-1, 1), (1, 18))})


def test_the_ceiling_is_opt_in_and_does_not_change_the_frozen_gate():
    """On by default it would widen every existing run's frame and re-price the verdicts that
    rest on it, as a side effect of adding a diagnostic.

    Driven on a frame where the arms actually differ. Asserting this over the all-zero
    fixture would hold with the foresight arm scoring the roster on consensus, on the weekly
    model, or on nothing at all.
    """
    g = _three_separate_numbers()
    plain = G.compare(g, weeks=[5])
    withc = G.compare(g, weeks=[5], ceiling=True)
    assert "foresight" not in plain.columns
    assert plain["consensus"].to_list() == withc["consensus"].to_list() == [41.0]
    assert plain["weekly"].to_list() == withc["weekly"].to_list() == [55.0]
    assert plain["diff"].to_list() == withc["diff"].to_list() == [14.0], (
        "asking for the ceiling changed the effect the gate reports")


# --- and an operator can ask for it (issue #134) --------------------------
#
# #43 built the arm and left it unreachable: `compare` took the option and nothing passed it
# one, so `docs/gate-power.md` stage 2 -- which compares each gate's MDE against its own
# ceiling -- could not be run for this gate at all.

def test_the_ceiling_and_the_effect_this_gate_reports_are_different_numbers():
    """The number stage 2 needs, and it has to be able to differ from the one it bounds."""
    got = G.compare(_three_separate_numbers(), weeks=[5], ceiling=True)
    assert got["foresight"].to_list() == [60.0]
    assert got["diff"].to_list() == [14.0]
    assert got["ceiling_diff"].to_list() == [19.0]
    assert got["ceiling_diff"][0] > got["diff"][0]


def test_the_ceiling_line_names_its_arm_its_unit_and_that_it_travels_nowhere():
    """Stage 2 reads three gates' ceilings and holds each against its own MDE. Three numbers
    in three units under one word is the confusion the line is written to prevent, so it
    carries the arm and the unit rather than a bare figure. Full foresight is the right arm
    *here* and the wrong one for the lineup gate, whose arms already share a projection.
    """
    paired = G.compare(_three_separate_numbers(), weeks=[5], ceiling=True)
    said = [ln for ln in _gate_run(paired).lines if "ceiling" in ln.lower()]
    assert len(said) == 1, "a ceiling that bounds the effect says one thing and no more"
    assert "+19.000" in said[0], "this gate quotes three places; +19.00 would be another's"
    assert G.UNIT in said[0]
    assert G.CEILING_ARM in said[0]
    assert "not comparable" in said[0]


def test_a_ceiling_that_does_not_bound_the_effect_says_so_loudly():
    """A bound that does not bound reads exactly like a tight one, and a tight one is what
    would license a verdict nothing supports. The draft gate has said this since #42; until
    #134 neither season gate could say it at all, and since #135 all three say it through
    one function, at this gate's three places."""
    paired = pl.DataFrame({"season": [2024, 2024], "roster": [0, 1], "diff": [0.4, 0.6],
                           "ceiling_diff": [0.1, 0.1]})
    said = [ln for ln in _gate_run(paired).lines if "CEILING BELOW THE EFFECT" in ln]
    assert len(said) == 1
    assert "+0.100" in said[0] and "+0.500" in said[0]


def test_a_run_that_asked_for_no_ceiling_prints_no_ceiling_line():
    """Not a blank and not a `nan` set against a unit -- the shape `paired_report` already
    uses for a field nothing computed, because `nan` beside a unit reads as a measurement.
    A VOID run and an empty frame take the same branch, neither carrying the column."""
    plain = G.compare(_three_separate_numbers(), weeks=[5])
    assert G.declared_ceiling(plain) is None
    assert not [ln for ln in _gate_run(plain).lines if "ceiling" in ln.lower()]
    empty = G.compare(_inputs(), weeks=[9])
    assert G.declared_ceiling(empty) is None
    assert not [ln for ln in _gate_run(empty).lines if "ceiling" in ln.lower()]


def test_the_ceiling_reaches_the_rule_this_gate_never_handed_it_to():
    """The one departure #135 makes on this gate, recorded in `docs/weekly-blend-gate.md` as
    the thing that could not happen: *"nothing here hands `summarise` a ceiling, so the
    NOT-RUNNABLE branch does not fire and cannot."* It now can. The lineup gate was wired
    this way under #134; one run means one wiring."""
    paired = G.compare(_three_separate_numbers(), weeks=[5], ceiling=True)
    run = _gate_run(paired)
    from hub.models.experiment import Field, reading
    assert reading(run.summary, "ceiling") is Field.VALUE
    assert run.summary["ceiling"] == pytest.approx(19.0)


# --- the fallback, and the three treatments a run reports under --------------
#
# #207. The arm under test scores on a mixture and the gate never said so: 54.2% model
# projection, 16.7% rank-interpolated, 29.1% unscoreable, measured 2026-09-07. The three
# treatments of the middle group are 1.5 points apart, further than any of them is from zero,
# and a run printed one of them. Which one is *primary* is #206 and was decided there; these
# hold that a run reports all three, and that the numbers it reports are ones it computed.

def _flat(lines):
    """The block as one line, so an assertion is about what it says and not where it folds.

    These blocks are wrapped to a terminal, so a sentence's line breaks move whenever a number
    in front of it changes width. Asserting on the folded text would make the width of `-1.004`
    load-bearing; asserting on the words holds the claim and lets the wrap move.
    """
    return " ".join(" ".join(lines).split())


def _mixture_fixture():
    """A column of exactly 1,000 cells split 542 / 167 / 291 -- the published shares.

    A thousand players over one covered week rather than a plausible roster, because the
    quantity under test is a ratio and the honest way to pin a ratio is to build one whose
    numerator and denominator are both counted by hand. 54.2 / 16.7 / 29.1 to one decimal is
    what the gate prints and what `docs/weekly-blend-gate.md` carries.
    """
    n = 1000
    projected = np.zeros((n, 18), dtype=bool)
    projected[:542, 4] = True
    weekly = np.full((n, 18), G.UNRANKED)
    weekly[:542, 4] = 9.0                 # the model's own number
    weekly[542:709, 4] = 5.0              # interpolated: no projection, but scoreable
    # 709:1000 keep UNRANKED -- no projection and nothing to interpolate from.
    return _inputs(rosters={2024: [list(range(n))]}, pos={2024: ["WR"] * n},
                   realised={2024: np.zeros((n, 18))}, consensus={2024: np.zeros((n, 18))},
                   weekly={2024: weekly}, projected={2024: projected},
                   addable={2024: np.ones((n, 18), dtype=bool)},
                   se={2024: np.zeros((n, 18))}, pool={2024: [[]]})


def test_the_mixture_shares_are_the_ones_the_gate_prints():
    """The three shares, counted rather than carried. They were learned by probing the arm and
    written into a table by hand; the gate reports them with the result now, which is what
    stops the published figures and the shipped code drifting apart."""
    mix = G.mixture(_mixture_fixture(), weeks=[5])
    assert mix["cells"] == 1000.0
    assert mix["projection"] == pytest.approx(0.542)
    assert mix["fallback"] == pytest.approx(0.167)
    assert mix["unscoreable"] == pytest.approx(0.291)
    said = _flat(G.mixture_report(mix))
    assert "1000 roster-week cells" in said
    assert "54.2% model projection" in said
    assert "16.7% rank-interpolated" in said
    assert "29.1% unscoreable" in said


def test_the_three_shares_partition_the_column_and_nothing_else():
    """They sum to one because every cell is exactly one of the three, and a cell the gate
    does not read is in none of them -- the same rule `coverage` follows, so the two blocks
    printed side by side describe one universe rather than two."""
    mix = G.mixture(_mixture_fixture(), weeks=[5])
    assert mix["projection"] + mix["fallback"] + mix["unscoreable"] == pytest.approx(1.0)
    off = G.mixture(_mixture_fixture()._replace(covered=set()), weeks=[5])
    assert off["cells"] == 0.0
    assert G.mixture_report(off) == [], "no cells is not a share of zero, it is no share"


def _fallback_column():
    """Ten players, six of them receivers, and five cells the model could not price.

    The shape is chosen so the *ordering among the unprojected* decides the lineup: one QB,
    two RBs and one TE fill their slots however they are scored, six receivers compete for
    three WR slots and the flex, and only one of the six carries a projection. So each
    treatment starts a different set, by construction rather than by luck -- which is the
    whole of what #206 says about this gate.
    """
    pos = ["QB", "RB", "RB", "TE"] + ["WR"] * 6
    n = len(pos)
    projected = np.zeros((n, 18), dtype=bool)
    projected[:5, 4] = True               # the four forced slots, and receiver 4

    weekly = np.full((n, 18), G.UNRANKED)
    weekly[:4, 4] = [10.0, 9.0, 8.0, 7.0]
    weekly[4, 4] = 100.0                  # the one projected receiver, top under every arm
    weekly[5:, 4] = [50.0, 40.0, 1.0, 2.0, 3.0]     # interpolated: 5 and 6 start, 9 flexes

    cons = np.zeros((n, 18))
    cons[:4, 4] = [-1.0, -2.0, -3.0, -4.0]
    cons[4:, 4] = [-70.0, -60.0, -50.0, -40.0, -30.0, -20.0]   # 9, 8, 7 are the ranked ones

    realised = np.zeros((n, 18))
    realised[4:, 4] = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    return _inputs(rosters={2024: [list(range(n))]}, pos={2024: pos},
                   realised={2024: realised}, consensus={2024: cons},
                   weekly={2024: weekly}, projected={2024: projected},
                   addable={2024: np.zeros((n, 18), dtype=bool)},
                   se={2024: np.zeros((n, 18))}, pool={2024: [[]]})


def test_the_unscoreable_treatment_benches_exactly_the_cells_the_model_could_not_price():
    """Strip the interpolation and an unprojected player becomes one this arm cannot start.
    Exactly those cells and no others: a treatment that also moved a projected cell would be
    measuring something besides the fallback."""
    g = _fallback_column()
    got = G.under_treatment(g, "unscoreable").weekly[2024]
    priced, col = g.projected[2024], g.weekly[2024]
    assert (got[~priced] == G.UNRANKED).all()
    assert (got[priced] == col[priced]).all()
    assert not np.array_equal(got, col), "the fixture has a fallback cell to strip"


def test_the_mixed_scale_treatment_is_the_column_44_removed():
    """The superseded one, reproduced rather than described: negated ECR in the cells with no
    projection and fantasy points everywhere else, which is the `np.where(np.isnan(mu), cons,
    mu)` #44 deleted. It is scored only because the published +0.215 was measured under it."""
    g = _fallback_column()
    got = G.under_treatment(g, "mixed scale").weekly[2024]
    priced, col, cons = g.projected[2024], g.weekly[2024], g.consensus[2024]
    assert (got[~priced] == cons[~priced]).all()
    assert (got[priced] == col[priced]).all()


def test_the_column_treatment_is_the_one_the_assembly_already_built():
    """It hands back the same object rather than a copy that happens to agree, so no run can
    score that row against a column the verdict was not read off."""
    g = _fallback_column()
    assert G.under_treatment(g, G.COLUMN_TREATMENT) is g
    with pytest.raises(ValueError, match="no such treatment"):
        G.under_treatment(g, "whatever-sounds-better")


def test_the_waiver_pool_does_not_move_with_the_fallback():
    """`addable` is the players *both* arms can score, which is already `projected` and ranked,
    so it is identical under all three. Moving it with the treatment would confound the
    fallback with the churn rule and the spread would stop being a spread across fallbacks."""
    g = _fallback_column()
    for t in G.TREATMENTS:
        other = G.under_treatment(g, t.name)
        assert np.array_equal(other.addable[2024], g.addable[2024])
        assert other.realised[2024] is g.realised[2024]
        assert other.consensus[2024] is g.consensus[2024]


def test_the_same_rows_score_to_three_different_numbers():
    """The whole of #207 in one assertion, on a fixture whose arithmetic is on paper.

    Consensus starts receivers 9, 8, 7 and 6 for 30 points. The arm under test starts 4, 5, 6
    and 9 under interpolation (19); 4, 5, 6 and 7 under *unscoreable*, where the five it
    cannot price tie at `UNRANKED` and the lineup takes them in order (7); and 4, 9, 8 and 7
    under the mixed scale, where ranks decide among the unprojected (28). One set of rows,
    three treatments, three effects.

    **`restrict=False` is the pre-#206 universe, and this test is why it stays reachable.**
    The spread it holds is the published finding; under the gate's own scored rows the same
    three treatments reach no scored cell and the three effects collapse to one, which is
    asserted directly below rather than by deleting this.
    """
    got = {e.treatment.name: e.summary["mean"]
           for e in G.treatment_effects(_fallback_column(), weeks=[5], restrict=False)}
    assert got == {"rank-interpolated": pytest.approx(19.0 - 30.0),
                   "unscoreable": pytest.approx(7.0 - 30.0),
                   "mixed scale": pytest.approx(28.0 - 30.0)}


def test_the_carried_row_is_the_frame_the_verdict_was_read_off():
    """Handed back rather than re-scored, so a run plays three arms and not four -- and so the
    assembled column's line in the table cannot disagree with the effect printed above it."""
    g = _fallback_column()
    paired = G.compare(g, weeks=[5], restrict=False)
    effects = G.treatment_effects(g, weeks=[5], primary=paired, restrict=False)
    mean = float(np.asarray(paired["diff"].to_numpy()).mean())
    assert effects[0].treatment.name == G.COLUMN_TREATMENT
    assert effects[0].summary["mean"] == pytest.approx(mean)
    assert effects[0].seasons["gain"].to_list() == [pytest.approx(mean)]
    # And it is handed to that row *only*. A frame reused across the table would print three
    # copies of one effect under three names, which is the single-treatment report wearing
    # the shape of the fix for it -- and, since #206, would also fake the check that the
    # treatments agree by handing all three the same answer.
    assert [e.summary["mean"] for e in effects[1:]] == [
        pytest.approx(7.0 - 30.0), pytest.approx(28.0 - 30.0)]


def _effect(name, role, mean, lo, hi):
    return G.TreatmentEffect(
        G.Treatment(name, role, f"why {name}"),
        {"n": 2000.0, "clusters": 4.0, "mean": mean, "lo": lo, "hi": hi, "p_better": 0.0,
         "se": 0.191, "mde": 0.768},
        _seasons([mean, mean, mean, mean]))


def _published():
    """The three effects as measured on 2026-09-07, on the mixture above."""
    return [_effect("rank-interpolated", "primary", -1.004, -1.347, -0.640),
            _effect("unscoreable", "comparison", 0.537, 0.133, 0.939),
            _effect("mixed scale", "superseded", 0.215, -0.242, 0.684)]


def test_the_three_published_effects_print_side_by_side_with_their_spread():
    """What a reader of the pre-#206 output got, at the numbers the document carried by hand.
    The spread is 1.541 -- larger than any of the three is from zero, which is the finding --
    and it is subtracted from the numbers in front of it rather than quoted.

    `restricted=False`, because that is the universe these three were measured on. The block
    says which universe it is printing, so the same three rows cannot be read as the check
    #206 turned them into.
    """
    said = _flat(G.treatment_report(_published(), restricted=False))
    assert "-1.004 95% CI [-1.347, -0.640]" in said
    assert "+0.537 95% CI [+0.133, +0.939]" in said
    assert "+0.215 95% CI [-0.242, +0.684]" in said
    assert "spread across treatments 1.541 " + G.UNIT in said
    assert "larger than any effect any of them reports (1.004)" in said
    assert "rank-interpolated (primary)" in said and "mixed scale (superseded)" in said
    assert "pre-#206 universe" in said, \
        "and it says that this is the universe where the choice was still live"
    assert "STILL REACHES THE RESULT" not in said, \
        "a live spread here is the finding, not an alarm"


def test_a_spread_smaller_than_the_effects_says_smaller():
    """The comparison is computed, not asserted. A gate whose fallback stopped mattering would
    print a line that reads correctly rather than one restating 2026-09-07."""
    tight = [_effect("rank-interpolated", "primary", -1.004, -1.1, -0.9),
             _effect("unscoreable", "comparison", -0.900, -1.0, -0.8)]
    said = _flat(G.treatment_report(tight, restricted=False))
    assert "spread across treatments 0.104" in said
    assert "smaller than any effect any of them reports (1.004)" in said


def test_one_treatment_is_not_a_comparison_and_prints_nothing():
    """A number beside itself is the single-treatment report this replaces, and a spread of
    zero printed across it would read as a finding. An unscored treatment is listed rather
    than dropped, because a missing row reads as one nobody ran."""
    assert G.treatment_report(_published()[:1], restricted=False) == []
    empty = G.TreatmentEffect(G.Treatment("empty", "comparison", "no rows"),
                              G.summarise(pl.DataFrame()), _seasons([]))
    said = _flat(G.treatment_report([*_published(), empty], restricted=False))
    assert "empty (comparison)" in said and "nothing scored" in said
    assert "spread across treatments 1.541" in said, "and it does not enter the spread"


# --- and #206: the fallback is not chosen, it is removed --------------------
#
# The three treatments above are 1.5 points apart, which is further than any of them is from
# zero, so the gate was reporting a modelling choice nobody had justified. #206's disposition
# is not a fourth choice: both arms score only the roster-weeks both can price, `method.md`
# rule 6 holds by construction rather than by two arms applying a fallback symmetrically, and
# the cells the treatments disagreed about are the cells nothing scores.
#
# What it costs is that the result speaks for that share of the slate and no more, which the
# gate prints, and that the same question about the rest is unanswered, which the gate also
# prints. These hold both halves.

def _priced_universe():
    """Twelve players, nine of them priceable, and a week whose arithmetic is on paper.

    One QB, two RBs and a TE fill their slots however they are scored. Eight receivers
    compete for three WR slots and the flex, and the last three of them are the cells the
    model cannot price -- carrying a huge *fallback* number and a top consensus rank, so that
    the pre-#206 universe starts them in both arms and the restriction can start them in
    neither. Their realised points are 1,000 each, so any arm that fields one says so loudly.

      restricted   consensus starts 8, 4, 5 and flexes 6 -> 0+1+2+3  =    6
                   weekly    starts 4, 5, 6 and flexes 7 -> 1+2+3+10 =   16, diff +10
      unrestricted consensus starts 9, 10, 11 and flexes 8          = 3000
                   weekly    starts 9, 10, 11 and flexes 4          = 3001, diff  +1
    """
    pos = ["QB", "RB", "RB", "TE"] + ["WR"] * 8
    n = len(pos)
    projected = np.zeros((n, 18), dtype=bool)
    projected[:9, 4] = True                     # the four forced slots and receivers 4-8

    cons = np.zeros((n, 18))
    cons[:, 4] = [-1.0, -2.0, -3.0, -4.0,
                  -50.0, -51.0, -52.0, -53.0, -49.0,      # the priceable receivers
                  -10.0, -11.0, -12.0]                    # and the three ranked best

    weekly = np.full((n, 18), G.UNRANKED)
    weekly[:, 4] = [10.0, 9.0, 8.0, 7.0,
                    30.0, 29.0, 28.0, 27.0, 1.0,
                    100.0, 90.0, 80.0]          # what the fallback put where no projection is

    realised = np.zeros((n, 18))
    realised[4:, 4] = [1.0, 2.0, 3.0, 10.0, 0.0, 1000.0, 1000.0, 1000.0]

    # The mask exactly as `assemble_universe` builds it: both arms can price him.
    addable = projected & (cons > G.UNRANKED)
    return _inputs(rosters={2024: [list(range(n))]}, pos={2024: pos},
                   realised={2024: realised}, consensus={2024: cons},
                   weekly={2024: weekly}, projected={2024: projected},
                   addable={2024: addable}, se={2024: np.zeros((n, 18))}, pool={2024: [[]]})


def test_neither_arm_starts_a_player_the_other_cannot_price():
    """The whole disposition in one frame. Both arms are restricted to the same nine cells,
    so the three the model cannot price -- worth a thousand points each, and ranked first by
    consensus -- are started by neither. Restricting one arm and not the other is the defect
    `_one_scale` names; restricting both is a smaller slate, which is what #206 chose."""
    got = G.compare(_priced_universe(), weeks=[5])
    assert got["consensus"].to_list() == [pytest.approx(6.0)]
    assert got["weekly"].to_list() == [pytest.approx(16.0)]
    assert got["diff"].to_list() == [pytest.approx(10.0)]


def test_the_pre_206_universe_started_them_in_both_arms():
    """The same rows unrestricted, which is what every published figure was measured on: the
    three unpriceable receivers are the top of the consensus page and carry the fallback's
    largest numbers, so both arms field all three and the gate compares two lineups chosen
    mostly by a guess. 3,000 against 3,001 for a difference of one point."""
    got = G.compare(_priced_universe(), weeks=[5], restrict=False)
    assert got["consensus"].to_list() == [pytest.approx(3000.0)]
    assert got["weekly"].to_list() == [pytest.approx(3001.0)]


def test_the_restriction_is_the_mask_the_gate_already_carried():
    """One definition, not two. `priced_by_both` is `addable` -- `(cons > UNRANKED) &
    projected`, built once in the assembly -- rather than a second `np.isnan` in this module
    deciding again what "both arms can price him" means. Two spellings of one idea is how the
    share a run prints comes to describe a different set of cells from the one it scored."""
    g = _priced_universe()
    assert G.priced_by_both(g, 2024) is g.addable[2024]


def test_the_fallback_cannot_move_the_restricted_result():
    """#206's claim, checked rather than asserted: the three treatments differ only in cells
    the restriction leaves out, so on the scored rows they are one number. Unrestricted, the
    same three rows and the same fixture give a spread of 2,985 points -- +1 where the
    fallback's guesses are fielded against -2,984 where they are not, which is the shape of
    the finding a real run put at 1.5 points."""
    g = _priced_universe()
    restricted = {e.treatment.name: e.summary["mean"]
                  for e in G.treatment_effects(g, weeks=[5])}
    assert set(restricted) == {"rank-interpolated", "unscoreable", "mixed scale"}
    assert list(restricted.values()) == [pytest.approx(10.0)] * 3, \
        "one number three times is the point: no treatment reaches a scored cell"

    loose = {e.treatment.name: e.summary["mean"]
             for e in G.treatment_effects(g, weeks=[5], restrict=False)}
    assert loose[G.COLUMN_TREATMENT] == pytest.approx(1.0)
    assert max(loose.values()) - min(loose.values()) == pytest.approx(2985.0), \
        "and the same fixture without the restriction is where the choice was worth points"


def test_a_spread_that_survives_the_restriction_is_an_alarm_and_not_a_finding():
    """A treatment reaching a scored cell means the restriction missed an arm, a week or the
    waiver rule -- and the effect above it is once again partly the fallback. Loud, in the
    register the VOID line and the ceiling use, because a spread printed quietly under a
    heading that says it should be zero reads as a rounding artifact."""
    said = _flat(G.treatment_report(_published()))
    assert "THE FALLBACK STILL REACHES THE RESULT" in said
    assert "disagree by 1.541" in said
    clean = [_effect("rank-interpolated", "was primary", -1.004, -1.347, -0.640),
             _effect("unscoreable", "was a comparison", -1.004, -1.347, -0.640)]
    quiet = _flat(G.treatment_report(clean))
    assert "spread across treatments 0.000" in quiet
    assert "STILL REACHES THE RESULT" not in quiet, "zero is the expected reading"
    assert "check on the restriction and not a choice between them" in quiet


def test_the_foresight_arm_is_restricted_with_the_other_two():
    """A ceiling that could start what the gate does not score would bound a different
    question, and would bound it about three hundred times too high on this fixture."""
    got = G.compare(_priced_universe(), weeks=[5], ceiling=True)
    assert got["foresight"].to_list() == [pytest.approx(16.0)]
    assert got["ceiling_diff"].to_list() == [pytest.approx(10.0)]
    loose = G.compare(_priced_universe(), weeks=[5], ceiling=True, restrict=False)
    assert loose["foresight"].to_list() == [pytest.approx(3010.0)]


def test_the_open_pool_sensitivity_does_not_carry_the_restriction_with_it():
    """`mask_pool` decides who may be *added* and the restriction decides who may be
    *started*. They were one mask read for one purpose until #206 and are now one mask read
    for two, so the sensitivity that opens the pool must not quietly open the lineup as
    well -- that would make `--open-pool` a different gate rather than a wider one."""
    got = G.compare(_priced_universe(), weeks=[5], mask_pool=False)
    assert got["weekly"].to_list() == [pytest.approx(16.0)]
    assert got["consensus"].to_list() == [pytest.approx(6.0)]


def _streamable():
    """A roster whose quarterback the model cannot price this week, and a free agent one.

    Nine held players -- one QB, two RBs, a TE and five receivers -- plus a free-agent QB.
    The held quarterback is the highest-scoring player on the roster and is outside the
    scored cells, so the *fieldable* lineup has an empty QB slot that a 3.0 free agent fills.
    Read on the whole roster, no swap improves anything and the answer is None.
    """
    pos = ["QB", "RB", "RB", "TE"] + ["WR"] * 5 + ["QB"]
    score = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 3.0])
    only = np.ones(len(pos), dtype=bool)
    only[0] = False
    return pos, score, only


def test_the_waiver_decision_reads_the_lineup_the_week_will_field():
    """A swap chosen against a lineup the restriction will not field is a swap chosen against
    a projection nothing scores. Both arms are handed the same `only`, so this is the
    restriction reaching the churn rule and not a rule either arm has to itself."""
    pos, score, only = _streamable()
    roster, pool, need = list(range(9)), [9], sum(STARTERS.values())
    assert G.waiver_swap(roster, pool, pos, score, need) is None, \
        "read on the whole roster the quarterback slot is already filled by the best player"
    assert G.waiver_swap(roster, pool, pos, score, need, only=only) == (9, 8), \
        "read on what the week can field it is empty, and streaming one is worth 3 points"


def test_a_player_outside_the_scored_cells_is_still_held_and_still_droppable():
    """The restriction is on the lineup arithmetic, not on the roster. A player nothing can
    price this week is not released -- he is unpriceable *this* week, and a rule that dropped
    him would be a roster rule invented by a measurement harness."""
    _, _, only = _streamable()
    kept = G.fieldable(range(9), only)
    assert 0 not in kept and kept == list(range(1, 9))
    assert G.fieldable(range(9), None) == list(range(9)), "and no mask is the whole roster"


def test_the_covered_share_is_counted_off_the_mask_the_gate_scored():
    """Criterion four: reported with the result rather than discovered by probing. Nine of
    twelve cells are priced by both arms, and the three that are not are three the model
    cannot project -- counted off `priced_by_both` itself, so the share printed beside a
    number is the share that number was measured over."""
    pop = G.priced_share(_priced_universe(), weeks=[5])
    assert pop["cells"] == 12.0
    assert pop["share"] == pytest.approx(0.75)
    assert pop["no_projection"] == pytest.approx(0.25)
    assert pop["no_rank"] == pytest.approx(0.0)
    off = G.priced_share(_priced_universe()._replace(covered=set()), weeks=[5])
    assert off["cells"] == 0.0
    assert G.priced_report(off) == [], "no cells is not a share of zero, it is no share"


def test_a_player_the_page_does_not_rank_leaves_the_scored_cells_too():
    """Both directions, and this one supersedes half a pre-registered rule. An unlisted
    player was *ranked last* on the argument that the absence is the incumbent's answer --
    which makes a refusal to price into a free lineup rule, the same thing that disqualified
    `unscoreable` in the other direction. `coverage` still counts him; nothing scores him."""
    g = _priced_universe()
    cons = g.consensus[2024].copy()
    cons[4, 4] = G.UNRANKED                      # receiver 4: projected, and now unlisted
    g = g._replace(consensus={2024: cons},
                   addable={2024: g.projected[2024] & (cons > G.UNRANKED)})
    pop = G.priced_share(g, weeks=[5])
    assert pop["share"] == pytest.approx(8 / 12)
    assert pop["no_rank"] == pytest.approx(1 / 12)
    assert G.coverage(g, weeks=[5])["unranked"] == pytest.approx(1 / 12) \
        == pytest.approx(pop["no_rank"]), \
        "still counted over every roster-week, including the ones nothing scores"
    # He was worth a point to the weekly arm and is now out of both arms' universe.
    assert G.compare(g, weeks=[5])["weekly"].to_list() == [pytest.approx(15.0)]


def test_a_roster_week_with_no_lineup_choice_left_is_counted_and_named():
    """The failure mode restricting the rows creates, and the reason it is printed.

    Shrink the priceable side until it fits inside the starting slots and every player either
    arm can price starts: the two arms field the identical team, the difference is zero by
    construction, and a gate that cannot disagree with itself is ADR-0012's structural zero
    arriving by a different road. It fired unprompted on the offline archive the first time
    #206 was run end to end, which is why it is a counted quantity rather than a caveat.
    """
    free = G.priced_share(_priced_universe(), weeks=[5])
    assert free["roster_weeks"] == 1.0
    assert free["forced"] == 0.0, "nine priceable against eight slots is a choice"

    g = _priced_universe()
    priced = g.projected[2024].copy()
    priced[8, 4] = False                         # eight priceable, and eight slots
    g = g._replace(projected={2024: priced},
                   addable={2024: priced & (g.consensus[2024] > G.UNRANKED)})
    pop = G.priced_share(g, weeks=[5])
    assert pop["forced"] == 1.0
    got = G.compare(g, weeks=[5])
    assert got["diff"].to_list() == [pytest.approx(0.0)], \
        "and the zero it warns about is a zero no arm chose"

    said = _flat(G.priced_report(pop))
    assert "no lineup choice in 100.0% of 1 scored roster-weeks" in said
    assert "THE ARMS NEVER DISAGREE" in said
    assert "could not have failed" in said
    assert "THE ARMS NEVER DISAGREE" not in _flat(G.priced_report(free)), \
        "the loud line is the 100% reading and not a threshold somebody picked"
    assert "no lineup choice in 0.0% of" in _flat(G.priced_report(free)), \
        "but the share is printed either way, because the in-between is the reader's call"


def test_the_result_says_what_it_covers_and_what_it_leaves_unanswered():
    """Criterion five, in the words the ticket asks for. A number that speaks for three
    quarters of a slate while reading as though it speaks for all of it is the failure this
    ticket is about, so the block says the share, says what is out and why, and says that the
    question about the rest is **unanswered** rather than absent, negative or implied."""
    said = _flat(G.priced_report(G.priced_share(_priced_universe(), weeks=[5])))
    assert "scored on the 75.0% of 12 roster-week cells both arms can price" in said
    assert "there is no fallback" in said.lower()
    assert "left out: 25.0% the model cannot project, 0.0% the consensus page does not rank"\
        in said
    assert "UNANSWERED: whether the weekly projection beats consensus on the 25.0% it "\
           "cannot price" in said
    assert "in either direction" in said, "unanswered is not a negative result"
    assert "What would answer it is a projection for those players" in said


# --- the gate names its own reads, however it is invoked (issue #247) ---------------------
#
# `main` is network-bound and `# pragma: no cover` for it; what is driven here is the whole of
# it with `assemble_universe` replaced by a fixture that records the one read it made. #192
# scoped `backtest.main` and `weekly_screen.main` with `reads_of_one_run` and left this gate
# unwired: called in-process by anything that has already read something -- the one gate run
# of #135 -- it inherited the enclosing run's reads and published a digest over bytes it never
# touched, a failure that looks exactly like a clean digest.

def test_the_gate_called_in_process_names_only_its_own_reads(monkeypatch, tmp_path, capsys):
    """Called from inside a run that has already read something, the gate's published digest
    covers the gate's reads and not the enclosing run's -- and the enclosing run still ends
    up holding both, because a scope narrows what a component reports and must never be a
    way for a run to lose a read."""
    from functools import partial

    from hub.config import data_digest
    from hub.fetch import nflverse as nv
    from hub.models.experiment import run_gate
    from hub.season import weekly_gate_data

    monkeypatch.setattr(nv, "_READ_THIS_RUN", {})
    outer = nv.Pin(source="player_stats", as_of=None, digest="0ut51de0", rows=1,
                   pinned_at=None)
    inner = nv.Pin(source="ff_rankings", as_of="2024-09-01", digest="1n51de01", rows=1,
                   pinned_at=None)
    # The enclosing run: something else in this process has read a source already.
    nv._remember(tmp_path / "the-enclosing-runs-entry.parquet", outer)

    def assembles(seasons, *, drafts, seed, shrink, expected):
        nv._remember(tmp_path / "the-gates-own-entry.parquet", inner)
        return _inputs()

    monkeypatch.setattr(weekly_gate_data, "assemble_universe", assembles)
    monkeypatch.setattr(G, "run_gate", partial(run_gate, record_width=False, bootstrap=100))
    out = tmp_path / "paired.parquet"
    assert G.main(["--run", "--seasons", "2024", "--drafts", "1", "--out", str(out)]) == 0
    stamped = pl.read_parquet(out)

    assert stamped["data_digest"].unique().to_list() == [data_digest([inner])], (
        "the gate's digest is not a digest over the gate's own read")
    assert stamped["data_digest"][0] != data_digest([outer, inner]), (
        "the gate published a digest over the enclosing run's reads as well as its own")
    assert "over 1 pinned source(s)" in capsys.readouterr().out
    assert sorted(p.source for p in nv.pins_this_run()) == ["ff_rankings", "player_stats"], (
        "the gate's read did not reach the run around it: scoping lost a read")
