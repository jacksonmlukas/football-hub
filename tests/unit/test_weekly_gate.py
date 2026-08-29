"""Gate B: does a lineup set off the Weekly projection beat one set off consensus rank?

All offline. The first live run returned +11.1 points per team-week at P(better) 100% and was
VOID -- 6.2% of roster-weeks were a join failure against a pre-registered 2% floor. These tests
pin the machinery that said so.
"""
import numpy as np
import polars as pl
import pytest

from hub.season import weekly_gate as G


def _pos(n_qb=2, n_rb=4, n_wr=5, n_te=2):
    return ["QB"] * n_qb + ["RB"] * n_rb + ["WR"] * n_wr + ["TE"] * n_te


# --- the lineup rule --------------------------------------------------------

def test_required_slots_fill_before_the_flex():
    pos = _pos()
    score = np.arange(len(pos), dtype=float)          # last players score highest
    idx = G.starters_by_score(pos, score)
    got = [pos[i] for i in idx]
    from hub.draft.season import FLEX_SLOTS, STARTERS
    for p, need in STARTERS.items():
        assert got.count(p) >= need, f"{p} short of its required {need}"
    assert len(idx) == sum(STARTERS.values()) + FLEX_SLOTS


def test_the_flex_takes_the_best_leftover_not_the_first():
    pos = ["QB", "RB", "RB", "RB", "WR", "WR", "WR", "WR", "TE"]
    score = np.array([1, 9, 8, 7, 6, 5, 4, 3, 2], dtype=float)
    idx = G.starters_by_score(pos, score)
    assert 3 in idx, "the third RB at 7 is the best flex-eligible leftover"


def test_a_quarterback_cannot_fill_the_flex():
    pos = ["QB", "QB", "RB", "RB", "WR", "WR", "WR", "TE"]
    score = np.array([9, 8, 1, 1, 1, 1, 1, 1], dtype=float)
    idx = G.starters_by_score(pos, score)
    assert [pos[i] for i in idx].count("QB") == 1, "the second QB scores highest and still sits"


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

def _cov_fixture():
    pos = _pos()
    n = len(pos)
    cons = np.zeros((n, 18))
    realised = np.zeros((n, 18))
    cons[0, 4] = G.UNRANKED          # unranked and scored -> a join failure
    realised[0, 4] = 12.0
    cons[1, 4] = G.UNRANKED          # unranked and scored nothing -> correctly benched
    return ({2024: [list(range(n))]}, {2024: cons}, {2024: realised})


def test_coverage_separates_a_correct_omission_from_a_join_failure():
    """A player who is out scores zero and being unranked is the incumbent's *answer*. A
    player who scored and was unranked is a name that did not match, and only that one biases
    the comparison."""
    ros, cons, realised = _cov_fixture()
    c = G.coverage(ros, cons, realised, {(2024, 5)}, weeks=[5])
    assert c["unranked"] == pytest.approx(2 / 13)
    assert c["join_failure"] == pytest.approx(1 / 13)


def test_coverage_ignores_weeks_the_incumbent_does_not_cover():
    ros, cons, realised = _cov_fixture()
    assert G.coverage(ros, cons, realised, set(), weeks=[5])["cells"] == 0


# --- the verdict, every branch ---------------------------------------------

def _summary(mean, lo, hi, clusters=60):
    return {"n": 800.0, "clusters": float(clusters), "mean": mean, "lo": lo, "hi": hi,
            "p_better": 1.0 if lo > 0 else 0.0}


def _seasons(gains):
    return pl.DataFrame({"season": list(range(2022, 2022 + len(gains))),
                         "gain": gains, "n": [200] * len(gains)})


def test_a_join_failure_voids_the_run_however_large_the_result():
    """The branch that fired on the first live run: +11.1 points per team-week at P 100%,
    which is the repo's own rule that a result too large to believe is a bug."""
    status, note = G.verdict(_summary(11.1, 10.1, 12.3), _seasons([10.0, 10.2, 13.2]),
                             {"cells": 680.0, "unranked": 0.157, "join_failure": 0.062})
    assert status == "VOID"
    assert "6.2%" in note and "2%" in note


def test_a_clean_join_lets_the_result_through():
    status, _ = G.verdict(_summary(0.9, 0.3, 1.5), _seasons([0.8, 1.0, 0.9]),
                          {"cells": 680.0, "unranked": 0.1, "join_failure": 0.005})
    assert status == "ADOPT"


def test_adopt_needs_every_season_as_well_as_the_interval():
    status, note = G.verdict(_summary(0.9, 0.3, 1.5), _seasons([-0.2, 1.4, 1.5]), None)
    assert status == "SHOW" and "2/3" in note


def test_losing_in_every_season_removes_the_module():
    status, note = G.verdict(_summary(-1.2, -1.8, -0.6), _seasons([-1.0, -1.3, -1.3]), None)
    assert status == "REMOVE" and "Delete" in note


def test_an_interval_containing_zero_is_shown_never_ranked_on():
    """The expected branch, and it carries an action rather than a disappointment."""
    status, note = G.verdict(_summary(0.2, -0.4, 0.8), _seasons([0.1, 0.4, 0.1]), None)
    assert status == "SHOW"
    assert "NEVER RANK ON" in note and "absence of evidence" in note


def test_nothing_measured_does_not_adopt():
    assert G.verdict(_summary(0.0, 0.0, 0.0, clusters=0), _seasons([]), None)[0] == "SHOW"


# --- pairing and the cluster bootstrap -------------------------------------

def _paired(rosters=6, weeks=10, gain=1.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(rosters):
        offset = rng.normal(0, 5.0)      # a roster-level effect, shared by all its weeks
        for w in range(1, weeks + 1):
            d = gain + offset + rng.normal(0, 1.0)
            rows.append({"season": 2024, "roster": k, "week": w,
                         "consensus": 100.0, "weekly": 100.0 + d, "diff": d})
    return pl.DataFrame(rows)


def test_the_bootstrap_resamples_rosters_not_rows():
    """A roster's weeks share its players, its bye and its draft. Resampling rows would treat
    ten readings as ten observations and report an interval far too narrow -- protocol item 3,
    which turned noise into an apparent 4-sigma result once already."""
    paired = _paired()
    clustered = G.cluster_bootstrap(paired, bootstrap=2000, seed=1)
    d = paired["diff"].to_numpy()
    rng = np.random.default_rng(1)
    naive = rng.integers(0, len(d), size=(2000, len(d)))
    naive_width = float(np.percentile(d[naive].mean(axis=1), 97.5)
                        - np.percentile(d[naive].mean(axis=1), 2.5))
    assert clustered["hi"] - clustered["lo"] > 2 * naive_width, \
        "clustering must widen the interval when the roster effect is real"
    assert clustered["clusters"] == 6 and clustered["n"] == 60


def test_an_empty_frame_reports_rather_than_crashing():
    s = G.cluster_bootstrap(pl.DataFrame())
    assert s["clusters"] == 0 and np.isnan(s["mean"])


def test_compare_emits_one_row_per_roster_week_and_skips_uncovered_weeks():
    pos = _pos()
    n = len(pos)
    rosters = {2024: [list(range(n))]}
    arrays = {2024: np.ones((n, 18))}
    scores = {2024: np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))}
    out = G.compare_universe(rosters, {2024: pos}, arrays, scores, scores, {2024: [[]]},
                             covered={(2024, 3), (2024, 4)}, weeks=[3, 4, 5])
    assert out.height == 2 and sorted(out["week"].to_list()) == [3, 4]
    assert (out["diff"] == 0.0).all(), "identical arms differ by nothing"


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
