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
    assert G.lineup_points(realised, pos, score).tolist() == [50.0, 50.0]
    static = np.column_stack([score[:, 0], score[:, 0]])
    assert G.lineup_points(realised, pos, static)[1] == 0.0, \
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
    return {(2024, 0): cons}, {(2024, 0): realised}


def test_coverage_separates_a_correct_omission_from_a_join_failure():
    """A player who is out scores zero and being unranked is the incumbent's *answer*. A
    player who scored and was unranked is a name that did not match, and only that one biases
    the comparison."""
    cons, realised = _cov_fixture()
    c = G.coverage(cons, realised, {(2024, 5)}, weeks=[5])
    assert c["unranked"] == pytest.approx(2 / 13)
    assert c["join_failure"] == pytest.approx(1 / 13)


def test_coverage_ignores_weeks_the_incumbent_does_not_cover():
    cons, realised = _cov_fixture()
    assert G.coverage(cons, realised, set(), weeks=[5])["cells"] == 0


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
    rosters = {2024: [[(f"p{i}", p) for i, p in enumerate(pos)]]}
    n = len(pos)
    arrays = {(2024, 0): np.ones((n, 18))}
    scores = {(2024, 0): np.tile(np.arange(n, dtype=float).reshape(-1, 1), (1, 18))}
    out = G.compare(rosters, arrays, scores, scores, covered={(2024, 3), (2024, 4)},
                    weeks=[3, 4, 5])
    assert out.height == 2 and sorted(out["week"].to_list()) == [3, 4]
    assert (out["diff"] == 0.0).all(), "identical arms differ by nothing"


# --- the score matrices, where the defaults are the whole argument ----------

def test_a_missing_value_takes_the_stated_default():
    """The rule the entire void analysis turned on: a player absent from the consensus page
    gets UNRANKED, a player with no realised row gets zero, and neither is a null that a
    later step silently reinterprets."""
    from hub.season.weekly_gate_data import _matrix
    m = _matrix(["a", "b"], {("a", 1): 5.0, ("a", 3): 7.0}, G.UNRANKED)
    assert m[0, 0] == 5.0 and m[0, 2] == 7.0
    assert m[0, 1] == G.UNRANKED, "a week he is missing from is unranked, not carried forward"
    assert (m[1, :] == G.UNRANKED).all(), "and a player missing entirely is unranked all season"


def test_the_matrix_covers_the_whole_regular_season():
    from hub.draft.season import REG_SEASON_WEEKS
    from hub.season.weekly_gate_data import _matrix
    assert _matrix(["a"], {}, 0.0).shape == (1, REG_SEASON_WEEKS)
