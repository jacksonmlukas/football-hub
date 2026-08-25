"""What a weekly injury designation costs, this week.

Not `INJURY_BETA`, which prices a *preseason* designation against a *season-long* projection.
A player ruled out in week 1 misses week 1 and plays the other sixteen, so those two numbers
answer different questions and are not comparable.

All offline.
"""
import numpy as np
import polars as pl
import pytest

from hub.models import injury


def _inj(rows):
    """(season, week, gsis_id, position, report_status, practice_status)."""
    cols = ["season", "week", "gsis_id", "position", "report_status", "practice_status"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"season": pl.Int32, "week": pl.Int32, "gsis_id": pl.Utf8,
                                "position": pl.Utf8, "report_status": pl.Utf8,
                                "practice_status": pl.Utf8})


def _stats(rows):
    """(season, week, player_id, position, points)."""
    cols = ["season", "week", "player_id", "position", "fantasy_points_ppr"]
    return pl.DataFrame({c: [r[i] for r in rows] for i, c in enumerate(cols)},
                        schema={"season": pl.Int32, "week": pl.Int32, "player_id": pl.Utf8,
                                "position": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _healthy_then(pid, healthy_pts, designated, season=2024):
    """`healthy_pts` weeks of clean play, then `designated` = [(week, status, practice, pts)]."""
    stats = [(season, w, pid, "WR", p) for w, p in enumerate(healthy_pts, start=1)]
    inj, off = [], len(healthy_pts)
    for i, (status, practice, pts) in enumerate(designated, start=1):
        inj.append((season, off + i, pid, "WR", status, practice))
        if pts is not None:
            stats.append((season, off + i, pid, "WR", pts))
    return _stats(stats), _inj(inj)


# --- the observation set --------------------------------------------------

def test_a_players_baseline_is_his_own_healthy_weeks():
    st, inj = _healthy_then("a", [10.0] * 8, [("Questionable", "Limited", 4.0)])
    obs = injury.observations(inj, st)
    assert obs.height == 1
    assert obs["baseline"][0] == pytest.approx(10.0)
    assert obs["delta"][0] == pytest.approx(-6.0)


def test_a_designated_week_with_no_stat_row_scores_zero_not_null():
    """He is exactly the outcome being priced. Dropping him would measure the cost of an
    injury among players who played through it."""
    st, inj = _healthy_then("a", [10.0] * 8, [("Out", "Did Not Participate", None)])
    obs = injury.observations(inj, st)
    assert obs.height == 1
    assert obs["pts"][0] == 0.0


def test_a_player_without_enough_healthy_weeks_is_excluded():
    """Below six, one good game defines the level everything else is measured against."""
    st, inj = _healthy_then("a", [10.0] * 3, [("Questionable", "Limited", 4.0)])
    assert injury.observations(inj, st).is_empty()


def test_healthy_weeks_are_not_themselves_observations():
    st, inj = _healthy_then("a", [10.0] * 8, [("Questionable", "Limited", 4.0)])
    obs = injury.observations(inj, st)
    assert obs["status"].to_list() == ["Questionable"]


def test_non_drafted_positions_are_excluded():
    st = _stats([(2024, w, "k", "K", 8.0) for w in range(1, 10)])
    inj = _inj([(2024, 10, "k", "K", "Questionable", "Limited")])
    assert injury.observations(inj, st).is_empty()


def test_duplicate_injury_rows_for_one_week_collapse():
    """nflverse occasionally carries two rows for one player-week."""
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)])
    inj = _inj([(2024, 9, "a", "WR", "Questionable", "Limited"),
                (2024, 9, "a", "WR", "Questionable", "Limited")])
    assert injury.observations(inj, st).height == 1


# --- retention is multiplicative, and that is the point -------------------

def test_out_retains_exactly_nothing():
    """The shape that additive cannot express: an Out player scores zero regardless of how
    good he is, and `retention = 0` says so exactly."""
    st, inj = _healthy_then("a", [10.0] * 8, [("Out", "Did Not Participate", None)] * 70)
    obs = injury.observations(inj, st)
    tab = injury.retention_table(obs, min_cell=10)
    assert tab["retention"][0] == pytest.approx(0.0)


def test_retention_is_a_ratio_of_totals_not_a_mean_of_ratios():
    """A player whose healthy baseline is near zero produces a ratio near infinity, and
    averaging those measures nothing."""
    st = _stats([(2024, w, "a", "WR", 20.0) for w in range(1, 9)]
                + [(2024, w, "b", "WR", 0.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 10.0), (2024, 9, "b", "WR", 0.0)])
    inj = _inj([(2024, 9, "a", "WR", "Questionable", "Limited"),
                (2024, 9, "b", "WR", "Questionable", "Limited")])
    tab = injury.retention_table(injury.observations(inj, st), min_cell=1)
    # totals: 10 points kept out of 20 baseline -> 0.5, and no nan from b's zero baseline
    assert tab["retention"][0] == pytest.approx(0.5)


def test_a_thin_cell_is_not_reported():
    st, inj = _healthy_then("a", [10.0] * 8, [("Questionable", "Limited", 4.0)])
    assert injury.retention_table(injury.observations(inj, st)).is_empty()


def test_an_unseen_cell_falls_back_rather_than_predicting_zero():
    """Zero would assert an unseen designation costs everything; the pooled rate is the
    honest default."""
    st, inj = _healthy_then("a", [10.0] * 8, [("Doubtful", "Did Not Participate", 0.0)])
    obs = injury.observations(inj, st)
    empty = injury.retention_table(obs)          # too thin, so no cells at all
    got = injury.predict_retention(obs, empty, fallback=0.5)
    assert got[0] == pytest.approx(5.0)


def test_prediction_scales_the_players_own_baseline():
    """Two players in the same cell get different predictions, because they are different
    players. A flat penalty cannot do that."""
    st = _stats([(2024, w, "big", "WR", 20.0) for w in range(1, 9)]
                + [(2024, w, "small", "WR", 5.0) for w in range(1, 9)]
                + [(2024, 9, "big", "WR", 10.0), (2024, 9, "small", "WR", 2.5)])
    inj = _inj([(2024, 9, "big", "WR", "Questionable", "Limited"),
                (2024, 9, "small", "WR", "Questionable", "Limited")])
    obs = injury.observations(inj, st).sort("baseline")
    tab = injury.retention_table(obs, min_cell=1)
    got = injury.predict_retention(obs, tab, fallback=1.0)
    assert got[1] > got[0], "the higher-baseline player must be predicted higher"


# --- the gate -------------------------------------------------------------

def _wf(base, out_zero, table, retention):
    return pl.DataFrame({"season": [2024, 2025], "n": [500, 500],
                         "mae_baseline": base, "mae_out_zero": out_zero,
                         "mae_table": table, "mae_retention": retention})


def test_a_table_that_beats_both_simple_rules_is_adopted():
    winner, text = injury.verdict(_wf([6.0, 6.0], [4.2, 4.2], [4.9, 4.9], [4.0, 4.0]))
    assert winner == "retention" and text.startswith("ADOPT")


def test_beating_only_the_null_is_not_enough():
    """Beating "ignore the injury report" shows injuries matter, which nobody doubts. The
    rule worth beating is the one a manager already follows for free."""
    winner, text = injury.verdict(_wf([6.0, 6.0], [4.2, 4.2], [5.0, 5.0], [4.5, 4.5]))
    assert winner == "out_zero" and text.startswith("KEEP")


def test_the_additive_table_losing_is_recorded_not_hidden():
    """It lost its own gate, and the reason -- Out is multiplicative -- is the finding."""
    winner, text = injury.verdict(_wf([6.0, 6.0], [4.2, 4.2], [4.9, 4.9], [4.0, 4.0]))
    assert "table" in text and winner == "retention"


def test_no_held_out_seasons_reports_nothing_measured():
    winner, text = injury.verdict(pl.DataFrame(schema={"season": pl.Int32}))
    assert winner == "baseline" and "nothing measured" in text


def test_the_walk_forward_fits_only_on_earlier_seasons():
    """The leak that would make any fitted table look good."""
    rows_st, rows_inj = [], []
    for season in (2023, 2024):
        for w in range(1, 9):
            rows_st.append((season, w, "a", "WR", 10.0))
        for w in range(9, 16):
            rows_inj.append((season, w, "a", "WR", "Questionable", "Limited"))
            rows_st.append((season, w, "a", "WR", 5.0 if season == 2023 else 1.0))
    obs = injury.observations(_inj(rows_inj), _stats(rows_st))
    wf = injury.walk_forward(obs, min_cell=1)
    assert wf["season"].to_list() == [2024]


def test_the_two_baselines_are_the_ones_a_person_would_use():
    import inspect
    src = inspect.getsource(injury.walk_forward)
    assert "out_zero" in src and "baseline" in src


# --- the CLI --------------------------------------------------------------

def test_help_needs_no_network():
    assert injury.main([]) == 0


def test_the_fit_path_runs_offline(monkeypatch, capsys, tmp_path):
    import nflreadpy as nfl
    rows_st, rows_inj = [], []
    for season in (2023, 2024):
        for pid in ("a", "b", "c"):
            for w in range(1, 9):
                rows_st.append((season, w, pid, "WR", 10.0))
            for w in range(9, 18):
                rows_inj.append((season, w, pid, "WR", "Questionable", "Limited"))
                rows_st.append((season, w, pid, "WR", 4.0))
    monkeypatch.setattr(nfl, "load_injuries", lambda *a, **k: _inj(rows_inj))
    monkeypatch.setattr(nfl, "load_player_stats", lambda *a, **k: _stats(rows_st))
    out = tmp_path / "t.parquet"
    assert injury.main(["--fit", "--seasons", "2023,2024", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "designated player-weeks" in text
    assert out.exists()


# --- does what is wrong with him add anything? ------------------------------
#
# `retention` prices a designation by (status, practice) and ignores the injury type. The gate
# for adding it is stricter than the one `retention` itself cleared, because the incumbent is
# now the thing that already won: every held-out season AND 2 se on the paired difference.

def _inj_typed(rows):
    """(season, week, gsis_id, position, report_status, practice_status, injury)."""
    base = _inj([r[:6] for r in rows])
    return base.with_columns(pl.Series("report_primary_injury", [r[6] for r in rows]))


def test_the_injury_type_is_carried_through():
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 4.0)])
    inj = _inj_typed([(2024, 9, "a", "WR", "Questionable", "Limited", "Hamstring")])
    assert injury.observations(inj, st)["injury"].to_list() == ["hamstring"]


def test_a_missing_injury_type_is_unknown_not_dropped():
    """53% of rows are null upstream. Dropping them would price injuries among the players
    whose injury happened to be reported."""
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 4.0)])
    obs = injury.observations(_inj([(2024, 9, "a", "WR", "Questionable", "Limited")]), st)
    assert obs.height == 1 and obs["injury"].to_list() == ["unknown"]


def test_an_absent_column_degrades_rather_than_raising():
    """Older nflverse slices, and every existing fixture in this file."""
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 4.0)])
    obs = injury.observations(_inj([(2024, 9, "a", "WR", "Questionable", "Limited")]), st)
    assert injury.predict_with_type(obs, injury.retention_table(obs, min_cell=1),
                                    {}, fallback=0.5).shape == (1,)


# --- the multiplier -------------------------------------------------------

def _typed_obs(n_per_type=40):
    rows_st, rows_inj = [], []
    for t, mult in (("Hamstring", 0.5), ("Ankle", 1.0)):
        for i in range(n_per_type):
            pid = f"{t}{i}"
            for w in range(1, 9):
                rows_st.append((2024, w, pid, "WR", 10.0))
            rows_st.append((2024, 9, pid, "WR", 10.0 * 0.6 * mult))
            rows_inj.append((2024, 9, pid, "WR", "Questionable", "Limited", t))
    return injury.observations(_inj_typed(rows_inj), _stats(rows_st))


def test_a_type_that_underperforms_the_table_gets_a_multiplier_below_one():
    obs = _typed_obs()
    tab = injury.retention_table(obs, min_cell=1)
    adj = injury.type_adjustment(obs, tab, fallback=0.6, k=1.0)
    assert adj["hamstring"] < 0.9 < adj["ankle"]


def test_shrinkage_pulls_a_thin_type_toward_no_adjustment():
    """A type with little evidence must change nothing, which is what lets thin types
    contribute in proportion to their evidence instead of being trusted or dropped."""
    obs = _typed_obs()
    tab = injury.retention_table(obs, min_cell=1)
    loose = injury.type_adjustment(obs, tab, fallback=0.6, k=1.0)
    tight = injury.type_adjustment(obs, tab, fallback=0.6, k=100000.0)
    assert abs(tight["hamstring"] - 1.0) < abs(loose["hamstring"] - 1.0)


def test_an_empty_adjustment_is_exactly_the_incumbent():
    """The candidate can only win by earning it: no adjustment reproduces `retention`."""
    obs = _typed_obs()
    tab = injury.retention_table(obs, min_cell=1)
    base = injury.predict_retention(obs, tab, fallback=0.6)
    assert injury.predict_with_type(obs, tab, {}, fallback=0.6) == pytest.approx(base)


def test_shrinkage_is_chosen_on_training_rows_only():
    obs = _typed_obs()
    tab = injury.retention_table(obs, min_cell=1)
    assert injury.fit_shrink(obs, tab, fallback=0.6) in injury.SHRINK_GRID


# --- the gate -------------------------------------------------------------

def _errs(gain, noise, seasons=(2024, 2025), n=400, seed=0):
    """Antithetic noise, so each season's mean difference is exactly `gain`."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, season in enumerate(seasons):
        e = rng.normal(0, noise, n // 2)
        for d in np.concatenate([e, -e]):
            rows.append((season, 5.0, 5.0 - gain[si] + float(d)))
    return pl.DataFrame({"season": [r[0] for r in rows], "k": [50.0] * len(rows),
                         "err_retention": [r[1] for r in rows],
                         "err_type": [r[2] for r in rows]})


def test_the_incumbent_is_retention_not_out_zero():
    """The thing to beat is what already won, not what it beat."""
    _, text = injury.type_verdict(_errs([0.0, 0.0], noise=1.0))
    assert "retention" in text and "out_zero" not in text


def test_a_type_effect_that_wins_everywhere_and_clears_two_se_is_adopted():
    winner, text = injury.type_verdict(_errs([0.5, 0.5], noise=1.0))
    assert winner == "type" and text.startswith("ADOPT")


def test_losing_one_season_is_not_enough():
    winner, text = injury.type_verdict(_errs([2.0, -0.5], noise=1.0))
    assert winner == "retention" and text.startswith("KEEP")


def test_a_gain_too_small_to_distinguish_from_noise_is_not_adopted():
    winner, _ = injury.type_verdict(_errs([0.2, 0.2], noise=3.0))
    assert winner == "retention"


def test_no_held_out_season_reports_nothing_measured():
    winner, text = injury.type_verdict(pl.DataFrame())
    assert winner == "retention" and "nothing measured" in text


def test_the_type_walk_forward_fits_only_on_earlier_seasons():
    rows_st, rows_inj = [], []
    for season in (2023, 2024):
        for i in range(30):
            pid = f"p{season}{i}"
            for w in range(1, 9):
                rows_st.append((season, w, pid, "WR", 10.0))
            rows_st.append((season, 9, pid, "WR", 5.0))
            rows_inj.append((season, 9, pid, "WR", "Questionable", "Limited", "Knee"))
    obs = injury.observations(_inj_typed(rows_inj), _stats(rows_st))
    assert injury.walk_forward_type(obs, min_cell=1)["season"].unique().to_list() == [2024]


def test_laterality_and_case_collapse_to_one_category():
    """nflverse passes the club's wording straight through: `Shoulder`, `Right Shoulder` and
    `left Shoulder` were three categories for one injury, splitting its evidence three ways.
    110 distinct raw values across 2022-25, 74 after this."""
    st = _stats([(2024, w, p, "WR", 10.0) for p in ("a", "b", "c") for w in range(1, 9)]
                + [(2024, 9, p, "WR", 4.0) for p in ("a", "b", "c")])
    inj = _inj_typed([(2024, 9, "a", "WR", "Questionable", "Limited", "Shoulder"),
                      (2024, 9, "b", "WR", "Questionable", "Limited", "Right Shoulder"),
                      (2024, 9, "c", "WR", "Questionable", "Limited", "left Shoulder")])
    assert set(injury.observations(inj, st)["injury"].to_list()) == {"shoulder"}


def test_laterality_is_only_stripped_from_the_front():
    """`right Thumb` is a thumb; a hypothetical injury whose name merely contains the word
    should not be mangled."""
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 4.0)])
    inj = _inj_typed([(2024, 9, "a", "WR", "Questionable", "Limited", "Upper right arm")])
    assert injury.observations(inj, st)["injury"].to_list() == ["upper right arm"]
