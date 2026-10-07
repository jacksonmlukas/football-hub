"""What a weekly injury designation costs, this week.

Not `INJURY_BETA`, which prices a *preseason* designation against a *season-long* projection.
A player ruled out in week 1 misses week 1 and plays the other sixteen, so those two numbers
answer different questions and are not comparable.

All offline.
"""
import numpy as np
import polars as pl
import pytest

from hub.fetch.replay import serve
from hub.ledger import Ledger
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


# --- the baseline is strictly prior (#361, S4; method.md rules 2 and 18) ----
#
# `observations` once took a designated week's baseline as the player's mean over the healthy
# weeks of the WHOLE season, so a week-9 designation was measured against weeks 10-14 -- the
# held-out arm handed the realised in-season level, a column no Sunday has. The control is a
# check that can vary with the thing it is about: change what happens AFTER a designated week
# and the baseline for that week must not move. `_leaks_the_future` is that check; it is
# trusted only because the planted lookahead below makes it fire.

def _season_with_a_late_run(late_pts):
    """Eight healthy weeks at 10, a designation in week 9, then five healthy weeks of
    `late_pts` -- the future, as far as week 9 is concerned."""
    stats = ([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
             + [(2024, 9, "a", "WR", 4.0)]
             + [(2024, w, "a", "WR", late_pts) for w in range(10, 15)])
    return _stats(stats), _inj([(2024, 9, "a", "WR", "Questionable", "Limited")])


def _leaks_the_future(build) -> bool:
    """True if the week-9 baseline `build(injuries, stats)` returns moves when only weeks
    10-14 do."""
    quiet = build(*reversed(_season_with_a_late_run(10.0)))
    loud = build(*reversed(_season_with_a_late_run(40.0)))
    return quiet["baseline"].to_list() != loud["baseline"].to_list()


def _lookahead_observations(injuries, stats):
    """THE PLANT: the shipped lookahead, as it stood before #361 -- baseline grouped on
    `(gsis_id, season)` over every healthy week of the season, designated week or no."""
    full = (stats.select("season", "week", pl.col("player_id").alias("gsis_id"),
                         pl.col("fantasy_points_ppr").alias("pts"))
                 .join(injuries.select("season", "week", "gsis_id",
                                       pl.col("report_status").alias("status")),
                       on=["season", "week", "gsis_id"], how="full", coalesce=True)
                 .with_columns(pl.col("status").fill_null("Healthy")))
    base = (full.filter(pl.col("status") == "Healthy").group_by(["gsis_id", "season"])
                .agg(pl.col("pts").mean().alias("baseline")))
    return full.filter(pl.col("status") != "Healthy").join(base, on=["gsis_id", "season"])


def test_the_future_leak_check_fires_on_the_planted_lookahead():
    """The positive control (rule 18). If this is green the check below is not decorative."""
    assert _leaks_the_future(_lookahead_observations)


def test_a_designated_weeks_baseline_does_not_move_with_the_weeks_after_it():
    assert not _leaks_the_future(injury.observations)


def test_the_baseline_is_the_expanding_mean_of_strictly_earlier_healthy_weeks():
    """Weeks 1-8 at 10 and week 9 designated: baseline 10 whatever follows. A second
    designation after a later, hotter healthy run sees that run, and only that far."""
    stats = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                   + [(2024, 9, "a", "WR", 4.0)]
                   + [(2024, w, "a", "WR", 20.0) for w in range(10, 14)]
                   + [(2024, 14, "a", "WR", 5.0)]
                   + [(2024, w, "a", "WR", 99.0) for w in range(15, 19)])
    inj = _inj([(2024, 9, "a", "WR", "Questionable", "Limited"),
                (2024, 14, "a", "WR", "Questionable", "Limited")])
    got = injury.observations(inj, stats).sort("week")
    assert got["baseline"].to_list() == pytest.approx([10.0, (80.0 + 80.0) / 12.0])
    assert got["healthy_weeks"].to_list() == [8, 12]


def test_the_prior_weeks_must_reach_the_minimum_before_the_week_is_scored():
    """Six healthy weeks of history, *before* the designation. A week-5 designation with
    thirteen healthy weeks after it used to qualify on the whole season's count."""
    st = _stats([(2024, w, "a", "WR", 10.0) for w in range(1, 5)]
                + [(2024, 5, "a", "WR", 3.0)]
                + [(2024, w, "a", "WR", 10.0) for w in range(6, 19)])
    inj = _inj([(2024, 5, "a", "WR", "Questionable", "Limited")])
    assert injury.observations(inj, st).is_empty()


def test_a_designation_does_not_borrow_another_seasons_weeks():
    st = _stats([(2023, w, "a", "WR", 30.0) for w in range(1, 10)]
                + [(2024, w, "a", "WR", 10.0) for w in range(1, 9)]
                + [(2024, 9, "a", "WR", 4.0)])
    inj = _inj([(2024, 9, "a", "WR", "Questionable", "Limited")])
    assert injury.observations(inj, st)["baseline"].to_list() == pytest.approx([10.0])


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
    rows_st, rows_inj = [], []
    for season in (2023, 2024):
        for pid in ("a", "b", "c"):
            for w in range(1, 9):
                rows_st.append((season, w, pid, "WR", 10.0))
            for w in range(9, 18):
                rows_inj.append((season, w, pid, "WR", "Questionable", "Limited"))
                rows_st.append((season, w, pid, "WR", 4.0))
    # As nflverse ships them, which is what the loader's contracts accept: the injury report
    # names a team, a game type and a player; the stats are narrowed to the five columns read.
    serve(injuries=_inj(rows_inj).with_columns(
              team=pl.lit("AAA"), game_type=pl.lit("REG"), full_name=pl.col("gsis_id")),
          player_stats=_stats(rows_st))
    out = tmp_path / "t.parquet"
    assert injury.main(["--fit", "--seasons", "2023,2024", "--out", str(out)],
                       ledger=Ledger(path=None)) == 0
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

SEASONS = (2021, 2022, 2023, 2024, 2025)


def _errs(gain, noise, seasons=SEASONS, n=400, seed=0, headroom=0.3):
    """Antithetic noise, so each season's mean difference is exactly `gain[i]`. `err_oracle`
    is the declared ceiling arm's error: `headroom` better than `retention`, every row; pass
    `headroom=None` for a frame that never measured a ceiling."""
    rng = np.random.default_rng(seed)
    rows = []
    for si, season in enumerate(seasons):
        e = rng.normal(0, noise, n // 2)
        for d in np.concatenate([e, -e]):
            rows.append((season, 5.0, 5.0 - gain[si] + float(d)))
    out = {"season": [r[0] for r in rows], "k": [50.0] * len(rows),
           "err_retention": [r[1] for r in rows], "err_type": [r[2] for r in rows]}
    if headroom is not None:
        out["err_oracle"] = [r[1] - headroom for r in rows]
    return pl.DataFrame(out)


_CLEAN = [0.2, 0.21, 0.19, 0.2, 0.2]


def test_the_incumbent_is_retention_not_out_zero():
    """The thing to beat is what already won, not what it beat."""
    _, text = injury.type_verdict(_errs([0.0] * 5, noise=1.0))
    assert "retention" in text and "out_zero" not in text


def test_a_type_effect_that_wins_everywhere_and_clears_the_interval_is_adopted():
    winner, text = injury.type_verdict(_errs(_CLEAN, noise=0.4))
    assert winner == "type" and text.startswith("ADOPT")


def test_losing_one_season_is_not_enough():
    winner, text = injury.type_verdict(_errs([0.2, 0.2, 0.2, 0.2, -0.2], noise=0.4))
    assert winner == "retention" and text.startswith("KEEP")


def test_a_gain_the_seasons_cannot_agree_on_is_not_adopted():
    """Positive in every season and still not distinguishable from nothing: the season is the
    cluster, five of them, and they scatter wider than their mean. The retired rule read this
    as 2 se over ~2,000 pooled rows."""
    winner, text = injury.type_verdict(_errs([0.02, 0.2, 0.01, 0.3, 0.02], noise=0.4))
    assert winner == "retention" and "interval contains zero" in text


def test_no_ceiling_measured_is_not_runnable_even_when_the_effect_is_clean():
    """S6 reaches this comparison for the first time (#343): the retired rule adopted on this
    frame. The same rows with `err_oracle` do adopt (above), so what moved the verdict is the
    ceiling and nothing else."""
    winner, text = injury.type_verdict(_errs(_CLEAN, noise=0.4, headroom=None))
    assert winner == "retention" and "NOT-RUNNABLE" in text
    assert "not measured a ceiling" in text
    assert "ceiling_diff" not in injury.type_frame(_errs(_CLEAN, noise=0.4, headroom=None)).columns


def test_a_ceiling_below_the_design_s_resolution_is_not_runnable():
    """Stage 2: the same clean gain against a headroom of 0.001 -- smaller than the smallest
    effect five seasons at this noise could resolve."""
    winner, text = injury.type_verdict(_errs(_CLEAN, noise=0.4, headroom=0.001))
    assert winner == "retention" and "NOT-RUNNABLE" in text and "ceiling of +0.001" in text


def test_the_type_arm_is_removed_when_worse_in_every_season():
    run = injury.type_run(_errs([-0.2, -0.21, -0.19, -0.2, -0.2], noise=0.4))
    assert run.verdict[0] == "REMOVE"
    assert injury.type_verdict(pl.DataFrame({}), run=run)[0] == "retention"


def test_the_ceiling_is_retention_minus_the_in_sample_oracle_per_row():
    frame = injury.type_frame(_errs(_CLEAN, noise=0.4))
    assert set(frame["ceiling_diff"].round(9).to_list()) == {0.3}
    assert {"season", "diff"} <= set(frame.columns)


def test_no_held_out_season_reports_nothing_measured():
    winner, text = injury.type_verdict(pl.DataFrame())
    assert winner == "retention" and "nothing measured" in text


def test_the_published_run_prints_stage_2_and_the_stamps_and_writes_one_entry():
    ledger = Ledger(path=None)
    run = injury.type_run(_errs(_CLEAN, noise=0.4), publish=True, seasons=SEASONS, ledger=ledger)
    text = "\n".join(run.lines)
    assert "MDE at 80% power" in text and "ceiling (" in text
    assert "data:" in text and "board:" in text and "commit:" in text
    assert [(e.name, e.recipe) for e in ledger._entries] == [
        ("injury_type", "baseline=strictly-prior,seasons=2021+2022+2023+2024+2025")]


def test_a_width_on_the_lookahead_baseline_does_not_compare_with_one_on_the_strictly_prior():
    """#436. The two baselines share config and data digests (the digests do not see code), so
    the recipe is the only thing separating their ledger entries. Planted: the pre-#361 entry,
    spelled as it was written (`seasons=...` alone), must NOT compare with a run now."""
    import dataclasses
    ledger = Ledger(path=None)
    injury.type_run(_errs(_CLEAN, noise=0.4), publish=True, seasons=SEASONS, ledger=ledger)
    (new,) = ledger._entries
    old = dataclasses.replace(new, recipe="seasons=2021+2022+2023+2024+2025")
    assert (old.config_digest, old.data_digest) == (new.config_digest, new.data_digest)
    assert not new.comparable(old)
    assert new.comparable(new)
    assert (new.recipe or "").startswith("baseline=strictly-prior")


def test_the_oracle_is_fitted_in_sample_on_the_held_out_season_itself():
    """The declared ceiling arm: its per-type multiplier comes from the season it is scored on,
    so on a fixture where type matters it is at least as good as the walk-forward arm."""
    rows_st, rows_inj = [], []
    for season in (2023, 2024):
        for t, mult in (("Hamstring", 0.5), ("Ankle", 1.0)):
            for i in range(30):
                pid = f"{t}{season}{i}"
                for w in range(1, 9):
                    rows_st.append((season, w, pid, "WR", 10.0))
                rows_st.append((season, 9, pid, "WR", 10.0 * 0.6 * mult))
                rows_inj.append((season, 9, pid, "WR", "Questionable", "Limited", t))
    obs = injury.observations(_inj_typed(rows_inj), _stats(rows_st))
    errs = injury.walk_forward_type(obs, min_cell=1)
    assert "err_oracle" in errs.columns
    oracle, typed, retention = (float(errs[c].to_numpy().mean())
                                for c in ("err_oracle", "err_type", "err_retention"))
    assert oracle <= typed + 1e-9
    assert oracle < retention


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
