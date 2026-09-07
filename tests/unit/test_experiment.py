"""The walk-forward paired experiment, which this repo runs twice.

`backtest` and `lineup_gate` both build a board as it stood before each season opened, load
what happened, play two arms, and report a paired interval. The season loop was byte-identical
in both and the report block differed by one word. None of it was reachable from a test,
because both copies lived inside a `main()` that needs a network.

All offline.
"""
import functools
import hashlib
import json
import math
import re
import statistics
from typing import Any

import numpy as np
import polars as pl
import pytest

from hub.models import experiment


def _board(n=8):
    return pl.DataFrame({"player": [f"P{i}" for i in range(n)],
                         "pos": ["RB", "WR"] * (n // 2)})


def _stats(season, n=8):
    rows = []
    for i in range(n):
        for wk in range(1, 5):
            rows.append((season, wk, f"id{i}", f"P{i}", "RB", 10.0 + i))
    return pl.DataFrame(
        {"season": [r[0] for r in rows], "week": [r[1] for r in rows],
         "player_id": [r[2] for r in rows], "player_display_name": [r[3] for r in rows],
         "position": [r[4] for r in rows],
         "fantasy_points_ppr": [r[5] for r in rows]})


class _Report:
    """The half of a Board's build report a Gate reads, stubbed.

    Structural, like the protocol it satisfies, and for the same reason: the module under
    test must not reach into `hub.draft`, so a test of its plumbing has no business doing it
    either. That the *real* report satisfies the protocol is one test below, once, where a
    drift between the two would be caught rather than assumed.
    """

    def __init__(self, *missing: str) -> None:
        self._missing = missing

    def corrections_missing(self) -> tuple[str, ...]:
        return self._missing


# --- the season loop --------------------------------------------------------

def test_inputs_are_gathered_per_season():
    boards, realised = experiment.walk_forward_inputs(
        [2023, 2024],
        lambda yr: (_board(), _Report()),
        load_stats=_stats)
    assert set(boards) == {2023, 2024} and set(realised) == {2023, 2024}
    assert boards[2023].height == 8 and realised[2024].height > 0


def test_the_board_is_built_as_of_that_season_s_opening():
    """A strategy scored against rankings published after the season is hindsight wearing a
    backtest's clothes. The rule lives in `hub.draft.board`, beside the `build` that enforces
    it -- it started here, which put draft-domain knowledge under `models/` and inverted the
    tree's one consistent direction.

    The as-of reads August 31 and used to read September 1. It selects the same rows either
    way: `board.consensus` is now inclusive of its as-of day, one convention with the loader
    that bounds the archive, so the date it asks for moved back by the day that change would
    otherwise have added. `tests/unit/test_consensus_page.py` holds the boundary itself.
    """
    import inspect

    from hub.draft import board
    src = inspect.getsource(board.board_as_of)
    assert 'as_of=f"{season}-08-31"' in src
    assert "season=season - 1" in src


def test_experiment_does_not_reach_into_draft():
    """Six `draft/` modules import `models/`; nothing should point back. A function-local
    import is the tell that one does."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(experiment))
    reaches = [n.module for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("hub.draft")]
    assert reaches == [], f"experiment reaches into draft: {reaches}"


# --- the Board a Gate is handed (issue #131) --------------------------------
#
# `build` degrades stage by stage on purpose, and two of those stages leave a column a
# **Correction** reads. Absorbing one does not leave a thinner Board -- it leaves one whose
# Corrected ADP is a different ranking, reported as having run. Every Gate in the repo reached
# the Board through this function and took `[0]`, so four seasons could be scored with one of
# them built that way and the published interval said nothing about it.


def test_a_gate_refuses_a_board_missing_a_correction_it_ranks_on():
    """The refusal, and the three things the sentence has to carry: which season, which term,
    and that the season is a *different ranking* rather than a thinner board -- because
    "built without td_luck" was read as the second for as long as anything read it at all.
    """
    with pytest.raises(experiment.CorrectionMissing) as refused:
        experiment.walk_forward_inputs(
            [2023, 2024],
            lambda yr: (_board(), _Report() if yr == 2023 else _Report("touchdown luck")),
            load_stats=_stats)
    said = str(refused.value)
    assert "2024" in said, "which season has to be in it; the other three are fine"
    assert "touchdown luck" in said
    assert "thinner" in said and "different ranking" in said


def test_a_gate_scores_a_board_that_carries_every_correction():
    """The other half, and the one that stops the guard being widened into a Gate that never
    runs. A whole Board is scored without comment."""
    boards, realised = experiment.walk_forward_inputs(
        [2023, 2024], lambda yr: (_board(), _Report()), load_stats=_stats)
    assert set(boards) == {2023, 2024} and set(realised) == {2023, 2024}


def test_a_board_that_computed_no_corrected_ranking_is_not_refused():
    """An ECR-only board -- which is every board `board_as_of` builds, since ESPN publishes
    ADP for the current season only -- has no corrected ranking to have been computed from a
    subset. `corrections_missing` already says so by returning nothing, and a Gate that
    refused it would refuse every backtest this repo runs."""
    from hub.draft.board import BuildReport
    experiment.require_corrections(2024, BuildReport(adp=False, td_luck=False))


def test_the_real_build_report_answers_the_protocol_a_gate_types_against():
    """`CorrectionReport` is structural so that this module never imports `hub.draft`, and the
    price of structural typing is that nothing checks the two ends agree. This is that check,
    once, against the class the three call sites actually hand over."""
    from hub.draft.board import BuildReport
    whole = BuildReport(adp=True, td_luck=True, durability=True)
    experiment.require_corrections(2024, whole)
    with pytest.raises(experiment.CorrectionMissing, match="durability"):
        experiment.require_corrections(2024, BuildReport(adp=True, td_luck=True))


def test_the_progress_hook_is_a_hook_not_a_print(capsys):
    """A caller under a line cap must be able to stay quiet, and this module must not own
    stdout -- the same reason `hub.draft.report` returns lines."""
    seen = []
    experiment.walk_forward_inputs([2024], lambda yr: (_board(), _Report()),
                                   load_stats=_stats, on_season=seen.append)
    assert seen == [2024]
    assert capsys.readouterr().out == ""


def test_no_hook_is_silent(capsys):
    experiment.walk_forward_inputs([2024], lambda yr: (_board(), _Report()),
                                   load_stats=_stats)
    assert capsys.readouterr().out == ""


def test_one_column_list_so_both_harnesses_share_a_cache_entry():
    """`nflverse._cache_path` keys on the sorted column set -- deliberately, so a caller
    asking for six columns is never served an earlier caller's five. Two harnesses asking
    for different slices therefore downloaded the same table twice."""
    import inspect

    from hub.draft import backtest
    from hub.season import lineup_gate
    for mod in (backtest, lineup_gate):
        src = inspect.getsource(mod)
        assert 'cols=["player_id"' not in src, f"{mod.__name__} still asks for its own slice"
    assert "fantasy_points_ppr" in experiment.PLAYER_STATS_COLS


# --- the paired report ------------------------------------------------------

def _summary(mean=-19.13, lo=-22.31, hi=-15.75, p=0.0, n=80):
    return {"n": n, "mean": mean, "lo": lo, "hi": hi, "p_better": p}


def test_the_report_names_both_arms():
    lines = experiment.paired_report(_summary(), arm_a="optimizer", arm_b="market")
    joined = "\n".join(lines)
    assert "optimizer - market" in joined
    assert "n=80" in joined and "-19.13" in joined
    assert "95% CI [-22.31, -15.75]" in joined
    assert "P(optimizer better) 0.0%" in joined


def test_the_unit_is_the_caller_s(capsys):
    """The two harnesses measure different things -- points per team game against points per
    game -- and the block was duplicated rather than parameterised for exactly that word."""
    a = "\n".join(experiment.paired_report(_summary(), arm_a="x", arm_b="y"))
    b = "\n".join(experiment.paired_report(_summary(), arm_a="x", arm_b="y",
                                           unit="points per game"))
    assert "points per team game" in a and "points per team game" not in b


def test_a_positive_result_keeps_its_sign():
    lines = experiment.paired_report(_summary(mean=0.17, lo=0.09, hi=0.25, p=0.998),
                                 arm_a="retention", arm_b="out_zero")
    assert "+0.17" in lines[0], "the sign is the finding; it must not be formatted away"


def test_the_report_returns_lines_and_prints_nothing(capsys):
    experiment.paired_report(_summary(), arm_a="a", arm_b="b")
    assert capsys.readouterr().out == ""


# --- the expanding-window split, which was written four times ---------------
#
# `margin.walk_forward`, `injury.walk_forward`, `injury.walk_forward_type` and
# `spread.walk_forward` each carried their own copy, already differing three ways. It is
# `docs/method.md` rule #2 -- the leakage invariant the repo records violating at 7.4 sigma --
# so four hand-written copies was four places a `<` could become a `<=` silently. Only one was
# pinned by a test. These are that test, against the one implementation.


def _seasons(pairs):
    """A frame of (season, row-id) pairs."""
    return pl.DataFrame({"season": [s for s, _ in pairs], "id": [i for _, i in pairs]})


def test_past_is_strictly_before_the_scored_season():
    """The invariant itself. Nothing fitted may have seen the season it is scored on."""
    df = _seasons([(y, i) for y in (2019, 2020, 2021, 2022) for i in range(5)])
    seen = []
    for yr, past, now in experiment.expanding_seasons(df):
        assert int(past["season"].max()) < yr, "past leaked the scored season"  # type: ignore[arg-type]
        assert set(now["season"].to_list()) == {yr}
        assert past.height + now.height == df.filter(pl.col("season") <= yr).height
        seen.append(yr)
    assert seen == [2020, 2021, 2022], "the earliest season is training data only"


def test_the_window_expands_rather_than_slides():
    """Every earlier season stays in `past` -- that is what makes it expanding."""
    df = _seasons([(y, i) for y in (2019, 2020, 2021) for i in range(3)])
    heights = {yr: past.height for yr, past, _ in experiment.expanding_seasons(df)}
    assert heights == {2020: 3, 2021: 6}


def test_min_past_is_rows_not_seasons():
    """`margin` needs two residuals before `fit` has a standard deviation to give."""
    df = _seasons([(2019, 0), (2020, 0), (2020, 1), (2021, 0)])
    assert [yr for yr, _, _ in experiment.expanding_seasons(df, min_past=2)] == [2021]
    assert [yr for yr, _, _ in experiment.expanding_seasons(df)] == [2020, 2021]


def test_a_thin_year_is_skipped_not_raised():
    """A walk-forward that stopped at the first thin year would report nothing at all."""
    df = _seasons([(2019, 0), (2020, 0), (2021, 0), (2022, 0)])
    assert [yr for yr, _, _ in experiment.expanding_seasons(df, min_past=3)] == [2022]


def test_one_season_yields_nothing():
    assert list(experiment.expanding_seasons(_seasons([(2021, 0), (2021, 1)]))) == []


def test_an_empty_frame_yields_nothing():
    assert list(experiment.expanding_seasons(
        pl.DataFrame({"season": [], "id": []}, schema={"season": pl.Int64, "id": pl.Int64}))) == []


def test_gaps_in_the_record_do_not_break_the_order():
    """Seasons are sorted, not counted -- a missing year is history, not a hole."""
    df = _seasons([(2018, 0), (2021, 0), (2022, 0)])
    assert [yr for yr, _, _ in experiment.expanding_seasons(df)] == [2021, 2022]


def test_the_season_column_is_nameable():
    df = pl.DataFrame({"yr": [2019, 2020], "id": [0, 1]})
    got = [yr for yr, _, _ in experiment.expanding_seasons(df, season_col="yr")]
    assert got == [2020]


def test_the_split_is_written_once():
    """The AST guard that stops it drifting back to four copies.

    A strict `<` on the season column is the leakage split and belongs to
    `expanding_seasons`. `margin` legitimately holds two `>=` trailing windows, but those cut
    *inside* `past`, which is already strictly earlier -- they are not this.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    offenders = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "experiment.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Compare) or not node.ops:
                continue
            if not isinstance(node.ops[0], (ast.Lt, ast.LtE)):
                continue
            left = node.left
            if (isinstance(left, ast.Call) and isinstance(left.func, ast.Attribute)
                    and left.func.attr == "col" and left.args
                    and isinstance(left.args[0], ast.Constant)
                    and left.args[0].value in ("season", "yr", "year")):
                offenders.append(f"{path.relative_to(src)}:{node.lineno}")
    assert not offenders, (
        "an expanding-window split written outside hub.models.experiment -- this is "
        f"docs/method.md rule #2 and it belongs in one place: {offenders}")


# --- the paired statistic, which was written twice --------------------------
#
# `spread.verdict` and `injury.type_verdict` held it verbatim, three of five lines
# byte-identical, under two names for one bar (`MIN_SE` and `TYPE_MIN_SE`, both 2.0).


def test_gain_is_positive_when_the_arm_has_the_smaller_error():
    """Sign convention: the difference is base minus arm, so positive favours the arm."""
    g = experiment.paired_gain([3.0, 3.0, 3.0], [1.0, 1.0, 1.0],
                               base_mae=[3.0], arm_mae=[1.0])
    assert g.mean == 2.0
    assert g.wins == 1


def test_the_standard_error_is_of_the_difference():
    """Hand-computed: d = [1, 2, 3], sd(ddof=1) = 1, se = 1/sqrt(3)."""
    import math
    g = experiment.paired_gain([2.0, 4.0, 6.0], [1.0, 2.0, 3.0],
                               base_mae=[4.0], arm_mae=[2.0])
    assert g.mean == 2.0
    assert math.isclose(g.se, 1.0 / math.sqrt(3))
    assert math.isclose(g.t, 2.0 / (1.0 / math.sqrt(3)))


def test_a_constant_difference_is_not_significant_by_division_by_zero():
    """Zero variance means zero standard error, and a t of 0 rather than an infinity."""
    g = experiment.paired_gain([2.0, 2.0], [1.0, 1.0], base_mae=[2.0], arm_mae=[1.0])
    assert g.se == 0.0 and g.t == 0.0


def test_one_observation_cannot_clear_a_significance_bar():
    g = experiment.paired_gain([5.0], [1.0], base_mae=[5.0], arm_mae=[1.0])
    assert g.se == 0.0 and g.t == 0.0


def test_seasons_won_counts_seasons_not_observations():
    """The every-season half of the gate. Two of three seasons won is not all three."""
    g = experiment.paired_gain([1.0] * 30, [1.0] * 30,
                               base_mae=[2.0, 2.0, 1.0], arm_mae=[1.0, 1.0, 3.0])
    assert (g.wins, g.seasons) == (2, 3)


def test_a_tie_is_not_a_win():
    g = experiment.paired_gain([1.0, 1.0], [1.0, 1.0], base_mae=[1.0], arm_mae=[1.0])
    assert g.wins == 0


def test_the_bar_is_declared_once():
    """One name for one threshold. It was `spread.MIN_SE` and `injury.TYPE_MIN_SE`, both 2.0,
    each commented "the repo's usual bar"."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    declared = []
    for path in sorted(src.rglob("*.py")):
        if path.name == "experiment.py":
            continue
        for node in ast.parse(path.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
            for t in targets:
                if isinstance(t, ast.Name) and t.id.endswith("MIN_SE"):
                    declared.append(f"{path.relative_to(src)}:{t.id}")
    assert not declared, f"a second significance bar: {declared}"


# --- the resampling unit, which is the interface's whole job ---

def _clustered(seed=0):
    """Six rosters, ten weeks each, with a real per-roster effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(6):
        offset = rng.normal(0, 3.0)
        for w in range(1, 11):
            rows.append({"season": 2024, "roster": k, "week": w,
                         "diff": offset + rng.normal(0, 1.0)})
    return pl.DataFrame(rows)


def test_the_cluster_order_does_not_move_the_interval():
    """Clusters are sorted before resampling. They used to arrive in `.unique()` order and the
    bootstrap indexes into that order, so a permutation moved the interval while leaving the
    mean alone -- which is why weekly-blend-gate.md records [-0.249, +0.659] for the same
    +0.215 a re-run reports as [-0.251, +0.663]."""
    df = _clustered()
    shuffled = df.sample(fraction=1.0, shuffle=True, seed=7)
    a = experiment.summarise(df, cluster=("season", "roster"), bootstrap=2000, seed=1)
    b = experiment.summarise(shuffled, cluster=("season", "roster"), bootstrap=2000, seed=1)
    assert a == b


def test_clustering_widens_the_interval_when_the_cluster_effect_is_real():
    df = _clustered()
    wide = experiment.summarise(df, cluster=("season", "roster"), bootstrap=2000, seed=1)
    narrow = experiment.summarise(df, bootstrap=2000, seed=1)
    assert wide["hi"] - wide["lo"] > narrow["hi"] - narrow["lo"]


def test_the_mean_is_the_same_either_way_for_balanced_clusters():
    """Only the interval should move. A mean that shifted would mean clustering had changed
    the estimate rather than its precision."""
    df = _clustered()
    wide = experiment.summarise(df, cluster=("season", "roster"), bootstrap=500, seed=1)
    narrow = experiment.summarise(df, bootstrap=500, seed=1)
    assert wide["mean"] == pytest.approx(narrow["mean"])
    assert wide["n"] == narrow["n"] == 60
    assert wide["clusters"] == 6


def test_an_unclustered_summary_still_reports_its_unit_count():
    """`clusters` is always present, so a reader never has to know which branch ran."""
    s = experiment.summarise(_clustered(), bootstrap=200, seed=1)
    assert s["clusters"] == s["n"] == 60


# --- #45 criterion 1: the season as the unit --------------------------------
#
# The load-bearing test of the whole ticket, and the one with a way to pass while proving
# nothing. `summarise(cluster=("season",))` and `summarise()` return the same interval on any
# frame whose rows are already independent, so a fixture built without a real season effect
# would make both assertions below true no matter what the clustering code did. Two ceiling
# tests in this repo stood on exactly that shape for months.
#
# So the fixture is built to the claim #45 actually makes -- *within-season rows are
# near-identical* -- and `test_the_fixture_would_catch_a_clustering_that_did_nothing` proves
# the fixture can tell the two estimators apart before anything else asserts on it.


def _near_identical_seasons(seed=0):
    """Four seasons, twenty rows each, with almost all the variance *between* seasons.

    The shape #45 argues the real gates have: rows inside a season share a board, a pool and
    one realisation of the year, so they differ by a whisker (sd 0.05) while the seasons
    differ by a lot (sd 3.0). Row-resampling sees eighty observations and reports an interval
    about sqrt(20) too narrow; season-resampling sees the four there are.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in range(2022, 2026):
        level = rng.normal(0.0, 3.0)
        for draft in range(20):
            rows.append({"season": season, "draft": draft,
                         "diff": level + rng.normal(0.0, 0.05)})
    return pl.DataFrame(rows)


def test_the_fixture_would_catch_a_clustering_that_did_nothing():
    """Before asserting that clustering widens, prove the fixture *can* show it not widening.

    A fixture on which the row bootstrap and the season bootstrap agree would let a
    `summarise` that ignored `cluster` entirely pass every assertion below it. Here they
    disagree by more than a factor of four, so an implementation that silently dropped the
    argument would fail rather than pass quietly.
    """
    df = _near_identical_seasons()
    row = experiment.summarise(df, bootstrap=2000, seed=1)
    season = experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=2000,
                                  seed=1)
    assert row["clusters"] == 80 and season["clusters"] == 4
    ratio = (season["hi"] - season["lo"]) / (row["hi"] - row["lo"])
    assert ratio > 4.0, f"the two estimators barely differ on this fixture (x{ratio:.2f})"


def test_clustering_on_the_season_widens_the_interval_on_near_identical_rows():
    """Criterion 1, first half. The interval that treats twenty near-identical rows as twenty
    observations is far too narrow, and clustering is what admits it."""
    df = _near_identical_seasons()
    row = experiment.summarise(df, bootstrap=2000, seed=1)
    season = experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=2000,
                                  seed=1)
    assert season["hi"] - season["lo"] > row["hi"] - row["lo"]
    assert season["se"] > row["se"]


def test_clustering_on_the_season_leaves_the_mean_alone_on_balanced_seasons():
    """Criterion 1, second half. Only the precision moves. A mean that shifted would mean
    clustering had changed the estimate rather than what is claimed about it -- and on
    balanced seasons the mean of the season means *is* the mean of the rows.

    **This half cannot fail on a `summarise` that ignores clustering entirely**, and that is
    not a weakness to hide: the equality it asserts is true precisely because both estimators
    give the same mean here. Mutation-tested on 2026-09-07 -- disabling the cluster branch
    leaves this green. So it also asserts the clustering actually *happened*, which is what
    stops it reading as though it covered both halves, and the widening half above is what
    carries the load.
    """
    df = _near_identical_seasons()
    row = experiment.summarise(df, bootstrap=2000, seed=1)
    season = experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=2000,
                                  seed=1)
    assert season["mean"] == pytest.approx(row["mean"])
    assert season["n"] == row["n"] == 80
    assert season["clusters"] == 4 and row["clusters"] == 80, \
        "the two summaries resampled the same unit, so the equality above proves nothing"


def test_an_unbalanced_season_moves_the_mean_and_that_is_the_estimator_not_a_bug():
    """The other half of the claim, stated so it cannot be mistaken for the first. Clustering
    reweights: each season counts once whatever its row count. On a *balanced* frame that is
    the same number as the row mean, which is why the criterion says balanced -- and asserting
    it on an unbalanced frame too would be asserting something false."""
    df = pl.concat([_near_identical_seasons(),
                    pl.DataFrame({"season": [2022] * 40, "draft": list(range(20, 60)),
                                  "diff": [50.0] * 40})])
    row = experiment.summarise(df, bootstrap=500, seed=1)
    season = experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=500,
                                  seed=1)
    assert season["mean"] != pytest.approx(row["mean"])


# --- #45 criterion 2: the reference distribution the cluster count justifies ------

# The published table, so the quantile is checked against something outside this repo rather
# than against itself. Two-sided 95%, i.e. the 0.975 quantile.
_T_TABLE = {1: 12.70620, 2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582,
            7: 2.364624, 8: 2.306004, 10: 2.228139, 30: 2.042272, 100: 1.983972,
            1000: 1.962339}


@pytest.mark.parametrize("df", sorted(_T_TABLE))
def test_the_t_quantile_matches_the_table(df):
    """`_t_cdf` closes in elementary functions for integer df and is inverted by bisection.
    Both halves are wrong in ways that would still look plausible -- an off-by-one in the odd
    recursion moves df=3 to 2.35 -- so it is pinned to the published values."""
    assert experiment.t_quantile(0.975, df) == pytest.approx(_T_TABLE[df], abs=1e-5)


def test_the_mde_is_the_t_quantile_not_the_normal_one():
    """Criterion 2, and the amendment of 2026-09-07 that motivates it. At four clusters the
    normal quantile is the wrong reference distribution, and it is wrong in the direction that
    flatters the gate: it makes the published SE and the published interval consistent with
    each other rather than correct."""
    normal = (statistics.NormalDist().inv_cdf(0.975)
              + statistics.NormalDist().inv_cdf(0.80))
    got = experiment.minimum_detectable_effect(3.67, clusters=4)
    assert got == pytest.approx((3.182446 + 0.8416212) * 3.67, abs=1e-4)
    assert got > normal * 3.67


def test_the_published_arithmetic_reproduces():
    """The two numbers #45 states, recomputed rather than trusted: 10.29 against 14.77 on the
    draft gate's published season-clustered SE of 3.67, and the 1.44x between them.

    The normal figure lands on 10.28 from an SE rounded to 3.67 and on 10.29 from the 3.6712
    that rounds to it, which is the rounding in the published table rather than a
    disagreement -- so the ratio, which is free of it, is what the assertion rests on."""
    se = 3.67
    z = statistics.NormalDist().inv_cdf(0.975) + statistics.NormalDist().inv_cdf(0.80)
    t = experiment.minimum_detectable_effect(se, clusters=4)
    assert z * se == pytest.approx(10.28, abs=0.01)
    assert t == pytest.approx(14.77, abs=0.01)
    assert t / (z * se) == pytest.approx(1.44, abs=0.005)


def test_the_standard_error_under_the_mde_is_the_intervals_own_bootstrap():
    """The half of the rule that a second bootstrap agreeing by luck would fake.

    **Asserting `mde == minimum_detectable_effect(se, k)` does not test this**, and the first
    version of this test did exactly that: `mde` is computed *from* `se`, so that identity
    holds however `se` was obtained. A `summarise` that drew a whole second bootstrap under a
    different seed and reported its standard deviation passed it -- measured, by mutation, on
    2026-09-07. That is precisely the defect `docs/gate-power.md` names.

    So the draws are reconstructed here from the documented algorithm -- the seed, the RNG, the
    index draw -- and `lo`, `hi` and `se` are all required to come out of that one vector. A
    second bootstrap cannot satisfy this one, because a different seed gives a different
    standard deviation and nothing about the reconstruction is approximate.
    """
    df = _near_identical_seasons()
    s = experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=2000, seed=1)

    units = (df.group_by(["season"]).agg(pl.col("diff").mean().alias("u"))
               .sort(["season"])["u"].to_numpy().astype(float))
    rng = np.random.default_rng(1)
    draws = units[rng.integers(0, len(units), size=(2000, len(units)))].mean(axis=1)

    assert s["lo"] == float(np.percentile(draws, 2.5))
    assert s["hi"] == float(np.percentile(draws, 97.5))
    assert s["se"] == float(draws.std(ddof=1)), \
        "the reported SE is not the standard deviation of the interval's own draws"
    assert s["mde"] == experiment.minimum_detectable_effect(s["se"], int(s["clusters"]))

    # And it tracks the interval it sits under: a wider interval is a bigger SE is a bigger
    # MDE. The row-clustered run of the same frame is the comparison.
    row = experiment.summarise(df, bootstrap=2000, seed=1)
    assert (s["hi"] - s["lo"] > row["hi"] - row["lo"]) and s["mde"] > row["mde"]


def test_a_single_cluster_has_no_mde_rather_than_an_infinite_one():
    """One observation has no degrees of freedom for a t. NaN is `Field.NO_DATA` and renders
    no line, which is the honest answer -- a run with one season has no power to report."""
    one = pl.DataFrame({"season": [2024] * 5, "diff": [1.0, 2.0, 3.0, 4.0, 5.0]})
    s = experiment.summarise(one, cluster=experiment.SEASON_CLUSTER, bootstrap=200, seed=1)
    assert s["clusters"] == 1
    assert experiment.reading(s, "mde") is experiment.Field.NO_DATA


# --- #45 criteria 3 and 4: what a four-cluster run has to show ---------------


def _small(clusters=4):
    df = pl.DataFrame({"season": list(range(2022, 2022 + clusters)),
                       "diff": [1.0 + i for i in range(clusters)]})
    return experiment.summarise(df, cluster=experiment.SEASON_CLUSTER, bootstrap=500, seed=1)


def test_the_t_interval_is_reported_beside_the_percentile_one_at_few_clusters():
    """Criterion 3. A nonparametric percentile bootstrap over four units under-covers, so both
    are printed and neither is presented as the answer."""
    lines = experiment.small_sample_report(_small(4))
    joined = "\n".join(lines)
    assert "95% t CI" in joined and "percentile" in joined
    assert "4 clusters" in joined


@pytest.mark.parametrize("k", [2, 4, 8])
def test_both_intervals_show_at_or_below_eight_clusters(k):
    assert experiment.small_sample_report(_small(k)) != []


def test_nothing_is_printed_above_eight_clusters():
    """The block is confined to the case that motivates it: a gate with many units prints
    exactly what it printed before."""
    many = experiment.summarise(_clustered(), cluster=("season", "roster"), bootstrap=200,
                                seed=1)
    assert many["clusters"] == 6
    big = experiment.summarise(_clustered(), bootstrap=200, seed=1)
    assert big["clusters"] == 60
    assert experiment.small_sample_report(big) == []


def test_the_two_intervals_actually_differ_so_printing_both_says_something():
    """Otherwise this is two renderings of one number. The t interval is the wider here,
    which is the direction the under-coverage argument predicts."""
    s = _small(4)
    half = experiment.t_quantile(0.975, 3) * s["se"]
    assert (s["mean"] - half) < s["lo"] and (s["mean"] + half) > s["hi"]


def test_the_four_season_means_are_printed():
    """Criterion 4. An interval over four replications is a claim a reader should be able to
    check by eye -- one season carrying the whole effect is visible in the four and invisible
    in the interval."""
    seasons = _gate_seasons([0.3, -0.5, 0.9, 0.1])
    lines = experiment.small_sample_report(_small(4), seasons)
    joined = "\n".join(lines)
    assert "4 replications" in joined
    for year, gain in zip(range(2022, 2026), ["+0.30", "-0.50", "+0.90", "+0.10"],
                          strict=True):
        assert f"{year} {gain}" in joined


def test_the_season_means_are_omitted_rather_than_faked_when_none_are_handed_in():
    joined = "\n".join(experiment.small_sample_report(_small(4)))
    assert "replications" not in joined


# --- #45 criterion 5: an interval that narrowed has to say so ----------------


def test_a_narrower_interval_prints_the_ratio_and_requires_review():
    """Criterion 5, and it exists because this happened. On 2026-09-07 the weekly screen
    (#169) came back with five of nine intervals *narrower* under clustering -- the opposite
    of what the argument predicts -- and it was caught only because someone compared."""
    got = experiment.narrowing(width=1.0, previous=2.0)
    assert got.requires_review
    joined = "\n".join(got.lines)
    assert "REQUIRES REVIEW" in joined and "NARROWER" in joined
    assert "0.50" in joined, "the ratio itself has to be on the line"


def test_a_wider_interval_is_noted_and_does_not_require_review():
    got = experiment.narrowing(width=2.0, previous=1.0)
    assert not got.requires_review and got.ratio == pytest.approx(2.0)
    assert "2.00x" in "\n".join(got.lines)


def test_an_equal_interval_does_not_require_review():
    """The boundary. Narrower is the trigger, not different."""
    assert not experiment.narrowing(width=1.0, previous=1.0).requires_review


def test_a_first_run_has_nothing_to_compare_and_says_nothing():
    got = experiment.narrowing(width=1.0, previous=None)
    assert got.lines == [] and not got.requires_review


def test_the_width_is_recorded_for_the_next_run_and_carries_the_review_flag(tmp_path):
    """Printed lines scroll past; the record is what a later reader has. Two runs of the same
    gate: the first has nothing to compare against, the second compares against the first."""
    path = tmp_path / "gate-width.json"
    first = {"lo": -2.0, "hi": 2.0, "clusters": 4.0}
    assert experiment.review_width("draft", first, path=path) == []
    assert json.loads(path.read_text())["draft"]["width"] == pytest.approx(4.0)

    second = {"lo": -0.5, "hi": 0.5, "clusters": 4.0}
    said = experiment.review_width("draft", second, path=path)
    assert "REQUIRES REVIEW" in "\n".join(said)
    entry = json.loads(path.read_text())["draft"]
    assert entry["requires_review"] is True and entry["width"] == pytest.approx(1.0)


def test_one_gate_s_history_is_not_another_s(tmp_path):
    """Keyed by name, so three gates do not overwrite each other and read a narrowing that is
    really a different gate's interval."""
    path = tmp_path / "gate-width.json"
    experiment.review_width("draft", {"lo": -2.0, "hi": 2.0, "clusters": 4.0}, path=path)
    assert experiment.review_width("weekly", {"lo": -0.5, "hi": 0.5, "clusters": 4.0},
                                   path=path) == []
    assert set(json.loads(path.read_text())) == {"draft", "weekly"}


def test_an_unreadable_history_costs_a_line_and_not_the_run(tmp_path):
    """CLAUDE.md's degradation rule. A gate that cannot read its own history still has a
    verdict; a harness that dies because a JSON file is half-written does not."""
    path = tmp_path / "gate-width.json"
    path.write_text("{ this is not json")
    assert experiment.review_width("draft", {"lo": -1.0, "hi": 1.0, "clusters": 4.0},
                                   path=path) == []


# --- the Gate, which was a rule three modules each remembered --------------------
#
# `CONTEXT.md` defines a Gate exactly -- does this beat the simplest thing that already
# works? -- and three modules answered it with the same three branches on a confidence
# interval. They had already diverged: the weekly gate required the sign to hold in every
# held-out season before adopting, the lineup gate and the draft backtest adopted on the
# pooled interval alone. Nobody decided that.
#
# The drift had already cost something. `hub.models.spread`'s verdict records in its own
# docstring that an earlier version checked the seasons alone and "would have adopted a model
# on a gain too small to distinguish from noise". That copy was found weak and strengthened;
# the others were never revisited, because nothing connected them.

_ACTIONS = experiment.Actions(
    adopt="ADOPT: the arm ships.",
    remove="REMOVE: delete it.",
    show="SHOW, NEVER RANK ON: printed beside the incumbent.")


def _gate_seasons(gains):
    return pl.DataFrame({"season": list(range(2022, 2022 + len(gains))),
                         "gain": [float(g) for g in gains],
                         "n": [10] * len(gains)})


def _gate_summary(lo, hi, clusters=80):
    return {"n": 800.0, "clusters": float(clusters), "mean": (lo + hi) / 2,
            "lo": lo, "hi": hi, "p_better": 0.5}


def test_an_interval_above_zero_in_every_season_adopts():
    status, said = experiment.gate(_gate_summary(0.4, 1.2), _gate_seasons([0.3, 0.5, 0.9]), _ACTIONS)
    assert status == "ADOPT"
    assert said.startswith("ADOPT: the arm ships.")
    assert "3/3" in said


def test_an_interval_above_zero_that_loses_a_season_does_not_adopt():
    """The half two of the three gates did not have, and the reason this is now one rule.
    A pooled interval that excludes zero while one season disagrees is the shape
    `hub.models.spread` was corrected for."""
    status, said = experiment.gate(_gate_summary(0.1, 1.2), _gate_seasons([0.3, -0.2, 0.9]), _ACTIONS)
    assert status == "SHOW"
    assert "2/3" in said


def test_an_interval_below_zero_in_every_season_removes():
    status, said = experiment.gate(_gate_summary(-1.2, -0.4), _gate_seasons([-0.3, -0.5, -0.9]),
                                   _ACTIONS)
    assert status == "REMOVE"
    assert said.startswith("REMOVE: delete it.")


def test_an_interval_below_zero_that_wins_a_season_does_not_remove():
    status, _ = experiment.gate(_gate_summary(-1.2, -0.1), _gate_seasons([-0.3, 0.2, -0.9]), _ACTIONS)
    assert status == "SHOW"


def test_an_interval_containing_zero_shows():
    status, said = experiment.gate(_gate_summary(-0.4, 0.9), _gate_seasons([0.3, -0.2, 0.9]), _ACTIONS)
    assert status == "SHOW"
    assert "absence of evidence" in said


def test_an_endpoint_of_exactly_zero_is_not_an_exclusion():
    """The boundary. A bootstrap that lands on zero has not excluded it, and
    ADR-0012's published interval is [-0.00, +0.00]."""
    assert experiment.gate(_gate_summary(0.0, 1.2), _gate_seasons([0.3, 0.5]), _ACTIONS)[0] == "SHOW"
    assert experiment.gate(_gate_summary(-1.2, 0.0), _gate_seasons([-0.3, -0.5]), _ACTIONS)[0] == "SHOW"


def test_nothing_measured_shows_rather_than_adopting_on_an_empty_interval():
    status, said = experiment.gate(_gate_summary(float("nan"), float("nan"), clusters=0),
                                   _gate_seasons([]), _ACTIONS)
    assert status == "SHOW"
    assert "Nothing measured" in said


def test_a_void_condition_preempts_every_branch():
    """A gate whose inputs are broken has no verdict to read. The condition itself is the
    caller's -- the weekly gate voids on a join-failure rate -- and the gate honours it
    rather than deciding what counts as broken."""
    status, said = experiment.gate(_gate_summary(0.4, 1.2), _gate_seasons([0.3, 0.5]), _ACTIONS,
                                   void="VOID: 4.0% of roster-weeks are a join failure.")
    assert status == "VOID"
    assert said.startswith("VOID:")


# --- the published verdicts must still fall out of the unified rule ------------

@pytest.mark.parametrize("what,lo,hi,gains,expected", [
    # ADR-0009: equity vs the market, n=80, losing in all four seasons.
    ("ADR-0009 championship equity", -23.16, -16.20, [-19.0, -21.0, -18.0, -20.0], "REMOVE"),
    # ADR-0012: the lineup optimiser, an interval that is zero to two places both ways.
    ("ADR-0012 lineup optimiser", -0.00, 0.00, [0.0, 0.0, 0.0, 0.0], "SHOW"),
    # The frozen weekly gate: +0.215, CI [-0.242, +0.684], three seasons of four.
    ("weekly gate", -0.242, 0.684, [0.4, 0.3, -0.2, 0.5], "SHOW"),
])
def test_the_recorded_verdicts_reproduce(what, lo, hi, gains, expected):
    """Unifying the rule tightened two of the three gates. If that had flipped a published
    decision it would be a different change entirely, so it is checked rather than hoped."""
    got = experiment.gate(_gate_summary(lo, hi), _gate_seasons(gains), _ACTIONS)[0]
    assert got == expected, f"{what} moved to {got}"

# --- the MDE and the ceiling, and the difference between no data and no slot ---------
#
# U4 of `docs/plans/2026-09-04-001-fix-pin-reprice-correct-board-plan.md` computes a minimum
# detectable effect inside `summarise`; U13 measures a ceiling on each gate's own harness and
# hands it in. Neither exists yet. The fields exist now so that when those units land, no call
# site changes -- which makes "this changed nothing" the whole property under test here.
#
# The first attempt gave both fields a NaN sentinel named `ABSENT` and a `present()` predicate,
# and justified it by citing `.github/scripts/heartbeat.sh`: `jq -r '.ts // 0'` made a missing
# timestamp an age of `now - 0`, the watchdog reported 56.7 years of staleness on every run and
# could never reach the branch that closes an incident -- "one sentinel standing for
# unreachable, unreadable and stale". The fix there was three words for three causes.
#
# It then did the thing the comment forbade. `summarise` already returns NaN for its mean,
# bounds and probability on an empty frame, meaning *the experiment had no rows*; `ABSENT` was
# the same NaN in the same dictionary meaning *this summary predates the field*, and `present`
# answered False to both. Measured 2026-09-05 on `summarise(pl.DataFrame())`:
# `present(s["mean"])` and `present(s["ceiling"])` were both `False` and the two values were
# both `float("nan")` -- nothing a reader was given could separate them.
#
# What replaced it carries the state in the mapping's *shape* rather than in a value: a field
# with nothing to say is not in the summary at all. Three states, three answers, from
# `experiment.reading` -- a value, no data, no slot. These tests hold that line, and the line
# that a *computed* zero is a value: a gate measured to have no headroom and a gate with no
# ceiling measured are different readings, and a reader who cannot tell them apart has the
# watchdog's number back.


def _paired_frame():
    return pl.DataFrame({"season": [2022, 2023, 2024, 2025],
                         "diff": [1.5, -0.5, 2.0, 0.25]})


# The three call sites, spelled as `backtest.main`, `lineup_gate.main` and `weekly_gate.main`
# spell them, with the block each rendered before the two fields existed. Byte-for-byte,
# because the acceptance criterion is byte-identical output for every existing caller and a
# `==` on the whole block is the only way to hold that rather than assert it.
BLOCKS_BEFORE: dict[str, tuple[dict[str, Any], list[str]]] = {
    "backtest": (
        {"arm_a": "optimizer", "arm_b": "market"},
        ["\n  n=80  optimizer - market = -19.13 points per team game",
         "  95% CI [-22.31, -15.75]   P(optimizer better) 0.0%"],
    ),
    "lineup_gate": (
        {"arm_a": "optimiser", "arm_b": "projections", "unit": "points per game"},
        ["\n  n=80  optimiser - projections = -19.13 points per game",
         "  95% CI [-22.31, -15.75]   P(optimiser better) 0.0%"],
    ),
    "weekly_gate": (
        {"arm_a": "weekly", "arm_b": "consensus", "unit": "points per team-week",
         "places": 3, "show_n": False},
        ["\n  weekly - consensus = -19.130 points per team-week",
         "  95% CI [-22.310, -15.750]   P(weekly better) 0.0%"],
    ),
}


@pytest.mark.parametrize("site", sorted(BLOCKS_BEFORE))
def test_an_existing_caller_s_block_is_byte_identical(site):
    """The load-bearing criterion. Not a line, not a blank, not a placeholder."""
    kwargs, before = BLOCKS_BEFORE[site]
    assert experiment.paired_report(_summary(), **kwargs) == before


@pytest.mark.parametrize("site", sorted(BLOCKS_BEFORE))
def test_a_summary_straight_out_of_summarise_renders_the_two_lines_plus_its_mde(site):
    """The goldens above are hand-built dicts, which cannot show whether the real product of
    `summarise` has grown a line. This does -- and since #45 it has grown exactly one.

    `gate-power.md` predicted this before it was built: *"supplying the key changes every
    gate's printed output, and moves the sweep digest."* The two original lines are still
    byte-identical, the new one is the MDE, and no caller here hands in a ceiling."""
    kwargs, before = BLOCKS_BEFORE[site]
    lines = experiment.paired_report(experiment.summarise(_paired_frame(), bootstrap=200,
                                                          seed=1), **kwargs)
    assert len(lines) == 3
    assert "MDE at 80% power" in lines[2]
    assert "ceiling" not in "\n".join(lines)
    # The two that were there are still exactly where and what they were. The numbers differ
    # from the goldens above -- those are a hand-built summary, this is a real frame -- so it
    # is the skeleton that is compared: every character that is not a digit or a sign.
    skeleton = functools.partial(re.sub, r"[-+0-9.]+", "#")
    assert [skeleton(ln) for ln in lines[:2]] == [skeleton(b) for b in before]


# --- the breadth proof ------------------------------------------------------
#
# Three hand-written goldens localise a failure; they cannot show the blocks are unchanged
# across the shapes a real run takes. The sweep below renders all three call sites over 41
# frames x 3 cluster settings x 2 seeds -- 738 blocks -- and hashes them, together with 80
# gate verdicts. Both digests were recorded from this module *as it stood before the sentinel
# changed* (working tree at 293ae3c, 2026-09-05), which is what makes them a proof of "current
# callers see the same bytes" rather than a restatement of whatever the code now does.

_SWEEP_CLUSTERS: tuple[tuple[str, ...] | None, ...] = (None, ("season",),
                                                       ("season", "roster"))
_SWEEP_SEEDS = (1, 7)


def _sweep_frames() -> list[pl.DataFrame]:
    """The empty frame, then forty shaped frames.

    Built from arithmetic rather than an RNG so the digests below cannot move under a numpy
    upgrade -- the bootstrap inside `summarise` is stream-dependent enough on its own.
    """
    schema = {"season": pl.Int64, "roster": pl.Int64, "week": pl.Int64, "diff": pl.Float64}
    frames = [pl.DataFrame(schema=schema)]
    for i in range(40):
        rows, k = [], 0
        for season in range(2022, 2023 + i % 4):
            for roster in range(1 + i % 3):
                for week in range(1, 2 + i % 5):
                    k += 1
                    rows.append({
                        "season": season, "roster": roster, "week": week,
                        "diff": round(math.sin(i * 1.7 + k * 0.31) * (1 + i % 3)
                                      + (i - 20) / 8, 6)})
        frames.append(pl.DataFrame(rows, schema=schema))
    return frames


def _sweep_labelled() -> list[tuple[str, list[str], dict[str, float]]]:
    """Every call site's rendered block, over every frame, cluster setting and seed -- with
    the summary that produced it, so an assertion can ask *why* a block has the lines it has
    rather than inferring it from the label."""
    out = []
    for f_i, frame in enumerate(_sweep_frames()):
        for cluster in _SWEEP_CLUSTERS:
            for seed in _SWEEP_SEEDS:
                s = experiment.summarise(frame, cluster=cluster, bootstrap=200, seed=seed)
                for site in sorted(BLOCKS_BEFORE):
                    kwargs, _ = BLOCKS_BEFORE[site]
                    out.append((f"{f_i}|{cluster}|{seed}|{site}",
                                experiment.paired_report(s, **kwargs), s))
    return out


def _sweep_blocks() -> list[tuple[str, list[str]]]:
    return [(label, lines) for label, lines, _ in _sweep_labelled()]


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n--\n".join(parts).encode()).hexdigest()[:16]


# **Moved once, by #45, and this is the record of it.** The digest was `bf5b1af281ea0f0c` from
# 2026-09-05 until `summarise` began computing an MDE. `docs/gate-power.md` predicted the move
# before the work started -- *"supplying the key changes every gate's printed output, and moves
# the sweep digest that `tests/unit/test_experiment.py` pins. That is the digest doing its
# job."* -- and `test_every_block_grew_exactly_the_mde_line` below is what says the move is
# only that. A re-recorded digest with no readable assertion beside it would be a rubber stamp.
BLOCK_SWEEP_DIGEST = "5c1be1a2ba3ba17f"

# Unmoved, and it has to be: `_gate_sweep({})` hands in neither an `mde` nor a `ceiling`, so
# the NOT-RUNNABLE branch cannot fire and all eighty verdicts are the ones ADR-0019 gave.
GATE_SWEEP_DIGEST = "7c0084a7ec69757d"


def test_the_whole_rendered_sweep_is_byte_identical():
    """738 blocks, one hash. If a field grew a line, changed a number or reordered, this moves
    and the readable assertions below say which of those it was."""
    blocks = _sweep_blocks()
    assert len(blocks) == 738
    assert _digest([f"{label}\n" + "\n".join(lines) for label, lines in blocks]) \
        == BLOCK_SWEEP_DIGEST


def test_every_block_grew_exactly_the_mde_line_and_nothing_else():
    """The readable half of the digest above, and the whole reason re-recording it is honest.

    A moved digest says *something* changed. This says what: every block that can carry an
    MDE grew that one line, in third place, below the interval; every block that cannot --
    an empty frame, or one whose cluster count is 1, which has no degrees of freedom for a t
    -- is byte-identical to the two lines it was; and no block anywhere grew a ceiling,
    because no caller in this sweep hands one in.
    """
    grew, unchanged = 0, 0
    for label, lines, s in _sweep_labelled():
        assert "ceiling" not in "\n".join(lines), label
        if experiment.reading(s, "mde") is experiment.Field.VALUE:
            assert len(lines) == 3, label
            assert lines[2].startswith("  MDE at 80% power "), label
            assert "95% CI" in lines[1], label
            grew += 1
        else:
            assert len(lines) == 2, label
            assert "MDE" not in "\n".join(lines), label
            unchanged += 1
    # Both branches are actually exercised: a test where every block took one arm would prove
    # only that arm. 18 empty-frame blocks plus the single-cluster ones stay at two lines.
    assert grew and unchanged, (grew, unchanged)
    assert grew + unchanged == 738


def test_only_the_empty_frame_renders_a_nan_across_the_sweep():
    """Where NaN reaches a reader it is because the experiment scored nothing -- 18 blocks,
    the empty frame at each of 3 cluster settings x 2 seeds x 3 call sites. That is also
    exactly what those callers printed before, so it stays."""
    nan_blocks = [label for label, lines in _sweep_blocks() if "nan" in "\n".join(lines)]
    assert len(nan_blocks) == 18
    assert {label.split("|")[0] for label in nan_blocks} == {"0"}


_GATE_INTERVALS = ((-1.2, -0.4), (-1.2, -0.1), (-0.4, 0.9), (0.0, 1.2), (-1.2, 0.0),
                   (0.1, 1.2), (0.4, 1.2), (-23.16, -16.20), (-0.242, 0.684), (-0.0, 0.0))
_GATE_GAINS = ((0.3, 0.5, 0.9), (0.3, -0.2, 0.9), (-0.3, -0.5, -0.9), (-0.3, 0.2, -0.9),
               (0.0, 0.0, 0.0), (1.0,), (-1.0,), (0.4, 0.3, -0.2, 0.5))


def _gate_sweep(extra: dict[str, float]) -> list[str]:
    """Every verdict on the interval x seasons grid, with `extra` merged into each summary."""
    out = []
    for lo, hi in _GATE_INTERVALS:
        for gains in _GATE_GAINS:
            status, said = experiment.gate(_gate_summary(lo, hi) | extra,
                                           _gate_seasons(gains), _ACTIONS)
            out.append(f"{lo}|{hi}|{gains}|{status}|{said}")
    return out


def test_the_eighty_gate_verdicts_are_unmoved():
    verdicts = _gate_sweep({})
    assert len(verdicts) == 80
    assert _digest(verdicts) == GATE_SWEEP_DIGEST


@pytest.mark.parametrize("extra", [
    {},
    {"ceiling": 1.2},
    {"ceiling": 0.0},
    {"mde": 0.44},
    {"mde": 0.44, "ceiling": 1.2},
    {"mde": float("nan"), "ceiling": float("nan")},
    {"mde": float("nan"), "ceiling": 1.2},
    {"mde": 99.0, "ceiling": float("nan")},
])
def test_a_runnable_gate_reaches_the_verdict_adr_0019_gave_it(extra):
    """The not-runnable branch fires on one condition and leaves every other case alone.

    Each row here is a state that must NOT trip it: no ceiling measured, no MDE computed,
    either of them present-but-no-data, and an MDE that sits comfortably below its ceiling.
    A gate that has not shown it cannot run has shown nothing, and `Field.NO_SLOT` and
    `Field.NO_DATA` are that third state rather than a licence to guess.
    """
    assert _gate_sweep(extra) == _gate_sweep({})


def test_an_mde_above_the_ceiling_is_not_runnable():
    """Criterion 8. The gate cannot separate a real effect from a perfect one, so it records
    that it cannot run instead of publishing a null."""
    status, said = experiment.gate(_gate_summary(-0.4, 0.9) | {"mde": 2.0, "ceiling": 1.2},
                                   _gate_seasons([0.3, -0.5, 0.9]), _ACTIONS)
    assert status == "NOT-RUNNABLE"
    assert "not planned" in said and "2.000" in said and "1.200" in said


def test_not_runnable_preempts_every_branch_but_void():
    """Criterion 8's *ordering*, which is the whole point of it. A gate whose MDE exceeds its
    ceiling would otherwise publish one of ADOPT, REMOVE or SHOW -- and SHOW, the null, is
    exactly the verdict `docs/gate-power.md` exists to stop an underpowered gate printing."""
    underpowered = {"mde": 99.0, "ceiling": 1.2}
    for lo, hi in _GATE_INTERVALS:
        for gains in _GATE_GAINS:
            summary = _gate_summary(lo, hi) | underpowered
            status, _ = experiment.gate(summary, _gate_seasons(gains), _ACTIONS)
            assert status == "NOT-RUNNABLE", (lo, hi, gains)
    # Including the branch that would otherwise ADOPT, and the empty one that would SHOW.
    assert experiment.gate(_gate_summary(0.4, 1.2) | underpowered,
                           _gate_seasons([0.3, 0.5, 0.9]), _ACTIONS)[0] == "NOT-RUNNABLE"
    assert experiment.gate(_gate_summary(0.4, 1.2, clusters=0) | underpowered,
                           _gate_seasons([0.3]), _ACTIONS)[0] == "NOT-RUNNABLE"


def test_void_still_preempts_not_runnable():
    """The one branch above it. A void gate's inputs are broken, which makes its MDE and its
    ceiling untrustworthy too -- there is nothing to compare."""
    status, said = experiment.gate(_gate_summary(-0.4, 0.9) | {"mde": 99.0, "ceiling": 1.2},
                                   _gate_seasons([0.3]), _ACTIONS, void="VOID: join failure.")
    assert status == "VOID" and said == "VOID: join failure."


def test_a_ceiling_of_zero_makes_any_positive_mde_not_runnable():
    """A measured zero is a value, not an absence -- a perfect arm gaining nothing over the
    incumbent is the strongest finding a ceiling can carry, and no gate can resolve an effect
    inside it. The comparison is signed for this reason rather than absolute."""
    status, _ = experiment.gate(_gate_summary(-0.4, 0.9) | {"mde": 0.01, "ceiling": 0.0},
                                _gate_seasons([0.3]), _ACTIONS)
    assert status == "NOT-RUNNABLE"


def test_an_mde_exactly_at_the_ceiling_still_runs():
    """The boundary, stated. `exceeds` is strict: a gate that can just resolve its ceiling has
    not been shown unable to run."""
    status, _ = experiment.gate(_gate_summary(-0.4, 0.9) | {"mde": 1.2, "ceiling": 1.2},
                                _gate_seasons([0.3, -0.5, 0.9]), _ACTIONS)
    assert status == "SHOW"


def test_a_runnable_gate_still_does_not_adopt_at_two_of_three_seasons():
    """Criterion 7, and the worked example `docs/method.md` records as correctly failed: an
    interval excluding zero is not enough on its own, and the every-season half is what stops
    one lucky season carrying a verdict. Asserted here *with* an MDE and a ceiling present, so
    the new branch cannot be what produced the answer."""
    summary = _gate_summary(0.1, 1.2) | {"mde": 0.44, "ceiling": 1.2}
    status, said = experiment.gate(summary, _gate_seasons([0.4, 0.3, -0.2]), _ACTIONS)
    assert status == "SHOW"
    assert "Won 2/3 seasons" in said
    assert "the sign is not consistent across seasons" in said


# --- the restated figures, so the documents cannot drift from the code -------


def test_the_restated_weekly_gate_figures_reproduce():
    """`docs/weekly-blend-gate.md`'s 2026-09-07 season-clustered restatement, recomputed.

    That gate builds its paired frame from the network and persists nothing, so the numbers on
    the page were derived rather than re-run -- a cluster bootstrap resamples the *cluster
    means*, and under `SEASON_CLUSTER` those are exactly the four per-season gains the page
    already published, over four balanced seasons. This is that derivation, pinned, so the
    document and the code cannot drift apart quietly.
    """
    published = {2022: -0.452, 2023: -1.490, 2024: -0.868, 2025: -1.204}
    frame = pl.DataFrame({"season": list(published), "diff": list(published.values())})
    s = experiment.summarise(frame, cluster=experiment.SEASON_CLUSTER, seed=0)

    assert s["clusters"] == 4
    assert s["mean"] == pytest.approx(-1.004, abs=0.001)
    assert (s["lo"], s["hi"]) == (pytest.approx(-1.347, abs=0.001),
                                  pytest.approx(-0.640, abs=0.001))
    assert s["se"] == pytest.approx(0.191, abs=0.001)
    assert s["mde"] == pytest.approx(0.768, abs=0.001)

    # The verdict does not move: every season is negative, so no resample reaches zero.
    seasons = pl.DataFrame({"season": list(published), "n": [500] * 4,
                            "gain": list(published.values())})
    assert experiment.gate(s, seasons, _ACTIONS)[0] == "REMOVE"


def test_the_weekly_percentile_interval_narrows_and_the_t_interval_widens():
    """The finding the restatement turns on, and the one criterion 5 exists to surface.

    Against the published roster-clustered [-1.391, -0.621] the season-clustered *percentile*
    interval is narrower -- a percentile bootstrap over four units cannot express a tail it
    never drew -- while the t interval on the same four units is much wider. Both facts on the
    same run, which is why the small-sample block prints both and why a narrowing has to
    announce itself rather than read as precision."""
    published = {2022: -0.452, 2023: -1.490, 2024: -0.868, 2025: -1.204}
    frame = pl.DataFrame({"season": list(published), "diff": list(published.values())})
    s = experiment.summarise(frame, cluster=experiment.SEASON_CLUSTER, seed=0)

    was = -0.621 - -1.391
    percentile = s["hi"] - s["lo"]
    t_width = 2 * experiment.t_quantile(0.975, 3) * s["se"]
    assert percentile < was, "the percentile interval no longer narrows; restate the doc"
    assert t_width > was
    assert percentile / was == pytest.approx(0.92, abs=0.01)

    said = experiment.narrowing(percentile, was)
    assert said.requires_review and "REQUIRES REVIEW" in "\n".join(said.lines)


def test_the_restated_draft_gate_mdes_reproduce():
    """`docs/gate-power.md`'s restated table: the same SEs, the t quantile instead of the
    normal, and the 1.44x that only matters at four clusters."""
    assert experiment.minimum_detectable_effect(1.79, 80) == pytest.approx(5.07, abs=0.01)
    assert experiment.minimum_detectable_effect(3.67, 4) == pytest.approx(14.77, abs=0.01)
    # Stage 1's margin against the gate's own reported effect, before and after.
    assert 19.66 / experiment.minimum_detectable_effect(3.67, 4) == pytest.approx(1.33,
                                                                                 abs=0.01)


# --- no data is not no slot -------------------------------------------------


def test_no_data_and_no_slot_are_different_answers():
    """The defect, in one assertion. On an empty frame the mean says *no rows were scored* and
    the ceiling says *nobody measured one*; those are different facts with different responses
    and the reader is now given words that separate them."""
    empty = experiment.summarise(pl.DataFrame())
    assert experiment.reading(empty, "mean") is experiment.Field.NO_DATA
    assert experiment.reading(empty, "ceiling") is experiment.Field.NO_SLOT
    # Since #45 `summarise` computes an MDE, so on an empty frame the answer moves from "this
    # producer does not compute the field" to "it does, and this run scored nothing into it" --
    # which puts all three states on one summary, where before it took two.
    assert experiment.reading(empty, "mde") is experiment.Field.NO_DATA
    assert experiment.reading(experiment.summarise(_paired_frame(), bootstrap=200, seed=1),
                              "mde") is experiment.Field.VALUE


def test_a_ceiling_measured_on_a_harness_survives_an_empty_frame():
    """The case that has to read correctly and could not before: the harness played its
    perfect-foresight arm and got a number, and this particular frame scored nothing. Two
    facts, both reportable, previously one NaN."""
    s = experiment.summarise(pl.DataFrame(), ceiling=0.0)
    assert experiment.reading(s, "ceiling") is experiment.Field.VALUE
    assert s["ceiling"] == 0.0
    assert experiment.reading(s, "mean") is experiment.Field.NO_DATA


def test_nan_means_no_data_and_nothing_else():
    """The invariant that keeps the two apart, stated both ways. A summary over rows carries
    no NaN anywhere; the empty summary carries NaN for exactly the six computed fields --
    four until #45 added `se` and `mde`, which are computed here and so have a slot to be
    empty in."""
    full = experiment.summarise(_paired_frame(), bootstrap=200, seed=1, ceiling=1.2)
    assert [k for k, v in full.items() if math.isnan(v)] == []
    empty = experiment.summarise(pl.DataFrame())
    assert sorted(k for k, v in empty.items() if math.isnan(v)) == [
        "hi", "lo", "mde", "mean", "p_better", "se"]


def test_the_empty_frame_still_reports_what_it_reported_before():
    """An experiment with no rows is not silently promoted to a missing field: its four
    computed numbers are still NaN and its two counts are still zero."""
    s = experiment.summarise(pl.DataFrame())
    assert (s["n"], s["clusters"]) == (0, 0)
    for field in ("mean", "lo", "hi", "p_better"):
        assert math.isnan(s[field]), field
        assert experiment.reading(s, field) is experiment.Field.NO_DATA


def test_a_summary_claims_the_mde_and_not_the_ceiling():
    """The key's presence is the claim that this producer computes the field, and #45 moves
    exactly one of the two: `summarise` now computes an MDE from its own bootstrap, and still
    cannot compute a ceiling -- measuring one means playing an extra arm and only the harness
    knows what its arms are."""
    s = experiment.summarise(_paired_frame(), bootstrap=200, seed=1)
    assert "mde" in s and "se" in s and "ceiling" not in s
    assert experiment.reading(s, "mde") is experiment.Field.VALUE
    assert experiment.reading(s, "ceiling") is experiment.Field.NO_SLOT


def test_the_optional_fields_do_not_move_the_numbers_that_were_already_there():
    """Adding a field must not perturb the bootstrap. Same seed, and these six are the values
    recorded from the run before the change."""
    s = experiment.summarise(_paired_frame(), bootstrap=500, seed=3)
    assert (s["n"], s["clusters"]) == (4.0, 4.0)
    assert (s["mean"], s["lo"], s["hi"], s["p_better"]) == (0.8125, -0.3125, 1.875, 0.944)


def test_a_ceiling_is_carried_from_the_caller():
    """U13 measures it on each gate's own harness against that gate's own incumbent, so it
    arrives from outside rather than being computed here."""
    s = experiment.summarise(_paired_frame(), bootstrap=200, seed=1, ceiling=1.20)
    assert s["ceiling"] == 1.20
    assert experiment.reading(s, "ceiling") is experiment.Field.VALUE


def test_the_declared_mapping_type_is_not_widened():
    """The constraint that shaped this and still binds: `summarise` is declared
    `dict[str, float]` and `backtest.verdict` and `lineup_gate.verdict` declare it on the way
    in. Widening the values to admit `None` was re-measured on 2026-09-05 at 30 pyrefly errors
    across six files, four of them in the three harnesses a prefactor must not touch. Absence
    is carried by the mapping's shape instead, which costs those callers nothing."""
    import inspect
    sig = inspect.signature(experiment.summarise)
    assert sig.return_annotation == "dict[str, float]"
    got = experiment.summarise(_paired_frame(), bootstrap=200, seed=1, ceiling=0.5)
    assert [k for k, v in got.items() if not isinstance(v, float | int)] == []


def test_a_measured_zero_is_a_value_not_an_absence():
    """The watchdog's `// 0` again, and the line the first attempt got right. A ceiling of zero
    says a perfect-foresight arm gains nothing over the incumbent, which is the strongest
    finding a ceiling can carry."""
    assert experiment.reading({"ceiling": 0.0}, "ceiling") is experiment.Field.VALUE
    assert experiment.reading({"ceiling": -0.0}, "ceiling") is experiment.Field.VALUE
    assert experiment.reading({}, "ceiling") is experiment.Field.NO_SLOT
    assert experiment.reading({"ceiling": float("nan")}, "ceiling") is experiment.Field.NO_DATA


# --- what the block renders -------------------------------------------------


def test_the_ceiling_line_renders_when_it_has_a_value():
    lines = experiment.paired_report(_summary() | {"ceiling": 1.2},
                                     arm_a="optimizer", arm_b="market")
    joined = "\n".join(lines)
    assert "ceiling (perfect foresight) +1.20 points per team game" in joined
    assert "MDE" not in joined, "the one with no slot must not appear at all"


def test_the_ceiling_line_names_the_arm_the_caller_gives_it():
    """Two of the three gates bound perfect foresight and the lineup gate bounds a perfect
    *spread*. A block that called the second one foresight would be the confusion
    `lineup_gate.foresight_lineup_points` exists as a separate function to prevent."""
    lines = experiment.paired_report(_summary() | {"ceiling": 1.2}, arm_a="a", arm_b="b",
                                     ceiling_arm="a perfect spread, not foresight")
    assert "ceiling (a perfect spread, not foresight) +1.20" in "\n".join(lines)
    assert "perfect foresight)" not in "\n".join(lines)


def test_a_caller_that_renders_its_own_ceiling_line_is_not_given_a_second_one():
    """`None` says the caller prints its own. The lineup gate passes it, because it has a
    `ceiling_report` that names the arm and warns when the ceiling fails to bound -- and
    without this the number appeared twice, once under the wrong name. Found by rendering the
    block rather than by any assertion, which is why there is now an assertion."""
    s = _summary() | {"ceiling": 1.2}
    assert experiment.paired_report(s, arm_a="a", arm_b="b", ceiling_arm=None) \
        == experiment.paired_report(_summary(), arm_a="a", arm_b="b")


def test_the_mde_line_renders_when_it_has_a_value():
    lines = experiment.paired_report(_summary() | {"mde": 0.44},
                                     arm_a="weekly", arm_b="consensus",
                                     unit="points per team-week", places=3, show_n=False)
    joined = "\n".join(lines)
    assert "MDE at 80% power +0.440 points per team-week" in joined
    assert "ceiling" not in joined


def test_a_measured_zero_still_renders():
    """The distinction the shape exists to keep. A gate whose ceiling is zero has no headroom
    to report and has to say so; omitting the line would read as never measured."""
    joined = "\n".join(experiment.paired_report(_summary() | {"ceiling": 0.0, "mde": 0.0},
                                                arm_a="a", arm_b="b"))
    assert "ceiling (perfect foresight) +0.00" in joined
    assert "MDE at 80% power +0.00" in joined


def test_a_field_with_no_data_renders_no_line_rather_than_a_line_reading_nan():
    """A slot that exists and has nothing in it yet -- U4's MDE over an empty frame -- is a
    third state, and the block's answer to it is silence. `nan` printed against a unit is the
    watchdog's 56.7 years."""
    lines = experiment.paired_report(_summary() | {"mde": float("nan"),
                                                   "ceiling": float("nan")},
                                     arm_a="a", arm_b="b")
    assert lines == experiment.paired_report(_summary(), arm_a="a", arm_b="b")
    assert "nan" not in "\n".join(lines)


def test_both_render_below_the_interval_in_a_stated_order():
    """Effect, then interval, then what the run could have resolved, then what there was to
    resolve. Each reads against the line above it."""
    lines = experiment.paired_report(_summary() | {"mde": 0.44, "ceiling": 1.2},
                                     arm_a="a", arm_b="b")
    assert len(lines) == 4
    assert "95% CI" in lines[1] and "MDE" in lines[2] and "ceiling" in lines[3]
