"""The walk-forward paired experiment, which this repo runs twice.

`backtest` and `lineup_gate` both build a board as it stood before each season opened, load
what happened, play two arms, and report a paired interval. The season loop was byte-identical
in both and the report block differed by one word. None of it was reachable from a test,
because both copies lived inside a `main()` that needs a network.

All offline.
"""
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


# --- the season loop --------------------------------------------------------

def test_inputs_are_gathered_per_season():
    boards, realised = experiment.walk_forward_inputs(
        [2023, 2024],
        lambda yr: (_board(), None),
        load_stats=_stats)
    assert set(boards) == {2023, 2024} and set(realised) == {2023, 2024}
    assert boards[2023].height == 8 and realised[2024].height > 0


def test_the_board_is_built_as_of_that_season_s_opening():
    """A strategy scored against rankings published after the season is hindsight wearing a
    backtest's clothes. The rule lives in `hub.draft.board`, beside the `build` that enforces
    it -- it started here, which put draft-domain knowledge under `models/` and inverted the
    tree's one consistent direction."""
    import inspect

    from hub.draft import board
    src = inspect.getsource(board.board_as_of)
    assert 'as_of=f"{season}-09-01"' in src
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


def test_the_progress_hook_is_a_hook_not_a_print(capsys):
    """A caller under a line cap must be able to stay quiet, and this module must not own
    stdout -- the same reason `hub.draft.report` returns lines."""
    seen = []
    experiment.walk_forward_inputs([2024], lambda yr: (_board(), None),
                                   load_stats=_stats, on_season=seen.append)
    assert seen == [2024]
    assert capsys.readouterr().out == ""


def test_no_hook_is_silent(capsys):
    experiment.walk_forward_inputs([2024], lambda yr: (_board(), None),
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
