"""The walk-forward paired experiment, which this repo runs twice.

`backtest` and `lineup_gate` both build a board as it stood before each season opened, load
what happened, play two arms, and report a paired interval. The season loop was byte-identical
in both and the report block differed by one word. None of it was reachable from a test,
because both copies lived inside a `main()` that needs a network.

All offline.
"""
import polars as pl

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
