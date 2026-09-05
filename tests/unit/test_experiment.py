"""The walk-forward paired experiment, which this repo runs twice.

`backtest` and `lineup_gate` both build a board as it stood before each season opened, load
what happened, play two arms, and report a paired interval. The season loop was byte-identical
in both and the report block differed by one word. None of it was reachable from a test,
because both copies lived inside a `main()` that needs a network.

All offline.
"""
import hashlib
import math
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
def test_a_summary_straight_out_of_summarise_renders_the_same_two_lines(site):
    """The goldens above are hand-built dicts, which cannot show whether the real product of
    `summarise` has grown a line. This does."""
    kwargs, _ = BLOCKS_BEFORE[site]
    lines = experiment.paired_report(experiment.summarise(_paired_frame(), bootstrap=200,
                                                          seed=1), **kwargs)
    assert len(lines) == 2
    assert "MDE" not in "\n".join(lines) and "ceiling" not in "\n".join(lines)


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


def _sweep_blocks() -> list[tuple[str, list[str]]]:
    """Every call site's rendered block, over every frame, cluster setting and seed."""
    out = []
    for f_i, frame in enumerate(_sweep_frames()):
        for cluster in _SWEEP_CLUSTERS:
            for seed in _SWEEP_SEEDS:
                s = experiment.summarise(frame, cluster=cluster, bootstrap=200, seed=seed)
                for site in sorted(BLOCKS_BEFORE):
                    kwargs, _ = BLOCKS_BEFORE[site]
                    out.append((f"{f_i}|{cluster}|{seed}|{site}",
                                experiment.paired_report(s, **kwargs)))
    return out


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n--\n".join(parts).encode()).hexdigest()[:16]


BLOCK_SWEEP_DIGEST = "bf5b1af281ea0f0c"
GATE_SWEEP_DIGEST = "7c0084a7ec69757d"


def test_the_whole_rendered_sweep_is_byte_identical():
    """738 blocks, one hash. If a field grew a line, changed a number or reordered, this moves
    and the readable assertions below say which of those it was."""
    blocks = _sweep_blocks()
    assert len(blocks) == 738
    assert _digest([f"{label}\n" + "\n".join(lines) for label, lines in blocks]) \
        == BLOCK_SWEEP_DIGEST


def test_no_block_in_the_sweep_grew_a_line():
    """The readable half of the digest above: today nothing computes an MDE and no caller
    hands in a ceiling, so every one of the 738 blocks is exactly the two lines it was."""
    for label, lines in _sweep_blocks():
        assert len(lines) == 2, label
        assert "MDE" not in "\n".join(lines) and "ceiling" not in "\n".join(lines)


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
    {"mde": 0.44, "ceiling": 1.2},
    {"mde": float("nan"), "ceiling": float("nan")},
])
def test_no_state_of_either_field_moves_a_gate_verdict(extra):
    """`gate` is ADR-0019's two halves and nothing else today. U4 adds a not-runnable branch
    ahead of every branch but VOID; until it does, a summary carrying these fields in any
    state must reach exactly the verdict a summary without them reaches."""
    assert _gate_sweep(extra) == _gate_sweep({})


# --- no data is not no slot -------------------------------------------------


def test_no_data_and_no_slot_are_different_answers():
    """The defect, in one assertion. On an empty frame the mean says *no rows were scored* and
    the ceiling says *nobody measured one*; those are different facts with different responses
    and the reader is now given words that separate them."""
    empty = experiment.summarise(pl.DataFrame())
    assert experiment.reading(empty, "mean") is experiment.Field.NO_DATA
    assert experiment.reading(empty, "ceiling") is experiment.Field.NO_SLOT
    assert experiment.reading(empty, "mde") is experiment.Field.NO_SLOT


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
    no NaN anywhere; the empty summary carries NaN for exactly the four computed fields."""
    full = experiment.summarise(_paired_frame(), bootstrap=200, seed=1, ceiling=1.2)
    assert [k for k, v in full.items() if math.isnan(v)] == []
    empty = experiment.summarise(pl.DataFrame())
    assert sorted(k for k, v in empty.items() if math.isnan(v)) == [
        "hi", "lo", "mean", "p_better"]


def test_the_empty_frame_still_reports_what_it_reported_before():
    """An experiment with no rows is not silently promoted to a missing field: its four
    computed numbers are still NaN and its two counts are still zero."""
    s = experiment.summarise(pl.DataFrame())
    assert (s["n"], s["clusters"]) == (0, 0)
    for field in ("mean", "lo", "hi", "p_better"):
        assert math.isnan(s[field]), field
        assert experiment.reading(s, field) is experiment.Field.NO_DATA


def test_a_summary_today_has_no_slot_for_either_field():
    """Nothing computes an MDE and no caller hands in a ceiling, so `summarise` claims
    neither. The key's presence is the claim that this producer computes the field."""
    s = experiment.summarise(_paired_frame(), bootstrap=200, seed=1)
    assert "mde" not in s and "ceiling" not in s
    assert experiment.reading(s, "mde") is experiment.Field.NO_SLOT
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
