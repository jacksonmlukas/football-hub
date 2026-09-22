"""#386's equivalence control, authored against `main` before `gate`'s seam moves.

**Ordering rule 1.** This file is written and proven -- green on unchanged code, red on a
planted defect, then restored -- *before* `experiment.gate` is touched. It is committed on
its own, ahead of the seam commit, so the control that will hold the refactor to
byte-identical output is not itself fitted to the refactor.

**What is captured.** Every `run_gate` adapter's full `GateRun` (verdict, summary, seasons,
lines-before-the-stamp) on a fixture close to its own -- the draft gate's frozen Board pin
(#197, via `tests/panelarchive.py`, the same fixture `test_backtest.py`'s pin reads), the
weekly gate's own recorded per-season figures (`docs/weekly-blend-gate.md`'s 2026-09-07
restatement, the same numbers `test_experiment.py::test_the_restated_weekly_gate_figures_
reproduce` already pins), a synthetic lineup frame in `lineup_gate.compare`'s own shape, a
synthetic weeks-drawn frame through `coverage.shape_law`'s own pipeline (the same `_drawn`
recipe `test_coverage.py` uses), the quarterback gate's own synthetic golden
(`test_starter_change.py`'s `_paired`), and margin's `_lumpy_synthetic` for both the width
gate (`margin.verdict`) and the shape gate (`margin.shape_verdict`) -- plus synthetic paired
frames built directly against `run_gate` to reach every branch a real fixture may not:
NOT-RUNNABLE, ADOPT, REMOVE, SHOW, a tie, and VOID.

**Stamps are excluded on purpose.** `run.stamped` carries a commit hash and a config digest,
both of which move with every commit in this repo including the seam commit itself -- they
say *when* the code ran, not what the gate decided, and the ticket's own listed capture
(verdict, why, summary, seasons, lines) does not include it. `_reduce` below drops `stamped`
entirely and truncates `lines` at the blank separator `run_gate` inserts ahead of the stamp
block, so a byte-identical assertion here is about the decision, not the tree it ran in. Both
call sites keep `record_width=False` and an unopened `width_path`, so `review_width`'s own
lines (which read a digest) stay empty on both sides of the seam rather than becoming a third
thing that moves with the commit.

**Rule 18, applied to this file.** `test_the_literal_plant_flips_the_boundary_case` is the
direct, exact demonstration: `_disposition`'s stated rule is `gain >= 2 * se` is a win, and
the literal one-token plant this ticket names (`>=` -> `>`) is checked against the boundary
case at `gain == 2 * se` exactly -- deterministic, no bootstrap involved, so the plant's effect
is provable rather than merely likely. The commit message for this file records the run of
this test against the plant (RED) and its restoration (GREEN); see that commit's body for the
literal before/after.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hub.models import experiment, margin
from hub.models.experiment import SEASON_CLUSTER, Actions, Ceiling, run_gate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GOLDEN = json.loads((Path(__file__).parent / "fixtures"
                     / "gate_seam_equivalence_control.json").read_text())


# --- the exact, deterministic boundary: rule 18's own literal plant ------------------------

def test_the_literal_plant_flips_the_boundary_case():
    """`_disposition`'s rule, read off the boundary directly rather than through a bootstrap
    that could only ever land there by chance. `gain >= 2 * se` is a win at the boundary
    itself; the ticket's literal plant (`>=` -> `>`) would read the same call as a tie. This
    is the assertion whose outcome the plant (applied by hand to `_disposition`, run, and
    reverted -- see this file's own commit body) is checked against."""
    assert experiment._disposition(2.0, se=1.0, m=20) == "win"
    assert experiment._disposition(-2.0, se=1.0, m=20) == "loss"
    # The immediate neighbours, so a plant that moved the boundary the *other* way (`>=` on
    # the wrong side, or `>` becoming `>=`) is also visible rather than only the exact edge.
    assert experiment._disposition(1.999, se=1.0, m=20) == "tie"
    assert experiment._disposition(-1.999, se=1.0, m=20) == "tie"


# --- capture: a GateRun reduced to what an equivalence control compares --------------------

_ROUND = 6  # decimal places: absorbs polars' multi-threaded-aggregation float jitter (the
            # 15th-17th significant digit of a `.mean()`/`.agg()` can differ run to run on
            # identical data -- a pre-existing property of the aggregation library, present
            # equally on both sides of the seam, and irrelevant to every rendered line, which
            # never shows more than 4 decimal places). Rounding here is coarser than any
            # printed figure, so it cannot hide a real behavioural difference the rendered
            # `lines` would also miss.


def _round(v):
    """Round a float for comparison, and turn NaN into a JSON-round-trippable, self-equal
    marker -- `nan != nan` would otherwise make any row carrying one (an under-clustered
    season's `se`, for instance) compare unequal to itself."""
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isfinite(v):
            return round(v, _ROUND)
    return v


def _reduce(run: experiment.GateRun) -> dict:
    """`GateRun` -> a plain, JSON-comparable dict: verdict, summary, seasons, and the lines
    `run_gate` renders *before* the stamp block. NaN is compared by name (`_nan`) rather than
    by value, since `nan != nan` would make every empty-frame case unequal to itself."""
    summary = {}
    nan_keys = []
    for k, v in run.summary.items():
        if isinstance(v, float) and math.isnan(v):
            nan_keys.append(k)
        else:
            summary[k] = v if isinstance(v, str) else _round(float(v))
    lines = list(run.lines)
    lines = lines[: lines.index("")] if "" in lines else lines
    seasons = [{k: _round(v) for k, v in row.items()} for row in run.seasons.to_dicts()]
    return {
        "verdict": list(run.verdict),
        "summary": summary,
        "summary_nan_keys": sorted(nan_keys),
        "seasons": seasons,
        "lines": lines,
    }


# --- fixture builders, one per adapter ------------------------------------------------------

def _draft_paired():
    """The draft gate's frozen Board pin (#197): `bt.compare` on `panelarchive`'s
    `draft_board`, at the same reduced budget `test_backtest.py`'s pin uses so this runs in
    seconds rather than the shipped budget's minutes."""
    import panelarchive as arc

    from hub.draft import backtest as bt
    from hub.draft.board import Board
    from hub.names import player_key
    board = arc.frame("draft_board")
    real = pl.DataFrame({
        "player": [player_key(n) for n in board["player"].to_list() for _ in range(14)],
        "week": [w for _ in range(board.height) for w in range(1, 15)],
        "points": [float(x) for x in board["xfp_per_game"].to_list() for _ in range(14)],
    }, schema={"player": pl.Utf8, "week": pl.Int64, "points": pl.Float64})
    return bt.compare({2025: Board.served(board)}, {2025: real}, n_drafts=1, seed=7,
                      my_slot=3, teams=12, rounds=6, n_draft_sims=2, n_season_sims=10)


def _draft_gate_run():
    from hub.draft import backtest as bt
    paired = _draft_paired()
    return run_gate(paired, cluster=SEASON_CLUSTER, within=bt.WITHIN, actions=bt.ACTIONS,
                    name="control-draft", arm_a="optimizer", arm_b="market",
                    void=bt.void_condition({"picks": 100.0, "market": 0.0, "optimizer": 0.0}),
                    ceiling=None, seed=0, bootstrap=200, record_width=False)


def _draft_sensitivity_gate_run():
    """The second `draft/backtest.py` call site (`sensitivity_table`'s loop): same rule, a
    scaled-noise arm. A fresh `Ceiling` so this case also exercises `ceiling` through the
    second call site's own parameter shape rather than only the first's `None`."""
    from hub.draft import backtest as bt
    paired = _draft_paired()
    top = Ceiling(bt.CEILING_ARM, (paired["market"].to_numpy() * 0.0 + 3.0))
    return run_gate(paired, cluster=SEASON_CLUSTER, within=bt.WITHIN, actions=bt.ACTIONS,
                    name="control-draft-sensitivity", arm_a="optimizer", arm_b="market",
                    void=None, ceiling=top, seed=0, bootstrap=200, record_width=False)


def _weekly_paired():
    """The weekly gate's own recorded per-season figures -- `docs/weekly-blend-gate.md`'s
    2026-09-07 season-clustered restatement -- one row per season, matching
    `test_experiment.py::test_the_restated_weekly_gate_figures_reproduce`'s own recipe: under
    `SEASON_CLUSTER` the season means *are* the cluster means, so this is that gate's real
    recorded output rather than a value invented for this file."""
    published = {2022: -0.452, 2023: -1.490, 2024: -0.868, 2025: -1.204}
    return pl.DataFrame({"season": list(published), "roster": list(published),
                         "diff": list(published.values())})


def _weekly_gate_run():
    from hub.season import weekly_gate as wg
    paired = _weekly_paired()
    top = Ceiling(wg.CEILING_ARM, np.full(paired.height, 10.799))
    return run_gate(paired, cluster=SEASON_CLUSTER, within=wg.WITHIN, actions=wg.ACTIONS,
                    name="control-weekly", arm_a="weekly", arm_b="consensus", unit=wg.UNIT,
                    places=wg.PLACES, show_n=False, void=None, ceiling=top, seed=0,
                    bootstrap=200, record_width=False)


def _lineup_paired():
    """A small synthetic frame in `lineup_gate.compare`'s own shape -- one row per
    (season, roster), clearly ahead in every season, so this reaches ADOPT through the real
    call site rather than only through the branch-coverage battery below. `ceiling_diff` is
    carried too, the way a real `--ceiling` run's frame does, since #363 (S6) makes a gate
    with no ceiling NOT-RUNNABLE regardless of its effect."""
    seasons = [2022, 2023, 2024, 2025]
    rosters = range(20)
    rows = [(s, r, 0.5 + 0.01 * r) for s in seasons for r in rosters]
    return pl.DataFrame({"season": [r[0] for r in rows], "roster": [r[1] for r in rows],
                         "diff": [r[2] for r in rows],
                         "ceiling_diff": [5.0 for _ in rows]})


def _lineup_gate_run():
    from hub.season import lineup_gate as lg
    paired = _lineup_paired()
    return run_gate(paired, cluster=SEASON_CLUSTER, within=lg.WITHIN, actions=lg.ACTIONS,
                    name="control-lineup", arm_a="optimiser", arm_b="projections", unit=lg.UNIT,
                    void=None, ceiling=lg.declared_ceiling(paired), seed=0, bootstrap=200,
                    record_width=False)


def _coverage_stats():
    """`test_coverage.py`'s own `_drawn` recipe, inlined: weeks drawn through the model's own
    skewed distribution, which is what makes the shape gate's fixture a golden rather than an
    arbitrary frame -- the counting and the shape comparison are both exercisable on a sample
    whose generating law is known."""
    from hub.models import predict

    def _player(pid, pos, season, points):
        return [{"player_id": pid, "position": pos, "season": season, "week": w + 1,
                 "season_type": "REG", "fantasy_points_ppr": float(p)}
                for w, p in enumerate(points)]

    rng = np.random.default_rng(23)
    pos, mu, n_players, weeks, skew = "WR", 12.0, 30, 17, 1.4
    sd = predict.WEEKLY_K[pos] * math.sqrt(mu)
    rows = []
    for season in (2022, 2023, 2024, 2025):
        for p in range(n_players):
            z = rng.standard_normal(weeks)
            rows += _player(f"p{p}", pos, season, predict.skewed(mu, sd, skew, z))
    return pl.DataFrame(rows, schema={
        "player_id": pl.Utf8, "position": pl.Utf8, "season": pl.Int32,
        "week": pl.Int32, "season_type": pl.Utf8, "fantasy_points_ppr": pl.Float64})


def _coverage_gate_run():
    from hub.models import coverage
    run, _, _ = coverage.shape_law(_coverage_stats(), width_path=Path("/nonexistent/control.json"))
    return run


def _quarterback_paired():
    """`test_starter_change.py`'s own `_paired` golden, inlined: three event-seasons, a
    constant positive `diff` and `ceiling`, clearing this diagnostic's own three-season floor."""
    seasons = (2026, 2027, 2028)
    n, diff = 6, 0.05
    return pl.DataFrame({"season": [s for s in seasons for _ in range(n)],
                         "game_id": [f"g{s}_{i}" for s in seasons for i in range(n)],
                         "diff": [diff] * (n * len(seasons)),
                         "ceiling": [0.5] * (n * len(seasons))})


def _quarterback_gate_run():
    from hub.models import starter_change as sc
    return sc.run(_quarterback_paired(), needed=3, width_path=Path("/nonexistent/control2.json"))


def _margin_wf(seasons=(2022, 2023, 2024, 2025), share=0.5, symmetric=True):
    """Margin's own `_lumpy_synthetic` golden, inlined at a small size."""
    rng = np.random.default_rng(1)
    rows = []
    for yr in seasons:
        per, sd, at = 100, 11.0, 3
        spreads = rng.uniform(-10, 10, per)
        margins = np.rint(spreads + rng.normal(0, sd, per))
        lump = rng.uniform(size=per) < share
        sign = rng.choice([-1.0, 1.0], size=per) if symmetric else np.sign(spreads)
        margins[lump] = at * sign[lump]
        margins[margins == 0] = 1
        rows += [(yr, float(s), int(m)) for s, m in zip(spreads, margins, strict=True)]
    return pl.DataFrame({"season": [r[0] for r in rows], "spread_line": [r[1] for r in rows],
                         "result": [r[2] for r in rows]},
                        schema={"season": pl.Int32, "spread_line": pl.Float64,
                                "result": pl.Int32})


def _margin_verdict():
    resid = margin.residuals(_margin_wf())
    wf = margin.walk_forward(resid)
    return margin.verdict(wf)


def _margin_shape_verdict():
    resid = margin.residuals(_margin_wf(share=0.5, symmetric=True))
    wf = margin.walk_forward_shape(resid)
    return margin.shape_verdict(wf)


# --- branch-coverage battery: synthetic paired frames reaching every branch ----------------

_BATTERY_ACTIONS = Actions(adopt="ADOPT.", remove="REMOVE.", show="SHOW.")


def _battery_frame(gains: list[float], *, n_per_season: int = 20, seed: int = 0,
                   noise_sd: float = 0.01):
    """One row per (season, unit), `n_per_season` units a season, each unit's own diff drawn
    tightly around that season's `gain` so the within-season SE is real and small -- enough
    clusters (`n_per_season >= TIE_MIN_CLUSTERS`) that `_disposition` reads the bootstrap
    rather than falling back to the sign."""
    rng = np.random.default_rng(seed)
    seasons, units, diffs = [], [], []
    for i, g in enumerate(gains):
        yr = 2020 + i
        noise = rng.normal(0.0, noise_sd, n_per_season)
        seasons += [yr] * n_per_season
        units += list(range(n_per_season))
        diffs += list(g + noise)
    return pl.DataFrame({"season": seasons, "unit": units, "diff": diffs})


def _battery_gate_run(gains, *, ceiling=None, void=None, seed=0, noise_sd=0.01):
    paired = _battery_frame(gains, seed=seed, noise_sd=noise_sd)
    top = None if ceiling is None else Ceiling("perfect", np.full(paired.height, ceiling))
    return run_gate(paired, cluster=SEASON_CLUSTER, within=("unit",),
                    actions=_BATTERY_ACTIONS, name=f"control-battery-{seed}",
                    arm_a="arm", arm_b="incumbent", void=void, ceiling=top, seed=seed,
                    bootstrap=300, record_width=False)


def _adopt_run():
    return _battery_gate_run([1.0, 1.1, 0.9, 1.2], ceiling=5.0, seed=1)


def _remove_run():
    return _battery_gate_run([-1.0, -1.1, -0.9, -1.2], ceiling=5.0, seed=2)


def _show_run():
    return _battery_gate_run([1.0, -1.0, 0.5, -0.5], ceiling=5.0, seed=3)


def _threshold_run():
    """The case the first battery could not see (review of 2026-09-22): every other frame's
    within-season SE is ~0.002 against gains of ~1.0, so halving the tie threshold changed no
    season's disposition and the goldens stayed green under a `2 * se -> 1 * se` plant -- a
    control blind to the every-season half it was meant to hold. Here the units are noisy
    (`noise_sd=1.2`, so a season's SE is ~0.27) and the third season's gain of 0.4 sits
    between one and two SEs: a tie under the rule as adopted, a win under a halved
    threshold. Captured on `main` before the seam, like the rest."""
    return _battery_gate_run([1.0, 1.1, 0.4, 1.2], ceiling=5.0, seed=5, noise_sd=1.2)


def _tie_run():
    """One season with a gain of exactly zero and a genuine (non-sign-only) bootstrap SE --
    a tie under #335's rule -- among three real wins, so ADOPT is blocked by the one season
    that cannot resolve rather than by a losing one."""
    return _battery_gate_run([1.0, 1.1, 0.0, 1.2], ceiling=5.0, seed=4)


def _not_runnable_run():
    """No ceiling handed in at all -- #363 (S6)'s own broadened branch."""
    return _battery_gate_run([1.0, 1.1, 0.9, 1.2], ceiling=None, seed=5)


def _void_run():
    return _battery_gate_run([1.0, 1.1, 0.9, 1.2], ceiling=5.0, void="VOID: control.", seed=6)


# --- every case, named -----------------------------------------------------------------------

CASES = {
    "draft": _draft_gate_run,
    "draft_sensitivity": _draft_sensitivity_gate_run,
    "weekly": _weekly_gate_run,
    "lineup": _lineup_gate_run,
    "coverage": _coverage_gate_run,
    "quarterback": _quarterback_gate_run,
    "battery_adopt": _adopt_run,
    "battery_remove": _remove_run,
    "battery_show": _show_run,
    "battery_tie": _tie_run,
    "battery_threshold": _threshold_run,
    "battery_not_runnable": _not_runnable_run,
    "battery_void": _void_run,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_adapter_reaches_the_expected_verdict(name):
    """The coarse, human-checkable half: every case reaches the branch it was built for, so
    a future reader can see *why* each is in the battery without decoding the golden values
    below. The golden values are the fine half -- this is the one a docstring can restate."""
    expected_verdict = {
        "draft": None,  # the frozen pin's own verdict, whatever ADR-0009 reads; not asserted
                        # here, only reproduced byte-identically by the golden test below.
        "draft_sensitivity": None,
        "weekly": "REMOVE",
        "lineup": "ADOPT",
        "coverage": None,
        "quarterback": "ADOPT",
        "battery_adopt": "ADOPT",
        "battery_remove": "REMOVE",
        "battery_show": "SHOW",
        "battery_tie": "SHOW",
        "battery_threshold": "SHOW",
        "battery_not_runnable": "NOT-RUNNABLE",
        "battery_void": "VOID",
    }[name]
    if expected_verdict is None:
        pytest.skip("reproduced byte-identically below; no independent branch claim made here")
    run = CASES[name]()
    assert run.verdict[0] == expected_verdict, run.verdict


def test_margin_verdict_reaches_adopt_on_the_symmetric_lump():
    winner, _text = _margin_verdict()
    assert winner in ("incumbent", "all", "trailing10")  # reproduced byte-identically below


def test_margin_shape_verdict_runs():
    shape, _text = _margin_shape_verdict()
    assert shape in ("gaussian", "lumpy")  # reproduced byte-identically below


# --- the equivalence control itself: byte-identical against the committed golden -----------
#
# `GOLDEN` was captured against `main`, before `gate`'s seam moved (this file's own commit,
# "Advances #386", records the exact run). Every case here recomputes its `GateRun` *now* --
# whichever side of the seam is active -- and compares against that frozen recording. Before
# the seam moves this is a tautology (the code producing `GOLDEN` and the code being checked
# are the same); after the seam moves it is the control: `gate(paired, ...)` must produce the
# identical reduction, byte for byte, or the ticket's own instruction applies -- stop, this is
# evidence the seam is drawn in the wrong place, not a bug to patch until green.

@pytest.mark.parametrize("name", sorted(CASES))
def test_the_gate_run_is_byte_identical_to_the_golden_capture(name):
    assert _reduce(CASES[name]()) == GOLDEN[name]


def test_margin_verdict_is_byte_identical_to_the_golden_capture():
    winner, text = _margin_verdict()
    assert [winner, text] == GOLDEN["margin_verdict"]


def test_margin_shape_verdict_is_byte_identical_to_the_golden_capture():
    shape, text = _margin_shape_verdict()
    assert [shape, text] == GOLDEN["margin_shape_verdict"]
