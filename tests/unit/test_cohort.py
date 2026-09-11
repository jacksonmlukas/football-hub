"""The Cohort: the rosters a season-level gate is scored on.

Both season-level gates draft twenty rosters a season, in seat 3, over fourteen rounds, with
the market strategy, seeded the same way -- and each built its own. The seed formula was
hand-copied into both, so if either drifted the two gates would silently score different
cohorts while describing themselves as identical harnesses.

The cause was a narrow interface, not carelessness. `backtest.play` returns names and
positions and discards the board indices and the undrafted remainder; the weekly gate's waiver
arm needs both, because a roster that changes week to week cannot be a fixed list of names. So
it could not use `play` and wrote the recipe again -- inside a `main()`, where no test reached
it.

What is asserted here is the recipe, not a golden roster: same ingredients in, same cohort out.
A frozen list of player indices would pass while the seed drifted, as long as it drifted in the
fixture too.
"""
import ast
import pathlib

import numpy as np
import polars as pl
import pytest

from hub.draft import cohort as C
from hub.draft.board import BuildReport


def _board(n=200):
    """A board wide enough to fill twelve teams for fourteen rounds -- 168 picks, so a
    narrower one silently produces short rosters rather than failing."""
    pos = [("QB", "RB", "WR", "TE")[i % 4] for i in range(n)]
    return pl.DataFrame({
        "player": [f"P{i:03d}" for i in range(n)],
        "pos": pos,
        "ecr": [float(i + 1) for i in range(n)],
        "adp": [float(i + 1) for i in range(n)],
        "vor": [float(n - i) for i in range(n)],
        "proj_blend": [float(n - i) / 4 for i in range(n)],
    })


def test_a_cohort_is_the_drafts_asked_for():
    got = C.cohort(_board(), 2024, drafts=5)
    assert len(got.rosters) == 5 and len(got.pool) == 5


def test_every_roster_is_the_rounds_asked_for():
    got = C.cohort(_board(), 2024, drafts=3, rounds=14)
    assert {len(r) for r in got.rosters} == {14}


def test_the_same_season_and_seed_give_the_same_cohort():
    """The property the hand-copied formula put at risk."""
    a = C.cohort(_board(), 2024, drafts=4, seed=0)
    b = C.cohort(_board(), 2024, drafts=4, seed=0)
    assert a.rosters == b.rosters and a.pool == b.pool


def test_a_different_seed_gives_a_different_cohort():
    """Otherwise the seed is decoration and the two gates score one draft twenty times."""
    a = C.cohort(_board(), 2024, drafts=4, seed=0)
    b = C.cohort(_board(), 2024, drafts=4, seed=7)
    assert a.rosters != b.rosters


def test_a_different_season_gives_a_different_cohort():
    """The season is in the seed, so a four-season gate is not one cohort scored four times."""
    assert C.cohort(_board(), 2024, drafts=4).rosters != \
        C.cohort(_board(), 2025, drafts=4).rosters


def test_the_recipe_is_its_ingredients_not_a_frozen_answer():
    """The assertion that catches a drifted seed. Run the documented ingredients directly and
    require the cohort to equal them -- a golden list of indices would pass while the formula
    moved, as long as it moved in the fixture too."""
    from hub.draft.backtest import market_strategy
    from hub.draft.optimize import simulate_remaining_draft
    from hub.draft.state import DraftState

    board = _board()
    got = C.cohort(board, 2024, drafts=2, seed=0)
    for k in range(2):
        room = simulate_remaining_draft(
            board, DraftState(taken=[]), my_slot=C.SLOT, teams=C.TEAMS, rounds=C.ROUNDS,
            rng=np.random.default_rng(C.seed_for(0, 2024, k)), my_pick=market_strategy())
        assert got.rosters[k] == [int(i) for i in room[C.SLOT - 1]]


def test_the_pool_is_everyone_nobody_drafted():
    """The waiver arm adds from it, so a drafted player leaking in would let a roster pick up
    someone another team already holds."""
    got = C.cohort(_board(), 2024, drafts=3)
    for k, pool in enumerate(got.pool):
        assert not set(pool) & set(got.rosters[k])
        assert len(set(pool)) == len(pool)


def test_the_pool_excludes_every_seat_not_only_mine():
    """Twelve teams draft; only one roster is scored. A pool that removed just my players
    would offer eleven other rosters' picks as free agents."""
    board = _board()
    got = C.cohort(board, 2024, drafts=1, rounds=14)
    drafted_by_the_room = board.height - len(got.pool[0])
    assert drafted_by_the_room == C.TEAMS * 14


def test_positions_are_carried_so_a_caller_need_not_re_read_the_board():
    got = C.cohort(_board(), 2024, drafts=1)
    assert len(got.pos) == 200
    assert all(p in ("QB", "RB", "WR", "TE") for p in got.pos)


def test_a_null_position_becomes_a_stated_placeholder_not_none():
    """Both gates index positions into a lineup rule that expects strings."""
    board = _board().with_columns(
        pl.when(pl.col("player") == "P000").then(None).otherwise(pl.col("pos")).alias("pos"))
    assert C.cohort(board, 2024, drafts=1).pos[0] == "NA"


@pytest.mark.parametrize("drafts", [0, 1])
def test_a_degenerate_cohort_is_a_shape_not_a_crash(drafts):
    got = C.cohort(_board(), 2024, drafts=drafts)
    assert len(got.rosters) == drafts


# --- the report travels with the frame (#231) ------------------------------
#
# `cohort` had no `report` parameter, so the two season-side gates -- both of which unpack
# `board, report = board_as_of(yr)` and have already read the report to refuse a season short
# a Correction -- handed the frame on alone and let `board.report_for` derive a fresh one from
# its columns. The re-derivation agrees on a historical board, which is why nothing ranked
# differently; agreeing today is exactly the condition that stops holding without saying so.


def _disagreeing_board(n=200):
    """A board carrying `adp`, where `adp` and `ecr` order the room differently.

    The stock fixture sets both to `i + 1`, so `w * adp + (1 - w) * ecr` is the same number
    whichever the report says -- a frame on which this seam cannot be observed at all. Here
    the draft market reverses consensus, so `blended_adp` produces a different `mu_pick`, the
    opponents perceive a different board, and the cohort that falls out is different.
    """
    df = _board(n)
    return df.with_columns(pl.Series("adp", [float(n - i) for i in range(n)]))


def test_the_report_the_caller_holds_is_used_and_not_re_derived():
    """The acceptance criterion, on a frame whose report disagrees with what it implies.

    `adp` is on this board, so `BuildReport.of_served` derives `adp=True` and `blended_adp`
    blends the draft market in. Handing `cohort` a report that says the stage did not run must
    produce the ECR-only room instead -- which is only observable if the passed report is what
    reaches `simulate_remaining_draft`. If it is dropped, both calls take the derivation and
    the two cohorts are identical.
    """
    board = _disagreeing_board()
    derived = C.cohort(board, 2024, drafts=2, seed=0)
    said_no_market = C.cohort(board, 2024, drafts=2, seed=0,
                              report=BuildReport(adp=False))
    assert derived.rosters != said_no_market.rosters, (
        "the report passed in was dropped and `report_for` derived one from the frame, which "
        "is the re-guess #199's report layer exists to end")
    # And the other direction, so the assertion above cannot pass on a report that is merely
    # *different* rather than *used*: a report agreeing with the frame reproduces it exactly.
    assert C.cohort(board, 2024, drafts=2, seed=0,
                    report=BuildReport(adp=True)).rosters == derived.rosters


def test_the_report_stays_optional():
    """The expand half of expand-then-contract. Every caller and fixture holding only a frame
    keeps working and keeps getting the derivation it already got -- which is what let the
    eleven #199 sites move one at a time, and what lets these two move without the rest."""
    import inspect
    param = inspect.signature(C.cohort).parameters["report"]
    assert param.default is None, (
        "making `report` required turns every frame-only caller into a break, which is the "
        "contract half of this change and not part of it")
    board = _disagreeing_board()
    assert C.cohort(board, 2024, drafts=1, seed=0).rosters == \
        C.cohort(board, 2024, drafts=1, seed=0, report=None).rosters


def test_no_gate_calls_the_cohort_without_the_report_it_is_holding():
    """The half of #231 that is about the callers, and it has to be a scan.

    `report` is optional so that a caller with only a frame keeps working -- which means a
    gate that stops passing it goes back to the re-derivation silently, and no assertion about
    `cohort`'s behaviour would notice. Both season-side gates already hold a `BuildReport`:
    `require_corrections` reads it before either of them drafts anything. Holding the recorded
    answer and letting it be inferred again downstream is the act this ticket closes, so what
    is pinned is that neither of them does it.
    """
    import ast
    import pathlib

    season = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub" / "season"
    bare = []
    for path in sorted(season.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "cohort"):
                continue
            if not any(kw.arg == "report" for kw in node.keywords):
                bare.append(f"{path.name}:{node.lineno}")
    assert not bare, (
        f"{bare} draft a Cohort without passing the report they unpacked from `board_as_of`, "
        f"so `simulate_remaining_draft` re-derives one from the frame's columns. Pass it: the "
        f"parameter is optional for frame-only callers, and a gate is not one.")


def test_the_cohort_is_what_the_gates_used_to_build_for_themselves():
    """The criterion that matters. Both published gate results were measured on the old
    recipe, so a cohort differing from it -- by a round, a seed, a strategy -- would
    invalidate a recorded number without saying so.

    `backtest.play` is what the lineup gate called; it is still here, so the equality can be
    asserted directly rather than against a frozen list that would drift with the fixture."""
    from hub.draft.backtest import market_strategy, play

    board = _board()
    got = C.cohort(board, 2024, drafts=3, seed=0)
    names_at = board["player"].to_list()
    for k in range(3):
        was, was_pos = play(board, market_strategy(), my_slot=C.SLOT, teams=C.TEAMS,
                            rounds=14, rng=np.random.default_rng(0 + 1000 * 2024 + k))
        assert [names_at[i] for i in got.rosters[k]] == was
        assert [got.pos[i] for i in got.rosters[k]] == was_pos


# --- the recipe is declared once (#200) --------------------------------------
#
# `seed_for` exists because "a formula copied by hand into two places is one that eventually
# differs in one", and by 2026-09-11 its constants had been copied into three more: the draft
# count as a bare `default=20` in three CLIs, and the round count as `optimize.DEFAULT_ROUNDS`
# and `cohort.ROUNDS`, two declarations of one 14 under two different reasons. The seed
# formula itself now has a sibling rather than a copy -- `backtest.draft_root` descends from
# the #195 seeding tree, and the Cohort deliberately does not (its docstring says why) -- so
# what is guarded is that each recipe is written where it is declared and nowhere else.

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"


def _walk(path: pathlib.Path):
    return ast.walk(ast.parse(path.read_text()))


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    """Line -> the name of the function that line sits inside, for reporting."""
    owner: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(line, node.name)
    return owner


def _is_season_times_1000(node: ast.AST) -> bool:
    """`1000 * season` or `season * 1000`: the Cohort formula's fingerprint."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return False
    sides = (node.left, node.right)
    return (any(isinstance(s, ast.Constant) and s.value == 1000 for s in sides)
            and any(isinstance(s, ast.Name) and s.id == "season" for s in sides))


def test_the_seed_formula_is_written_once():
    """The AST guard the ticket asks for: the Cohort's draw is stated in `seed_for` and the
    backtest's root in `draft_root`, and neither expression appears anywhere else in `src/`.

    Two fingerprints. The integer formula is `1000 * season` inside an expression -- the
    thing that was hand-copied into both season-side gates and then into `compare` and
    `ceiling`. The tree's is `root_seed` called with coordinates: one argument is a caller
    handing a root down (`diagnose`, `win_probability`), which is what the pass-through
    exists for; two or more is a caller re-deriving the backtest's room recipe.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        owner = _enclosing_functions(tree)
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0)
            where = f"{path.relative_to(SRC)}:{line} in {owner.get(line)}"
            if _is_season_times_1000(node) and not (
                    path.name == "cohort.py" and owner.get(line) == "seed_for"):
                offenders.append(f"{where} -- the Cohort seed formula, restated")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "root_seed" and len(node.args) > 1 and not (
                        path.name == "backtest.py" and owner.get(line) == "draft_root")):
                offenders.append(f"{where} -- the backtest's room recipe, restated")
    assert not offenders, (
        "a room recipe written outside its declaration -- `cohort.seed_for` for the Cohort "
        f"and `backtest.draft_root` for the backtest; call the declaration: {offenders}")


def test_rounds_is_one_declaration():
    """`cohort.ROUNDS` names `optimize.DEFAULT_ROUNDS` rather than restating 14 beside it.

    Read off the source rather than compared by value: `assert ROUNDS == DEFAULT_ROUNDS`
    passes on two independent 14s, which is the state this test exists to end.
    """
    from hub.draft.optimize import DEFAULT_ROUNDS

    assigned = [node.value for node in _walk(SRC / "draft" / "cohort.py")
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ROUNDS" for t in node.targets)]
    assert len(assigned) == 1
    assert isinstance(assigned[0], ast.Name) and assigned[0].id == "DEFAULT_ROUNDS", (
        "cohort.ROUNDS is declared as a literal beside optimize.DEFAULT_ROUNDS; two "
        "declarations of one number are two numbers as soon as one is edited")
    assert C.ROUNDS == DEFAULT_ROUNDS


@pytest.mark.parametrize("cli", [
    SRC / "draft" / "backtest.py",
    SRC / "season" / "lineup_gate.py",
    SRC / "season" / "weekly_gate.py",
])
def test_each_cli_defaults_its_draft_count_from_the_declaration(cli):
    """`--drafts` defaults to `DRAFTS`, the name, in all three Gate CLIs -- not to a 20 that
    happens to agree with it today."""
    defaults = []
    for node in _walk(cli):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == "--drafts"):
            continue
        defaults += [kw.value for kw in node.keywords if kw.arg == "default"]
    assert len(defaults) == 1, f"{cli.name} declares --drafts {len(defaults)} times"
    got = defaults[0]
    assert isinstance(got, ast.Name) and got.id == "DRAFTS", (
        f"{cli.name} defaults --drafts to {ast.unparse(got)} rather than to "
        f"`cohort.DRAFTS`; the literal is the recipe restated where nothing updates it")
