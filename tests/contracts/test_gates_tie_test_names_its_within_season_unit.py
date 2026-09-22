"""Every gate names its within-season repeated-measure unit, and this is what stops one of
them drifting back to guessing.

ADR-0019's #335 amendment: a season counts as a *win* only if its gain clears its own noise,
and "its own noise" is a bootstrap over the within-season repeated-measure unit --
`docs/method.md` rule 3's unit, different for every gate (the draft room, the roster, the
player, the event). `experiment.run_gate` has no default for `within`, the same way it has
none for `cluster` (`tests/contracts/test_gates_cluster_on_the_season.py`) -- a caller that
omitted it would get a `TypeError`, not a guess. This is the companion contract: the five
`run_gate` call sites #335 names all pass `within=`, and what they pass resolves to a
declared, non-empty tuple rather than an inline guess a later edit could quietly narrow.

Coverage's gate is not in `test_gates_cluster_on_the_season.py`'s `HARNESSES` -- an earlier
omission, not a decision this file repeats. #335's own acceptance criteria names five
`run_gate` call sites, and this reads all five.
"""
import ast
import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"

# The five `run_gate` call sites #335 names, as (module path, dotted import name).
HARNESSES = {
    "backtest": ("draft/backtest.py", "hub.draft.backtest"),
    "coverage": ("models/coverage.py", "hub.models.coverage"),
    "starter_change": ("models/starter_change.py", "hub.models.starter_change"),
    "weekly_gate": ("season/weekly_gate.py", "hub.season.weekly_gate"),
    "lineup_gate": ("season/lineup_gate.py", "hub.season.lineup_gate"),
}


def _run_gate_calls(path: pathlib.Path) -> list[ast.Call]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else None)
        if name == "run_gate":
            out.append(node)
    return out


def _within_arg(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "within":
            return kw.value
    return None


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_every_run_gate_call_passes_within(harness):
    """The default is a `TypeError`, so an omitted argument cannot reach production silently
    the way an omitted `cluster` could before this parameter existed."""
    rel, _ = HARNESSES[harness]
    calls = _run_gate_calls(SRC / rel)
    assert calls, f"{harness} does not call run_gate"
    for call in calls:
        assert _within_arg(call) is not None, (
            f"{harness} calls run_gate with no within -- run_gate's own signature has no "
            f"default, so this would already be a TypeError at runtime; this test is what "
            f"catches it before that call is made")


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_the_within_unit_resolves_to_a_declared_non_empty_tuple(harness):
    """Resolved through the module, the same way the cluster contract reads `cluster=`: an
    alias is fine, an inline guess or an empty tuple is not."""
    rel, dotted = HARNESSES[harness]
    module = importlib.import_module(dotted)
    for call in _run_gate_calls(SRC / rel):
        arg = _within_arg(call)
        assert isinstance(arg, ast.Name), (
            f"{harness} passes a within-season unit this test cannot resolve "
            f"({ast.dump(arg) if arg is not None else 'nothing'}); if it is now a literal or "
            f"an expression, read it here rather than deleting this")
        got = getattr(module, arg.id)
        assert isinstance(got, tuple) and got, (
            f"{harness}'s within-season unit ({arg.id} = {got!r}) must be a declared, "
            f"non-empty tuple of column names")


# The per-gate units ADR-0019's #335 amendment states, so a reader has one place naming what
# `within` resolves to at each site without opening five files.
DECLARED_UNITS = {
    "backtest": ("draft",),
    "coverage": ("player_id",),
    "starter_change": ("season",),       # one cluster per season: a real no-op (#335 item 2)
    "weekly_gate": ("roster",),
    "lineup_gate": ("roster",),
}


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_the_within_unit_matches_the_amendment_s_table(harness):
    rel, dotted = HARNESSES[harness]
    module = importlib.import_module(dotted)
    for call in _run_gate_calls(SRC / rel):
        arg = _within_arg(call)
        assert isinstance(arg, ast.Name)
        got = tuple(getattr(module, arg.id))
        assert got == DECLARED_UNITS[harness], (
            f"{harness} resamples {got}, not {DECLARED_UNITS[harness]} -- update this table "
            f"and the ADR-0019 amendment together, never one without the other")
