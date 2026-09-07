"""Every gate resamples the season, and this is what stops one of them drifting back.

Issue #45's claim is about the data: within a season the rows share a board, a player pool, a
schedule and one realisation of the year, so the honest replication count is the number of
seasons. The claim is either true or false about the data -- it is not a setting -- which
means all three harnesses have to make it, and the failure mode is not that one of them makes
a *different* claim loudly. It is that one of them quietly makes none.

**This repo has already been bitten by exactly that, twice, and recorded it both times.**
`experiment.gate` exists because three modules each remembered ADR-0019's rule and two of them
remembered it wrong -- "one copy of a rule had been corrected and the others were never
revisited, because nothing connected them". `tests/contracts/test_every_contract_is_applied.py`
opens on the same shape. The cluster was the third instance in flight: `weekly_gate` had been
corrected from the row to the roster and the other two had not been corrected at all, so the
draft gate and the lineup gate were still resampling rows while the weekly gate's docstring
explained why that was the error that once produced an apparent 4-sigma result.

So the enforcing piece is not the three call sites. It is this.

The assertion is on the **call**, not on a constant. A harness that declared
`CLUSTER = ("season",)` and then passed something else -- or passed nothing, which is the
default and is the mistake -- would satisfy any check that only read the module's constants.
This resolves the argument actually written at the call site through the module it is written
in, so the thing asserted is the value `summarise` receives.
"""
import ast
import importlib
import pathlib

import pytest

from hub.models.experiment import SEASON_CLUSTER

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"

# The three harnesses `CONTEXT.md` calls Gates, as (module path, dotted import name).
HARNESSES = {
    "backtest": ("draft/backtest.py", "hub.draft.backtest"),
    "lineup_gate": ("season/lineup_gate.py", "hub.season.lineup_gate"),
    "weekly_gate": ("season/weekly_gate.py", "hub.season.weekly_gate"),
}


def _summarise_calls(path: pathlib.Path) -> list[ast.Call]:
    """Every `summarise(...)` call in a module, however it is spelled."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else (
            fn.attr if isinstance(fn, ast.Attribute) else None)
        if name == "summarise":
            out.append(node)
    return out


def _cluster_arg(call: ast.Call) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == "cluster":
            return kw.value
    return None


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_the_gate_passes_a_cluster_at_all(harness):
    """The default is no cluster, so an omitted argument is the row -- silently, and with the
    same interval shape a correct run produces. Nothing about the output says which happened,
    which is why this is asserted rather than reviewed."""
    rel, _ = HARNESSES[harness]
    calls = _summarise_calls(SRC / rel)
    assert calls, f"{harness} calls summarise nowhere"
    for call in calls:
        assert _cluster_arg(call) is not None, (
            f"{harness} calls summarise with no cluster -- the default is the row, and "
            f"`summarise`'s own docstring says there is no safe default")


@pytest.mark.parametrize("harness", sorted(HARNESSES))
def test_the_cluster_the_gate_passes_is_the_season(harness):
    """Resolved through the module, so an alias is fine and a wrong value is not.

    `weekly_gate.CLUSTER` is bound to `SEASON_CLUSTER` and keeps its name because the module's
    own history is written around it; what matters is the value that reaches `summarise`, so
    that is what is read -- by looking the argument's name up on the imported module rather
    than by matching the identifier `SEASON_CLUSTER` as text.
    """
    rel, dotted = HARNESSES[harness]
    module = importlib.import_module(dotted)
    for call in _summarise_calls(SRC / rel):
        arg = _cluster_arg(call)
        assert isinstance(arg, ast.Name), (
            f"{harness} passes a cluster this test cannot resolve "
            f"({ast.dump(arg) if arg is not None else 'nothing'}); if it is now a literal or "
            f"an expression, read it here rather than deleting this")
        got = getattr(module, arg.id)
        assert tuple(got) == tuple(SEASON_CLUSTER), (
            f"{harness} resamples {tuple(got)}, not the season")


def test_no_harness_declares_a_second_season_cluster():
    """One name for one claim, the same rule `MIN_SE` is held to. A harness that wrote its own
    `("season",)` would be a second declaration of a shared claim about the data, and the way
    these drift is that one of them is later corrected."""
    offenders = []
    for harness, (rel, dotted) in sorted(HARNESSES.items()):
        module = importlib.import_module(dotted)
        for node in ast.parse((SRC / rel).read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else (
                [node.target] if isinstance(node, ast.AnnAssign) and node.value else [])
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                value = getattr(module, t.id, None)
                if isinstance(value, tuple) and tuple(value) == tuple(SEASON_CLUSTER) \
                        and value is not SEASON_CLUSTER:
                    offenders.append(f"{harness}:{t.id}")
    assert not offenders, f"a second season cluster: {offenders}"


def test_the_shared_declaration_is_where_the_gates_read_it_from():
    """And it is one object, so an alias cannot drift from what it aliases."""
    from hub.season import weekly_gate

    assert weekly_gate.CLUSTER is SEASON_CLUSTER
    assert SEASON_CLUSTER == ("season",)
