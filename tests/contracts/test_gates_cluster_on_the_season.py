"""Every gate resamples the season, and this is what stops one of them drifting back.

Issue #45's claim is about the data: within a season the rows share a board, a player pool, a
schedule and one realisation of the year, so the honest replication count is the number of
seasons. The claim is either true or false about the data -- it is not a setting -- which
means all the harnesses have to make it, and the failure mode is not that one of them makes a
*different* claim loudly. It is that one of them quietly makes none.

**This repo has already been bitten by exactly that, twice, and recorded it both times.**
`experiment.gate` exists because three modules each remembered ADR-0019's rule and two of them
remembered it wrong -- "one copy of a rule had been corrected and the others were never
revisited, because nothing connected them". The cluster was the third instance in flight:
`weekly_gate` had been corrected from the row to the roster and the other two had not been
corrected at all, so the draft gate and the lineup gate were still resampling rows while the
weekly gate's docstring explained why that was the error that once produced an apparent
4-sigma result.

**#387.** Before, this file AST-walked four call sites for a hand-kept `HARNESSES` dict, and
its own docstring already recorded the drift: "coverage's gate is not in ... `HARNESSES`". A
`Harness` (#387, `hub.models.experiment.Harness`) is not free to name a cluster at all --
`Harness.run`/`Harness.decide` always read `SEASON_CLUSTER` themselves, never a field a
declaration could get wrong -- so the enforcing piece moves from "does every call site pass
the right value" to "does every gate go through a Harness at all", which
`tests/gate_harnesses.py`'s discovery is.
"""
import pathlib

from gate_harnesses import GATE_MODULES, all_harnesses

from hub.models.experiment import SEASON_CLUSTER


def test_every_gate_declares_a_harness_and_harnesses_never_name_a_cluster():
    """`Harness` (see `hub.models.experiment`) has no `cluster` field at all -- so a gate that
    goes through one cannot pass the row, or anything else, in its place. What this checks is
    that every gate module actually has one to go through."""
    found = all_harnesses()
    assert not hasattr(next(iter(found.values())), "cluster"), (
        "Harness grew a cluster field -- the point of this contract is that a gate reads the "
        "season off Harness.run/Harness.decide, which read SEASON_CLUSTER themselves, not off "
        "a field a declaration could name wrong")


def test_every_gate_module_resamples_the_season():
    """Every gate module reads the season one of two ways, and both are held here.

    `coverage`, `starter_change` and `margin` call `Harness.run`/`Harness.decide` directly,
    which read `SEASON_CLUSTER` themselves -- no `cluster=` at the call site at all, and
    nothing here to get wrong. `backtest`, `weekly_gate` and `lineup_gate` still call
    `run_gate` by name (so `tests/unit`'s monkeypatches keep working -- #387's own report says
    why), so those three still spell `cluster=` explicitly; what this holds is that whatever
    they spell resolves to `SEASON_CLUSTER`, the same thing this file has always held.
    """
    import ast
    import importlib
    import inspect

    for module in GATE_MODULES:
        mod = importlib.import_module(module)
        src = inspect.getsource(mod)
        tree = ast.parse(src)
        attr_calls = {node.func.attr for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        name_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id in ("run_gate", "gate", "summarise")]
        if {"run", "decide"} & attr_calls and not name_calls:
            continue  # reads SEASON_CLUSTER through Harness.run/Harness.decide; nothing to check
        assert name_calls, (
            f"{module} calls neither Harness.run/Harness.decide nor run_gate/gate/summarise "
            f"directly -- nothing here resamples the season")
        for call in name_calls:
            cluster_kw = next((kw.value for kw in call.keywords if kw.arg == "cluster"), None)
            assert cluster_kw is not None, (
                f"{module} calls {ast.unparse(call.func)} with no cluster -- the default is "
                f"the row and summarise's own docstring says there is no safe default")
            assert isinstance(cluster_kw, ast.Name), (
                f"{module} passes a cluster this test cannot resolve "
                f"({ast.dump(cluster_kw)}); if it is now a literal or an expression, read it "
                f"here rather than deleting this")
            got = getattr(mod, cluster_kw.id)
            assert tuple(got) == tuple(SEASON_CLUSTER), (
                f"{module} resamples {tuple(got)}, not the season")


def test_the_shared_declaration_is_where_the_gates_read_it_from():
    """And it is one object, so an alias cannot drift from what it aliases."""
    from hub.season import weekly_gate

    assert weekly_gate.CLUSTER is SEASON_CLUSTER
    assert SEASON_CLUSTER == ("season",)


def test_season_cluster_is_declared_once():
    """One name for one claim, the same rule `MIN_SE` is held to: `SEASON_CLUSTER` lives in
    `hub.models.experiment` and nowhere else redeclares `("season",)` as its own module-level
    tuple -- a second declaration is exactly how these drift, one corrected and the other not."""
    import ast

    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path == root / "models" / "experiment.py":
            continue
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets, val = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, val = [node.target], node.value
            else:
                continue
            for t in targets:
                if not isinstance(t, ast.Name):
                    continue
                try:
                    value = ast.literal_eval(val)
                except (ValueError, TypeError):
                    continue
                if value == tuple(SEASON_CLUSTER):
                    offenders.append(f"{path.relative_to(root)}:{t.id}")
    assert not offenders, f"a second season cluster: {offenders}"
