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
        # Only the names this module took from `hub.models.experiment`: a module of its own
        # `summarise` (`hub.models.spread` has a per-season MAE table by that name, #343) is
        # not the function this contract is about, and was read as it before the three
        # hand-built harnesses became gates.
        from_experiment = {a.asname or a.name for node in ast.walk(tree)
                           if isinstance(node, ast.ImportFrom)
                           and node.module == "hub.models.experiment" for a in node.names}
        name_calls = [node for node in ast.walk(tree)
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                      and node.func.id in ("run_gate", "gate", "summarise")
                      and node.func.id in from_experiment]
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


# --- paired_gain, the other function (#311) ----------------------------------------------
#
# `run_gate` takes `cluster` with no default and the AST walk above reads it off every call.
# `paired_gain` is the function the gates that bypass `run_gate` read their significance half
# from, and until #311 it computed its standard error over the row with no cluster argument at
# all -- the one function in the repo making the claim "each row is independent", which is the
# claim this file exists to keep any harness from making. #343 routed `spread` and `injury`
# through `Harness.run` (which reads `SEASON_CLUSTER` itself), so `hub.exhibits.weekly_projection`'s
# diagnostic contrast is the one call site left today; this holds every call site there is
# or will be, not the one that happens to exist.


def paired_gain_violations(source: str, where: str) -> list[str]:
    """Every `paired_gain(...)` call in `source` that does not name the season as its cluster.

    The argument is a row-parallel array, not a column name, so the call has to *read the season
    column*: `cluster=errs["season"].to_numpy()` is held, `cluster=errs["week"]`, no `cluster`
    at all, and a bare name this test cannot resolve are each a violation -- the last on the
    same principle the `run_gate` walk applies to its own unresolvable argument.
    """
    import ast

    out = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute) else None)
        if name != "paired_gain":
            continue
        cluster = next((kw.value for kw in node.keywords if kw.arg == "cluster"), None)
        if cluster is None:
            out.append(f"{where}:{node.lineno} calls paired_gain with no cluster")
            continue
        reads = {c.value for c in ast.walk(cluster) if isinstance(c, ast.Constant)}
        names = {n.id for n in ast.walk(cluster) if isinstance(n, ast.Name)}
        if "season" not in reads and "SEASON_CLUSTER" not in names:
            out.append(f"{where}:{node.lineno} passes a cluster that does not read the season: "
                       f"{ast.unparse(cluster)}")
    return out


def test_cluster_has_no_default_on_every_function_that_decides():
    """A signature check, not a convention: omit it and Python says so."""
    import inspect

    from hub.models import experiment

    for fn in (experiment.paired_gain, experiment.run_gate, experiment.gate):
        param = inspect.signature(fn).parameters["cluster"]
        assert param.default is inspect.Parameter.empty, (
            f"{fn.__name__}'s cluster has a default ({param.default!r}); what one independent "
            f"observation is has no safe one")


def test_every_paired_gain_call_in_the_source_passes_the_season():
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
    calls, bad = 0, []
    for path in sorted(root.rglob("*.py")):
        if path == root / "models" / "experiment.py":
            continue
        src = path.read_text()
        calls += src.count("paired_gain(")
        bad += paired_gain_violations(src, str(path.relative_to(root)))
    assert calls, "no paired_gain call found -- the walk is vacuous, or the function is gone"
    assert not bad, f"paired_gain without the season as its cluster: {bad}"


def test_the_paired_gain_walk_sees_the_plants():
    """Rule 18: the check can fail. Three planted calls -- no cluster, the week, the row -- are
    each reported, and the two forms the repo uses are not."""
    held = ('g = paired_gain(a, b, cluster=errs["season"].to_numpy(), '
            'season=errs["season"], within=errs["week"])\n'
            'h = experiment.paired_gain(a, b, cluster=SEASON_CLUSTER, season=s, within=w)\n')
    assert paired_gain_violations(held, "ok.py") == []
    assert len(paired_gain_violations(
        "paired_gain(a, b, season=s, within=w)\n", "none.py")) == 1
    assert len(paired_gain_violations(
        'paired_gain(a, b, cluster=errs["week"].to_numpy(), season=s, within=w)\n',
        "week.py")) == 1
    assert len(paired_gain_violations(
        "paired_gain(a, b, cluster=np.arange(len(a)), season=s, within=w)\n", "row.py")) == 1
