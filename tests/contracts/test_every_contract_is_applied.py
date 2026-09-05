"""A contract that is declared and never applied reads as coverage.

`hub.contracts` describes eleven third-party sources, and its whole value is failing when one
of them changes shape. Four were applied to nothing: both CFBD endpoints, the odds snapshots
every prediction is priced from, and an undocumented ESPN endpoint that a scheduled job now
hits every ten minutes on a Sunday.

**The repo had already been bitten by this and fixed exactly one instance.** `hub.draft.board`
still carries the sentence: *"this contract was declared in `hub.contracts` and applied to
nothing at all."* One was found; four survived, because nothing connected them. That is the
same shape as the Gate in #14, where one copy of a rule had been corrected and the others were
never revisited.

So the enforcing piece is not the four call sites -- it is this, which makes a fifth inert
contract impossible to add without someone deciding not to.
"""
import ast
import pathlib

import pytest

from hub import contracts

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"


def _declared() -> set[str]:
    """Every `Contract` object in the module, by the name it is bound to."""
    tree = ast.parse((SRC / "contracts.py").read_text())
    out = set()
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "Contract"):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return out


def _registries(tree: ast.AST, declared: set[str]) -> dict[str, set[str]]:
    """Module-level dicts whose values are declared contracts, by the name they are bound to.

    `hub.fetch.nflverse` maps source name to contract in `SOURCES` and validates through the
    variable it takes out; `hub.fetch.cfbd` does the same with `CONTRACTS`. That is the
    better pattern and the guard has to follow it, or it pushes the next author into a style
    rather than a behaviour.
    """
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        # Both forms. Each real registry in this repo is annotated -- `SOURCES: dict[str,
        # Contract | None] = {...}` -- which is an `AnnAssign`, and a resolver that read only
        # bare assignments reported all eight of them as unapplied. Caught by running it.
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        held = {v.id for v in value.values if isinstance(v, ast.Name)} & declared
        if held:
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = held
    return out


def _yields(value: ast.AST, declared: set[str],
            registries: dict[str, set[str]]) -> set[str]:
    """Which declared contracts an expression can evaluate to. Empty means "cannot tell"."""
    if isinstance(value, ast.Name):
        return {value.id} & declared
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
        return registries.get(value.value.id, set())          # SOURCES[source]
    if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get" and isinstance(value.func.value, ast.Name)):
        return registries.get(value.func.value.id, set())     # CONTRACTS.get(endpoint)
    return set()


def _bindings(tree: ast.AST, declared: set[str],
              registries: dict[str, set[str]]) -> dict[str, set[str]]:
    """Local names, and the declared contracts each can be holding when it is validated."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            target, value = node.targets[0].id, node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value        # the walrus in cfbd
        else:
            continue
        if got := _yields(value, declared, registries):
            out.setdefault(target, set()).update(got)
    return out


def _applied_in(tree: ast.AST, declared: set[str]) -> set[str]:
    """The contracts this module actually validates, by resolving what each call validates.

    **The relationship, not two tokens in the same file.** This was "the module names a
    contract somewhere and calls `.validate` somewhere", which is true of a module that
    mentions the right contract in a comment and validates a different one -- and that is
    precisely the property the declaration exists to state. It arrived that way honestly: the
    strict `NAME.validate(` form false-positived on nflverse's registry, and loosening it far
    enough to admit the registry loosened it past the point of checking anything.

    A receiver this cannot resolve contributes nothing rather than everything, so a new
    indirection reads as unapplied and is fixed here, in the open.
    """
    registries = _registries(tree, declared)
    bindings = _bindings(tree, declared, registries)
    out: set[str] = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "validate"):
            continue
        recv = node.func.value
        if isinstance(recv, ast.Name):
            out |= ({recv.id} & declared) or bindings.get(recv.id, set())
    return out


def _applied() -> set[str]:
    declared, out = _declared(), set()
    for path in SRC.rglob("*.py"):
        if path.name == "contracts.py":
            continue
        out |= _applied_in(ast.parse(path.read_text()), declared)
    return out


def test_the_scan_finds_the_contracts_that_exist():
    """The test's own premise. A parser that quietly matched nothing would make every
    assertion below vacuously true, which is the failure mode of a guard like this."""
    found = _declared()
    assert len(found) >= 10, f"only found {sorted(found)}; the declaration shape has changed"
    assert {"DRAFT_BOARD", "ODDS_SNAPSHOT", "ESPN_SCOREBOARD"} <= found


def test_every_declared_contract_is_applied_somewhere():
    unapplied = _declared() - _applied()
    assert not unapplied, (
        f"declared and applied to nothing: {sorted(unapplied)}. A contract nobody calls reads "
        f"as coverage -- someone opening `hub.contracts` sees the source described and "
        f"concludes the fetch layer is guarded. Apply it at the fetcher that reads that "
        f"source, or delete the declaration.")


def _guard(source: str) -> set[str]:
    """Run the guard over a planted module. The guard itself, not a restatement of it."""
    import textwrap
    return _applied_in(ast.parse(textwrap.dedent(source)), _declared())


def test_a_contract_imported_and_never_validated_does_not_count():
    """Importing a contract and forgetting to call it looks exactly like applying it, from
    the import line."""
    assert _guard("""
        from hub.contracts import ODDS_SNAPSHOT
        def snapshot(df):
            return df
    """) == set()


def test_validating_a_different_contract_does_not_count_as_applying_this_one():
    """**The hole.** The guard asked whether the module named a contract and called
    `.validate` -- both true of a module that mentions the right one in a comment and
    validates another. That is the whole property the contract exists to state, and a guard
    that cannot see the difference is the third of its kind found on 2026-09-04, beside the
    odds credit floor and the secret scan."""
    got = _guard("""
        from hub.contracts import ODDS_SNAPSHOT, PBP
        def snapshot(df):
            # ODDS_SNAPSHOT is the shape every prediction is priced from.
            return PBP.validate(df)
    """)
    assert got == {"PBP"}, f"the odds snapshots are not enforced here, but {got} says so"


def test_a_bare_mention_in_a_comment_is_not_enforcement():
    assert _guard("""
        from hub.contracts import ODDS_SNAPSHOT
        def snapshot(df, other):
            # applied via ODDS_SNAPSHOT downstream
            return other.validate(df)
    """) == set()


def test_a_direct_call_counts():
    assert _guard("""
        from hub.contracts import ODDS_SNAPSHOT
        def snapshot(df):
            return ODDS_SNAPSHOT.validate(df)
    """) == {"ODDS_SNAPSHOT"}


def test_a_contract_reached_through_a_registry_still_counts():
    """The false positive that caused the loosening. `hub.fetch.nflverse` binds source name
    to contract and validates through the variable, and a guard that only saw the direct form
    would report six correctly-enforced contracts as inert."""
    assert _guard("""
        from hub.contracts import PBP, SCHEDULES, Contract
        SOURCES: dict[str, Contract | None] = {"pbp": PBP, "schedules": SCHEDULES}
        def load(source, df):
            contract = SOURCES[source]
            if contract is not None:
                contract.validate(df)
    """) == {"PBP", "SCHEDULES"}


def test_a_registry_reached_through_get_and_a_walrus_still_counts():
    """`hub.fetch.cfbd`'s form, which is a walrus over `.get` rather than a subscript."""
    assert _guard("""
        from hub.contracts import CFBD_GAMES, CFBD_LINES
        CONTRACTS = {"games": CFBD_GAMES, "lines": CFBD_LINES}
        def week(endpoint, df):
            if (contract := CONTRACTS.get(endpoint)) is not None:
                contract.validate(df)
    """) == {"CFBD_GAMES", "CFBD_LINES"}


def test_an_unresolvable_receiver_counts_for_nothing_rather_than_everything():
    """A new indirection reads as unapplied, so it is fixed here in the open rather than
    quietly widening what the guard will accept."""
    assert _guard("""
        from hub.contracts import PBP
        def load(get_contract, df):
            get_contract().validate(df)
    """) == set()


@pytest.mark.parametrize("name", sorted(_declared()))
def test_each_contract_names_whether_it_has_met_real_data(name):
    """Two of these were written from documentation and have never seen a live response, so
    a first failure is as likely to mean "the guess was wrong" as "the source broke". The
    reader of a red build should not have to work out which."""
    assert isinstance(getattr(contracts, name).verified_against_live, bool)
