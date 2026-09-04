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


def _applied() -> set[str]:
    """Every declared contract a module both names and validates.

    Two conditions, and neither alone is enough. A module that calls `.validate()` proves a
    contract is enforced *somewhere* in it; naming the contract proves it is that one.

    Deliberately not "a `NAME.validate()` call", which is what I wrote first. `hub.fetch.
    nflverse` maps source name to contract in a registry and validates through the variable,
    which is the better pattern -- and a guard that only recognised the direct form would have
    reported six correctly-enforced contracts as inert and pushed the next author into a
    style rather than a behaviour.
    """
    declared, out = _declared(), set()
    for path in SRC.rglob("*.py"):
        if path.name == "contracts.py":
            continue
        tree = ast.parse(path.read_text())
        validates = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                        and n.func.attr == "validate" for n in ast.walk(tree))
        if not validates:
            continue
        out |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} & declared
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


def test_a_contract_imported_and_never_validated_does_not_count():
    """The gap the two conditions close. Importing a contract and forgetting to call it looks
    exactly like applying it, from the import line."""
    import textwrap
    fake = ast.parse(textwrap.dedent("""
        from hub.contracts import ODDS_SNAPSHOT
        def snapshot(df):
            return df
    """))
    validates = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "validate" for n in ast.walk(fake))
    assert not validates, "a module that never validates must not count as applying anything"


@pytest.mark.parametrize("name", sorted(_declared()))
def test_each_contract_names_whether_it_has_met_real_data(name):
    """Two of these were written from documentation and have never seen a live response, so
    a first failure is as likely to mean "the guess was wrong" as "the source broke". The
    reader of a red build should not have to work out which."""
    assert isinstance(getattr(contracts, name).verified_against_live, bool)
