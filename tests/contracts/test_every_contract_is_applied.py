"""A contract that is declared and never applied reads as coverage.

`hub.contracts` describes fourteen third-party sources, and its whole value is failing when
one of them changes shape. Four were applied to nothing: both CFBD endpoints, the odds snapshots
every prediction is priced from, and an undocumented ESPN endpoint that a scheduled job now
hits every ten minutes on a Sunday.

**The repo had already been bitten by this and fixed exactly one instance.** `hub.draft.board`
still carries the sentence: *"this contract was declared in `hub.contracts` and applied to
nothing at all."* One was found; four survived, because nothing connected them. That is the
same shape as the Gate in #14, where one copy of a rule had been corrected and the others were
never revisited.

So the enforcing piece is not the four call sites -- it is this, which makes a fifth inert
contract impossible to add without someone deciding not to.

The second half of the file asks the other question a declaration makes: not "is this applied
anywhere" but "has it ever met the thing it describes". `verified_against_live` records that,
and the only assertion on it was that the attribute is a `bool` -- on a field declared
`bool = True`. Flipping both CFBD contracts to `True`, which erases the entire distinction the
field exists to record, left all twenty tests here green. Measured on 2026-09-05.
"""
import ast
import functools
import pathlib

import polars as pl
import pytest

from hub import contracts

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"
TESTS = pathlib.Path(__file__).resolve().parents[1]
FIXTURES = TESTS / "golden" / "fixtures"


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
    reader of a red build should not have to work out which.

    That the attribute is a `bool` is all this ever asserted, on a field declared
    `bool = True`. What the flag *says* is checked below.

    Three states now, not two: `None` is "nobody has measured", and it is the default, so a
    newly declared contract starts by saying nothing rather than by claiming it has met a
    live response. `isinstance(..., bool)` would have read `None` as a broken declaration."""
    value = getattr(contracts, name).verified_against_live
    assert value is True or value is False or value is None, (
        f"{name}.verified_against_live is {value!r}; the three states are True (a captured "
        f"payload proves it), False (only ever a hand-built shape), and None (unmeasured)")


# --- has the declaration ever met the thing it describes? ---------------------
#
# The flag is not asked to be true; it is asked to agree with evidence the repo already keeps.
# Every source is frozen under `tests/golden/fixtures/`, and that directory's README calls the
# `.synthetic` marker in a filename "not decoration": it marks a payload hand-built from docs
# because no key for that source exists on this machine. So the question "has this contract
# met a real response" has an answer in the tree -- which frozen payload is the contract
# actually validated against, and was that payload captured or guessed.
#
# The escape hatch is the honest one and the README already prescribes it: the day a key is
# added, the fixture is replaced with a real capture under a name without `.synthetic`, and
# the flag follows. Renaming the file is the edit; there is no list here to keep in step.
#
# **What the evidence can say, and what it cannot.** It resolved five of eleven contracts on
# 2026-09-05 -- PBP, FF_OPPORTUNITY, SCHEDULES, CFBD_GAMES, CFBD_LINES -- because those are
# the ones a test validates against a frozen payload. (#33 took it to eight of fourteen by
# freezing a capture of each source it added; the five below that resolve to nothing are the
# same five.) The other six resolved to nothing, and
# for a flag that defaulted to `True` that meant six contracts claiming they had met live
# data on no evidence at all: a contract written from documentation, exactly like the two
# this section exists for, could be added and nothing would go red.
#
# Six is now five, and how it moved is the point. ESPN_SCOREBOARD was in that group while a
# real capture of the endpoint sat under tests/golden/fixtures/ -- the test reading that file
# asserted its fields by hand, and a hand-written assert is not something this scan can see,
# so the contract said "unmeasured" beside its own evidence. The fix was to route the capture
# through `ESPN_SCOREBOARD.validate`, which is the escape hatch below rather than an exception
# to it: the flag followed the resolver, and the resolver followed the payload.
#
# Nobody has measured whether the five still unresolved have ever met a live response, and
# inventing an answer would be worse than saying nothing, so the fix is not to guess -- it is
# to stop silence reading as a claim. `None` is the default and means unmeasured, which is a
# different sentence in a violation from "written from documentation", and the three tests
# below together make the flag a function of the evidence rather than a claim beside it:
# a capture requires `True`, hand-built-only requires `False`, no evidence requires `None`.
# The way to move a contract off `None` is to validate it against a frozen payload, not to
# edit a list.


@functools.cache
def _fixtures() -> dict[str, bool]:
    """Every frozen payload, and whether it is a capture of a real response."""
    return {p.name: ".synthetic." not in p.name for p in FIXTURES.glob("*.json")}


def _payload_names(node: ast.AST, fixtures: dict[str, bool]) -> set[str]:
    """The frozen payloads an expression names outright.

    Matched against the files that exist rather than against a `.json` suffix, so a fixture
    renamed out from under this reads as no evidence rather than as a capture.
    """
    return {n.value for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in fixtures}


def _payload_bindings(scope: ast.AST, fixtures: dict[str, bool]) -> dict[str, set[str]]:
    """Local names, and the frozen payload each one can be holding."""
    out: dict[str, set[str]] = {}
    for node in ast.walk(scope):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            target, value = node.targets[0].id, node.value
        elif (isinstance(node, ast.AnnAssign) and node.value is not None
                and isinstance(node.target, ast.Name)):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.NamedExpr) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        else:
            continue
        if got := _payload_names(value, fixtures):
            out.setdefault(target, set()).update(got)
    return out


def _receiver_contracts(recv: ast.AST, declared: set[str]) -> set[str]:
    """Which declared contract a `.validate` receiver is, for the fixture-test forms."""
    if isinstance(recv, ast.Name):
        return {recv.id} & declared
    if isinstance(recv, ast.Call):
        # `shape_only(PBP).validate(df)`. Two of the fixture tests drop the volume floor
        # before validating, because a fixture is a handful of rows on purpose -- so a
        # resolver reading only a bare name finds no evidence for `PBP` or `FF_OPPORTUNITY`.
        return {a.id for a in recv.args if isinstance(a, ast.Name)} & declared
    return set()


def _exercised_in(scope: ast.AST, declared: set[str],
                  fixtures: dict[str, bool]) -> dict[str, set[str]]:
    """Which frozen payloads each contract is validated against, within one function.

    One hop from the validated expression to a local binding, and deliberately not two.
    `test_odds_fixture_parses_to_the_lines_table_shape` is two hops -- the payload, then rows
    built from it by hand, then a frame built from those -- and it should not count:
    `ODDS_SNAPSHOT` describes a table this repo writes under column names it chose, so the
    frame it validates is the parser's output rather than the shape a third party returned.
    A hop-limit is a blunt way to draw that line, and it draws it where the evidence is.
    """
    bindings = _payload_bindings(scope, fixtures)
    out: dict[str, set[str]] = {}
    for node in ast.walk(scope):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "validate" and node.args):
            continue
        names = _receiver_contracts(node.func.value, declared)
        if not names:
            continue
        got = _payload_names(node.args[0], fixtures)
        if not got:                          # `df = frame("x.json"); CONTRACT.validate(df)`
            for n in ast.walk(node.args[0]):
                if isinstance(n, ast.Name):
                    got |= bindings.get(n.id, set())
        if not got:
            continue                         # no payload behind it, so nothing to say
        for name in names:
            out.setdefault(name, set()).update(got)
    return out


@functools.cache
def _exercised() -> dict[str, frozenset[str]]:
    """Every contract the suite validates against a frozen payload, and which payloads.

    Scoped one function at a time, which is not a detail. `df` is bound in eight functions of
    `test_source_contracts.py` and names four different payloads across them; resolved over
    the module as a whole, every one of those reaches every call in the file, and `CFBD_GAMES`
    -- whose only fixture is hand-built -- comes back holding `nflverse_pbp.json` and reads as
    verified against a real capture. Measured on the version of this that did that.
    """
    declared, fixtures = _declared(), _fixtures()
    out: dict[str, set[str]] = {}
    for path in sorted(TESTS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for name, got in _exercised_in(node, declared, fixtures).items():
                out.setdefault(name, set()).update(got)
    return {name: frozenset(got) for name, got in out.items()}


def test_the_payload_scan_finds_both_kinds_of_evidence():
    """This half's premise. Both ends can quietly match nothing -- a moved fixture directory,
    a renamed helper, a `.validate` reached some new way -- and a scan that resolves no
    contract makes the two assertions below vacuously true, which is the exact defect the
    section was added for."""
    fixtures = _fixtures()
    captured = sorted(n for n, real in fixtures.items() if real)
    hand_built = sorted(n for n, real in fixtures.items() if not real)
    assert len(captured) >= 3, f"no captured payloads under {FIXTURES}: {sorted(fixtures)}"
    assert len(hand_built) >= 2, f"no hand-built payloads under {FIXTURES}: {sorted(fixtures)}"

    seen = _exercised()
    assert {"CFBD_GAMES", "CFBD_LINES", "PBP", "SCHEDULES"} <= set(seen), (
        f"the scan resolved {sorted(seen)}; it has stopped seeing contracts that are "
        f"demonstrably validated against a frozen payload in tests/contracts/")
    # Naming four was not enough. Five contracts resolved on 2026-09-05 and FF_OPPORTUNITY
    # was the fifth, so a resolver that stopped following one form could drop it and still
    # satisfy the line above -- and a contract whose evidence disappears is exactly a
    # contract whose flag has nothing left to disagree with. A floor, because the resolved
    # set can only be grown by wiring more evidence, never shrunk by accident -- so it moves
    # up when evidence is added, and #33 moved it from five to eight by freezing a capture of
    # the rankings archive, the injury report and snap counts.
    assert len(seen) >= 8, (
        f"the scan resolved {len(seen)} contracts ({sorted(seen)}), down from the eight "
        f"measured on 2026-09-05; evidence has gone missing rather than been added, so "
        f"some contract's declaration is no longer being checked against anything")
    assert seen["CFBD_GAMES"] == frozenset({"cfbd_games.synthetic.json"}), (
        f"CFBD_GAMES resolved to {sorted(seen['CFBD_GAMES'])}; it should hold its own "
        f"payload, not every payload named in the file that validates it")


def test_a_contract_only_ever_checked_against_a_hand_built_shape_says_it_is_unverified():
    """The flip this exists to catch. Both CFBD contracts are validated only against payloads
    written by hand from CFBD's documentation, so `verified_against_live=True` on either is a
    claim the tree contradicts -- and it used to be a claim nothing read.

    `False` exactly, not merely "not True". The tree answers this question here, so a
    declaration that says `None` -- unmeasured -- while a hand-built payload sits behind it
    is throwing away the more useful of the two sentences a violation can carry."""
    fixtures, wrong = _fixtures(), []
    for name, seen in sorted(_exercised().items()):
        flag = getattr(contracts, name).verified_against_live
        if not any(fixtures[p] for p in seen) and flag is not False:
            wrong.append(f"{name} (verified_against_live={flag!r}, "
                         f"checked only against {', '.join(sorted(seen))})")
    assert not wrong, (
        f"every frozen payload these are checked against was hand-built from documentation, "
        f"so each must declare `verified_against_live=False`: {wrong}. That flag is what "
        f"tells the reader of a red build whether to suspect the declaration or the source, "
        f"so a contract that has only ever met a shape we guessed must say so. If a real "
        f"capture now exists, replace the `.synthetic` fixture with it -- see "
        f"tests/golden/fixtures/README.md.")


def test_a_contract_checked_against_a_real_capture_does_not_call_itself_a_guess():
    """The other direction, which goes stale the quieter way: a fixture gets replaced with a
    real capture and the flag it was written to explain is left behind. `None` fails here for
    the same reason `False` does -- a capture is an answer, and the declaration should carry
    it rather than say nobody looked."""
    fixtures, wrong = _fixtures(), []
    for name, seen in sorted(_exercised().items()):
        real = sorted(p for p in seen if fixtures[p])
        flag = getattr(contracts, name).verified_against_live
        if real and flag is not True:
            wrong.append(f"{name} (verified_against_live={flag!r}, "
                         f"checked against {', '.join(real)})")
    assert not wrong, (
        f"a captured payload proves these have met a real response, so each must declare "
        f"`verified_against_live=True`: {wrong}. Every violation from a contract that does "
        f"not carries a note about the declaration being the likelier suspect, which sends "
        f"the reader of a red build to the wrong one.")


def test_a_contract_with_no_evidence_at_all_cannot_claim_it_has_met_live_data():
    """**The gap #66 was opened for.** The two tests above only constrain contracts the
    evidence resolves, which was five of eleven on 2026-09-05. The other six resolved to no
    frozen payload at all and the flag still defaulted to *verified*, so a contract written
    from documentation -- exactly like the two CFBD ones this whole section exists for --
    could be declared and nothing would go red.

    The fix is not to guess at those six. Nobody has measured whether the unexercised
    nflverse contracts have ever met a live response, and `None` says exactly that: it is
    the default, so silence now reads as silence rather than as a claim. Moving a contract
    off `None` means validating it against a frozen payload, not editing a list here."""
    seen = _exercised()
    wrong = []
    for name in sorted(_declared()):
        flag = getattr(contracts, name).verified_against_live
        if name not in seen and flag is not None:
            wrong.append(f"{name} (verified_against_live={flag!r})")
    assert not wrong, (
        f"claims to know whether it has met live data, and no frozen payload in the tree "
        f"says either way: {wrong}. Nothing in tests/ validates these contracts against a "
        f"fixture, so the answer is unmeasured and the declaration must say so with `None` "
        f"-- the state that exists precisely so an unmeasured contract is not collapsed "
        f"into one known to be hand-built. To claim otherwise, validate it against a frozen "
        f"payload under tests/golden/fixtures/ and let this scan see it.")


def test_a_violation_from_an_unverified_contract_names_the_likelier_suspect():
    """The behaviour the flag buys, which nothing asserted. An empty frame is enough to fail
    any of them -- what is being read is the sentence, not the failure."""
    unverified = [n for n in sorted(_declared())
                  if getattr(contracts, n).verified_against_live is False]
    assert unverified, ("no contract declares itself unverified, so this asserts nothing; "
                        "both CFBD endpoints were written from documentation")
    for name in unverified:
        with pytest.raises(contracts.ContractViolation) as raised:
            getattr(contracts, name).validate(pl.DataFrame())
        assert "suspect the declaration before the source" in str(raised.value), (
            f"{name} has never met a live response and its violation does not say so")


def test_a_violation_from_an_unmeasured_contract_does_not_borrow_the_guess_sentence():
    """The distinction the third state exists for, read where a person actually meets it.

    "Written from documentation" is a claim about how a contract was authored, and it is true
    of the two CFBD ones. It is not known to be true of the six nothing validates against a
    payload -- nobody has measured those -- so a violation from one must not say it. Nor may
    it say nothing, which is what a flag defaulting to *verified* did: a reader of that red
    build was told the source is the only suspect."""
    unmeasured = [n for n in sorted(_declared())
                  if getattr(contracts, n).verified_against_live is None]
    assert unmeasured, ("no contract declares its provenance unmeasured, so this asserts "
                        "nothing; six resolved to no frozen payload on 2026-09-05")
    for name in unmeasured:
        with pytest.raises(contracts.ContractViolation) as raised:
            getattr(contracts, name).validate(pl.DataFrame())
        message = str(raised.value)
        assert "written from documentation" not in message, (
            f"{name}'s provenance is unmeasured, and its violation asserts it was written "
            f"from documentation -- which is the CFBD contracts' story, not a measurement "
            f"anyone made about this one")
        assert "nothing in this repo records" in message, (
            f"{name} has no frozen payload behind it and its violation does not say so, so "
            f"the reader of a red build is left to suspect the source alone")


def _evidence(source: str) -> dict[str, set[str]]:
    """Run the payload resolver over a planted module. The resolver itself, not a restatement."""
    import textwrap
    out: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for name, got in _exercised_in(node, _declared(), _fixtures()).items():
                out.setdefault(name, set()).update(got)
    return out


def test_a_hand_built_payload_is_evidence_only_of_the_documented_shape():
    assert _evidence("""
        from hub.contracts import CFBD_GAMES
        def test_documented_shape():
            CFBD_GAMES.validate(frame("cfbd_games.synthetic.json"))
    """) == {"CFBD_GAMES": {"cfbd_games.synthetic.json"}}


def test_a_payload_reached_through_a_local_and_a_dropped_volume_floor_still_counts():
    """The local binding every captured source reaches its payload through, and the dropped
    volume floor two of them add. A resolver that missed either would report those sources as
    having no evidence at all, which constrains nothing."""
    assert _evidence("""
        from hub.contracts import PBP
        def test_real_slice():
            df = frame("nflverse_pbp.json").with_columns(pl.col("week").cast(pl.Int32))
            shape_only(PBP).validate(df)
    """) == {"PBP": {"nflverse_pbp.json"}}


def test_a_payload_does_not_leak_across_the_functions_that_rebind_it():
    """**The hole a module-wide scan opens.** `df` names four different payloads across eight
    functions of `test_source_contracts.py`; resolved module-wide, `CFBD_GAMES` came back
    holding the real play-by-play capture and read as verified -- the guard agreeing with the
    flip it exists to catch."""
    assert _evidence("""
        from hub.contracts import CFBD_GAMES, PBP
        def test_documented_shape():
            df = frame("cfbd_games.synthetic.json")
            CFBD_GAMES.validate(df)
        def test_real_slice():
            df = frame("nflverse_pbp.json")
            PBP.validate(df)
    """) == {"CFBD_GAMES": {"cfbd_games.synthetic.json"}, "PBP": {"nflverse_pbp.json"}}


def test_a_frame_assembled_by_hand_is_not_the_payload_it_was_read_from():
    """`ODDS_SNAPSHOT`'s columns are ones this repo chose, and the frame it validates is what
    the parser produced from the payload rather than the payload's own shape. Counting that
    as the contract meeting a response would be the tautology again, one indirection out."""
    assert _evidence("""
        from hub.contracts import ODDS_SNAPSHOT
        def test_parses_to_the_lines_table_shape():
            events = load("odds_spreads.synthetic.json")
            rows = [{"game_id": "g", "close_spread": median_home_spread(e)} for e in events]
            ODDS_SNAPSHOT.validate(pl.DataFrame(rows))
    """) == {}


def test_a_string_that_names_no_frozen_payload_is_not_evidence():
    """Dropping `.synthetic` from the string is not how a contract becomes verified; the
    fixture has to exist under that name, which means someone replaced it with a capture."""
    assert _evidence("""
        from hub.contracts import CFBD_GAMES
        def test_documented_shape():
            CFBD_GAMES.validate(frame("cfbd_games.json"))
    """) == {}


def test_an_unresolvable_frame_counts_for_nothing_rather_than_everything():
    """Same rule as the half above: an indirection this cannot follow reads as no evidence,
    which constrains nothing, rather than as evidence of whatever was nearby."""
    assert _evidence("""
        from hub.contracts import PBP
        def test_real_slice(load_frame):
            PBP.validate(load_frame("nflverse_pbp.json").pipe(reshape))
    """) == {"PBP": {"nflverse_pbp.json"}}
