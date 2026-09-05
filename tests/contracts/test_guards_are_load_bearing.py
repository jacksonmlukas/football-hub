"""Every guard this repo declares, proved by deleting it and watching the suite go red.

Eight guards were found incapable of firing in the week of 2026-09-04, and **every one of
them had passing tests**: the odds credit floor whose state lived in a gitignored tree, the
secret scanner whose patterns were BRE passed to `grep -E`, the contract guard that checked
for two tokens rather than a relationship, the workflow schedule check whose two greps were
never tied together, a publish guard that was literally dead code, a canary that proved one
pattern of five, a `verified_against_live` assertion that was a tautology, and a model
comparison whose store path could never be reached.

The common shape: **the test asserted the outcome the guard was meant to produce, rather
than that the guard produced it.** A guard that has been quietly deleted, or that never
matched anything, produces the same green suite.

Each was found by a person or an agent thinking to try the excision by hand. That is not a
habit, it is a run of luck, and it is what this file replaces.

**How to declare a guard.** Wrap it where it lives:

    # GUARD name-in-kebab-case: one line on what it protects.
    if not payload.get(count_key):
        ...
    # /GUARD

The grammar, the record type, the excision and the way a child run is read are all
`tests/guardlib.py`, shared with the shell gate's harness in `tests/unit/test_preflight.py`
and the page's in `tests/contracts/test_dashboard_escapes.py` -- three copies of one
convention until issue #65. Read that module for the bracket's one resolution rule; the only
thing this file adds to it is the derivation below, which is why the bracket is usually
absent here.

The marker sits with the code for the reason ADR-0006 keeps fitted constants with their
provenance: a declaration that lives away from the thing it declares is one that stops
matching it. Adding a guard means marking it; marking it means this file proves it.

**The trap this harness is built around.** Excising a block can leave source that does not
parse -- an `if` that was a function's only statement, say. Every test then fails at
collection, which looks exactly like "the guard is load-bearing" and is not. The first
hand-run excision in this repo hit precisely that and reported a misleading zero. So a
mutant that will not parse is a *harness* failure naming the guard's markers, never a pass.
"""
import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from guardlib import (
    CHILD_FLAGS,
    Guard,
    declared,
    excise,
    marker,
    outcome,
    selector_args,
    selector_file,
)

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"
TESTS = ROOT / "tests"

# `# GUARD <name> [<selectors>]: <why>` ... `# /GUARD`, at any indentation.
GUARD = marker("GUARD", "#")


def _selectors(guard: Guard) -> tuple[str, ...]:
    """The tests that must notice, derived when they can be so they cannot drift.

    `src/hub/season/roster.py` -> `tests/unit/test_roster.py`, and this repo also names some
    after the package -- `src/hub/fetch/odds.py` -> `tests/unit/test_fetch_odds.py` -- so the
    suffix form is tried too. A shared module whose protection is exercised through its
    callers has no such filename to derive from and says so in the marker instead:
    `# GUARD name [unit/test_x.py]: why`, the same selector-relative-to-`tests/` the other two
    harnesses write. That is a path in a comment, which is a second thing to keep in step, so
    it is the exception rather than the rule: the premise test below fails if it stops
    resolving. The alternative was a harness that reports a covered guard as dead, which is
    worse than a path that can go stale loudly.
    """
    if guard.selectors:
        return guard.selectors
    assert guard.path is not None
    stem = guard.path.stem
    for pattern in (f"tests/**/test_{stem}.py", f"tests/**/test_*_{stem}.py"):
        if found := sorted(ROOT.glob(pattern)):
            return (str(found[0].relative_to(TESTS)),)
    return ()


def _declared(root: Path = SRC) -> list[Guard]:
    out: list[Guard] = []
    for path in sorted(root.rglob("*.py")):
        out.extend(declared(path.read_text(), GUARD, path))
    return out


def _without(guard: Guard) -> str:
    """The module's source with that guard's block removed, markers and all."""
    assert guard.path is not None
    return excise(guard.path.read_text(), guard)


def _run_against(mutant_src: Path, selectors: tuple[str, ...]) -> subprocess.CompletedProcess:
    """Run the tests a guard names against a mutated copy of the source tree.

    `-o pythonpath=` and not the environment variable. `pyproject.toml` sets
    `pythonpath = ["src", "tests"]`, which pytest inserts at the front of `sys.path` ahead of
    anything `PYTHONPATH` contributes -- so the first version of this ran every mutant against
    the real source and reported all four guards green. The harness caught itself, which is
    the only reason this comment exists rather than a ninth dead guard.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", *selector_args(selectors, TESTS), "-x", *CHILD_FLAGS,
         "-o", f"pythonpath={mutant_src}"],
        cwd=ROOT, capture_output=True, text=True, timeout=300, env=os.environ.copy(),
    )


# --- the harness's own premise ------------------------------------------------
#
# A scan that matched nothing would make every assertion below vacuously true, which is the
# exact failure this file exists to catch. It would be a fine joke and a real defect.

def test_the_scan_finds_the_guards_that_exist():
    found = _declared()
    assert len(found) >= 4, f"only found {[g.name for g in found]}"
    names = {g.name for g in found}
    assert "last-good-not-blanked" in names, (
        "the publish guard is the one this whole habit came from; if it is unmarked, the "
        "marker syntax has drifted")


def test_every_guard_names_a_test_module_that_exists():
    missing = [(g.name, s) for g in _declared() for s in (_selectors(g) or ("",))
               if not s or not selector_file(s, TESTS).exists()]
    assert not missing, (
        f"declared guards whose selectors do not resolve: {missing}. A selector is a path "
        f"relative to `tests/`, or is derived by the convention `src/hub/<x>.py` -> "
        f"`tests/**/test_<x>.py` or `test_*_<x>.py`; add the module or fix the marker.")


def test_a_guard_that_cannot_be_removed_cleanly_is_a_harness_error(tmp_path):
    """The trap. Excising a block can leave source that will not parse, and every test then
    fails at collection -- which looks identical to a load-bearing guard. The first hand-run
    excision in this repo hit exactly that and reported a misleading zero."""
    mod = tmp_path / "hub" / "brittle.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(textwrap.dedent("""\
        def f(x):
            # GUARD only-statement: removing this leaves a function with no body.
            if x:
                return 1
            # /GUARD
    """))
    guard = _declared(tmp_path / "hub")[0]
    with pytest.raises(SyntaxError):
        ast.parse(_without(guard))


def test_the_harness_notices_a_guard_that_does_nothing(tmp_path):
    """The positive control. A guard whose removal changes no behaviour must be reported,
    or this file is itself the ninth dead guard."""
    mod = tmp_path / "hub" / "inert.py"
    mod.parent.mkdir(parents=True)
    mod.write_text(textwrap.dedent("""\
        def f(x):
            # GUARD does-nothing: asserts something already true.
            if x is None and x is not None:
                raise ValueError("unreachable")
            # /GUARD
            return x
    """))
    guard = _declared(tmp_path / "hub")[0]
    mutant = _without(guard)
    ast.parse(mutant)                                   # parses, so a red run means the test
    assert "unreachable" not in mutant
    # `f` behaves identically with and without it -- which is what a real run would surface
    # as "the target tests still pass".
    ns_with, ns_without = {}, {}
    exec(mod.read_text(), ns_with)
    exec(mutant, ns_without)
    assert ns_with["f"](3) == ns_without["f"](3) == 3


def test_a_child_that_errored_is_told_apart_from_one_that_failed(tmp_path):
    """The other trap, and the one all three harnesses now decide with the same code.

    A child run that goes red because it *crashed* -- at import, at collection, in a fixture
    -- proves nothing about the guard that was excised. Until issue #65 this file asked
    whether the word "error" appeared in the child's output, which any traceback mentioning
    one satisfies, and which the very next assertion (`returncode != 0`) made true anyway, so
    it never added a check at all. `guardlib.outcome` reads pytest's own short summary.

    Two real child runs, because the thing being proved is that pytest's output can be read
    -- a hand-written string would prove the regex matches the string it was written from.
    """
    (tmp_path / "test_it_fails.py").write_text("def test_x():\n    assert False\n")
    (tmp_path / "test_it_errors.py").write_text("raise RuntimeError('boom')\n")

    def run(name):
        return subprocess.run(
            [sys.executable, "-m", "pytest", str(tmp_path / name), *CHILD_FLAGS, "-p",
             "no:randomly", "--no-header", "-o", "addopts="],
            cwd=tmp_path, capture_output=True, text=True, timeout=120, env=os.environ.copy())

    failing, erroring = outcome(run("test_it_fails.py")), outcome(run("test_it_errors.py"))
    assert failing.is_evidence and failing.failed == {"test_x"}, failing.stdout
    assert not erroring.is_evidence, (
        f"a child that crashed at collection was read as a guard firing:\n{erroring.stdout}")
    assert erroring.errors, erroring.stdout
    assert erroring.returncode != 0, "the crashing child was not even red"
    assert "error" in erroring.stdout.lower(), (
        "the substring check this replaced would have been satisfied by this run, which is "
        "the point: red and full of the word 'error' is not a failing test")


# --- the habit ----------------------------------------------------------------

@pytest.mark.parametrize("guard", _declared(), ids=repr)
def test_removing_the_guard_turns_its_tests_red(guard, tmp_path):
    """Delete the guard, run the tests it names against the mutant, require failure.

    Three outcomes have to be told apart, and the middle one is the whole point. A test
    *failure* means the guard is load-bearing. A green run means nothing proves it fires. A
    child that goes red *without a test failing* -- a mutant that imports but errors during
    collection or setup -- is a harness error and evidence of nothing. Until issue #65 that
    last one was decided by asking whether the word "error" appeared in the output, which is
    a condition any traceback satisfies and which the very next assertion made true anyway;
    `guardlib.outcome` reads pytest's own short summary instead.
    """
    mutant = _without(guard)
    try:
        ast.parse(mutant)
    except SyntaxError as e:                            # pragma: no cover - marker error
        pytest.fail(f"removing guard {guard.name!r} from {guard.path.name} leaves source "
                    f"that will not parse ({e}). Widen the markers so the block stands "
                    f"alone -- as written, a red run would prove nothing.")

    tree = tmp_path / "src"
    for path in SRC.rglob("*.py"):
        dst = tree / "hub" / path.relative_to(SRC)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(mutant if path == guard.path else path.read_text())

    selectors = _selectors(guard)
    assert selectors, f"no test module derives from {guard.path.name}"
    got = outcome(_run_against(tree, selectors))
    if got.errors and not got.failed:                   # pragma: no cover - harness error
        pytest.fail(
            f"the mutant run for {guard.name!r} is red without a test failing, which is a "
            f"harness error and not evidence: {got.why_not_evidence()}.\n"
            + got.stdout[-2000:])
    assert got.is_evidence, (
        f"{' '.join(selectors)} still passes with guard {guard.name!r} removed, so nothing "
        f"proves it fires. It guards: {guard.why}\n"
        f"Either the tests assert the outcome rather than the guard, or the guard is dead. "
        f"({got.why_not_evidence()})")
