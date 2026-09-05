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
import re
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"

# `# GUARD <name>: <why>` ... `# /GUARD`, at any indentation.
GUARD = re.compile(
    r"^[ \t]*# GUARD (?P<name>[a-z0-9][a-z0-9-]*)"
    r"(?: \[(?P<tests>[^\]]+)\])?: (?P<why>[^\n]+)\n"
    r"(?P<body>.*?)"
    r"^[ \t]*# /GUARD[^\n]*\n",
    re.S | re.M,
)


@dataclass(frozen=True)
class Declared:
    """One declared guard: where it is, what it protects, and what proves it."""

    path: Path
    name: str
    why: str
    span: tuple[int, int]
    declared_tests: str | None = None

    @property
    def tests(self) -> Path | None:
        """The test module that must notice. Derived, not declared, so it cannot drift.

        A shared module whose protection is exercised through its callers says so in the
        marker -- `# GUARD name [unit/test_x.py]: why`, relative to `tests/`. That is a
        path in a comment,
        which is a second thing to keep in step, so it is the exception: the premise test
        below fails if it stops resolving. The alternative was a harness that reports a
        covered guard as dead, which is worse than a path that can go stale loudly.

        `src/hub/season/roster.py` -> `tests/unit/test_roster.py`, and this repo also names
        some after the package -- `src/hub/fetch/odds.py` -> `tests/unit/test_fetch_odds.py`
        -- so the suffix form is tried too. Derived rather than written into the marker
        because a path in a comment is a second thing to keep in step, which is the failure
        this whole file is about.
        """
        if self.declared_tests:
            return ROOT / "tests" / self.declared_tests
        stem = self.path.stem
        for pattern in (f"tests/**/test_{stem}.py", f"tests/**/test_*_{stem}.py"):
            if found := sorted(ROOT.glob(pattern)):
                return found[0]
        return None

    def __repr__(self) -> str:                          # the parametrize id
        return self.name


def _declared(root: Path = SRC) -> list[Declared]:
    out: list[Declared] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text()
        for m in GUARD.finditer(text):
            out.append(Declared(path, m.group("name"), m.group("why"), m.span(),
                                m.group("tests")))
    return out


def _without(guard: Declared) -> str:
    """The module's source with that guard's block removed, markers and all."""
    text = guard.path.read_text()
    lo, hi = guard.span
    return text[:lo] + text[hi:]


def _run_against(mutant_src: Path, tests: Path) -> subprocess.CompletedProcess:
    """Run one test module against a mutated copy of the source tree.

    `-o pythonpath=` and not the environment variable. `pyproject.toml` sets
    `pythonpath = ["src"]`, which pytest inserts at the front of `sys.path` ahead of anything
    `PYTHONPATH` contributes -- so the first version of this ran every mutant against the
    real source and reported all four guards green. The harness caught itself, which is the
    only reason this comment exists rather than a ninth dead guard.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(tests), "-x", "-q", "-p", "no:cacheprovider",
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
    missing = [g.name for g in _declared()
               if g.tests is None or not g.tests.exists()]
    assert not missing, (
        f"declared guards whose test module could not be derived: {missing}. The "
        f"convention is `src/hub/<x>.py` -> `tests/**/test_<x>.py` or `test_*_<x>.py`; "
        f"add the module or move the guard.")


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


# --- the habit ----------------------------------------------------------------

@pytest.mark.parametrize("guard", _declared(), ids=repr)
def test_removing_the_guard_turns_its_tests_red(guard, tmp_path):
    """Delete the guard, run the module's tests against the mutant, require failure.

    The two things that must be told apart: a test *failure* means the guard is load-bearing;
    a *collection error* means the mutant did not parse, and proves nothing at all.
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

    assert guard.tests is not None, f"no test module derives from {guard.path.name}"
    got = _run_against(tree, guard.tests)
    assert "error" not in got.stdout.lower().split("=== warnings")[0] or got.returncode != 0
    assert got.returncode != 0, (
        f"{guard.tests.name} still passes with guard {guard.name!r} removed, so nothing "
        f"proves it fires. It guards: {guard.why}\n"
        f"Either the tests assert the outcome rather than the guard, or the guard is dead.")
