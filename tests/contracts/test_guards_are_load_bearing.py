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
`tests/guardlib.py`, shared with the shell scan's harness in `tests/unit/test_preflight.py`
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
from dataclasses import dataclass
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


# --- the next refusal ---------------------------------------------------------
#
# Everything above proves the guards that *are* declared. Nothing above makes the next one
# get declared, and twenty-five are twenty-five because someone swept for them by hand on
# 2026-09-04. A registry that only a sweep can grow starts decaying the day the sweep ends,
# which is the shape of all eight dead guards that week: something that reads as protection
# while nothing keeps it true (#62).
#
# **Why this is not a scan for `raise`.** There are 84 `raise` statements under `src/hub` and
# 10 of them sit inside a declared guard. The other 74 are argument validation, domain
# signalling a caller catches (`NoLegalLineup`, `Infeasible`, `NoOverlap`) and `SystemExit`
# from a `main`. Asking for a marker or a written exemption on all 74 is a gate nobody
# finishes and everybody turns off, and this repo has the incident: `preflight_public.sh`
# carried a `noqa`-shaped exemption budget until it was replaced by named exceptions. A
# noisy gate is worse than none, because it is switched off *and* it was the last thing
# looking.
#
# `raise` is also the wrong unit for the case that prompted the ticket. The contract
# validator's five unmarked refusals do not raise -- they append to a list that one later
# `raise` turns into a `ContractViolation`. A scan for `raise` would have found none of them.
#
# **Why this is not a count.** A baseline of "25 declared guards, update deliberately" says
# a number is allowed to change, not which refusal is unmarked and why. `test_preflight.py`
# ran exactly that shape -- `outside <= 1` -- until 2026-09-05, and it read as "one unmarked
# guard is fine": the allowance was already spent on the gitleaks call, and marking or
# deleting that call would have freed the budget for the next unmarked guard to spend in
# silence. The same shape has bitten this repo three times, most recently a canary comment
# claiming six steps over five.
#
# **What this is instead: a refusal vocabulary, per module.** The shell scan has one exact
# way to refuse -- `fail=1` -- so its scan is a substring and has no judgement in it at all.
# Python has no repo-wide equivalent, but a *module* can have one, and where it does the
# judgement is made once, in writing, rather than per `raise` by whoever is reading.
# `WATCHED` below records the modules where that is true, the shapes that make a statement a
# refusal there, and why. Inside a watched module every one must sit in a `# GUARD` (proved
# by excision above) or a `# UNPROVED` (explained, with the test that expires the exemption)
# -- the two verbs already shared with the shell scan through `guardlib.marker`, not a third
# dialect.
#
# **Read off the AST, not off the text.** The shell scan greps, because `bash` leaves nothing
# better to read; here the first cut grepped too and went red on this module's own
# docstring, which names both refusal shapes in a sentence explaining that they are watched.
# That is the fires-on-its-own-explanation failure, reached in one edit, and patching it with
# a comment-line exclusion would only have moved it into the next docstring or error string.
# `ast` knows a call from a sentence about one, so a refusal is located by shape and prose is
# free to describe it. The exclusion the shell scan needs does not exist here.
#
# **What it costs, stated rather than discovered.** The gate sees only what `WATCHED` lists.
# A new refusal in an unwatched module is as silent today as every refusal was yesterday, so
# this narrows the hole rather than closing it. That is deliberate and it is the honest
# trade: a scope can only under-cover, and adding to it strengthens the gate, whereas a
# budget can be spent. `contracts.py` is the entry it starts with because it is the highest-
# leverage refusal code in the repo -- fourteen contracts, every fetch boundary, one function
# -- and because it is the module whose declared dtypes were "read by nothing" for months by
# its own docstring's account. Widening it is one line per module and one honest sentence.
#
# **The second entry, and what it still leaves out.** `fetch/nflverse.py` is watched for
# `ContractViolation` and `WideFrameRefused`. Each is raised in this repo for one reason --
# a frame reached a boundary that nothing can vouch for, or a frame was asked for in a shape
# this module will not return -- and `grep -rn "except ContractViolation" src tests scripts`,
# repeated for `WideFrameRefused`, finds no catcher either time, so for neither is there a
# caller to whom one is ordinary control flow. That is the property that makes the judgement
# safe to make once here, the same one `contracts.py` has.
#
# The module raises three types and every one of the eight is a refusal: `WideFrameRefused`
# six times, `UnattributedPoints` once, and this `ContractViolation`. `WideFrameRefused` was
# left out when this entry was written, because two of its six could not be marked that day
# and marking the other four would have put a green tick beside the two that could not.
# Both are dealt with now (#62), and how is the useful half:
#
#   * `load_rankings`'s unknown-page refusal was not load-bearing under the tests as written.
#     Deleting it by hand on 2026-09-06 left all 77 of `unit/test_fetch_nflverse.py` green,
#     because `_raw_ff_rankings` then refuses with the same exception type and a message that
#     still contains the two page names the test asserted on. That is the "asserted the
#     outcome, not that this guard produced it" shape from the week of 2026-09-04, found here
#     rather than by luck. The test now matches `load_rankings`'s own wording and asserts that
#     the archive was never reached for, so the two refusals are told apart by *where* each
#     fires rather than by what it says.
#   * `_raw_ff_rankings`'s own refusal is what stops a bad partition key reaching
#     `nfl.load_ff_rankings`, so a mutant without it makes the child run fetch the live 1.83M
#     -row archive. That is a fixture rather than an argument: `no_live_rankings` in
#     `unit/test_fetch_nflverse.py` shuts nflreadpy's door, and the excision then proves the
#     guard against a call that never leaves the machine. `UNPROVED_HERE` stays empty, which
#     is the honest outcome -- an exemption nobody had to spend.
#
# `UnattributedPoints` is still unwatched and is the next shape to add. Saying so is the
# point of the entry rather than an apology for it: an under-covering scope says which
# refusals it is not reading and why, where a budget says only that a number moved.

PY_UNPROVED = marker("UNPROVED", "#")

# No Python refusal is exempt today. Named as an empty set rather than left out so that the
# first one has somewhere to go that is not "delete the scan": the mechanism exists, it is
# the shell scan's, and the test below fails if a block appears that this set does not know.
UNPROVED_HERE: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Watched:
    """A module whose refusals are guards by default, and the shapes that identify one.

    `collects` names the local lists whose entries become a refusal -- appending to one is
    the contract validator's way of saying no. `raises` names the exception types raised to
    refuse. Both, because a check that raised directly rather than appending would otherwise
    be a refusal neither shape sees.
    """

    path: Path
    collects: tuple[str, ...]
    raises: tuple[str, ...]
    why: str


def _entry(stem: str) -> Watched:
    """One entry by module name, for the controls that need a particular module's shape.

    `WATCHED[0]` was fine while there was one entry and becomes a silent mis-aim the moment
    there are two: a control written for `contracts.py`'s collector would start running
    against whichever entry happens to be first, and pass for the wrong reason.
    """
    (found,) = [w for w in WATCHED if w.path.stem == stem]
    return found


WATCHED = (
    Watched(
        path=SRC / "contracts.py",
        collects=("problems",),
        raises=("ContractViolation",),
        why=("Both of this module's verbs refuse and neither does anything else: every entry "
             "appended to `problems` becomes a `ContractViolation` at the foot of "
             "`Contract.validate`, and the one raise outside it -- `conform` meeting a "
             "consumer that named a column the contract does not declare -- is the same kind "
             "of statement, reached before any frame is read. So there is no ordinary control "
             "flow here for a scan to mistake for a guard, which is what makes the judgement "
             "safe to make once, in this entry, rather than per statement by whoever is "
             "reading. `Normalisation` is the module's one non-refusing shape and it collects "
             "and raises nothing, so it is invisible to this scan by construction rather than "
             "by exemption."),
    ),
    Watched(
        path=SRC / "fetch" / "nflverse.py",
        collects=(),
        raises=("ContractViolation", "WideFrameRefused"),
        why=("This is the loader every nflverse number in the repo comes through, so a row "
             "that gets past it is a row in a published board. A `ContractViolation` raised "
             "here says the frame cannot be vouched for and is not being served; a "
             "`WideFrameRefused` says it was asked for in a shape this module will not "
             "return. Nothing in `src`, `tests` or `scripts` catches either, so there is no "
             "caller for whom one is ordinary control flow, which is what makes the "
             "judgement safe to make once in this entry. `collects` is empty because this "
             "module has no accumulator -- it refuses by raising, at the statement that "
             "found the problem. Still narrower than the module's full vocabulary: "
             "`UnattributedPoints` is unwatched, and the note above says why."),
    ),
)


def _refusals(text: str, watched: Watched) -> list[tuple[int, int, str]]:
    """Every refusal in `text`, as (character offset, line number, source line).

    Located by shape rather than by substring. A grep for `problems.append(` also matches the
    docstring that explains it is watched, which is how the first version of this went red on
    its own explanation; `ast` tells a call from a sentence about one, so the module is free
    to describe what it refuses.
    """
    starts, offset = [], 0
    for line in text.splitlines(keepends=True):
        starts.append(offset)
        offset += len(line)
    lines = text.splitlines()

    found = []
    for node in ast.walk(ast.parse(text)):
        lineno = None
        if isinstance(node, ast.Call):
            fn = node.func
            if (isinstance(fn, ast.Attribute) and fn.attr == "append"
                    and isinstance(fn.value, ast.Name) and fn.value.id in watched.collects):
                lineno = node.lineno
        elif isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name) and exc.id in watched.raises:
                lineno = node.lineno
        if lineno is not None:
            found.append((starts[lineno - 1], lineno, lines[lineno - 1].strip()))
    return sorted(found)


def _undeclared(text: str, watched: Watched) -> list[str]:
    """The refusals in `text` inside neither a `# GUARD` nor an `# UNPROVED` block.

    Takes the text rather than a path so the controls below can splice a refusal in and watch
    this name it. Reported by line and by source, never as a count: the report has to say
    *which* refusal is undeclared, or it is the budget this replaces wearing a longer message.
    """
    covered: set[int] = set()
    for pattern in (GUARD, PY_UNPROVED):
        for m in pattern.finditer(text):
            covered.update(range(*m.span()))
    return [f"line {lineno}: {src}"
            for offset, lineno, src in _refusals(text, watched) if offset not in covered]


def test_every_refusal_in_a_watched_module_is_declared():
    """The gate. A seventh contract check cannot land the way the first six did."""
    for watched in WATCHED:
        undeclared = _undeclared(watched.path.read_text(), watched)
        assert undeclared == [], (
            f"these refusals in {watched.path.name} sit inside no # GUARD and no # UNPROVED "
            f"block, so nothing proves they fire:\n  " + "\n  ".join(undeclared) + "\n"
            f"That module is watched because: {watched.why}\n"
            f"Wrap each in `# GUARD name: why` and let the excision harness above prove it, "
            f"or -- only if it genuinely cannot be proved -- declare it as an `# UNPROVED` "
            f"block naming the test that expires the exemption, and add its name to "
            f"UNPROVED_HERE.")


def test_the_watched_shapes_still_find_the_module_they_watch():
    """The premise, and the failure this whole file exists to catch.

    Rename `problems` to `issues` and the gate above finds nothing, reports no undeclared
    refusals, and stays green forever over a module it has stopped reading. A scan that has
    gone blind is indistinguishable from a module with nothing to find, which is the
    "passing tests, dead guard" shape from the week of 2026-09-04 -- so the denominator is
    asserted rather than assumed, and each declared shape has to account for itself.

    *Declared* shape. A module that refuses by raising and never accumulates -- which is
    every refusal in `fetch/nflverse.py` -- declares `collects=()`, and an empty tuple has
    nothing to account for. What must never pass is an entry declaring no shape at all: that
    reads a module and finds nothing by construction, which is this same failure written into
    the entry rather than arrived at by drift."""
    for watched in WATCHED:
        text = watched.path.read_text()
        alone = [(f"{field}={names!r}", probe)
                 for field, names, probe in (
                     ("collects", watched.collects,
                      Watched(watched.path, watched.collects, (), watched.why)),
                     ("raises", watched.raises,
                      Watched(watched.path, (), watched.raises, watched.why)))
                 if names]
        assert alone, (
            f"{watched.path.name} is watched for no refusal shape at all, so the gate reads "
            f"it and finds nothing however many refusals it grows. An entry declares at "
            f"least one of `collects` and `raises`.")
        for shape, probe in alone:
            assert _refusals(text, probe), (
                f"nothing in {watched.path.name} matches its declared {shape} any more, so "
                f"the gate is reading less than it claims and would stay green over any "
                f"number of undeclared refusals. Re-point WATCHED at whatever that module "
                f"refuses with now.")


# Every (module, shape) pair `WATCHED` declares. The controls below are parametrized over
# these rather than written against one entry, because an entry nothing splices into is an
# entry nobody has watched read anything -- which is what `WATCHED[0]` quietly became the
# moment there were two.
COLLECT_SHAPES = [(w, name) for w in WATCHED for name in w.collects]
RAISE_SHAPES = [(w, exc) for w in WATCHED for exc in w.raises]


@pytest.mark.parametrize(("watched", "collector"), COLLECT_SHAPES,
                         ids=[f"{w.path.stem}-{n}" for w, n in COLLECT_SHAPES])
def test_a_new_undeclared_refusal_is_named_immediately(watched, collector):
    """The positive control. Splice a refusal onto the watched module and require the report
    to say what it is and where -- not "1 refusal is undeclared", which is the count restated.

    Without this the gate only ever runs against a module that already satisfies it, which is
    the same vacuum as a canary proving one pattern of five."""
    spliced = (watched.path.read_text()
               + f'\ndef _later({collector}):\n    {collector}.append("x")\n')
    got = _undeclared(spliced, watched)
    assert len(got) == 1, f"the splice was not the only undeclared refusal: {got}"
    assert f'{collector}.append("x")' in got[0], (
        f"the report does not say what it is: {got[0]}")
    assert got[0].startswith("line "), f"nor where it is: {got[0]}"


@pytest.mark.parametrize(("watched", "exc"), RAISE_SHAPES,
                         ids=[f"{w.path.stem}-{e}" for w, e in RAISE_SHAPES])
def test_a_check_that_raises_instead_of_appending_is_caught_too(watched, exc):
    """The hole the second shape closes. A seventh check could refuse directly rather than
    adding to `problems`, and a gate that only knew the list would not see it.

    It is also the *only* shape `fetch/nflverse.py` has: that module keeps no accumulator and
    refuses at the statement that found the problem, so this parametrization is what proves
    its entry reads it at all."""
    spliced = (watched.path.read_text()
               + f'\ndef _later(df):\n    raise {exc}("straight to the exit")\n')
    got = _undeclared(spliced, watched)
    assert len(got) == 1 and exc in got[0], got
    assert got[0].startswith("line "), f"the report does not say where it is: {got[0]}"


def test_prose_naming_a_refusal_is_not_read_as_one():
    """Why this reads the AST. `Contract.validate`'s own docstring names both watched shapes
    in the sentence explaining that they are watched, and the first version of this gate went
    red on it -- a gate firing on its own explanation, which is the shape that gets a gate
    deleted rather than satisfied. Kept as a test because the cheap fix (skip comment lines,
    as the shell scan must) passes today and breaks on the next docstring or error string."""
    watched = _entry("contracts")
    spliced = (watched.path.read_text()
               + '\ndef _later():\n    """Mentions problems.append( and ContractViolation."""\n'
                 '    return "raise ContractViolation( in a string, too"\n')
    assert _undeclared(spliced, watched) == [], (
        "a sentence about a refusal was read as a refusal, so the gate now fires on any "
        "attempt to document it")


def test_an_unproved_block_declares_a_refusal_the_same_way_a_guard_does():
    """The explained half of "marked or explained", proved rather than asserted.

    A refusal that cannot be proved by excision -- the shell scan's optional gitleaks call is
    the one live example -- has to have somewhere to go, or the person who meets one deletes
    the gate instead. Both verbs come from `guardlib.marker`, so this checks the second one is
    actually wired here rather than only in `test_preflight.py`."""
    watched = _entry("contracts")
    text = watched.path.read_text()
    naked = text + '\ndef _later(problems):\n    problems.append("x")\n'
    covered = (text + '\ndef _later(problems):\n'
                      '    # UNPROVED later-check [contracts/test_contracts.py]: why not.\n'
                      '    problems.append("x")\n'
                      '    # /UNPROVED\n')
    assert len(_undeclared(naked, watched)) == 1
    assert _undeclared(covered, watched) == [], (
        "an # UNPROVED block does not cover the refusal it wraps, so the only way past this "
        "gate is a # GUARD -- and a refusal that cannot be proved has nowhere to go.")


def test_no_python_refusal_claims_an_exemption_it_has_not_argued_for():
    """The exemptions are named, not counted -- the distinction `test_preflight.py` drew when
    it replaced `outside <= 1` with `UNPROVED_HERE`. An `# UNPROVED` block appearing in a
    watched module without being added here is a new unproved refusal that nobody argued
    for, which is the budget shape by another route."""
    for watched in WATCHED:
        found = {g.name for g in declared(watched.path.read_text(), PY_UNPROVED)}
        assert found <= UNPROVED_HERE, (
            f"{watched.path.name} declares {sorted(found - UNPROVED_HERE)} as unprovable by "
            f"excision, which this file does not know about. A refusal that cannot be proved "
            f"is not a spare slot: say which one it is and why, in both places.")


def test_every_unproved_block_names_a_test_that_expires_it():
    """An exemption with no expiry is a permanent one. For an `# UNPROVED` the bracket names
    the test that fails the day the exemption stops holding -- the same one meaning the
    bracket has had since #65 -- so a stale one is reported here rather than never."""
    stale = [(g.name, s)
             for w in WATCHED for g in declared(w.path.read_text(), PY_UNPROVED, w.path)
             for s in (g.selectors or ("",))
             if not s or not selector_file(s, TESTS).exists()]
    assert not stale, (
        f"these # UNPROVED brackets do not resolve to a file under tests/: {stale}. The "
        f"bracket is one or more pytest selectors relative to `tests/` -- see "
        f"tests/guardlib.py.")


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
