"""One way to declare a guard, in every language this repo ships.

Three harnesses prove guards by deleting them -- `tests/contracts/test_guards_are_load_bearing.py`
for the Python source, `tests/unit/test_preflight.py` for the shell gate, and
`tests/contracts/test_dashboard_escapes.py` for the page's script. They were written in
parallel from the same brief and converged on the thing that matters (a mutant that no longer
parses is a harness error, never a pass) while diverging on the grammar, the record type, the
excision helper and the way "the child run went red" is read. This module is the part that
does not have to differ. What genuinely does differ -- how a mutant is *run* -- stays in each
harness, because Python imports a mutated tree, bash gets a mutated script, and node gets a
mutated page, and an abstraction over those three would hide more than it removed.

**Declaring a guard.** Wrap it where it lives, in that language's line comment:

    # GUARD name-in-kebab-case [unit/test_thing.py]: one line on what it protects.
    ...
    # /GUARD

    // GUARD name-in-kebab-case [contracts/test_page.py]: one line on what it protects.
    ...
    // /GUARD

The marker sits with the code for the reason ADR-0006 keeps fitted constants with their
provenance: a declaration that lives away from the thing it declares is one that stops
matching it.

**The bracket, by one rule.** It names the tests that must go red when the block is deleted,
written as pytest selectors *relative to `tests/`*, separated by spaces. A selector is a file
(`unit/test_provenance.py`) or a node id (`unit/test_preflight.py::test_a_clean_repo_passes`);
both are what you would hand pytest, so there is nothing to translate when reading one. It
was three rules before issue #65 -- a path here, a `-k` expression there, and nothing at all
on the page -- which is one syntax with three meanings, the failure this whole habit exists
to catch.

The bracket is omitted in exactly one place, and it is a derivation rather than a fourth
meaning: guards in `src/` whose tests follow the `src/hub/<x>.py` -> `tests/**/test_<x>.py`
filename convention leave it off, and the Python harness derives the same selector from the
filename. Derived beats written down when a convention exists, because a path in a comment is
a second thing to keep in step; where no convention exists (one shell script, six guards; one
page) the selectors are written and the harness's control run fails loudly if one stops
resolving.

**Reading a child run.** `outcome` parses pytest's own short summary rather than searching its
output for words. A child that goes red because it crashed at collection proves nothing about
the guard, and "the output contains 'error'" is satisfied by any traceback that mentions one.
`Outcome.failed` is the evidence; `Outcome.errors` without it is a harness error.
"""
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# The flags every child run needs for `outcome` to be able to read it: `-rfE` asks for the
# short summary of failures *and* errors, which is what tells the two apart.
CHILD_FLAGS = ("-q", "-rfE", "-p", "no:cacheprovider")


def marker(verb: str, comment: str) -> re.Pattern[str]:
    """The block grammar for one verb (`GUARD`, `UNPROVED`) in one comment syntax (`#`, `//`).

    One regex, built twice per language instead of copied three times. `name` is kebab-case
    so it can be a pytest parametrize id unchanged; `selectors` is the optional bracket; `why`
    is the one-line sentence, which must describe what *this block's own excision* shows.
    """
    c = re.escape(comment)
    return re.compile(
        rf"^[ \t]*{c} {verb} (?P<name>[a-z0-9][a-z0-9-]*)"
        rf"(?: \[(?P<selectors>[^\]]+)\])?: (?P<why>[^\n]+)\n"
        rf"(?P<body>.*?)"
        rf"^[ \t]*{c} /{verb}[^\n]*\n",
        re.S | re.M,
    )


@dataclass(frozen=True)
class Guard:
    """One declared guard: what it protects, where its markers sit, and what must notice.

    Frozen so it can be a parametrize argument, and `__repr__` is the name so `ids=repr`
    prints `credit-floor-refuses` rather than a dataclass dump.
    """

    name: str
    why: str
    span: tuple[int, int]
    selectors: tuple[str, ...] = ()
    path: Path | None = None

    def __repr__(self) -> str:                          # the parametrize id
        return self.name


def declared(text: str, pattern: re.Pattern[str], path: Path | None = None) -> list[Guard]:
    """Every guard `pattern` finds in `text`, in the order they appear."""
    return [
        Guard(
            name=m.group("name"),
            why=m.group("why"),
            span=m.span(),
            selectors=tuple((m.group("selectors") or "").split()),
            path=path,
        )
        for m in pattern.finditer(text)
    ]


def excise(text: str, guard: Guard) -> str:
    """`text` with that guard's block removed, markers and all."""
    lo, hi = guard.span
    return text[:lo] + text[hi:]


def selector_file(selector: str, tests_root: Path) -> Path:
    """The file a selector points at, so a stale one can be reported rather than silently run."""
    return tests_root / selector.split("::")[0]


def selector_args(selectors: tuple[str, ...] | list[str], tests_root: Path) -> list[str]:
    """Selectors as pytest arguments: relative to `tests/` in the marker, absolute on the
    command line, so a child run started from anywhere resolves them the same way."""
    args = []
    for selector in selectors:
        path, sep, rest = selector.partition("::")
        args.append(f"{tests_root / path}{sep}{rest}")
    return args


# `FAILED tests/x.py::test_y[param] - AssertionError` / `ERROR tests/x.py`, from `-rfE`.
_SUMMARY = re.compile(r"^(?P<verdict>FAILED|ERROR) (?P<nodeid>\S+)", re.M)


def _name(nodeid: str) -> str:
    """`tests/x.py::test_y[param]` -> `test_y`. The base name, so a parametrized failure and
    an unparametrized one are the same test to a caller comparing sets."""
    return nodeid.split("::")[-1].split("[")[0]


@dataclass(frozen=True)
class Outcome:
    """What a child pytest run actually reported, parsed rather than matched for substrings.

    The distinction this exists for: a red run means the guard is load-bearing *only* if a
    test failed. A run that errored at collection -- because the excision left source that
    imports, parses, but cannot be set up -- is red for a reason that has nothing to do with
    the guard, and reading it as proof is the trap all three harnesses were built around.
    Until issue #65 one harness decided this by asking whether the word "error" appeared in
    the output, which any traceback satisfies.
    """

    returncode: int
    failed: frozenset[str]
    errors: frozenset[str]
    stdout: str

    @property
    def is_evidence(self) -> bool:
        """True when at least one test failed, which is the only red that proves anything."""
        return bool(self.failed)

    def why_not_evidence(self) -> str:
        """One line on why a red run proves nothing, for a harness-error message."""
        if self.errors:
            return (f"the child errored rather than failed ({sorted(self.errors)}), so the "
                    f"mutant broke collection or setup rather than tripping a test")
        if self.returncode == 5:
            return "the child selected no tests at all, so nothing was ever going to notice"
        return (f"the child reported no test failures at all (exit {self.returncode}), so "
                f"there is no failing test to point at")


def outcome(got: subprocess.CompletedProcess[str]) -> Outcome:
    """Read a child pytest run. Requires `CHILD_FLAGS`, or the summary is not there to read."""
    failed, errors = set(), set()
    for m in _SUMMARY.finditer(got.stdout):
        (failed if m.group("verdict") == "FAILED" else errors).add(_name(m.group("nodeid")))
    return Outcome(got.returncode, frozenset(failed), frozenset(errors), got.stdout)
