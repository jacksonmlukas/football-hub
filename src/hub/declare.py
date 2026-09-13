"""A fitted constant declares its own digest coverage where it is written (#253).

ADR-0006 keeps a fitted constant beside its provenance and hashes it into the model version
from there. What decided *which* constants were hashed grew into four mechanisms, each added
to patch the last: a wholesale module list keyed one way (`FITTED_MODULES`), an extras list
keyed another (`FITTED_EXTRA`), an exclusion list keyed a third (`NOT_IN_DIGEST`), and a
module-level opt-out string copied into twenty-six modules and read only by a test --
plus the AST rule that made every assignment in a listed module a constant, and the
float-only scan in the test that keyed by file stem. A module author had to learn all of
them to declare one number.

Now the number says. Three spellings, one question asked once per constant, at the line
that defines it:

    TALENT_CV = fitted(0.32)                  # a measurement: an interval and a write-up
    MIN_GAMES = chosen(10)                    # a stated choice a prediction reads
    _EIG_FLOOR = not_an_input(1e-8, "...")    # a number no prediction can reach, and why

`fitted` and `chosen` are both **covered** -- coverage is owed by anything that changes a
prediction, measured or not (`FLEX_SHARES`, #184) -- and differ only in what they claim
about the number. `not_an_input` is the one way out, and it carries its argument, because
an exclusion is a decision on the record and not a module quietly falling off a list.

The digest is derived by walking the declarations: `declarations()` reads every module
under `hub` once per process and finds the module-level names assigned through one of the
three, and `covered()` reads their live values. The three functions are the identity at
run time; what they add is the fact that the source can be read for them. The values are
read live rather than captured at the call, so a test that rebinds a module attribute --
the way every digest test did before #253 -- still moves the digest, and so that
`fitted_digest` can be asked what it would be with one constant elsewhere without touching
any module at all.

A leaf: it imports the standard library and nothing from `hub`, so `hub.config` and every
module holding a constant can take a spelling from it without a cycle.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

SRC = Path(__file__).resolve().parent
PACKAGE = "hub"


def fitted(value: T) -> T:
    """A measurement with a confidence interval and a write-up in `docs/`; in the digest."""
    return value


def chosen(value: T) -> T:
    """A stated choice that a prediction reads; in the digest, and not claiming to be a
    measurement. The shape of a random draw, a pin, a threshold chosen from data."""
    return value


def not_an_input(value: T, why: str) -> T:
    """A number no prediction can reach, kept out of the digest for the reason given: a
    cache bound, a harness's threshold, the recorded output of a fit a test guards."""
    return value


SPELLINGS = frozenset({fitted.__name__, chosen.__name__, not_an_input.__name__})
EXCLUDED = not_an_input.__name__

# The fewest words an exclusion may argue itself in. A bare marker would pass the walk
# while saying nothing, and the whole point of the record it replaces was that an exclusion
# is a claim a later reader can check.
MIN_REASON_WORDS = 8


@dataclass(frozen=True)
class Declaration:
    """One constant's declaration: where it is, which spelling, and the argument if excluded."""
    module: str          # dotted, `hub.models.predict`
    name: str
    kind: str            # one of `SPELLINGS`
    why: str | None      # the argument, for `not_an_input`; None otherwise
    line: int

    @property
    def key(self) -> str:
        """`stem.NAME`, the key the digest text is written in. Stems are held unique across
        declaring modules by `declarations()`, so the key names one constant."""
        return f"{self.module.rsplit('.', 1)[-1]}.{self.name}"

    @property
    def covered(self) -> bool:
        return self.kind != EXCLUDED


def _spelling(call: ast.expr) -> str | None:
    """Which of the three a call is, whether spelled bare or as `declare.fitted`."""
    if not isinstance(call, ast.Call):
        return None
    f = call.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
    return name if name in SPELLINGS else None


def _reason(call: ast.Call) -> str | None:
    args = list(call.args[1:]) + [k.value for k in call.keywords if k.arg == "why"]
    got = args[0] if args else None
    return got.value if isinstance(got, ast.Constant) and isinstance(got.value, str) else None


def declared_in(source: str, module: str) -> list[Declaration]:
    """The declarations one module's source makes: module-level assignments of one name
    through one of the three spellings. A call anywhere else -- inside a function, in a
    tuple unpacking -- is not a declaration, and an exclusion without its argument is
    refused here rather than recorded as silence."""
    out: list[Declaration] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        else:
            continue
        kind = _spelling(value)
        if kind is None or not isinstance(target, ast.Name):
            continue
        assert isinstance(value, ast.Call)
        why = _reason(value) if kind == EXCLUDED else None
        if kind == EXCLUDED and (why is None or len(why.split()) < MIN_REASON_WORDS):
            raise ValueError(
                f"{module}.{target.id} (line {node.lineno}) is declared not an input without "
                f"an argument of at least {MIN_REASON_WORDS} words. An exclusion is a decision "
                f"on the record; a bare marker is the silence #187, #199 and #201 each were.")
        out.append(Declaration(module, target.id, kind, why, node.lineno))
    return out


@lru_cache(maxsize=1)
def declarations() -> tuple[Declaration, ...]:
    """Every declaration under `hub`, read from the source once per process.

    The tree is read rather than imported: importing every module to ask it would pull
    duckdb, nflreadpy and the rest into a digest call, and a declaration is a fact about
    the source. Two declaring modules with one stem would make `key` ambiguous and are
    refused -- the stem-keyed scan this replaces would have let `hub.draft.season` and
    `hub.season.*` collide in silence.
    """
    found: list[Declaration] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).with_suffix("")
        module = ".".join((PACKAGE, *rel.parts)) if rel.name != "__init__" else PACKAGE
        found.extend(declared_in(path.read_text(), module))
    stems: dict[str, str] = {}
    for d in found:
        stem = d.module.rsplit(".", 1)[-1]
        if stems.setdefault(stem, d.module) != d.module:
            raise RuntimeError(
                f"{stems[stem]} and {d.module} both declare constants and share the stem "
                f"{stem!r}, so a digest key `{stem}.NAME` would name two things. Rename one.")
    return tuple(found)


def covered() -> dict[str, Any]:
    """`{key: live value}` for every covered declaration, the way the digest reads them."""
    return {d.key: getattr(import_module(d.module), d.name)
            for d in declarations() if d.covered}


def excluded() -> dict[str, str]:
    """`{key: why}` for every declaration that argued itself out."""
    return {d.key: d.why or "" for d in declarations() if not d.covered}
