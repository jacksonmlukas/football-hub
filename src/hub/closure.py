"""The first-party import closure of the code an arm runs, and the exemptions it walks with
(#439, #456, #442).

These lived in `hub.ledger` beside `code_digest` and the comparison machinery: two reasons to
change in one module. The ledger's code declaration (`Harness.arm_modules`, #435/#439) and the
forward reading's arm pin (`weekly_forward.PINNED_ARM_MODULES`, #456) both ask the same
question -- *which modules can an edit change what this arm computes?* -- and both answer it
with `import_closure` and `CLOSURE_EXEMPT`, so there is one walker and one list.

**Relative imports are not followed** (`level != 0`). No file under `src/hub` uses one, and
`tests/contracts/test_every_gate_declares_the_code_it_runs.py` holds that: a relative import
would be a module the walker never reaches, so an edit there would leave a ledger key and a
forward pin unmoved. Write `from hub.x import y`.

A leaf: stdlib only, so `hub.ledger` and `hub.models.experiment` can use it without a cycle.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from collections.abc import Iterable
from functools import cache
from pathlib import Path

# name (exact) or "prefix.*" -> why a change there cannot change what an arm computes, or where
# another digest already covers it. A module here is neither required nor descended into.
CLOSURE_EXEMPT: dict[str, str] = {
    "hub.models.experiment": "the shared gate rule, run by every gate; naming it would end every "
                             "gate's history on any edit there (reasoned in run_gate's docstring)",
    "hub.fetch.*": "data loaders and caches: the bytes they return are what data_digest pins, "
                   "and a loader change that alters bytes moves it",
    "hub.config": "config_digest hashes the resolved config and every fitted constant in "
                  "FITTED_MODULES",
    "hub.atomic": "atomic file writes; no arm computation",
    "hub.jsonio": "JSON read/write and timestamp helpers; no arm computation",
    "hub.paths": "path constants; no arm computation",
    "hub.store": "the on-disk table store behind the loaders; the bytes it returns are pinned "
                 "(reached by hub.draft.board and hub.models.conformal)",
    "hub.cli": "CLI plumbing (`unavailable`); runs after the verdict, not in an arm",
    "hub.contracts": "schema assertions on loaded data; they raise or pass, they do not compute",
    "hub.declare": "decision/chosen markers; identity decorators",
    "hub.ledger": "the comparison machinery itself, not an arm",
}


def _key_matches(key: str, mod: str) -> bool:
    return mod == key or (key.endswith(".*") and mod.startswith(key[:-1]))


def exempt_key(mod: str, exempt: dict[str, str] | None = None) -> str | None:
    """The `CLOSURE_EXEMPT` key that exempts `mod`, or `None`. `exempt` defaults to the table."""
    table = CLOSURE_EXEMPT if exempt is None else exempt
    return next((k for k in table if _key_matches(k, mod)), None)


def closure_exempt(mod: str) -> bool:
    return exempt_key(mod) is not None


def module_source(name: str) -> Path:
    """The source file of dotted module `name`: the one resolver `code_digest` and
    `module_digests` share (#442). Raises `ValueError` for a name with no `.py` source and for
    a package, whose `__init__` is not the code an arm runs."""
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise ValueError(f"{name!r} is not a module with a source file")
    if spec.submodule_search_locations is not None:
        raise ValueError(f"{name!r} is a package; name the modules whose code the arms run")
    return Path(spec.origin)


def first_party_imports(mod: str, prefix: str = "hub") -> set[str]:
    """First-party plain modules `mod` imports, anywhere in its source (local imports too).
    Relative imports are skipped (see the module docstring)."""
    spec = importlib.util.find_spec(mod)
    if spec is None or spec.origin is None:
        raise ValueError(f"{mod!r} is not a module with a source file")
    out: set[str] = set()
    for n in ast.walk(ast.parse(Path(spec.origin).read_text())):
        if isinstance(n, ast.Import):
            names = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            names = [n.module, *(f"{n.module}.{a.name}" for a in n.names)]
        else:
            continue
        for name in names:
            if name != prefix and not name.startswith(prefix + "."):
                continue
            try:
                s = importlib.util.find_spec(name)
            except (ImportError, AttributeError):  # `from mod import attr`: not a module
                continue
            if s is not None and s.origin and s.submodule_search_locations is None:
                out.add(name)
    return out


def walk(roots: Iterable[str], prefix: str = "hub",
         exempt: dict[str, str] | None = None) -> tuple[set[str], set[str]]:
    """`(closure, exempt_hit)` from `roots`: every non-exempt first-party module reachable by
    import, and the `exempt` table keys that some import reached on the way (a key no walk ever
    hits is a dead exemption, which the contract flags). An exempt module is neither returned
    nor descended into."""
    table = CLOSURE_EXEMPT if exempt is None else exempt
    seen: set[str] = set()
    hit: set[str] = set()
    todo = list(roots)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        key = exempt_key(m, table)
        if key is not None:
            hit.add(key)
            continue
        seen.add(m)
        todo += first_party_imports(m, prefix)
    return seen, hit


def import_closure(roots: Iterable[str], prefix: str = "hub") -> set[str]:
    """Every non-exempt first-party module reachable from `roots` by import (#439, #456).

    An exempt module (`CLOSURE_EXEMPT`) is neither returned nor descended into."""
    return walk(roots, prefix)[0]


@cache
def closure_of(roots: tuple[str, ...]) -> tuple[str, ...]:
    """`import_closure(roots)` sorted, cached per roots tuple: what `Harness.arm_modules` is.
    Cached because a noise sweep runs one harness many times in a process; the sources do not
    change under a running process."""
    return tuple(sorted(import_closure(roots)))


def module_digests(modules: Iterable[str]) -> dict[str, str]:
    """`{module: 12-char sha256 of its source bytes}`, for a pin that can name what moved (#456)."""
    return {name: hashlib.sha256(module_source(name).read_bytes()).hexdigest()[:12]
            for name in sorted(set(modules))}
