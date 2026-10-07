"""Every gate's `Harness` names the code its arms run, so the Ledger can see a code change (#435).

`config_digest` hashes config and fitted constants, not code: #361 changed the injury retention
baseline and the new interval was compared with the old, because the key had no term for it.
`Harness.arm_modules` is that term, declared beside the arms. A harness that declares too little
defeats it silently: the first version declared the draft gate without `championship_equity`
(which scores arm B) and the weekly gate without `weekly_gate_data`, `weekly` and `panel` (which
produce the arm), so an edit to any of them kept the old history comparable (#439).

So this holds every harness found by `gate_harnesses.all_harnesses()` (the walk, not a list) to:

1. it declares modules, including its own, and every name resolves to a source file;
2. **the first-party import closure is declared** (#439). The AST of the home module and of each
   declared module is walked for `hub.*` imports -- module level and function local -- and the
   walk follows them. Any module reached that is neither declared nor exempt fails. An exempt
   module is neither required nor descended into.

`code_digest` therefore covers exactly the declared modules; the closure is what the declaration
is checked against, not something hashed on its own.

**The exemptions** (`EXEMPT`, each with its reason) are the places where a code change cannot
change what an arm computes, or where another digest already covers it. Everything else a gate
reaches is declared, including modules that look peripheral: a declaration that is too long breaks
comparability on an unrelated edit, which is the direction this ledger errs in; one that is too
short is the defect.

Rule 18: `_check` is run against planted harnesses -- no modules, one omitting its own module, a
nonexistent module, and one that omits a module its arm imports (module level and function local,
in a throwaway package) -- each must fail, and the same harness with the module listed passes.
"""
import ast
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest
from gate_harnesses import all_harnesses

from hub.ledger import code_digest
from hub.models.experiment import Actions, Harness

_ACTIONS = Actions(adopt="A", remove="R", show="S")

# name (exact) or "prefix.*" -> why a change there cannot change what an arm computes.
EXEMPT: dict[str, str] = {
    "hub.models.experiment": "the shared gate rule, run by every gate; naming it would end every "
                             "gate's history on any edit there (reasoned in run_gate's docstring)",
    "hub.fetch.*": "data loaders and caches: the bytes they return are what data_digest pins, "
                   "and a loader change that alters bytes moves it",
    "hub.config": "config_digest hashes the resolved config and every fitted constant in "
                  "FITTED_MODULES",
    "hub.atomic": "atomic file writes; no arm computation",
    "hub.jsonio": "JSON read/write and timestamp helpers; no arm computation",
    "hub.paths": "path constants; no arm computation",
    "hub.store": "the on-disk table store behind the loaders; the bytes it returns are pinned",
    "hub.cli": "CLI plumbing (`unavailable`); runs after the verdict, not in an arm",
    "hub.contracts": "schema assertions on loaded data; they raise or pass, they do not compute",
    "hub.declare": "decision/chosen markers; identity decorators",
    "hub.ledger": "the comparison machinery itself, not an arm",
    "hub.season.survivor": "reached for the NFL_WEEKS constant (the schedule length) only; the "
                           "survivor pool's code is no gate's arm",
}


def _exempt(mod: str) -> bool:
    return any(mod == k or (k.endswith(".*") and mod.startswith(k[:-1])) for k in EXEMPT)


def _imports(mod: str, prefix: str = "hub") -> set[str]:
    """First-party plain modules `mod` imports, anywhere in its source (local imports too)."""
    spec = importlib.util.find_spec(mod)
    assert spec is not None and spec.origin is not None, mod
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


def closure(home: str, declared: tuple[str, ...], prefix: str = "hub") -> set[str]:
    """Every non-exempt first-party module reachable from `home` and `declared`."""
    seen: set[str] = set()
    todo = [home, *declared]
    while todo:
        m = todo.pop()
        if m in seen or _exempt(m):
            continue
        seen.add(m)
        todo += _imports(m, prefix)
    return seen


def _check(key: str, harness: Harness, prefix: str = "hub") -> None:
    home = key.rsplit(".", 1)[0]
    assert harness.arm_modules, f"{key} declares no arm_modules, so its ledger key cannot see code"
    assert home in harness.arm_modules, (
        f"{key} does not name its own module {home!r} in arm_modules {harness.arm_modules}")
    code_digest(harness.arm_modules)  # raises, naming the module, if one does not resolve
    missing = sorted(closure(home, harness.arm_modules, prefix) - set(harness.arm_modules))
    assert not missing, (
        f"{key} runs {missing} but does not declare them in arm_modules: an edit there would "
        f"leave its old ledger history comparable (#439). Declare each, or add it to EXEMPT "
        f"with the reason a change there cannot change an arm.")


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_each_harness_declares_the_modules_its_arms_run(key):
    _check(key, all_harnesses()[key])


def _planted(*modules: str) -> Harness:
    return Harness(name="planted", arm_a="a", arm_b="b", within=("x",), ceiling_arm="an arm",
                   actions=_ACTIONS, arm_modules=modules)


def test_a_planted_harness_with_no_modules_fails():
    with pytest.raises(AssertionError, match="declares no arm_modules"):
        _check("hub.models.planted.PLANTED", _planted())


def test_a_planted_harness_that_omits_its_own_module_fails():
    with pytest.raises(AssertionError, match="does not name its own module"):
        _check("hub.models.planted.PLANTED", _planted("hub.models.market"))


def test_a_planted_harness_naming_a_module_that_does_not_exist_fails():
    with pytest.raises(ValueError, match="not a module with a source file"):
        _check("hub.models.planted.PLANTED",
               _planted("hub.models.planted", "hub.models.nothing_here"))


@pytest.fixture
def planted_tree(tmp_path, monkeypatch):
    """A throwaway `hub`-like package `planted439` whose gate imports one helper at module level
    and one inside a function, plus an exempt loader."""
    pkg = tmp_path / "planted439"
    (pkg / "fetch").mkdir(parents=True)
    for f in ("__init__.py", "fetch/__init__.py"):
        (pkg / f).write_text("")
    (pkg / "gate.py").write_text(
        "from planted439 import helper\n"
        "from planted439.fetch import loader\n"
        "def arm():\n    from planted439 import late\n    return helper, late, loader\n")
    (pkg / "helper.py").write_text("X = 1\n")
    (pkg / "late.py").write_text("Y = 2\n")
    (pkg / "fetch" / "loader.py").write_text("Z = 3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    for m in [m for m in sys.modules if m.startswith("planted439")]:
        del sys.modules[m]
    monkeypatch.setitem(EXEMPT, "planted439.fetch.*", "the planted loader")
    return "planted439.gate.PLANTED"


def test_a_planted_harness_omitting_a_module_its_arm_imports_fails_and_listing_it_passes(
        planted_tree):
    """Rule 18 for the closure: the module-level import (`helper`) and the function-local one
    (`late`) are each found; the exempt loader is not asked for; naming both passes."""
    key = planted_tree
    with pytest.raises(AssertionError, match=r"planted439\.helper.*planted439\.late"):
        _check(key, _planted("planted439.gate"), prefix="planted439")
    with pytest.raises(AssertionError, match=r"planted439\.late"):
        _check(key, _planted("planted439.gate", "planted439.helper"), prefix="planted439")
    _check(key, _planted("planted439.gate", "planted439.helper", "planted439.late"),
           prefix="planted439")


def test_the_closure_follows_through_a_declared_module_to_what_it_imports(tmp_path, monkeypatch):
    """Transitive: the gate imports `mid`, which imports `deep`. Declaring `mid` is not enough."""
    pkg = tmp_path / "planted439b"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "gate.py").write_text("from planted439b import mid\n")
    (pkg / "mid.py").write_text("from planted439b import deep\n")
    (pkg / "deep.py").write_text("W = 4\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    with pytest.raises(AssertionError, match=r"planted439b\.deep"):
        _check("planted439b.gate.P", _planted("planted439b.gate", "planted439b.mid"),
               prefix="planted439b")
    _check("planted439b.gate.P",
           _planted("planted439b.gate", "planted439b.mid", "planted439b.deep"),
           prefix="planted439b")
