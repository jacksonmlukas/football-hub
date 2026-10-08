"""Every gate's `Harness` names the code its arms run, so the Ledger can see a code change (#435).

`config_digest` hashes config and fitted constants, not code: #361 changed the injury retention
baseline and the new interval was compared with the old, because the key had no term for it.
`Harness.arm_modules` is that term. Since #442 it is **derived**, not hand-kept: a harness
declares `arm_roots` (its own module, beside its arms) and `arm_modules` is the first-party
import closure of those roots (`hub.closure.import_closure`). The first version (#435) declared
the draft gate without `championship_equity` (which scores arm B) and the weekly gate without
`weekly_gate_data`, `weekly` and `panel` (which produce the arm), so an edit to any of them kept
the old history comparable (#439); a list the closure derives cannot omit one. The cost is
recorded on #442 and is intended: any new first-party import in an arm moves that gate's
`code_digest`.

So this holds every harness found by `gate_harnesses.all_harnesses()` (the walk, not a list) to:

1. it declares roots, including its own module, and every module of the derived closure
   resolves to a source file;
2. the derivation is the closure: the AST of each root is walked for `hub.*` imports -- module
   level and function local -- and the walk follows them, stopping at an exempt module (neither
   required nor descended into). Planted trees prove the walk finds both import kinds, skips the
   exempt loader and goes through a middle module; and the derivation of `margin`'s harness is
   pinned to the 9 modules its hand-kept list named, so deriving moved no ledger key;
3. **no exemption is dead** (#442): every `CLOSURE_EXEMPT` key is reached by an import from some
   harness or from the forward reading's `ARM_ROOTS`. A key nothing reaches is a claim about code
   that no arm runs, which hides the day one does. (#442 suspected `hub.store` of being one; it
   is reached, by `hub.draft.board` and `hub.models.conformal`, so it stays -- this check is what
   shows it);
4. **no relative imports** under `src/hub` (#442): the walker follows absolute imports only, so a
   relative one is a module neither the ledger's key nor the forward pin would ever see.

**The exemptions** (`hub.closure.CLOSURE_EXEMPT`, each with its reason) are the places where a
code change cannot change what an arm computes, or where another digest already covers it. The
planted trees pass a copy (`{**CLOSURE_EXEMPT, ...}`) to the walker, never the production table.

Rule 18: `_check` is run against planted harnesses -- no roots, one omitting its own module, a
nonexistent module -- each must fail; the dead-exemption and relative-import checks are run
against planted inputs the same way.
"""
import ast
import importlib
import pathlib

import pytest
from gate_harnesses import all_harnesses

from hub.closure import CLOSURE_EXEMPT, walk
from hub.ledger import code_digest
from hub.models.experiment import Actions, Harness
from hub.season.weekly_forward import ARM_ROOTS

_ACTIONS = Actions(adopt="A", remove="R", show="S")
_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "hub"


def _check(key: str, harness: Harness) -> None:
    home = key.rsplit(".", 1)[0]
    assert harness.arm_roots, f"{key} declares no arm_roots, so its ledger key cannot see code"
    assert home in harness.arm_roots, (
        f"{key} does not name its own module {home!r} in arm_roots {harness.arm_roots}")
    assert set(harness.arm_roots) <= set(harness.arm_modules)
    code_digest(harness.arm_modules)  # raises, naming the module, if one does not resolve


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_each_harness_declares_the_roots_its_arms_run_from(key):
    _check(key, all_harnesses()[key])


def _planted(*roots: str) -> Harness:
    return Harness(name="planted", arm_a="a", arm_b="b", within=("x",), ceiling_arm="an arm",
                   actions=_ACTIONS, arm_roots=roots)


def test_a_planted_harness_with_no_roots_fails():
    with pytest.raises(AssertionError, match="declares no arm_roots"):
        _check("hub.models.planted.PLANTED", _planted())


def test_a_planted_harness_that_omits_its_own_module_fails():
    with pytest.raises(AssertionError, match="does not name its own module"):
        _check("hub.models.planted.PLANTED", _planted("hub.models.market"))


def test_a_planted_harness_naming_a_module_that_does_not_exist_fails():
    with pytest.raises(ValueError, match=r"'hub\.models\.nothing_here' is not a module"):
        _check("hub.models.margin.PLANTED",
               _planted("hub.models.margin", "hub.models.nothing_here"))


def test_deriving_the_closure_gave_margins_harness_the_modules_its_list_named():
    """#442's control: the derivation is the hand-kept list it replaced, so deriving moved no
    gate's `code_digest`. (Measured for all 11 harnesses before the lists were deleted; this one
    stays as the standing pin.)"""
    assert _planted("hub.models.margin").arm_modules == (
        "hub.models.base", "hub.models.components", "hub.models.conformal",
        "hub.models.coverage", "hub.models.margin", "hub.models.market", "hub.models.predict",
        "hub.models.scoring_rules", "hub.models.volume")


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
    return "planted439.gate"


# A copy, never the production table: the planted trees add their own loader to it.
_PLANTED_EXEMPT = {**CLOSURE_EXEMPT, "planted439.fetch.*": "the planted loader"}


def test_the_walk_finds_module_level_and_function_local_imports_and_skips_the_exempt_loader(
        planted_tree):
    """Rule 18 for the derivation: the module-level import (`helper`) and the function-local one
    (`late`) are each found; the exempt loader is neither returned nor asked for."""
    got, hit = walk([planted_tree], "planted439", _PLANTED_EXEMPT)
    assert got == {"planted439.gate", "planted439.helper", "planted439.late"}
    assert "planted439.fetch.*" in hit
    without, _ = walk([planted_tree], "planted439", CLOSURE_EXEMPT)
    assert "planted439.fetch.loader" in without, "the production table must not exempt it"
    assert "planted439.fetch.*" not in CLOSURE_EXEMPT


def test_the_closure_follows_through_a_middle_module_to_what_it_imports(tmp_path, monkeypatch):
    """Transitive: the gate imports `mid`, which imports `deep`; only the gate is a root."""
    pkg = tmp_path / "planted439b"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "gate.py").write_text("from planted439b import mid\n")
    (pkg / "mid.py").write_text("from planted439b import deep\n")
    (pkg / "deep.py").write_text("W = 4\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    got, _ = walk(["planted439b.gate"], "planted439b")
    assert got == {"planted439b.gate", "planted439b.mid", "planted439b.deep"}


def _dead_exemptions(roots: list[str], exempt: dict[str, str]) -> list[str]:
    return sorted(set(exempt) - walk(roots, "hub", exempt)[1])


def test_every_exemption_is_reached_by_some_harness_or_the_forward_arm():
    """#442: an exemption nothing imports is a claim about code no arm runs. (`hub.season.survivor`
    was the opposite -- reached for one constant, with the wrong reason -- and was fixed by
    moving the constant.)"""
    roots = [r for h in all_harnesses().values() for r in h.arm_roots] + list(ARM_ROOTS)
    dead = _dead_exemptions(roots, CLOSURE_EXEMPT)
    assert not dead, (
        f"{dead} in CLOSURE_EXEMPT is imported by no harness's arm and no forward-arm root: "
        f"prune it, or the day an arm does reach it the entry will hide that.")


def test_a_planted_dead_exemption_is_flagged_and_a_live_one_is_not():
    """Control for the check above: an exemption for a module nothing imports is found, and
    `hub.fetch.*` (reached by the weekly arm's loaders) is not."""
    exempt = {**CLOSURE_EXEMPT, "hub.nothing_imports_this": "planted"}
    dead = _dead_exemptions(["hub.models.margin", *ARM_ROOTS], exempt)
    assert "hub.nothing_imports_this" in dead
    assert "hub.fetch.*" not in dead


def relative_imports(source: str) -> list[int]:
    """Line numbers of the relative imports in `source`."""
    return [n.lineno for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.ImportFrom) and n.level != 0]


def test_no_module_under_src_hub_uses_a_relative_import():
    """#442: `hub.closure` follows absolute imports only, so a relative one would be a module the
    ledger's key and the forward pin never see. Write `from hub.x import y`."""
    found = {str(p.relative_to(_SRC)): lines for p in sorted(_SRC.rglob("*.py"))
             if (lines := relative_imports(p.read_text()))}
    assert not found, f"relative imports (use absolute): {found}"


def test_the_relative_import_scan_finds_a_planted_one():
    assert relative_imports("import os\nfrom . import x\n") == [2]
    assert relative_imports("from ..pkg import y\n") == [1]
    assert relative_imports("from hub import x\n") == []
