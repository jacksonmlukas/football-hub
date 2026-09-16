"""Every module-level number under `hub` has declared its digest coverage (#253).

`hub.declare` is the mechanism -- `fitted`, `chosen`, `not_an_input` at the constant, the
digest walked off those declarations -- and `tests/unit/test_config.py` holds what the walk
finds. This file holds the other half: **a number nobody declared is refused by name**, so
the question "does this identify a model version" is asked once per constant and cannot be
skipped by a module never having been registered. Before #253 that question was answered by
a wholesale module list, an extras list, an exclusion list and a module-level opt-out string
in twenty-six modules, and three separate escapes reached production through the seams.

**What is scanned, stated rather than hidden.** A public upper-case module-level name whose
value holds a float literal anywhere inside it -- the breadth the scan had before, now with
no directory filter and no stem key. Widening to every `int` flags a hundred and thirty
names -- cache sizes, API tiers, print widths -- so the integers known to matter (the shape
of a random draw, #201; the nfeloqb pin, #271) are held by name below rather than by breadth.

**One ratchet, only shrinking.** `SIGNATURE_DEFAULTS` is the escape ADR-0006 recorded and
#253 restates: a function-signature default that sets the extent of a draw has no
module-level name to declare, and each is named here until it is given one. A second,
`UNDECLARED`, named the fourteen numbers in modules another lane owned the afternoon #253
landed; #296 declared them and emptied it, and the module-level opt-out string it carried
their reasons from is gone from the tree.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hub import declare

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"

# The signature defaults that set the extent of a draw and have no name to declare (ADR-0006
# and #201's open escape). `n_sims` in `availability` and `evaluate` set the precision of a
# Monte Carlo estimate, not the extent of a simulated season, and are named for the same
# reason: a default that shapes a draw is on the record whichever kind it is.
SIGNATURE_DEFAULTS: dict[str, str] = {
    **{f"draft/backtest.py:{fn}:{arg}": f"the extent of the {kind} draw; #201's open escape"
       for fn in ("compare", "diagnose", "noise_sensitivity")
       for arg, kind in (("n_draft_sims", "draft"), ("n_season_sims", "season"))},
    "draft/availability.py:availability:n_sims": "the precision of the availability estimate",
    "draft/evaluate.py:evaluate:n_sims": "the precision of an offline harness's estimate",
    # Exhibits: not a dependency of the product (`test_the_exhibit_is_not_a_dependency.py`),
    # so a draw's extent there moves no published number.
    "exhibits/championship_equity.py:champion_probability:n_sims": "an exhibit's draw",
    "exhibits/championship_equity.py:win_probability:n_draft_sims": "an exhibit's draw",
    "exhibits/championship_equity.py:win_probability:n_season_sims": "an exhibit's draw",
    "exhibits/leverage.py:seed_value:n_sims": "an exhibit's draw",
    "exhibits/leverage.py:simulate:n_sims": "an exhibit's draw",
    "exhibits/leverage.py:sweep_row:n_sims": "an exhibit's draw",
}

# The integers the scan's breadth cannot see and #201 and #271 established must be covered:
# each is a declaration in the tree, held by name.
SHAPES = ("config.REG_SEASON_WEEKS", "league.PLAYOFF_TEAMS", "league.PLAYOFF_ROUNDS",
          "optimize.DEFAULT_ROUNDS", "cohort.ROUNDS", "cohort.DRAFTS", "nfeloqb.COMMIT",
          "board.MIN_GAMES", "regression.MIN_GAMES", "schedule.STALE_AFTER_DAYS")


def _holds_a_float(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, float)
               for n in ast.walk(node))


def _module_level_floats(path: Path) -> list[tuple[str, ast.expr]]:
    """`(NAME, value)` for every public upper-case module-level assignment holding a float."""
    out = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            if (isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_")
                    and _holds_a_float(value)):
                out.append((t.id, value))
    return out


def _declared_keys() -> dict[str, declare.Declaration]:
    return {f"{d.module.replace('hub.', '', 1).replace('.', '/')}.py:{d.name}": d
            for d in declare.declarations()}


def test_every_module_level_float_is_declared():
    declared = _declared_keys()
    undeclared = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for name, _ in _module_level_floats(path):
            key = f"{rel}:{name}"
            if key not in declared:
                undeclared.append(key)
    assert not undeclared, (
        f"module-level numbers nobody has declared: {undeclared}. Say what each is where it "
        f"is written -- `fitted(v)`, `chosen(v)` or `not_an_input(v, why)` from `hub.declare` "
        f"-- so the digest can read it or the exclusion is on the record.")


def test_the_opt_out_string_is_gone_from_the_tree():
    """`NOT_FITTED_BECAUSE` was the fourth mechanism; #253 stopped reading it and #296
    removed the last five copies. A module carrying it again would be a module-level
    exclusion the walk cannot see -- the state #253 collapsed."""
    carrying = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                      if "NOT_FITTED_BECAUSE" in p.read_text())
    assert carrying == [], f"NOT_FITTED_BECAUSE is back in {carrying}; declare the numbers instead"


@pytest.mark.parametrize("key", SHAPES)
def test_an_integer_that_shapes_a_draw_or_pins_a_source_is_declared(key):
    got = {d.key: d for d in declare.declarations()}
    assert key in got and got[key].covered, f"{key} is not a covered declaration"


def test_a_signature_default_that_shapes_a_draw_is_declared_or_named():
    """The escape ADR-0006 recorded, restated. A parameter named for a number of simulations
    with a literal default is a model input with no module-level name; each is listed until
    it is given one, and a new one is refused here rather than arriving unseen."""
    found = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            a = node.args
            params = a.posonlyargs + a.args + a.kwonlyargs
            defaults = [None] * (len(a.posonlyargs) + len(a.args) - len(a.defaults)) + list(a.defaults)
            defaults += list(a.kw_defaults)
            for param, default in zip(params, defaults, strict=True):
                if param.arg.endswith("_sims") and isinstance(default, ast.Constant):
                    found.append(f"{rel}:{node.name}:{param.arg}")
    unnamed = sorted(set(found) - set(SIGNATURE_DEFAULTS))
    gone = sorted(set(SIGNATURE_DEFAULTS) - set(found))
    assert not unnamed, (
        f"signature defaults that shape a draw and are declared nowhere: {unnamed}. Give "
        f"each a module-level name declared through `hub.declare`, or name it here with why.")
    assert not gone, f"SIGNATURE_DEFAULTS names defaults that no longer exist: {gone}"
