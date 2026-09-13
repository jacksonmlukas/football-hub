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

**Two ratchets, both only shrinking.** `UNDECLARED` names the numbers in modules another
lane owned the afternoon #253 landed, with their module's old opt-out reason carried until
that lane declares them; a listed name that is declared or gone fails as stale, and a
number outside the list fails as undeclared. `SIGNATURE_DEFAULTS` is the escape ADR-0006
recorded and #253 restates: a function-signature default that sets the extent of a draw has
no module-level name to declare, and each is named here until it is given one.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from hub import declare

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"

# Modules owned by another lane when #253 landed (2026-09-12); each still carries its
# `NOT_FITTED_BECAUSE` string, read by nothing since #253. The reason is the module's own,
# carried here so the exclusion stays on the record until the constant argues for itself.
UNDECLARED: dict[str, str] = {
    "draft/backtest.py:PROGRESS_POLL": "the draft gate's harness; nothing here predicts",
    "draft/backtest.py:PROGRESS_GRACE": "the draft gate's harness; nothing here predicts",
    "draft/backtest.py:NOISE_SCALES": "the draft gate's harness; nothing here predicts",
    "draft/backtest.py:VOID_FLOOR":
        "a pre-registered guard on which runs are reported, the same shape as weekly_gate.VOID_FLOOR",
    "models/eval.py:DEFAULT_HOLDOUT": "model-comparison harness; it reads predictions, never makes them",
    "models/margin.py:FITTED_SD": "the recorded output of the MARGIN_SD fit, held by a test",
    "models/margin.py:FITTED_SE": "the recorded output of the MARGIN_SD fit, held by a test",
    "models/margin.py:FITTED_KEY_EXCESS": "the recorded output of the MARGIN_SD fit, held by a test",
    "models/margin.py:FITTED_SHAPE_GAIN": "the recorded output of the MARGIN_SD fit, held by a test",
    "models/margin.py:FITTED_SHAPE_SE": "the recorded output of the MARGIN_SD fit, held by a test",
    "models/margin.py:FITTED_SHAPE_CEILING": "the recorded output of the MARGIN_SD fit, held by a test",
    "season/pool.py:DEFAULT_CONCENTRATIONS": "the axis `sensitivity` sweeps, not a value any figure is computed at",
    "season/pool.py:DECISIVE_SIGMA": "the evidential bar the gates are stated at, a convention",
    "season/survivor.py:MIN_PROB": "a floor that keeps a zero out of a log; a setting",
}
STILL_OPTED_OUT: frozenset[str] = frozenset(
    {k.split(":")[0] for k in UNDECLARED})

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
          "board.MIN_GAMES", "regression.MIN_GAMES", "quarterback.ELO_PER_POINT",
          "quarterback.STALE_AFTER_DAYS")


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


def test_every_module_level_float_is_declared_or_named_as_undeclared():
    declared = _declared_keys()
    undeclared, stale = [], []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for name, _ in _module_level_floats(path):
            key = f"{rel}:{name}"
            if key in declared:
                if key in UNDECLARED:
                    stale.append(key)
            elif key not in UNDECLARED:
                undeclared.append(key)
    assert not undeclared, (
        f"module-level numbers nobody has declared: {undeclared}. Say what each is where it "
        f"is written -- `fitted(v)`, `chosen(v)` or `not_an_input(v, why)` from `hub.declare` "
        f"-- so the digest can read it or the exclusion is on the record.")
    assert not stale, f"declared now, so no longer undeclared; drop from UNDECLARED: {stale}"


def test_the_undeclared_list_only_shrinks():
    """A name that is gone from its module has left the ratchet with it."""
    gone = []
    for key in UNDECLARED:
        rel, name = key.split(":")
        if name not in {n for n, _ in _module_level_floats(SRC / rel)}:
            gone.append(key)
    assert not gone, f"UNDECLARED names constants that no longer exist: {gone}"


def test_the_opt_out_string_is_read_by_nothing_and_survives_only_where_named():
    """`NOT_FITTED_BECAUSE` was the fourth mechanism; since #253 no test and no digest reads
    it. It survives only in the modules `UNDECLARED` names, until their lane declares."""
    carrying = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py")
                if "NOT_FITTED_BECAUSE" in p.read_text()}
    assert carrying == STILL_OPTED_OUT, (
        f"NOT_FITTED_BECAUSE is read by nothing; new={sorted(carrying - STILL_OPTED_OUT)} "
        f"gone={sorted(STILL_OPTED_OUT - carrying)}")
    # And nothing reads it: no test, no digest. A reader would make the string a mechanism
    # again, which is the state #253 collapsed.
    # A read is an attribute access or a `getattr`; the name in a docstring recording what
    # was retired is prose, and this file itself is the one place it is spelled by design.
    here = Path(__file__)
    reads = re.compile(r"\.NOT_FITTED_BECAUSE\b|getattr\([^)]*['\"]NOT_FITTED_BECAUSE['\"]")
    readers = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").rglob("*.py")
                     if p != here and reads.search(p.read_text()))
    assert readers == [], f"NOT_FITTED_BECAUSE is read again by {readers}"


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
