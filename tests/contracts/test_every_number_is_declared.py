"""Every module-level number under `hub` has declared its digest coverage (#253, #321).

`hub.declare` is the mechanism -- `fitted`, `chosen`, `not_an_input` at the constant, the
digest walked off those declarations -- and `tests/unit/test_config.py` holds what the walk
finds. This file holds the other half: **a number nobody declared is refused by name**, so
the question "does this identify a model version" is asked once per constant and cannot be
skipped by a module never having been registered. Before #253 that question was answered by
a wholesale module list, an extras list, an exclusion list and a module-level opt-out string
in twenty-six modules, and three separate escapes reached production through the seams.

**What is scanned, stated rather than hidden.** Any module-level name -- public or
underscore-prefixed, upper- or lower-case, bound at column zero or under an `if`, `try`,
`with`, loop or `match` (`declare.module_level`) -- that either holds a float literal anywhere
in its value or *is* a float, or a container of floats, once the module is imported (a
derived `1 / 3` or `ModelConfig().x` has no literal and is a float all the same). Until #321
the scan read only public upper-case names written at column zero, and of eight spellings
audit IV tried, one was seen. Widening to every `int` flags a hundred and thirty names --
cache sizes, API tiers, print widths -- so the integers known to matter (the shape of a
random draw, #201; the nfeloqb pin, #271) are held by name below rather than by breadth.

**Two ratchets, only shrinking.** `SIGNATURE_DEFAULTS` is the escape ADR-0006 recorded and
#253 restates: a function-signature default that is a float, or sets the extent of a draw
(`*_sims`), has no module-level name to declare, and each is named here until it is given
one. `DATACLASS_DEFAULTS` is the same for a dataclass field's default. Both used to be
narrower than their claim -- the signature scan matched only names ending `_sims`, so
`draws`, `alpha` and `k` were unratcheted, and a dataclass field was not scanned at all
(#321). A third, `UNDECLARED`, named fourteen numbers in modules another lane owned the
afternoon #253 landed; #296 declared them and emptied it.
"""
from __future__ import annotations

import ast
import importlib
import numbers
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from hub import declare

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "hub"


def _draw(arg: str, kind: str) -> str:
    return f"the extent of the {kind} draw; #201's open escape"


# Every signature default that is a float or sets the extent of a draw, and has no
# module-level name to declare (ADR-0006, #201's open escape, #321). `n_sims` in
# `availability` and `evaluate` set the precision of a Monte Carlo estimate, not the extent
# of a simulated season, and are named for the same reason: a default that shapes a draw is
# on the record whichever kind it is. Methods are keyed by their qualified name.
SIGNATURE_DEFAULTS: dict[str, str] = {
    **{f"draft/backtest.py:{fn}:{arg}": _draw(arg, kind)
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
    # --- the float defaults #321 widened the scan to see -------------------------------
    **{f"exhibits/leverage.py:{fn}:{arg}": "an exhibit's dial; the exhibit is not a dependency"
       for fn, arg in (("simulate", "k"), ("simulate", "vol"), ("simulate", "cv_mult"),
                       ("team_mean", "k"), ("team_mean", "vol"), ("team_mean", "cv_mult"),
                       ("calibrate", "vol"), ("calibrate", "cv_mult"), ("calibrate", "lo"),
                       ("calibrate", "hi"), ("calibrated", "vol"), ("calibrated", "cv_mult"),
                       ("sweep_row", "vol"), ("sweep_row", "cv_mult"))},
    "exhibits/championship_equity.py:win_probability:opp_noise": "an exhibit's dial; 1.0 is the "
        "fitted pick-noise law itself",
    **{f"{fn}:opp_noise": "the room's scale over the fitted pick noise, 1.0 being the fitted law "
       "itself; a sensitivity dial each caller passes"
       for fn in ("draft/backtest.py:optimizer_strategy", "draft/backtest.py:play",
                  "draft/backtest.py:compare", "draft/backtest.py:ceiling",
                  "draft/optimize.py:simulate_remaining_draft")},
    "draft/availability.py:fit_pick_noise:default": "the (a, b) of sigma(mu) = a + b*mu used "
        "when a room's history is too thin to fit; owed a name beside the fitted law",
    "draft/availability.py:noise_from_picks:default": "the same fallback pick-noise law; owed a "
        "name beside the fitted one",
    "models/base.py:Conformalized.__init__:alpha": "the miscoverage a wrapper calibrates to; "
        "config's `model.conformal_alpha` is the digest-covered twin",
    "models/correlate.py:significant:se_threshold": "a display filter on a correlation table",
    "models/predict.py:nearest_correlation:tol": "a numerical convergence tolerance",
    "models/predict.py:moments:floor_sd": "zero means no floor; the old floor of 2.0 is gone "
        "(its docstring says why)",
    "models/scoring_rules.py:log_loss:eps": "a numerical clip against log(0)",
    "models/starter_change.py:_log_loss:eps": "a numerical clip against log(0)",
    "models/spread.py:_predict:w": "a harness candidate's blend weight, passed by the "
        "experiment that fits it",
    "season/lineup.py:optimize:opp_sd": "the opponent's spread when a caller names none; "
        "`main` takes it as a flag",
    "season/pool.py:weekly:outlay": "zero outlay: the free pick, the baseline the week costs against",
    "season/weekly_gate.py:compare:z": "the gate's own dial, zero for the null arm",
    "season/weekly_gate.py:treatment_effects:z": "the gate's own dial, zero for the null arm",
}

# A dataclass field's default that is a float, and has no module-level name to declare.
# `config.py`'s are fields of `HubConfig`, which `config_digest` hashes whole, resolved;
# `PropPrice.p_nonzero` is a field of a priced row, its own output and no input.
DATACLASS_DEFAULTS: dict[str, str] = {
    **{f"config.py:{cls}:{field}": "a HubConfig field; config_digest hashes the resolved config"
       for cls, field in (("DraftConfig", "espn_weight"), ("DraftConfig", "projection_lambda"),
                          ("DraftConfig", "z_clip"), ("DraftConfig", "correction_clamp_frac"),
                          ("DraftConfig", "sos_ridge"), ("ModelConfig", "conformal_alpha"))},
    **{f"config.py:PoolConfig:{field}": "a HubConfig field, but `pool` is left out of "
       "config_digest on purpose and hashed by `pool_digest` instead"
       for field in ("entry_fee", "buyback_fee", "field_concentration")},
    "models/props.py:PropPrice:p_nonzero": "a priced row's own output, the share of weeks "
        "with a nonzero stat; nothing reads it as an input",
    # The survivor pool's records (NamedTuples): each field is a figure the simulation
    # reports about itself, and its default is the "none to report" zero, not a setting.
    **{f"season/pool.py:{cls}:{field}": "a record's own reported figure; the default is the "
       "'nothing to report' zero, and the survivor pool is outside config_digest on purpose"
       for cls, field in (("Candidate", "dollars_se"), ("Candidate", "survives_se"),
                          ("EntryOutcome", "last_out"), ("EntryOutcome", "replans"),
                          ("EntryOutcome", "share_sd"), ("Weekly", "given_up_se"),
                          ("Weekly", "outlay"), ("_Week", "concentration"))},
}

# The integers the scan's breadth cannot see and #201 and #271 established must be covered:
# each is a declaration in the tree, held by name.
SHAPES = ("config.REG_SEASON_WEEKS", "league.PLAYOFF_TEAMS", "league.PLAYOFF_ROUNDS",
          "optimize.DEFAULT_ROUNDS", "cohort.ROUNDS", "cohort.DRAFTS", "nfeloqb.COMMIT",
          "board.MIN_GAMES", "regression.MIN_GAMES", "schedule.STALE_AFTER_DAYS")


# --- the scans, each a function of a source so a planted one can be asked --------------


def _holds_a_float(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, float)
               for n in ast.walk(node))


def _is_floaty(v: Any, depth: int = 0) -> bool:
    """A float, or a container of one, by what the object *is* once imported."""
    if isinstance(v, bool):
        return False
    if isinstance(v, numbers.Real) and not isinstance(v, numbers.Integral):
        return True
    if depth < 3 and isinstance(v, Mapping):
        return any(_is_floaty(x, depth + 1) for x in v.values())
    if depth < 3 and isinstance(v, (list, tuple, set, frozenset)):
        return any(_is_floaty(x, depth + 1) for x in v)
    return False


def _target_names(target: ast.expr) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _assignments(source: str) -> Iterator[tuple[str, ast.expr]]:
    """`(name, value)` for every module-level binding of a name by assignment, wherever the
    statement sits (`declare.module_level`). Dunders are the interpreter's."""
    for node in declare.module_level(ast.parse(source).body):
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        for t in targets:
            for name in _target_names(t):
                if not (name.startswith("__") and name.endswith("__")):
                    yield name, value


def undeclared_floats(source: str, module: str, live: Mapping[str, Any]) -> list[str]:
    """Names this module binds at module level that hold a float -- a literal in the source
    or, in `live`, a float or a container of floats at run time -- and that no declaration
    in the same source covers."""
    declared = {d.name for d in declare.declared_in(source, module)}
    held = {name for name, value in _assignments(source)
            if _holds_a_float(value) or _is_floaty(live.get(name))}
    return sorted(held - declared)


def _defs(node: ast.AST, prefix: str = "") -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function with the dotted name a reader would give it, methods under their class."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            yield from _defs(child, f"{prefix}{child.name}.")
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield prefix + child.name, child
            yield from _defs(child, f"{prefix}{child.name}.<locals>.")
        else:
            yield from _defs(child, prefix)


def signature_defaults(source: str, rel: str) -> set[str]:
    """`file:function:arg` for every default that is a float, or that sets the extent of a
    draw (a `*_sims` parameter with a constant default)."""
    found: set[str] = set()
    for qual, node in _defs(ast.parse(source)):
        a = node.args
        params = a.posonlyargs + a.args + a.kwonlyargs
        defaults: list[ast.expr | None] = (
            [None] * (len(a.posonlyargs) + len(a.args) - len(a.defaults)) + list(a.defaults)
            + list(a.kw_defaults))
        for param, default in zip(params, defaults, strict=True):
            if default is None:
                continue
            if (param.arg.endswith("_sims") and isinstance(default, ast.Constant)) \
                    or _holds_a_float(default):
                found.add(f"{rel}:{qual}:{param.arg}")
    return found


def _is_dataclass(node: ast.ClassDef) -> bool:
    marks = [ast.unparse(d) for d in node.decorator_list] + [ast.unparse(b) for b in node.bases]
    return any("dataclass" in m or "NamedTuple" in m for m in marks)


def dataclass_defaults(source: str, rel: str) -> set[str]:
    """`file:Class:field` for every dataclass (or NamedTuple) field whose default is a float."""
    return {f"{rel}:{cls.name}:{stmt.target.id}"
            for cls in ast.walk(ast.parse(source)) if isinstance(cls, ast.ClassDef)
            and _is_dataclass(cls)
            for stmt in cls.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
            and stmt.value is not None and _holds_a_float(stmt.value)}


def _module_of(rel: Path) -> str:
    parts = rel.with_suffix("").parts
    return ".".join(("hub", *(parts[:-1] if parts[-1] == "__init__" else parts)))


def _files() -> Iterator[tuple[str, str, str]]:
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC)
        yield rel.as_posix(), _module_of(rel), path.read_text()


# --- the tree -----------------------------------------------------------------------


def test_every_module_level_float_is_declared():
    undeclared = []
    for rel, module, source in _files():
        live = vars(importlib.import_module(module))
        undeclared += [f"{rel}:{name}" for name in undeclared_floats(source, module, live)]
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


def test_a_signature_default_that_is_a_float_or_shapes_a_draw_is_declared_or_named():
    """The escape ADR-0006 recorded, restated. A parameter named for a number of simulations
    with a literal default, or any parameter whose default is a float, is a model input with
    no module-level name; each is listed until it is given one, and a new one is refused here
    rather than arriving unseen."""
    found = {k for rel, _, source in _files() for k in signature_defaults(source, rel)}
    unnamed = sorted(found - set(SIGNATURE_DEFAULTS))
    gone = sorted(set(SIGNATURE_DEFAULTS) - found)
    assert not unnamed, (
        f"signature defaults that are floats or shape a draw and are declared nowhere: "
        f"{unnamed}. Give each a module-level name declared through `hub.declare`, or name it "
        f"here with why.")
    assert not gone, f"SIGNATURE_DEFAULTS names defaults that no longer exist: {gone}"


def test_a_dataclass_field_default_that_is_a_float_is_declared_or_named():
    found = {k for rel, _, source in _files() for k in dataclass_defaults(source, rel)}
    unnamed = sorted(found - set(DATACLASS_DEFAULTS))
    gone = sorted(set(DATACLASS_DEFAULTS) - found)
    assert not unnamed, (
        f"dataclass fields whose default is a float and that are named nowhere: {unnamed}. "
        f"A field has no module-level name to declare, so list it in DATACLASS_DEFAULTS with "
        f"the reason it is covered or no input.")
    assert not gone, f"DATACLASS_DEFAULTS names fields that no longer exist: {gone}"


# --- the controls: one planted source per spelling the scan was blind to (#321) --------
#
# Each asserts both directions on the same planted source: it is flagged as written, and the
# same source with the declaration the message asks for is not. A scan that flagged everything
# passes the first; one that flagged nothing passes the second; only one that reads the
# spelling passes both.

NESTED = {
    "if": "if True:\n    X = 0.5\n",
    "else": "if False:\n    pass\nelse:\n    X = 0.5\n",
    "try": "try:\n    X = 0.5\nexcept ImportError:\n    pass\n",
    "except": "try:\n    pass\nexcept ImportError:\n    X = 0.5\n",
    "finally": "try:\n    pass\nfinally:\n    X = 0.5\n",
    "with": "import contextlib\nwith contextlib.suppress(OSError):\n    X = 0.5\n",
    "for": "for _i in range(1):\n    X = 0.5\n",
    "match": "match 1:\n    case 1:\n        X = 0.5\n",
}
IMPORT = "from hub.declare import chosen\n"


@pytest.mark.parametrize("where", NESTED)
def test_a_float_bound_inside_a_compound_statement_is_seen(where):
    planted = NESTED[where]
    assert undeclared_floats(planted, "hub.x", {}) == ["X"]
    declared = IMPORT + planted.replace("0.5", "chosen(0.5)")
    assert undeclared_floats(declared, "hub.x", {}) == []
    assert [d.name for d in declare.declared_in(declared, "hub.x")] == ["X"]


def test_a_dataclass_field_default_is_seen():
    planted = ("from dataclasses import dataclass\n\n@dataclass(frozen=True)\nclass C:\n"
               "    n: int = 3\n    shrink: float = 0.25\n")
    assert dataclass_defaults(planted, "x.py") == {"x.py:C:shrink"}
    assert dataclass_defaults(planted.replace("0.25", "3"), "x.py") == set()
    assert dataclass_defaults(
        "from typing import NamedTuple\n\nclass N(NamedTuple):\n    a: float = 1.5\n",
        "x.py") == {"x.py:N:a"}
    assert dataclass_defaults("class Plain:\n    a: float = 1.5\n", "x.py") == set()


def test_a_signature_default_is_seen_whatever_it_is_called():
    """`SIGNATURE_DEFAULTS` matched only a name ending `_sims`, so `draws`, `alpha` and `k`
    went through unratcheted."""
    planted = ("def f(a, draws=2000.0, *, alpha: float = 0.05, k=1.5):\n    pass\n\n"
               "class C:\n    def m(self, tol=1e-9):\n        pass\n")
    assert signature_defaults(planted, "x.py") == {
        "x.py:f:draws", "x.py:f:alpha", "x.py:f:k", "x.py:C.m:tol"}
    # what the old rule saw of it: nothing
    assert not [n for n in ("draws", "alpha", "k", "tol") if n.endswith("_sims")]
    assert signature_defaults("def f(a, n_sims=100, b=2, c=None, d='x'):\n    pass\n",
                              "x.py") == {"x.py:f:n_sims"}
    assert signature_defaults("def f(a, b=2, c=None):\n    pass\n", "x.py") == set()
    # a tuple default holds floats
    assert signature_defaults("def f(d=(2.0, 0.18)):\n    pass\n", "x.py") == {"x.py:f:d"}


def test_a_private_float_and_a_float_with_no_literal_are_seen():
    assert undeclared_floats("_FLOOR = 1e-8\n", "hub.x", {}) == ["_FLOOR"]
    assert undeclared_floats(IMPORT + "_FLOOR = chosen(1e-8)\n", "hub.x", {}) == []
    # no float literal anywhere in the source: only the imported value says it is a float
    source = "THIRD = 1 / 3\nALPHA = config.alpha\nGRID = tuple(g / 4 for g in range(3))\n"
    live: dict[str, Any] = {"THIRD": 1 / 3, "ALPHA": 0.5, "GRID": (0.0, 0.25, 0.5)}
    assert undeclared_floats(source, "hub.x", live) == ["ALPHA", "GRID", "THIRD"]
    assert undeclared_floats("N = 3\nS = 'x'\nB = True\n", "hub.x",
                             {"N": 3, "S": "x", "B": True}) == []
    declared = IMPORT + "THIRD = chosen(1 / 3)\nALPHA = chosen(0.5)\nGRID = chosen((0.0, 0.25))\n"
    assert undeclared_floats(declared, "hub.x", live) == []


def test_a_float_unpacked_from_a_tuple_cannot_be_declared_and_is_seen():
    assert undeclared_floats("A, B = 1.5, 2.5\n", "hub.x", {}) == ["A", "B"]
