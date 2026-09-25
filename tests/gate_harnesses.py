"""Discover every gate's `Harness` (#387) by attribute, so the three contract tests over the
gates iterate a discovery rather than a hand-kept tuple.

Before this, three separate files each kept their own tuple naming the same gate modules
(`test_each_gate_declares_its_ceiling_arm.GATES`, `test_gates_tie_test_names_its_within_season_
unit.HARNESSES`, `test_gates_cluster_on_the_season.HARNESSES`) and had drifted from each other
and from the seven call sites they were meant to describe -- coverage was missing from two,
margin from all three, and one of the three still called its own dict `HARNESS` even though it
named ASTs, not harnesses. One discovery, here, shared by all three.

**The discovery is a real tree walk, the same shape `test_every_check_has_a_positive_control.
py`'s `_decisions` uses for decision functions, not a second hand-kept list of module names
wearing a new name.** `harnesses_under` imports every module under a root package and inspects
what it binds at top level; anything that is a `Harness` instance counts. A gate that renamed
its declaration, or a temporary module planting an eighth one anywhere in the tree, is found
without this file changing -- which is the whole point (rule 18, `docs/method.md`): a planted
harness in a directory the discovery walks must be seen, or the discovery is not proven.
"""
from __future__ import annotations

import importlib
import pathlib

from hub.models.experiment import Harness

# The six modules `docs/gate-power.md` and #387 call gates -- read once here so a caller
# wanting only "the six gates" (as opposed to "everything the tree walk found") has a name for
# it, not a re-derivation. `hub.models.margin` declares two harnesses (`SHAPE_HARNESS`,
# `WIDTH_HARNESS`) for its two verdicts; every other module declares one (named `HARNESS`),
# matching #387's count of seven harnesses in all.
GATE_MODULES: tuple[str, ...] = (
    "hub.draft.backtest",
    "hub.season.weekly_gate",
    "hub.season.lineup_gate",
    "hub.models.coverage",
    "hub.models.starter_change",
    "hub.models.margin",
)


def harnesses_under(root: pathlib.Path, package: str) -> dict[str, Harness]:
    """`"module.NAME" -> Harness`, for every module-level `Harness` instance found by
    importing every `.py` file under `root` as `package.<its dotted path>`.

    A real import, not an AST scan: a `Harness` is a runtime object, constructed from fields
    that may themselves be imports or expressions (`lineup_gate.HARNESS`'s `ceiling_arm`, for
    one), so nothing short of importing the module can say what it actually bound. Every
    module under `src/hub` imports cleanly with no network or filesystem side effect at import
    time -- CLAUDE.md's rule against reading `data/` is a rule about what a module's
    *functions* do when called, not about importing the module itself -- so this is cheap and
    safe to run on every collection of this file's callers.
    """
    out: dict[str, Harness] = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).with_suffix("")
        parts = rel.parts
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        dotted = ".".join((package, *parts)) if parts else package
        mod = importlib.import_module(dotted)
        for name, value in vars(mod).items():
            if isinstance(value, Harness):
                out[f"{dotted}.{name}"] = value
    return out


def all_harnesses() -> dict[str, Harness]:
    """Every `Harness` in this repo's own `src/hub` tree -- seven, over six modules, today."""
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "hub"
    return harnesses_under(src, "hub")


def gate_harnesses() -> dict[str, Harness]:
    """`module -> its Harness`, for every module in `GATE_MODULES` that declares exactly one.

    `hub.models.margin` declares two and is read by `all_harnesses` instead -- a caller asking
    "one per gate module" (the ceiling-arm and within-unit contracts, which read a single
    declaration per module in their per-module checks) wants this; a caller counting every
    harness in the repo wants `all_harnesses`.
    """
    found = all_harnesses()
    by_module: dict[str, dict[str, Harness]] = {}
    for key, harness in found.items():
        module = key.rsplit(".", 1)[0]
        by_module.setdefault(module, {})[key] = harness
    return {module: next(iter(names.values())) for module, names in by_module.items()
            if module in GATE_MODULES and len(names) == 1}


def assert_declares_ceiling_arm(key, harness):
    """The contract's one check, callable on any harness -- so the planted control in
    `test_harness_discovery_is_proven.py` runs this, not a copy of it. Used by
    `test_each_gate_declares_its_ceiling_arm.py`."""
    arm = harness.ceiling_arm
    assert isinstance(arm, str) and arm.strip(), (
        f"{key} has no ceiling_arm. Stage 2 of docs/gate-power.md is applied to each gate's "
        f"*declared* arm, and a gate that declares none has nothing for the rule to read.")
