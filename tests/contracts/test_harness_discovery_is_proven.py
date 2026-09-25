"""Rule 18, applied to `tests/gate_harnesses.py` itself (#387).

The three contracts over the gates -- `test_each_gate_declares_its_ceiling_arm.py`,
`test_gates_tie_test_names_its_within_season_unit.py`, `test_gates_cluster_on_the_season.py`
-- all iterate `gate_harnesses.all_harnesses()`/`gate_harnesses()` rather than a hand-kept
tuple. That is only worth doing if the discovery itself can be seen to work: a planted eighth
`Harness` in a module the walk reaches must be found, and a planted `Harness` missing a
`ceiling_arm` must fail to construct at all (`Harness` is a `NamedTuple` with no default for
it) -- both are checked here, directly, the way `test_every_check_has_a_positive_control.py`'s
own last test plants a decision function in a temporary tree rather than trusting its regex.
"""
from pathlib import Path

import pytest
from gate_harnesses import harnesses_under

from hub.models.experiment import Actions, Harness


def test_a_planted_eighth_harness_is_discovered(tmp_path: Path):
    """A `Harness` bound at module scope, anywhere the walk reaches, is found -- not only in
    the six modules `GATE_MODULES` already names. This is what makes the other three contracts'
    "iterate the declarations" honest: they ask this same function, not a list of module names
    an eighth gate could be added without ever joining."""
    # A package name distinct from the real `hub` -- `import_module("hub.models.planted")`
    # would resolve `hub.models` to the already-loaded real package, whose `__path__` does not
    # include `tmp_path`, and never look here at all.
    pkg = tmp_path / "planted_tree" / "models"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "planted_tree" / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "planted.py").write_text(
        "from hub.models.experiment import Actions, Harness\n"
        "PLANTED = Harness(name='planted', arm_a='a', arm_b='b', within=('x',), "
        "ceiling_arm='an arm', actions=Actions(adopt='A', remove='R', show='S'))\n",
        encoding="utf-8")
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        found = harnesses_under(tmp_path / "planted_tree", "planted_tree")
    finally:
        sys.path.remove(str(tmp_path))
        for name in list(sys.modules):
            if name.startswith("planted_tree"):
                del sys.modules[name]
    assert "planted_tree.models.planted.PLANTED" in found, found
    assert found["planted_tree.models.planted.PLANTED"].ceiling_arm == "an arm"


def test_a_harness_with_no_ceiling_arm_cannot_be_built():
    """`Harness.ceiling_arm` has no default -- so a gate that forgot to declare one fails at
    construction, before any contract has to notice its absence. This is the planted failure
    `test_each_gate_declares_its_ceiling_arm_by_name` exists to catch if it were ever possible
    to reach; `Harness`'s own shape makes reaching it a `TypeError` instead."""
    with pytest.raises(TypeError):
        Harness(name="planted", arm_a="a", arm_b="b", within=("x",),  # type: ignore[call-arg]
                actions=Actions(adopt="A", remove="R", show="S"))
