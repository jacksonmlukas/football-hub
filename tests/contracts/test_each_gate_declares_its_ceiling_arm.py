"""Stage 2 compares a gate's MDE against *its declared ceiling arm* -- so each declares one.

`docs/gate-power.md` stage 2 was pre-registered against a "foresight ceiling". #43 then built
the lineup gate's ceiling as a variance oracle, deliberately: a foresight lineup is the best XI
after the fact and bounds nothing interesting, where a perfect *spread* with the projection
untouched bounds what any spread model could deliver. The two arms are not interchangeable and
foresight is strictly the larger, so the variance oracle is the stricter comparator.

#138 amended the rule to read "its declared ceiling arm" rather than exposing the weaker arm to
make the rule read literally. That amendment is honest only if every gate *declares* an arm by
name where it prints, so no two gates' numbers can be read as one quantity -- which is what this
holds. A gate whose ceiling line said only "ceiling" would let three numbers in three units be
tabulated as one, and the rule would be comparing against a word.

**#387.** Before, this file hand-kept `GATES`, a tuple of module names, and read `CEILING_ARM`
off each -- a fifth registry, alongside the three other hand-kept tuples #387's own ticket
found drifting (coverage was in none of them, margin in none). Every gate now declares one or
more `Harness` (`hub.models.experiment.Harness`; `hub.models.margin` declares two, for its two
verdicts), whose `ceiling_arm` field *is* the declared arm, so this iterates
`tests/gate_harnesses.py`'s discovery -- ten harnesses over nine modules -- rather than
re-listing which module has one.
"""
import pathlib

import pytest
from gate_harnesses import GATE_MODULES, all_harnesses, assert_declares_ceiling_arm


def test_twelve_harnesses_over_the_ten_gate_modules():
    """The discovery itself. #387 fixed it at seven declarations over six modules (margin has
    two, for its two verdicts; every other gate module has one); #343 routed the spread
    candidates, the injury type verdict and the component calibration through the Gate, which
    is three more modules and three more declarations -- and ADR-0019's claim that every gate
    reads the one rule is what makes the count ten rather than seven; #360 declared a second
    injury harness (retention against `out_zero`), which makes it eleven; #465 declares a twelfth in
    `favourite_longshot`, a tenth module."""
    found = all_harnesses()
    assert len(found) == 12, f"expected twelve harnesses, found {len(found)}: {sorted(found)}"
    assert len(GATE_MODULES) == 10, f"expected ten gate modules, found {sorted(GATE_MODULES)}"


@pytest.mark.parametrize("key", sorted(all_harnesses()))
def test_every_harness_declares_its_ceiling_arm_by_name(key):
    assert_declares_ceiling_arm(key, all_harnesses()[key])


def test_the_arms_are_distinct_so_no_two_numbers_read_as_one():
    """Two *gate modules* naming their arm identically would invite exactly the tabulation the
    amendment exists to prevent -- unless they genuinely bound the same quantity, which the
    draft and weekly gates (both foresight) do not: one is a season known in advance to a
    drafter, the other a week known in advance to a projection.

    One arm per module, not per harness: `margin`'s two verdicts (`SHAPE_HARNESS`,
    `WIDTH_HARNESS`) deliberately share `CEILING_ARM` -- both bound the same in-sample
    "perfect P(win | spread)" -- which is one declared quantity read by two verdicts, not two
    gates converging on one name by accident."""
    arms = {}
    for key, h in all_harnesses().items():
        arms[key.rsplit(".", 1)[0]] = h.ceiling_arm
    assert len(set(arms.values())) == len(arms), f"two gate modules share an arm name: {arms}"


def test_the_rule_names_the_declared_arm_not_foresight():
    """The amendment itself, held. Stage 2 must not quietly revert to 'foresight ceiling'."""
    doc = (pathlib.Path(__file__).resolve().parents[2] / "docs" / "gate-power.md").read_text()
    stage2 = doc[doc.index("**Stage 2"):doc.index("**Stage 2") + 900]
    assert "declared ceiling arm" in stage2, (
        "stage 2 no longer reads 'its declared ceiling arm'; the #138 amendment was reverted")
    assert "exceeds its foresight ceiling" not in stage2, (
        "stage 2 says 'foresight ceiling' again, which the lineup gate does not produce under "
        "the arm it declares")
    assert "2026-09-11" in stage2 or "amended" in stage2.lower(), (
        "an amended pre-registration that does not say it was amended is worse than none")


@pytest.mark.parametrize("module", sorted(GATE_MODULES))
def test_each_gate_module_prints_its_arms_on_its_ceiling_lines(module):
    """Declaring it is not enough; the line a reader sees has to carry it. Held on the source
    rather than by running a network gate: every arm this module's Harness(es) declare has to
    appear somewhere in the module beyond the Harness construction itself -- a module constant,
    a CLI help string, or both."""
    import importlib
    import inspect

    src = inspect.getsource(importlib.import_module(module))
    arms = {key.rsplit(".", 1)[1]: h.ceiling_arm for key, h in all_harnesses().items()
            if key.rsplit(".", 1)[0] == module}
    for name, arm in arms.items():
        assert src.count(arm) >= 1, (
            f"{module}'s {name} declares ceiling_arm={arm!r} but nothing in the module prints "
            f"it")
